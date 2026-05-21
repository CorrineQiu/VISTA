from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import (
    concat_box_prediction_layers as torchvision_concat_box_prediction_layers,
)

logger = logging.getLogger(__name__)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(max(w - 1, 0)))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(max(h - 1, 0)))
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i : i + 1], boxes[order[1:]]).squeeze(0)
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.stack(keep) if len(keep) > 0 else boxes.new_zeros((0,), dtype=torch.long)


def encode_boxes(
    reference_boxes: torch.Tensor,
    proposals: torch.Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    wx, wy, ww, wh = weights
    ex_widths = proposals[:, 2] - proposals[:, 0]
    ex_heights = proposals[:, 3] - proposals[:, 1]
    ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

    gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
    gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths.clamp(min=1e-6)
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights.clamp(min=1e-6)
    targets_dw = ww * torch.log(gt_widths.clamp(min=1e-6) / ex_widths.clamp(min=1e-6))
    targets_dh = wh * torch.log(gt_heights.clamp(min=1e-6) / ex_heights.clamp(min=1e-6))
    return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)


def decode_boxes(
    rel_codes: torch.Tensor,
    boxes: torch.Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    wx, wy, ww, wh = weights
    boxes = boxes.to(rel_codes.dtype)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = rel_codes[:, 0] / wx
    dy = rel_codes[:, 1] / wy
    dw = (rel_codes[:, 2] / ww).clamp(max=math.log(1000.0 / 16))
    dh = (rel_codes[:, 3] / wh).clamp(max=math.log(1000.0 / 16))

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(rel_codes)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0 / 9.0, reduction: str = "sum") -> torch.Tensor:
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n * n / beta, n - 0.5 * beta)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def roi_align_single(feature_map: torch.Tensor, rois: torch.Tensor, output_size: int, spatial_scale: float) -> torch.Tensor:
    dtype = feature_map.dtype
    device = feature_map.device
    _, c, h, w = feature_map.shape
    rois = rois.to(dtype=torch.float32, device=device)

    if rois.numel() == 0:
        return torch.zeros((0, c, output_size, output_size), device=device, dtype=dtype)

    outputs = []
    # Self-contained RoIAlign replacement: sample continuous box coordinates with bilinear interpolation.
    # This avoids the old integer crop + adaptive pooling approximation, which is too coarse for STA boxes.
    lin = torch.linspace(0.5 / output_size, 1.0 - 0.5 / output_size, output_size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    for roi in rois:
        b = int(roi[0].item())
        x1, y1, x2, y2 = roi[1:] * spatial_scale
        x1 = x1.clamp(0, max(w - 1, 0))
        y1 = y1.clamp(0, max(h - 1, 0))
        x2 = x2.clamp(min=x1 + 1e-3, max=max(w - 1, 0))
        y2 = y2.clamp(min=y1 + 1e-3, max=max(h - 1, 0))

        grid_x = x1 + xx * (x2 - x1)
        grid_y = y1 + yy * (y2 - y1)
        if w > 1:
            grid_x = grid_x / (w - 1) * 2.0 - 1.0
        else:
            grid_x = grid_x * 0.0
        if h > 1:
            grid_y = grid_y / (h - 1) * 2.0 - 1.0
        else:
            grid_y = grid_y * 0.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        pooled = F.grid_sample(
            feature_map[b : b + 1].float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(dtype=dtype)
        outputs.append(pooled[0])

    return torch.stack(outputs, dim=0).to(dtype=dtype)


def multi_scale_roi_align(features: Dict[str, torch.Tensor], boxes: List[torch.Tensor], image_shapes: List[Tuple[int, int]], output_size: int = 7) -> torch.Tensor:
    feat_names = ["p2", "p3", "p4", "p5", "p6"]
    strides = {"p2": 4.0, "p3": 8.0, "p4": 16.0, "p5": 32.0, "p6": 64.0}

    all_rois = []
    levels = []
    for b, bxs in enumerate(boxes):
        if bxs.numel() == 0:
            continue
        bxs = bxs.to(dtype=torch.float32)
        wh = (bxs[:, 2:] - bxs[:, :2]).clamp(min=1.0)
        areas = torch.sqrt(wh[:, 0] * wh[:, 1])
        lvl = torch.floor(4 + torch.log2(areas / 224.0 + 1e-6))
        lvl = lvl.clamp(min=2, max=6).to(torch.long)
        batch_idx = bxs.new_full((bxs.shape[0], 1), float(b))
        all_rois.append(torch.cat([batch_idx, bxs], dim=1))
        levels.append(lvl)

    feat0 = next(iter(features.values()))
    c = feat0.shape[1]
    dtype = feat0.dtype
    device = feat0.device

    if len(all_rois) == 0:
        return torch.zeros((0, c, output_size, output_size), device=device, dtype=dtype)

    rois = torch.cat(all_rois, dim=0).to(dtype=torch.float32, device=device)
    levels = torch.cat(levels, dim=0).to(device=device)
    out = torch.zeros((rois.shape[0], c, output_size, output_size), device=device, dtype=dtype)

    for level_name in feat_names:
        lvl_id = int(level_name[1])
        inds = torch.where(levels == lvl_id)[0]
        if inds.numel() == 0:
            continue
        pooled = roi_align_single(
            features[level_name],
            rois[inds],
            output_size=output_size,
            spatial_scale=1.0 / strides[level_name],
        )
        out[inds] = pooled.to(dtype=out.dtype, device=out.device)

    return out


class AnchorGenerator(nn.Module):
    def __init__(self, sizes: Sequence[Sequence[int]], aspect_ratios: Sequence[Sequence[float]]):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def generate_anchors(self, sizes: Sequence[int], aspect_ratios: Sequence[float], device) -> torch.Tensor:
        anchors = []
        for size in sizes:
            area = float(size * size)
            for ar in aspect_ratios:
                w = math.sqrt(area / ar)
                h = ar * w
                anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def forward(self, features: Dict[str, torch.Tensor], image_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        feat_names = ["p2", "p3", "p4", "p5", "p6"]
        strides = {"p2": 4.0, "p3": 8.0, "p4": 16.0, "p5": 32.0, "p6": 64.0}
        device = next(iter(features.values())).device
        anchors_over_all_levels = []
        for name, sizes, ars in zip(feat_names, self.sizes, self.aspect_ratios):
            base = self.generate_anchors(sizes, ars, device)
            _, _, h, w = features[name].shape
            stride = strides[name]
            shifts_x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            shifts_y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
            anchors_level = (base[None, :, :] + shifts[:, None, :]).reshape(-1, 4)
            anchors_over_all_levels.append(anchors_level)
        anchors_cat = torch.cat(anchors_over_all_levels, dim=0)
        return [anchors_cat.clone() for _ in image_shapes]


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits = []
        bbox_reg = []
        for name in ["p2", "p3", "p4", "p5", "p6"]:
            t = F.relu(self.conv(features[name]))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer: torch.Tensor, N: int, A: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls: List[torch.Tensor], box_regression: List[torch.Tensor], num_anchors_per_level: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level, A in zip(box_cls, box_regression, num_anchors_per_level):
        N, AxC, H, W = box_cls_per_level.shape
        C = AxC // A
        box_cls_flattened.append(permute_and_flatten(box_cls_per_level, N, A, C, H, W))
        box_regression_flattened.append(permute_and_flatten(box_regression_per_level, N, A, 4, H, W))
    return torch.cat(box_cls_flattened, dim=1).squeeze(-1), torch.cat(box_regression_flattened, dim=1)


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc6.weight, a=1)
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.kaiming_uniform_(self.fc7.weight, a=1)
        nn.init.constant_(self.fc7.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class STAPredictionHead(nn.Module):
    def __init__(
        self,
        representation_size: int,
        num_nouns: int,
        num_verbs: int,
        verb_background: bool = False,
    ):
        super().__init__()
        self.verb_background = bool(verb_background)
        self.noun_classifier = nn.Linear(representation_size, num_nouns + 1)
        self.box_regressor = nn.Linear(representation_size, (num_nouns + 1) * 4)
        self.verb_classifier = nn.Linear(representation_size, num_verbs + int(self.verb_background))
        self.ttc_regressor = nn.Linear(representation_size, 1)
        self.score_regressor = nn.Linear(representation_size, 1)
        for layer in [
            self.noun_classifier,
            self.box_regressor,
            self.verb_classifier,
            self.ttc_regressor,
            self.score_regressor,
        ]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noun_logits = self.noun_classifier(x)
        box_regression = self.box_regressor(x)
        verb_logits = self.verb_classifier(x)
        ttc_pred = F.softplus(self.ttc_regressor(x)).squeeze(-1)
        score_logits = self.score_regressor(x).squeeze(-1)
        return noun_logits, box_regression, verb_logits, ttc_pred, score_logits


class PyramidBuilder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.p4 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.up_p3 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.up_p2 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.down_p5 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.down_p6 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        p4 = self.p4(self.in_proj(x))
        p3 = self.up_p3(F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False))
        p2 = self.up_p2(F.interpolate(p3, scale_factor=2.0, mode="bilinear", align_corners=False))
        p5 = self.down_p5(p4)
        p6 = self.down_p6(p5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}


def _load_vjepa_encoder(checkpoint: str, model_kwargs: Dict, frames_per_clip: int, video_resolution: int):
    if model_kwargs.get("use_v2_1", True):
        try:
            import app.vjepa_2_1.models.vision_transformer as vit
        except ModuleNotFoundError:
            import src.models.vision_transformer as vit
    else:
        import src.models.vision_transformer as vit

    enc_kwargs = dict(model_kwargs["encoder"])
    model_name = enc_kwargs.pop("model_name")
    checkpoint_key = enc_kwargs.pop("checkpoint_key", "target_encoder")
    encoder = vit.__dict__[model_name](img_size=video_resolution, num_frames=frames_per_clip, **enc_kwargs)
    state = torch.load(checkpoint, map_location="cpu")
    pretrained = state[checkpoint_key]
    pretrained = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pretrained.items()}
    current = encoder.state_dict()
    for k, v in current.items():
        if k not in pretrained or pretrained[k].shape != v.shape:
            pretrained[k] = v
    msg = encoder.load_state_dict(pretrained, strict=False)
    logger.info("Loaded V-JEPA encoder with msg: %s", msg)
    return encoder


class FrozenVJEPA2_1DualEncoder(nn.Module):
    def __init__(self, checkpoint: str, model_kwargs: Dict, frames_per_clip: int, video_resolution: int):
        super().__init__()
        self.encoder = _load_vjepa_encoder(checkpoint, model_kwargs, frames_per_clip, video_resolution)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.tubelet_size = getattr(self.encoder, "tubelet_size", 2)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.encoder, "forward_features"):
                try:
                    return self.encoder.forward_features(video)
                except Exception:
                    pass
            return self.encoder(video)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.encoder, "forward_image"):
                return self.encoder.forward_image(image)
            if hasattr(self.encoder, "forward_features"):
                try:
                    return self.encoder.forward_features(image, num_frames=1)
                except Exception:
                    pass
            return self.encoder(image.unsqueeze(2))

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self._encode_video(video)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self._encode_image(image)

    def forward(self, video: torch.Tensor, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"video_tokens": self.encode_video(video), "image_tokens": self.encode_image(image)}


class CachedFeatureOnlyVideoEncoder(nn.Module):
    """Placeholder used when cached JEPA global tokens bypass the video encoder."""

    def __init__(self, embed_dim: int, patch_size: int = 16, tubelet_size: int = 2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "V-JEPA encoder load was skipped; cached video_global_token is required for every sample."
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "V-JEPA encoder load was skipped; cached video_global_token is required for every sample."
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalAttentiveProbe(nn.Module):
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class FrameGuidedTemporalPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.out = nn.LayerNorm(dim)

    def forward(self, image_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(image_tokens)
        kv = self.norm_kv(video_tokens)
        y, _ = self.cross(q, kv, kv, need_weights=False)
        return self.out(image_tokens + y)


class SumConvFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(32, dim), nn.GELU())

    def forward(self, image_tokens: torch.Tensor, aligned_video_tokens: torch.Tensor, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        fused_tokens = image_tokens + aligned_video_tokens
        b, _, d = fused_tokens.shape
        h = image.shape[-2] // patch_size
        w = image.shape[-1] // patch_size
        grid = fused_tokens.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        return self.conv(grid)


@dataclass
class STAPaperHeadOutputs:
    pred_boxes: List[torch.Tensor]
    noun_logits: List[torch.Tensor]
    verb_logits: List[torch.Tensor]
    ttc_pred: List[torch.Tensor]
    scores: List[torch.Tensor]


@dataclass
class STAPaperOutputs:
    pred_boxes: List[torch.Tensor]
    noun_logits: List[torch.Tensor]
    verb_logits: List[torch.Tensor]
    ttc_pred: List[torch.Tensor]
    scores: List[torch.Tensor]
    loss_dict: Dict[str, torch.Tensor]
    head_outputs: Optional[List[STAPaperHeadOutputs]] = None
    selected_head_index: int = 0
    selected_head_indices: Optional[List[int]] = None
    num_prediction_heads: int = 4


class STAPaperFaithfulModel(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        backbone_kwargs: Dict,
        frames_per_clip: int = 8,
        video_resolution: int = 384,
        num_nouns: int = 128,
        num_verbs: int = 81,
        proposal_topk_train: int = 2000,
        proposal_topk_test: int = 300,
        rpn_pre_nms_topk_train: int = 2000,
        rpn_pre_nms_topk_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        roi_pool_size: int = 7,
        fpn_dim: int = 256,
        probe_depth: int = 2,
        num_heads: int = 8,
        representation_size: int = 1024,
        class_topk_per_proposal: int = 5,
        verb_topk_per_proposal: int = 3,
        detections_per_img: int = 100,
        box_score_thresh: float = 0.001,
        box_nms_thresh: float = 0.55,
        box_reg_weights: Sequence[float] = (10.0, 10.0, 5.0, 5.0),
        inference_score: str = "objectness_quality_noun_verb",
        verb_loss_weight: float = 0.1,
        ttc_loss_weight: float = 0.5,
        score_loss_weight: float = 1.0,
        num_prediction_heads: int = 4,
        verb_background: bool = True,
    ):
        super().__init__()
        self.backbone = FrozenVJEPA2_1DualEncoder(checkpoint, backbone_kwargs, frames_per_clip, video_resolution)
        dim = self.backbone.embed_dim
        self.patch_size = int(self.backbone.patch_size)
        self.num_nouns = int(num_nouns)
        self.num_verbs = int(num_verbs)
        self.proposal_topk_train = int(proposal_topk_train)
        self.proposal_topk_test = int(proposal_topk_test)
        self.rpn_pre_nms_topk_train = int(rpn_pre_nms_topk_train)
        self.rpn_pre_nms_topk_test = int(rpn_pre_nms_topk_test)
        self.rpn_nms_thresh = float(rpn_nms_thresh)
        self.roi_pool_size = int(roi_pool_size)
        self.fpn_dim = int(fpn_dim)
        self.representation_size = int(representation_size)
        self.class_topk_per_proposal = max(1, int(class_topk_per_proposal))
        self.verb_topk_per_proposal = max(1, int(verb_topk_per_proposal))
        self.detections_per_img = int(detections_per_img)
        self.box_score_thresh = float(box_score_thresh)
        self.box_nms_thresh = float(box_nms_thresh)
        if len(tuple(box_reg_weights)) != 4:
            raise ValueError(f"box_reg_weights must contain four values, got {box_reg_weights}")
        self.box_reg_weights = tuple(float(v) for v in box_reg_weights)
        self.inference_score = str(inference_score)
        self.verb_loss_weight = float(verb_loss_weight)
        self.ttc_loss_weight = float(ttc_loss_weight)
        self.score_loss_weight = float(score_loss_weight)
        self.num_prediction_heads = max(1, int(num_prediction_heads))
        self.verb_background = bool(verb_background)

        self.video_probe = TemporalAttentiveProbe(dim=dim, depth=probe_depth, num_heads=num_heads)
        self.temporal_pool = FrameGuidedTemporalPooling(dim=dim, num_heads=num_heads)
        self.fusion = SumConvFusion(dim=dim)
        self.pyramid = PyramidBuilder(in_dim=dim, out_dim=fpn_dim)

        self.anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        self.rpn_head = RPNHead(fpn_dim, self.anchor_generator.num_anchors_per_location()[0])

        self.box_head = TwoMLPHead(fpn_dim * roi_pool_size * roi_pool_size, representation_size)
        self.global_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, representation_size), nn.GELU())
        self.context_fc = nn.Sequential(
            nn.Linear(representation_size * 2, representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(inplace=True),
        )
        self.prediction_head = STAPredictionHead(
            representation_size,
            num_nouns,
            num_verbs,
            verb_background=self.verb_background,
        )
        self.extra_prediction_heads = nn.ModuleList(
            [
                STAPredictionHead(
                    representation_size,
                    num_nouns,
                    num_verbs,
                    verb_background=self.verb_background,
                )
                for _ in range(self.num_prediction_heads - 1)
            ]
        )

    def _iter_prediction_heads(self):
        yield self.prediction_head
        for head in self.extra_prediction_heads:
            yield head

    def _select_prediction_heads(self, head_indices: Optional[List[int]] = None):
        heads = list(self._iter_prediction_heads())
        if head_indices is None:
            return list(enumerate(heads))
        selected = []
        seen = set()
        for idx in head_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(heads):
                raise ValueError(f"head index {idx} out of range for {len(heads)} prediction heads")
            if idx not in seen:
                selected.append((idx, heads[idx]))
                seen.add(idx)
        if not selected:
            raise ValueError("head_indices selected no prediction heads")
        return selected

    def _build_fused_image_maps(
        self,
        image: torch.Tensor,
        image_shapes: List[Tuple[int, int]],
        video_tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        fused_maps = []
        for b, (img_h, img_w) in enumerate(image_shapes):
            image_b = image[b : b + 1, :, :img_h, :img_w].contiguous()
            image_tokens_b = self.backbone.encode_image(image_b)
            aligned_video_b = self.temporal_pool(image_tokens_b, video_tokens[b : b + 1])
            fused_map_b = self.fusion(image_tokens_b, aligned_video_b, image_b, self.patch_size)
            fused_maps.append(fused_map_b)
        return fused_maps

    def _build_pyramid_batch(self, fused_maps: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        per_sample = [self.pyramid(x) for x in fused_maps]
        out: Dict[str, torch.Tensor] = {}
        for level_name in ["p2", "p3", "p4", "p5", "p6"]:
            max_h = max(p[level_name].shape[-2] for p in per_sample)
            max_w = max(p[level_name].shape[-1] for p in per_sample)
            ref = per_sample[0][level_name]
            batch = ref.new_zeros((len(per_sample), ref.shape[1], max_h, max_w))
            for b, p in enumerate(per_sample):
                feat = p[level_name]
                _, _, h, w = feat.shape
                batch[b, :, :h, :w] = feat[0]
            out[level_name] = batch
        return out

    def _flatten_rpn(self, objectness: List[torch.Tensor], bbox_regression: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return concat_box_prediction_layers(objectness, bbox_regression, self.anchor_generator.num_anchors_per_location())

    def _anchor_valid_masks(self, anchors: List[torch.Tensor], image_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        valid_masks = []
        for anchors_per_img, (img_h, img_w) in zip(anchors, image_shapes):
            center_x = 0.5 * (anchors_per_img[:, 0] + anchors_per_img[:, 2])
            center_y = 0.5 * (anchors_per_img[:, 1] + anchors_per_img[:, 3])
            valid_masks.append(
                (center_x >= 0.0)
                & (center_y >= 0.0)
                & (center_x < float(img_w))
                & (center_y < float(img_h))
            )
        return valid_masks

    def _compute_rpn_losses(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        anchors: List[torch.Tensor],
        targets: List[Dict],
        valid_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = objectness.device
        labels_all = []
        matched_gt_boxes_all = []
        sampled_inds_all = []
        sampled_pos_inds_all = []
        for b, (anchors_per_img, tgt) in enumerate(zip(anchors, targets)):
            gt_boxes = tgt["boxes"].to(device)
            num_anchors = anchors_per_img.shape[0]
            labels = anchors_per_img.new_full((num_anchors,), -1, dtype=torch.long)
            matched_gt_boxes = anchors_per_img.new_zeros((num_anchors, 4))
            if valid_masks is None:
                valid = torch.ones((num_anchors,), dtype=torch.bool, device=device)
            else:
                valid = valid_masks[b].to(device=device, dtype=torch.bool)
            valid_inds = torch.where(valid)[0]
            anchors_valid = anchors_per_img[valid_inds]
            if valid_inds.numel() == 0:
                pass
            elif gt_boxes.numel() == 0:
                labels[valid_inds] = 0
            else:
                iou = box_iou(anchors_valid, gt_boxes)
                matched_vals, matched_idx = iou.max(dim=1)
                labels_valid = anchors_per_img.new_full((valid_inds.numel(),), -1, dtype=torch.long)
                labels_valid[matched_vals < 0.3] = 0
                labels_valid[matched_vals >= 0.7] = 1
                gt_best = iou.argmax(dim=0)
                labels_valid[gt_best] = 1
                labels[valid_inds] = labels_valid
                matched_gt_boxes[valid_inds] = gt_boxes[matched_idx]
            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where(labels == 0)[0]
            num_pos = min(128, pos_idx.numel())
            num_neg = min(256 - num_pos, neg_idx.numel())
            if pos_idx.numel() > 0:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:num_pos]]
            if neg_idx.numel() > 0:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:num_neg]]
            sampled = torch.cat([pos_idx, neg_idx], dim=0)
            labels_all.append(labels)
            matched_gt_boxes_all.append(matched_gt_boxes)
            sampled_inds_all.append(sampled + b * objectness.shape[1])
            sampled_pos_inds_all.append(pos_idx + b * objectness.shape[1])

        labels_cat = torch.cat(labels_all, dim=0)
        matched_gt_boxes_cat = torch.cat(matched_gt_boxes_all, dim=0)
        sampled_inds = torch.cat(sampled_inds_all, dim=0) if len(sampled_inds_all) > 0 else torch.empty((0,), dtype=torch.long, device=device)
        sampled_pos_inds = torch.cat(sampled_pos_inds_all, dim=0) if len(sampled_pos_inds_all) > 0 else torch.empty((0,), dtype=torch.long, device=device)
        objectness_flat = objectness.reshape(-1)
        loss_objectness = F.binary_cross_entropy_with_logits(objectness_flat[sampled_inds], labels_cat[sampled_inds].float()) if sampled_inds.numel() > 0 else objectness_flat.sum() * 0.0

        pos_inds = sampled_pos_inds
        if pos_inds.numel() > 0:
            anchors_cat = torch.cat(anchors, dim=0)
            regression_targets = encode_boxes(matched_gt_boxes_cat[pos_inds], anchors_cat[pos_inds])
            loss_rpn_box_reg = smooth_l1_loss(pred_bbox_deltas.reshape(-1, 4)[pos_inds], regression_targets, beta=1.0 / 9.0, reduction="sum") / max(1, pos_inds.numel())
        else:
            loss_rpn_box_reg = objectness.sum() * 0.0
        return {"loss_rpn_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}

    def _filter_proposals(self, proposals: torch.Tensor, scores: torch.Tensor, image_shape: Tuple[int, int], pre_nms_topk: int, post_nms_topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = scores.sigmoid()
        keep = torch.argsort(scores, descending=True)[:pre_nms_topk]
        proposals = proposals[keep]
        scores = scores[keep]
        proposals = clip_boxes_to_image(proposals, image_shape)
        keep = remove_small_boxes(proposals, 1.0)
        proposals = proposals[keep]
        scores = scores[keep]
        keep = nms(proposals, scores, self.rpn_nms_thresh)
        keep = keep[:post_nms_topk]
        return proposals[keep], scores[keep]

    def _generate_proposals(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        anchors: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        valid_masks: Optional[List[torch.Tensor]] = None,
        loss_mode: bool = False,
        pre_nms_topk: Optional[int] = None,
        post_nms_topk: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        proposals_out = []
        scores_out = []
        pre_nms_topk = int(pre_nms_topk if pre_nms_topk is not None else (self.rpn_pre_nms_topk_train if loss_mode else self.rpn_pre_nms_topk_test))
        post_nms_topk = int(post_nms_topk if post_nms_topk is not None else (self.proposal_topk_train if loss_mode else self.proposal_topk_test))
        for b in range(objectness.shape[0]):
            anchors_b = anchors[b]
            deltas_b = pred_bbox_deltas[b]
            scores = objectness[b]
            if valid_masks is not None:
                valid = valid_masks[b].to(device=scores.device, dtype=torch.bool)
                anchors_b = anchors_b[valid]
                deltas_b = deltas_b[valid]
                scores = scores[valid]
            props = decode_boxes(deltas_b, anchors_b)
            props, scores = self._filter_proposals(props, scores, image_shapes[b], pre_nms_topk, post_nms_topk)
            proposals_out.append(props)
            scores_out.append(scores)
        return proposals_out, scores_out

    @torch.no_grad()
    def predict_rpn_proposals(
        self,
        video: torch.Tensor,
        image: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        topk: Optional[int] = None,
        pre_nms_topk: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        image_shapes = (
            [(int(t["size"][0]), int(t["size"][1])) for t in targets]
            if targets is not None
            else [(int(image.shape[-2]), int(image.shape[-1])) for _ in range(image.shape[0])]
        )
        z_v = self.backbone.encode_video(video)
        h_v = self.video_probe(z_v)
        fused_maps = self._build_fused_image_maps(image, image_shapes, h_v)
        pyramid = self._build_pyramid_batch(fused_maps)
        rpn_objectness, rpn_bbox_regression = self.rpn_head(pyramid)
        rpn_objectness_flat, rpn_bbox_regression_flat = self._flatten_rpn(
            rpn_objectness, rpn_bbox_regression
        )
        anchors = self.anchor_generator(pyramid, image_shapes)
        valid_masks = self._anchor_valid_masks(anchors, image_shapes)
        return self._generate_proposals(
            rpn_objectness_flat,
            rpn_bbox_regression_flat,
            anchors,
            image_shapes,
            valid_masks=valid_masks,
            loss_mode=False,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=topk,
        )

    def _assign_targets_to_proposals(self, proposals: List[torch.Tensor], targets: List[Dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        matched_idxs = []
        labels = []
        matched_boxes = []
        matched_verbs = []
        matched_ttc = []
        matched_ious = []
        for proposals_per_img, targets_per_img in zip(proposals, targets):
            gt_boxes = targets_per_img["boxes"]
            gt_labels = targets_per_img["noun_labels"]
            gt_verbs = targets_per_img["verb_labels"]
            gt_ttc = targets_per_img["ttc"]
            if gt_boxes.numel() == 0:
                device = proposals_per_img.device
                matched_idxs.append(torch.full((proposals_per_img.shape[0],), -1, dtype=torch.long, device=device))
                labels.append(torch.zeros((proposals_per_img.shape[0],), dtype=torch.long, device=device))
                matched_boxes.append(torch.zeros((proposals_per_img.shape[0], 4), dtype=torch.float32, device=device))
                matched_verbs.append(torch.zeros((proposals_per_img.shape[0],), dtype=torch.long, device=device))
                matched_ttc.append(torch.zeros((proposals_per_img.shape[0],), dtype=torch.float32, device=device))
                matched_ious.append(torch.zeros((proposals_per_img.shape[0],), dtype=torch.float32, device=device))
                continue
            iou = box_iou(proposals_per_img, gt_boxes)
            matched_vals, matched_idx = iou.max(dim=1)
            lbl = gt_labels[matched_idx] + 1
            lbl[matched_vals < 0.5] = 0
            matched_idxs.append(matched_idx)
            labels.append(lbl)
            matched_boxes.append(gt_boxes[matched_idx])
            matched_verbs.append(gt_verbs[matched_idx])
            matched_ttc.append(gt_ttc[matched_idx])
            matched_ious.append(matched_vals)
        return matched_idxs, labels, matched_boxes, matched_verbs, matched_ttc, matched_ious

    def _subsample_proposals(
        self,
        labels: List[torch.Tensor],
        force_keep: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        sampled_inds = []
        for img_idx, lbl in enumerate(labels):
            positive = torch.where(lbl > 0)[0]
            negative = torch.where(lbl == 0)[0]
            num_pos = min(128, positive.numel())
            forced_pos = positive.new_zeros((0,))
            if force_keep is not None and img_idx < len(force_keep):
                forced = force_keep[img_idx].to(device=lbl.device, dtype=torch.long)
                forced = forced[(forced >= 0) & (forced < lbl.numel())]
                forced = forced[lbl[forced] > 0]
                if forced.numel() > 0:
                    forced_pos = torch.unique(forced)[:num_pos]

            if positive.numel() > 0:
                if forced_pos.numel() > 0:
                    remaining_pos = positive[~torch.isin(positive, forced_pos)]
                else:
                    remaining_pos = positive
                num_random_pos = max(0, num_pos - forced_pos.numel())
                if remaining_pos.numel() > 0 and num_random_pos > 0:
                    random_pos = remaining_pos[torch.randperm(remaining_pos.numel(), device=remaining_pos.device)[:num_random_pos]]
                else:
                    random_pos = positive.new_zeros((0,))
                positive = torch.cat([forced_pos, random_pos], dim=0)
            num_neg = min(512 - positive.numel(), negative.numel())
            if negative.numel() > 0:
                negative = negative[torch.randperm(negative.numel(), device=negative.device)[:num_neg]]
            sampled_inds.append(torch.cat([positive, negative], dim=0))
        return sampled_inds

    def _training_sample_proposals(self, proposals: List[torch.Tensor], proposal_scores: List[torch.Tensor], targets: List[Dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        proposals_with_gt = []
        proposal_scores_with_gt = []
        gt_force_keep = []
        for p, s, t in zip(proposals, proposal_scores, targets):
            gt_boxes = t["boxes"].to(p.device)
            num_props = p.shape[0]
            num_gt = gt_boxes.shape[0]
            proposals_with_gt.append(torch.cat([p, gt_boxes], dim=0))
            proposal_scores_with_gt.append(torch.cat([s, s.new_ones((num_gt,))], dim=0))
            gt_force_keep.append(torch.arange(num_props, num_props + num_gt, device=p.device, dtype=torch.long))
        proposals = proposals_with_gt
        proposal_scores = proposal_scores_with_gt
        _, labels, matched_boxes, matched_verbs, matched_ttc, matched_ious = self._assign_targets_to_proposals(proposals, targets)
        sampled_inds = self._subsample_proposals(labels, force_keep=gt_force_keep)
        proposals = [p[idx] for p, idx in zip(proposals, sampled_inds)]
        proposal_scores = [s[idx] for s, idx in zip(proposal_scores, sampled_inds)]
        labels = [l[idx] for l, idx in zip(labels, sampled_inds)]
        matched_boxes = [b[idx] for b, idx in zip(matched_boxes, sampled_inds)]
        matched_verbs = [v[idx] for v, idx in zip(matched_verbs, sampled_inds)]
        matched_ttc = [t[idx] for t, idx in zip(matched_ttc, sampled_inds)]
        matched_ious = [q[idx] for q, idx in zip(matched_ious, sampled_inds)]
        return proposals, proposal_scores, labels, matched_boxes, matched_verbs, matched_ttc, matched_ious

    def _roi_features(self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor], global_token: torch.Tensor, image_shapes: List[Tuple[int, int]]):
        box_features = multi_scale_roi_align(features, proposals, image_shapes, output_size=self.roi_pool_size)
        box_features = self.box_head(box_features)
        counts = [p.shape[0] for p in proposals]
        ctx_global = self.global_proj(global_token)
        fused = []
        start = 0
        for b, n in enumerate(counts):
            local = box_features[start : start + n]
            glb = ctx_global[b].unsqueeze(0).expand(n, -1)
            enriched = self.context_fc(torch.cat([local, glb], dim=1))
            fused.append(local + enriched)
            start += n
        fused = torch.cat(fused, dim=0) if len(fused) > 0 else box_features.new_zeros((0, self.representation_size))
        return box_features, fused

    def _run_prediction_head(self, fused: torch.Tensor, head: nn.Module):
        noun_logits, box_regression, verb_logits, ttc_pred, score_logits = head(fused)
        return noun_logits, box_regression, verb_logits, ttc_pred, score_logits

    def _roi_forward(self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor], proposal_scores: List[torch.Tensor], global_token: torch.Tensor, image_shapes: List[Tuple[int, int]]):
        del proposal_scores
        box_features, fused = self._roi_features(features, proposals, global_token, image_shapes)
        noun_logits, box_regression, verb_logits, ttc_pred, score_logits = self.prediction_head(fused)
        return noun_logits, box_regression, verb_logits, ttc_pred, score_logits, box_features, fused

    def _compute_roi_losses(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        verb_logits: torch.Tensor,
        ttc_pred: torch.Tensor,
        score_logits: torch.Tensor,
        proposals: List[torch.Tensor],
        labels: List[torch.Tensor],
        regression_targets_boxes: List[torch.Tensor],
        regression_targets_verbs: List[torch.Tensor],
        regression_targets_ttc: List[torch.Tensor],
        regression_targets_ious: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = class_logits.device
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_boxes_cat = torch.cat(regression_targets_boxes, dim=0)
        regression_targets_verbs_cat = torch.cat(regression_targets_verbs, dim=0)
        regression_targets_ttc_cat = torch.cat(regression_targets_ttc, dim=0)
        regression_targets_ious_cat = torch.cat(regression_targets_ious, dim=0).to(device)

        loss_classifier = F.cross_entropy(class_logits, labels_cat)
        quality_targets = torch.where(labels_cat > 0, regression_targets_ious_cat.clamp(0.0, 1.0), torch.zeros_like(regression_targets_ious_cat))
        loss_score = F.binary_cross_entropy_with_logits(score_logits, quality_targets)

        sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
        if sampled_pos_inds_subset.numel() > 0:
            labels_pos = labels_cat[sampled_pos_inds_subset]
            box_regression = box_regression.reshape(class_logits.shape[0], self.num_nouns + 1, 4)
            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                encode_boxes(
                    regression_targets_boxes_cat[sampled_pos_inds_subset],
                    torch.cat(proposals, dim=0)[sampled_pos_inds_subset],
                    weights=self.box_reg_weights,
                ),
                beta=1.0 / 9.0,
                reduction="sum",
            ) / labels_cat.numel()
            loss_ttc = F.smooth_l1_loss(ttc_pred[sampled_pos_inds_subset], regression_targets_ttc_cat[sampled_pos_inds_subset])
        else:
            box_loss = class_logits.sum() * 0.0
            loss_ttc = class_logits.sum() * 0.0
        if self.verb_background:
            verb_targets = regression_targets_verbs_cat.to(device=device, dtype=torch.long) + 1
            verb_targets = torch.where(labels_cat > 0, verb_targets, torch.zeros_like(verb_targets))
            loss_verb = F.cross_entropy(verb_logits, verb_targets) if labels_cat.numel() > 0 else class_logits.sum() * 0.0
        elif sampled_pos_inds_subset.numel() > 0:
            loss_verb = F.cross_entropy(verb_logits[sampled_pos_inds_subset], regression_targets_verbs_cat[sampled_pos_inds_subset])
        else:
            loss_verb = class_logits.sum() * 0.0
        return {
            "loss_noun_classifier": loss_classifier,
            "loss_box_reg": box_loss,
            "loss_verb": self.verb_loss_weight * loss_verb,
            "loss_ttc": self.ttc_loss_weight * loss_ttc,
            "loss_interaction_score": self.score_loss_weight * loss_score,
        }

    def _inference_from_roi(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        verb_logits: torch.Tensor,
        ttc_pred: torch.Tensor,
        score_logits: torch.Tensor,
        proposals: List[torch.Tensor],
        proposal_scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        pred_boxes_out, noun_logits_out, verb_logits_out, ttc_out, scores_out = [], [], [], [], []
        start = 0
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_nouns + 1, 4)
        probs_all = F.softmax(class_logits, dim=1)
        for b, props in enumerate(proposals):
            n = props.shape[0]
            cls_b = class_logits[start : start + n]
            probs_b = probs_all[start : start + n]
            box_reg_b = box_regression[start : start + n]
            verb_b = verb_logits[start : start + n]
            ttc_b = ttc_pred[start : start + n]
            score_b = score_logits[start : start + n]
            proposal_scores_b = proposal_scores[b]
            start += n
            if n == 0:
                device = cls_b.device if cls_b.numel() > 0 else props.device
                pred_boxes_out.append(torch.zeros((0, 4), device=device))
                noun_logits_out.append(torch.zeros((0, self.num_nouns), device=device))
                verb_logits_out.append(torch.zeros((0, self.num_verbs), device=device))
                ttc_out.append(torch.zeros((0,), device=device))
                scores_out.append(torch.zeros((0,), device=device))
                continue
            fg_probs = probs_b[:, 1:]
            class_topk = min(self.class_topk_per_proposal, fg_probs.shape[1])
            noun_scores, labels = fg_probs.topk(class_topk, dim=1)
            proposal_indices = torch.arange(n, device=props.device).unsqueeze(1).expand(n, class_topk).reshape(-1)
            labels = labels.reshape(-1)
            noun_scores = noun_scores.reshape(-1)
            labels_plus = labels + 1
            boxes = decode_boxes(
                box_reg_b[proposal_indices, labels_plus],
                props[proposal_indices],
                weights=self.box_reg_weights,
            )
            boxes = clip_boxes_to_image(boxes, image_shapes[b])
            keep = remove_small_boxes(boxes, 1.0)
            boxes = boxes[keep]
            noun_scores = noun_scores[keep]
            labels = labels[keep]
            proposal_indices = proposal_indices[keep]
            verb_scores_all = verb_b[proposal_indices].softmax(dim=-1)
            if self.verb_background:
                # Match StillFast v2: verb class 0 is background during training,
                # but official STA predictions use foreground verb ids only.
                verb_scores_all = verb_scores_all[:, 1:]
            verb_topk = min(self.verb_topk_per_proposal, verb_scores_all.shape[1])
            verb_scores, verb_labels = verb_scores_all.topk(verb_topk, dim=1)
            boxes = boxes.unsqueeze(1).expand(-1, verb_topk, -1).reshape(-1, 4)
            noun_scores = noun_scores.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            labels = labels.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            proposal_indices = proposal_indices.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            verb_scores = verb_scores.reshape(-1)
            verb_labels = verb_labels.reshape(-1)
            quality_prob = score_b[proposal_indices].sigmoid()
            objectness_prob = proposal_scores_b[proposal_indices].clamp(0.0, 1.0)
            if self.inference_score == "noun":
                scores = noun_scores
            elif self.inference_score == "noun_verb":
                scores = noun_scores * verb_scores
            elif self.inference_score == "quality_noun_verb":
                scores = quality_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness_noun_verb":
                scores = objectness_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness_quality_noun_verb":
                scores = objectness_prob * quality_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness":
                scores = objectness_prob
            elif self.inference_score == "quality":
                scores = quality_prob
            else:
                raise ValueError(f"Unsupported inference_score={self.inference_score}")
            score_keep = scores >= self.box_score_thresh
            boxes = boxes[score_keep]
            scores = scores[score_keep]
            labels = labels[score_keep]
            proposal_indices = proposal_indices[score_keep]
            verb_labels = verb_labels[score_keep]
            ttc_keep = ttc_b[proposal_indices].clamp_min(0.0)
            noun_logits_keep = cls_b.new_full((labels.shape[0], self.num_nouns), -50.0)
            if labels.numel() > 0:
                noun_logits_keep[torch.arange(labels.shape[0], device=labels.device), labels] = 50.0
            verb_logits_keep = verb_b.new_full((labels.shape[0], self.num_verbs), -50.0)
            if verb_labels.numel() > 0:
                verb_logits_keep[torch.arange(verb_labels.shape[0], device=verb_labels.device), verb_labels] = 50.0
            if boxes.numel() > 0:
                keep_all = []
                compound_labels = labels * self.num_verbs + verb_labels
                for cls_id in compound_labels.unique():
                    cls_inds = torch.where(compound_labels == cls_id)[0]
                    cls_keep = nms(boxes[cls_inds], scores[cls_inds], self.box_nms_thresh)
                    keep_all.append(cls_inds[cls_keep])
                keep_nms = torch.cat(keep_all, dim=0) if keep_all else labels.new_zeros((0,), dtype=torch.long)
                keep_nms = keep_nms[torch.argsort(scores[keep_nms], descending=True)]
                keep_nms = keep_nms[: self.detections_per_img]
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                ttc_keep = ttc_keep[keep_nms]
                noun_logits_keep = noun_logits_keep[keep_nms]
                verb_logits_keep = verb_logits_keep[keep_nms]
            pred_boxes_out.append(boxes)
            noun_logits_out.append(noun_logits_keep)
            verb_logits_out.append(verb_logits_keep)
            ttc_out.append(ttc_keep)
            scores_out.append(scores)
        return pred_boxes_out, noun_logits_out, verb_logits_out, ttc_out, scores_out


    def forward(
        self,
        video: torch.Tensor,
        image: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        compute_loss: bool = False,
        return_predictions: Optional[bool] = None,
        head_indices: Optional[List[int]] = None,
    ) -> STAPaperOutputs:
        if return_predictions is None:
            return_predictions = not self.training

        image_shapes = (
            [(int(t["size"][0]), int(t["size"][1])) for t in targets]
            if targets is not None
            else [(int(image.shape[-2]), int(image.shape[-1])) for _ in range(image.shape[0])]
        )

        z_v = self.backbone.encode_video(video)
        h_v = self.video_probe(z_v)
        fused_maps = self._build_fused_image_maps(image, image_shapes, h_v)
        pyramid = self._build_pyramid_batch(fused_maps)
        global_token = h_v.mean(dim=1)

        rpn_objectness, rpn_bbox_regression = self.rpn_head(pyramid)
        rpn_objectness_flat, rpn_bbox_regression_flat = self._flatten_rpn(
            rpn_objectness, rpn_bbox_regression
        )
        anchors = self.anchor_generator(pyramid, image_shapes)
        valid_masks = self._anchor_valid_masks(anchors, image_shapes)

        loss_mode = (targets is not None) and (self.training or compute_loss)
        loss_dict: Dict[str, torch.Tensor] = {}

        # 先只做 loss 分支
        if loss_mode:
            loss_dict.update(
                self._compute_rpn_losses(
                    rpn_objectness_flat,
                    rpn_bbox_regression_flat,
                    anchors,
                    targets,
                    valid_masks=valid_masks,
                )
            )

            proposals_train, proposal_scores_train = self._generate_proposals(
                rpn_objectness_flat.detach(),
                rpn_bbox_regression_flat.detach(),
                anchors,
                image_shapes,
                valid_masks=valid_masks,
                loss_mode=True,
            )
            (
                proposals_train,
                proposal_scores_train,
                labels,
                matched_boxes,
                matched_verbs,
                matched_ttc,
                matched_ious,
            ) = self._training_sample_proposals(proposals_train, proposal_scores_train, targets)

            _, fused_train = self._roi_features(pyramid, proposals_train, global_token, image_shapes)
            roi_loss_sums: Dict[str, torch.Tensor] = {}
            for head_idx, head in enumerate(self._iter_prediction_heads()):
                class_logits_tr, box_regression_tr, verb_logits_tr, ttc_pred_tr, score_logits_tr = self._run_prediction_head(
                    fused_train, head
                )
                head_losses = self._compute_roi_losses(
                    class_logits_tr,
                    box_regression_tr,
                    verb_logits_tr,
                    ttc_pred_tr,
                    score_logits_tr,
                    proposals_train,
                    labels,
                    matched_boxes,
                    matched_verbs,
                    matched_ttc,
                    matched_ious,
                )
                for name, value in head_losses.items():
                    roi_loss_sums[name] = roi_loss_sums.get(name, value.new_zeros(())) + value
                    loss_dict[f"loss_head{head_idx}_{name.removeprefix('loss_')}"] = value.detach()

            for name, value in roi_loss_sums.items():
                loss_dict[name] = value / float(self.num_prediction_heads)

            # 尽早释放这些大 tensor 的 Python 引用
            del fused_train
            del proposals_train, proposal_scores_train, labels, matched_boxes, matched_verbs, matched_ttc, matched_ious

        # 只有需要时才做 prediction 分支
        if return_predictions:
            proposals_pred, proposal_scores_pred = self._generate_proposals(
                rpn_objectness_flat.detach(),
                rpn_bbox_regression_flat.detach(),
                anchors,
                image_shapes,
                valid_masks=valid_masks,
                loss_mode=False,
            )
            _, fused_pred = self._roi_features(pyramid, proposals_pred, global_token, image_shapes)
            head_outputs: List[STAPaperHeadOutputs] = []
            selected_heads = self._select_prediction_heads(head_indices)
            selected_head_indices = [idx for idx, _ in selected_heads]
            for _, head in selected_heads:
                class_logits, box_regression, verb_logits, ttc_pred, score_logits = self._run_prediction_head(
                    fused_pred, head
                )
                pred_boxes_h, noun_logits_h, verb_logits_h, ttc_h, scores_h = self._inference_from_roi(
                    class_logits,
                    box_regression,
                    verb_logits,
                    ttc_pred,
                    score_logits,
                    proposals_pred,
                    proposal_scores_pred,
                    image_shapes,
                )
                head_outputs.append(
                    STAPaperHeadOutputs(
                        pred_boxes=pred_boxes_h,
                        noun_logits=noun_logits_h,
                        verb_logits=verb_logits_h,
                        ttc_pred=ttc_h,
                        scores=scores_h,
                    )
                )
            pred_boxes = head_outputs[0].pred_boxes
            noun_logits = head_outputs[0].noun_logits
            verb_logits_out = head_outputs[0].verb_logits
            ttc_out = head_outputs[0].ttc_pred
            scores_out = head_outputs[0].scores
            del fused_pred
        else:
            pred_boxes, noun_logits, verb_logits_out, ttc_out, scores_out = [], [], [], [], []
            head_outputs = None
            selected_head_indices = None

        if loss_dict:
            total_loss_keys = (
                "loss_rpn_objectness",
                "loss_rpn_box_reg",
                "loss_noun_classifier",
                "loss_box_reg",
                "loss_verb",
                "loss_ttc",
                "loss_interaction_score",
            )
            loss_dict["loss_total"] = sum(loss_dict[k] for k in total_loss_keys if k in loss_dict)
        else:
            loss_dict["loss_total"] = rpn_objectness_flat.sum() * 0.0

        return STAPaperOutputs(
            pred_boxes=pred_boxes,
            noun_logits=noun_logits,
            verb_logits=verb_logits_out,
            ttc_pred=ttc_out,
            scores=scores_out,
            loss_dict=loss_dict,
            head_outputs=head_outputs,
            selected_head_index=0,
            selected_head_indices=selected_head_indices,
            num_prediction_heads=self.num_prediction_heads,
        )


class CocoFasterRCNNVJEPASTAModel(nn.Module):
    """
    StillFast-style STA model:
    - COCO-pretrained Faster R-CNN still branch supplies ResNet-FPN, RPN, ROIAlign, and box head.
    - Frozen V-JEPA video branch supplies temporal context.
    - STA-specific multi-head predictors produce noun, verb, TTC, box, and quality scores.
    """

    def __init__(
        self,
        checkpoint: str,
        backbone_kwargs: Dict,
        frames_per_clip: int = 8,
        video_resolution: int = 384,
        num_nouns: int = 128,
        num_verbs: int = 81,
        proposal_topk_train: int = 2000,
        proposal_topk_test: int = 300,
        rpn_pre_nms_topk_train: int = 2000,
        rpn_pre_nms_topk_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        roi_pool_size: int = 7,
        probe_depth: int = 2,
        num_heads: int = 8,
        representation_size: int = 1024,
        class_topk_per_proposal: int = 5,
        verb_topk_per_proposal: int = 3,
        detections_per_img: int = 100,
        box_score_thresh: float = 0.001,
        box_nms_thresh: float = 0.55,
        box_reg_weights: Sequence[float] = (10.0, 10.0, 5.0, 5.0),
        inference_score: str = "objectness_quality_noun_verb",
        verb_loss_weight: float = 0.1,
        ttc_loss_weight: float = 0.5,
        score_loss_weight: float = 1.0,
        num_prediction_heads: int = 4,
        verb_background: bool = True,
        detector_pretrained: bool = True,
        detector_weights_path: Optional[str] = None,
        detector_trainable_backbone_layers: int = 3,
        freeze_detector_backbone: bool = False,
        freeze_detector_rpn: bool = False,
        freeze_detector_box_head: bool = False,
        fpn_context_fusion: bool = True,
        skip_vjepa_encoder_load: bool = True,
        cached_vjepa_dim: int = 1664,
        **unused_kwargs,
    ):
        super().__init__()
        if unused_kwargs:
            logger.info("Ignoring unused model kwargs for COCO detector model: %s", sorted(unused_kwargs))

        self.num_nouns = int(num_nouns)
        self.num_verbs = int(num_verbs)
        self.class_topk_per_proposal = max(1, int(class_topk_per_proposal))
        self.verb_topk_per_proposal = max(1, int(verb_topk_per_proposal))
        self.detections_per_img = int(detections_per_img)
        self.box_score_thresh = float(box_score_thresh)
        self.box_nms_thresh = float(box_nms_thresh)
        self.box_reg_weights = tuple(float(v) for v in box_reg_weights)
        if len(self.box_reg_weights) != 4:
            raise ValueError(f"box_reg_weights must contain four values, got {box_reg_weights}")
        self.inference_score = str(inference_score)
        self.verb_loss_weight = float(verb_loss_weight)
        self.ttc_loss_weight = float(ttc_loss_weight)
        self.score_loss_weight = float(score_loss_weight)
        self.num_prediction_heads = max(1, int(num_prediction_heads))
        self.verb_background = bool(verb_background)
        self.fpn_context_fusion = bool(fpn_context_fusion)

        weights = None
        if detector_pretrained and not detector_weights_path:
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        # Torchvision ignores trainable_backbone_layers when weights=None and would warn.
        # For local COCO checkpoints we load weights manually, then enforce the setting below.
        torchvision_trainable_layers = None if detector_weights_path else int(detector_trainable_backbone_layers)
        self.detector = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=torchvision_trainable_layers,
            rpn_pre_nms_top_n_train=int(rpn_pre_nms_topk_train),
            rpn_pre_nms_top_n_test=int(rpn_pre_nms_topk_test),
            rpn_post_nms_top_n_train=int(proposal_topk_train),
            rpn_post_nms_top_n_test=int(proposal_topk_test),
            rpn_nms_thresh=float(rpn_nms_thresh),
            box_score_thresh=float(box_score_thresh),
            box_nms_thresh=float(box_nms_thresh),
            box_detections_per_img=int(detections_per_img),
            box_roi_pool=None,
        )
        self.detector.roi_heads.box_coder.weights = self.box_reg_weights
        if detector_weights_path:
            self._load_detector_weights(detector_weights_path)
        self._set_detector_trainable_backbone_layers(int(detector_trainable_backbone_layers))

        # The COCO box predictor is not used; STA heads below replace it.
        for p in self.detector.roi_heads.box_predictor.parameters():
            p.requires_grad_(False)
        if freeze_detector_backbone:
            for p in self.detector.backbone.parameters():
                p.requires_grad_(False)
        if freeze_detector_rpn:
            for p in self.detector.rpn.parameters():
                p.requires_grad_(False)
        if freeze_detector_box_head:
            for p in self.detector.roi_heads.box_head.parameters():
                p.requires_grad_(False)

        if skip_vjepa_encoder_load:
            enc_kwargs = dict(backbone_kwargs.get("encoder", {}))
            self.video_encoder = CachedFeatureOnlyVideoEncoder(
                embed_dim=int(cached_vjepa_dim),
                patch_size=int(enc_kwargs.get("patch_size", 16)),
                tubelet_size=int(enc_kwargs.get("tubelet_size", 2)),
            )
            self.skip_vjepa_encoder_load = True
            logger.info(
                "Skipping V-JEPA encoder load; expecting cached video_global_token with dim=%d",
                self.video_encoder.embed_dim,
            )
        else:
            self.video_encoder = FrozenVJEPA2_1DualEncoder(
                checkpoint,
                backbone_kwargs,
                frames_per_clip,
                video_resolution,
            )
            self.skip_vjepa_encoder_load = False
        vjepa_dim = self.video_encoder.embed_dim
        detector_representation_size = int(self.detector.roi_heads.box_head.fc7.out_features)
        if detector_representation_size != int(representation_size):
            logger.info(
                "Using detector ROI representation_size=%d instead of config value=%d",
                detector_representation_size,
                int(representation_size),
            )
        self.representation_size = detector_representation_size

        self.video_probe = TemporalAttentiveProbe(dim=vjepa_dim, depth=probe_depth, num_heads=num_heads)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(vjepa_dim),
            nn.Linear(vjepa_dim, self.representation_size),
            nn.GELU(),
        )
        self.fpn_context_proj = nn.Sequential(
            nn.LayerNorm(vjepa_dim),
            nn.Linear(vjepa_dim, 512),
        )
        # Start from an identity FiLM. This lets the improved run reuse the
        # StillFast-style detector path without disturbing COCO initialization.
        nn.init.zeros_(self.fpn_context_proj[-1].weight)
        nn.init.zeros_(self.fpn_context_proj[-1].bias)
        self.context_fc = nn.Sequential(
            nn.Linear(self.representation_size * 2, self.representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.representation_size, self.representation_size),
            nn.ReLU(inplace=True),
        )
        self.prediction_head = STAPredictionHead(
            self.representation_size,
            self.num_nouns,
            self.num_verbs,
            verb_background=self.verb_background,
        )
        self.extra_prediction_heads = nn.ModuleList(
            [
                STAPredictionHead(
                    self.representation_size,
                    self.num_nouns,
                    self.num_verbs,
                    verb_background=self.verb_background,
                )
                for _ in range(self.num_prediction_heads - 1)
            ]
        )
        self._log_detector_trainable_params()

    @staticmethod
    def _count_trainable_params(module: nn.Module) -> int:
        return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)

    def _set_detector_trainable_backbone_layers(self, trainable_layers: int) -> None:
        """Mirror torchvision's ResNet-FPN trainable_backbone_layers after manual weight load."""
        trainable_layers = max(0, min(5, int(trainable_layers)))
        body = getattr(self.detector.backbone, "body", None)
        if body is None:
            logger.warning("Detector backbone has no .body; cannot enforce trainable_backbone_layers")
            return
        if trainable_layers == 5:
            layers_to_train = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
        else:
            layers_to_train = tuple(["layer4", "layer3", "layer2", "layer1"][:trainable_layers])
        for name, parameter in body.named_parameters():
            parameter.requires_grad_(any(name.startswith(layer) for layer in layers_to_train))
        logger.info(
            "Detector ResNet body trainable_backbone_layers=%d layers=%s",
            trainable_layers,
            ",".join(layers_to_train) if layers_to_train else "none",
        )

    def _log_detector_trainable_params(self) -> None:
        logger.info(
            "COCO detector trainable params: backbone=%d rpn=%d roi_box_head=%d sta_heads=%d video_probe=%d fpn_context_fusion=%s",
            self._count_trainable_params(self.detector.backbone),
            self._count_trainable_params(self.detector.rpn),
            self._count_trainable_params(self.detector.roi_heads.box_head),
            self._count_trainable_params(self.prediction_head) + self._count_trainable_params(self.extra_prediction_heads),
            self._count_trainable_params(self.video_probe),
            self.fpn_context_fusion,
        )

    def _load_detector_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(ckpt, dict) and key in ckpt:
                state = ckpt[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported detector checkpoint format: {path}")
        cleaned = {}
        for k, v in state.items():
            name = str(k)
            for prefix in ("module.", "detector."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            cleaned[name] = v
        msg = self.detector.load_state_dict(cleaned, strict=False)
        logger.info("Loaded detector weights from %s with msg: %s", path, msg)

    def _iter_prediction_heads(self):
        yield self.prediction_head
        for head in self.extra_prediction_heads:
            yield head

    def _select_prediction_heads(self, head_indices: Optional[List[int]] = None):
        heads = list(self._iter_prediction_heads())
        if head_indices is None:
            return list(enumerate(heads))
        selected = []
        seen = set()
        for idx in head_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(heads):
                raise ValueError(f"head index {idx} out of range for {len(heads)} prediction heads")
            if idx not in seen:
                selected.append((idx, heads[idx]))
                seen.add(idx)
        if not selected:
            raise ValueError("head_indices selected no prediction heads")
        return selected

    def _image_list(self, image: torch.Tensor, image_shapes: List[Tuple[int, int]]) -> ImageList:
        return ImageList(image, image_shapes)

    def _detector_targets(self, targets: List[Dict], device: torch.device) -> List[Dict[str, torch.Tensor]]:
        det_targets = []
        for t in targets:
            boxes = t["boxes"].to(device)
            labels = t["noun_labels"].to(device=device, dtype=torch.long) + 1
            det_targets.append({"boxes": boxes, "labels": labels})
        return det_targets

    def _video_global_token(self, video: torch.Tensor) -> torch.Tensor:
        z_v = self.video_encoder.encode_video(video)
        h_v = self.video_probe(z_v)
        return h_v.mean(dim=1)

    def _target_global_token(
        self,
        targets: Optional[List[Dict]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not targets:
            return None
        if not all("video_global_token" in t for t in targets):
            return None
        return torch.stack(
            [t["video_global_token"].to(device=device, dtype=torch.float32) for t in targets],
            dim=0,
        )

    def _fuse_fpn_context(
        self,
        features: Dict[str, torch.Tensor],
        global_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.fpn_context_fusion:
            return features
        gamma_beta = self.fpn_context_proj(global_token.float())
        gamma, beta = gamma_beta.chunk(2, dim=1)
        fused = {}
        for name, feat in features.items():
            if feat.shape[1] != gamma.shape[1]:
                fused[name] = feat
                continue
            g = torch.tanh(gamma).to(device=feat.device, dtype=feat.dtype).view(feat.shape[0], -1, 1, 1)
            b = beta.to(device=feat.device, dtype=feat.dtype).view(feat.shape[0], -1, 1, 1)
            fused[name] = feat * (1.0 + g) + b
        return fused

    def _roi_features(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        global_token: torch.Tensor,
    ) -> torch.Tensor:
        box_features = self.detector.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.detector.roi_heads.box_head(box_features)
        counts = [p.shape[0] for p in proposals]
        ctx_global = self.global_proj(global_token)
        fused = []
        start = 0
        for b, n in enumerate(counts):
            local = box_features[start : start + n]
            if n == 0:
                fused.append(local)
            else:
                glb = ctx_global[b].unsqueeze(0).expand(n, -1)
                fused.append(local + self.context_fc(torch.cat([local, glb], dim=1)))
            start += n
        if not fused:
            return box_features.new_zeros((0, self.representation_size))
        return torch.cat(fused, dim=0)

    def _run_prediction_head(self, fused: torch.Tensor, head: STAPredictionHead):
        noun_logits, box_regression, verb_logits, ttc_pred, score_logits = head(fused)
        return noun_logits, box_regression, verb_logits, ttc_pred, score_logits

    def _rpn_forward_with_scores(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        compute_losses: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Torchvision RPN forward that preserves objectness scores for STA ranking."""
        features_list = list(features.values())
        objectness, pred_bbox_deltas = self.detector.rpn.head(features_list)
        anchors = self.detector.rpn.anchor_generator(images, features_list)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            int(s[0] * s[1] * s[2]) for s in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = torchvision_concat_box_prediction_layers(
            objectness,
            pred_bbox_deltas,
        )
        proposals = self.detector.rpn.box_coder.decode(
            pred_bbox_deltas.detach(),
            anchors,
        )
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.detector.rpn.filter_proposals(
            proposals,
            objectness,
            images.image_sizes,
            num_anchors_per_level,
        )

        losses: Dict[str, torch.Tensor] = {}
        if compute_losses and self.detector.rpn.training:
            if targets is None:
                raise ValueError("targets should not be None when training RPN")
            labels, matched_gt_boxes = self.detector.rpn.assign_targets_to_anchors(
                anchors,
                targets,
            )
            regression_targets = self.detector.rpn.box_coder.encode(
                matched_gt_boxes,
                anchors,
            )
            loss_objectness, loss_rpn_box_reg = self.detector.rpn.compute_loss(
                objectness,
                pred_bbox_deltas,
                labels,
                regression_targets,
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, scores, losses

    def _matched_quality_targets(
        self,
        proposals: List[torch.Tensor],
        labels: List[torch.Tensor],
        matched_idxs: List[torch.Tensor],
        targets: List[Dict],
    ) -> List[torch.Tensor]:
        quality_targets = []
        for props, lbl, midx, tgt in zip(proposals, labels, matched_idxs, targets):
            gt_boxes = tgt["boxes"].to(props.device)
            quality = props.new_zeros((props.shape[0],))
            pos = lbl > 0
            if gt_boxes.numel() > 0 and pos.any():
                matched_boxes = gt_boxes[midx[pos]]
                quality[pos] = box_iou(props[pos], matched_boxes).diag().clamp(0.0, 1.0)
            quality_targets.append(quality)
        return quality_targets

    def _matched_sta_targets(
        self,
        labels: List[torch.Tensor],
        matched_idxs: List[torch.Tensor],
        targets: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        verbs = []
        ttcs = []
        for lbl, midx, tgt in zip(labels, matched_idxs, targets):
            device = lbl.device
            gt_verbs = tgt["verb_labels"].to(device=device, dtype=torch.long)
            gt_ttc = tgt["ttc"].to(device=device, dtype=torch.float32)
            if gt_verbs.numel() == 0:
                verbs.append(torch.zeros_like(lbl, dtype=torch.long))
                ttcs.append(torch.zeros((lbl.shape[0],), dtype=torch.float32, device=device))
                continue
            safe_idx = midx.clamp(min=0, max=max(int(gt_verbs.numel()) - 1, 0))
            verbs.append(gt_verbs[safe_idx])
            ttcs.append(gt_ttc[safe_idx])
        return torch.cat(verbs, dim=0), torch.cat(ttcs, dim=0)

    def _compute_sta_losses(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        verb_logits: torch.Tensor,
        ttc_pred: torch.Tensor,
        score_logits: torch.Tensor,
        proposals: List[torch.Tensor],
        matched_idxs: List[torch.Tensor],
        labels: List[torch.Tensor],
        regression_targets: List[torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits,
            box_regression,
            labels,
            regression_targets,
        )
        labels_cat = torch.cat(labels, dim=0)
        quality_targets = torch.cat(
            self._matched_quality_targets(proposals, labels, matched_idxs, targets),
            dim=0,
        ).to(device=score_logits.device, dtype=score_logits.dtype)
        loss_score = F.binary_cross_entropy_with_logits(score_logits, quality_targets)

        matched_verbs, matched_ttc = self._matched_sta_targets(labels, matched_idxs, targets)
        matched_verbs = matched_verbs.to(device=verb_logits.device, dtype=torch.long)
        matched_ttc = matched_ttc.to(device=ttc_pred.device, dtype=ttc_pred.dtype)
        pos = labels_cat.to(device=ttc_pred.device) > 0
        if self.verb_background:
            verb_targets = matched_verbs + 1
            verb_targets = torch.where(
                labels_cat.to(device=verb_targets.device) > 0,
                verb_targets,
                torch.zeros_like(verb_targets),
            )
            loss_verb = F.cross_entropy(verb_logits, verb_targets)
        elif pos.any():
            loss_verb = F.cross_entropy(verb_logits[pos], matched_verbs[pos])
        else:
            loss_verb = class_logits.sum() * 0.0
        if pos.any():
            loss_ttc = F.smooth_l1_loss(ttc_pred[pos], matched_ttc[pos])
        else:
            loss_ttc = class_logits.sum() * 0.0

        return {
            "loss_noun_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "loss_verb": self.verb_loss_weight * loss_verb,
            "loss_ttc": self.ttc_loss_weight * loss_ttc,
            "loss_interaction_score": self.score_loss_weight * loss_score,
        }

    def _postprocess_sta(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        verb_logits: torch.Tensor,
        ttc_pred: torch.Tensor,
        score_logits: torch.Tensor,
        proposals: List[torch.Tensor],
        proposal_scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        pred_boxes_out, noun_logits_out, verb_logits_out, ttc_out, scores_out = [], [], [], [], []
        start = 0
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_nouns + 1, 4)
        probs_all = F.softmax(class_logits, dim=1)
        for b, props in enumerate(proposals):
            n = props.shape[0]
            cls_b = class_logits[start : start + n]
            probs_b = probs_all[start : start + n]
            box_reg_b = box_regression[start : start + n]
            verb_b = verb_logits[start : start + n]
            ttc_b = ttc_pred[start : start + n]
            score_b = score_logits[start : start + n]
            proposal_scores_b = proposal_scores[b]
            start += n
            if n == 0:
                device = props.device
                pred_boxes_out.append(torch.zeros((0, 4), device=device))
                noun_logits_out.append(torch.zeros((0, self.num_nouns), device=device))
                verb_logits_out.append(torch.zeros((0, self.num_verbs), device=device))
                ttc_out.append(torch.zeros((0,), device=device))
                scores_out.append(torch.zeros((0,), device=device))
                continue

            fg_probs = probs_b[:, 1:]
            class_topk = min(self.class_topk_per_proposal, fg_probs.shape[1])
            noun_scores, labels = fg_probs.topk(class_topk, dim=1)
            proposal_indices = torch.arange(n, device=props.device).unsqueeze(1).expand(n, class_topk).reshape(-1)
            labels = labels.reshape(-1)
            noun_scores = noun_scores.reshape(-1)
            labels_plus = labels + 1
            boxes = decode_boxes(
                box_reg_b[proposal_indices, labels_plus],
                props[proposal_indices],
                weights=self.box_reg_weights,
            )
            boxes = clip_boxes_to_image(boxes, image_shapes[b])
            keep = remove_small_boxes(boxes, 1.0)
            boxes = boxes[keep]
            noun_scores = noun_scores[keep]
            labels = labels[keep]
            proposal_indices = proposal_indices[keep]

            verb_scores_all = verb_b[proposal_indices].softmax(dim=-1)
            if self.verb_background:
                verb_scores_all = verb_scores_all[:, 1:]
            verb_topk = min(self.verb_topk_per_proposal, verb_scores_all.shape[1])
            verb_scores, verb_labels = verb_scores_all.topk(verb_topk, dim=1)
            boxes = boxes.unsqueeze(1).expand(-1, verb_topk, -1).reshape(-1, 4)
            noun_scores = noun_scores.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            labels = labels.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            proposal_indices = proposal_indices.unsqueeze(1).expand(-1, verb_topk).reshape(-1)
            verb_scores = verb_scores.reshape(-1)
            verb_labels = verb_labels.reshape(-1)

            quality_prob = score_b[proposal_indices].sigmoid()
            objectness_prob = proposal_scores_b[proposal_indices].clamp(0.0, 1.0)
            if self.inference_score == "noun":
                scores = noun_scores
            elif self.inference_score == "noun_verb":
                scores = noun_scores * verb_scores
            elif self.inference_score == "quality_noun_verb":
                scores = quality_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness_noun_verb":
                scores = objectness_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness_quality_noun_verb":
                scores = objectness_prob * quality_prob * noun_scores * verb_scores
            elif self.inference_score == "objectness":
                scores = objectness_prob
            elif self.inference_score == "quality":
                scores = quality_prob
            else:
                raise ValueError(f"Unsupported inference_score={self.inference_score}")

            score_keep = scores >= self.box_score_thresh
            boxes = boxes[score_keep]
            scores = scores[score_keep]
            labels = labels[score_keep]
            proposal_indices = proposal_indices[score_keep]
            verb_labels = verb_labels[score_keep]
            ttc_keep = ttc_b[proposal_indices].clamp_min(0.0)

            noun_logits_keep = cls_b.new_full((labels.shape[0], self.num_nouns), -50.0)
            if labels.numel() > 0:
                noun_logits_keep[torch.arange(labels.shape[0], device=labels.device), labels] = 50.0
            verb_logits_keep = verb_b.new_full((labels.shape[0], self.num_verbs), -50.0)
            if verb_labels.numel() > 0:
                verb_logits_keep[torch.arange(verb_labels.shape[0], device=verb_labels.device), verb_labels] = 50.0

            if boxes.numel() > 0:
                compound_labels = labels * self.num_verbs + verb_labels
                keep_all = []
                for cls_id in compound_labels.unique():
                    cls_inds = torch.where(compound_labels == cls_id)[0]
                    cls_keep = nms(boxes[cls_inds], scores[cls_inds], self.box_nms_thresh)
                    keep_all.append(cls_inds[cls_keep])
                keep_nms = torch.cat(keep_all, dim=0) if keep_all else labels.new_zeros((0,), dtype=torch.long)
                keep_nms = keep_nms[torch.argsort(scores[keep_nms], descending=True)]
                keep_nms = keep_nms[: self.detections_per_img]
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                ttc_keep = ttc_keep[keep_nms]
                noun_logits_keep = noun_logits_keep[keep_nms]
                verb_logits_keep = verb_logits_keep[keep_nms]

            pred_boxes_out.append(boxes)
            noun_logits_out.append(noun_logits_keep)
            verb_logits_out.append(verb_logits_keep)
            ttc_out.append(ttc_keep)
            scores_out.append(scores)
        return pred_boxes_out, noun_logits_out, verb_logits_out, ttc_out, scores_out

    @torch.no_grad()
    def predict_rpn_proposals(
        self,
        video: torch.Tensor,
        image: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        topk: Optional[int] = None,
        pre_nms_topk: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        del video, pre_nms_topk
        image_shapes = (
            [(int(t["size"][0]), int(t["size"][1])) for t in targets]
            if targets is not None
            else [(int(image.shape[-2]), int(image.shape[-1])) for _ in range(image.shape[0])]
        )
        images = self._image_list(image, image_shapes)
        features = self.detector.backbone(images.tensors)
        if targets is not None and "video_global_token" in targets[0]:
            global_token = torch.stack([t["video_global_token"].to(image.device) for t in targets], dim=0)
            features = self._fuse_fpn_context(features, global_token)
        proposals, scores, _ = self._rpn_forward_with_scores(
            images,
            features,
            None,
            compute_losses=False,
        )
        if topk is not None:
            proposals = [p[: int(topk)] for p in proposals]
            scores = [s[: int(topk)] for s in scores]
        return proposals, scores

    def forward(
        self,
        video: torch.Tensor,
        image: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        compute_loss: bool = False,
        return_predictions: Optional[bool] = None,
        head_indices: Optional[List[int]] = None,
    ) -> STAPaperOutputs:
        if return_predictions is None:
            return_predictions = not self.training

        image_shapes = (
            [(int(t["size"][0]), int(t["size"][1])) for t in targets]
            if targets is not None
            else [(int(image.shape[-2]), int(image.shape[-1])) for _ in range(image.shape[0])]
        )
        images = self._image_list(image, image_shapes)
        det_targets = self._detector_targets(targets, image.device) if targets is not None else None

        global_token = self._target_global_token(targets, image.device)
        if global_token is None:
            global_token = self._video_global_token(video)
        features = self.detector.backbone(images.tensors)
        features = self._fuse_fpn_context(features, global_token)

        loss_mode = (targets is not None) and (self.training or compute_loss)
        rpn_targets = det_targets if self.training and det_targets is not None else None
        proposals, proposal_scores, rpn_losses = self._rpn_forward_with_scores(
            images,
            features,
            rpn_targets,
            compute_losses=(rpn_targets is not None),
        )
        loss_dict: Dict[str, torch.Tensor] = {}
        if loss_mode:
            if "loss_objectness" in rpn_losses:
                loss_dict["loss_rpn_objectness"] = rpn_losses["loss_objectness"]
            if "loss_rpn_box_reg" in rpn_losses:
                loss_dict["loss_rpn_box_reg"] = rpn_losses["loss_rpn_box_reg"]
            proposals_train, matched_idxs, labels, regression_targets = self.detector.roi_heads.select_training_samples(
                proposals,
                det_targets,
            )
            fused_train = self._roi_features(features, proposals_train, image_shapes, global_token)
            roi_loss_sums: Dict[str, torch.Tensor] = {}
            for head_idx, head in enumerate(self._iter_prediction_heads()):
                class_logits_tr, box_regression_tr, verb_logits_tr, ttc_pred_tr, score_logits_tr = self._run_prediction_head(
                    fused_train,
                    head,
                )
                head_losses = self._compute_sta_losses(
                    class_logits_tr,
                    box_regression_tr,
                    verb_logits_tr,
                    ttc_pred_tr,
                    score_logits_tr,
                    proposals_train,
                    matched_idxs,
                    labels,
                    regression_targets,
                    targets,
                )
                for name, value in head_losses.items():
                    roi_loss_sums[name] = roi_loss_sums.get(name, value.new_zeros(())) + value
                    loss_dict[f"loss_head{head_idx}_{name.removeprefix('loss_')}"] = value.detach()
            for name, value in roi_loss_sums.items():
                loss_dict[name] = value / float(self.num_prediction_heads)

        if return_predictions:
            fused_pred = self._roi_features(features, proposals, image_shapes, global_token)
            head_outputs: List[STAPaperHeadOutputs] = []
            selected_heads = self._select_prediction_heads(head_indices)
            selected_head_indices = [idx for idx, _ in selected_heads]
            for _, head in selected_heads:
                class_logits, box_regression, verb_logits, ttc_pred, score_logits = self._run_prediction_head(
                    fused_pred,
                    head,
                )
                pred_boxes_h, noun_logits_h, verb_logits_h, ttc_h, scores_h = self._postprocess_sta(
                    class_logits,
                    box_regression,
                    verb_logits,
                    ttc_pred,
                    score_logits,
                    proposals,
                    proposal_scores,
                    image_shapes,
                )
                head_outputs.append(
                    STAPaperHeadOutputs(
                        pred_boxes=pred_boxes_h,
                        noun_logits=noun_logits_h,
                        verb_logits=verb_logits_h,
                        ttc_pred=ttc_h,
                        scores=scores_h,
                    )
                )
            pred_boxes = head_outputs[0].pred_boxes
            noun_logits = head_outputs[0].noun_logits
            verb_logits_out = head_outputs[0].verb_logits
            ttc_out = head_outputs[0].ttc_pred
            scores_out = head_outputs[0].scores
        else:
            pred_boxes, noun_logits, verb_logits_out, ttc_out, scores_out = [], [], [], [], []
            head_outputs = None
            selected_head_indices = None

        if loss_dict:
            total_loss_keys = (
                "loss_rpn_box_reg",
                "loss_rpn_objectness",
                "loss_noun_classifier",
                "loss_box_reg",
                "loss_verb",
                "loss_ttc",
                "loss_interaction_score",
            )
            loss_dict["loss_total"] = sum(loss_dict[k] for k in total_loss_keys if k in loss_dict)
        else:
            loss_dict["loss_total"] = image.sum() * 0.0

        return STAPaperOutputs(
            pred_boxes=pred_boxes,
            noun_logits=noun_logits,
            verb_logits=verb_logits_out,
            ttc_pred=ttc_out,
            scores=scores_out,
            loss_dict=loss_dict,
            head_outputs=head_outputs,
            selected_head_index=0,
            selected_head_indices=selected_head_indices,
            num_prediction_heads=self.num_prediction_heads,
        )


# Keep eval.py unchanged while switching this new package to the COCO detector model.
LegacyVJEPATokenDetectorSTAPaperFaithfulModel = STAPaperFaithfulModel
STAPaperFaithfulModel = CocoFasterRCNNVJEPASTAModel
