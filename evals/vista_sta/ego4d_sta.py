from __future__ import annotations

import json
import os
import random
# from dataclasses import dataclass
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import logging
import hashlib
import time
import gc
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)
_PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def _cache_root() -> Path:
    root = Path(
        os.environ.get(
            "EGO4D_STA_CACHE_ROOT",
            str(Path(__file__).resolve().parent / "cache"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _wait_for_cache_file(path: Path) -> None:
    timeout_s = int(os.environ.get("EGO4D_STA_CACHE_TIMEOUT_SEC", "7200"))
    poll_s = float(os.environ.get("EGO4D_STA_CACHE_POLL_SEC", "5"))
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for cache file: {path}")
        time.sleep(poll_s)


def _load_label_map_cache(path: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    noun_id_to_idx = {int(k): int(v) for k, v in payload["noun_id_to_idx"].items()}
    verb_id_to_idx = {int(k): int(v) for k, v in payload["verb_id_to_idx"].items()}
    return noun_id_to_idx, verb_id_to_idx


def _count_samples_in_cache(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload["samples"])


def _read_rgb_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), copy=True)


def _read_rgb_image_square(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        size = int(size)
        if size > 0 and img.size != (size, size):
            img = img.resize((size, size), _PIL_BILINEAR)
        return np.array(img, copy=True)


def jepa_global_feature_path(feature_root: str, sample: "STASampleDirItem") -> Path:
    safe_uid = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(sample.uid))
    split = str(sample.split or "unknown")
    return Path(feature_root) / split / f"{safe_uid}.pt"


def _normalize_clip(clip: torch.Tensor) -> torch.Tensor:
    if clip.dtype != torch.float32:
        clip = clip.float()
    clip.div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype, device=clip.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype, device=clip.device)
    clip.sub_(mean.view(1, 1, 1, 3)).div_(std.view(1, 1, 1, 3))
    return clip.permute(3, 0, 1, 2).contiguous()


def _normalize_image(img: torch.Tensor) -> torch.Tensor:
    if img.dtype != torch.float32:
        img = img.float()
    img.div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device)
    img.sub_(mean.view(1, 1, 3)).div_(std.view(1, 1, 3))
    return img.permute(2, 0, 1).contiguous()


def _resize_short_side_image(
    img: torch.Tensor,
    short_side: int,
    max_size: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    _, h, w = img.shape
    scale = float(short_side) / float(min(h, w))
    if max_size is not None and max(h, w) * scale > max_size:
        scale = float(max_size) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return x, scale


def _resize_square_clip(clip: torch.Tensor, size: int) -> torch.Tensor:
    x = clip.permute(1, 0, 2, 3)  # [T, C, H, W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3).contiguous()


def _sample_history_frame_paths(paths: Sequence[str], frames_per_clip: int) -> List[str]:
    """Return a fixed-length clip while preserving temporal order."""
    if len(paths) == 0:
        raise ValueError("history_frame_paths is empty")
    frames_per_clip = int(frames_per_clip)
    if frames_per_clip <= 0:
        return list(paths)
    if len(paths) == frames_per_clip:
        return list(paths)
    if len(paths) > frames_per_clip:
        return list(paths[len(paths) - frames_per_clip :])
    pad = [paths[0]] * (frames_per_clip - len(paths))
    return pad + list(paths)


def _horizontal_flip_clip(clip: torch.Tensor) -> torch.Tensor:
    return torch.flip(clip, dims=[-1])


def _horizontal_flip_image_and_boxes(
    img: torch.Tensor,
    boxes_xyxy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, w = img.shape
    img = torch.flip(img, dims=[-1])
    if boxes_xyxy.numel() == 0:
        return img, boxes_xyxy
    boxes = boxes_xyxy.clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = w - x2
    boxes[:, 2] = w - x1
    return img, boxes


def _clip_boxes_to_image(boxes_xyxy: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    _, h, w = image.shape
    boxes = boxes_xyxy.clone()
    max_x1 = float(max(w - 2, 0))
    max_y1 = float(max(h - 2, 0))
    max_x2 = float(max(w - 1, 1))
    max_y2 = float(max(h - 1, 1))
    boxes[:, 0] = boxes[:, 0].clamp(0.0, max_x1)
    boxes[:, 1] = boxes[:, 1].clamp(0.0, max_y1)
    boxes[:, 2] = boxes[:, 2].clamp(0.0, max_x2)
    boxes[:, 3] = boxes[:, 3].clamp(0.0, max_y2)
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0).clamp(max=max_x2)
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0).clamp(max=max_y2)
    return boxes


def _extract_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 优先 future_prediction_target.objects
    fut = payload.get("future_prediction_target", {})
    objs = fut.get("objects", [])
    if isinstance(objs, list) and len(objs) > 0:
        return objs

    # fallback 到 raw_annotation.objects
    raw = payload.get("raw_annotation", {})
    objs = raw.get("objects", [])
    return objs if isinstance(objs, list) else []


def _extract_box(obj: Dict[str, Any]):
    return obj.get("box_xyxy") or obj.get("box") or obj.get("bbox")


# def build_label_maps_from_roots(
#     split_roots: List[str],
# ) -> Tuple[Dict[int, int], Dict[int, int]]:
#     noun_ids = set()
#     verb_ids = set()
    
#     for split_root in split_roots:
#         root_path = Path(split_root)
#         if not root_path.exists():
#             continue

#         for lp in sorted(root_path.rglob("label.json")):
#             with open(lp, "r", encoding="utf-8") as f:
#                 payload = json.load(f)

#             objs = _extract_objects(payload)
#             for obj in objs:
#                 noun_raw = obj.get("noun_category_id", obj.get("noun_id"))
#                 verb_raw = obj.get("verb_category_id", obj.get("verb_id"))
#                 if noun_raw is not None and verb_raw is not None:
#                     noun_ids.add(int(noun_raw))
#                     verb_ids.add(int(verb_raw))

#     if len(noun_ids) == 0 or len(verb_ids) == 0:
#         raise ValueError(
#             f"No valid noun/verb ids found in provided roots: {split_roots}"
#         )

#     noun_id_to_idx = {k: i for i, k in enumerate(sorted(noun_ids))}
#     verb_id_to_idx = {k: i for i, k in enumerate(sorted(verb_ids))}
#     return noun_id_to_idx, verb_id_to_idx

def _label_map_cache_path(split_roots: List[str]) -> Path:
    roots = [str(Path(p).resolve()) for p in split_roots]
    key = hashlib.md5("||".join(sorted(roots)).encode("utf-8")).hexdigest()[:12]
    return _cache_root() / f"label_maps_{key}.json"

def _sample_manifest_cache_path(
    split_root: str,
    require_labels: bool,
    noun_id_to_idx: Dict[int, int],
    verb_id_to_idx: Dict[int, int],
) -> Path:
    sig = {
        "split_root": str(Path(split_root).resolve()),
        "require_labels": bool(require_labels),
        "noun_ids": sorted(int(k) for k in noun_id_to_idx.keys()),
        "verb_ids": sorted(int(k) for k in verb_id_to_idx.keys()),
    }
    key = hashlib.md5(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return _cache_root() / f"samples_{key}.json"

def build_label_maps_from_roots(
    split_roots: List[str],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    split_roots = [str(Path(p).resolve()) for p in split_roots]
    cache_path = _label_map_cache_path(split_roots)

    if cache_path.exists():
        if rank == 0:
            logger.info("[label_map] loading cache from %s", cache_path)
        return _load_label_map_cache(cache_path)

    if distributed and rank != 0:
        if rank == 1:
            logger.info("[label_map] waiting for rank0 to build cache: %s", cache_path)
        _wait_for_cache_file(cache_path)
        return _load_label_map_cache(cache_path)

    noun_ids = set()
    verb_ids = set()

    for split_root in split_roots:
        root_path = Path(split_root)
        if not root_path.exists():
            if rank == 0:
                logger.info("[label_map] split_root not found: %s", split_root)
            continue

        label_paths = sorted(root_path.rglob("label.json"))
        if rank == 0:
            logger.info(
                "[label_map] start split_root=%s label_files=%d",
                split_root, len(label_paths)
            )

        t0 = time.time()
        for i, lp in enumerate(label_paths, 1):
            with open(lp, "r", encoding="utf-8") as f:
                payload = json.load(f)

            objs = _extract_objects(payload)
            for obj in objs:
                noun_raw = obj.get("noun_category_id", obj.get("noun_id"))
                verb_raw = obj.get("verb_category_id", obj.get("verb_id"))
                if noun_raw is not None and verb_raw is not None:
                    noun_ids.add(int(noun_raw))
                    verb_ids.add(int(verb_raw))

            if rank == 0 and (i % 2000 == 0 or i == len(label_paths)):
                logger.info(
                    "[label_map] split_root=%s progress=%d/%d elapsed=%.1fs",
                    split_root, i, len(label_paths), time.time() - t0
                )

    if len(noun_ids) == 0 or len(verb_ids) == 0:
        raise ValueError(f"No valid noun/verb ids found in provided roots: {split_roots}")

    noun_id_to_idx = {k: i for i, k in enumerate(sorted(noun_ids))}
    verb_id_to_idx = {k: i for i, k in enumerate(sorted(verb_ids))}

    if rank == 0:
        payload = {
            "split_roots": split_roots,
            "noun_id_to_idx": noun_id_to_idx,
            "verb_id_to_idx": verb_id_to_idx,
        }
        _atomic_write_json(cache_path, payload)
        logger.info(
            "[label_map] cache written to %s (nouns=%d verbs=%d)",
            cache_path, len(noun_id_to_idx), len(verb_id_to_idx)
        )

    return noun_id_to_idx, verb_id_to_idx


def prebuild_sta_caches(
    train_split_root: str,
    val_split_root: str,
    test_split_root: Optional[str] = None,
    build_test_cache: bool = False,
) -> Dict[str, Any]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    noun_id_to_idx, verb_id_to_idx = build_label_maps_from_roots([train_split_root, val_split_root])
    label_map_cache = _label_map_cache_path([train_split_root, val_split_root])

    train_cache = _sample_manifest_cache_path(
        split_root=train_split_root,
        require_labels=True,
        noun_id_to_idx=noun_id_to_idx,
        verb_id_to_idx=verb_id_to_idx,
    )
    val_cache = _sample_manifest_cache_path(
        split_root=val_split_root,
        require_labels=True,
        noun_id_to_idx=noun_id_to_idx,
        verb_id_to_idx=verb_id_to_idx,
    )

    test_cache = None
    if build_test_cache and test_split_root is not None:
        test_cache = _sample_manifest_cache_path(
            split_root=test_split_root,
            require_labels=False,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
        )

    if rank != 0:
        _wait_for_cache_file(label_map_cache)
        _wait_for_cache_file(train_cache)
        _wait_for_cache_file(val_cache)
        if test_cache is not None:
            _wait_for_cache_file(test_cache)
        test_count = _count_samples_in_cache(test_cache) if test_cache is not None and test_cache.exists() else None
        return {
            "cache_root": str(_cache_root()),
            "num_nouns": len(noun_id_to_idx),
            "num_verbs": len(verb_id_to_idx),
            "train_samples": _count_samples_in_cache(train_cache),
            "val_samples": _count_samples_in_cache(val_cache),
            "test_samples": test_count,
        }

    if not train_cache.exists():
        train_ds = Ego4DSTASampleDirDataset(
            split_root=train_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=True,
            train=True,
        )
        del train_ds
        gc.collect()
    if not val_cache.exists():
        val_ds = Ego4DSTASampleDirDataset(
            split_root=val_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=True,
            train=False,
        )
        del val_ds
        gc.collect()
    test_count = None
    if test_cache is not None and not test_cache.exists():
        test_ds = Ego4DSTASampleDirDataset(
            split_root=test_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=False,
            train=False,
        )
        del test_ds
        gc.collect()

    train_count = _count_samples_in_cache(train_cache)
    val_count = _count_samples_in_cache(val_cache)
    if test_cache is not None and test_cache.exists():
        test_count = _count_samples_in_cache(test_cache)

    return {
        "cache_root": str(_cache_root()),
        "num_nouns": len(noun_id_to_idx),
        "num_verbs": len(verb_id_to_idx),
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
    }

@dataclass
class STASampleDirItem:
    uid: str
    split: str
    sample_dir: str
    video_uid: str
    frame: int
    history_frame_paths: List[str]
    highres_image_path: str
    boxes_xyxy: List[List[float]]
    noun_ids_raw: List[int]
    verb_ids_raw: List[int]
    ttc: List[float]
    has_labels: bool


class Ego4DSTASampleDirDataset(Dataset):
    def __init__(
        self,
        split_root: str,
        noun_id_to_idx: Optional[Dict[int, int]] = None,
        verb_id_to_idx: Optional[Dict[int, int]] = None,
        build_label_map: bool = False,
        require_labels: Optional[bool] = None,
        video_short_side: int = 384,
        image_short_side_train: Sequence[int] = (640, 672, 704, 736, 768, 800),
        image_short_side_val: int = 800,
        image_max_size: int = 1333,
        train: bool = True,
        horizontal_flip_prob: float = 0.0,
        frames_per_clip: int = 8,
        jepa_feature_root: Optional[str] = None,
        use_jepa_feature_cache: bool = False,
    ):
        super().__init__()
        self.split_root = split_root
        self.train = train
        self.video_short_side = int(video_short_side)
        self.image_short_side_train = list(image_short_side_train)
        self.image_short_side_val = int(image_short_side_val)
        self.image_max_size = int(image_max_size)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.frames_per_clip = int(frames_per_clip)
        self.jepa_feature_root = jepa_feature_root
        self.use_jepa_feature_cache = bool(use_jepa_feature_cache)

        root_path = Path(split_root)
        if not root_path.exists():
            raise FileNotFoundError(f"split_root does not exist: {split_root}")

        if require_labels is None:
            require_labels = root_path.name.lower() in {"train", "val"}
        self.require_labels = bool(require_labels)

        if build_label_map:
            noun_id_to_idx, verb_id_to_idx = build_label_maps_from_roots([split_root])

        assert noun_id_to_idx is not None and verb_id_to_idx is not None
        self.noun_id_to_idx = noun_id_to_idx
        self.verb_id_to_idx = verb_id_to_idx
        self.idx_to_noun_id = {v: k for k, v in self.noun_id_to_idx.items()}
        self.idx_to_verb_id = {v: k for k, v in self.verb_id_to_idx.items()}

        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        rank0 = rank == 0

        sample_cache_path = _sample_manifest_cache_path(
            split_root=split_root,
            require_labels=self.require_labels,
            noun_id_to_idx=self.noun_id_to_idx,
            verb_id_to_idx=self.verb_id_to_idx,
        )

        def _load_sample_cache(path: Path) -> List[STASampleDirItem]:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return [STASampleDirItem(**x) for x in payload["samples"]]

        # 先尝试直接读缓存，避免任何目录扫描
        if sample_cache_path.exists():
            if rank0:
                logger.info("[dataset-cache] loading cache from %s", sample_cache_path)
            self.samples = _load_sample_cache(sample_cache_path)
            if len(self.samples) == 0:
                raise ValueError(f"sample cache is empty: {sample_cache_path}")
            return

        # 多卡时，非 rank0 等 rank0 生成缓存
        if distributed and not rank0:
            if rank == 1:
                logger.info("[dataset-cache] waiting for rank0 to build cache: %s", sample_cache_path)
            _wait_for_cache_file(sample_cache_path)
            self.samples = _load_sample_cache(sample_cache_path)
            if len(self.samples) == 0:
                raise ValueError(f"sample cache is empty: {sample_cache_path}")
            return

        # 只有 rank0 或单卡且缓存不存在时，才真正扫描目录
        label_paths = sorted(root_path.rglob("label.json"))

        if rank0:
            logger.info(
                "[dataset-init] split_root=%s label_files=%d require_labels=%s",
                split_root, len(label_paths), self.require_labels
            )

        if len(label_paths) == 0:
            raise FileNotFoundError(
                f"No label.json found under split_root={split_root}"
            )

        self.samples: List[STASampleDirItem] = []
        kept = 0
        skipped_unlabeled = 0
        skipped_invalid = 0

        t0 = time.time()
        for i, lp in enumerate(label_paths, 1):
            sd = str(lp.parent)
            with open(lp, "r", encoding="utf-8") as f:
                payload = json.load(f)

            split_name = str(payload.get("split", root_path.name)).lower()
            observation = payload["observation"]

            objects = _extract_objects(payload)
            has_labels = len(objects) > 0

            if self.require_labels and not has_labels:
                skipped_unlabeled += 1
                continue

            history_frame_paths = [
                os.path.join(sd, x["file"])
                for x in observation["history_frames"]
            ]
            highres_image_path = os.path.join(
                sd,
                observation["last_history_frame_highres_file"],
            )

            boxes: List[List[float]] = []
            nouns: List[int] = []
            verbs: List[int] = []
            ttcs: List[float] = []

            for obj in objects:
                box = _extract_box(obj)
                noun_raw = obj.get("noun_category_id", obj.get("noun_id"))
                verb_raw = obj.get("verb_category_id", obj.get("verb_id"))
                ttc_raw = obj.get("time_to_contact", obj.get("ttc"))

                if box is None or noun_raw is None or verb_raw is None or ttc_raw is None:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box]
                if x2 <= x1 or y2 <= y1:
                    continue

                noun_raw = int(noun_raw)
                verb_raw = int(verb_raw)
                if noun_raw not in self.noun_id_to_idx or verb_raw not in self.verb_id_to_idx:
                    continue

                boxes.append([x1, y1, x2, y2])
                nouns.append(self.noun_id_to_idx[noun_raw])
                verbs.append(self.verb_id_to_idx[verb_raw])
                ttcs.append(float(ttc_raw))

            if self.require_labels and len(boxes) == 0:
                skipped_invalid += 1
                continue

            self.samples.append(
                STASampleDirItem(
                    uid=str(payload["uid"]),
                    split=split_name,
                    sample_dir=sd,
                    video_uid=str(payload["video_uid"]),
                    frame=int(payload["frame"]),
                    history_frame_paths=history_frame_paths,
                    highres_image_path=highres_image_path,
                    boxes_xyxy=boxes,
                    noun_ids_raw=nouns,
                    verb_ids_raw=verbs,
                    ttc=ttcs,
                    has_labels=has_labels,
                )
            )
            kept += 1

            if rank0 and (i % 2000 == 0 or i == len(label_paths)):
                logger.info(
                    "[dataset-init] split_root=%s progress=%d/%d kept=%d skipped_unlabeled=%d skipped_invalid=%d elapsed=%.1fs",
                    split_root, i, len(label_paths), kept, skipped_unlabeled, skipped_invalid, time.time() - t0
                )
        
        if len(self.samples) == 0:
            raise ValueError(
                f"After filtering, dataset is empty for split_root={split_root}. "
                f"require_labels={self.require_labels}"
            )

        if rank0:
            payload = {
                "split_root": split_root,
                "require_labels": self.require_labels,
                "samples": [asdict(x) for x in self.samples],
            }
            _atomic_write_json(sample_cache_path, payload)
            logger.info(
                "[dataset-cache] cache written to %s samples=%d",
                sample_cache_path,
                len(self.samples),
            )
            
        if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
            print(
                f"[Ego4DSTASampleDirDataset] split_root={split_root} "
                f"label_files={len(label_paths)} kept={kept} "
                f"skipped_unlabeled={skipped_unlabeled} "
                f"skipped_invalid={skipped_invalid} "
                f"require_labels={self.require_labels}"
            )
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        cached_global_token = None
        if self.use_jepa_feature_cache and self.jepa_feature_root:
            feat_path = jepa_global_feature_path(self.jepa_feature_root, sample)
            if feat_path.exists():
                cached_global_token = torch.load(feat_path, map_location="cpu").float()

        if cached_global_token is None:
            history_frame_paths = _sample_history_frame_paths(
                sample.history_frame_paths,
                self.frames_per_clip,
            )
            clip_np = np.stack(
                [_read_rgb_image_square(p, self.video_short_side) for p in history_frame_paths],
                axis=0,
            )
            clip = _normalize_clip(torch.from_numpy(clip_np))
        else:
            clip = torch.zeros((3, 1, 1, 1), dtype=torch.float32)
        img_np = _read_rgb_image(sample.highres_image_path)
        image = _normalize_image(torch.from_numpy(img_np))

        boxes = (
            torch.tensor(sample.boxes_xyxy, dtype=torch.float32)
            if len(sample.boxes_xyxy) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        noun_labels = (
            torch.tensor(sample.noun_ids_raw, dtype=torch.long)
            if len(sample.noun_ids_raw) > 0
            else torch.zeros((0,), dtype=torch.long)
        )
        verb_labels = (
            torch.tensor(sample.verb_ids_raw, dtype=torch.long)
            if len(sample.verb_ids_raw) > 0
            else torch.zeros((0,), dtype=torch.long)
        )
        ttc = (
            torch.tensor(sample.ttc, dtype=torch.float32)
            if len(sample.ttc) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        image_short_side = (
            random.choice(self.image_short_side_train)
            if self.train
            else self.image_short_side_val
        )
        image, scale = _resize_short_side_image(
            image,
            short_side=image_short_side,
            max_size=self.image_max_size,
        )
        if boxes.numel() > 0:
            boxes = boxes * scale
            boxes = _clip_boxes_to_image(boxes, image)

        if self.train and random.random() < self.horizontal_flip_prob:
            clip = _horizontal_flip_clip(clip)
            image, boxes = _horizontal_flip_image_and_boxes(image, boxes)
            boxes = _clip_boxes_to_image(boxes, image)

        target = {
            "uid": sample.uid,
            "boxes": boxes,
            "noun_labels": noun_labels,
            "verb_labels": verb_labels,
            "ttc": ttc,
            "size": torch.tensor(
                [image.shape[-2], image.shape[-1]],
                dtype=torch.long,
            ),
            "orig_size": torch.tensor(
                [img_np.shape[0], img_np.shape[1]],
                dtype=torch.long,
            ),
            "video_id": sample.video_uid,
            "frame": sample.frame,
            "has_labels": sample.has_labels,
        }
        if cached_global_token is not None:
            target["video_global_token"] = cached_global_token

        return {
            "video": clip,
            "image": image,
            "target": target,
        }


def collate_sta_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = torch.stack([x["video"] for x in batch], dim=0)

    max_h = max(x["image"].shape[-2] for x in batch)
    max_w = max(x["image"].shape[-1] for x in batch)
    # Match torchvision detection padding convention for FPN/RPN stride alignment.
    size_divisible = 32
    max_h = ((max_h + size_divisible - 1) // size_divisible) * size_divisible
    max_w = ((max_w + size_divisible - 1) // size_divisible) * size_divisible
    images = []
    for item in batch:
        img = item["image"]
        padded = img.new_zeros((img.shape[0], max_h, max_w))
        padded[:, : img.shape[-2], : img.shape[-1]] = img
        images.append(padded)
    images = torch.stack(images, dim=0)

    targets = [x["target"] for x in batch]
    return {"video": videos, "image": images, "targets": targets}
