
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
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


class SetCriterion(nn.Module):
    """
    Criterion wrapper for STAPaperFaithfulModel.

    The model computes training losses internally and exposes:
      outputs.loss_dict = {
        "loss_rpn_objectness": ...,
        "loss_rpn_box_reg": ...,
        "loss_noun_classifier": ...,
        "loss_box_reg": ...,
        "loss_verb": ...,
        "loss_ttc": ...,
        "loss_total": ...
      }

    We keep the criterion API expected by eval.py.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, outputs, targets: List[Dict]):
        del targets
        loss_dict = getattr(outputs, "loss_dict", None)
        if loss_dict is None:
            raise ValueError("outputs has no loss_dict; expected STAPaperFaithfulModel outputs")

        total = loss_dict.get("loss_total", None)
        if total is None:
            total = sum(v for k, v in loss_dict.items() if k.startswith("loss_"))

        logs = {
            "loss_rpn_objectness": float(loss_dict.get("loss_rpn_objectness", total.new_zeros(())).detach()),
            "loss_rpn_box_reg": float(loss_dict.get("loss_rpn_box_reg", total.new_zeros(())).detach()),
            "loss_noun_classifier": float(loss_dict.get("loss_noun_classifier", total.new_zeros(())).detach()),
            "loss_box_reg": float(loss_dict.get("loss_box_reg", total.new_zeros(())).detach()),
            "loss_verb": float(loss_dict.get("loss_verb", total.new_zeros(())).detach()),
            "loss_ttc": float(loss_dict.get("loss_ttc", total.new_zeros(())).detach()),
            "loss_interaction_score": float(loss_dict.get("loss_interaction_score", total.new_zeros(())).detach()),
        }
        for key, value in sorted(loss_dict.items()):
            if key.startswith("loss_head"):
                logs[key] = float(value.detach())
        return total, logs, None


def compute_proxy_hits(outputs, targets: List[Dict], topk: int = 5, ttc_tolerance: float = 0.25):
    recalls = {"box": 0.0, "noun": 0.0, "noun_verb": 0.0, "overall": 0.0}
    count = 0
    for preds_boxes, preds_noun_logits, preds_verb_logits, preds_ttc, preds_scores, tgt in zip(
        outputs.pred_boxes, outputs.noun_logits, outputs.verb_logits, outputs.ttc_pred, outputs.scores, targets
    ):
        gt_boxes = tgt["boxes"].to(preds_boxes.device if preds_boxes.numel() > 0 else tgt["boxes"].device)
        if gt_boxes.numel() == 0:
            continue
        count += 1
        if preds_scores.numel() == 0:
            continue
        order = torch.topk(preds_scores, k=min(topk, preds_scores.numel()), largest=True).indices
        pred_boxes = preds_boxes[order]
        pred_nouns = preds_noun_logits[order].argmax(dim=-1)
        pred_verbs = preds_verb_logits[order].argmax(dim=-1)
        pred_ttc_b = preds_ttc[order]
        gt_nouns = tgt["noun_labels"].to(pred_boxes.device)
        gt_verbs = tgt["verb_labels"].to(pred_boxes.device)
        gt_ttc = tgt["ttc"].to(pred_boxes.device)
        hit_box = hit_noun = hit_nv = hit_all = False
        iou = box_iou(pred_boxes, gt_boxes)
        for g in range(gt_boxes.shape[0]):
            valid = iou[:, g] > 0.5
            if valid.any():
                hit_box = True
                noun_valid = valid & (pred_nouns == gt_nouns[g])
                if noun_valid.any():
                    hit_noun = True
                    nv_valid = noun_valid & (pred_verbs == gt_verbs[g])
                    if nv_valid.any():
                        hit_nv = True
                        all_valid = nv_valid & (torch.abs(pred_ttc_b - gt_ttc[g]) <= ttc_tolerance)
                        if all_valid.any():
                            hit_all = True
        recalls["box"] += float(hit_box)
        recalls["noun"] += float(hit_noun)
        recalls["noun_verb"] += float(hit_nv)
        recalls["overall"] += float(hit_all)
    if count == 0:
        return {k: 0.0 for k in recalls}
    return {k: 100.0 * v / count for k, v in recalls.items()}


def build_results_json(outputs, targets: List[Dict], idx_to_noun_id: Dict[int, int], idx_to_verb_id: Dict[int, int], topk: int = 100) -> Dict:
    results = {}
    for preds_boxes, preds_noun_logits, preds_verb_logits, preds_ttc, preds_scores, tgt in zip(
        outputs.pred_boxes, outputs.noun_logits, outputs.verb_logits, outputs.ttc_pred, outputs.scores, targets
    ):
        uid = str(tgt["uid"])
        if preds_scores.numel() == 0:
            results[uid] = []
            continue
        order = torch.topk(preds_scores, k=min(topk, preds_scores.numel()), largest=True).indices
        resized_h, resized_w = tgt["size"].tolist()
        orig_h, orig_w = tgt["orig_size"].tolist()
        scale_x = float(orig_w) / float(max(1, resized_w))
        scale_y = float(orig_h) / float(max(1, resized_h))
        boxes = preds_boxes[order].detach().float().cpu().clone()
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(orig_w - 1))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(orig_h - 1))
        noun_indices = preds_noun_logits[order].argmax(dim=-1).detach().cpu().tolist()
        verb_indices = preds_verb_logits[order].argmax(dim=-1).detach().cpu().tolist()
        ttcs = preds_ttc[order].detach().float().cpu().tolist()
        scores = preds_scores[order].detach().float().cpu().tolist()
        boxes_list = boxes.tolist()
        sample_results = []
        for box, noun_idx, verb_idx, ttc, score in zip(boxes_list, noun_indices, verb_indices, ttcs, scores):
            sample_results.append(
                {
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "noun_category_id": int(idx_to_noun_id[noun_idx]),
                    "verb_category_id": int(idx_to_verb_id[verb_idx]),
                    "time_to_contact": float(ttc),
                    "score": float(score),
                }
            )
        results[uid] = sample_results
    return {"version": "1.0", "challenge": "ego4d_short_term_object_interaction_anticipation", "results": results}
