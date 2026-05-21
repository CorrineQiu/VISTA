#!/usr/bin/env python3
"""Ensemble multiple STA prediction JSON files from different heads.

The input/output schema matches Ego4D STA submission JSON:
{
  "version": "1.0",
  "challenge": "...",
  "results": {
    "<sample_uid>": [
      {"box": [x1, y1, x2, y2], "noun_category_id": int,
       "verb_category_id": int, "time_to_contact": float, "score": float}
    ]
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _group_key(pred: dict[str, Any], class_aware: str) -> tuple[Any, ...]:
    if class_aware == "noun_verb":
        return (pred.get("noun_category_id"), pred.get("verb_category_id"))
    if class_aware == "noun":
        return (pred.get("noun_category_id"),)
    if class_aware == "none":
        return ("all",)
    raise ValueError(f"Unsupported class_aware={class_aware}")


def _nms_group(preds: list[dict[str, Any]], thresh: float) -> list[dict[str, Any]]:
    preds = sorted(preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for pred in preds:
        box = pred.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        if all(_iou_xyxy(box, kept_pred["box"]) <= thresh for kept_pred in kept):
            kept.append(pred)
    return kept


def _ensemble_sample(
    per_head_preds: list[list[dict[str, Any]]],
    head_weights: list[float],
    nms_thresh: float,
    topk: int,
    class_aware: str,
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for head_idx, preds in enumerate(per_head_preds):
        weight = head_weights[head_idx]
        for pred in preds:
            item = dict(pred)
            item["score"] = float(item.get("score", 0.0)) * weight
            pooled.append(item)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for pred in pooled:
        groups.setdefault(_group_key(pred, class_aware), []).append(pred)

    kept: list[dict[str, Any]] = []
    for group_preds in groups.values():
        kept.extend(_nms_group(group_preds, nms_thresh))
    kept.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
    return kept[:topk]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Head merged JSON files.")
    parser.add_argument("--output", required=True, help="Output ensembled JSON path.")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--nms-thresh", type=float, default=0.55)
    parser.add_argument(
        "--class-aware",
        choices=("noun_verb", "noun", "none"),
        default="noun_verb",
        help="Apply NMS independently within this class grouping.",
    )
    parser.add_argument(
        "--head-weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-head score weights. Defaults to equal weights.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    payloads = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payloads.append(json.load(f))

    if args.head_weights is None:
        head_weights = [1.0] * len(payloads)
    else:
        head_weights = list(args.head_weights)
        if len(head_weights) != len(payloads):
            raise ValueError("--head-weights must match number of --inputs")

    all_keys = set()
    for payload in payloads:
        all_keys.update(payload.get("results", {}).keys())

    out = {
        "version": payloads[0].get("version", "1.0"),
        "challenge": payloads[0].get(
            "challenge", "ego4d_short_term_object_interaction_anticipation"
        ),
        "results": {},
    }

    for idx, key in enumerate(sorted(all_keys), start=1):
        per_head_preds = [payload.get("results", {}).get(key, []) for payload in payloads]
        out["results"][key] = _ensemble_sample(
            per_head_preds,
            head_weights=head_weights,
            nms_thresh=float(args.nms_thresh),
            topk=int(args.topk),
            class_aware=str(args.class_aware),
        )
        if idx % 1000 == 0:
            print(f"[ensemble] processed {idx}/{len(all_keys)} samples", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[ensemble] wrote {output} samples={len(out['results'])}")


if __name__ == "__main__":
    main()
