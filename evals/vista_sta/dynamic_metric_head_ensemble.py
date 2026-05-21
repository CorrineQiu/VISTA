#!/usr/bin/env python3
"""Metric-aware dynamic ensemble for Ego4D STA head predictions.

This script fuses multiple head-level STA submission JSONs.  The scoring and
clustering are intentionally aligned with the primary Ego4D STA metric:
noun+verb must match, IoU must exceed 0.5, and TTC must be within 0.25s.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHALLENGE = "ego4d_short_term_object_interaction_anticipation"


def iou_xyxy(a: list[float], b: list[float]) -> float:
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


def ttc_affinity(a: float, b: float, tolerance: float) -> float:
    diff = abs(float(a) - float(b))
    if diff > tolerance:
        return 0.0
    return 1.0 - diff / max(tolerance, 1e-6)


def same_primary_match(
    a: dict[str, Any],
    b: dict[str, Any],
    iou_thresh: float,
    ttc_tolerance: float,
) -> bool:
    if int(a["noun_category_id"]) != int(b["noun_category_id"]):
        return False
    if int(a["verb_category_id"]) != int(b["verb_category_id"]):
        return False
    if abs(float(a["time_to_contact"]) - float(b["time_to_contact"])) > ttc_tolerance:
        return False
    return iou_xyxy(a["box"], b["box"]) >= iou_thresh


def metric_affinity(
    a: dict[str, Any],
    b: dict[str, Any],
    iou_thresh: float,
    ttc_tolerance: float,
) -> float:
    if int(a["noun_category_id"]) != int(b["noun_category_id"]):
        return 0.0
    if int(a["verb_category_id"]) != int(b["verb_category_id"]):
        return 0.0
    box_iou = iou_xyxy(a["box"], b["box"])
    if box_iou < iou_thresh:
        return 0.0
    ttc_sim = ttc_affinity(float(a["time_to_contact"]), float(b["time_to_contact"]), ttc_tolerance)
    if ttc_sim <= 0.0:
        return 0.0
    return box_iou * (0.75 + 0.25 * ttc_sim)


def top_scores(preds: list[dict[str, Any]], k: int) -> list[float]:
    return [float(p.get("score", 0.0)) for p in preds[:k]]


def dynamic_head_weights(
    per_head_preds: list[list[dict[str, Any]]],
    agreement_topk: int,
    iou_thresh: float,
    ttc_tolerance: float,
) -> list[float]:
    """Estimate per-sample head reliability from confidence and cross-head consensus."""

    num_heads = len(per_head_preds)
    if num_heads == 0:
        return []

    conf_quality: list[float] = []
    for preds in per_head_preds:
        scores5 = top_scores(preds, 5)
        scores20 = top_scores(preds, 20)
        if not scores5:
            conf_quality.append(0.0)
            continue
        top1 = scores5[0]
        mean5 = sum(scores5) / len(scores5)
        mass20 = sum(scores20)
        peakiness = top1 / max(mass20, top1, 1e-12)
        conf_quality.append(math.sqrt(max(top1, 0.0) * max(mean5, 0.0)) * (0.75 + 0.25 * peakiness))

    agreement_quality: list[float] = []
    for head_idx, preds in enumerate(per_head_preds):
        candidates = preds[:agreement_topk]
        own_mass = sum(float(p.get("score", 0.0)) for p in candidates)
        if own_mass <= 0.0:
            agreement_quality.append(0.0)
            continue
        matched_mass = 0.0
        for pred in candidates:
            pred_score = float(pred.get("score", 0.0))
            if pred_score <= 0.0:
                continue
            other_support = 0.0
            for other_idx, other_preds in enumerate(per_head_preds):
                if other_idx == head_idx:
                    continue
                best = 0.0
                for other in other_preds[:agreement_topk]:
                    aff = metric_affinity(pred, other, iou_thresh=iou_thresh, ttc_tolerance=ttc_tolerance)
                    if aff <= 0.0:
                        continue
                    best = max(best, aff * float(other.get("score", 0.0)))
                other_support += best
            matched_mass += pred_score * other_support
        denom = own_mass * max(1, num_heads - 1)
        agreement_quality.append(matched_mass / max(denom, 1e-12))

    raw: list[float] = []
    for conf, agreement in zip(conf_quality, agreement_quality):
        if conf <= 0.0:
            raw.append(0.0)
            continue
        raw.append((conf + 1e-12) ** 0.60 * (0.08 + agreement) ** 0.40)

    total = sum(raw)
    if total <= 0.0:
        active = [1.0 if preds else 0.0 for preds in per_head_preds]
        total_active = sum(active)
        return [v / total_active if total_active > 0.0 else 1.0 / num_heads for v in active]
    return [v / total for v in raw]


def weighted_average(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return sum(values) / max(1, len(values))
    return sum(v * w for v, w in zip(values, weights)) / total


def fuse_cluster(
    cluster: list[dict[str, Any]],
    head_weights: list[float],
    num_heads: int,
) -> dict[str, Any]:
    best_by_head: dict[int, dict[str, Any]] = {}
    for pred in cluster:
        head_idx = int(pred["_head_idx"])
        prev = best_by_head.get(head_idx)
        if prev is None or float(pred["_adjusted_score"]) > float(prev["_adjusted_score"]):
            best_by_head[head_idx] = pred

    members = list(best_by_head.values())
    member_scores = [float(p["score"]) for p in members]
    member_weights = [
        max(head_weights[int(p["_head_idx"])], 1e-6) * max(float(p["score"]), 1e-9) ** 0.75
        for p in members
    ]

    box = [
        weighted_average([float(p["box"][coord]) for p in members], member_weights)
        for coord in range(4)
    ]
    if box[2] < box[0]:
        box[0], box[2] = box[2], box[0]
    if box[3] < box[1]:
        box[1], box[3] = box[3], box[1]

    ttc = weighted_average([float(p["time_to_contact"]) for p in members], member_weights)

    present_weight = sum(head_weights[int(p["_head_idx"])] for p in members)
    weighted_sum = sum(head_weights[int(p["_head_idx"])] * float(p["score"]) for p in members)
    avg_score = weighted_sum / max(present_weight, 1e-12)
    max_score = max(member_scores) if member_scores else 0.0
    support_count = len(best_by_head)

    support_factor = 0.72 + 0.28 * math.sqrt(max(0.0, min(1.0, present_weight)))
    if num_heads > 1:
        support_factor += 0.10 * (support_count - 1) / (num_heads - 1)
    score = (0.70 * avg_score + 0.30 * max_score) * support_factor

    anchor = max(members, key=lambda p: float(p["_adjusted_score"]))
    return {
        "box": [float(x) for x in box],
        "noun_category_id": int(anchor["noun_category_id"]),
        "verb_category_id": int(anchor["verb_category_id"]),
        "time_to_contact": float(ttc),
        "score": float(score),
        "_support_count": support_count,
        "_support_weight": float(present_weight),
    }


def ensemble_one_sample(
    per_head_preds: list[list[dict[str, Any]]],
    head_weights: list[float],
    input_topk: int,
    output_topk: int,
    cluster_iou: float,
    final_iou: float,
    ttc_tolerance: float,
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    num_heads = len(per_head_preds)
    for head_idx, preds in enumerate(per_head_preds):
        for pred in preds[:input_topk]:
            box = pred.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            score = float(pred.get("score", 0.0))
            if score <= 0.0:
                continue
            item = {
                "box": [float(x) for x in box],
                "noun_category_id": int(pred["noun_category_id"]),
                "verb_category_id": int(pred["verb_category_id"]),
                "time_to_contact": float(pred["time_to_contact"]),
                "score": score,
                "_head_idx": head_idx,
                "_adjusted_score": score * head_weights[head_idx] * num_heads,
            }
            pooled.append(item)

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for pred in pooled:
        grouped[(int(pred["noun_category_id"]), int(pred["verb_category_id"]))].append(pred)

    clusters: list[list[dict[str, Any]]] = []
    for group_preds in grouped.values():
        group_preds.sort(key=lambda p: float(p["_adjusted_score"]), reverse=True)
        group_clusters: list[list[dict[str, Any]]] = []
        anchors: list[dict[str, Any]] = []
        for pred in group_preds:
            best_idx = -1
            best_aff = 0.0
            for idx, anchor in enumerate(anchors):
                aff = metric_affinity(pred, anchor, iou_thresh=cluster_iou, ttc_tolerance=ttc_tolerance)
                if aff > best_aff:
                    best_aff = aff
                    best_idx = idx
            if best_idx >= 0:
                group_clusters[best_idx].append(pred)
            else:
                anchors.append(pred)
                group_clusters.append([pred])
        clusters.extend(group_clusters)

    fused = [fuse_cluster(cluster, head_weights, num_heads=num_heads) for cluster in clusters]
    fused.sort(key=lambda p: float(p["score"]), reverse=True)

    final: list[dict[str, Any]] = []
    for pred in fused:
        duplicate = False
        for kept in final:
            if not same_primary_match(pred, kept, iou_thresh=final_iou, ttc_tolerance=ttc_tolerance):
                continue
            if float(pred["score"]) <= float(kept["score"]):
                duplicate = True
                break
        if duplicate:
            continue
        final.append(
            {
                "box": pred["box"],
                "noun_category_id": pred["noun_category_id"],
                "verb_category_id": pred["verb_category_id"],
                "time_to_contact": pred["time_to_contact"],
                "score": pred["score"],
            }
        )
        if len(final) >= output_topk:
            break
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Merged head JSON files.")
    parser.add_argument("--output", required=True, help="Output submission JSON.")
    parser.add_argument("--zip-output", default=None, help="Optional zip archive path.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    parser.add_argument("--version", default="2.0", choices=("1.0", "2.0"))
    parser.add_argument("--input-topk", type=int, default=100)
    parser.add_argument("--output-topk", type=int, default=100)
    parser.add_argument("--agreement-topk", type=int, default=40)
    parser.add_argument("--cluster-iou", type=float, default=0.50)
    parser.add_argument("--final-iou", type=float, default=0.50)
    parser.add_argument("--ttc-tolerance", type=float, default=0.25)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    payloads = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payloads.append(json.load(f))

    all_keys = sorted(set().union(*(set(payload.get("results", {}).keys()) for payload in payloads)))
    out = {"version": args.version, "challenge": CHALLENGE, "results": {}}

    weight_sums = [0.0 for _ in payloads]
    support_hist: Counter[int] = Counter()
    pred_count_hist: Counter[int] = Counter()

    for idx, uid in enumerate(all_keys, start=1):
        per_head_preds = [
            sorted(payload.get("results", {}).get(uid, []), key=lambda p: float(p.get("score", 0.0)), reverse=True)
            for payload in payloads
        ]
        weights = dynamic_head_weights(
            per_head_preds,
            agreement_topk=int(args.agreement_topk),
            iou_thresh=float(args.cluster_iou),
            ttc_tolerance=float(args.ttc_tolerance),
        )
        for head_idx, weight in enumerate(weights):
            weight_sums[head_idx] += weight
        support_hist[sum(1 for weight in weights if weight >= 0.20)] += 1

        preds = ensemble_one_sample(
            per_head_preds,
            head_weights=weights,
            input_topk=int(args.input_topk),
            output_topk=int(args.output_topk),
            cluster_iou=float(args.cluster_iou),
            final_iou=float(args.final_iou),
            ttc_tolerance=float(args.ttc_tolerance),
        )
        pred_count_hist[len(preds)] += 1
        out["results"][uid] = preds

        if idx % 1000 == 0:
            print(f"[dynamic-ensemble] processed {idx}/{len(all_keys)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    zip_output = Path(args.zip_output) if args.zip_output else None
    if zip_output is not None:
        zip_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(output, arcname=output.name)

    summary = {
        "method": "metric_aware_dynamic_confidence_consensus_wbf",
        "inputs": [str(path) for path in paths],
        "output": str(output),
        "zip_output": str(zip_output) if zip_output is not None else None,
        "version": args.version,
        "num_uids": len(all_keys),
        "input_topk_per_head": int(args.input_topk),
        "output_topk_per_uid": int(args.output_topk),
        "cluster_iou": float(args.cluster_iou),
        "final_iou": float(args.final_iou),
        "ttc_tolerance": float(args.ttc_tolerance),
        "average_dynamic_head_weights": {
            f"head{idx}": weight_sums[idx] / max(1, len(all_keys)) for idx in range(len(weight_sums))
        },
        "active_head_count_histogram_weight_ge_0.20": dict(sorted(support_hist.items())),
        "prediction_count_histogram": dict(sorted(pred_count_hist.items())),
        "output_bytes": output.stat().st_size,
        "zip_bytes": zip_output.stat().st_size if zip_output is not None and zip_output.exists() else None,
    }
    summary_output = Path(args.summary_output) if args.summary_output else output.with_name(output.stem + "_summary.json")
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[dynamic-ensemble] wrote {output} samples={len(all_keys)} bytes={output.stat().st_size}")
    if zip_output is not None:
        print(f"[dynamic-ensemble] wrote {zip_output} bytes={zip_output.stat().st_size}")
    print(f"[dynamic-ensemble] wrote {summary_output}")


if __name__ == "__main__":
    main()
