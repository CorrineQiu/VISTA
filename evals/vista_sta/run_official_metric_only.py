from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

K = 5
IOU_THRESH = 0.5
TTC_THRESH = 0.25
CHALLENGE = "ego4d_short_term_object_interaction_anticipation"
VERSION = "1.0"


@dataclass
class Pred:
    example_id: str
    box: Tuple[float, float, float, float]
    noun: int
    verb: int
    ttc: float
    score: float


@dataclass
class GT:
    example_id: str
    box: Tuple[float, float, float, float]
    noun: int
    verb: int
    ttc: float


class EvaluationError(Exception):
    pass


class MissingUidError(EvaluationError):
    def __init__(self, uids: Sequence[str]) -> None:
        self.uids = list(map(str, uids))

    def __str__(self) -> str:
        preview = ", ".join(self.uids[:10])
        if len(self.uids) > 10:
            preview += f", ... (+{len(self.uids) - 10} more)"
        return f"Prediction json 缺少 {len(self.uids)} 个 GT uid: {preview}"



def normalize_box(x: Any) -> Optional[Tuple[float, float, float, float]]:
    if x is None:
        return None
    if isinstance(x, list) and len(x) == 4:
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])
    if isinstance(x, tuple) and len(x) == 4:
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])
    if isinstance(x, dict):
        if all(k in x for k in ["x1", "y1", "x2", "y2"]):
            return float(x["x1"]), float(x["y1"]), float(x["x2"]), float(x["y2"])
        if all(k in x for k in ["xmin", "ymin", "xmax", "ymax"]):
            return float(x["xmin"]), float(x["ymin"]), float(x["xmax"]), float(x["ymax"])
        if all(k in x for k in ["left", "top", "right", "bottom"]):
            return float(x["left"]), float(x["top"]), float(x["right"]), float(x["bottom"])
    return None



def compute_iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    preds = np.expand_dims(preds, 1)
    gts = np.expand_dims(gts, 0)

    def area(boxes: np.ndarray) -> np.ndarray:
        width = boxes[..., 2] - boxes[..., 0] + 1.0
        height = boxes[..., 3] - boxes[..., 1] + 1.0
        width[width < 0] = 0.0
        height[height < 0] = 0.0
        return width * height

    ixmin = np.maximum(gts[..., 0], preds[..., 0])
    iymin = np.maximum(gts[..., 1], preds[..., 1])
    ixmax = np.minimum(gts[..., 2], preds[..., 2])
    iymax = np.minimum(gts[..., 3], preds[..., 3])

    area_preds = area(preds)
    area_gts = area(gts)
    area_intersections = area(np.stack([ixmin, iymin, ixmax, iymax], axis=-1))
    return area_intersections / (area_preds + area_gts - area_intersections + 1e-11)



def ap_from_pr(prec: np.ndarray, rec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = np.max((mpre[i], mpre[i + 1]))
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))



def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    zero = b == 0
    b[zero] = 1.0
    a[zero] = 0.0
    return a / b



def validate_results_json(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> None:
    if "results" not in pred_json or not isinstance(pred_json["results"], dict):
        raise EvaluationError("Prediction json 顶层必须包含 results dict")
    if pred_json.get("challenge") != CHALLENGE:
        raise EvaluationError(f"challenge 必须是 {CHALLENGE!r}，当前是 {pred_json.get('challenge')!r}")
    if pred_json.get("version") != VERSION:
        raise EvaluationError(f"version 必须是 {VERSION!r}，当前是 {pred_json.get('version')!r}")
    if "annotations" not in gt_json or not isinstance(gt_json["annotations"], list):
        raise EvaluationError("GT json 顶层必须包含 annotations list")

    gt_uids = [str(x["uid"]) for x in gt_json["annotations"] if isinstance(x, dict) and "uid" in x]
    res_uids = set(map(str, pred_json["results"].keys()))
    missing = [uid for uid in gt_uids if uid not in res_uids]
    if missing:
        raise MissingUidError(missing)

    for uid, pred_list in pred_json["results"].items():
        if not isinstance(pred_list, list):
            raise EvaluationError(f"results[{uid!r}] 必须是 list")
        for i, p in enumerate(pred_list):
            if not isinstance(p, dict):
                raise EvaluationError(f"results[{uid!r}][{i}] 必须是 dict")
            box = p.get("box")
            if not isinstance(box, list) or len(box) != 4:
                raise EvaluationError(f"results[{uid!r}][{i}]['box'] 必须是长度为 4 的 list")
            for key in ["noun_category_id", "verb_category_id"]:
                if key not in p or not isinstance(p[key], int):
                    raise EvaluationError(f"results[{uid!r}][{i}]['{key}'] 必须是 int")
            for key in ["time_to_contact", "score"]:
                if key not in p or not isinstance(p[key], (int, float)):
                    raise EvaluationError(f"results[{uid!r}][{i}]['{key}'] 必须是数值")



def parse_predictions(pred_json: Dict[str, Any]) -> List[Pred]:
    out: List[Pred] = []
    for example_id, pred_list in pred_json["results"].items():
        for p in pred_list:
            box = normalize_box(p.get("box"))
            if box is None:
                continue
            out.append(
                Pred(
                    example_id=str(example_id),
                    box=box,
                    noun=int(p["noun_category_id"]),
                    verb=int(p["verb_category_id"]),
                    ttc=float(p["time_to_contact"]),
                    score=float(p["score"]),
                )
            )
    return out



def parse_gts(gt_json: Dict[str, Any]) -> List[GT]:
    out: List[GT] = []
    for ann in gt_json["annotations"]:
        uid = str(ann["uid"])
        for obj in ann.get("objects", []):
            box = normalize_box(obj.get("box"))
            if box is None:
                continue
            out.append(
                GT(
                    example_id=uid,
                    box=box,
                    noun=int(obj["noun_category_id"]),
                    verb=int(obj["verb_category_id"]),
                    ttc=float(obj["time_to_contact"]),
                )
            )
    return out



def ann_to_gt_arrays(ann: Dict[str, Any]) -> Dict[str, np.ndarray]:
    objs = ann.get("objects", [])
    if len(objs) == 0:
        return {}
    return {
        "boxes": np.vstack([np.asarray(obj["box"], dtype=np.float64) for obj in objs]),
        "nouns": np.asarray([int(obj["noun_category_id"]) for obj in objs], dtype=np.int64),
        "verbs": np.asarray([int(obj["verb_category_id"]) for obj in objs], dtype=np.int64),
        "ttcs": np.asarray([float(obj["time_to_contact"]) for obj in objs], dtype=np.float64),
    }



def preds_to_arrays(pred_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if len(pred_list) == 0:
        return {}
    return {
        "boxes": np.vstack([np.asarray(p["box"], dtype=np.float64) for p in pred_list]),
        "nouns": np.asarray([int(p["noun_category_id"]) for p in pred_list], dtype=np.int64),
        "verbs": np.asarray([int(p["verb_category_id"]) for p in pred_list], dtype=np.int64),
        "ttcs": np.asarray([float(p["time_to_contact"]) for p in pred_list], dtype=np.float64),
        "scores": np.asarray([float(p["score"]) for p in pred_list], dtype=np.float64),
    }


class STAMeanAveragePrecisionLite:
    def __init__(self, iou_threshold: float = IOU_THRESH, ttc_threshold: float = TTC_THRESH, top_k: int = K):
        self.iou_threshold = float(iou_threshold)
        self.ttc_threshold = float(ttc_threshold)
        self.top_k = int(top_k)
        self.true_positives: List[np.ndarray] = []
        self.confidence_scores: List[np.ndarray] = []
        self.predicted_classes: List[np.ndarray] = []
        self.gt_classes: List[np.ndarray] = []

    @staticmethod
    def metric_names() -> List[str]:
        return [
            "noun_top5_map",
            "noun_verb_top5_map",
            "noun_ttc_top5_map",
            "overall_top5_map",
        ]

    def _map_classes(self, arrs: Dict[str, np.ndarray]) -> np.ndarray:
        nouns = arrs["nouns"]
        return np.vstack([nouns, nouns, nouns, nouns]).T

    def _match_matrix(self, pred: Dict[str, Any], gts: Dict[str, np.ndarray], overlaps: np.ndarray) -> np.ndarray:
        nouns = pred["nouns"] == gts["nouns"]
        verbs = pred["verbs"] == gts["verbs"]
        ttcs = np.abs(pred["ttcs"] - gts["ttcs"]) <= self.ttc_threshold
        boxes = overlaps.ravel() > self.iou_threshold
        tp_noun = boxes & nouns
        tp_noun_verb = boxes & nouns & verbs
        tp_noun_ttc = boxes & nouns & ttcs
        tp_overall = boxes & nouns & verbs & ttcs
        return np.vstack([tp_noun, tp_noun_verb, tp_noun_ttc, tp_overall]).T

    def add(self, preds: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]) -> None:
        if len(labels) == 0:
            return
        self.gt_classes.append(self._map_classes(labels))
        if len(preds) == 0:
            return

        predicted_boxes = preds["boxes"]
        predicted_scores = preds["scores"]
        predicted_classes = self._map_classes(preds)
        true_positives = np.zeros((len(predicted_boxes), 4), dtype=np.float64)

        gt_boxes = labels["boxes"]
        ious = compute_iou_matrix(predicted_boxes, gt_boxes)
        gt_matched = np.zeros((len(gt_boxes), 4), dtype=np.float64)

        for i in predicted_scores.argsort()[::-1]:
            overlaps = ious[i].reshape(-1, 1)
            matchings = self._match_matrix(
                {
                    "nouns": preds["nouns"][i],
                    "verbs": preds["verbs"][i],
                    "ttcs": preds["ttcs"][i],
                },
                labels,
                overlaps,
            )
            overlaps = np.tile(overlaps, (1, matchings.shape[1]))
            matchings[gt_matched == 1] = 0
            overlaps[matchings == 0] = -1.0
            jj = overlaps.argmax(0)
            i_matchings = matchings[jj, np.arange(len(jj))].astype(bool)
            true_positives[i, i_matchings] = 1.0
            gt_matched[jj, np.arange(len(jj))] += i_matchings.astype(np.float64)

        if self.top_k > 1:
            fp_budget = (self.top_k - 1) * len(labels["boxes"])
            order = predicted_scores.argsort()[::-1]
            sorted_tp = true_positives[order, :].astype(np.float64)
            sorted_fp = 1.0 - sorted_tp
            sorted_tp[(sorted_fp.cumsum(0) <= fp_budget) & (sorted_fp == 1)] = np.nan
            true_positives = sorted_tp
            predicted_scores = predicted_scores[order]
            predicted_classes = predicted_classes[order]

        self.true_positives.append(true_positives)
        self.confidence_scores.append(predicted_scores)
        self.predicted_classes.append(predicted_classes)

    def evaluate(self) -> Dict[str, float]:
        names = self.metric_names()
        if not self.gt_classes:
            return {name: 0.0 for name in names}

        gt_classes = np.concatenate(self.gt_classes, axis=0)
        if self.predicted_classes:
            predicted_classes = np.concatenate(self.predicted_classes, axis=0)
            true_positives = np.concatenate(self.true_positives, axis=0)
            confidence_scores = np.concatenate(self.confidence_scores, axis=0)
        else:
            predicted_classes = np.zeros((0, 4), dtype=np.int64)
            true_positives = np.zeros((0, 4), dtype=np.float64)
            confidence_scores = np.zeros((0,), dtype=np.float64)

        metrics: Dict[str, float] = {}
        for metric_idx, name in enumerate(names):
            _gt_classes = gt_classes[:, metric_idx]
            _predicted_classes = predicted_classes[:, metric_idx] if predicted_classes.size else np.zeros((0,), dtype=np.int64)
            _true_positives = true_positives[:, metric_idx] if true_positives.size else np.zeros((0,), dtype=np.float64)
            _confidence_scores = confidence_scores

            classes = np.unique(_gt_classes)
            ap_values: List[float] = []
            for c in classes:
                tp = _true_positives[_predicted_classes == c]
                cs = _confidence_scores[_predicted_classes == c]
                ngt = int(np.sum(_gt_classes == c))

                if len(tp) > 0:
                    valid = ~np.isnan(tp)
                    tp = tp[valid]
                    cs = cs[valid]
                    if len(tp) > 0 and ngt > 0:
                        order = cs.argsort()[::-1]
                        tps = tp[order]
                        tp_cum = tps.cumsum()
                        fp_cum = (1.0 - tps).cumsum()
                        prec = safe_div(tp_cum, tp_cum + fp_cum)
                        rec = safe_div(tp_cum, np.full_like(tp_cum, fill_value=float(ngt)))
                        ap_values.append(100.0 * ap_from_pr(prec, rec))
                    elif not (len(tp) == 0 and ngt == 0):
                        ap_values.append(0.0)
                elif ngt > 0:
                    ap_values.append(0.0)

            metrics[name] = float(np.mean(ap_values)) if ap_values else 0.0
        return metrics



def summarize_boxes(preds: List[Pred], gts: List[GT]) -> Dict[str, Any]:
    def count_bad_xyxy(items: Sequence[Any]) -> int:
        bad = 0
        for x in items:
            x1, y1, x2, y2 = x.box
            if x2 < x1 or y2 < y1:
                bad += 1
        return bad

    gt_examples: Dict[str, int] = {}
    for g in gts:
        gt_examples[g.example_id] = gt_examples.get(g.example_id, 0) + 1
    pred_examples: Dict[str, int] = {}
    for p in preds:
        pred_examples[p.example_id] = pred_examples.get(p.example_id, 0) + 1

    return {
        "num_pred_examples": len(pred_examples),
        "num_gt_examples": len(gt_examples),
        "avg_preds_per_example": (sum(pred_examples.values()) / max(1, len(pred_examples))) if pred_examples else 0.0,
        "avg_gts_per_example": (sum(gt_examples.values()) / max(1, len(gt_examples))) if gt_examples else 0.0,
        "bad_pred_xyxy": count_bad_xyxy(preds),
        "bad_gt_xyxy": count_bad_xyxy(gts),
    }



def compute_metrics_from_json(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> Dict[str, float]:
    metric = STAMeanAveragePrecisionLite(top_k=K)
    for ann in gt_json["annotations"]:
        uid = str(ann["uid"])
        gt = ann_to_gt_arrays(ann)
        pred = preds_to_arrays(pred_json["results"].get(uid, []))
        metric.add(pred, gt)
    metrics = metric.evaluate()
    metrics["primary_metric"] = metrics["overall_top5_map"]
    return metrics



def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful Ego4D STA Top-5 mAP evaluator")
    parser.add_argument("pred_json", type=str)
    parser.add_argument("gt_json", type=str)
    args = parser.parse_args()

    pred_path = Path(args.pred_json)
    gt_path = Path(args.gt_json)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_json = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    validate_results_json(pred_json, gt_json)

    preds = parse_predictions(pred_json)
    gts = parse_gts(gt_json)
    summary = summarize_boxes(preds, gts)

    print(f"[info] pred_json={pred_path}")
    print(f"[info] gt_json={gt_path}")
    print(f"[info] num_preds={len(preds)}")
    print(f"[info] num_gts={len(gts)}")
    print(f"[info] num_pred_examples={summary['num_pred_examples']}")
    print(f"[info] num_gt_examples={summary['num_gt_examples']}")
    print(f"[info] avg_preds_per_example={summary['avg_preds_per_example']:.4f}")
    print(f"[info] avg_gts_per_example={summary['avg_gts_per_example']:.4f}")
    print(f"[info] bad_pred_xyxy={summary['bad_pred_xyxy']}")
    print(f"[info] bad_gt_xyxy={summary['bad_gt_xyxy']}")
    if preds:
        print(f"[debug] first_pred={preds[0]}")
    if gts:
        print(f"[debug] first_gt={gts[0]}")

    metrics = compute_metrics_from_json(pred_json, gt_json)
    print(f"noun_top5_map: {metrics['noun_top5_map']:.6f}")
    print(f"noun_verb_top5_map: {metrics['noun_verb_top5_map']:.6f}")
    print(f"noun_ttc_top5_map: {metrics['noun_ttc_top5_map']:.6f}")
    print(f"*overall_top5_map: {metrics['overall_top5_map']:.6f}")
    print(f"primary_metric: {metrics['primary_metric']:.6f}")


if __name__ == "__main__":
    main()
