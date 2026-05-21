from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from evals.vista_sta.ego4d_sta import (
    Ego4DSTASampleDirDataset,
    STASampleDirItem,
    _normalize_clip,
    _read_rgb_image_square,
    _sample_history_frame_paths,
    build_label_maps_from_roots,
    jepa_global_feature_path,
)
from evals.vista_sta.models import FrozenVJEPA2_1DualEncoder


logger = logging.getLogger("extract_jepa_global_features")


def _expand_config_strings(obj):
    if isinstance(obj, dict):
        return {k: _expand_config_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_config_strings(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def _load_clip(sample: STASampleDirItem, frames_per_clip: int, video_short_side: int) -> torch.Tensor:
    frame_paths = _sample_history_frame_paths(sample.history_frame_paths, frames_per_clip)
    clip_np = np.stack(
        [_read_rgb_image_square(p, video_short_side) for p in frame_paths],
        axis=0,
    )
    return _normalize_clip(torch.from_numpy(clip_np))


class JepaFeatureExtractionDataset(Dataset):
    def __init__(
        self,
        samples: List[STASampleDirItem],
        output_root: str,
        frames_per_clip: int,
        video_short_side: int,
    ) -> None:
        self.samples = samples
        self.output_root = output_root
        self.frames_per_clip = int(frames_per_clip)
        self.video_short_side = int(video_short_side)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        clip = _load_clip(sample, self.frames_per_clip, self.video_short_side)
        out_path = str(jepa_global_feature_path(self.output_root, sample))
        return {"clip": clip, "out_path": out_path}


def _collate_feature_batch(batch):
    return {
        "clip": torch.stack([x["clip"] for x in batch], dim=0),
        "out_path": [x["out_path"] for x in batch],
    }


def _build_samples(cfg: dict, split_names: List[str]) -> List[STASampleDirItem]:
    data_cfg = cfg["experiment"]["data"]
    roots = [data_cfg["train_split_root"], data_cfg["val_split_root"]]
    noun_id_to_idx, verb_id_to_idx = build_label_maps_from_roots(roots)
    samples: List[STASampleDirItem] = []
    for split in split_names:
        root = data_cfg[f"{split}_split_root"]
        require_labels = split.lower() in {"train", "val"}
        ds = Ego4DSTASampleDirDataset(
            split_root=root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=require_labels,
            train=False,
            video_short_side=int(data_cfg.get("video_resolution", 384)),
            frames_per_clip=int(data_cfg.get("frames_per_clip", cfg["model_kwargs"].get("frames_per_clip", 8))),
        )
        samples.extend(ds.samples)
    return samples


def _build_samples_from_manifest_cache(
    cfg: dict,
    split_names: List[str],
    sample_cache_dir: str,
) -> List[STASampleDirItem]:
    data_cfg = cfg["experiment"]["data"]
    wanted_roots = {
        str(Path(data_cfg[f"{split}_split_root"]).resolve())
        for split in split_names
    }
    samples: List[STASampleDirItem] = []
    cache_dir = Path(sample_cache_dir)
    for path in sorted(cache_dir.glob("samples_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        split_root = str(Path(payload.get("split_root", "")).resolve())
        if split_root not in wanted_roots:
            continue
        loaded = [STASampleDirItem(**x) for x in payload.get("samples", [])]
        samples.extend(loaded)
        logger.info(
            "loaded manifest cache %s root=%s samples=%d",
            path,
            split_root,
            len(loaded),
        )
    if not samples:
        raise FileNotFoundError(
            f"No matching samples_*.json found in {sample_cache_dir} for roots={sorted(wanted_roots)}"
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sample-cache-dir", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = _expand_config_strings(yaml.safe_load(f))

    data_cfg = cfg["experiment"]["data"]
    model_cfg = cfg["model_kwargs"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if args.sample_cache_dir:
        samples = _build_samples_from_manifest_cache(cfg, splits, args.sample_cache_dir)
    else:
        samples = _build_samples(cfg, splits)
    if args.reverse_order:
        samples = list(reversed(samples))
    samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_id]
    total_before_filter = len(samples)
    skipped_existing = 0
    missing_samples = []
    for sample in samples:
        out_path = jepa_global_feature_path(str(output_root), sample)
        if out_path.exists():
            skipped_existing += 1
        else:
            missing_samples.append(sample)
    samples = missing_samples
    if args.limit > 0:
        samples = samples[: args.limit]

    logger.info(
        "start extracting split=%s shard=%d/%d samples=%d output=%s device=%s batch=%d",
        ",".join(splits),
        args.shard_id,
        args.num_shards,
        len(samples),
        output_root,
        device,
        args.batch_size,
    )
    logger.info(
        "cache status total_before_filter=%d skipped_existing=%d missing_after_limit=%d",
        total_before_filter,
        skipped_existing,
        len(samples),
    )

    encoder = FrozenVJEPA2_1DualEncoder(
        checkpoint=str(model_cfg["checkpoint"]),
        model_kwargs=model_cfg.get("backbone_kwargs", {}),
        frames_per_clip=int(model_cfg.get("frames_per_clip", data_cfg.get("frames_per_clip", 8))),
        video_resolution=int(model_cfg.get("video_short_side", data_cfg.get("video_resolution", 384))),
    ).to(device)
    encoder.eval()

    frames_per_clip = int(data_cfg.get("frames_per_clip", model_cfg.get("frames_per_clip", 8)))
    video_short_side = int(data_cfg.get("video_resolution", model_cfg.get("video_short_side", 384)))
    batch_size = max(1, int(args.batch_size))

    done = 0
    t0 = time.time()

    loader = DataLoader(
        JepaFeatureExtractionDataset(
            samples=samples,
            output_root=str(output_root),
            frames_per_clip=frames_per_clip,
            video_short_side=video_short_side,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
        collate_fn=_collate_feature_batch,
    )

    for step, batch in enumerate(loader, 1):
        video = batch["clip"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = encoder.encode_video(video)
            if tokens.ndim == 3:
                pooled = tokens.float().mean(dim=1)
            elif tokens.ndim == 2:
                pooled = tokens.float()
            else:
                pooled = tokens.float().flatten(1).mean(dim=1, keepdim=True)
        for out_path_str, feat in zip(batch["out_path"], pooled):
            out_path = Path(out_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                continue
            tmp_path = out_path.with_name(
                f"{out_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
            )
            torch.save(feat.detach().to(torch.float16).cpu(), tmp_path)
            try:
                os.link(tmp_path, out_path)
                done += 1
            except FileExistsError:
                pass
            finally:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
        processed = min(step * batch_size, len(samples))
        if processed % max(1, args.log_interval) < batch_size or processed == len(samples):
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-6)
            remain = (len(samples) - processed) / max(rate, 1e-6)
            logger.info(
                "progress shard=%d/%d processed_missing=%d/%d wrote=%d skipped_existing=%d elapsed=%.1fs eta=%.1fs",
                args.shard_id,
                args.num_shards,
                processed,
                len(samples),
                done,
                skipped_existing,
                elapsed,
                remain,
            )
    logger.info(
        "done shard=%d/%d wrote=%d skipped_existing=%d elapsed=%.1fs",
        args.shard_id,
        args.num_shards,
        done,
        skipped_existing,
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
