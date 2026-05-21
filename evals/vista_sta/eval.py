from __future__ import annotations

import csv
import gc
import json
import logging
import os
import pprint
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
    MixedPrecision,
)

from .dataloader import filter_annotations, init_data
from .ego4d_sta import prebuild_sta_caches
from .losses import SetCriterion, compute_proxy_hits, build_results_json
from .models import STAPaperFaithfulModel
from .utils import (
    AverageMeter,
    barrier,
    init_opt,
    merge_result_jsons,
    reduce_dict,
    run_official_evaluator,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pp = pprint.PrettyPrinter(indent=4)
MAX_LOGGED_HEADS = 4

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed_all(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = False


def _parse_head_indices(spec, num_heads: int) -> list[int]:
    spec = str(spec or "all").strip().lower()
    if spec in {"all", "*"}:
        return list(range(num_heads))
    if spec in {"first", "head0", "0"}:
        return [0]
    if spec in {"none", "off", "false"}:
        return []
    indices = []
    for token in spec.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("head"):
            token = token.removeprefix("head")
        idx = int(token)
        if idx < 0 or idx >= num_heads:
            raise ValueError(f"head index {idx} out of range for {num_heads} prediction heads")
        if idx not in indices:
            indices.append(idx)
    return indices


def _normalize_dump_predictions(spec) -> str:
    spec = str(spec or "official_only").strip().lower()
    aliases = {
        "true": "all",
        "1": "all",
        "yes": "all",
        "false": "off",
        "0": "off",
        "no": "off",
        "none": "off",
        "official": "official_only",
    }
    return aliases.get(spec, spec)


def _configure_cpu_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    torch_threads = max(1, int(os.environ.get("TORCH_NUM_THREADS", "1")))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _setup_file_logger(log_txt: Path):
    root_logger = logging.getLogger()
    abs_path = str(log_txt.resolve())
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path
        for h in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_txt, mode="a")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _freeze_video_probe_for_cached_features(model: torch.nn.Module) -> int:
    """Freeze the temporal probe when cached JEPA global tokens bypass it."""
    probe = getattr(model, "video_probe", None)
    if probe is None:
        return 0
    frozen = 0
    for p in probe.parameters():
        if p.requires_grad:
            frozen += int(p.numel())
        p.requires_grad_(False)
    return frozen


def _append_csv_row(csv_path: Path, row: dict):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_train_loss",
        "proxy_box",
        "proxy_noun",
        "proxy_noun_verb",
        "proxy_overall",
        "official_primary_metric",
        "best_head",
        "official_best_head",
        "best_metric",
        "is_best",
    ]
    for head_idx in range(MAX_LOGGED_HEADS):
        fieldnames.extend(
            [
                f"proxy_head{head_idx}_box",
                f"proxy_head{head_idx}_noun",
                f"proxy_head{head_idx}_noun_verb",
                f"proxy_head{head_idx}_overall",
                f"official_head{head_idx}_primary_metric",
            ]
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in fieldnames})


def _metric_csv_row(epoch, train_loss, metrics, best_metric, is_best):
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_train_loss": metrics.get("val_train_loss", None),
        "proxy_box": metrics.get("proxy_box", None),
        "proxy_noun": metrics.get("proxy_noun", None),
        "proxy_noun_verb": metrics.get("proxy_noun_verb", None),
        "proxy_overall": metrics.get("proxy_overall", None),
        "official_primary_metric": metrics.get("official_primary_metric", None),
        "best_head": metrics.get("best_head", None),
        "official_best_head": metrics.get("official_best_head", None),
        "best_metric": best_metric,
        "is_best": int(is_best),
    }
    for head_idx in range(MAX_LOGGED_HEADS):
        for key in ("box", "noun", "noun_verb", "overall"):
            row[f"proxy_head{head_idx}_{key}"] = metrics.get(f"proxy_head{head_idx}_{key}", None)
        row[f"official_head{head_idx}_primary_metric"] = metrics.get(
            f"official_head{head_idx}_primary_metric", None
        )
    return row


def _to_device_targets(targets, device):
    out = []
    for tgt in targets:
        out.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tgt.items()})
    return out


def build_model(args_model, num_nouns, num_verbs):
    def _bool_arg(name: str, default: bool) -> bool:
        value = args_model.get(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    return STAPaperFaithfulModel(
        checkpoint=args_model["checkpoint"],
        backbone_kwargs=args_model["backbone_kwargs"],
        frames_per_clip=int(args_model.get("frames_per_clip", 8)),
        video_resolution=int(args_model.get("video_short_side", 384)),
        num_nouns=num_nouns,
        num_verbs=num_verbs,
        proposal_topk_train=int(args_model.get("proposal_topk_train", 2000)),
        proposal_topk_test=int(args_model.get("proposal_topk_test", 300)),
        rpn_pre_nms_topk_train=int(args_model.get("rpn_pre_nms_topk_train", 2000)),
        rpn_pre_nms_topk_test=int(args_model.get("rpn_pre_nms_topk_test", 1000)),
        rpn_nms_thresh=float(args_model.get("rpn_nms_thresh", 0.7)),
        roi_pool_size=int(args_model.get("roi_pool_size", 7)),
        fpn_dim=int(args_model.get("fpn_dim", 256)),
        probe_depth=int(args_model.get("probe_depth", 2)),
        num_heads=int(args_model.get("num_heads", 8)),
        representation_size=int(args_model.get("representation_size", 1024)),
        class_topk_per_proposal=int(args_model.get("class_topk_per_proposal", 5)),
        verb_topk_per_proposal=int(args_model.get("verb_topk_per_proposal", 3)),
        detections_per_img=int(args_model.get("detections_per_img", 100)),
        box_score_thresh=float(args_model.get("box_score_thresh", 0.001)),
        box_nms_thresh=float(args_model.get("box_nms_thresh", 0.55)),
        box_reg_weights=args_model.get("box_reg_weights", (10.0, 10.0, 5.0, 5.0)),
        inference_score=str(args_model.get("inference_score", "objectness_quality_noun_verb")),
        verb_loss_weight=float(args_model.get("verb_loss_weight", 0.1)),
        ttc_loss_weight=float(args_model.get("ttc_loss_weight", 0.5)),
        score_loss_weight=float(args_model.get("score_loss_weight", 1.0)),
        num_prediction_heads=int(args_model.get("num_prediction_heads", 4)),
        verb_background=_bool_arg("verb_background", True),
        detector_pretrained=_bool_arg("detector_pretrained", True),
        detector_weights_path=args_model.get("detector_weights_path", None),
        detector_trainable_backbone_layers=int(args_model.get("detector_trainable_backbone_layers", 3)),
        freeze_detector_backbone=_bool_arg("freeze_detector_backbone", False),
        freeze_detector_rpn=_bool_arg("freeze_detector_rpn", False),
        freeze_detector_box_head=_bool_arg("freeze_detector_box_head", False),
        fpn_context_fusion=_bool_arg("fpn_context_fusion", True),
        skip_vjepa_encoder_load=_bool_arg("skip_vjepa_encoder_load", True),
        cached_vjepa_dim=int(args_model.get("cached_vjepa_dim", 1664)),
    )


def _dist_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["device_ids"] = [torch.cuda.current_device()]
        torch.distributed.barrier(**kwargs)


def _build_model_serialized(args_model, num_nouns, num_verbs, device, rank, world_size, distributed, serialize_model_init):
    if not distributed or world_size <= 1 or not serialize_model_init:
        logger.info("rank=%d building model without serialization", rank)
        model = build_model(args_model, num_nouns=num_nouns, num_verbs=num_verbs).to(device)
        logger.info(
            "rank=%d model.to(device) done alloc=%.2fGB reserved=%.2fGB",
            rank,
            torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0,
            torch.cuda.memory_reserved(device) / 1024**3 if torch.cuda.is_available() else 0.0,
        )
        return model

    model = None
    for init_rank in range(world_size):
        if rank == init_rank:
            logger.info("rank=%d serialized model init slot %d/%d start", rank, init_rank, world_size)
            t0 = time.time()
            model = build_model(args_model, num_nouns=num_nouns, num_verbs=num_verbs).to(device)
            logger.info(
                "rank=%d serialized model init slot %d/%d done in %.1fs alloc=%.2fGB reserved=%.2fGB",
                rank,
                init_rank,
                world_size,
                time.time() - t0,
                torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0,
                torch.cuda.memory_reserved(device) / 1024**3 if torch.cuda.is_available() else 0.0,
            )
        _dist_barrier()
    if model is None:
        raise RuntimeError(f"rank={rank} failed to build model during serialized init")
    gc.collect()
    return model


def _is_fsdp_model(model) -> bool:
    return isinstance(model, FSDP)


def _is_ddp_model(model) -> bool:
    return isinstance(model, DDP)


def _unwrap_model(model):
    return model.module if _is_ddp_model(model) else model


def _broadcast_best_info(best_metric: float, is_best: bool, device: torch.device, distributed: bool, rank: int):
    if not distributed:
        return best_metric, is_best
    t = torch.zeros(2, device=device, dtype=torch.float32)
    if rank == 0:
        t[0] = float(best_metric)
        t[1] = 1.0 if is_best else 0.0
    torch.distributed.broadcast(t, src=0)
    return float(t[0].item()), bool(int(t[1].item()))


def _make_checkpoint_state(
    epoch: int,
    model,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler,
    best_metric: float,
    metrics: dict,
    args_eval: dict,
    idx_to_noun_id: dict,
    idx_to_verb_id: dict,
    fsdp_active: bool,
):
    if fsdp_active:
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            model_state = model.state_dict()
        optimizer_state = FSDP.full_optim_state_dict(model, optimizer)
    else:
        model_state = _unwrap_model(model).state_dict()
        optimizer_state = optimizer.state_dict()

    return {
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "wd_scheduler": None if wd_scheduler is None else wd_scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics,
        "args_eval": args_eval,
        "idx_to_noun_id": idx_to_noun_id,
        "idx_to_verb_id": idx_to_verb_id,
        "use_fsdp": bool(fsdp_active),
    }


def _save_checkpoint_bundle(
    folder: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler,
    best_metric: float,
    metrics: dict,
    args_eval: dict,
    idx_to_noun_id: dict,
    idx_to_verb_id: dict,
    is_best: bool,
    fsdp_active: bool,
    distributed: bool,
    rank: int,
):
    state = _make_checkpoint_state(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        scaler=scaler,
        best_metric=best_metric,
        metrics=metrics,
        args_eval=args_eval,
        idx_to_noun_id=idx_to_noun_id,
        idx_to_verb_id=idx_to_verb_id,
        fsdp_active=fsdp_active,
    )

    barrier(distributed)
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(state, folder / f"epoch-{epoch:03d}.pt")
        torch.save(state, folder / "latest.pt")
        if is_best:
            torch.save(state, folder / "best.pt")
    barrier(distributed)


def _load_checkpoint(path: Path, model, optimizer=None, scheduler=None, wd_scheduler=None, scaler=None, fsdp_active: bool = False):
    ckpt = torch.load(path, map_location="cpu")
    model_to_load = _unwrap_model(model)
    state = ckpt["model"]
    if getattr(model_to_load, "skip_vjepa_encoder_load", False):
        before = len(state)
        state = {k: v for k, v in state.items() if not str(k).startswith("video_encoder.encoder.")}
        logger.info(
            "Filtered skipped V-JEPA encoder weights from checkpoint: kept=%d dropped=%d",
            len(state),
            before - len(state),
        )

    if fsdp_active:
        full_state_cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            msg = model.load_state_dict(state, strict=False)

        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            loaded_osd = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(loaded_osd)
    else:
        msg = model_to_load.load_state_dict(state, strict=False)
        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

    logger.info("Loaded checkpoint %s with msg: %s", path, msg)

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if wd_scheduler is not None and ckpt.get("wd_scheduler") is not None:
        wd_scheduler.load_state_dict(ckpt["wd_scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def train_one_epoch(
    ipe,
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler,
    data_loader,
    use_bfloat16,
    grad_accum_steps=1,
    rank=0,
):
    model.train(True)
    _data_loader = iter(data_loader)
    meters = {"loss": AverageMeter()}

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for itr in range(ipe):
        if rank == 0 and itr == 0:
            logger.info("[train-debug] about to fetch first batch")

        try:
            batch = next(_data_loader)
        except StopIteration:
            _data_loader = iter(data_loader)
            batch = next(_data_loader)

        if rank == 0 and itr == 0:
            logger.info(
                "[train-debug] fetched first batch video_shape=%s image_shape=%s num_targets=%d",
                tuple(batch["video"].shape),
                tuple(batch["image"].shape),
                len(batch["targets"]),
            )

        video = batch["video"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        targets = _to_device_targets(batch["targets"], device)

        if rank == 0 and itr == 0:
            logger.info("[train-debug] moved first batch to device")

        amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=torch.cuda.is_available(),
        ):
            if rank == 0 and itr == 0:
                logger.info("[train-debug] about to run first forward")
            outputs = model(video, image, targets=targets)
            if rank == 0 and itr == 0:
                logger.info("[train-debug] first forward finished")

            loss, logs, _ = criterion(outputs, targets)

            # 梯度累计：把 loss 平均到每个 micro-step
            loss_to_backward = loss / grad_accum_steps

        if rank == 0 and itr == 0:
            logger.info("[train-debug] first loss computed = %.6f", float(loss.detach()))

        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        # 只在累计满 grad_accum_steps，或最后一个 iteration 时更新参数
        do_step = ((itr + 1) % grad_accum_steps == 0) or (itr == ipe - 1)

        if do_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # 只有真正 update 时才推进 scheduler / wd_scheduler
            scheduler.step()
            wd_scheduler.step()

        # meter 记录原始 loss，不记录缩放后的 loss_to_backward
        meters["loss"].update(float(loss.detach()), n=video.shape[0])
        for k, v in logs.items():
            meters.setdefault(k, AverageMeter()).update(float(v), n=video.shape[0])

        if rank == 0 and (itr % 20 == 0 or itr == ipe - 1):
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            num_boxes = sum(int(t["boxes"].shape[0]) for t in batch["targets"])

            msg = (
                f"[sta_conservative_frozen_new_fixed_0507] train itr={itr}/{ipe} "
                f"loss={meters['loss'].avg:.4f} "
                f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB "
                f"img={tuple(batch['image'].shape)} boxes={num_boxes} "
                f"accum={grad_accum_steps} step_now={int(do_step)}"
            )
            for name in sorted(meters.keys()):
                if name == "loss":
                    continue
                msg += f" {name}={meters[name].avg:.4f}"
            logger.info(msg)

    return {k: m.avg for k, m in meters.items()}


@torch.inference_mode()
def validate_and_dump(
    ipe,
    device,
    model,
    criterion,
    data_loader,
    use_bfloat16,
    idx_to_noun_id,
    idx_to_verb_id,
    results_dir: Path,
    epoch: int,
    distributed: bool,
    rank: int,
    world_size: int,
    topk_predictions: int,
    evaluator_script: str | None,
    annotations_val: str | None,
    run_official_metrics: bool = True,
    official_eval_heads: str = "proxy_best",
    compute_val_loss: bool = False,
    max_val_batches: int | None = None,
    validation_heads: str = "all",
    dump_predictions: str = "official_only",
    compute_proxy_metrics: bool = True,
    log_interval: int = 100,
):
    model.train(False)
    validation_t0 = time.time()
    base_model = _unwrap_model(model)
    num_model_heads = max(1, int(getattr(base_model, "num_prediction_heads", 1)))
    eval_head_indices = _parse_head_indices(validation_heads, num_model_heads)
    if not eval_head_indices:
        raise ValueError("validation_heads selected no heads")
    official_eval_heads_norm = str(official_eval_heads or "proxy_best").strip().lower()
    dump_mode = _normalize_dump_predictions(dump_predictions)
    if rank == 0 and num_model_heads > MAX_LOGGED_HEADS:
        logger.info("num_prediction_heads=%d; CSV records the first %d heads", num_model_heads, MAX_LOGGED_HEADS)

    def _head_suffix(head_idx: int) -> str:
        return "" if head_idx == 0 else f"_head{head_idx}"

    def _per_rank_json(head_idx: int, rank_idx: int) -> Path:
        return results_dir / f"epoch_{epoch:03d}{_head_suffix(head_idx)}_rank{rank_idx}.json"

    def _merged_json(head_idx: int) -> Path:
        return results_dir / f"epoch_{epoch:03d}{_head_suffix(head_idx)}_merged.json"

    def _empty_payload() -> dict:
        return {
            "version": "1.0",
            "challenge": "ego4d_short_term_object_interaction_anticipation",
            "results": {},
        }

    _data_loader = iter(data_loader)
    metrics = {"val_train_loss": AverageMeter()} if compute_val_loss else {}
    proxy_meters = {
        head_idx: {
            "box": AverageMeter(),
            "noun": AverageMeter(),
            "noun_verb": AverageMeter(),
            "overall": AverageMeter(),
        }
        for head_idx in eval_head_indices
    }

    official_needed = bool(run_official_metrics and annotations_val is not None and evaluator_script)
    if dump_mode in {"off", "none"}:
        dump_head_indices: list[int] = []
    elif dump_mode in {"all", "*"}:
        dump_head_indices = list(eval_head_indices)
    elif dump_mode in {"official_only", "official"}:
        if not official_needed:
            dump_head_indices = []
        elif official_eval_heads_norm in {"proxy_best", "best", "auto"}:
            dump_head_indices = list(eval_head_indices)
        elif official_eval_heads_norm in {"all", "*"}:
            dump_head_indices = list(eval_head_indices)
        elif official_eval_heads_norm in {"none", "off", "false", "0"}:
            dump_head_indices = []
        else:
            dump_head_indices = [h for h in _parse_head_indices(official_eval_heads_norm, num_model_heads) if h in eval_head_indices]
    else:
        raise ValueError(f"Unsupported dump_predictions={dump_predictions}")
    local_payloads = {head_idx: _empty_payload() for head_idx in dump_head_indices}

    eval_ipe = min(ipe, int(max_val_batches)) if max_val_batches is not None and int(max_val_batches) > 0 else ipe

    if rank == 0:
        logger.info(
            "validation epoch=%03d start val_iters=%d heads=%d compute_val_loss=%s official_eval_heads=%s",
            epoch,
            eval_ipe,
            len(eval_head_indices),
            compute_val_loss,
            official_eval_heads,
        )
        logger.info(
            "validation epoch=%03d selected_heads=%s dump_predictions=%s dump_heads=%s",
            epoch,
            eval_head_indices,
            dump_mode,
            dump_head_indices,
        )

    for itr in range(eval_ipe):
        # try:
        #     batch = next(_data_loader)
        # except Exception:
        #     _data_loader = iter(data_loader)
        #     batch = next(_data_loader)
        try:
            batch = next(_data_loader)
        except StopIteration:
            _data_loader = iter(data_loader)
            batch = next(_data_loader)
    
        video = batch["video"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        targets = _to_device_targets(batch["targets"], device)

        amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            pred_outputs = model(
                video,
                image,
                targets=targets,
                return_predictions=True,
                head_indices=eval_head_indices,
            )
            head_outputs = pred_outputs.head_outputs or [pred_outputs]
            returned_head_indices = pred_outputs.selected_head_indices or eval_head_indices
            if len(head_outputs) != len(returned_head_indices):
                raise RuntimeError(
                    f"Model returned {len(head_outputs)} head outputs for {len(returned_head_indices)} head indices"
                )

        if compute_val_loss:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                loss_outputs = model(
                    video,
                    image,
                    targets=targets,
                    compute_loss=True,
                    return_predictions=False,
                )
                loss, _, _ = criterion(loss_outputs, targets)
            metrics["val_train_loss"].update(float(loss.detach()), n=video.shape[0])

        for head_idx, head_output in zip(returned_head_indices, head_outputs):
            if compute_proxy_metrics:
                proxy = compute_proxy_hits(head_output, targets, topk=5)
                for k, v in proxy.items():
                    proxy_meters[head_idx][k].update(v, n=video.shape[0])

            if head_idx in local_payloads:
                payload = build_results_json(
                    head_output,
                    batch["targets"],
                    idx_to_noun_id,
                    idx_to_verb_id,
                    topk=topk_predictions,
                )
                local_payloads[head_idx]["results"].update(payload["results"])

        progress_interval = max(1, int(log_interval))
        if rank == 0 and ((itr + 1) % progress_interval == 0 or (itr + 1) == eval_ipe):
            elapsed = time.time() - validation_t0
            sec_per_iter = elapsed / float(itr + 1)
            eta = sec_per_iter * float(max(eval_ipe - itr - 1, 0))
            logger.info(
                "validation epoch=%03d progress=%d/%d elapsed=%.1fs sec_per_iter=%.3f eta=%.1fs",
                epoch,
                itr + 1,
                eval_ipe,
                elapsed,
                sec_per_iter,
                eta,
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        logger.info(
            "validation epoch=%03d writing per-rank json heads=%d results_dir=%s",
            epoch,
            len(local_payloads),
            results_dir,
        )
    if distributed and world_size > 1:
        for write_rank in range(world_size):
            if rank == write_rank:
                for head_idx, payload in local_payloads.items():
                    out_json = _per_rank_json(head_idx, rank)
                    logger.info(
                        "validation epoch=%03d rank=%d writing json head=%d path=%s",
                        epoch,
                        rank,
                        head_idx,
                        out_json,
                    )
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    logger.info(
                        "validation epoch=%03d rank=%d wrote json head=%d bytes=%d",
                        epoch,
                        rank,
                        head_idx,
                        out_json.stat().st_size,
                    )
            barrier(distributed)
    else:
        for head_idx, payload in local_payloads.items():
            with open(_per_rank_json(head_idx, rank), "w", encoding="utf-8") as f:
                json.dump(payload, f)

    barrier(distributed)

    base_stats = {}
    if compute_val_loss:
        base_stats["val_train_loss"] = metrics["val_train_loss"].avg
    if compute_proxy_metrics:
        for head_idx, meters in proxy_meters.items():
            base_stats.update({f"proxy_head{head_idx}_{k}": v.avg for k, v in meters.items()})
    base_stats = reduce_dict(base_stats, distributed)

    official_heads = []
    if rank == 0:
        proxy_best_head = max(
            eval_head_indices,
            key=lambda h: base_stats.get(f"proxy_head{h}_overall", float("-inf")),
        )
        if official_eval_heads_norm in {"all", "*"}:
            official_heads = list(eval_head_indices)
        elif official_eval_heads_norm in {"proxy_best", "best", "auto"}:
            official_heads = [proxy_best_head]
        elif official_eval_heads_norm in {"none", "off", "false", "0"}:
            official_heads = []
        else:
            official_heads = _parse_head_indices(official_eval_heads_norm, num_model_heads)
        official_heads = sorted({h for h in official_heads if h in eval_head_indices})
        base_stats["proxy_best_head"] = int(proxy_best_head)

    official_metrics_by_head = {}
    official_t0 = time.time()
    if rank == 0:
        for head_idx in dump_head_indices:
            merged_json = _merged_json(head_idx)
            merge_t0 = time.time()
            logger.info(
                "validation epoch=%03d merging json head=%d/%d path=%s",
                epoch,
                head_idx,
                num_model_heads - 1,
                merged_json,
            )
            merge_result_jsons(
                [_per_rank_json(head_idx, r) for r in range(world_size)],
                merged_json,
            )
            logger.info(
                "validation epoch=%03d merged json head=%d elapsed=%.1fs",
                epoch,
                head_idx,
                time.time() - merge_t0,
            )
            if (
                run_official_metrics
                and head_idx in official_heads
                and annotations_val is not None
                and evaluator_script
            ):
                metric_t0 = time.time()
                logger.info(
                    "validation epoch=%03d official evaluator start head=%d json=%s",
                    epoch,
                    head_idx,
                    merged_json,
                )
                official_metrics = run_official_evaluator(
                    merged_json, Path(annotations_val), evaluator_script
                )
                logger.info(
                    "validation epoch=%03d official evaluator done head=%d elapsed=%.1fs",
                    epoch,
                    head_idx,
                    time.time() - metric_t0,
                )
                if official_metrics is not None:
                    official_metrics_by_head[head_idx] = official_metrics
        missing_official_heads = sorted(set(official_heads) - set(dump_head_indices))
        if missing_official_heads and run_official_metrics:
            logger.warning(
                "validation epoch=%03d skipped official heads without dumped json: %s",
                epoch,
                missing_official_heads,
            )
    barrier(distributed)
    official_elapsed = time.time() - official_t0 if rank == 0 else 0.0

    if rank == 0:
        official_primary_by_head = {}
        for head_idx, official_metrics in official_metrics_by_head.items():
            base_stats.update({f"official_head{head_idx}_{k}": float(v) for k, v in official_metrics.items()})
            if "primary_metric" in official_metrics:
                official_primary_by_head[head_idx] = float(official_metrics["primary_metric"])

        if official_primary_by_head:
            best_head = max(official_primary_by_head, key=official_primary_by_head.get)
            selection_source = "official_primary_metric"
            base_stats["official_primary_metric"] = float(official_primary_by_head[best_head])
            base_stats["official_best_head"] = int(best_head)
        else:
            best_head = max(
                eval_head_indices,
                key=lambda h: base_stats.get(f"proxy_head{h}_overall", float("-inf")),
            )
            selection_source = "proxy_overall"

        base_stats["best_head"] = int(best_head)
        for key in ("box", "noun", "noun_verb", "overall"):
            base_stats[f"proxy_{key}"] = base_stats.get(f"proxy_head{best_head}_{key}", None)

        head_proxy_overall = {
            head_idx: base_stats.get(f"proxy_head{head_idx}_overall", None)
            for head_idx in eval_head_indices
        }
        logger.info(
            "[head-selection] epoch=%03d source=%s best_head=%d proxy_overall=%s official_primary=%s",
            epoch,
            selection_source,
            best_head,
            head_proxy_overall,
            official_primary_by_head,
        )
        logger.info(
            "validation epoch=%03d done total=%.1fs official=%.1fs heads=%d official_heads=%s val_iters=%d",
            epoch,
            time.time() - validation_t0,
            official_elapsed,
            len(eval_head_indices),
            official_heads,
            eval_ipe,
        )

    return base_stats


def main(args_eval, resume_preempt=False):
    _configure_cpu_threads()
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", "./runs/ego4d_vjepa2_1_vitG_384")
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = (args_eval.get("tag", "") or "").strip()
    save_subdir = args_eval.get("save_subdir", "action_anticipation_frozen")
    load_checkpoint_mode = args_eval.get("load_checkpoint", "latest")

    args_exp = args_eval["experiment"]
    args_data = args_exp["data"]
    args_opt = args_exp["optimization"]
    args_eval_cfg = args_exp.get("evaluation", {})
    args_model = args_eval["model_kwargs"]
    requested_bfloat16 = bool(args_opt.get("use_bfloat16", True))
    use_bfloat16 = requested_bfloat16
    serialize_model_init = bool(args_eval.get("serialize_model_init", True))

    # # 顶层 launcher 会把 use_fsdp 放在 args_eval 顶层
    # use_fsdp = bool(args_eval.get("use_fsdp", False) or args_opt.get("use_fsdp", False))

    if "use_fsdp" in args_eval:
        use_fsdp = bool(args_eval["use_fsdp"])
    else:
        use_fsdp = bool(args_opt.get("use_fsdp", False))

    # main.py initializes the process group before entering this module.  In the
    # mp.Process launcher path it does not export RANK/WORLD_SIZE, so use the
    # process group as the source of truth when it is available.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
        distributed = world_size > 1
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed = False

    if requested_bfloat16 and torch.cuda.is_available():
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
        if not bf16_supported:
            use_bfloat16 = False
            if rank == 0:
                logger.warning("use_bfloat16=True requested but CUDA device does not report BF16 support; falling back to FP16 AMP.")

    allow_tf32 = bool(args_opt.get("allow_tf32", args_eval.get("allow_tf32", True)))
    cudnn_benchmark = bool(args_opt.get("cudnn_benchmark", args_eval.get("cudnn_benchmark", False)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if rank == 0:
        logger.info(
            "runtime: torch_threads=%s interop_threads=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s num_workers_per_rank=%s world_size=%s use_bfloat16=%s allow_tf32=%s cudnn_benchmark=%s",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get("MKL_NUM_THREADS"),
            os.environ.get("OPENBLAS_NUM_THREADS"),
            args_data.get("num_workers", None),
            world_size,
            use_bfloat16,
            allow_tf32,
            cudnn_benchmark,
        )

    cache_root = args_eval.get("cache_root", None)
    if cache_root:
        os.environ["EGO4D_STA_CACHE_ROOT"] = str(cache_root)

    # 每个子进程只可见一张卡，所以固定用 cuda:0
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # if rank != 0:
    #     logging.getLogger().setLevel(logging.WARNING)
    #     logger.setLevel(logging.WARNING)

    if use_fsdp and not distributed:
        raise RuntimeError(
            "use_fsdp=True but distributed initialization failed. "
            "Abort instead of silently falling back to single-GPU."
        )

    folder = Path(pretrain_folder) / save_subdir
    if eval_tag:
        folder = folder / eval_tag

    # 模型权重 / 推理结果仍然放原目录；val-only 可以显式覆盖结果目录，便于评估指定训练目录的 checkpoint。
    results_dir = Path(args_eval.get("results_dir", folder / "results"))
    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"

    # 评估日志单独放，避免和训练冲突
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")

    eval_log_dir = Path(
        args_eval.get(
            "log_dir",
            str(Path(__file__).resolve().parent / "logs"),
        )
    )
    if eval_tag:
        eval_log_dir = eval_log_dir / eval_tag
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    log_txt = eval_log_dir / f"log_{job_id}.txt"
    log_csv = eval_log_dir / f"log_r{rank}_{job_id}.csv"
    rank_log_txt = eval_log_dir / f"log_rank{rank}_{job_id}.txt"

    best_metric = float("-inf")
    start_epoch = 0

    folder.mkdir(parents=True, exist_ok=True)

    _setup_file_logger(rank_log_txt)
    logger.info("rank=%d rank_log_txt = %s", rank, rank_log_txt.resolve())

    if rank == 0:
        logger.info("log_txt = %s", log_txt.resolve())
        logger.info("log_csv = %s", log_csv.resolve())

        cfg_dump = folder / "run_config.json"
        with open(cfg_dump, "w", encoding="utf-8") as f:
            json.dump(args_eval, f, indent=2)

    prebuild_cache = bool(args_eval.get("prebuild_cache", True))
    if prebuild_cache:
        cache_stats = prebuild_sta_caches(
            train_split_root=args_data["train_split_root"],
            val_split_root=args_data["val_split_root"],
            test_split_root=args_data.get("test_split_root", None),
            build_test_cache=bool(args_eval.get("build_test_cache", False)),
        )
        if rank == 0:
            logger.info("Prebuilt cache stats: %s", cache_stats)
    
    ann_train = filter_annotations(
        dataset=args_data.get("dataset", "Ego4D_STA"),
        train_split_root=args_data["train_split_root"],
        val_split_root=args_data["val_split_root"],
        test_split_root=None,
        build_train=True,
        build_val=False,
        build_test=False,
        train_include_val=bool(args_data.get("train_include_val", False)),
        val_holdout_count=int(args_data.get("val_holdout_count", 0)),
    )

    train_annotations = ann_train["train"]
    noun_id_to_idx = ann_train["nouns"]
    verb_id_to_idx = ann_train["verbs"]

    idx_to_noun_id = {v: k for k, v in noun_id_to_idx.items()}
    idx_to_verb_id = {v: k for k, v in verb_id_to_idx.items()}
    
    if rank == 0:
        logger.info("Loaded train dataset: train=%d", len(train_annotations))

    _, train_loader, train_data_info = init_data(
        annotations=train_annotations,
        batch_size=int(args_opt.get("batch_size", 1)),
        dataset=args_data.get("dataset", "Ego4D_STA"),
        video_short_side=int(args_data.get("video_resolution", 384)),
        frames_per_clip=int(args_data.get("frames_per_clip", args_model.get("frames_per_clip", 8))),
        rank=rank,
        world_size=world_size,
        num_workers=int(args_data.get("num_workers", 0)),
        pin_mem=bool(args_data.get("pin_memory", False)),
        persistent_workers=bool(args_data.get("persistent_workers", False)),
        prefetch_factor=int(args_data.get("prefetch_factor", 2)),
        training=True,
        image_short_side_train=args_data.get("image_short_side_train", [640, 672, 704, 736, 768, 800]),
        image_short_side_val=int(args_data.get("image_short_side_val", 800)),
        image_max_size=int(args_data.get("image_max_size", 1333)),
        horizontal_flip_prob=float(args_data.get("horizontal_flip_prob", 0.0)),
        jepa_feature_root=args_data.get("jepa_feature_root", None),
        use_jepa_feature_cache=bool(args_data.get("use_jepa_feature_cache", False)),
    )

    ipe = train_loader.num_batches
    if rank == 0:
        logger.info("iterations_per_epoch train=%d", ipe)

    # ipe = train_loader.num_batches
    # val_ipe = val_loader.num_batches
    # if rank == 0:
    #     logger.info("iterations_per_epoch train=%d val=%d", ipe, val_ipe)

    fsdp_active = bool(use_fsdp and distributed and world_size > 1)

    if fsdp_active:
        from functools import partial
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        logger.info("rank=%d building model for FSDP", rank)
        model = build_model(args_model, num_nouns=len(noun_id_to_idx), num_verbs=len(verb_id_to_idx))
        logger.info("rank=%d about to wrap FSDP", rank)
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=20_000_000)

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            limit_all_gathers=True,
            use_orig_params=False,
            sync_module_states=True,
        )
        logger.info("rank=%d FSDP wrap done", rank)
    else:
        model = _build_model_serialized(
            args_model=args_model,
            num_nouns=len(noun_id_to_idx),
            num_verbs=len(verb_id_to_idx),
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            serialize_model_init=serialize_model_init,
        )
        if bool(args_data.get("use_jepa_feature_cache", False)) and bool(args_model.get("freeze_video_probe_when_using_cache", True)):
            frozen_probe_params = _freeze_video_probe_for_cached_features(model)
            if frozen_probe_params > 0:
                logger.info(
                    "rank=%d froze video_probe params=%d because use_jepa_feature_cache=True",
                    rank,
                    frozen_probe_params,
                )
        if distributed and world_size > 1:
            find_unused_parameters = bool(args_eval.get("ddp_find_unused_parameters", False))
            logger.info("rank=%d about to wrap DDP find_unused_parameters=%s", rank, find_unused_parameters)
            model = DDP(model, device_ids=[0], find_unused_parameters=find_unused_parameters)
            logger.info("rank=%d DDP wrap done", rank)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("rank=%d params total=%d trainable=%d", rank, n_params, n_trainable)
        
    # model = build_model(args_model, num_nouns=len(ann["nouns"]), num_verbs=len(ann["verbs"]))

    # fsdp_active = bool(use_fsdp and distributed and world_size > 1)

    # if fsdp_active:
    #     mp_policy = MixedPrecision(
    #         param_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    #         reduce_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    #         buffer_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    #     )
    #     model = FSDP(
    #         model,   # 关键：不要先 model.to(device)
    #         device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    #         sharding_strategy=ShardingStrategy.FULL_SHARD,
    #         mixed_precision=mp_policy,
    #         limit_all_gathers=True,
    #         use_orig_params=True,
    #         sync_module_states=True,
    #     )
    # else:
    #     model = model.to(device)
    #     if distributed and world_size > 1:
    #         model = DDP(model, device_ids=[0], find_unused_parameters=False)

    criterion = SetCriterion().to(device)

    grad_accum_steps = max(1, int(args_opt.get("grad_accum_steps", 1)))
    num_updates_per_epoch = max(1, (ipe + grad_accum_steps - 1) // grad_accum_steps)
    if rank == 0:
        logger.info(
            "optimizer_updates_per_epoch=%d (train_iters=%d grad_accum_steps=%d)",
            num_updates_per_epoch,
            ipe,
            grad_accum_steps,
        )

    # 注意：wrap 后再建 optimizer
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        model,
        iterations_per_epoch=num_updates_per_epoch,
        num_epochs=int(args_opt.get("num_epochs", 16)),
        kwargs={
            "weight_decay": args_opt.get("weight_decay", 1e-4),
            "final_weight_decay": args_opt.get("final_weight_decay", args_opt.get("weight_decay", 1e-4)),
            "lr": args_opt.get("lr", 1e-4),
            "start_lr": args_opt.get("start_lr", args_opt.get("lr", 1e-4)),
            "final_lr": args_opt.get("final_lr", 0.0),
            "warmup": args_opt.get("warmup", 0.0),
            "detector_lr_mult": args_opt.get("detector_lr_mult", 1.0),
        },
        use_bfloat16=use_bfloat16,
    )
    if rank == 0:
        logger.info(
            "optimizer param_groups=%s",
            [
                {
                    "name": group.get("name", str(i)),
                    "params": sum(p.numel() for p in group["params"]),
                    "start_lr": group.get("mc_start_lr", group.get("lr")),
                    "ref_lr": group.get("mc_ref_lr", group.get("lr")),
                }
                for i, group in enumerate(optimizer.param_groups)
            ],
        )

    eval_split = str(args_eval_cfg.get("eval_split", "val")).strip().lower()
    if eval_split not in {"val", "test"}:
        raise ValueError(f"Unsupported eval_split={eval_split!r}; expected 'val' or 'test'")
    eval_is_test = eval_split == "test"

    disable_validation = bool(args_eval_cfg.get("disable_validation", False))
    val_loader = None
    val_ipe = 0
    if disable_validation:
        if rank == 0:
            logger.info("Validation disabled by config; checkpoints will still be saved every epoch.")
    else:
        ann_val = filter_annotations(
            dataset=args_data.get("dataset", "Ego4D_STA"),
            train_split_root=args_data["train_split_root"],
            val_split_root=args_data["val_split_root"],
            test_split_root=args_data.get("test_split_root", None),
            build_train=False,
            build_val=not eval_is_test,
            build_test=eval_is_test,
            val_holdout_count=int(args_data.get("val_holdout_count", 0)),
        )

        val_annotations = ann_val["test" if eval_is_test else "val"]
        if val_annotations is None:
            raise ValueError(f"Requested eval_split={eval_split!r}, but no dataset was built")

        if rank == 0:
            logger.info("Loaded %s dataset: %s=%d", eval_split, eval_split, len(val_annotations))

        val_batch_size = int(args_eval_cfg.get("eval_batch_size", args_eval_cfg.get("batch_size", args_opt.get("batch_size", 1))))
        if rank == 0:
            logger.info("%s batch_size=%d train batch_size=%d", eval_split, val_batch_size, int(args_opt.get("batch_size", 1)))

        _, val_loader, _ = init_data(
            annotations=val_annotations,
            batch_size=val_batch_size,
            dataset=args_data.get("dataset", "Ego4D_STA"),
            video_short_side=int(args_data.get("video_resolution", 384)),
            frames_per_clip=int(args_data.get("frames_per_clip", args_model.get("frames_per_clip", 8))),
            rank=rank,
            world_size=world_size,
            num_workers=int(args_data.get("num_workers", 0)),
            pin_mem=bool(args_data.get("pin_memory", False)),
            persistent_workers=bool(args_data.get("persistent_workers", False)),
            prefetch_factor=int(args_data.get("prefetch_factor", 2)),
            training=False,
            image_short_side_train=args_data.get("image_short_side_train", [640, 672, 704, 736, 768, 800]),
            image_short_side_val=int(args_data.get("image_short_side_val", 800)),
            image_max_size=int(args_data.get("image_max_size", 1333)),
            horizontal_flip_prob=0.0,
            jepa_feature_root=args_data.get("jepa_feature_root", None),
            use_jepa_feature_cache=bool(args_data.get("use_jepa_feature_cache", False)),
        )

        val_ipe = val_loader.num_batches
        if rank == 0:
            logger.info("iterations_per_epoch %s=%d", eval_split, val_ipe)
    
    ckpt_path = None
    if load_checkpoint_mode == "best":
        ckpt_path = best_path
    elif load_checkpoint_mode == "latest":
        ckpt_path = latest_path
    elif load_checkpoint_mode:
        ckpt_path = Path(load_checkpoint_mode)

    if resume_checkpoint and ckpt_path is not None and ckpt_path.exists():
        ckpt = _load_checkpoint(
            ckpt_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            scaler=scaler,
            fsdp_active=fsdp_active,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        best_metric = float(ckpt.get("best_metric", float("-inf")))
        if ckpt.get("scheduler") is None:
            scheduler._step = float(start_epoch * num_updates_per_epoch)
            logger.info(
                "Checkpoint missing scheduler state; fallback scheduler_step=%.0f",
                scheduler._step,
            )
        if ckpt.get("wd_scheduler") is None:
            wd_scheduler._step = float(start_epoch * num_updates_per_epoch)
            logger.info(
                "Checkpoint missing wd scheduler state; fallback wd_scheduler_step=%.0f",
                wd_scheduler._step,
            )
        if rank == 0:
            logger.info("Resumed from %s at epoch=%s best_metric=%.4f", ckpt_path, start_epoch, best_metric)

    if val_only and (ckpt_path is None or not ckpt_path.exists()):
        raise FileNotFoundError(f"val_only/test inference requested but checkpoint does not exist: {ckpt_path}")

    if val_only and ckpt_path is not None and ckpt_path.exists():
        ckpt = _load_checkpoint(ckpt_path, model, fsdp_active=fsdp_active)
        start_epoch = int(ckpt.get("epoch", start_epoch) or start_epoch)

    if val_only:
        metrics = validate_and_dump(
            ipe=val_ipe,
            device=device,
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            idx_to_noun_id=idx_to_noun_id,
            idx_to_verb_id=idx_to_verb_id,
            results_dir=results_dir,
            epoch=start_epoch,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            topk_predictions=int(args_eval_cfg.get("prediction_topk", 100)),
            evaluator_script=args_eval_cfg.get("evaluator_script", None),
            annotations_val=None if eval_is_test else args_data.get("evaluator_gt_json", None),
            run_official_metrics=bool(args_eval_cfg.get("run_official", True)) and not eval_is_test,
            official_eval_heads=str(args_eval_cfg.get("official_eval_heads", "proxy_best")),
            compute_val_loss=bool(args_eval_cfg.get("compute_val_loss", False)),
            max_val_batches=args_eval_cfg.get("max_val_batches", None),
            validation_heads=str(args_eval_cfg.get("validation_heads", "all")),
            dump_predictions=str(args_eval_cfg.get("dump_predictions", "official_only")),
            compute_proxy_metrics=bool(args_eval_cfg.get("compute_proxy_metrics", True)),
            log_interval=int(args_eval_cfg.get("validation_log_interval", 100)),
        )
        if rank == 0:
            logger.info("Validation metrics: %s", metrics)
            _append_csv_row(
                log_csv,
                _metric_csv_row(
                    epoch=start_epoch,
                    train_loss=None,
                    metrics=metrics,
                    best_metric=best_metric,
                    is_best=0,
                ),
            )
        return

    num_epochs = int(args_opt.get("num_epochs", 16))
    validate_interval = max(1, int(args_eval_cfg.get("validate_interval", 1)))
    official_eval_interval = max(1, int(args_eval_cfg.get("official_eval_interval", 1)))
    skip_initial_validation_epochs = max(0, int(args_eval_cfg.get("skip_initial_validation_epochs", 0)))
    run_official = bool(args_eval_cfg.get("run_official", True))
    best_metric_source = str(args_eval_cfg.get("best_metric_source", "official")).strip().lower()
    if rank == 0 and skip_initial_validation_epochs > 0:
        logger.info(
            "Skipping validation for the first %d epoch(s); first validation can run at epoch %d",
            skip_initial_validation_epochs,
            skip_initial_validation_epochs + 1,
        )

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_data_info, "sampler") and getattr(train_data_info, "sampler", None) is not None:
            train_data_info.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            ipe=ipe,
            device=device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            scaler=scaler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            grad_accum_steps=int(args_opt.get("grad_accum_steps", 1)),
            rank=rank,
        )

        epoch_num = epoch + 1
        validation_warmup_done = epoch_num > skip_initial_validation_epochs
        should_validate = (not disable_validation) and validation_warmup_done and (
            (epoch_num % validate_interval == 0) or (epoch_num == num_epochs)
        )
        should_run_official = (not disable_validation) and run_official and (
            (epoch_num % official_eval_interval == 0) or (epoch_num == num_epochs)
        )
        if should_validate:
            val_stats = validate_and_dump(
                ipe=val_ipe,
                device=device,
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                use_bfloat16=use_bfloat16,
                idx_to_noun_id=idx_to_noun_id,
                idx_to_verb_id=idx_to_verb_id,
                results_dir=results_dir,
                epoch=epoch_num,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                topk_predictions=int(args_eval_cfg.get("prediction_topk", 100)),
                evaluator_script=args_eval_cfg.get("evaluator_script", None),
                annotations_val=args_data.get("evaluator_gt_json", None),
                run_official_metrics=should_run_official,
                official_eval_heads=str(args_eval_cfg.get("official_eval_heads", "proxy_best")),
                compute_val_loss=bool(args_eval_cfg.get("compute_val_loss", False)),
                max_val_batches=args_eval_cfg.get("max_val_batches", None),
                validation_heads=str(args_eval_cfg.get("validation_heads", "all")),
                dump_predictions=str(args_eval_cfg.get("dump_predictions", "official_only")),
                compute_proxy_metrics=bool(args_eval_cfg.get("compute_proxy_metrics", True)),
                log_interval=int(args_eval_cfg.get("validation_log_interval", 100)),
            )
        else:
            val_stats = {}
            if rank == 0 and not validation_warmup_done:
                logger.info(
                    "epoch=%s validation skipped during initial warmup (%d/%d)",
                    epoch_num,
                    epoch_num,
                    skip_initial_validation_epochs,
                )

        if best_metric_source in {"official", "official_primary", "official_primary_metric"}:
            metric_value = val_stats.get("official_primary_metric", None)
        elif best_metric_source in {"proxy", "proxy_overall"}:
            metric_value = val_stats.get("proxy_overall", None)
        elif best_metric_source in {"official_or_proxy", "auto"}:
            metric_value = val_stats.get("official_primary_metric", val_stats.get("proxy_overall", None))
        else:
            raise ValueError(f"Unsupported best_metric_source={best_metric_source}")
        current_metric = float(metric_value) if metric_value is not None else float("-inf")
        if rank == 0:
            if should_validate and metric_value is None and best_metric_source in {"official", "official_primary", "official_primary_metric"}:
                logger.info(
                    "epoch=%s best checkpoint not updated because best_metric_source=%s and no official_primary_metric was produced",
                    epoch_num,
                    best_metric_source,
                )
            best_updated = metric_value is not None and current_metric > best_metric
            if best_updated:
                best_metric = current_metric
            is_best = best_updated
        else:
            is_best = False

        best_metric, is_best = _broadcast_best_info(best_metric, is_best, device, distributed, rank)

        _save_checkpoint_bundle(
            folder=folder,
            epoch=epoch_num,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            scaler=scaler,
            best_metric=best_metric,
            metrics={"train": train_stats, "val": val_stats},
            args_eval=args_eval,
            idx_to_noun_id=idx_to_noun_id,
            idx_to_verb_id=idx_to_verb_id,
            is_best=is_best,
            fsdp_active=fsdp_active,
            distributed=distributed,
            rank=rank,
        )

        if rank == 0:
            logger.info("epoch=%s train=%s val=%s", epoch_num, train_stats, val_stats)
            _append_csv_row(
                log_csv,
                _metric_csv_row(
                    epoch=epoch_num,
                    train_loss=train_stats.get("loss", None),
                    metrics=val_stats,
                    best_metric=best_metric,
                    is_best=is_best,
                ),
            )
            if is_best:
                logger.info(
                    "New best checkpoint at epoch %s with primary metric %.4f from head %s",
                    epoch_num,
                    best_metric,
                    val_stats.get("best_head", None),
                )
