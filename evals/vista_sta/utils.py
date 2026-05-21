
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def init_distributed():
    try:
        from src.utils.distributed import init_distributed as repo_init_distributed
        out = repo_init_distributed()
        if isinstance(out, tuple):
            if len(out) == 2:
                world_size, rank = out
            elif len(out) >= 3:
                world_size, rank = out[0], out[1]
            else:
                world_size, rank = 1, 0
        else:
            world_size, rank = 1, 0
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
        distributed = int(world_size) > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()
        return int(rank), int(world_size), int(local_rank), bool(distributed)
    except Exception:
        pass

    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', 0))))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', os.environ.get('PMI_SIZE', 1))))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    distributed = world_size > 1
    if distributed and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        timeout_s = int(os.environ.get("TORCH_DIST_TIMEOUT_SEC", "7200"))
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=timeout_s),
        )
    distributed = distributed and torch.distributed.is_available() and torch.distributed.is_initialized()
    return rank, world_size, local_rank, distributed


def reduce_dict(input_dict: Dict[str, float], distributed: bool) -> Dict[str, float]:
    if not input_dict:
        return input_dict
    if not distributed:
        return input_dict
    keys = sorted(input_dict.keys())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    values = torch.tensor([input_dict[k] for k in keys], device=device, dtype=torch.float32)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values = values / torch.distributed.get_world_size()
    return {k: float(v) for k, v in zip(keys, values.cpu().tolist())}


def barrier(distributed: bool) -> None:
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["device_ids"] = [torch.cuda.current_device()]
        torch.distributed.barrier(**kwargs)


def init_opt(model: torch.nn.Module, iterations_per_epoch: int, num_epochs: int, kwargs: Dict, use_bfloat16: bool = False):
    start_lr = float(kwargs.get('start_lr', kwargs.get('lr', 1e-4)))
    ref_lr = float(kwargs.get('lr', 1e-4))
    ref_wd = float(kwargs.get('weight_decay', 1e-4))
    final_wd = float(kwargs.get('final_weight_decay', ref_wd))
    warmup_steps = int(float(kwargs.get('warmup', 0.0)) * iterations_per_epoch)
    final_lr = float(kwargs.get('final_lr', 0.0))

    def _make_group(params, lr_mult: float, name: str):
        params = [p for p in params if p.requires_grad]
        if not params:
            return None
        return {
            'params': params,
            'lr': start_lr * lr_mult,
            'weight_decay': ref_wd,
            'mc_warmup_steps': warmup_steps,
            'mc_start_lr': start_lr * lr_mult,
            'mc_ref_lr': ref_lr * lr_mult,
            'mc_final_lr': final_lr * lr_mult,
            'mc_ref_wd': ref_wd,
            'mc_final_wd': final_wd,
            'name': name,
        }

    detector_lr_mult = float(kwargs.get('detector_lr_mult', 1.0))
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model
    if detector_lr_mult != 1.0 and hasattr(base_model, "detector"):
        detector_params = [p for p in base_model.detector.parameters() if p.requires_grad]
        detector_param_ids = {id(p) for p in detector_params}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in detector_param_ids
        ]
        param_groups = [
            g for g in (
                _make_group(detector_params, detector_lr_mult, "detector"),
                _make_group(other_params, 1.0, "heads"),
            )
            if g is not None
        ]
    else:
        param_groups = [_make_group([p for p in model.parameters() if p.requires_grad], 1.0, "all")]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineLRSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
    scaler = None
    if torch.cuda.is_available() and not use_bfloat16:
        scaler = torch.amp.GradScaler("cuda")
    return optimizer, scaler, scheduler, wd_scheduler


class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get('mc_ref_lr')
            final_lr = group.get('mc_final_lr')
            start_lr = group.get('mc_start_lr')
            warmup_steps = group.get('mc_warmup_steps')
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            group['lr'] = new_lr

    def state_dict(self) -> Dict:
        return {
            'T_max': self.T_max,
            '_step': self._step,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.T_max = int(state_dict.get('T_max', self.T_max))
        self._step = float(state_dict.get('_step', 0.0))


class CosineWDSchedule(object):
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get('mc_ref_wd')
            final_wd = group.get('mc_final_wd')
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group['weight_decay'] = new_wd

    def state_dict(self) -> Dict:
        return {
            'T_max': self.T_max,
            '_step': self._step,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.T_max = int(state_dict.get('T_max', self.T_max))
        self._step = float(state_dict.get('_step', 0.0))


def empty_results_json() -> Dict:
    return {
        'version': '1.0',
        'challenge': 'ego4d_short_term_object_interaction_anticipation',
        'results': {},
    }


def merge_result_jsons(paths: List[Path], out_path: Path) -> Path:
    merged = empty_results_json()
    for p in paths:
        if not p.exists():
            continue
        with open(p, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        merged['results'].update(payload.get('results', {}))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f)
    return out_path


def run_official_evaluator(results_json: Path, annotations_json: Path, evaluator_script: Optional[str]) -> Optional[Dict[str, float]]:
    if not evaluator_script:
        return None
    if not Path(evaluator_script).exists():
        logger.warning('Official evaluator script not found: %s', evaluator_script)
        return None
    cmd = [sys.executable, evaluator_script, str(results_json), str(annotations_json)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        logger.warning('Official evaluator failed with code %s\n%s', proc.returncode, proc.stdout)
        return None
    metrics: Dict[str, float] = {}
    primary = None
    lines = proc.stdout.splitlines()
    for line in lines:
        m = re.match(r'^(.*?):\s*([0-9]+(?:\.[0-9]+)?)\s*$', line.strip())
        if not m:
            continue
        name = m.group(1).strip()
        val = float(m.group(2))
        metrics[name] = val
        if name.startswith('*'):
            primary = val
    if primary is None and metrics:
        primary = list(metrics.values())[-1]
    if primary is not None:
        metrics['primary_metric'] = primary
    logger.info('Official evaluator output: %s', metrics)
    return metrics or None
