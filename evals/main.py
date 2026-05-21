# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))

import yaml

from evals.scaffold import main as eval_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--val_only", action="store_true", help="only run eval", default=False)
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to debug with checkpointing.",
)
parser.add_argument(
    "--folder",
    type=str,
    help="location to save logs",
    default="",
)
parser.add_argument("--override_config_folder", action="store_true")
parser.add_argument("--checkpoint", type=str, help="location of pretrained ckpt")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--use_fsdp", action="store_true")


def _expand_config_strings(obj):
    if isinstance(obj, dict):
        return {k: _expand_config_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_config_strings(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def _configure_master_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" in os.environ:
        return
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id.isdigit():
        os.environ["MASTER_PORT"] = str(20000 + int(slurm_job_id) % 40000)
    else:
        os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)


def process_main(args, rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"

    import logging
    import torch
    torch_threads = max(1, int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if args.val_only:
            params["val_only"] = True

        if args.checkpoint:
            params["model_kwargs"]["checkpoint"] = args.checkpoint

        if args.model_name:
            params["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = args.model_name

        if args.batch_size:
            params["experiment"]["optimization"]["batch_size"] = args.batch_size

        if args.override_config_folder:
            params["folder"] = args.folder
        params["use_fsdp"] = args.use_fsdp
        params = _expand_config_strings(params)
        logger.info("loaded params...")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    requested_world_size = world_size
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    if requested_world_size > 1 and world_size <= 1:
        raise RuntimeError(
            f"Distributed initialization failed: requested world_size={requested_world_size}, got world_size={world_size}."
        )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    logger.info(
        "Running... (rank: %s/%s) torch_threads=%s interop_threads=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s",
        rank,
        world_size,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
    )

    # Launch the eval with loaded config
    eval_main(params["eval_name"], args_eval=params)


if __name__ == "__main__":
    args = parser.parse_args()
    _configure_master_env()
    if args.debugmode:
        # FSDP debugging (use torchrun)
        if args.use_fsdp:
            process_main(
                args=args,
                rank=int(os.environ["RANK"]),
                fname=args.fname,
                world_size=int(os.environ["WORLD_SIZE"]),
                devices=args.devices,
            )
        # Single-GPU debugging
        else:
            process_main(args=args, rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        processes = []
        for rank in range(num_gpus):
            process = mp.Process(target=process_main, args=(args, rank, args.fname, num_gpus, args.devices))
            process.start()
            processes.append(process)

        def stop_alive_processes(alive_ranks):
            for alive_rank in list(alive_ranks):
                process = processes[alive_rank]
                if process.is_alive():
                    process.terminate()
            deadline = time.time() + 30
            while alive_ranks and time.time() < deadline:
                for alive_rank in list(alive_ranks):
                    process = processes[alive_rank]
                    process.join(timeout=1)
                    if process.exitcode is not None:
                        alive_ranks.remove(alive_rank)
            for alive_rank in list(alive_ranks):
                process = processes[alive_rank]
                if process.is_alive():
                    process.kill()
            for alive_rank in list(alive_ranks):
                processes[alive_rank].join(timeout=5)

        failed = []
        alive = set(range(num_gpus))
        try:
            while alive:
                for rank in list(alive):
                    process = processes[rank]
                    process.join(timeout=1)
                    if process.exitcode is None:
                        continue
                    alive.remove(rank)
                    if process.exitcode != 0:
                        failed.append((rank, process.exitcode))
                if failed:
                    stop_alive_processes(alive)
                    break
        except KeyboardInterrupt:
            stop_alive_processes(alive)
            raise
        if failed:
            raise SystemExit(f"One or more distributed workers failed: {failed}")
