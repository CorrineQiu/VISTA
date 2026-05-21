from __future__ import annotations
import os
from logging import getLogger
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

from .ego4d_sta import (
    Ego4DSTASampleDirDataset,
    collate_sta_batch,
    build_label_maps_from_roots,
)

logger = getLogger(__name__)


def filter_annotations(
    dataset,
    train_split_root,
    val_split_root,
    test_split_root=None,
    build_train=True,
    build_val=True,
    build_test=False,
    train_include_val=False,
    val_holdout_count=0,
    **kwargs,
):
    del dataset, kwargs
    val_holdout_count = max(0, int(val_holdout_count))

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if rank == 0:
        logger.info(
            "[filter_annotations] start train=%s val=%s test=%s",
            train_split_root, val_split_root, test_split_root
        )

    noun_id_to_idx, verb_id_to_idx = build_label_maps_from_roots(
        [train_split_root, val_split_root]
    )

    if rank == 0:
        logger.info(
            "[filter_annotations] label maps built nouns=%d verbs=%d",
            len(noun_id_to_idx), len(verb_id_to_idx)
        )

    train_ds = None
    val_ds = None
    test_ds = None

    if build_train:
        if rank == 0:
            logger.info("[filter_annotations] building train dataset")
        train_ds = Ego4DSTASampleDirDataset(
            split_root=train_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=True,
            train=True,
        )
        if train_include_val:
            if rank == 0:
                logger.info(
                    "[filter_annotations] adding val split into train dataset; holdout_count=%d",
                    val_holdout_count,
                )
            val_as_train_ds = Ego4DSTASampleDirDataset(
                split_root=val_split_root,
                noun_id_to_idx=noun_id_to_idx,
                verb_id_to_idx=verb_id_to_idx,
                build_label_map=False,
                require_labels=True,
                train=True,
            )
            if val_holdout_count > 0:
                start = min(val_holdout_count, len(val_as_train_ds))
                val_as_train_ds = Subset(val_as_train_ds, range(start, len(val_as_train_ds)))
            train_ds = ConcatDataset([train_ds, val_as_train_ds])

    if build_val:
        if rank == 0:
            logger.info("[filter_annotations] building val dataset")
        val_ds = Ego4DSTASampleDirDataset(
            split_root=val_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=True,
            train=False,
        )
        if val_holdout_count > 0:
            stop = min(val_holdout_count, len(val_ds))
            val_ds = Subset(val_ds, range(stop))

    if build_test and test_split_root is not None:
        if rank == 0:
            logger.info("[filter_annotations] building test dataset")
        test_ds = Ego4DSTASampleDirDataset(
            split_root=test_split_root,
            noun_id_to_idx=noun_id_to_idx,
            verb_id_to_idx=verb_id_to_idx,
            build_label_map=False,
            require_labels=False,
            train=False,
        )

    return {
        "nouns": noun_id_to_idx,
        "verbs": verb_id_to_idx,
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }


def init_data(
    annotations,
    batch_size,
    dataset,
    video_short_side=384,
    frames_per_clip=8,
    rank=0,
    world_size=1,
    num_workers=2,
    pin_mem=True,
    persistent_workers=False,
    prefetch_factor=2,
    training=True,
    image_short_side_train=(640, 672, 704, 736, 768, 800),
    image_short_side_val=800,
    image_max_size=1333,
    horizontal_flip_prob=0.0,
    jepa_feature_root=None,
    use_jepa_feature_cache=False,
):
    del dataset

    ds = annotations
    def _apply_runtime_cfg(one_ds):
        one_ds.video_short_side = int(video_short_side)
        one_ds.frames_per_clip = int(frames_per_clip)
        one_ds.image_short_side_train = list(image_short_side_train)
        one_ds.image_short_side_val = int(image_short_side_val)
        one_ds.image_max_size = int(image_max_size)
        one_ds.train = bool(training)
        one_ds.horizontal_flip_prob = float(horizontal_flip_prob if training else 0.0)
        one_ds.jepa_feature_root = jepa_feature_root
        one_ds.use_jepa_feature_cache = bool(use_jepa_feature_cache)

    def _apply_runtime_cfg_recursive(one_ds):
        _apply_runtime_cfg(one_ds)
        for child in getattr(one_ds, "datasets", []):
            _apply_runtime_cfg_recursive(child)
        child = getattr(one_ds, "dataset", None)
        if child is not None:
            _apply_runtime_cfg_recursive(child)

    _apply_runtime_cfg_recursive(ds)

    sampler = DistributedSampler(
        ds,
        shuffle=training,
        num_replicas=world_size,
        rank=rank,
    ) if world_size > 1 else None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": (sampler is None and training),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "persistent_workers": (num_workers > 0) and persistent_workers,
        "collate_fn": collate_sta_batch,
        "drop_last": training,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(ds, **loader_kwargs)
    loader.num_batches = len(loader)
    loader.num_samples = len(ds)

    data_info = type(
        "DataInfo",
        (),
        {
            "dataloader": loader,
            "sampler": sampler,
            "set_epoch": lambda self, epoch: sampler.set_epoch(epoch) if sampler is not None else None,
        },
    )()
    return ds, loader, data_info
