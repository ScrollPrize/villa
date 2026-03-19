from __future__ import annotations

import os
from typing import Any, Mapping

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def resolve_distributed_config(config: Mapping[str, Any]) -> dict[str, int | bool]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(config.get("use_ddp", False)) or env_world_size > 1
    return {
        "use_ddp": use_ddp,
        "world_size": env_world_size if use_ddp else 1,
        "rank": int(os.environ.get("RANK", "0")) if use_ddp else 0,
        "local_rank": int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) if use_ddp else 0,
    }


def build_distributed_sampler(
    dataset: Dataset[Any],
    *,
    is_distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> DistributedSampler[Any] | None:
    if not is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )
