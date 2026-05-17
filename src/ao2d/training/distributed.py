from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(distributed, rank, world_size, local_rank, device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(module: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.distributed:
        return module
    if ctx.device.type == "cuda":
        return DistributedDataParallel(module, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DistributedDataParallel(module)


def unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def make_sampler(dataset: Dataset | None, ctx: DistributedContext, shuffle: bool) -> DistributedSampler | None:
    if dataset is None or not ctx.distributed:
        return None
    return DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=shuffle, drop_last=shuffle)


def set_sampler_epoch(sampler, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_metrics(metrics: dict[str, float], ctx: DistributedContext) -> dict[str, float]:
    if not ctx.distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[k]) for k in keys], device=ctx.device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / ctx.world_size
    return {k: float(v) for k, v in zip(keys, values.tolist(), strict=True)}

