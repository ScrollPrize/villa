"""Batch dice under DDP must match one process over the whole batch.

With ddp=True and batch_dice=True both dice losses all-gather their per-class
statistics and sum away the world axis. Two gloo ranks each hold half of a
batch; their loss and their input gradients must equal those of a single
process that sees the whole batch with ddp=False.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vesuvius.models.training.loss.nnunet_losses import (
    MemoryEfficientSoftDiceLoss,
    SoftDiceLoss,
)


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)

WORLD = 2
LOSSES = {"soft": SoftDiceLoss, "memory_efficient": MemoryEfficientSoftDiceLoss}


def _batch():
    generator = torch.Generator().manual_seed(0)
    probabilities = torch.rand(4, 3, 8, 8, generator=generator)
    labels = torch.randint(0, 3, (4, 1, 8, 8), generator=generator)
    return probabilities, labels


def _loss(name, ddp):
    return LOSSES[name](apply_nonlin=None, batch_dice=True, do_bg=True, smooth=1.0, ddp=ddp)


def _rank_half(rank, tensor):
    half = tensor.shape[0] // WORLD
    return tensor[rank * half:(rank + 1) * half]


def _worker(rank, name, init_file, out_prefix):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    try:
        probabilities, labels = _batch()
        local = _rank_half(rank, probabilities).clone().requires_grad_(True)
        loss = _loss(name, ddp=True)(local, _rank_half(rank, labels))
        loss.backward()
        torch.save({"loss": loss.detach(), "grad": local.grad}, f"{out_prefix}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_batch_dice_under_ddp_matches_one_process(tmp_path, name):
    probabilities, labels = _batch()
    probabilities.requires_grad_(True)
    reference = _loss(name, ddp=False)(probabilities, labels)
    reference.backward()

    out_prefix = str(tmp_path / "rank")
    mp.spawn(
        _worker,
        args=(name, str(tmp_path / "process_group"), out_prefix),
        nprocs=WORLD,
        join=True,
    )

    for rank in range(WORLD):
        result = torch.load(f"{out_prefix}.{rank}")
        torch.testing.assert_close(result["loss"], reference.detach())
        torch.testing.assert_close(result["grad"], _rank_half(rank, probabilities.grad))
