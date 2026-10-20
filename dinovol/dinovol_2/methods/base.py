"""The boundary between a training algorithm and the shared runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


class TrainingMethod(nn.Module):
    """Own models, objectives and algorithm state; leave backward/stepping to Trainer.

    Forward must use the method's DDP-wrapped model when distributed. Metrics are
    detached scalar tensors so the trainer can transfer them to the host together.
    Hooks run once per optimizer step, except forward (once per microbatch).
    """
    name: str

    def forward(self, batch, step: int) -> StepOutput:
        raise NotImplementedError

    def parameter_groups(self, **kwargs):
        raise NotImplementedError

    def gradient_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def before_optimizer_step(self, step: int) -> None:
        pass

    def after_optimizer_step(self, step: int) -> None:
        pass

    def adjust_optimizer_groups(self, groups, step: int) -> None:
        pass

    def step_metrics(self, step: int) -> dict[str, float]:
        return {}

    def evaluation_backbone(self) -> nn.Module:
        raise NotImplementedError

    def evaluate(self, batch, step: int) -> Mapping[str, float]:
        output = self(batch, step)
        return {name: float(value) for name, value in {"loss": output.loss, **output.metrics}.items()}

    def dataset_config(self, config, *, training: bool):
        return dict(config)

    def build_collate(self, dataset):
        from dinovol_2.ops.collate import collate_multicrop_batch
        return collate_multicrop_batch

    def checkpoint_state(self) -> dict[str, Any]:
        return {"model": self.state_dict()}

    def load_checkpoint_state(self, state: Mapping[str, Any], step: int):
        self.load_state_dict(state["model"])
        return ()

    def rank_checkpoint_state(self) -> dict[str, Any]:
        """Small, CPU-resident algorithm state that differs between ranks."""
        return {}

    def load_rank_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        pass

    def load_optimizer_state(self, optimizer, state, added=()) -> None:
        optimizer.load_state_dict(state)

    def save_monitor_image(self, batch, step, metrics, *, directory, config):
        return None
