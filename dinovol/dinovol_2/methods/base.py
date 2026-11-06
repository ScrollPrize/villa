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

    def forward_backward(self, batch, step: int, *, autocast, sync_context, backward) -> StepOutput:
        """Compute the objective and backpropagate it for one microbatch.

        autocast() is the forward precision context. sync_context(sync) wraps every DDP
        forward+backward; only sync=True on the final backward of the last microbatch
        all-reduces gradients. backward(loss) scales and backpropagates. Methods that
        need several backward passes per microbatch (source chunks) override this.
        """
        with sync_context(True):
            with autocast():
                output = self(batch, step)
            backward(output.loss)
        return output

    def parameter_groups(self, **kwargs):
        raise NotImplementedError

    def gradient_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def before_optimizer_step(self, step: int) -> None:
        pass

    def after_optimizer_step(self, step: int) -> None:
        pass

    def optimizer_step_skipped(self, step: int) -> None:
        """Called instead of after_optimizer_step when the AMP scaler skipped the update."""
        self.after_optimizer_step(step)

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

    def build_processor(self, dataset, device):
        """Parent-process image processing and collation for a dataset's worker samples."""
        from dinovol_2.dataset.cuda_pipeline import BatchProcessor
        return BatchProcessor(dataset.config, self.build_collate(dataset), device=device)

    def monitor_subset(self, batch, count: int):
        """The first `count` samples of a collated batch, for save_monitor_image."""
        batch_size = int(batch["batch_size"])
        n_global_views = int(batch["n_global_views"])
        sample_count = min(batch_size, count)
        global_crops = batch["collated_global_crops"]
        index = torch.tensor(
            [view * batch_size + sample for view in range(n_global_views) for sample in range(sample_count)],
            dtype=torch.long, device=global_crops.device,
        )
        return {
            "collated_global_crops": global_crops.index_select(0, index),
            "n_global_views": n_global_views,
            "batch_size": sample_count,
        }

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
