from __future__ import annotations

from dataclasses import dataclass

from ink.core.types import DataBundle


@dataclass
class StitchTraining:
    experiment: object
    train_loader: object
    val_loader: object

    def run_step(self, batch):
        del batch
        raise NotImplementedError("stitched training runner is not implemented")

    def run_epoch(self, *, optimizer_setup=None, device=None):
        del optimizer_setup, device
        raise NotImplementedError("stitched training runner is not implemented")

    def run(self, **kwargs):
        del kwargs
        raise NotImplementedError("stitched training runner is not implemented")


@dataclass(frozen=True)
class StitchTrainer:
    def build(self, *, experiment, data: DataBundle, logger=None) -> StitchTraining:
        del logger
        return StitchTraining(
            experiment=experiment,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
        )
