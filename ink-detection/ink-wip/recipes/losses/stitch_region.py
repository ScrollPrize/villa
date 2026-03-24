from __future__ import annotations

from dataclasses import dataclass, field

from ink.recipes.losses.dice_bce import DiceBCEBatch
from ink.recipes.losses.reporting import resolve_train_output


@dataclass(frozen=True)
class StitchRegionLoss:
    patch_loss: object = field(default_factory=DiceBCEBatch)

    def compute(self, batch):
        if not callable(self.patch_loss):
            raise TypeError("patch_loss must be callable")
        output = resolve_train_output(
            self.patch_loss,
            batch.logits[None, None],
            batch.targets[None, None],
            valid_mask=batch.valid_mask[None, None],
        )
        loss = output.loss
        if getattr(loss, "ndim", 0) > 0:
            loss = loss.reshape(-1).mean()
        components = output.components

        def _component_value(*keys: str):
            for key in keys:
                if key not in components:
                    continue
                value = components[key]
                if getattr(value, "ndim", 0) > 0:
                    value = value.reshape(-1).mean()
                return value.detach()
            return None

        dice_loss = _component_value("dice_loss")
        if dice_loss is None:
            dice_loss = loss.new_tensor(0.0)
        dice = _component_value("dice")
        if dice is None:
            dice = 1.0 - dice_loss
        bce = _component_value("bce", "bce_loss")
        if bce is None:
            bce = loss.new_tensor(0.0)

        return {
            "loss": loss,
            "region_loss": loss,
            "dice": dice,
            "bce": bce,
            "dice_loss": dice_loss,
        }
