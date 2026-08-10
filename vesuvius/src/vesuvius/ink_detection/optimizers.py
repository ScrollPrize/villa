"""Training parameter groups and delegation to vesuvius optimizer creation."""

from __future__ import annotations

from torch import nn

from vesuvius.ink_detection.config import TrainingConfig


class OptimizerParamGroupTarget:
    """Expose an authored parameter-group list through parameters()."""

    def __init__(self, param_groups: list[dict]) -> None:
        self._param_groups = param_groups

    def parameters(self) -> list[dict]:
        return self._param_groups


def plan_optimizer_target(
    model: nn.Module, config: TrainingConfig
) -> nn.Module | OptimizerParamGroupTarget:
    """Freeze or separately scale a configured pretrained shared encoder."""

    model_mapping = config.ink.to_mapping()
    model_config = model_mapping.get("model_config") or {}
    pretrained_backbone = model_config.get("pretrained_backbone")
    if not pretrained_backbone:
        return model
    encoder_lr_mult = float(config.optimizer.encoder_lr_mult)

    freeze_encoder = bool(
        model_mapping.get("freeze_encoder", False)
        or model_config.get("freeze_encoder", False)
    )
    encoder_params = list(model.shared_encoder.parameters())
    if freeze_encoder:
        for parameter in encoder_params:
            parameter.requires_grad = False

    if not freeze_encoder and encoder_lr_mult == 1.0:
        return model

    encoder_param_ids = {id(parameter) for parameter in encoder_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_param_ids
    ]
    param_groups: list[dict] = []
    if other_params:
        param_groups.append({"params": other_params})
    if not freeze_encoder and encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": (
                    config.optimizer.learning_rate
                    * encoder_lr_mult
                ),
            }
        )
    if not param_groups:
        raise ValueError(
            "No trainable parameters remain after applying freeze_encoder"
        )
    return OptimizerParamGroupTarget(param_groups)


def create_training_optimizer(model: nn.Module, config: TrainingConfig):
    """Create the configured optimizer after applying ink-specific groups."""

    from vesuvius.models.training.optimizers import create_optimizer

    optimizer = config.optimizer
    return create_optimizer(
        {
            "name": optimizer.name,
            "learning_rate": optimizer.learning_rate,
            "weight_decay": optimizer.weight_decay,
            "betas": optimizer.betas,
            "momentum": optimizer.momentum,
            "nesterov": optimizer.nesterov,
        },
        plan_optimizer_target(model, config),
    )
