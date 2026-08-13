from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from ..train import BaseTrainer
from ...datasets.mutex_affinity_dataset import MutexAffinityDataset, TargetSpec
from ...evaluation.affinity_metrics import AffinityMetricConfig, AffinityStatsMetric


class MutexAffinityTrainer(BaseTrainer):
    """
    Trainer that optimises the CNN to predict Mutex Watershed affinities.

    The trainer expects the dataset to provide:
      - target tensors named after the configured affinity heads
      - optional masks stored under ``f\"{target}_mask\"``

    Losses follow Section 5.1 of the Mutex Watershed paper, using the
    Sørensen-Dice formulation for both attractive (on the complement)
    and repulsive edges.
    """

    MASK_SUFFIX = "_mask"

    def __init__(self, mgr=None, verbose: bool = True) -> None:
        super().__init__(mgr=mgr, verbose=verbose)
        if getattr(self.mgr, "enable_deep_supervision", False):
            self.mgr.enable_deep_supervision = False
            self.mgr.tr_configs["enable_deep_supervision"] = False
            if verbose:
                print("Deep supervision disabled for MutexAffinityTrainer")
        self._current_masks: Dict[str, torch.Tensor] = {}
        self._affinity_specs: Dict[str, TargetSpec] = {}
        self._mutex_targets: Sequence[str] = ()
        smoothing_cfg = getattr(self.mgr, "affinity_label_smoothing", None)
        if smoothing_cfg is None:
            smoothing_cfg = (self.mgr.tr_configs or {}).get("affinity_label_smoothing")
        self._label_smoothing = float(smoothing_cfg) if smoothing_cfg is not None else 0.0
        if self._label_smoothing < 0.0 or self._label_smoothing >= 0.5:
            raise ValueError(
                f"affinity_label_smoothing must be in [0, 0.5); received {self._label_smoothing}."
            )
        self.mgr.tr_configs["affinity_label_smoothing"] = self._label_smoothing

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------
    def _build_dataset_for_mgr(self, mgr, *, is_training: bool):
        dataset = MutexAffinityDataset(mgr, is_training=is_training)
        self._affinity_specs = dict(dataset.affinity_specs)
        self._mutex_targets = tuple(self._affinity_specs.keys())
        return dataset

    def _initialize_evaluation_metrics(self):
        base_metrics = super()._initialize_evaluation_metrics()
        for target_name in self._mutex_targets:
            spec = self._affinity_specs.get(target_name)
            invert = bool(spec.invert_for_loss) if spec is not None else False
            config = AffinityMetricConfig(
                apply_sigmoid=True,
                invert=invert,
                name=f"{target_name}_affinity",
            )
            base_metrics[target_name] = [AffinityStatsMetric(config=config)]
        return base_metrics

    # ------------------------------------------------------------------
    # Forward pass plumbing
    # ------------------------------------------------------------------
    def _get_model_outputs(self, model, data_dict):
        mask_tensors: Dict[str, torch.Tensor] = {}
        for key, value in list(data_dict.items()):
            if key.endswith(self.MASK_SUFFIX) and hasattr(value, "to"):
                target_name = key[: -len(self.MASK_SUFFIX)]
                mask_tensors[target_name] = value.to(self.device)

        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in {"image", "patch_info", "is_unlabeled", "regression_keys"}
            and hasattr(v, "to")
            and not k.endswith(self.MASK_SUFFIX)
        }

        raw_outputs = model(inputs)

        if isinstance(raw_outputs, dict):
            outputs_dict = raw_outputs
        elif isinstance(raw_outputs, (list, tuple)):
            if raw_outputs and isinstance(raw_outputs[0], dict):
                outputs_dict = raw_outputs[0]
            else:
                outputs_dict = {}
                target_names = list(self._mutex_targets)
                for idx, tensor in enumerate(raw_outputs):
                    name = target_names[idx] if idx < len(target_names) else f"output_{idx}"
                    outputs_dict[name] = tensor
        else:
            primary_name = next(iter(self._mutex_targets), "output")
            outputs_dict = {primary_name: raw_outputs}

        self._current_masks = mask_tensors
        return inputs, targets_dict, outputs_dict

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        return self._compute_mutex_losses(
            outputs,
            targets_dict,
            loss_fns,
            fallback_fn=super()._compute_train_loss,
        )

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        with torch.no_grad():
            total_loss, task_losses = self._compute_mutex_losses(
                outputs,
                targets_dict,
                loss_fns,
                fallback_fn=super()._compute_validation_loss,
            )
        return task_losses

    def _compute_mutex_losses(
        self,
        outputs,
        targets_dict,
        loss_fns,
        *,
        fallback_fn,
    ):
        filtered_targets = {
            name: tensor
            for name, tensor in targets_dict.items()
            if not name.endswith(self.MASK_SUFFIX)
        }

        custom_targets = [t for t in filtered_targets if t in self._mutex_targets]
        other_targets = [t for t in filtered_targets if t not in custom_targets]

        total_loss: Optional[torch.Tensor] = None
        task_losses: Dict[str, torch.Tensor] = {}

        for target_name in custom_targets:
            pred = outputs[target_name]
            gt = filtered_targets[target_name]
            mask = self._current_masks.get(target_name)
            spec = self._affinity_specs.get(target_name)
            invert = bool(spec.invert_for_loss) if spec is not None else False
            loss = self._sorensen_dice_loss(
                pred,
                gt,
                mask,
                invert=invert,
                label_smoothing=self._label_smoothing,
            )
            task_losses[target_name] = loss.detach().cpu().item()
            total_loss = loss if total_loss is None else total_loss + loss

        if other_targets:
            other_outputs = {name: outputs[name] for name in other_targets if name in outputs}
            missing_preds = [
                name for name in other_targets
                if name not in other_outputs and not name.endswith("_skel")
            ]
            if missing_preds:
                missing_str = ", ".join(sorted(missing_preds))
                raise KeyError(
                    f"Missing model outputs for targets: {missing_str}. "
                    "Ensure the network produces predictions for all configured non-affinity targets."
                )
            other_targets_dict = {name: filtered_targets[name] for name in other_targets}
            other_loss_fns = {name: loss_fns[name] for name in other_targets if name in loss_fns}
            fallback_result = fallback_fn(
                other_outputs,
                other_targets_dict,
                other_loss_fns,
            )
            if isinstance(fallback_result, tuple):
                base_total, base_losses = fallback_result
            else:
                base_total, base_losses = None, fallback_result

            if base_total is not None:
                total_loss = base_total if total_loss is None else total_loss + base_total
            task_losses.update(base_losses)

        if total_loss is None:
            # No targets were present; create a zero tensor with gradient support
            sample_tensor = next(iter(outputs.values()))
            total_loss = torch.zeros(
                (),
                dtype=sample_tensor.dtype,
                device=sample_tensor.device,
                requires_grad=True,
            )

        return total_loss, task_losses

    @staticmethod
    def _sorensen_dice_loss(
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        invert: bool,
        label_smoothing: float,
    ) -> torch.Tensor:
        """
        Compute a Sørensen-Dice style loss with optional complement handling.

        Parameters
        ----------
        prediction : torch.Tensor
            Model logits or probabilities shaped [B, C, ...].
        ground_truth : torch.Tensor
            Ground-truth affinities in [0, 1] with the same shape as prediction.
        mask : torch.Tensor, optional
            Boolean/float mask marking valid edges. Must be broadcastable to prediction.
        invert : bool
            If True, operates on (1 - prediction) and (1 - ground_truth) to match
            the attractive-edge formulation in Eq. (33) of the paper.
        label_smoothing : float
            Amount of binary label smoothing to apply to ground truth before computing Dice.
        """
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if isinstance(ground_truth, (list, tuple)):
            ground_truth = ground_truth[0]

        prob = torch.sigmoid(prediction)
        if not torch.is_tensor(ground_truth):
            target = torch.as_tensor(ground_truth, dtype=prob.dtype, device=prob.device)
        else:
            target = ground_truth.to(prob.dtype)

        if label_smoothing:
            smooth = float(label_smoothing)
            # Binary label smoothing: move targets away from {0, 1}.
            target = target * (1.0 - smooth) + 0.5 * smooth

        if invert:
            prob = 1.0 - prob
            target = 1.0 - target

        if mask is not None:
            if isinstance(mask, (list, tuple)):
                mask = mask[0]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=prob.dtype, device=prob.device)
            else:
                mask = mask.to(prob.dtype)
            prob = prob * mask
            target = target * mask

        dims = MutexAffinityTrainer._sum_dims(prob.ndim)
        numerator = (prob * target).sum(dim=dims)
        denominator = (prob.pow(2) + target.pow(2)).sum(dim=dims)

        eps = torch.finfo(prob.dtype).tiny
        dice = numerator / (denominator + eps)
        loss = -dice.mean()
        return loss

    @staticmethod
    def _sum_dims(ndim: int) -> Tuple[int, ...]:
        if ndim < 2:
            return (0,)
        # Sum over batch and spatial axes, keep channel dimension
        dims = [0] + list(range(2, ndim))
        return tuple(dims)
