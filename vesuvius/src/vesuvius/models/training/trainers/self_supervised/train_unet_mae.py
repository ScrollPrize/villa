# reimplementation of https://github.com/MIC-DKFZ/nnssl/blob/openneuro/src/nnssl/training/nnsslTrainer/masked_image_modeling/BaseMAETrainer.py

import torch
from torch import nn
from contextlib import nullcontext

from vesuvius.models.training.train import BaseTrainer


class MAEMSELoss(nn.Module):
    """MSE loss for Masked Autoencoder that only computes loss on masked regions.

    Expects mask with 1 for visible (unmasked) and 0 for masked regions.
    Loss is averaged over masked locations only.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == model_output.dim() - 1:
            # mask: [B, 1, ...] -> expand to [B, C, ...]
            mask = mask.expand(model_output.shape[:2] + mask.shape[2:])

        reconstruction_loss = (model_output - target) ** 2
        masked_loss = reconstruction_loss * (1 - mask)

        denom = torch.sum(1 - mask)
        denom = torch.clamp(denom, min=1.0)
        loss = torch.sum(masked_loss) / denom
        return loss


def _create_blocky_mask(spatial_size, block_size: int, sparsity: float, device: torch.device) -> torch.Tensor:
    """Create a low-res binary mask then upsample by repeating blocks.

    - spatial_size: tuple of length 2 or 3 (H,W) or (D,H,W)
    - block_size: integer block edge length in voxels/pixels
    - sparsity: fraction of blocks set to 0 (masked)
    Returns mask shaped [1, *spatial_size] with 1=visible, 0=masked
    """
    if len(spatial_size) not in (2, 3):
        raise ValueError(f"Expected 2D or 3D spatial_size, got {spatial_size}")

    # Compute small grid size
    small = tuple(max(1, s // block_size) for s in spatial_size)
    flat = torch.ones(int(torch.tensor(small).prod().item()), device=device)
    n_masked = int(sparsity * flat.shape[0])
    if n_masked > 0:
        perm = torch.randperm(flat.shape[0], device=device)
        flat[perm[:n_masked]] = 0.0
    small_mask = flat.view(*small)

    # Repeat to full resolution
    if len(spatial_size) == 2:
        H, W = spatial_size
        rep_h = max(1, H // small[0])
        rep_w = max(1, W // small[1])
        mask = small_mask.repeat_interleave(rep_h, dim=0).repeat_interleave(rep_w, dim=1)
        mask = mask[:H, :W]
        mask = mask.unsqueeze(0)  # [1, H, W]
    else:
        D, H, W = spatial_size
        rep_d = max(1, D // small[0])
        rep_h = max(1, H // small[1])
        rep_w = max(1, W // small[2])
        mask = (
            small_mask.repeat_interleave(rep_d, dim=0)
            .repeat_interleave(rep_h, dim=1)
            .repeat_interleave(rep_w, dim=2)
        )
        mask = mask[:D, :H, :W]
        mask = mask.unsqueeze(0)  # [1, D, H, W]

    return mask


class TrainUNetMAE(BaseTrainer):

    def __init__(self, mgr=None, verbose: bool = True):
        # default masking parameters
        self.mask_ratio = 0.75
        self.block_size = 16  # in voxels/pixels
        self.grad_clip = 12.0

        if mgr is not None:
            self.mask_ratio = float(getattr(mgr, 'mask_ratio', self.mask_ratio))
            self.block_size = int(getattr(mgr, 'mae_block_size', self.block_size))

            mgr.allow_unlabeled_data = True
            mgr.only_spatial_and_intensity = True
            mgr.enable_deep_supervision = False # deep supervision would be problematic / overly complex for this task

            # Set MAE-specific optimization defaults, but only if user did NOT specify them in config/CLI
            trc = getattr(mgr, 'tr_configs', {}) if hasattr(mgr, 'tr_configs') else {}
            if 'optimizer' not in trc:
                mgr.optimizer = 'SGD'
            if 'initial_lr' not in trc:
                mgr.initial_lr = 1e-2
            if 'weight_decay' not in trc:
                mgr.weight_decay = 3e-5
            if 'scheduler' not in trc:
                mgr.scheduler = 'poly'
            # Make Poly exponent explicit if scheduler kwargs not provided at all
            if 'scheduler_kwargs' not in trc and not hasattr(mgr, 'scheduler_kwargs'):
                mgr.scheduler_kwargs = {'exponent': 0.9}

            in_ch = getattr(mgr, 'in_channels', 1)
            if not hasattr(mgr, 'targets') or mgr.targets is None:
                mgr.targets = {}
            mgr.targets['mae'] = {
                'out_channels': in_ch,
                'activation': 'none',
                'weight': 1.0,
                'task_type': 'regression',
            }

        super().__init__(mgr, verbose)

        self._is_mae_training = True  # we keep model in train mode during eval so we still get masking
        self._mae_loss = MAEMSELoss()
        self._current_mask = None

    # --- losses --- #
    def _build_loss(self):
        return {'mae': [(self._mae_loss, 1.0)]}

    # --- metrics (disable classification metrics for MAE) --- #
    def _initialize_evaluation_metrics(self):
        return {}

    # --- forward helpers with masking --- #
    def _make_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create a blocky binary mask [B,1,*] with 1=visible, 0=masked."""
        b, _, *spatial = inputs.shape
        device = inputs.device
        mask_1 = _create_blocky_mask(tuple(spatial), self.block_size, self.mask_ratio, device)
        mask = mask_1.unsqueeze(0).expand(b, -1, *spatial)  # [B,1,*]
        return mask

    def _get_model_outputs(self, model, data_dict):
        """Apply mask to inputs, run model, and build MAE targets/outputs."""
        inputs = data_dict["image"].to(self.device)

        mask = self._make_mask(inputs).to(inputs.dtype)
        masked_inputs = inputs * mask
        outputs = model(masked_inputs)

        if 'mae' not in outputs:
            raise RuntimeError("Model does not have a 'mae' head. Ensure mgr.targets includes 'mae'.")

        out_mae = outputs['mae']
        if isinstance(out_mae, (list, tuple)) and len(out_mae) > 0:
            out_mae = out_mae[0]
        outputs['mae'] = out_mae

        targets_dict = {'mae': inputs}
        self._current_mask = mask

        return inputs, targets_dict, outputs

    # --- loss computation --- #
    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        out = outputs['mae']
        tgt = targets_dict['mae']
        mask = self._current_mask

        if mask is None:
            raise ValueError('No loss mask detected, mae training requires a loss mask!')

        # Ensure mask can broadcast to [B,C,*]
        if mask.shape[1] == 1 and out.shape[1] > 1:
            mask = mask.expand(out.shape[0], out.shape[1], *mask.shape[2:])

        loss = self._mae_loss(out, tgt, mask)
        task_losses = {'mae': loss.detach().cpu().item()}
        return loss, task_losses

    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        inputs = data_dict["image"].to(self.device)

        ctx = nullcontext()
        if use_amp:
            if self.device.type == 'cuda':
                ctx = torch.amp.autocast('cuda')
            else:
                ctx = torch.amp.autocast(self.device.type)

        with ctx:
            mask = self._make_mask(inputs).to(inputs.dtype)
            masked_inputs = inputs * mask
            outputs = model(masked_inputs)
            out_mae = outputs['mae']
            if isinstance(out_mae, (list, tuple)) and len(out_mae) > 0:
                out_mae = out_mae[0]
            outputs['mae'] = out_mae

            if mask.shape[1] == 1 and out_mae.shape[1] > 1:
                mask_for_loss = mask.expand(out_mae.shape[0], out_mae.shape[1], *mask.shape[2:])
            else:
                mask_for_loss = mask

            loss = self._mae_loss(out_mae, inputs, mask_for_loss)

        task_losses = {'mae': loss.detach().cpu().item()}
        targets_dict = {'mae': inputs}
        return task_losses, inputs, targets_dict, outputs
