import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedReconstructionLoss(nn.Module):
    """
    Masked-region reconstruction loss for 2-D or 3-D MAE style pre-training.

    Args
    ----
    base_loss         :  'mse' | 'l1' | 'smooth_l1'
    normalize_targets :  normalise pred/target per-sample using *visible* voxels
    reduction         :  'mean' (masked-mean), 'sum', or 'none'
    eps               :  small constant for numerical stability
    variance_threshold:  minimum std deviation required for normalization (default: 0.1)
                        If std < threshold, normalization is skipped to avoid numerical instability
    """

    def __init__(self,
                 base_loss: str = 'mse',
                 normalize_targets: bool = True,
                 reduction: str = 'mean',
                 eps: float = 1e-6,
                 variance_threshold: float = 0.1,
                 use_robust_norm: bool = True,
                 max_loss_value: float = 100.0,
                 log_high_losses: bool = True):
        super().__init__()

        # choose base loss
        if base_loss == 'mse':
            self.base_loss_fn = F.mse_loss
        elif base_loss == 'l1':
            self.base_loss_fn = F.l1_loss
        elif base_loss == 'smooth_l1':
            self.base_loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f'Unknown base_loss: {base_loss}')

        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f'Unknown reduction: {reduction}')

        self.normalize_targets = normalize_targets
        self.reduction        = reduction
        self.eps              = eps
        self.variance_threshold = variance_threshold
        self.use_robust_norm = use_robust_norm
        self.max_loss_value = max_loss_value
        self.log_high_losses = log_high_losses
        self._high_loss_count = 0

    # --------------------------------------------------------------------- #
    # utility                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _check_mask(mask: torch.Tensor, pred: torch.Tensor):
        """
        Ensures mask has shape (B,1,...) and is {0,1}.  Raises otherwise.
        """
        if mask.dim() != pred.dim():
            raise ValueError(f'Mask must have same #dims as pred; '
                             f'got {mask.shape} vs {pred.shape}')

        if mask.shape[0] != pred.shape[0]:
            raise ValueError('Batch dim of mask/pred differ.')

        # force channel==1
        if mask.shape[1] != 1:
            raise ValueError(f'Mask channel dim must be 1; got {mask.shape}')

        if not mask.dtype.is_floating_point:
            mask = mask.float()

        # clamp to {0,1}
        mask = mask.clamp(0, 1)
        return mask

    # --------------------------------------------------------------------- #
    # forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self,
                pred : torch.Tensor,
                target: torch.Tensor,
                mask : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B,C,H,W) or (B,C,D,H,W)
        target : same shape as pred
        mask   : (B,1,H,W) or (B,1,D,H,W) – 1 = masked, 0 = visible
        """
        if pred.shape != target.shape:
            raise ValueError(f'pred and target shapes differ: '
                             f'{pred.shape} vs {target.shape}')

        mask = self._check_mask(mask, pred)  # (B,1, ...)

        # ------------------------------------------------------------------ #
        # normalisation (optional)                                           #
        # ------------------------------------------------------------------ #
        if self.normalize_targets:
            visible = 1.0 - mask                 # (B,1, ...)
            dims    = tuple(range(2, pred.dim()))  # spatial dims  reduce over

            # count visible voxels, clamp to avoid /0
            num_vis = visible.sum(dim=dims, keepdim=True).clamp_min_(1.0)

            if self.use_robust_norm:
                # Use robust statistics (median/MAD) for normalization
                # This is more resistant to outliers
                batch_size = target.shape[0]
                
                for b in range(batch_size):
                    # Extract visible mask for this sample
                    # visible has shape (B, 1, ...) so we need to handle the channel dimension
                    visible_mask_b = visible[b, 0] > 0  # Remove channel dimension
                    
                    if visible_mask_b.sum() > 0:
                        # Extract visible values from target and pred
                        # target[b] has shape (C, D, H, W) or (C, H, W)
                        # We need to apply mask across all channels
                        target_b = target[b]  # Shape: (C, ...)
                        pred_b = pred[b]      # Shape: (C, ...)
                        
                        # Flatten spatial dimensions for easier indexing
                        target_b_flat = target_b.view(target_b.shape[0], -1)  # (C, N)
                        visible_mask_flat = visible_mask_b.view(-1)  # (N,)
                        
                        # Extract visible values across all channels
                        visible_values = target_b_flat[:, visible_mask_flat].flatten()
                        
                        if visible_values.numel() > 0:
                            # Compute median
                            median = visible_values.median()
                            
                            # Compute MAD (Median Absolute Deviation)
                            mad = (visible_values - median).abs().median()
                            
                            # Scale MAD to be comparable to std
                            scaled_mad = 1.4826 * mad
                            
                            # Avoid division by zero
                            if scaled_mad < self.variance_threshold:
                                # Fall back to IQR-based scaling
                                q75 = visible_values.quantile(0.75)
                                q25 = visible_values.quantile(0.25)
                                iqr = q75 - q25
                                scaled_mad = max(iqr / 1.35, self.eps)
                            
                            # Normalize this sample
                            if scaled_mad > self.variance_threshold:
                                target[b] = (target[b] - median) / scaled_mad
                                pred[b] = (pred[b] - median) / scaled_mad
            else:
                # Original mean/std normalization
                mean = (target * visible).sum(dim=dims, keepdim=True) / num_vis
                var  = ((target - mean).pow(2) * visible).sum(dim=dims, keepdim=True) / num_vis
                std  = (var + self.eps).sqrt()

                # Only normalize if std is above threshold to avoid numerical instability
                # Create a mask for samples with sufficient variance
                normalize_mask = (std > self.variance_threshold).float()
                
                # Apply normalization only where std is sufficient
                # For samples with low std, keep original values
                target_normalized = (target - mean) / std
                pred_normalized = (pred - mean) / std
                
                # Use normalized values where std > threshold, original values otherwise
                target = target * (1 - normalize_mask) + target_normalized * normalize_mask
                pred = pred * (1 - normalize_mask) + pred_normalized * normalize_mask

        # ------------------------------------------------------------------ #
        # pixel/voxel-wise base loss                                         #
        # ------------------------------------------------------------------ #
        loss = self.base_loss_fn(pred, target, reduction='none')  # (B,C,...)

        # Check for NaN or Inf in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            if self.log_high_losses:
                print(f"WARNING: NaN or Inf detected in loss computation!")
                print(f"  Pred stats - min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}")
                print(f"  Target stats - min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}")
            # Replace NaN/Inf with max_loss_value
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), 
                             torch.tensor(self.max_loss_value, device=loss.device, dtype=loss.dtype), 
                             loss)

        # Clamp loss values to prevent extreme spikes
        loss_before_clamp = loss.clone()
        loss = torch.clamp(loss, min=0.0, max=self.max_loss_value)
        
        # Log if clamping was needed
        if self.log_high_losses and (loss_before_clamp > self.max_loss_value).any():
            self._high_loss_count += 1
            max_loss_val = loss_before_clamp.max().item()
            if self._high_loss_count <= 10 or self._high_loss_count % 100 == 0:  # Log first 10, then every 100th
                print(f"WARNING: High loss detected (count: {self._high_loss_count})")
                print(f"  Max loss before clamping: {max_loss_val:.4f}")
                print(f"  Loss clamped to: {self.max_loss_value}")
                # Find which samples had high loss
                batch_max_losses = loss_before_clamp.view(loss.shape[0], -1).max(dim=1)[0]
                for b in range(loss.shape[0]):
                    if batch_max_losses[b] > self.max_loss_value:
                        print(f"  Sample {b} had max loss: {batch_max_losses[b].item():.4f}")

        # apply mask  (broadcast channel dim automatically)
        loss = loss * mask

        if self.reduction == 'none':
            return loss

        if self.reduction == 'sum':
            return loss.sum()

        # 'mean' over *masked* voxels only
        num_masked = mask.sum()
        if num_masked > 0:
            mean_loss = loss.sum() / num_masked
            
            # Final check on mean loss
            if self.log_high_losses and mean_loss > 10.0:  # Warn if mean loss is very high
                print(f"WARNING: High mean loss: {mean_loss.item():.4f}")
                
            return mean_loss
        # nothing masked – return 0 with grad
        return pred.new_tensor(0.0, requires_grad=True)
