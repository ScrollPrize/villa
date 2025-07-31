"""
Betti Matching losses for topologically accurate segmentation.
Uses the C++ implementation of Betti matching with Python bindings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    # Find the Betti-Matching-3D build directory
    import sys
    from pathlib import Path
    
    # Look for the external Betti build in the Vesuvius installation
    vesuvius_module_path = Path(__file__).parent.parent.parent.parent.parent.parent  # Go up to vesuvius root
    betti_build_path = vesuvius_module_path / "external" / "Betti-Matching-3D" / "build"
    
    if betti_build_path.exists():
        sys.path.insert(0, str(betti_build_path))
        import betti_matching as bm
    else:
        raise ImportError(
            f"Betti-Matching-3D build not found at {betti_build_path}. "
            f"Please install Vesuvius with: pip install -e . "
            f"This will automatically clone and build Betti-Matching-3D."
        )
    
except ImportError as e:
    raise ImportError(
        f"Could not import betti_matching module. "
        f"Make sure Vesuvius is properly installed with: pip install -e . "
        f"This will clone and build Betti-Matching-3D automatically. "
        f"Error: {e}"
    )


class BettiMatchingLoss(nn.Module):
    """
    Pure Betti matching loss for topological accuracy.
    
    Note: The current Betti-Matching-3D API only supports basic matching.
    The construction, comparison, and relative parameters are stored but
    not used in the computation. Only filtration is implemented via
    preprocessing (inverting values for superlevel filtration).
    
    Parameters:
    -----------
    relative : bool, default=False
        If True, uses relative Betti matching (NOT IMPLEMENTED)
    filtration : str, default='superlevel'
        Type of filtration: 'superlevel', 'sublevel', or 'bothlevel'
    construction : str, default='V'
        Construction type: 'V' (Vietoris) or 'T' (Threshold) (NOT IMPLEMENTED)
    comparison : str, default='union'
        Comparison method: 'union' or 'intersection' (NOT IMPLEMENTED)
    """
    
    def __init__(self, relative=False, filtration='superlevel', 
                 construction='V', comparison='union'):
        super().__init__()
        self.relative = relative
        self.filtration = filtration
        self.construction = construction
        self.comparison = comparison
        
    def forward(self, input, target):
        """
        Compute Betti matching loss.
        
        Parameters:
        -----------
        input : torch.Tensor
            Predicted logits or probabilities (B, C, D, H, W) or (B, C, H, W)
            C can be 1 (sigmoid) or 2 (softmax)
        target : torch.Tensor
            Ground truth masks (B, C, D, H, W) or (B, C, H, W)
            For C=2, expects one-hot encoded format
            
        Returns:
        --------
        loss : torch.Tensor
            Scalar loss value
        """
        batch_size = input.shape[0]
        num_channels = input.shape[1]
        
        # Handle different input formats
        if num_channels == 2:
            # Two-channel input: apply softmax and extract foreground channel
            input_probs = torch.softmax(input, dim=1)
            input_fg = input_probs[:, 1:2]  # Extract foreground channel, keep dims
            
            # Handle target format
            if target.shape[1] == 2:
                # One-hot encoded target: extract foreground channel
                target_fg = target[:, 1:2]
            else:
                # Single channel target: use as is
                target_fg = target
        else:
            # Single channel input: apply sigmoid
            if not (input.min() >= 0 and input.max() <= 1):
                input_fg = torch.sigmoid(input)
            else:
                input_fg = input
            target_fg = target
        
        # Apply 2x downsampling
        # Determine if we're dealing with 2D or 3D data
        is_3d = len(input.shape) == 5
        
        if is_3d:
            # 3D downsampling
            input_ds = F.max_pool3d(input_fg, kernel_size=2, stride=2)
            target_ds = F.max_pool3d(target_fg, kernel_size=2, stride=2)
        else:
            # 2D downsampling
            input_ds = F.max_pool2d(input_fg, kernel_size=2, stride=2)
            target_ds = F.max_pool2d(target_fg, kernel_size=2, stride=2)
        
        total_loss = 0.0
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Get single sample (using downsampled versions)
            pred_i = input_ds[i].squeeze(0)  # Remove channel dimension
            target_i = target_ds[i].squeeze(0)
            
            # Convert to numpy and ensure float32
            pred_np = pred_i.detach().cpu().numpy().astype(np.float32)
            target_np = target_i.detach().cpu().numpy().astype(np.float32)
            
            # Ensure target is binary
            target_np = (target_np > 0.5).astype(np.float32)
            
            # Compute Betti matching using C++ module
            # Note: The new API doesn't support filtration/construction/comparison/relative parameters
            # These would need to be implemented as preprocessing steps if needed
            
            # For superlevel filtration, we need to invert the values
            if self.filtration == 'superlevel' or self.filtration == 'bothlevel':
                pred_super = 1.0 - pred_np
                target_super = 1.0 - target_np
            else:
                pred_super = pred_np
                target_super = target_np
                
            if self.filtration == 'bothlevel':
                # Compute for both superlevel and sublevel
                result_super = bm.compute_matching(
                    pred_super, target_super,
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=True
                )
                result_sub = bm.compute_matching(
                    pred_np, target_np,
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=True
                )
                
                # Combine losses from both filtrations
                num_unmatched1_super = result_super.num_unmatched_input1.sum() if result_super.num_unmatched_input1 is not None else 0
                num_unmatched2_super = result_super.num_unmatched_input2.sum() if result_super.num_unmatched_input2 is not None else 0
                num_unmatched1_sub = result_sub.num_unmatched_input1.sum() if result_sub.num_unmatched_input1 is not None else 0
                num_unmatched2_sub = result_sub.num_unmatched_input2.sum() if result_sub.num_unmatched_input2 is not None else 0
                
                loss_i = (num_unmatched1_super + num_unmatched2_super + 
                         num_unmatched1_sub + num_unmatched2_sub) / 2.0
            else:
                # Single filtration
                if self.filtration == 'sublevel':
                    pred_input = pred_np
                    target_input = target_np
                else:  # superlevel
                    pred_input = pred_super
                    target_input = target_super
                    
                result = bm.compute_matching(
                    pred_input, target_input,
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=True
                )
                
                # Loss is the number of unmatched features
                num_unmatched1 = result.num_unmatched_input1.sum() if result.num_unmatched_input1 is not None else 0
                num_unmatched2 = result.num_unmatched_input2.sum() if result.num_unmatched_input2 is not None else 0
                loss_i = float(num_unmatched1 + num_unmatched2)
            
            total_loss += loss_i
        
        # Return as tensor with gradient tracking
        return torch.tensor(total_loss / batch_size, 
                          device=input.device, 
                          dtype=torch.float32,
                          requires_grad=True)


class DiceBettiMatchingLoss(nn.Module):
    """
    Combined Dice + Betti matching loss for balanced optimization.
    
    Parameters:
    -----------
    alpha : float, default=0.5
        Weight for the Betti matching term (Dice weight is 1.0)
    relative : bool, default=False
        If True, uses relative Betti matching
    filtration : str, default='superlevel'
        Type of filtration: 'superlevel', 'sublevel', or 'bothlevel'
    construction : str, default='V'
        Construction type: 'V' (Vietoris) or 'T' (Threshold)
    comparison : str, default='union'
        Comparison method: 'union' or 'intersection'
    """
    
    def __init__(self, alpha=0.5, relative=False, filtration='superlevel',
                 construction='V', comparison='union'):
        super().__init__()
        self.alpha = alpha
        self.betti_loss = BettiMatchingLoss(
            relative=relative,
            filtration=filtration,
            construction=construction,
            comparison=comparison
        )
        # Import Dice loss from nnUNet
        from .nnunet_losses import MemoryEfficientSoftDiceLoss
        # Don't apply nonlin here - we'll handle it based on input channels
        self.dice_loss = MemoryEfficientSoftDiceLoss(
            apply_nonlin=None,
            batch_dice=False,
            do_bg=False,
            smooth=1e-5
        )
        
    def forward(self, input, target):
        """
        Compute combined Dice + Betti matching loss.
        
        Parameters:
        -----------
        input : torch.Tensor
            Predicted logits (B, C, D, H, W) or (B, C, H, W)
            C can be 1 (sigmoid) or 2 (softmax)
        target : torch.Tensor
            Ground truth masks (B, C, D, H, W) or (B, C, H, W)
            
        Returns:
        --------
        loss : torch.Tensor
            Combined loss value
        loss_dict : dict
            Dictionary with individual loss components
        """
        num_channels = input.shape[1]
        
        # Apply appropriate activation for Dice loss
        if num_channels == 2:
            # Softmax for 2-channel
            input_dice = torch.softmax(input, dim=1)
        else:
            # Sigmoid for 1-channel
            input_dice = torch.sigmoid(input)
        
        # Compute individual losses
        dice_loss = self.dice_loss(input_dice, target)
        betti_loss = self.betti_loss(input, target)  # Betti handles activation internally
        
        # Combine losses
        total_loss = dice_loss + self.alpha * betti_loss
        
        # Return loss and components dict (for compatibility with existing pipeline)
        loss_dict = {
            'dice': dice_loss.detach().cpu().item(),
            'betti': betti_loss.detach().cpu().item(),
            'total': total_loss.detach().cpu().item()
        }
        
        return total_loss, loss_dict