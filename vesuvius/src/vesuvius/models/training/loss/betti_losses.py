"""
Betti Matching losses for topologically accurate segmentation.
from https://github.com/nstucki/Betti-matching
Uses the C++ implementation of Betti matching with Python bindings found here https://github.com/nstucki/Betti-Matching-3D

To install and use this loss, make sure you run the build_betti.py script in vesuvius/utils
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
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
            f"Please run the build_betti.py script in vesuvius/utils/ "
            f"This will automatically clone and build Betti-Matching-3D."
            f"You may need to force the system lbstdc++ with the following "
            f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        )
    
except ImportError as e:
    raise ImportError(
        f"Could not import betti_matching module. "
        f"Please run the build_betti.py script in vesuvius/utils/  "
        f"This will clone and build Betti-Matching-3D automatically. "
        f"You may need to force the system lbstdc++ with the following "
        f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        f"Error: {e}"
    )


class BettiMatchingLoss(nn.Module):
    """
    Betti matching loss for topological accuracy. See https://github.com/nstucki/Betti-matching for details

    filtration is implemented via preprocessing (inverting values for superlevel filtration).
    
    Parameters:
    -----------
    relative : bool, default=False
        If True, uses relative Betti matching (NOT IMPLEMENTED)
    filtration : str, default='superlevel'
        Type of filtration: 'superlevel', 'sublevel', or 'bothlevel'
    """
    
    def __init__(self,
                 filtration='superlevel'
):
        super().__init__()
        self.filtration = filtration
        
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

        if num_channels == 2:
            input_probs = torch.softmax(input, dim=1)
            input_fg = input_probs[:, 1:2]  # Extract foreground channel, keep dims

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

        is_3d = len(input.shape) == 5

        # we use pooling here to downsample the target and gt. betti matching is very expensive,
        # and in some (admittedly not super robust) testing, we've found that downsampling 2x preserves
        # most of the features found / matched
        # TODO : test if avgpool / meanpool / other work better, in prelim tests maxpool preserved thin structures best
        if is_3d:
            input_ds = F.max_pool3d(input_fg, kernel_size=2, stride=2)
            target_ds = F.max_pool3d(target_fg, kernel_size=2, stride=2)
        else:
            input_ds = F.max_pool2d(input_fg, kernel_size=2, stride=2)
            target_ds = F.max_pool2d(target_fg, kernel_size=2, stride=2)
        
        total_loss = 0.0

        for i in range(batch_size):
            pred_i = input_ds[i].squeeze(0)  # Remove channel dimension
            target_i = target_ds[i].squeeze(0)

            pred_np = pred_i.detach().cpu().numpy().astype(np.float32)
            target_np = target_i.detach().cpu().numpy().astype(np.float32)

            target_np = (target_np > 0.5).astype(np.float32)

            # For superlevel filtration, we need to invert the values
            if self.filtration == 'superlevel' or self.filtration == 'bothlevel':
                pred_super = 1.0 - pred_np
                target_super = 1.0 - target_np
            else:
                pred_super = pred_np
                target_super = target_np
                
            if self.filtration == 'bothlevel':
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

                num_unmatched1_super = result_super.num_unmatched_input1.sum() if result_super.num_unmatched_input1 is not None else 0
                num_unmatched2_super = result_super.num_unmatched_input2.sum() if result_super.num_unmatched_input2 is not None else 0
                num_unmatched1_sub = result_sub.num_unmatched_input1.sum() if result_sub.num_unmatched_input1 is not None else 0
                num_unmatched2_sub = result_sub.num_unmatched_input2.sum() if result_sub.num_unmatched_input2 is not None else 0
                
                loss_i = (num_unmatched1_super + num_unmatched2_super + 
                         num_unmatched1_sub + num_unmatched2_sub) / 2.0
            else:
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

        return torch.tensor(total_loss / batch_size, 
                          device=input.device, 
                          dtype=torch.float32,
                          requires_grad=True)
