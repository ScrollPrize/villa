"""
Wrapper classes to integrate Primus transformer architecture with NetworkFromConfig.

This module provides adapter classes that make the Primus architecture compatible
with the existing NetworkFromConfig system that expects encoder-decoder separation
and task-specific decoders.
"""

from typing import Tuple, Dict, Optional, List
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import RotaryEmbeddingCat

from .transformers.primus import _PRIMUS_CONFIGS
from .transformers.eva import Eva
from .transformers.patch_encode_decode import PatchEmbed, PatchDecode, LayerNormNd


class PrimusEncoder(nn.Module):
    """
    Encoder wrapper for Primus that extracts features using PatchEmbed + EVA.
    Compatible with NetworkFromConfig's encoder interface.
    """
    
    def __init__(
        self,
        input_channels: int,
        config_name: str,  # 'S', 'B', 'M', or 'L'
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        drop_path_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        pos_emb_type: str = "rope",
        mlp_ratio=4 * 2 / 3,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        super().__init__()
        
        # Get configuration
        conf = _PRIMUS_CONFIGS[config_name]
        self.embed_dim = conf["embed_dim"]
        self.eva_depth = conf["eva_depth"]
        self.eva_numheads = conf["eva_numheads"]
        
        self.patch_embed_size = patch_embed_size
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.num_register_tokens = num_register_tokens
        
        # Check input shape compatibility
        assert len(input_shape) in (2, 3), "Only 2D and 3D inputs are supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)]), \
            f"Input shape {input_shape} must be divisible by patch_embed_size {patch_embed_size}"
        self.ndim = len(input_shape)

        # Calculate patch grid dimensions
        self.patch_grid = tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)])
        
        # Initialize patch embedding
        self.patch_embed = PatchEmbed(patch_embed_size, input_channels, self.embed_dim)
        
        # Initialize EVA transformer
        self.eva = Eva(
            embed_dim=self.embed_dim,
            depth=self.eva_depth,
            num_heads=self.eva_numheads,
            ref_feat_shape=self.patch_grid,
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            pos_emb_type=pos_emb_type,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
        
        # Initialize mask token for restoration as learnable parameter
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)
        
        # Initialize register tokens if needed
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, self.embed_dim))
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None
        
        # Store output info for decoder compatibility
        self.output_channels = [self.embed_dim]  # Single output channel for transformer
        self.strides = [patch_embed_size]  # Stride is the patch embedding size
        
        # These are for compatibility with existing decoder interface
        # Set dimension-aware ops
        if self.ndim == 2:
            self.conv_op = nn.Conv2d
            self.norm_op = nn.InstanceNorm2d
            self.kernel_sizes = [[3, 3]]
        else:
            self.conv_op = nn.Conv3d
            self.norm_op = nn.InstanceNorm3d
            self.kernel_sizes = [[3, 3, 3]]
        self.conv_bias = True
        self.norm_op_kwargs = {"affine": True, "eps": 1e-5}
        self.dropout_op = None
        self.dropout_op_kwargs = None
        self.nonlin = nn.GELU
        self.nonlin_kwargs = {}
        
    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        if keep_indices is None:
            return x, None

        B, num_kept, C = x.shape
        device = x.device

        # Initialize with mask tokens expanded to full sequence
        restored = self.mask_token.expand(B, num_patches, -1).clone()
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        # keep_indices shape: (B, num_kept)
        batch_indices = torch.arange(B, device=device)[:, None].expand(-1, num_kept)

        # Assign kept patches to their positions
        restored[batch_indices, keep_indices] = x
        restored_mask[batch_indices, keep_indices] = True

        return restored, restored_mask
    
    def forward(self, x, ret_mask=False):
        """
        Forward pass through the Primus encoder.
        Returns a list with a single feature map to maintain compatibility.
        If ret_mask is True, also returns the full resolution mask for MAE training.

        Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) inputs.
        """
        # Store original shape for mask expansion
        full_spatial = x.shape[2:]  # Full resolution spatial dims

        # Apply patch embedding
        x = self.patch_embed(x)
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        num_patches = int(torch.tensor(spatial_shape).prod().item())

        # Rearrange to sequence format
        if self.ndim == 2:
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            x = rearrange(x, "b c d h w -> b (d h w) c")

        # Add register tokens if present
        if self.register_tokens is not None:
            x = torch.cat(
                (self.register_tokens.expand(x.shape[0], -1, -1), x),
                dim=1,
            )

        # Pass through EVA transformer
        x, keep_indices = self.eva(x)

        # Remove register tokens if they were added
        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1]:]

        # Restore full sequence with mask tokens
        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)

        # Reshape back to spatial format
        if self.ndim == 2:
            H, W = spatial_shape
            x = rearrange(restored_x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
            # Prepare full resolution mask if requested
            if ret_mask and restoration_mask is not None:
                mask = rearrange(restoration_mask, "b (h w) -> b h w", h=H, w=W)
                full_mask = (
                    mask.repeat_interleave(full_spatial[0] // H, dim=1)
                    .repeat_interleave(full_spatial[1] // W, dim=2)
                )
                full_mask = full_mask[:, None, ...]  # [B, 1, H, W]
            else:
                full_mask = None
        else:
            D, H, W = spatial_shape
            x = rearrange(restored_x, "b (d h w) c -> b c d h w", d=D, h=H, w=W).contiguous()
            # Prepare full resolution mask if requested
            if ret_mask and restoration_mask is not None:
                mask = rearrange(restoration_mask, "b (d h w) -> b d h w", d=D, h=H, w=W)
                full_mask = (
                    mask.repeat_interleave(full_spatial[0] // D, dim=1)
                    .repeat_interleave(full_spatial[1] // H, dim=2)
                    .repeat_interleave(full_spatial[2] // W, dim=3)
                )
                full_mask = full_mask[:, None, ...]  # [B, 1, D, H, W]
            else:
                full_mask = None

        # Return as a list to maintain compatibility with skip connection interface
        if ret_mask:
            return [x], full_mask
        return [x]


class PrimusDecoder(nn.Module):
    """
    Decoder wrapper for Primus using PatchDecode for upsampling.
    Matches official MIC-DKFZ implementation (no transformer decoder layers).
    Compatible with NetworkFromConfig's decoder interface.
    """

    def __init__(
        self,
        encoder: PrimusEncoder,
        num_classes: int,
        norm="LayerNormNd",
        activation="GELU",
        **_unused,
    ):
        super().__init__()


        # Store encoder reference without registering as submodule to avoid
        # duplicate state_dict keys when decoder is used with separate_decoders=True
        object.__setattr__(self, '_encoder_ref', encoder)
        self.num_classes = num_classes

        # Parse normalization layer
        norm_layer = self._get_norm_layer(norm)

        # Parse activation function
        act_fn = self._get_activation(activation)

        # Initialize PatchDecode for upsampling
        self.patch_decoder = PatchDecode(
            patch_size=encoder.patch_embed_size,
            embed_dim=encoder.embed_dim,
            out_channels=num_classes,
            norm=norm_layer,
            activation=act_fn,
        )
    
    def _get_norm_layer(self, norm):
        """Get normalization layer from string or class."""
        if not isinstance(norm, str):
            return norm
            
        norm_lower = norm.lower()
        ndim = getattr(self.encoder, "ndim", 3)
        if "layernorm" in norm_lower:
            return LayerNormNd
        elif "batchnorm" in norm_lower:
            if "2d" in norm_lower or ndim == 2:
                return nn.BatchNorm2d
            return nn.BatchNorm3d
        elif "instancenorm" in norm_lower:
            if "2d" in norm_lower or ndim == 2:
                return nn.InstanceNorm2d
            return nn.InstanceNorm3d
        elif "groupnorm" in norm_lower:
            def _group_norm(num_channels):
                num_groups = 8
                if num_channels % num_groups != 0:
                    num_groups = 1
                return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            return _group_norm
        else:
            # Default to LayerNormNd
            print(f"Unknown norm type '{norm}', defaulting to LayerNormNd")
            return LayerNormNd
    
    def _get_activation(self, activation):
        """Get activation function from string or class."""
        if not isinstance(activation, str):
            return activation
            
        act_lower = activation.lower()
        if "gelu" in act_lower:
            return nn.GELU
        elif "relu" in act_lower and "leaky" not in act_lower:
            return nn.ReLU
        elif "leakyrelu" in act_lower or "leaky_relu" in act_lower:
            return nn.LeakyReLU
        elif "silu" in act_lower or "swish" in act_lower:
            return nn.SiLU
        elif "tanh" in act_lower:
            return nn.Tanh
        elif "elu" in act_lower:
            return nn.ELU
        elif "selu" in act_lower:
            return nn.SELU
        elif "prelu" in act_lower:
            return nn.PReLU
        else:
            # Default to GELU
            print(f"Unknown activation type '{activation}', defaulting to GELU")
            return nn.GELU

    @property
    def encoder(self):
        """Access the encoder reference (not a submodule to avoid duplicate state_dict keys)."""
        return self._encoder_ref

    def forward(self, features: List[torch.Tensor]):
        """
        Forward pass through the Primus decoder.

        Args:
            features: List of feature maps from encoder (expects single element for Primus)

        Returns:
            Upsampled segmentation map
        """
        # Primus encoder returns a single feature map
        x = features[0] if isinstance(features, list) else features

        # Apply patch decoder for upsampling
        output = self.patch_decoder(x)

        return output


class PrimusNetwork(nn.Module):
    """
    Complete Primus network with task-specific decoders.
    This class integrates with NetworkFromConfig for multi-task learning.
    """
    
    def __init__(
        self,
        input_channels: int,
        config_name: str,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        targets: Dict[str, Dict],
        **kwargs
    ):
        super().__init__()
        
        # Extract decoder parameters from kwargs
        self.decoder_depth = kwargs.pop('decoder_depth', 2)
        self.decoder_num_heads = kwargs.pop('decoder_num_heads', 12)
        
        # Initialize shared encoder
        self.shared_encoder = PrimusEncoder(
            input_channels=input_channels,
            config_name=config_name,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            **kwargs
        )
        
        # Initialize task-specific decoders
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        
        for target_name, target_info in targets.items():
            # Get output channels for this task
            out_channels = target_info.get('out_channels', target_info.get('channels', input_channels))
            
            # Create decoder for this task
            self.task_decoders[target_name] = PrimusDecoder(
                encoder=self.shared_encoder,
                num_classes=out_channels,
                decoder_depth=self.decoder_depth,
                decoder_num_heads=self.decoder_num_heads,
            )
            
            # Set up activation if specified
            activation_str = target_info.get("activation", "none")
            if activation_str.lower() == "sigmoid":
                self.task_activations[target_name] = nn.Sigmoid()
            elif activation_str.lower() == "softmax":
                self.task_activations[target_name] = nn.Softmax(dim=1)
            else:
                self.task_activations[target_name] = None
                
            print(f"Primus task '{target_name}' configured with {out_channels} output channels")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Returns:
            Dictionary of task outputs
        """
        # Get features from encoder
        features = self.shared_encoder(x)
        
        # Process through each task decoder
        results = {}
        for task_name, decoder in self.task_decoders.items():
            logits = decoder(features)
            
            # Apply activation if not training and activation is specified
            activation_fn = self.task_activations[task_name]
            if activation_fn is not None and not self.training:
                logits = activation_fn(logits)
                
            results[task_name] = logits
            
        return results
