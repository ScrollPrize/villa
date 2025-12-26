import math
from typing import Tuple, Callable, Optional

import numpy as np
import torch
from timm.layers import (
    trunc_normal_,
    apply_keep_indices_nlc,
    RotaryEmbeddingCat,
    set_fused_attn,
)
from timm.layers.patch_dropout import PatchDropoutWithIndices  # Using this instead of PatchDropout for indices support
from timm.models.eva import EvaBlock
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from .pope import PoPEEmbedding, PoPEBlock

set_fused_attn(True)

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

class Eva(nn.Module):
    """Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    
    Key differences from reference implementation:
      * Uses PatchDropoutWithIndices instead of PatchDropout(return_indices=True)
      * For 2D ROPE embeddings, directly indexes instead of using apply_keep_indices_nlc 
        to avoid batch dimension issues with apply_rot_embed_cat
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qkv_fused: bool = True,  # Fused QKV is faster (~10-15% speedup)
        mlp_ratio: float = 4 * 2 / 3,
        swiglu_mlp: bool = True,
        scale_mlp: bool = True,
        scale_attn_inner: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        drop_path_rate: float = 0.0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        use_abs_pos_emb: bool = True,
        use_rot_pos_emb: bool = True,
        dynamic_img_size: bool = False,
        ref_feat_shape: Optional[Tuple[int, ...]] = None,  # 224/14
        num_reg_tokens: int = 0,
        pos_emb_type: str = "rope",
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        block_fn=EvaBlock,
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        self.num_prefix_tokens = num_reg_tokens

        num_patches = np.prod(ref_feat_shape)

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            # NOTE: Using PatchDropoutWithIndices from timm which properly returns indices
            # The reference uses PatchDropout with return_indices=True but that param doesn't exist in standard timm
            self.patch_drop = PatchDropoutWithIndices(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = None

        pos_emb_type = pos_emb_type.lower()
        if pos_emb_type not in ("rope", "pope"):
            raise ValueError(f"Unsupported pos_emb_type '{pos_emb_type}'. Use 'rope' or 'pope'.")

        if pos_emb_type == "pope" and rope_impl is RotaryEmbeddingCat:
            rope_impl = PoPEEmbedding
        if pos_emb_type == "pope" and block_fn is EvaBlock:
            block_fn = PoPEBlock

        # Determine RoPE configuration based on spatial dimensions
        self.head_dim = embed_dim // num_heads
        num_spatial = len(ref_feat_shape) if ref_feat_shape is not None else 3

        if use_rot_pos_emb:
            if pos_emb_type == "pope":
                rope_dim = self.head_dim
                self.effective_rope_dim = rope_dim
            elif num_spatial == 3:
                # For 3D: rope covers 2/3 of head_dim (split across 3 spatial dimensions)
                rope_dim = round(self.head_dim / 1.5)
                assert rope_dim == self.head_dim / 1.5, "rope dim must be divisible by (num_heads * 1.5)"
                assert rope_dim % 4 == 0, "rope dim must be divisible by 4"
                # Effective dim after RotaryEmbeddingCat: (rope_dim // 4) * 6
                self.effective_rope_dim = (rope_dim // 4) * 6
            else:
                # For 2D: use head_dim but effective will be truncated to multiple of 4
                rope_dim = self.head_dim
                # Effective dim after RotaryEmbeddingCat: (rope_dim // 4) * 4
                self.effective_rope_dim = (rope_dim // 4) * 4

            self.rope = rope_impl(
                rope_dim, in_pixels=False, feat_shape=ref_feat_shape, ref_feat_shape=ref_feat_shape, **rope_kwargs
            )
        else:
            self.rope = None
            self.effective_rope_dim = 0

        # Cache RoPE embeddings for static input shapes (avoids recomputation each forward)
        if use_rot_pos_emb and not dynamic_img_size:
            self._cached_rope = self._pad_rope_embed(self.rope.get_embed())
        else:
            self._cached_rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = block_fn
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {"pos_embed", "cls_token"}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )
        return matcher

    def _pad_rope_embed(self, rope_embed: torch.Tensor) -> torch.Tensor:
        """
        Pad RoPE embedding to match head_dim when effective_rope_dim < head_dim.

        The embedding is [seq_len, effective_dim * 2] (sin and cos concatenated).
        We pad to [seq_len, head_dim * 2] with identity values (sin=0, cos=1).
        """
        effective_dim = rope_embed.shape[-1] // 2
        if effective_dim >= self.head_dim:
            return rope_embed

        seq_len = rope_embed.shape[0]
        pad_size = self.head_dim - effective_dim

        # Split into sin and cos halves
        sin_emb, cos_emb = rope_embed.tensor_split(2, dim=-1)

        # Pad sin with 0s and cos with 1s (identity rotation)
        sin_pad = torch.zeros(seq_len, pad_size, device=rope_embed.device, dtype=rope_embed.dtype)
        cos_pad = torch.ones(seq_len, pad_size, device=rope_embed.device, dtype=rope_embed.dtype)

        sin_emb = torch.cat([sin_emb, sin_pad], dim=-1)
        cos_emb = torch.cat([cos_emb, cos_pad], dim=-1)

        # Concatenate back
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            raise NotImplementedError("dynamic_img_size is not implemented at the moment")
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed_3d(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            # Use cached RoPE if available, otherwise compute and pad
            if self._cached_rope is not None:
                rot_pos_embed = self._cached_rope
            elif self.rope is not None:
                rot_pos_embed = self._pad_rope_embed(self.rope.get_embed())
            else:
                rot_pos_embed = None

        # Ensure RoPE embeddings are on the same device/dtype as input (AMP-safe)
        if rot_pos_embed is not None:
            rot_pos_embed = rot_pos_embed.to(device=x.device, dtype=x.dtype)

        if pos_embed is not None:
            x = x + pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                pos_embed_has_batch = (
                    rot_pos_embed.ndim >= 3 and rot_pos_embed.shape[0] == x.shape[0]
                )
                rot_pos_embed = apply_keep_indices_nlc(
                    x, rot_pos_embed, keep_indices, pos_embed_has_batch=pos_embed_has_batch
                )
            return x, rot_pos_embed, keep_indices
        else:
            return x, rot_pos_embed, None

    def forward_features(self, x):
        x, rot_pos_embed, keep_indices = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed, use_reentrant=False)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x, keep_indices

    def forward(self, x):
        x, keep_indices = self.forward_features(x)
        return x, keep_indices
