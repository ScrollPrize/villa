"""Frozen DINO CNN/transformer with the upstream Vesuvius residual decoder."""
from types import SimpleNamespace

import torch
from torch import nn

from dinovol_2.model.backbone import build_backbone
from dinovol_2.training.checkpoint import backbone_state
from dinovol_2.training.config import as_legacy_config
from .vesuvius import components


def load_encoder(path, branch="teacher"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = as_legacy_config(checkpoint["config"], legacy_checkpoint=True)["model"]
    encoder = build_backbone(config)
    state, actual_branch = backbone_state(checkpoint, preferred_branch=branch)
    if branch != actual_branch:
        raise ValueError(f"Requested {branch} weights, but only found {actual_branch}")
    encoder.load_state_dict(state, strict=True)
    return encoder, {"step": checkpoint.get("step"), "branch": actual_branch, "model": config}


class SurfaceModel(nn.Module):
    def __init__(self, encoder, *, blocks=2, vesuvius_source=None):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        if encoder.embedding_type != "deeper":
            raise ValueError("Surface skips require the deeper CNN embedding")
        cnn = encoder.down_projection
        if tuple(cnn.stage_strides[0]) != (1, 1, 1):
            raise ValueError("The first CNN stage must supply a full-resolution skip")
        encoder.return_embed_skips = True
        encoder.grad_checkpointing = False
        channels = [*cnn.skip_channels, encoder.embed_dim]
        adapter = SimpleNamespace(
            output_channels=channels, strides=cnn.stage_strides,
            kernel_sizes=[(3, 3, 3)] * len(channels), conv_op=nn.Conv3d,
            conv_bias=True, norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        )
        decoder, _ = components(vesuvius_source)
        self.decoder = decoder(adapter, basic_block="BasicBlockD", num_classes=2,
                               n_conv_per_stage=blocks, deep_supervision=False)
        # Upstream creates unused heads for deep-supervision checkpoint compatibility.
        # Keep its state layout, but train only the head actually used by forward().
        for head in self.decoder.seg_layers[:-1]:
            head.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        spatial = tuple(image.shape[2:])
        patch = self.encoder.patch_size
        if any(n % p for n, p in zip(spatial, patch)):
            raise ValueError(f"Crop {spatial} must be divisible by patch size {patch}")
        grid = tuple(n // p for n, p in zip(spatial, patch))
        self.encoder.global_crops_size = spatial
        self.encoder.local_crops_size = spatial
        self.encoder.global_ref_feat_shape = grid
        self.encoder.local_ref_feat_shape = grid
        # no_grad, rather than inference_mode: decoder backward needs to save inputs.
        with torch.no_grad():
            features = self.encoder.forward_features(image)
            bottleneck = features["x_norm_patchtokens"].transpose(1, 2).reshape(
                image.shape[0], self.encoder.embed_dim, *grid)
        return self.decoder([*features["embed_skips"], bottleneck])
