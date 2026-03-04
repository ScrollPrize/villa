import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_group_norm_groups(num_channels: int, desired_groups: int) -> int:
    num_channels = int(num_channels)
    desired_groups = int(desired_groups)
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    desired_groups = max(1, min(desired_groups, num_channels))
    for groups in range(desired_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def replace_batchnorm_with_groupnorm(module: nn.Module, *, desired_groups: int = 32) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = int(child.num_features)
            groups = _pick_group_norm_groups(num_channels, desired_groups)
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
            if getattr(child, "affine", False):
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_batchnorm_with_groupnorm(child, desired_groups=desired_groups)
    return module


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale, *, norm="batch", group_norm_groups=32):
        super().__init__()
        norm = str(norm).lower()
        if norm not in {"batch", "group"}:
            raise ValueError(f"Unknown norm: {norm!r}")

        def _norm2d(num_channels: int) -> nn.Module:
            if norm == "group":
                groups = _pick_group_norm_groups(num_channels, int(group_norm_groups))
                return nn.GroupNorm(num_groups=groups, num_channels=int(num_channels))
            return nn.BatchNorm2d(int(num_channels))

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                _norm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True),
            ) for i in range(1, len(encoder_dims))
        ])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


__all__ = [
    "_pick_group_norm_groups",
    "replace_batchnorm_with_groupnorm",
    "Decoder",
]
