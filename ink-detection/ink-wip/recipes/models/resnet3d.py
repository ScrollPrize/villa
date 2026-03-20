from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ink.core.types import DataBundle


_RESNET3D_ENCODER_DIMS = {
    50: [256, 512, 1024, 2048],
    101: [256, 512, 1024, 2048],
    152: [256, 512, 1024, 2048],
}


def _default_backbone_pretrained_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "weights" / "r3d50_KM_200ep.pth")


def _pick_group_norm_groups(num_channels: int, desired_groups: int) -> int:
    num_channels = int(num_channels)
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels!r}")
    desired_groups = int(desired_groups)
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

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                    _norm2d(encoder_dims[i - 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )
        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            feature_maps[i - 1] = self.convs[i - 1](f)

        return self.up(self.logit(feature_maps[0]))


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
        forward_features=False,
    ):
        super().__init__()
        self.forward_features = forward_features
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                raise NotImplementedError("shortcut_type='A' is not supported by the standalone ResNet3D port")
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.forward_features:
            return [x1, x2, x3, x4]

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def generate_model(model_depth, **kwargs):
    if model_depth == 10:
        return ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    if model_depth == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    if model_depth == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    if model_depth == 50:
        return ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    if model_depth == 101:
        return ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    if model_depth == 152:
        return ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    if model_depth == 200:
        return ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    raise ValueError(f"Unsupported model_depth={model_depth!r}")


def _resolve_resnet3d_depth(depth):
    depth = int(depth)
    if depth not in _RESNET3D_ENCODER_DIMS:
        raise ValueError(
            f"Unsupported resnet3d_model_depth={depth!r}. "
            f"Expected one of {sorted(_RESNET3D_ENCODER_DIMS)!r}."
        )
    return depth


def _build_backbone_with_optional_pretrained(*, depth, norm, group_norm_groups, pretrained, backbone_pretrained_path):
    backbone = generate_model(
        model_depth=depth,
        n_input_channels=1,
        forward_features=True,
        n_classes=1039,
    )

    if bool(pretrained):
        pretrained_path = Path(str(backbone_pretrained_path))
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Missing backbone pretrained weights: {pretrained_path}. "
                "Set model.backbone_pretrained_path to a valid file."
            )
        backbone_ckpt = torch.load(pretrained_path, map_location="cpu")
        state_dict = backbone_ckpt.get("state_dict", backbone_ckpt)
        conv1_weight = state_dict["conv1.weight"]
        state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
        backbone.load_state_dict(state_dict, strict=False)

    if str(norm).lower() == "group":
        replace_batchnorm_with_groupnorm(backbone, desired_groups=int(group_norm_groups))
    return backbone


class ResNet3DSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        depth=50,
        norm="batch",
        group_norm_groups=32,
        pretrained=True,
        backbone_pretrained_path=None,
    ):
        super().__init__()
        depth = _resolve_resnet3d_depth(depth)
        self.backbone = _build_backbone_with_optional_pretrained(
            depth=depth,
            norm=norm,
            group_norm_groups=group_norm_groups,
            pretrained=pretrained,
            backbone_pretrained_path=backbone_pretrained_path or _default_backbone_pretrained_path(),
        )
        self.decoder = Decoder(
            encoder_dims=_RESNET3D_ENCODER_DIMS[depth],
            upscale=1,
            norm=norm,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        feature_maps = self.backbone(x)
        pooled_feature_maps = [torch.max(feature_map, dim=2)[0] for feature_map in feature_maps]
        return self.decoder(pooled_feature_maps)


@dataclass(frozen=True)
class ResNet3D:
    depth: int = 50
    norm: str = "batch"
    group_norm_groups: int = 32
    pretrained: bool = True
    backbone_pretrained_path: str = field(default_factory=_default_backbone_pretrained_path)

    def build(self, *, data: DataBundle, runtime=None, augment=None):
        del data, runtime, augment
        return ResNet3DSegmentationModel(
            depth=self.depth,
            norm=self.norm,
            group_norm_groups=self.group_norm_groups,
            pretrained=self.pretrained,
            backbone_pretrained_path=self.backbone_pretrained_path,
        )
