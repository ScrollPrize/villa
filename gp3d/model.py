import pytorch_lightning as pl
from torch.optim import AdamW
import segmentation_models_pytorch as smp

from warmup_scheduler import GradualWarmupScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

def conv3x3x3(in_planes, out_planes, stride):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, downsample):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, downsample):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes, 1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, 1)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VolumetricResNet(nn.Module):
    def __init__(self, block, layers, base_planes=64):
        super().__init__()
        self.inplanes = base_planes
        self.planes = [base_planes, base_planes * 2, base_planes * 4, base_planes * 8]
        self.conv1 = nn.Conv3d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.planes[0], layers[0])
        self.layer2 = self._make_layer(block, self.planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.planes[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = [block(
            in_planes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample
        )]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def create_volumetric_resnet(depth, **kwargs):
    assert depth in [10, 18, 34, 50, 101, 152], f"Unsupported depth: {depth}"
    if depth == 10:
        return VolumetricResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif depth == 18:
        return VolumetricResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif depth == 34:
        return VolumetricResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif depth == 50:
        return VolumetricResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif depth == 101:
        return VolumetricResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif depth == 152:
        return VolumetricResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class Decoder3D(nn.Module):
    def __init__(self, encoder_dims, output_size):
        super().__init__()
        self.output_size = output_size
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    encoder_dims[i] + encoder_dims[i - 1],
                    encoder_dims[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm3d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))
        ])
        self.logit = nn.Conv3d(encoder_dims[0], 1, kernel_size=1)

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode='trilinear', align_corners=False)
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        if x.shape[-3:] != (self.output_size, self.output_size, self.output_size):
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size, self.output_size),
                mode='trilinear',
                align_corners=False
            )
        return x


class InkDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        self.backbone = create_volumetric_resnet(depth=RESNET_DEPTH)

        # Get encoder dimensions by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.rand(1, 1, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)
            encoder_outputs = self.backbone(dummy_input)
            encoder_dims = [x.size(1) for x in encoder_outputs]
            print(f"ResNet{RESNET_DEPTH} encoder dimensions: {encoder_dims}")
            print(f"Processing {CHUNK_SIZE}³ chunks -> {OUTPUT_SIZE}³ outputs")

            # Show spatial dimension flow through network
            print("\nDimension flow through network:")
            print(f"Input: 1x{CHUNK_SIZE}x{CHUNK_SIZE}x{CHUNK_SIZE}")
            print(f"After conv1 (stride=2): {self.backbone.inplanes}x32x32x32")
            for i, feat in enumerate(encoder_outputs):
                print(f"After layer{i + 1}: {feat.shape[1]}x{feat.shape[2]}x{feat.shape[3]}x{feat.shape[4]}")

        self.decoder = Decoder3D(encoder_dims=encoder_dims, output_size=OUTPUT_SIZE)
        self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = self.normalization(x)
        feat_maps = self.backbone(x)
        pred_mask = self.decoder(feat_maps)
        return pred_mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y = F.interpolate(
            y.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE),
            mode='trilinear',
            align_corners=False
        )

        loss = self.loss_func(outputs, y)
        if torch.isnan(loss):
            print("Loss nan encountered")

        self.log_dict({
            "train/loss": loss,
            "train/lr": self.trainer.optimizers[0].param_groups[0]['lr'],
            "train/epoch": self.current_epoch,
            "train/step": self.global_step,
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, xyxy = batch
        outputs = self(x)
        y = F.interpolate(
            y.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZE),
            mode='trilinear',
            align_corners=False
        )
        loss = self.loss_func(outputs, y)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)
        return [optimizer], [scheduler]


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super().__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]