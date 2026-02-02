import time
import os.path as osp
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from group_dro import GroupDROComputer
from warmup_scheduler import GradualWarmupScheduler
from models.resnetall import generate_model
# from models.i3dallnl import InceptionI3d
from train_resnet3d_lib.config import CFG, log


# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')


def _pick_group_norm_groups(num_channels: int, desired_groups: int) -> int:
    num_channels = int(num_channels)
    desired_groups = int(desired_groups)
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    desired_groups = max(1, min(desired_groups, num_channels))
    for g in range(desired_groups, 0, -1):
        if num_channels % g == 0:
            return g
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
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

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


class StitchManager:
    def __init__(
        self,
        *,
        stitch_val_dataloader_idx=None,
        stitch_pred_shape=None,
        stitch_segment_id=None,
        stitch_all_val=False,
        stitch_downsample=1,
        stitch_all_val_shapes=None,
        stitch_all_val_segment_ids=None,
        stitch_train_shapes=None,
        stitch_train_segment_ids=None,
        stitch_train=False,
        stitch_train_every_n_epochs=1,
    ):
        self.downsample = max(1, int(stitch_downsample or 1))
        self.buffers = {}
        self.segment_ids = {}
        self.train_buffers = {}
        self.train_segment_ids = []
        self.train_loaders = []
        self.train_enabled = bool(stitch_train)
        self.train_every_n_epochs = max(1, int(stitch_train_every_n_epochs or 1))
        self.borders_by_split = {"train": {}, "val": {}}

        if bool(stitch_all_val):
            if stitch_all_val_shapes is None or stitch_all_val_segment_ids is None:
                raise ValueError("stitch_all_val requires stitch_all_val_shapes and stitch_all_val_segment_ids")
            if len(stitch_all_val_shapes) != len(stitch_all_val_segment_ids):
                raise ValueError(
                    "stitch_all_val_shapes and stitch_all_val_segment_ids must have the same length "
                    f"(got {len(stitch_all_val_shapes)} vs {len(stitch_all_val_segment_ids)})"
                )

            for loader_idx, (segment_id, shape) in enumerate(zip(stitch_all_val_segment_ids, stitch_all_val_shapes)):
                h = int(shape[0])
                w = int(shape[1])
                ds_h = (h + self.downsample - 1) // self.downsample
                ds_w = (w + self.downsample - 1) // self.downsample
                self.buffers[int(loader_idx)] = (
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                )
                self.segment_ids[int(loader_idx)] = str(segment_id)
        else:
            stitch_enabled = (stitch_val_dataloader_idx is not None) and (stitch_pred_shape is not None)
            if stitch_enabled:
                h = int(stitch_pred_shape[0])
                w = int(stitch_pred_shape[1])
                ds_h = (h + self.downsample - 1) // self.downsample
                ds_w = (w + self.downsample - 1) // self.downsample
                idx = int(stitch_val_dataloader_idx)
                self.buffers[idx] = (
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                )
                self.segment_ids[idx] = str(stitch_segment_id or idx)

        if stitch_train_shapes is not None or stitch_train_segment_ids is not None:
            stitch_train_shapes = stitch_train_shapes or []
            stitch_train_segment_ids = stitch_train_segment_ids or []
            if len(stitch_train_shapes) != len(stitch_train_segment_ids):
                raise ValueError(
                    "stitch_train_shapes and stitch_train_segment_ids must have the same length "
                    f"(got {len(stitch_train_shapes)} vs {len(stitch_train_segment_ids)})"
                )

            for segment_id, shape in zip(stitch_train_segment_ids, stitch_train_shapes):
                h = int(shape[0])
                w = int(shape[1])
                ds_h = (h + self.downsample - 1) // self.downsample
                ds_w = (w + self.downsample - 1) // self.downsample
                self.train_buffers[str(segment_id)] = (
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                )
                self.train_segment_ids.append(str(segment_id))

        self.enabled = len(self.buffers) > 0

    def set_borders(self, *, train_borders=None, val_borders=None):
        if train_borders is not None:
            self.borders_by_split["train"] = dict(train_borders)
        if val_borders is not None:
            self.borders_by_split["val"] = dict(val_borders)

    def set_train_loaders(self, loaders, segment_ids):
        self.train_loaders = list(loaders or [])
        self.train_segment_ids = [str(x) for x in (segment_ids or [])]

    def accumulate_to_buffers(self, *, outputs, xyxys, pred_buf, count_buf):
        ds = self.downsample
        y_preds = torch.sigmoid(outputs).to("cpu")
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            x1_ds = x1 // ds
            y1_ds = y1 // ds
            x2_ds = (x2 + ds - 1) // ds
            y2_ds = (y2 + ds - 1) // ds
            target_h = y2_ds - y1_ds
            target_w = x2_ds - x1_ds
            if target_h <= 0 or target_w <= 0:
                continue

            pred_patch = y_preds[i].unsqueeze(0).float()
            if pred_patch.shape[-2:] != (target_h, target_w):
                pred_patch = F.interpolate(
                    pred_patch,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

            pred_buf[y1_ds:y2_ds, x1_ds:x2_ds] += pred_patch.squeeze(0).squeeze(0).numpy()
            count_buf[y1_ds:y2_ds, x1_ds:x2_ds] += 1.0

    def accumulate_val(self, *, outputs, xyxys, dataloader_idx):
        if not self.enabled:
            return
        idx = int(dataloader_idx)
        if idx not in self.buffers:
            return
        pred_buf, count_buf = self.buffers[idx]
        self.accumulate_to_buffers(outputs=outputs, xyxys=xyxys, pred_buf=pred_buf, count_buf=count_buf)

    def run_train_stitch_pass(self, model):
        if not self.train_enabled:
            return False
        if not self.train_loaders or not self.train_segment_ids:
            return False
        if len(self.train_loaders) != len(self.train_segment_ids):
            raise ValueError(
                "train stitch loaders/segment_ids length mismatch "
                f"({len(self.train_loaders)} vs {len(self.train_segment_ids)})"
            )

        epoch = int(getattr(model, "current_epoch", 0))
        if self.train_every_n_epochs > 1:
            if ((epoch + 1) % self.train_every_n_epochs) != 0:
                return False

        t0 = time.perf_counter()
        from train_resnet3d_lib.config import log
        log(f"train stitch pass start epoch={epoch}")

        for segment_id in self.train_segment_ids:
            if segment_id not in self.train_buffers:
                raise ValueError(f"Missing train stitch buffers for segment_id={segment_id!r}")
            pred_buf, count_buf = self.train_buffers[segment_id]
            pred_buf.fill(0)
            count_buf.fill(0)

        was_training = model.training
        precision_context = nullcontext()
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            strategy = getattr(trainer, "strategy", None)
            precision_plugin = getattr(strategy, "precision_plugin", None) if strategy is not None else None
            if precision_plugin is None:
                precision_plugin = getattr(trainer, "precision_plugin", None)
            if precision_plugin is not None and hasattr(precision_plugin, "forward_context"):
                precision_context = precision_plugin.forward_context()
        try:
            model.eval()
            with torch.inference_mode(), precision_context:
                for loader, segment_id in zip(self.train_loaders, self.train_segment_ids):
                    pred_buf, count_buf = self.train_buffers[str(segment_id)]
                    for batch in loader:
                        x, _y, xyxys, _g = batch
                        x = x.to(model.device, non_blocking=True)
                        outputs = model(x)
                        self.accumulate_to_buffers(
                            outputs=outputs,
                            xyxys=xyxys,
                            pred_buf=pred_buf,
                            count_buf=count_buf,
                        )
        finally:
            if was_training:
                model.train()

        log(f"train stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
        return True

    def on_validation_epoch_end(self, model):
        if not self.enabled or not self.buffers:
            return

        sanity_checking = bool(model.trainer is not None and getattr(model.trainer, "sanity_checking", False))
        is_global_zero = bool(model.trainer is None or model.trainer.is_global_zero)
        train_configured = bool(self.train_loaders) and bool(self.train_segment_ids) and bool(self.train_buffers)
        stitch_train_mode = bool(self.train_enabled and train_configured)

        did_run_train_stitch = False
        if stitch_train_mode and (not is_global_zero):
            # Only rank-0 runs the extra train visualization pass + logging.
            for pred_buf, count_buf in self.buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)
            return
        if stitch_train_mode and (not sanity_checking):
            did_run_train_stitch = self.run_train_stitch_pass(model)
            if not did_run_train_stitch:
                # train stitching is enabled but only runs every N epochs; fall back to val-only logging.
                stitch_train_mode = False

        log_train_stitch = bool(stitch_train_mode and did_run_train_stitch)

        images = []
        captions = []

        if log_train_stitch:
            segment_to_val = {}
            for loader_idx, (pred_buf, count_buf) in self.buffers.items():
                segment_id = self.segment_ids.get(loader_idx, str(loader_idx))
                stitched = np.divide(
                    pred_buf,
                    count_buf,
                    out=np.zeros_like(pred_buf),
                    where=count_buf != 0,
                )
                segment_to_val[str(segment_id)] = (np.clip(stitched, 0, 1), (count_buf != 0))

            segment_to_train = {}
            for segment_id in self.train_segment_ids:
                pred_buf, count_buf = self.train_buffers[str(segment_id)]
                stitched = np.divide(
                    pred_buf,
                    count_buf,
                    out=np.zeros_like(pred_buf),
                    where=count_buf != 0,
                )
                segment_to_train[str(segment_id)] = (np.clip(stitched, 0, 1), (count_buf != 0))

            segment_ids = sorted(set(segment_to_val.keys()) | set(segment_to_train.keys()))
            for segment_id in segment_ids:
                base = None
                if segment_id in segment_to_val:
                    base = segment_to_val[segment_id][0].copy()
                elif segment_id in segment_to_train:
                    base = segment_to_train[segment_id][0].copy()
                else:
                    continue

                if segment_id in segment_to_train and segment_id in segment_to_val:
                    train_img, train_has = segment_to_train[segment_id]
                    val_img, val_has = segment_to_val[segment_id]
                    base[train_has] = train_img[train_has]
                    base[val_has] = val_img[val_has]

                rgb = np.repeat(base[..., None], 3, axis=2)
                train_border = self.borders_by_split.get("train", {}).get(str(segment_id))
                val_border = self.borders_by_split.get("val", {}).get(str(segment_id))
                if train_border is not None:
                    rgb[train_border.astype(bool)] = np.array([1.0, 0.0, 0.0], dtype=rgb.dtype)
                if val_border is not None:
                    rgb[val_border.astype(bool)] = np.array([0.0, 0.0, 1.0], dtype=rgb.dtype)

                images.append(rgb)
                has_train = segment_id in segment_to_train and bool(segment_to_train[segment_id][1].any())
                has_val = segment_id in segment_to_val and bool(segment_to_val[segment_id][1].any())
                if has_train and has_val:
                    split_tag = "train+val"
                elif has_train:
                    split_tag = "train"
                elif has_val:
                    split_tag = "val"
                else:
                    split_tag = "none"
                captions.append(f"{segment_id} ({split_tag} ds={self.downsample})")
        else:
            want_color = bool(self.train_enabled)
            for loader_idx, (pred_buf, count_buf) in self.buffers.items():
                stitched = np.divide(
                    pred_buf,
                    count_buf,
                    out=np.zeros_like(pred_buf),
                    where=count_buf != 0,
                )
                segment_id = self.segment_ids.get(loader_idx, str(loader_idx))
                base = np.clip(stitched, 0, 1)
                if want_color:
                    rgb = np.repeat(base[..., None], 3, axis=2)
                    train_border = self.borders_by_split.get("train", {}).get(str(segment_id))
                    val_border = self.borders_by_split.get("val", {}).get(str(segment_id))
                    if train_border is not None:
                        rgb[train_border.astype(bool)] = np.array([1.0, 0.0, 0.0], dtype=rgb.dtype)
                    if val_border is not None:
                        rgb[val_border.astype(bool)] = np.array([0.0, 0.0, 1.0], dtype=rgb.dtype)
                    images.append(rgb)
                else:
                    images.append(base)
                captions.append(f"{segment_id} (val ds={self.downsample})")

        if (not sanity_checking) and (model.trainer is None or model.trainer.is_global_zero):
            if isinstance(model.logger, WandbLogger):
                step = None
                try:
                    step = int(getattr(model.trainer, "global_step", 0))
                except Exception:
                    step = None
                if step is None:
                    model.logger.log_image(key="masks", images=images, caption=captions)
                else:
                    model.logger.log_image(key="masks", images=images, caption=captions, step=step)

        # reset stitch buffers
        for pred_buf, count_buf in self.buffers.values():
            pred_buf.fill(0)
            count_buf.fill(0)
        if log_train_stitch:
            for pred_buf, count_buf in self.train_buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)

class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        size=256,
        enc='',
        with_norm=False,
        objective="erm",
        loss_mode="batch",
        robust_step_size=None,
        group_counts=None,
        group_dro_gamma=0.1,
        group_dro_btl=False,
        group_dro_alpha=None,
        group_dro_normalize_loss=False,
        group_dro_min_var_weight=0.0,
        group_dro_adj=None,
        total_steps=780,
        n_groups=1,
        group_names=None,
        stitch_val_dataloader_idx=None,
        stitch_pred_shape=None,
        stitch_segment_id=None,
        stitch_all_val=False,
        stitch_downsample=1,
        stitch_all_val_shapes=None,
        stitch_all_val_segment_ids=None,
        stitch_train_shapes=None,
        stitch_train_segment_ids=None,
        stitch_train=False,
        stitch_train_every_n_epochs=1,
        norm="batch",
        group_norm_groups=32,
        erm_group_topk=0,
    ):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()

        self.n_groups = int(n_groups)
        if group_names is None:
            group_names = [str(i) for i in range(self.n_groups)]
        self.group_names = list(group_names)

        self.group_dro = None
        if str(self.hparams.objective).lower() == "group_dro":
            if robust_step_size is None:
                raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")
            if group_counts is None:
                raise ValueError("group_counts is required when training.objective is group_dro")

            self.group_dro = GroupDROComputer(
                n_groups=self.n_groups,
                group_counts=group_counts,
                alpha=group_dro_alpha,
                gamma=group_dro_gamma,
                adj=group_dro_adj,
                min_var_weight=group_dro_min_var_weight,
                step_size=robust_step_size,
                normalize_loss=group_dro_normalize_loss,
                btl=group_dro_btl,
            )

        self.erm_group_topk = int(erm_group_topk or 0)
        if self.erm_group_topk < 0:
            raise ValueError(f"erm_group_topk must be >= 0, got {self.erm_group_topk}")

        self.loss_func1 = smp.losses.DiceLoss(mode="binary")
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)

        self.backbone = generate_model(
            model_depth=50,
            n_input_channels=1,
            forward_features=True,
            n_classes=1039,
        )

        norm = str(norm).lower()
        group_norm_groups = int(group_norm_groups)
        init_ckpt_path = getattr(CFG, "init_ckpt_path", None)
        if not init_ckpt_path:
            backbone_pretrained_path = getattr(CFG, "backbone_pretrained_path", "./r3d50_KM_200ep.pth")
            if not osp.exists(backbone_pretrained_path):
                raise FileNotFoundError(
                    f"Missing backbone pretrained weights: {backbone_pretrained_path}. "
                    "Either place r3d50_KM_200ep.pth next to train_resnet3d.py, set CFG.backbone_pretrained_path, "
                    "or pass --init_ckpt_path to fine-tune from a previous run."
                )
            backbone_ckpt = torch.load(backbone_pretrained_path, map_location="cpu")
            state_dict = backbone_ckpt.get("state_dict", backbone_ckpt)
            conv1_weight = state_dict["conv1.weight"]
            state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict, strict=False)

        if norm == "group":
            replace_batchnorm_with_groupnorm(self.backbone, desired_groups=group_norm_groups)

        was_training = self.backbone.training
        try:
            self.backbone.eval()
            with torch.no_grad():
                encoder_dims = [x.size(1) for x in self.backbone(torch.rand(1, 1, 20, 256, 256))]
        finally:
            if was_training:
                self.backbone.train()

        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups)

        if self.hparams.with_norm:
            if norm == "group":
                self.normalization = nn.GroupNorm(num_groups=1, num_channels=1)
            else:
                self.normalization = nn.BatchNorm3d(num_features=1)

        self._stitcher = StitchManager(
            stitch_val_dataloader_idx=stitch_val_dataloader_idx,
            stitch_pred_shape=stitch_pred_shape,
            stitch_segment_id=stitch_segment_id,
            stitch_all_val=bool(stitch_all_val),
            stitch_downsample=int(stitch_downsample or 1),
            stitch_all_val_shapes=stitch_all_val_shapes,
            stitch_all_val_segment_ids=stitch_all_val_segment_ids,
            stitch_train_shapes=stitch_train_shapes,
            stitch_train_segment_ids=stitch_train_segment_ids,
            stitch_train=bool(stitch_train),
            stitch_train_every_n_epochs=int(stitch_train_every_n_epochs or 1),
        )

    def set_stitch_borders(self, *, train_borders=None, val_borders=None):
        self._stitcher.set_borders(train_borders=train_borders, val_borders=val_borders)

    def set_train_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_train_loaders(loaders, segment_ids)

    def _accumulate_stitch_predictions(self, *, outputs, xyxys, pred_buf, count_buf):
        self._stitcher.accumulate_to_buffers(outputs=outputs, xyxys=xyxys, pred_buf=pred_buf, count_buf=count_buf)

    def _run_train_stitch_pass(self):
        return self._stitcher.run_train_stitch_pass(self)

    def on_train_epoch_start(self):
        device = self.device
        self._train_loss_sum = torch.tensor(0.0, device=device)
        self._train_dice_sum = torch.tensor(0.0, device=device)
        self._train_count = torch.tensor(0.0, device=device)

        self._train_group_loss_sum = torch.zeros(self.n_groups, device=device)
        self._train_group_dice_sum = torch.zeros(self.n_groups, device=device)
        self._train_group_count = torch.zeros(self.n_groups, device=device)

    def _update_train_stats(self, per_sample_loss, per_sample_dice, group_idx):
        self._train_loss_sum += per_sample_loss.sum()
        self._train_dice_sum += per_sample_dice.sum()
        self._train_count += float(per_sample_loss.numel())

        group_idx = group_idx.long()
        self._train_group_loss_sum.scatter_add_(0, group_idx, per_sample_loss)
        self._train_group_dice_sum.scatter_add_(0, group_idx, per_sample_dice)
        self._train_group_count.scatter_add_(
            0,
            group_idx,
            torch.ones_like(per_sample_loss, dtype=self._train_group_count.dtype),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)

        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        targets = targets.float()

        smooth_factor = 0.25
        soft_targets = (1.0 - targets) * smooth_factor + targets * (1.0 - smooth_factor)

        bce = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
        bce = bce.mean(dim=(1, 2, 3))

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        eps = 1e-7
        dice = (2 * intersection + eps) / (union + eps)

        dice_loss = 1.0 - dice
        per_sample_loss = 0.5 * dice_loss + 0.5 * bce
        return per_sample_loss, dice, bce, dice_loss

    def compute_group_avg(self, values, group_idx):
        group_idx = group_idx.long()
        group_map = (
            group_idx
            == torch.arange(self.n_groups, device=group_idx.device).unsqueeze(1).long()
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_avg = (group_map @ values.view(-1)) / group_denom
        return group_avg, group_count

    def compute_batch_loss(self, logits, targets):
        return 0.5 * self.loss_func1(logits, targets) + 0.5 * self.loss_func2(logits, targets)

    def training_step(self, batch, batch_idx):
        x, y, g = batch
        outputs = self(x)

        objective = str(self.hparams.objective).lower()
        loss_mode = str(self.hparams.loss_mode).lower()
        g = g.long()
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)

        if objective == "erm":
            if loss_mode == "batch":
                dice_loss = self.loss_func1(outputs, y)
                bce_loss = self.loss_func2(outputs, y)
                loss = 0.5 * dice_loss + 0.5 * bce_loss
                self.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
            elif loss_mode == "per_sample":
                if self.erm_group_topk > 0:
                    group_loss, group_count = self.compute_group_avg(per_sample_loss, g)
                    present = group_count > 0
                    if present.any():
                        present_losses = group_loss[present]
                        k = min(int(self.erm_group_topk), int(present_losses.numel()))
                        topk_losses, _ = torch.topk(present_losses, k, largest=True)
                        loss = topk_losses.mean()
                    else:
                        loss = per_sample_loss.mean()

                    if self.global_step % CFG.print_freq == 0:
                        if present.any():
                            worst_group_loss = group_loss[present].max()
                        else:
                            worst_group_loss = group_loss.max()
                        self.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)
                        for group_idx, group_name in enumerate(self.group_names):
                            safe_group_name = str(group_name).replace("/", "_")
                            self.log(
                                f"train/group_{group_idx}_{safe_group_name}/loss",
                                group_loss[group_idx],
                                on_step=True,
                                on_epoch=False,
                            )
                            self.log(
                                f"train/group_{group_idx}_{safe_group_name}/count",
                                group_count[group_idx],
                                on_step=True,
                                on_epoch=False,
                            )
                else:
                    loss = per_sample_loss.mean()
                self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)
            else:
                raise ValueError(f"Unknown training.loss_mode: {self.hparams.loss_mode!r}")
        elif objective == "group_dro":
            if loss_mode != "per_sample":
                raise ValueError("GroupDRO requires training.loss_mode=per_sample")
            if self.group_dro is None:
                raise RuntimeError("GroupDRO objective was set but group_dro computer was not initialized")

            robust_loss, group_loss, group_count, _weights = self.group_dro.loss(per_sample_loss, g)
            loss = robust_loss
            self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)

            if self.global_step % CFG.print_freq == 0:
                present = group_count > 0
                if present.any():
                    worst_group_loss = group_loss[present].max()
                else:
                    worst_group_loss = group_loss.max()
                self.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)

                for group_idx, group_name in enumerate(self.group_names):
                    safe_group_name = str(group_name).replace("/", "_")
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/loss",
                        group_loss[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/count",
                        group_count[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/adv_prob",
                        self.group_dro.adv_probs[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
        else:
            raise ValueError(f"Unknown training.objective: {self.hparams.objective!r}")

        self._update_train_stats(per_sample_loss, per_sample_dice, g)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if self._train_count.item() > 0:
            avg_loss = self._train_loss_sum / self._train_count
            avg_dice = self._train_dice_sum / self._train_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_dice = torch.tensor(0.0, device=self.device)

        group_count = self._train_group_count
        group_loss = self._train_group_loss_sum / group_count.clamp_min(1)
        group_dice = self._train_group_dice_sum / group_count.clamp_min(1)

        self.log("train/epoch_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/epoch_avg_dice", avg_dice, on_step=False, on_epoch=True, prog_bar=False)
        for group_idx, group_name in enumerate(self.group_names):
            safe_group_name = str(group_name).replace("/", "_")
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_loss",
                group_loss[group_idx],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_dice",
                group_dice[group_idx],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_count",
                group_count[group_idx],
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_start(self):
        device = self.device
        self._val_loss_sum = torch.tensor(0.0, device=device)
        self._val_dice_sum = torch.tensor(0.0, device=device)
        self._val_bce_sum = torch.tensor(0.0, device=device)
        self._val_dice_loss_sum = torch.tensor(0.0, device=device)
        self._val_count = torch.tensor(0.0, device=device)

        self._val_group_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_bce_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_count = torch.zeros(self.n_groups, device=device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, g = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)

        self._val_loss_sum += per_sample_loss.sum()
        self._val_dice_sum += per_sample_dice.sum()
        self._val_bce_sum += per_sample_bce.sum()
        self._val_dice_loss_sum += per_sample_dice_loss.sum()
        self._val_count += float(per_sample_loss.numel())

        g = g.long()
        self._val_group_loss_sum.scatter_add_(0, g, per_sample_loss)
        self._val_group_dice_sum.scatter_add_(0, g, per_sample_dice)
        self._val_group_bce_sum.scatter_add_(0, g, per_sample_bce)
        self._val_group_dice_loss_sum.scatter_add_(0, g, per_sample_dice_loss)
        self._val_group_count.scatter_add_(0, g, torch.ones_like(per_sample_loss, dtype=self._val_group_count.dtype))

        self._stitcher.accumulate_val(outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)

        return {"loss": per_sample_loss.mean()}

    def on_validation_epoch_end(self):
        if self._val_count.item() > 0:
            avg_loss = self._val_loss_sum / self._val_count
            avg_dice = self._val_dice_sum / self._val_count
            avg_bce = self._val_bce_sum / self._val_count
            avg_dice_loss = self._val_dice_loss_sum / self._val_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_dice = torch.tensor(0.0, device=self.device)
            avg_bce = torch.tensor(0.0, device=self.device)
            avg_dice_loss = torch.tensor(0.0, device=self.device)

        group_count = self._val_group_count
        group_loss = self._val_group_loss_sum / group_count.clamp_min(1)
        group_dice = self._val_group_dice_sum / group_count.clamp_min(1)
        group_bce = self._val_group_bce_sum / group_count.clamp_min(1)
        group_dice_loss = self._val_group_dice_loss_sum / group_count.clamp_min(1)

        present = group_count > 0
        if present.any():
            worst_group_loss = group_loss[present].max()
            worst_group_dice = group_dice[present].min()
        else:
            worst_group_loss = group_loss.max()
            worst_group_dice = group_dice.min()

        self.log("val/avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val/worst_group_loss", worst_group_loss, on_epoch=True, prog_bar=True)
        self.log("val/avg_dice", avg_dice, on_epoch=True, prog_bar=False)
        self.log("val/worst_group_dice", worst_group_dice, on_epoch=True, prog_bar=False)
        self.log("val/avg_bce_loss", avg_bce, on_epoch=True, prog_bar=False)
        self.log("val/avg_dice_loss", avg_dice_loss, on_epoch=True, prog_bar=False)

        for group_idx, group_name in enumerate(self.group_names):
            safe_group_name = str(group_name).replace("/", "_")
            self.log(f"val/group_{group_idx}_{safe_group_name}/loss", group_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice", group_dice[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/bce_loss", group_bce[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice_loss", group_dice_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/count", group_count[group_idx], on_epoch=True)
        self._stitcher.on_validation_epoch_end(self)

    def configure_optimizers(self):
        if bool(getattr(CFG, "exclude_weight_decay_bias_norm", False)) and float(getattr(CFG, "weight_decay", 0.0) or 0.0) > 0:
            decay_params = []
            no_decay_params = []
            for _, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if int(getattr(param, "ndim", 0)) < 2:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer = AdamW(
                [
                    {"params": decay_params, "weight_decay": float(CFG.weight_decay)},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=CFG.lr,
                weight_decay=0.0,
            )
        else:
            optimizer = AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler_name = str(getattr(CFG, "scheduler", "OneCycleLR")).lower()
        steps_per_epoch = int(self.hparams.total_steps)
        epochs = int(CFG.epochs)

        if scheduler_name == "onecyclelr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=CFG.lr,
                pct_start=float(getattr(CFG, "onecycle_pct_start", 0.15)),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                div_factor=float(getattr(CFG, "onecycle_div_factor", 25.0)),
                final_div_factor=float(getattr(CFG, "onecycle_final_div_factor", 1e2)),
            )
            interval = "step"
        elif scheduler_name == "cosine":
            total_steps = max(1, steps_per_epoch * epochs)
            warmup_pct = float(getattr(CFG, "cosine_warmup_pct", 0.0) or 0.0)
            warmup_pct = max(0.0, min(1.0, warmup_pct))
            warmup_steps = int(round(total_steps * warmup_pct))
            warmup_steps = max(0, min(warmup_steps, total_steps - 1))

            eta_min = float(getattr(CFG, "min_lr", 0.0))

            if warmup_steps > 0:
                warmup_factor = float(getattr(CFG, "warmup_factor", 1.0) or 1.0)
                if warmup_factor <= 0:
                    raise ValueError(f"warmup_factor must be > 0, got {warmup_factor}")
                start_factor = 1.0 / warmup_factor

                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(start_factor),
                    end_factor=1.0,
                    total_iters=int(warmup_steps),
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(total_steps - warmup_steps),
                    eta_min=float(eta_min),
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[int(warmup_steps)],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(total_steps),
                    eta_min=float(eta_min),
            )
            interval = "step"
        elif scheduler_name == "gradualwarmupschedulerv2":
            scheduler = get_scheduler(CFG, optimizer)
            interval = "epoch"
        else:
            raise ValueError(
                f"Unsupported scheduler={CFG.scheduler!r}. Supported: 'OneCycleLR' | 'cosine' | 'GradualWarmupSchedulerV2'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
