import pytorch_lightning as pl
import torch

from train_resnet3d_lib.modeling.losses import build_bce_targets, compute_per_sample_loss_and_dice
from train_resnet3d_lib.modeling.model_config import (
    ModelConfig,
    ObjectiveConfig,
    StitchConfig,
    coerce_model_config,
    coerce_objective_config,
    coerce_stitch_config,
)
from train_resnet3d_lib.modeling.optimizers_runtime import configure_optimizers as configure_optimizers_runtime
from train_resnet3d_lib.modeling.runtime_init import (
    initialize_regression_state,
    save_regression_hyperparameters,
)
from train_resnet3d_lib.modeling.train_val_runtime import (
    accumulate_train_stats,
    accumulate_validation_stats,
    compute_objective_loss,
    finalize_training_batch,
    initialize_validation_metrics,
    log_train_epoch_metrics,
    log_validation_epoch_metrics,
    reset_train_epoch_accumulators,
    reset_validation_epoch_accumulators,
    sync_validation_accumulators,
    update_validation_stream_metrics,
)


class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        *,
        model_cfg: ModelConfig | dict,
        objective_cfg: ObjectiveConfig | dict,
        stitch_cfg: StitchConfig | dict,
    ):
        super(RegressionPLModel, self).__init__()
        model_cfg = coerce_model_config(model_cfg)
        objective_cfg = coerce_objective_config(objective_cfg)
        stitch_cfg = coerce_stitch_config(stitch_cfg)
        save_regression_hyperparameters(
            self,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            stitch_cfg=stitch_cfg,
        )
        initialize_regression_state(
            self,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            stitch_cfg=stitch_cfg,
        )

    def set_stitch_borders(self, *, train_borders=None, val_borders=None):
        self._stitcher.set_borders(train_borders=train_borders, val_borders=val_borders)

    def set_train_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_train_loaders(loaders, segment_ids)

    def set_log_only_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_log_only_loaders(loaders, segment_ids)

    def _accumulate_stitch_predictions(self, *, outputs, xyxys, pred_buf, count_buf, offset=(0, 0)):
        self._stitcher.accumulate_to_buffers(
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            offset=offset,
        )

    def on_train_epoch_start(self):
        reset_train_epoch_accumulators(self)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        return compute_per_sample_loss_and_dice(
            logits,
            targets,
            loss_recipe=self.loss_recipe,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def build_bce_targets(self, targets):
        return build_bce_targets(
            targets,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def training_step(self, batch, batch_idx):
        x, y, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        loss = compute_objective_loss(
            self,
            outputs=outputs,
            targets=y,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )

        finalize_training_batch(
            self,
            loss=loss,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            group_idx=group_idx,
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        log_train_epoch_metrics(self)

    def on_validation_epoch_start(self):
        reset_validation_epoch_accumulators(self)
        initialize_validation_metrics(self)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        accumulate_validation_stats(
            self,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )
        update_validation_stream_metrics(self, outputs=outputs, targets=y)

        self._stitcher.accumulate_val(outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)
        return {"loss": per_sample_loss.mean()}

    def on_validation_epoch_end(self):
        sync_validation_accumulators(self)
        log_validation_epoch_metrics(self)
        self._stitcher.on_validation_epoch_end(self)

    def configure_optimizers(self):
        return configure_optimizers_runtime(self)


__all__ = ["RegressionPLModel"]
