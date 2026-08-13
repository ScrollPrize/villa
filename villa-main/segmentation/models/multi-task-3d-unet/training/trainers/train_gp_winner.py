from configuration.config_manager import ConfigManager
from training.trainers.basetrainer import BaseTrainer
from builders.models.ink import TimeSFormerInk
from builders.models.youssef_creations import get_scheduler, CFG
from torch.optim import AdamW
import segmentation_models_pytorch as smp


class TrainerTimesFormer(BaseTrainer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.mgr = ConfigManager(config_file)
        self.mgr.train_patch_size = (16, 256, 256)

    def _build_model(self):
        model = TimeSFormerInk(pred_shape=self.mgr.train_patch_size[2])
        return model

    def _build_loss(self):

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        loss_fns = {}
        for task_name, task_info in self.mgr.tasks.items():
            loss_fn = self.loss_func
            loss_fns[task_name] = loss_fn

        return loss_fns

    def _get_optimizer(self, model):
        optimizer = AdamW(model, lr=3e-5)
        return optimizer

    def _get_scheduler(self, optimizer):
        config = CFG
        scheduler = get_scheduler(config, optimizer)
        return scheduler
