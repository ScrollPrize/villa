from configuration.config_manager import ConfigManager
from training.trainers.basetrainer import BaseTrainer
from builders.models.ink import FirstLettersi3dModel


class TrainerFirstLetters(BaseTrainer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.mgr = ConfigManager(config_file)
        self.mgr.train_patch_size = (20, 256, 256)

    def _build_model(self):
        model = FirstLettersi3dModel(pred_shape=self.mgr.train_patch_size[2])
        return model

    def _build_loss(self):
        pass