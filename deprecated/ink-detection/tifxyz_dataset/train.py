import json
import wandb
import torch
import numpy as np 

import accelerate
from accelerate.utils import TorchDynamoPlugin

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss

from vesuvius.neural_tracing.nets.models import make_model

from .tifxyz_dataset import TifxyzInkDataset

################################
# THIS IS A STUB -- DO NOT USE #
################################
################################
# THIS IS A STUB -- DO NOT USE #
################################
################################
# THIS IS A STUB -- DO NOT USE #
################################

class TifxyzInkTrainer():
    def __init__(config_path):
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        dynamo_plugin = TorchDynamoPlugin(
            backend   = "inductor",
            mode      = "default",
            fullgraph = False,
            dynamic   = False
        )

        dataloader_config = accelerate.DataLoaderConfiguration(
            non_blocking = True
        )

        accelerator = accelerate.Accelerator(
            mixed_precision             = config.get('mixed_precision', "fp16"),
            gradient_accumulation_steps = config.get('grad_acc_steps', 1),
            dynamo_plugin               = dynamo_plugin,
            dataloader_config           = dataloader_config
        )

        if 'wandb_project' in config and accelerator.is_main_process:
            wandb_kwargs = {
                'project' : config['wandb_project'],
                'entity'  : config['wandb_entity'],
                'config'  : config
            }

            wandb.init(**wandb_kwargs)

        train_ds = TifxyzInkDataset(
            config,
            apply_augmentation=True
        )

        val_ds = TifxyzInkDataset(
            config,
            apply_augmentation=False
        )

        num_patches = len(train_ds)
        num_val     = max(1, num_patches * config.get('val_fraction', 0.1))
        num_train   = num_patches - num_val

        raise NotImplementedError("Do not use this trainer, it is not done, it does nothing.")

