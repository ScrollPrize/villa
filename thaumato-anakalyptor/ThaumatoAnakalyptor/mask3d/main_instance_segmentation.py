import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    elif os.path.exists(f"{cfg.general.save_dir}/last-epoch.ckpt"):
        print("EXPERIMENT ALREADY EXIST")
        cfg["trainer"][
            "resume_from_checkpoint"
        ] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    # Adding ModelCheckpoint to automatically save model weights
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.general.save_dir),  # Directory to save the model
        filename='weights-{epoch}-{step}',  # Name of the file
        save_top_k=1,                       # Save only the top k models
        save_last=True,                     # Save the last model as well
        monitor='val_loss',                 # Metric to monitor for saving
        mode='min',                         # Minimize the monitored metric
        save_weights_only=True,             # Save only the weights
    )
    callbacks.append(checkpoint_callback)

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        accelerator="gpu",
        strategy='ddp',
        callbacks=callbacks,
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        accelerator="gpu",
        strategy='ddp',
        logger=loggers,
        **cfg.trainer,
    )
    runner.test(model)


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
