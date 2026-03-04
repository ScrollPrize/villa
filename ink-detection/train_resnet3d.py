import os.path as osp

from train_resnet3d_lib.config import CFG
from train_resnet3d_lib import training as tr
from train_resnet3d_lib.runtime import orchestration
from train_resnet3d_lib.runtime import wandb_runtime
from train_resnet3d_lib.datasets_builder import build_datasets


__all__ = ["CFG", "parse_args", "main"]


def parse_args():
    return orchestration.parse_args()


def main():
    args = parse_args()
    orchestration.log_startup(args)
    base_config = orchestration.load_and_validate_base_config(
        args.metadata_json,
        base_dir=osp.dirname(orchestration.__file__),
    )
    preinit_overrides = wandb_runtime.load_wandb_preinit_overrides()
    wandb_logger = wandb_runtime.init_wandb_logger(args, base_config, preinit_overrides=preinit_overrides)
    merged_config = orchestration.merge_config(
        base_config,
        wandb_logger,
        args,
        preinit_overrides=preinit_overrides,
    )
    run_state = orchestration.prepare_run(args, merged_config, wandb_logger)
    data_state = build_datasets(run_state)
    model = tr.build_model(run_state, data_state, wandb_logger)
    trainer = tr.build_trainer(args, wandb_logger)
    if args.stitch_only:
        tr.validate(trainer, model, data_state, run_state)
    else:
        tr.fit(trainer, model, data_state, run_state)


if __name__ == "__main__":
    main()
