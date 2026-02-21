from train_resnet3d_lib.config import CFG
from train_resnet3d_lib import training as tr
from train_resnet3d_lib import orchestration
from train_resnet3d_lib.datasets_builder import build_datasets


__all__ = ["CFG", "parse_args", "main"]


def parse_args():
    return orchestration.parse_args()


def main():
    args = parse_args()
    orchestration.log_startup(args)
    base_config = orchestration.load_base_config(args)
    preinit_overrides = orchestration.load_wandb_preinit_overrides()
    wandb_logger = orchestration.init_wandb_logger(args, base_config, preinit_overrides=preinit_overrides)
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
    tr.fit(trainer, model, data_state, run_state)


if __name__ == "__main__":
    main()
