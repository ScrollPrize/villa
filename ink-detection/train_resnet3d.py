from train_resnet3d_lib.config import CFG
from train_resnet3d_lib import training as tr


__all__ = ["CFG", "parse_args", "main"]


def parse_args():
    return tr.parse_args()


def main():
    args = parse_args()
    tr.log_startup(args)
    base_config = tr.load_base_config(args)
    wandb_logger, wandb_info = tr.init_wandb_logger(args, base_config)
    merged_config = tr.merge_config(base_config, wandb_logger, args)
    run_state = tr.prepare_run(args, merged_config, wandb_logger, wandb_info)
    data_state = tr.build_datasets(run_state)
    model = tr.build_model(run_state, data_state, wandb_logger)
    trainer = tr.build_trainer(args, wandb_logger)
    tr.fit(trainer, model, data_state, run_state)


if __name__ == "__main__":
    main()
