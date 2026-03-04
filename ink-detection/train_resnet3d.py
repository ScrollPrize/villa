import math

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
    if args.max_steps_per_epoch is not None:
        if int(args.max_steps_per_epoch) < 1:
            raise ValueError(f"--max_steps_per_epoch must be >= 1, got {args.max_steps_per_epoch}")
        CFG.max_steps_per_epoch = int(args.max_steps_per_epoch)
        orchestration.log(
            f"runtime override max_steps_per_epoch={CFG.max_steps_per_epoch} "
            "(train DataLoader uses shuffle=False; sampler controls sample order)"
        )
    else:
        CFG.max_steps_per_epoch = None
    run_state = orchestration.prepare_run(args, merged_config, wandb_logger)
    data_state = build_datasets(run_state)
    if args.max_steps_per_epoch is not None:
        base_epochs = int(getattr(CFG, "epochs", 1))
        accum = max(1, int(getattr(CFG, "accumulate_grad_batches", 1) or 1))
        raw_micro_steps_per_epoch = int(len(data_state["train_loader"]))
        uncapped_optimizer_steps_per_epoch = int(math.ceil(raw_micro_steps_per_epoch / accum))
        capped_optimizer_steps_per_epoch = int(data_state["steps_per_epoch"])
        if capped_optimizer_steps_per_epoch < 1:
            raise ValueError(
                "max_steps_per_epoch produced 0 optimizer steps per epoch; "
                f"len(train_loader)={raw_micro_steps_per_epoch}, "
                f"max_steps_per_epoch={args.max_steps_per_epoch}, "
                f"accumulate_grad_batches={accum}"
            )
        target_total_optimizer_steps = int(uncapped_optimizer_steps_per_epoch * base_epochs)
        effective_epochs = int(math.ceil(target_total_optimizer_steps / capped_optimizer_steps_per_epoch))
        effective_epochs = max(1, effective_epochs)
        CFG.epochs = effective_epochs
        orchestration.log(
            "dynamic epoch scaling "
            f"base_epochs={base_epochs} uncapped_optimizer_steps_per_epoch={uncapped_optimizer_steps_per_epoch} "
            f"capped_optimizer_steps_per_epoch={capped_optimizer_steps_per_epoch} "
            f"target_total_optimizer_steps={target_total_optimizer_steps} effective_epochs={effective_epochs}"
        )

    model = tr.build_model(run_state, data_state, wandb_logger)
    trainer = tr.build_trainer(args, wandb_logger)
    tr.fit(trainer, model, data_state, run_state)


if __name__ == "__main__":
    main()
