"""Command-line entry point and compatibility imports for pretraining."""
import argparse
import json
import torch.distributed as dist

from dinovol_2.training.trainer import Trainer
from dinovol_2.training.config import normalize_config
from dinovol_2.training.schedules import CosineScheduler
from dinovol_2.methods.dino import dino_loss_term_count


class DinoIBOTPretrainer(Trainer):
    """Compatibility entry point. New code should use Trainer and trainer.method."""
    @property
    def model_module(self):
        return self.method.model_module

    @property
    def model(self):
        return self.method.model

    def verify_batch_pipeline(self, batch, step=0):
        from dinovol_2.training.diagnostics import verify_batch_pipeline
        return verify_batch_pipeline(self.method, batch, step)

    def verify_train_step(self, batch, step=0):
        from dinovol_2.training.diagnostics import verify_train_step
        return verify_train_step(self, batch, step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal standalone 3D DINO+iBOT pretrainer")
    parser.add_argument("config", type=str)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    config = normalize_config(config)
    runtime = config["runtime"]
    if args.resume_from is not None:
        runtime["resume_from"] = args.resume_from
        runtime["resume"] = True
    if args.no_resume:
        runtime["resume"] = False
        runtime["auto_resume"] = False
        runtime.pop("resume_from", None)
    if args.ddp:
        runtime["use_ddp"] = True

    trainer = Trainer(config)
    try:
        trainer.fit()
    finally:
        trainer._finish_wandb()
        if trainer.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
