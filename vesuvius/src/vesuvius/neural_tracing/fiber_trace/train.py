from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatchBuilder
from vesuvius.neural_tracing.fiber_trace.losses import compute_fiber_trace_loss
from vesuvius.neural_tracing.fiber_trace.model import build_fiber_trace_model


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config JSON must be an object, got {type(config).__name__}")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def run_training(config: dict[str, Any]) -> None:
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if "seed" in config:
        torch.manual_seed(int(config["seed"]))

    batch_builder = FiberTraceBatchBuilder(config)
    model = build_fiber_trace_model(config).to(device)
    opt_cfg = dict(config.get("optimizer", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", opt_cfg.get("lr", 1e-3))),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    loss_cfg = dict(config.get("loss", {}))
    steps = int(config.get("num_steps", config.get("steps", 1)))
    log_every = max(1, int(config.get("log_every", 1)))
    model.train()
    for step in range(1, steps + 1):
        batch = batch_builder.sample_batch().to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.volume, batch.cond_fw_xyz, batch.cond_up_xyz)
        losses = compute_fiber_trace_loss(
            outputs,
            batch,
            temperature=float(loss_cfg.get("temperature", 0.1)),
            contrastive_weight=float(loss_cfg.get("contrastive_weight", 1.0)),
            fw_weight=float(loss_cfg.get("fw_weight", 1.0)),
            up_weight=float(loss_cfg.get("up_weight", 1.0)),
            max_contrastive_samples=int(loss_cfg.get("max_contrastive_samples", 4096)),
        )
        losses.total.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == steps:
            print(
                f"step={step} total={float(losses.total.detach().cpu()):.6f} "
                f"contrastive={float(losses.contrastive.detach().cpu()):.6f} "
                f"fw={float(losses.fw.detach().cpu()):.6f} "
                f"up={float(losses.up.detach().cpu()):.6f}",
                flush=True,
            )

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "steps": steps,
            },
            path,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the fiber tracing MVP model.")
    parser.add_argument("config", help="Path to a fiber trace training config JSON.")
    args = parser.parse_args(argv)
    run_training(_load_config(args.config))


if __name__ == "__main__":
    main()
