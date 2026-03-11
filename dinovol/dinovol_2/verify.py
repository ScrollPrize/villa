from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dinovol_2.pretrain import DinoIBOTPretrainer
else:
    from .pretrain import DinoIBOTPretrainer

def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_verification_report(
    config: Mapping[str, Any],
    *,
    step: int = 0,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    resolved_config = dict(config)
    if use_amp is not None:
        resolved_config["use_amp"] = bool(use_amp)
    trainer = DinoIBOTPretrainer(resolved_config)
    dataloader = trainer.build_dataloader()
    batch = next(iter(dataloader))
    report = trainer.verify_train_step(batch, step=step)
    report["config_path"] = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one DINOv2 pretraining batch end to end.")
    parser.add_argument("config", type=str, help="Path to the JSON config file.")
    parser.add_argument("--step", type=int, default=0, help="Scheduler step to use for teacher temperature.")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP for this verification run without editing the config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON verification report.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    report = build_verification_report(
        config,
        step=args.step,
        use_amp=False if args.no_amp else None,
    )
    report["config_path"] = str(config_path)

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
