import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_folds(raw_folds):
    if raw_folds is None:
        return [None]
    folds = []
    for item in str(raw_folds).split(","):
        token = item.strip()
        if token == "":
            raise ValueError("empty token in --folds")
        if not token.isdigit():
            raise ValueError(f"invalid fold value in --folds: {token!r}")
        folds.append(int(token))
    if not folds:
        raise ValueError("--folds produced an empty set")
    return folds


def parse_tags(raw_tags):
    if raw_tags is None:
        return []
    tags = []
    for item in str(raw_tags).split(","):
        token = item.strip()
        if token == "":
            raise ValueError("empty token in --wandb_tags")
        tags.append(token)
    if not tags:
        raise ValueError("--wandb_tags produced an empty set")
    return tags


def normalize_optional_fold_for_tag(raw_fold):
    if raw_fold is None:
        return None
    if isinstance(raw_fold, bool):
        raise TypeError(f"cv_fold must not be boolean, got {raw_fold!r}")
    if isinstance(raw_fold, int):
        return raw_fold
    if isinstance(raw_fold, float):
        if not float(raw_fold).is_integer():
            return str(raw_fold)
        return int(raw_fold)
    if isinstance(raw_fold, str):
        stripped = raw_fold.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return str(raw_fold)


def normalize_accumulate_grad_batches_for_tag(raw_value):
    if raw_value is None:
        return 1
    if isinstance(raw_value, bool):
        raise TypeError(
            "metadata.training_hyperparameters.training.accumulate_grad_batches must be a positive integer, "
            f"got boolean {raw_value!r}"
        )
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        if not float(raw_value).is_integer():
            raise ValueError(
                "metadata.training_hyperparameters.training.accumulate_grad_batches must be a positive integer, "
                f"got {raw_value!r}"
            )
        value = int(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped.isdigit():
            raise ValueError(
                "metadata.training_hyperparameters.training.accumulate_grad_batches must be a positive integer, "
                f"got {raw_value!r}"
            )
        value = int(stripped)
    else:
        raise TypeError(
            "metadata.training_hyperparameters.training.accumulate_grad_batches must be a positive integer, "
            f"got {type(raw_value).__name__}"
        )
    if value < 1:
        raise ValueError(
            "metadata.training_hyperparameters.training.accumulate_grad_batches must be >= 1, "
            f"got {value}"
        )
    return value


def main():
    parser = argparse.ArgumentParser(description="Run ERM training once per group (no sweep).")
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--outputs_path", type=str, default=None)
    parser.add_argument(
        "--init_ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint to initialize weights from (fine-tune). Passed to train_resnet3d.py.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names. Defaults to all groups present in training.train_segments.",
    )
    parser.add_argument(
        "--group_key",
        type=str,
        default=None,
        help="Segments metadata key used to define groups (defaults to group_dro.group_key or base_path).",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated cv folds to run (e.g. '0,1,2'). "
            "If omitted, keep metadata.training.cv_fold as-is."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional override for training_hyperparameters.training.lr.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optional override for training_hyperparameters.training.weight_decay.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training_hyperparameters.training.epochs (must be >= 1).",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Optional comma-separated W&B tags to append to each run.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.metadata_json is None:
        metadata_path = (project_root / "metadata.json").resolve()
    else:
        metadata_path = Path(args.metadata_json).expanduser().resolve()
    base_metadata = load_json(metadata_path)
    base_segments = base_metadata.get("segments", {}) or {}

    training_cfg = base_metadata.get("training", {}) or {}
    train_segments = training_cfg.get("train_segments") or list(base_segments.keys())
    if not train_segments:
        raise ValueError("No segments found in metadata (training.train_segments and segments are both empty).")

    val_segments = list(training_cfg.get("val_segments") or train_segments)
    if not val_segments:
        raise ValueError("training.val_segments is empty (and no default could be inferred).")

    group_key = args.group_key
    if group_key is None:
        group_key = (base_metadata.get("group_dro", {}) or {}).get("group_key")
    if group_key is None:
        group_key = "base_path"

    group_to_segments = {}
    for segment_id in train_segments:
        seg_meta = base_segments.get(segment_id, {}) or {}
        group_name = seg_meta.get(group_key)
        if group_name is None:
            group_name = seg_meta.get("base_path", segment_id)
        group_name = str(group_name)
        group_to_segments.setdefault(group_name, []).append(segment_id)

    if args.groups is None:
        group_names = sorted(group_to_segments.keys())
    else:
        group_names = [s.strip() for s in str(args.groups).split(",") if s.strip()]
    folds = parse_folds(args.folds)
    custom_wandb_tags = parse_tags(args.wandb_tags)

    train_script = (project_root / "train_resnet3d.py").resolve()

    tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="per_group_erm_")
    tmp_dir = Path(tmp_dir_ctx.name)
    print(f"writing temporary metadata overrides to {tmp_dir}")

    try:
        for group_name in group_names:
            segment_ids = group_to_segments.get(group_name)
            if not segment_ids:
                raise ValueError(f"Group {group_name!r} is not present in training.train_segments (group_key={group_key!r}).")

            safe_group_name = group_name.replace("/", "_")

            for fold in folds:
                run_name = f"erm_group_{safe_group_name}"
                if fold is not None:
                    run_name = f"{run_name}_fold{int(fold)}"

                md = json.loads(json.dumps(base_metadata))
                md.setdefault("training", {})
                if not isinstance(md["training"], dict):
                    raise TypeError(f"metadata.training must be an object, got {type(md['training']).__name__}")
                md["training"]["objective"] = "erm"
                md["training"]["sampler"] = "shuffle"
                md["training"]["loss_mode"] = "batch"
                md["training"]["train_segments"] = list(segment_ids)
                md["training"]["val_segments"] = val_segments
                if fold is not None:
                    md["training"]["cv_fold"] = int(fold)

                if args.lr is not None or args.weight_decay is not None or args.epochs is not None:
                    md.setdefault("training_hyperparameters", {})
                    if not isinstance(md["training_hyperparameters"], dict):
                        raise TypeError(
                            "metadata.training_hyperparameters must be an object, "
                            f"got {type(md['training_hyperparameters']).__name__}"
                        )
                    md["training_hyperparameters"].setdefault("training", {})
                    if not isinstance(md["training_hyperparameters"]["training"], dict):
                        raise TypeError(
                            "metadata.training_hyperparameters.training must be an object, "
                            f"got {type(md['training_hyperparameters']['training']).__name__}"
                        )
                    if args.lr is not None:
                        md["training_hyperparameters"]["training"]["lr"] = float(args.lr)
                    if args.weight_decay is not None:
                        md["training_hyperparameters"]["training"]["weight_decay"] = float(args.weight_decay)
                    if args.epochs is not None:
                        if int(args.epochs) < 1:
                            raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
                        md["training_hyperparameters"]["training"]["epochs"] = int(args.epochs)

                md.setdefault("wandb", {})
                if not isinstance(md["wandb"], dict):
                    raise TypeError(f"metadata.wandb must be an object, got {type(md['wandb']).__name__}")
                existing_tags = md["wandb"].get("tags", [])
                if existing_tags is None:
                    existing_tags = []
                if not isinstance(existing_tags, list):
                    raise TypeError(
                        "metadata.wandb.tags must be a list of strings, "
                        f"got {type(existing_tags).__name__}"
                    )
                normalized_existing_tags = []
                for idx, tag in enumerate(existing_tags):
                    if not isinstance(tag, str):
                        raise TypeError(f"metadata.wandb.tags[{idx}] must be a string, got {type(tag).__name__}")
                    normalized_tag = tag.strip()
                    if not normalized_tag:
                        raise ValueError(f"metadata.wandb.tags[{idx}] must be non-empty")
                    normalized_existing_tags.append(normalized_tag)

                if "training_hyperparameters" not in md or not isinstance(md["training_hyperparameters"], dict):
                    raise KeyError("metadata must define an object at key 'training_hyperparameters'")
                if "training" not in md["training_hyperparameters"] or not isinstance(
                    md["training_hyperparameters"]["training"], dict
                ):
                    raise KeyError("metadata.training_hyperparameters must define an object at key 'training'")
                effective_accumulate_grad_batches = normalize_accumulate_grad_batches_for_tag(
                    md["training_hyperparameters"]["training"].get("accumulate_grad_batches")
                )
                effective_fold = normalize_optional_fold_for_tag(md["training"].get("cv_fold"))

                auto_wandb_tags = [
                    "erm",
                    "shuffle",
                    "hparam_sweep",
                    f"accum{effective_accumulate_grad_batches}",
                    "per_group_erm",
                    f"group_{safe_group_name}",
                ]
                if effective_fold is not None:
                    auto_wandb_tags.append(f"fold{effective_fold}")
                if args.epochs is not None:
                    auto_wandb_tags.append(f"epochs{int(args.epochs)}")
                if args.lr is not None:
                    auto_wandb_tags.append(f"lr_{float(args.lr):.2e}")
                if args.weight_decay is not None:
                    auto_wandb_tags.append(f"wd_{float(args.weight_decay):.2e}")

                merged_tags = []
                seen_tags = set()
                for tag in normalized_existing_tags + custom_wandb_tags + auto_wandb_tags:
                    if tag in seen_tags:
                        continue
                    seen_tags.add(tag)
                    merged_tags.append(tag)
                md["wandb"]["tags"] = merged_tags

                per_group_metadata_path = tmp_dir / f"metadata_{run_name}.json"
                per_group_metadata_path.write_text(json.dumps(md, indent=2))

                cmd = [
                    sys.executable,
                    str(train_script),
                    "--metadata_json",
                    str(per_group_metadata_path),
                    "--run_name",
                    run_name,
                ]
                if args.outputs_path is not None:
                    cmd += ["--outputs_path", str(args.outputs_path)]
                if args.init_ckpt_path is not None:
                    cmd += ["--init_ckpt_path", str(args.init_ckpt_path)]
                cmd += passthrough

                print("running:", " ".join(shlex.quote(c) for c in cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
    finally:
        tmp_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
