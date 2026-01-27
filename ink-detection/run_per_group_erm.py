import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Run ERM training once per group (no sweep).")
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
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
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.metadata_json is None:
        metadata_path = (Path(__file__).resolve().parent / "metadata.json").resolve()
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

    train_script = (Path(__file__).resolve().parent / "train_resnet3d.py").resolve()

    tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="per_group_erm_")
    tmp_dir = Path(tmp_dir_ctx.name)
    print(f"writing temporary metadata overrides to {tmp_dir}")

    try:
        for group_name in group_names:
            segment_ids = group_to_segments.get(group_name)
            if not segment_ids:
                raise ValueError(f"Group {group_name!r} is not present in training.train_segments (group_key={group_key!r}).")

            safe_group_name = group_name.replace("/", "_")
            run_name = f"erm_group_{safe_group_name}"

            md = json.loads(json.dumps(base_metadata))
            md.setdefault("training", {})
            md["training"]["objective"] = "erm"
            md["training"]["sampler"] = "shuffle"
            md["training"]["loss_mode"] = "batch"
            md["training"]["train_segments"] = list(segment_ids)
            md["training"]["val_segments"] = val_segments

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
            if args.project is not None:
                cmd += ["--project", str(args.project)]
            if args.entity is not None:
                cmd += ["--entity", str(args.entity)]
            if args.wandb_group is not None:
                cmd += ["--wandb_group", str(args.wandb_group)]
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
