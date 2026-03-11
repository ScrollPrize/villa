import argparse
import copy
import os
import os.path as osp

import torch
from torch.utils.data import Dataset

from train_resnet3d_lib.config import (
    CFG,
    log,
)
from train_resnet3d_lib.runtime import orchestration
from train_resnet3d_lib import training as tr
from train_resnet3d_lib.data.patch_index_cache import (
    extract_infer_patch_coordinates_cached,
)
from train_resnet3d_lib.data.normalization_stats import (
    prepare_run_fold_normalization_stats,
)
from train_resnet3d_lib.data.datasets_runtime import LazyZarrXyOnlyDataset, build_eval_loader
from train_resnet3d_lib.data.augmentations import get_transforms
from train_resnet3d_lib.data.patching import _mask_component_bboxes_downsample
from train_resnet3d_lib.data.image_readers import (
    get_segment_layer_range as _segment_layer_range,
    get_segment_meta as _segment_meta,
    get_segment_reverse_layers as _segment_reverse_layers,
    read_fragment_mask_for_shape,
)
from train_resnet3d_lib.data.zarr_volume import ZarrSegmentVolume


__all__ = ["parse_args", "main"]


class StitchValidationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        out_h = int(CFG.size) // 4
        out_w = int(CFG.size) // 4
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"invalid resized label shape from CFG.size={CFG.size!r}")
        self._dummy_label = torch.zeros((1, out_h, out_w), dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, xy = self.base_dataset[idx]
        return image, self._dummy_label.clone(), xy, 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--init_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        default=None,
        help="Load model state directly from a Lightning .ckpt during validate.",
    )
    parser.add_argument(
        "--checkpoint_group",
        action="append",
        default=None,
        help=(
            "Repeatable group spec '<checkpoint_dir>:<epoch_csv>' for multi-checkpoint runs, "
            "e.g. --checkpoint_group /path/runA/checkpoints:4,9 --checkpoint_group /path/runB/checkpoints:14,19."
        ),
    )
    parser.add_argument(
        "--checkpoint_run_prefix",
        type=str,
        default="stitch",
        help="Run-name prefix used with --checkpoint_group.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--outputs_path", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision mode. Use --precision auto to derive from metadata use_amp.",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=None,
        help="Override metadata valid batch size for this run only.",
    )
    return parser.parse_args()


def _parse_epoch_csv(epoch_csv):
    if not isinstance(epoch_csv, str):
        raise TypeError(f"epoch csv must be a string, got {type(epoch_csv).__name__}")
    raw_items = [part.strip() for part in epoch_csv.split(",")]
    epochs = []
    for raw_epoch in raw_items:
        if not raw_epoch:
            continue
        if not raw_epoch.isdigit():
            raise ValueError(f"epoch csv contains non-integer value: {raw_epoch!r}")
        epochs.append(int(raw_epoch))
    if not epochs:
        raise ValueError("epoch csv must contain at least one integer epoch")
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"epoch csv contains duplicates: {epochs!r}")
    return epochs


def _checkpoint_run_base_name(checkpoint_dir):
    normalized = checkpoint_dir.rstrip("/\\")
    base = osp.basename(normalized)
    if base == "checkpoints":
        return osp.basename(osp.dirname(normalized))
    return base


def resolve_checkpoint_jobs(args):
    if args.checkpoint_group is None:
        if not args.init_ckpt_path and not args.resume_from_ckpt:
            raise ValueError("provide either --init_ckpt_path or --resume_from_ckpt for stitch inference")
        return [
            {
                "init_ckpt_path": args.init_ckpt_path,
                "resume_from_ckpt": args.resume_from_ckpt,
                "run_name": args.run_name,
                "label": str(args.init_ckpt_path or args.resume_from_ckpt),
            }
        ]

    if args.init_ckpt_path is not None or args.resume_from_ckpt is not None:
        raise ValueError(
            "--checkpoint_group cannot be combined with --init_ckpt_path or --resume_from_ckpt"
        )
    if args.run_name is not None:
        raise ValueError("--run_name cannot be used with --checkpoint_group; use --checkpoint_run_prefix instead")

    prefix = str(args.checkpoint_run_prefix).strip()
    if not prefix:
        raise ValueError("--checkpoint_run_prefix must be non-empty when --checkpoint_group is used")

    jobs = []
    seen_ckpt_paths = set()
    for idx, group_spec in enumerate(args.checkpoint_group):
        if not isinstance(group_spec, str):
            raise TypeError(
                f"--checkpoint_group[{idx}] must be a string, got {type(group_spec).__name__}"
            )
        if ":" not in group_spec:
            raise ValueError(
                f"--checkpoint_group[{idx}] must use format '<checkpoint_dir>:<epoch_csv>', got {group_spec!r}"
            )
        checkpoint_dir_part, epoch_csv = group_spec.rsplit(":", 1)
        checkpoint_dir = checkpoint_dir_part.strip()
        if not checkpoint_dir:
            raise ValueError(
                f"--checkpoint_group[{idx}] has empty checkpoint_dir in spec {group_spec!r}"
            )
        if not osp.isabs(checkpoint_dir):
            checkpoint_dir = osp.join(os.getcwd(), checkpoint_dir)
        if not osp.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"--checkpoint_group[{idx}] checkpoint directory not found: {checkpoint_dir}"
            )

        epochs = _parse_epoch_csv(epoch_csv)
        run_base = _checkpoint_run_base_name(checkpoint_dir)
        if not run_base:
            raise ValueError(
                f"--checkpoint_group[{idx}] resolved empty run base name from checkpoint_dir={checkpoint_dir!r}"
            )
        for epoch in epochs:
            ckpt_path = osp.join(checkpoint_dir, f"epochepoch={epoch}.ckpt")
            if ckpt_path in seen_ckpt_paths:
                raise ValueError(f"duplicate checkpoint in --checkpoint_group specs: {ckpt_path}")
            if not osp.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"--checkpoint_group[{idx}] missing checkpoint for epoch {epoch}: {ckpt_path}"
                )
            seen_ckpt_paths.add(ckpt_path)
            ckpt_stem = osp.splitext(osp.basename(ckpt_path))[0]
            run_name = f"{prefix}_{run_base}_{ckpt_stem}"
            jobs.append(
                {
                    "init_ckpt_path": ckpt_path,
                    "resume_from_ckpt": None,
                    "run_name": run_name,
                    "label": f"{run_base}:{ckpt_stem}",
                }
            )

    if not jobs:
        raise ValueError("no checkpoints resolved from --checkpoint_group")
    return jobs


def resolve_stitch_config(merged_config):
    stitch_cfg = dict(merged_config.get("stitch") or {})
    segment_ids = [str(segment_id).strip() for segment_id in list(stitch_cfg.get("segment_ids") or [])]
    segment_ids = [segment_id for segment_id in segment_ids if segment_id]
    mask_suffix = str(stitch_cfg.get("mask_suffix", "_val"))
    return segment_ids, mask_suffix


def validate_stitch_eval_schedule():
    stitch_every_n_epochs = max(1, int(getattr(CFG, "eval_stitch_every_n_epochs", 1) or 1))
    stitch_every_n_epochs_plus_one = bool(getattr(CFG, "eval_stitch_every_n_epochs_plus_one", False))
    eval_epoch = 1
    if stitch_every_n_epochs_plus_one and stitch_every_n_epochs > 1:
        stitch_metrics_runs_this_validate = False
    else:
        stitch_metrics_runs_this_validate = (eval_epoch % stitch_every_n_epochs) == 0

    if bool(getattr(CFG, "eval_stitch_metrics", False)) and not stitch_metrics_runs_this_validate:
        raise ValueError(
            "stitch metrics are enabled, but one-shot stitch validation runs eval_epoch=1 and "
            "current schedule would skip metrics. Set metadata_json.stitch.eval_every_n_epochs=1 and "
            "metadata_json.stitch.eval_plus_one=false (or disable eval_stitch_metrics)."
        )

    log(
        "stitch eval config "
        f"eval_stitch_metrics={bool(getattr(CFG, 'eval_stitch_metrics', False))} "
        f"eval_stitch_every_n_epochs={stitch_every_n_epochs} "
        f"eval_stitch_every_n_epochs_plus_one={stitch_every_n_epochs_plus_one} "
        f"eval_topological_metrics_every_n_epochs={int(getattr(CFG, 'eval_topological_metrics_every_n_epochs', 1))} "
        f"eval_save_stitch_debug_images={bool(getattr(CFG, 'eval_save_stitch_debug_images', False))} "
        "eval_save_stitch_debug_images_every_n_epochs="
        f"{int(getattr(CFG, 'eval_save_stitch_debug_images_every_n_epochs', 1))}"
    )


def build_stitch_data_state(run_state, *, segment_ids, mask_suffix):
    segments_metadata = run_state["segments_metadata"]
    requested_segment_ids = [str(segment_id) for segment_id in segment_ids]

    if str(getattr(CFG, "data_backend", "zarr")).strip().lower() != "zarr":
        raise ValueError(
            "stitch_train_resnet3d.py requires training.data_backend='zarr' "
            f"(got {getattr(CFG, 'data_backend', None)!r})"
        )

    shared_volume_cache = {}

    valid_transform = get_transforms(data="valid", cfg=CFG)
    val_loaders = []
    val_stitch_shapes = []
    val_stitch_segment_ids = []
    group_idx_by_segment = {}
    total_patch_count = 0
    val_mask_bboxes = {}
    stitch_use_roi = bool(getattr(CFG, "stitch_use_roi", False))
    stitch_downsample = int(getattr(CFG, "stitch_downsample", 1))

    for segment_id in requested_segment_ids:
        seg_meta = _segment_meta(segments_metadata, segment_id)
        layer_range = _segment_layer_range(seg_meta, segment_id)
        reverse_layers = _segment_reverse_layers(seg_meta, segment_id)
        volume = shared_volume_cache.get(segment_id)
        if volume is None:
            volume = ZarrSegmentVolume(
                segment_id,
                seg_meta,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
            )
            shared_volume_cache[segment_id] = volume
        fragment_mask = read_fragment_mask_for_shape(
            segment_id,
            volume.shape[:2],
            mask_suffix=mask_suffix,
        )
        xyxys = extract_infer_patch_coordinates_cached(
            fragment_mask=fragment_mask,
            fragment_id=segment_id,
            mask_suffix=mask_suffix,
            split_name="val",
        )
        patch_count = int(xyxys.shape[0])
        if patch_count <= 0:
            raise ValueError(
                f"segment {segment_id!r} produced zero inference patches from mask_suffix={mask_suffix!r}"
            )
        total_patch_count += patch_count
        log(
            f"stitch dataset segment={segment_id} mask_suffix={mask_suffix!r} "
            f"shape={tuple(fragment_mask.shape)} patches={patch_count}"
        )

        xy_only_dataset = LazyZarrXyOnlyDataset(
            {segment_id: volume},
            {segment_id: xyxys},
            CFG,
            transform=valid_transform,
        )
        val_dataset = StitchValidationDataset(xy_only_dataset)
        val_loader = build_eval_loader(val_dataset)
        loader_batch_size = int(val_loader.batch_size)
        expected_batch_size = int(CFG.valid_batch_size)
        if loader_batch_size != expected_batch_size:
            raise ValueError(
                "stitch loader batch size mismatch: "
                f"loader={loader_batch_size} cfg.valid_batch_size={expected_batch_size}"
            )
        loader_batches = int(len(val_loader))
        log(
            f"stitch loader segment={segment_id} "
            f"batch_size={loader_batch_size} batches={loader_batches}"
        )
        val_loaders.append(val_loader)
        val_stitch_shapes.append(tuple(fragment_mask.shape))
        val_stitch_segment_ids.append(segment_id)
        group_idx_by_segment[segment_id] = 0

        if stitch_use_roi:
            bboxes = _mask_component_bboxes_downsample(
                fragment_mask,
                stitch_downsample,
            )
            if int(bboxes.shape[0]) > 0:
                val_mask_bboxes[segment_id] = bboxes

    if total_patch_count <= 0:
        raise ValueError("no stitch patches were produced")
    if not val_loaders:
        raise ValueError("no stitch validation loaders were built")

    total_steps = sum(int(len(loader)) for loader in val_loaders)
    if total_steps <= 0:
        total_steps = 1

    return {
        "train_loader": None,
        "val_loaders": val_loaders,
        "group_names": ["stitch"],
        "group_idx_by_segment": group_idx_by_segment,
        "train_group_counts": [total_patch_count],
        "steps_per_epoch": total_steps,
        "train_stitch_loaders": [],
        "train_stitch_shapes": [],
        "train_stitch_segment_ids": [],
        "train_mask_borders": {},
        "train_mask_bboxes": {},
        "val_mask_borders": {},
        "val_mask_bboxes": val_mask_bboxes,
        "log_only_stitch_loaders": [],
        "log_only_stitch_shapes": [],
        "log_only_stitch_segment_ids": [],
        "log_only_mask_bboxes": {},
        "include_train_xyxys": False,
        "stitch_val_dataloader_idx": 0,
        "stitch_pred_shape": val_stitch_shapes[0],
        "stitch_segment_id": val_stitch_segment_ids[0],
        "val_stitch_shapes": val_stitch_shapes,
        "val_stitch_segment_ids": val_stitch_segment_ids,
        "shared_volume_cache": shared_volume_cache,
    }

def main():
    args = parse_args()
    checkpoint_jobs = resolve_checkpoint_jobs(args)

    log(f"start pid={os.getpid()} cwd={os.getcwd()}")
    log(
        "args "
        f"metadata_json={args.metadata_json!r} "
        f"outputs_path={args.outputs_path!r} devices={args.devices} accelerator={args.accelerator!r} "
        f"precision={args.precision!r} run_name={args.run_name!r} valid_batch_size={args.valid_batch_size!r} "
        f"init_ckpt_path={args.init_ckpt_path!r} resume_from_ckpt={args.resume_from_ckpt!r} "
        f"checkpoint_group={args.checkpoint_group!r}"
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    log(
        f"torch cuda_available={cuda_available} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"device_count={device_count}"
    )

    base_config, preinit_overrides = orchestration.load_base_config_and_preinit(
        metadata_json=args.metadata_json,
        base_dir=osp.dirname(orchestration.__file__),
    )

    data_state = None
    expected_segment_ids = None
    expected_mask_suffix = None
    total_jobs = int(len(checkpoint_jobs))
    log(f"resolved checkpoint jobs={total_jobs}")

    for job_idx, job in enumerate(checkpoint_jobs, start=1):
        run_args = argparse.Namespace(**copy.deepcopy(vars(args)))
        run_args.init_ckpt_path = job["init_ckpt_path"]
        run_args.resume_from_ckpt = job["resume_from_ckpt"]
        run_args.run_name = job["run_name"]

        log(
            f"starting stitch run {job_idx}/{total_jobs} "
            f"label={job['label']!r} run_name={run_args.run_name!r} "
            f"init_ckpt_path={run_args.init_ckpt_path!r} resume_from_ckpt={run_args.resume_from_ckpt!r}"
        )
        wandb_logger, merged_config = orchestration.prepare_wandb_and_merged_config(
            run_args,
            base_config,
            preinit_overrides=preinit_overrides,
        )
        if run_args.valid_batch_size is not None:
            valid_batch_size = int(run_args.valid_batch_size)
            if valid_batch_size <= 0:
                raise ValueError(f"--valid_batch_size must be > 0, got {valid_batch_size}")
            CFG.valid_batch_size = valid_batch_size
            log(f"override valid_batch_size={CFG.valid_batch_size}")

        segment_ids, mask_suffix = resolve_stitch_config(merged_config)
        if not segment_ids:
            raise ValueError("metadata_json.stitch.segment_ids must contain at least one segment for stitch runs")
        if run_args.run_name is None:
            if len(segment_ids) == 1:
                run_args.run_name = f"stitch_{segment_ids[0]}"
            else:
                run_args.run_name = f"stitch_{len(segment_ids)}segments"
        cfg_val_mask_suffix = str(getattr(CFG, "val_mask_suffix", ""))
        resolved_mask_suffix = str(mask_suffix)
        if cfg_val_mask_suffix != resolved_mask_suffix:
            CFG.val_mask_suffix = resolved_mask_suffix
            log(f"override val_mask_suffix={CFG.val_mask_suffix!r} for stitch validation/metrics")

        normalized_segment_ids = tuple(str(segment_id) for segment_id in segment_ids)
        normalized_mask_suffix = str(mask_suffix)
        if expected_segment_ids is None:
            expected_segment_ids = normalized_segment_ids
            expected_mask_suffix = normalized_mask_suffix
        else:
            if normalized_segment_ids != expected_segment_ids:
                raise ValueError(
                    "all checkpoint runs must use the same stitch segment_ids when reusing loaders; "
                    f"expected={expected_segment_ids!r} got={normalized_segment_ids!r}"
                )
            if normalized_mask_suffix != expected_mask_suffix:
                raise ValueError(
                    "all checkpoint runs must use the same stitch mask_suffix when reusing loaders; "
                    f"expected={expected_mask_suffix!r} got={normalized_mask_suffix!r}"
                )

        validate_stitch_eval_schedule()
        CFG.stitch_all_val = True
        run_state = orchestration.prepare_run(run_args, merged_config, wandb_logger)
        if data_state is None:
            data_state = build_stitch_data_state(
                run_state,
                segment_ids=segment_ids,
                mask_suffix=mask_suffix,
            )
            log("stitch data state initialized once and will be reused for remaining checkpoints")
        prepare_run_fold_normalization_stats(
            run_state=run_state,
            data_backend="zarr",
            volume_cache=data_state["shared_volume_cache"],
        )
        model = tr.build_model(run_state, data_state, wandb_logger)
        trainer = tr.build_trainer(run_args, wandb_logger)
        tr.validate(trainer, model, data_state, run_state)


if __name__ == "__main__":
    main()
