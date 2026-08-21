#!/usr/bin/env python3
"""Rewrite a legacy Spiral checkpoint for the current VC3D Spiral workspace.

The source checkpoint is never modified.  Only embedded configuration
dictionaries are migrated; model, optimiser, scheduler, and RNG state are
carried through unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from checkpoint_io import load_checkpoint_cpu
from config import Config


MODEL_KEYS = (
    "num_flow_integration_steps",
    "flow_integration_solver",
    "num_flow_timesteps",
    "num_flow_stages",
    "flow_bounds_z_margin",
    "flow_bounds_radius",
    "flow_voxel_resolution",
    "flow_field_type",
    "gap_expander_logit_resolution",
    "gap_expander_num_windings",
    "gap_expander_lr_scale",
    "linear_z_resolution",
    "initial_dr_per_winding",
)

LEGACY_RENAMES = {
    "random_seed": "optimizer_random_seed",
    "distributed_split_batch": "optimizer_distributed_split_batch",
    "learning_rate": "optimizer_learning_rate",
    "exp_lr_schedule": "optimizer_exp_lr_schedule",
    "lr_final_factor": "optimizer_lr_final_factor",
    "num_training_steps": "optimizer_num_training_steps",
    **{key: f"model_{key}" for key in MODEL_KEYS},
    "num_patches_per_step": "sample_count_patches_per_step",
    "num_patches_per_step_for_dt": "sample_count_patches_per_step_for_dt",
    "num_points_per_patch": "sample_count_points_per_patch",
    "unverified_num_patches_per_step":
        "sample_count_unverified_patches_per_step",
    "unverified_num_patches_per_step_for_dt":
        "sample_count_unverified_patches_per_step_for_dt",
    "unverified_num_points_per_patch":
        "sample_count_unverified_points_per_patch",
    "rel_winding_num_pcls": "sample_count_relative_winding_pcls",
    "rel_winding_num_patch_pairs_per_pcl":
        "sample_count_relative_winding_patch_pairs_per_pcl",
    "abs_winding_num_pcls": "sample_count_absolute_winding_pcls",
    "abs_winding_num_points_per_pcl":
        "sample_count_absolute_winding_points_per_pcl",
    "unattached_pcl_num_per_step":
        "sample_count_unattached_pcls_per_step",
    "unattached_pcl_num_points_per_step":
        "sample_count_unattached_pcl_points_per_step",
    "track_num_per_step": "sample_count_tracks_per_step",
    "track_num_points_per_step": "sample_count_track_points_per_step",
    "dense_normals_num_points": "sample_count_dense_normal_points",
    "regularisation_num_points": "sample_count_regularisation_points",
    "dense_spacing_num_pairs": "sample_count_dense_spacing_pairs",
    "dense_spacing_count_extra_pairs":
        "sample_count_dense_spacing_count_extra_pairs",
    "dense_spacing_density_extra_pairs":
        "sample_count_dense_spacing_density_extra_pairs",
    "dense_spacing_density_chunk_pairs":
        "sample_count_dense_spacing_density_chunk_pairs",
    "min_spacing_independent_samples":
        "sample_count_minimum_spacing_independent_samples",
    "dense_attachment_num_points":
        "sample_count_dense_attachment_points",
    "patch_dt_target_num_points": "sample_count_patch_dt_target_points",
    "dt_target_num_points_per_strip":
        "sample_count_dt_target_points_per_strip",
    "shell_num_samples": "sample_count_shell_samples",
    "interactive_influence_footprint_points":
        "sample_count_influence_footprint_points",
    "interactive_influence_anchor_lattice_points":
        "sample_count_influence_anchor_lattice_points",
    "interactive_influence_anchor_geometry_points":
        "sample_count_influence_anchor_geometry_points",
    "interactive_influence_anchor_samples_per_step":
        "sample_count_influence_anchor_samples_per_step",
    "erode_patches": "patch_erode_patches",
    "disable_patches": "input_disable_patches",
    "use_verified_patches": "input_use_verified_patches",
    "use_unverified_patches": "input_use_unverified_patches",
    "use_normals": "input_use_normals",
    "use_surf_sdt": "input_use_surf_sdt",
    "use_tracks": "input_use_tracks",
    "use_gradient_magnitude": "input_use_gradient_magnitude",
    "use_fibers": "input_use_fibers",
    "unverified_patch_radius_loss_margin":
        "patch_unverified_patch_radius_loss_margin",
    "unverified_patch_radius_loss_inv":
        "patch_unverified_patch_radius_loss_inv",
    "unverified_patch_radius_within_norm_p":
        "patch_unverified_patch_radius_within_norm_p",
    "unverified_patch_dt_norm_p": "patch_unverified_patch_dt_norm_p",
    "unverified_patch_dt_within_patch_norm_p":
        "patch_unverified_patch_dt_within_patch_norm_p",
    "unverified_patch_dt_loss_margin":
        "patch_unverified_patch_dt_loss_margin",
    "unverified_patch_exclusion_radius":
        "patch_unverified_patch_exclusion_radius",
    "rel_winding_adjacent_patches_only":
        "pcl_rel_winding_adjacent_patches_only",
    "stratified_pcl_sampling": "pcl_stratified_pcl_sampling",
    "fiber_min_point_spacing": "pcl_fiber_min_point_spacing",
    "unattached_pcl_min_point_spacing":
        "pcl_unattached_pcl_min_point_spacing",
    "max_track_crossing_per_step": "track_max_track_crossing_per_step",
    "min_walk_steps_per_track": "track_min_walk_steps_per_track",
    "max_walk_steps_per_track": "track_max_walk_steps_per_track",
    "n_walks_per_track": "track_max_walks_per_track",
    "grad_mag_encode_scale": "dense_grad_mag_encode_scale",
    "grad_mag_factor": "dense_grad_mag_factor",
    "spacing_integration_steps": "dense_spacing_integration_steps",
    "min_spacing_d_min_wv": "dense_min_spacing_d_min_wv",
    "sym_dirichlet_finite_difference_epsilon":
        "model_sym_dirichlet_finite_difference_epsilon",
    "weight_decay_gap_expander": "optimizer_weight_decay_gap_expander",
    "weight_decay_flow_field": "optimizer_weight_decay_flow_field",
    "save_png_visualizations": "output_save_png_visualizations",
    "interactive_influence_enabled": "influence_enabled",
    "interactive_influence_z": "influence_z",
    "interactive_influence_windings": "influence_windings",
    "interactive_influence_theta_frac": "influence_theta_frac",
    "interactive_influence_disable_dt_frac": "influence_disable_dt_frac",
    "interactive_influence_sigma": "influence_sigma",
    "interactive_influence_anchor_ramp_power":
        "influence_anchor_ramp_power",
}

# These sampling settings moved out of the durable config, and the track-walk
# setting was replaced by current controls.
LEGACY_REMOVED = {
    "patch_strip_sampling",
    "patch_2d_sampling_max_area",
    "track_walk_require_loop_consistency",
}

CONFIG_FIELDS = ("cfg", "requested_config", "resolved_config")


def migrate_config(source: Mapping) -> tuple[dict, list[str], list[str], list[str]]:
    """Return a current-schema config and a summary of the migration."""
    defaults = Config().as_dict()
    migrated = dict(defaults)
    renamed = []
    removed = []
    consumed = set()

    for old_key, value in source.items():
        if old_key in defaults:
            migrated[old_key] = value
            consumed.add(old_key)
        elif old_key in LEGACY_RENAMES:
            new_key = LEGACY_RENAMES[old_key]
            migrated[new_key] = value
            renamed.append(f"{old_key} -> {new_key}")
            consumed.add(old_key)
        elif old_key in LEGACY_REMOVED:
            removed.append(old_key)
            consumed.add(old_key)

    unknown = sorted(set(source) - consumed)
    if unknown:
        raise ValueError(
            "checkpoint contains configuration keys with no known migration: "
            + ", ".join(unknown)
        )

    # Validate types, ranges, enum values, and the exact current key set.
    validated = Config(migrated).as_dict()
    added = sorted(set(defaults) - {
        LEGACY_RENAMES.get(key, key)
        for key in source
        if key not in LEGACY_REMOVED
    })
    return validated, sorted(renamed), sorted(removed), added


def update_checkpoint(checkpoint: dict) -> tuple[dict, dict]:
    """Migrate all embedded config snapshots without touching tensor state."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    if "spiral_and_transform" not in checkpoint:
        raise ValueError("checkpoint has no 'spiral_and_transform' model state")

    updated = dict(checkpoint)
    reports = {}
    fallback = checkpoint.get("cfg")
    if not isinstance(fallback, Mapping):
        raise ValueError("checkpoint has no 'cfg' configuration dictionary")

    for field in CONFIG_FIELDS:
        source = checkpoint.get(field, fallback)
        if not isinstance(source, Mapping):
            raise ValueError(f"checkpoint field {field!r} is not a dictionary")
        migrated, renamed, removed, added = migrate_config(source)
        updated[field] = migrated
        reports[field] = {
            "renamed": renamed,
            "removed": removed,
            "added_from_current_defaults": added,
        }
    return updated, reports


def default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_updated{source.suffix}")


def save_atomic(checkpoint: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a pre-VC3D Spiral checkpoint's embedded configuration "
            "to the schema required by this workspace."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="source .ckpt file")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="destination (default: <source_stem>_updated.ckpt)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination (the source is never overwritten)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report changes without writing a checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.checkpoint.expanduser().resolve(strict=True)
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None else default_output_path(source)
    )
    if destination == source:
        raise ValueError("refusing to overwrite the source checkpoint")
    if destination.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"destination already exists: {destination}; pass --overwrite")

    print(f"[spiral] loading {source}", flush=True)
    checkpoint = load_checkpoint_cpu(source)
    updated, reports = update_checkpoint(checkpoint)
    report = reports["cfg"]
    print(
        f"[spiral] cfg: {len(report['renamed'])} renamed, "
        f"{len(report['removed'])} removed, "
        f"{len(report['added_from_current_defaults'])} added from current defaults",
        flush=True,
    )

    if args.dry_run:
        print("[spiral] dry run complete; no file written")
        return 0

    print(f"[spiral] writing {destination}", flush=True)
    save_atomic(updated, destination)

    # Reload through the same mmap path used by the resident fitter and verify
    # the exact invariant that VC3D checks before starting a session.
    del checkpoint, updated
    reloaded = load_checkpoint_cpu(destination)
    expected_keys = set(Config().as_dict())
    if not isinstance(reloaded, dict) or set(reloaded.get("cfg", {})) != expected_keys:
        raise RuntimeError("written checkpoint failed current-schema validation")
    print(f"[spiral] updated checkpoint: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"update_checkpoint.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
