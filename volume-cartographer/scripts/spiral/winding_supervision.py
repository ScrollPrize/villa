"""GPU-resident relative-winding supervision exported from neural phase rays."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from loss_maps import record_loss_samples
from sample_spiral import get_theta_and_radii, unwrap_shifted_radii


ARTIFACT_TYPE = "winding_inference_crossings"
FORMAT_VERSION = 1


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_array(root: Path, description: dict, *, verify: bool) -> np.ndarray:
    path = root / description["file"]
    if verify and _sha256(path) != description["sha256"]:
        raise ValueError(f"winding-inference array checksum mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(description["shape"]):
        raise ValueError(f"winding-inference array shape mismatch: {path}")
    if np.dtype(value.dtype).str != str(description["dtype"]):
        raise ValueError(f"winding-inference array dtype mismatch: {path}")
    return value


class WindingInferenceStore:
    """Flat ragged crossing rays copied once to the fitting device."""

    def __init__(self, path, device, *, verify=True, z_range=None):
        self.path = str(Path(path).resolve())
        root = Path(self.path)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"winding-inference manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(f"not a winding-inference crossing store: {root}")
        if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(
                f"unsupported winding-inference format version: "
                f"{manifest.get('format_version')!r}")
        if manifest.get("coordinate_order") != "zyx":
            raise ValueError("winding-inference coordinates must use zyx order")
        identity_view = copy.deepcopy(manifest)
        claimed_fingerprint = identity_view.pop("fingerprint", None)
        identity_view.pop("elapsed_seconds", None)
        identity_view.pop("export_workers", None)
        identity_view.pop("rays_per_task", None)
        for shard in identity_view.get("shards", []):
            shard.pop("elapsed_seconds", None)
        if claimed_fingerprint != _canonical_digest(identity_view):
            raise ValueError("winding-inference manifest fingerprint mismatch")

        origins = []
        steps = []
        crossing_t = []
        crossing_level = []
        offsets = [np.array([0], dtype=np.int64)]
        crossing_base = 0
        for shard in manifest["shards"]:
            shard_root = root / shard["name"]
            arrays = shard["arrays"]
            origins.append(_load_array(
                shard_root, arrays["ray_origin_zyx"], verify=verify))
            steps.append(_load_array(
                shard_root, arrays["ray_step_zyx"], verify=verify))
            crossing_t.append(_load_array(
                shard_root, arrays["crossing_t"], verify=verify))
            crossing_level.append(_load_array(
                shard_root, arrays["crossing_level"], verify=verify))
            seed_winding = _load_array(
                shard_root, arrays["seed_winding"], verify=verify)
            if len(seed_winding) != len(origins[-1]):
                raise ValueError(
                    f"winding-inference seed/ray counts disagree: {shard_root}")
            local_offsets = _load_array(
                shard_root, arrays["crossing_offsets"], verify=verify)
            offsets.append(np.asarray(local_offsets[1:], dtype=np.int64)
                           + crossing_base)
            crossing_base += int(local_offsets[-1])

        def concatenate(items, shape, dtype):
            if not items:
                return np.empty(shape, dtype=dtype)
            return np.concatenate(items).astype(dtype, copy=False)

        origin_np = concatenate(origins, (0, 3), np.float32)
        step_np = concatenate(steps, (0, 3), np.float32)
        t_np = concatenate(crossing_t, (0,), np.float32)
        level_np = concatenate(crossing_level, (0,), np.int16)
        offset_np = np.concatenate(offsets)
        if len(offset_np) != len(origin_np) + 1:
            raise ValueError("winding-inference ray/offset counts disagree")
        if int(offset_np[-1]) != len(t_np) or len(t_np) != len(level_np):
            raise ValueError("winding-inference crossing arrays disagree")

        device = torch.device(device)
        self.origin = torch.from_numpy(origin_np).to(device)
        self.step = torch.from_numpy(step_np).to(device)
        self.offset = torch.from_numpy(offset_np).to(device)
        self.crossing_t = torch.from_numpy(t_np).to(device)
        self.crossing_level = torch.from_numpy(level_np.astype(
            np.int32, copy=False)).to(device)
        self.length = self.offset[1:] - self.offset[:-1]
        # A pair is valid only when both crossings land in [z_begin, z_end).
        # Crossings are stored in ascending t and z is linear in t, so a ray
        # whose first/last-crossing z interval misses the slab can never yield
        # a valid pair; drop it from the samplers instead of spending
        # transform evaluations on pairs the loss mask will zero.
        if z_range is not None and len(self.crossing_t):
            z_begin, z_end = float(z_range[0]), float(z_range[1])
            first_z = (self.origin[:, 0]
                       + self.crossing_t[self.offset[:-1]] * self.step[:, 0])
            last_z = (self.origin[:, 0]
                      + self.crossing_t[(self.offset[1:] - 1).clamp(min=0)]
                      * self.step[:, 0])
            self._z_eligible = (
                (torch.maximum(first_z, last_z) >= z_begin)
                & (torch.minimum(first_z, last_z) < z_end))
        else:
            self._z_eligible = torch.ones(
                len(self.length), dtype=torch.bool, device=device)
        self.density_rays = torch.nonzero(
            (self.length >= 2) & self._z_eligible,
            as_tuple=False).squeeze(-1)
        self._relative_rays = {}
        self.manifest = manifest
        self.fingerprint = {
            "artifact_type": ARTIFACT_TYPE,
            "format_version": FORMAT_VERSION,
            "fingerprint": str(manifest["fingerprint"]),
            "num_rays": int(manifest["num_rays"]),
            "num_crossings": int(manifest["num_crossings"]),
        }

    @property
    def device(self):
        return self.origin.device

    @property
    def num_z_eligible_rays(self):
        return int(self._z_eligible.sum().item())

    def _choose(self, values, count, *, generator=None):
        count = int(count)
        if count <= 0 or not len(values):
            return values.new_empty((0,))
        choices = torch.randint(
            len(values), (count,), device=self.device, generator=generator)
        return values[choices]

    def sample_adjacent(self, count, *, generator=None):
        ray = self._choose(self.density_rays, count, generator=generator)
        if not len(ray):
            return self._empty_samples()
        length = self.length[ray]
        start = torch.floor(torch.rand(
            len(ray), device=self.device, generator=generator
        ) * (length - 1).to(torch.float32)).to(torch.int64)
        return self._materialize(ray, start, start + 1)

    def sample_relative(
        self, count, min_delta, max_delta, *, generator=None,
    ):
        min_delta, max_delta = int(min_delta), int(max_delta)
        if min_delta < 1 or max_delta < min_delta:
            raise ValueError("relative pair delta must satisfy 1 <= min <= max")
        eligible = self._relative_rays.get(min_delta)
        if eligible is None:
            eligible = torch.nonzero(
                (self.length >= min_delta + 1) & self._z_eligible,
                as_tuple=False).squeeze(-1)
            self._relative_rays[min_delta] = eligible
        ray = self._choose(eligible, count, generator=generator)
        if not len(ray):
            return self._empty_samples()
        length = self.length[ray]
        maximum = torch.minimum(
            length - 1, torch.full_like(length, max_delta))
        separation = min_delta + torch.floor(torch.rand(
            len(ray), device=self.device, generator=generator
        ) * (maximum - min_delta + 1).to(torch.float32)).to(torch.int64)
        start = torch.floor(torch.rand(
            len(ray), device=self.device, generator=generator
        ) * (length - separation).to(torch.float32)).to(torch.int64)
        return self._materialize(ray, start, start + separation)

    def _empty_samples(self):
        return {
            "points": self.origin.new_empty((0, 2, 3)),
            "target": self.origin.new_empty((0,)),
        }

    def materialize_flat(self, flat_indices):
        """Materialize arbitrary global crossing rows as ZYX coordinates."""
        flat = torch.as_tensor(
            flat_indices, dtype=torch.int64, device=self.device)
        if not flat.numel():
            return self.origin.new_empty((*flat.shape, 3))
        if flat.min().item() < 0 or flat.max().item() >= len(self.crossing_t):
            raise IndexError("winding-inference crossing index is out of range")
        ray = torch.searchsorted(self.offset[1:], flat, right=True)
        return (
            self.origin[ray]
            + self.crossing_t[flat, None] * self.step[ray]
        )

    def _materialize(self, ray, first, second):
        first_flat = self.offset[ray] + first
        second_flat = self.offset[ray] + second
        flat = torch.stack([first_flat, second_flat], dim=-1)
        t = self.crossing_t[flat]
        points = self.origin[ray, None, :] + t[..., None] * self.step[ray, None, :]
        levels = self.crossing_level[flat]
        return {
            "points": points,
            "target": (levels[:, 1] - levels[:, 0]).to(torch.float32),
        }


def load_winding_inference_store(path, device, *, verify=True, z_range=None):
    return WindingInferenceStore(path, device, verify=verify, z_range=z_range)


def _component_residual(
    spiral_pairs, sample_pairs, target, dr_per_winding, z_begin, z_end,
):
    theta, _radius, shifted = get_theta_and_radii(
        spiral_pairs[..., 1:], dr_per_winding)
    # A two-point unwrap is exact even for large winding separations: the
    # shifted radius is non-modular (it grows by dr_per_winding every
    # winding), so the only ambiguity unwrap_shifted_radii resolves here is
    # the single possible theta=0 wrap between the pair's two points.
    unwrapped, _adjustments = unwrap_shifted_radii(
        theta, shifted, dr_per_winding, dim=-1)
    predicted = (unwrapped[:, 1] - unwrapped[:, 0]) / dr_per_winding
    valid = (
        (sample_pairs[..., 0] >= float(z_begin))
        & (sample_pairs[..., 0] < float(z_end))
        & torch.isfinite(sample_pairs).all(dim=-1)
        & torch.isfinite(spiral_pairs).all(dim=-1)
    ).all(dim=-1) & torch.isfinite(predicted) & (target > 0)
    return predicted - target, valid


def _metrics(name, residual, valid):
    values = residual.detach().abs()[valid]
    result = {f"{name}_valid_fraction": float(valid.float().mean().item())
              if len(valid) else 0.0}
    if len(values):
        quantiles = torch.quantile(
            values.float(), torch.tensor([0.5, 0.95], device=values.device))
        result.update({
            f"{name}_residual_abs_mean": float(values.float().mean().item()),
            f"{name}_residual_abs_p50": float(quantiles[0].item()),
            f"{name}_residual_abs_p95": float(quantiles[1].item()),
        })
    return result


def get_winding_inference_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    store,
    cfg,
    z_begin,
    z_end,
    *,
    generator=None,
):
    """Sample both inference components and evaluate one shared transform."""
    relative_count = (int(cfg["sample_count_winding_model_relative_pairs"])
                      if cfg["loss_weight_dense_spacing"] > 0 else 0)
    density_count = (int(cfg["sample_count_winding_model_density_pairs"])
                     if cfg["loss_weight_dense_spacing_density"] > 0 else 0)
    pair_delta = cfg["winding_model_relative_pair_delta"]
    relative = store.sample_relative(
        relative_count, pair_delta[0], pair_delta[1], generator=generator)
    density = store.sample_adjacent(density_count, generator=generator)
    counts = (len(relative["target"]), len(density["target"]))
    all_points = torch.cat([relative["points"], density["points"]], dim=0)
    all_targets = torch.cat([relative["target"], density["target"]], dim=0)
    zero = dr_per_winding.new_zeros(())
    if not len(all_targets):
        return {
            "dense_spacing_winding_model_relative": zero,
            "dense_spacing_winding_model_density": zero,
        }, {}

    spiral = slice_to_spiral_transform(all_points.reshape(-1, 3)).reshape(-1, 2, 3)
    losses = {}
    metrics = {}
    cursor = 0
    for name, count in zip((
        "dense_spacing_winding_model_relative",
        "dense_spacing_winding_model_density",
    ), counts):
        component_spiral = spiral[cursor : cursor + count]
        component_points = all_points[cursor : cursor + count]
        component_target = all_targets[cursor : cursor + count]
        cursor += count
        residual, valid = _component_residual(
            component_spiral, component_points, component_target,
            dr_per_winding, z_begin, z_end)
        per_pair = F.huber_loss(
            residual, torch.zeros_like(residual), reduction="none",
            delta=float(cfg["winding_model_huber_delta"]),
        )
        valid_f = valid.to(per_pair.dtype)
        losses[name] = (per_pair * valid_f).sum() / valid_f.sum().clamp(min=1)
        metrics.update(_metrics(name, residual, valid))
        if count:
            record_loss_samples(
                name, component_spiral.mean(dim=1), residual.detach().abs(), valid)
    return losses, metrics


def get_winding_model_patch_relative_loss(
    slice_to_spiral_transform,
    dr_per_winding,
    store,
    patches_dict,
    patch_atlas,
    *,
    crossing_map,
    cfg,
    z_begin,
    z_end,
):
    """Propagate model-relative assignments across their attached patches."""
    from losses import get_patch_relative_pair_loss

    pair_requests = store.sample_pair_requests(cfg)
    return get_patch_relative_pair_loss(
        slice_to_spiral_transform,
        dr_per_winding,
        patches_dict,
        patch_atlas,
        pair_requests,
        crossing_map=crossing_map,
        cfg=cfg,
        z_begin=z_begin,
        z_end=z_end,
        loss_name="winding_model_patch_relative",
    )
