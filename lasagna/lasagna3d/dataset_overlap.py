"""Pairwise surface-overlap diagnostics for TifxyzLasagnaDataset.

For each training patch, this command finds candidate surface pairs
(consecutive in chain ordering, from different source segments) and
samples signed distances between them using the distance transform of
one surface and the normals of the other. Per patch, it reports the
pair with the lowest 1% percentile (i.e. the worst intersection).

The signed distance for a point P on surface A against surface B is:

    d_signed = |P - nearest_B(P)| * sign((P - nearest_B(P)) . n_A(P))

where `n_A(P)` is A's own surface normal at P. Intersection shows up
as negative samples in the lower tail.

Stats are always reported with the median forced positive — if the raw
median is negative, samples are negated before computing stats. The
resulting `sign_flipped` flag records whether that happened.

Results are written as one JSON line per patch to `--out`.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure lasagna/ dir is on sys.path so we can import sibling modules.
_LASAGNA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)


TAG = "[lasagna3d dataset overlap]"

_PCT_LABELS = ("0.1", "1", "5", "15", "25", "50", "75", "85", "95", "99", "99.9")
_PCT_VALUES = tuple(float(x) for x in _PCT_LABELS)
_TOPK_SUMMARY = 20


def _dataset_display_name(dataset_cfg: dict, fallback_idx: int) -> str:
    segments_path = dataset_cfg.get("segments_path")
    if segments_path:
        parent = Path(segments_path).parent.name
        if parent:
            return parent
    volume_path = dataset_cfg.get("volume_path")
    if volume_path:
        return Path(str(volume_path).rstrip("/")).name or f"dataset{fallback_idx}"
    return f"dataset{fallback_idx}"


def _edt_with_indices_torch(mask_not_b_uint8):
    """CuPy EDT + feature transform, returned as torch CUDA tensors.

    Input: ``mask_not_b_uint8`` — a (Z, Y, X) uint8 tensor on CUDA,
    with 1 where B is absent and 0 on B's surface.

    Returns:
        dist: (Z, Y, X) float32 torch CUDA tensor.
        inds: (3, Z, Y, X) int64 torch CUDA tensor — nearest-B voxel
            index for every voxel.

    CuPy has no in-place torch path, so we round-trip via DLPack —
    no host copy. ``edt_torch`` in ``tifxyz_labels`` does the same
    for the distances-only case; this helper adds the feature
    transform.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    cp_mask = cp.from_dlpack(mask_not_b_uint8.contiguous())
    dist_cp, inds_cp = cupy_edt(
        cp_mask, return_distances=True, return_indices=True,
    )
    dist = torch.from_dlpack(dist_cp).to(torch.float32)
    # cupy EDT returns int32 indices; upcast to int64 so torch's
    # advanced indexing is happy (and matches long() elsewhere).
    inds = torch.from_dlpack(inds_cp).to(torch.int64)
    return dist, inds


def _signed_distance_samples_gpu(
    points_a_zyx: torch.Tensor,   # (M, 3) float32 CUDA
    normals_a_zyx: torch.Tensor,  # (M, 3) float32 CUDA
    mask_b: torch.Tensor,         # (Z, Y, X) bool   CUDA
) -> torch.Tensor:
    """Return (M,) float32 CUDA tensor of signed distances from points on A to B.

    Math is identical to the previous numpy version; only the device
    changed. All gathers and arithmetic stay on CUDA.
    """
    Z, Y, X = mask_b.shape
    if points_a_zyx.shape[0] == 0 or not bool(mask_b.any()):
        return torch.empty(0, dtype=torch.float32, device=mask_b.device)

    mask_not_b = (~mask_b).to(torch.uint8)
    dist_b, inds_b = _edt_with_indices_torch(mask_not_b)

    pz = points_a_zyx[:, 0].floor().clamp(0, Z - 1).long()
    py = points_a_zyx[:, 1].floor().clamp(0, Y - 1).long()
    px = points_a_zyx[:, 2].floor().clamp(0, X - 1).long()

    d = dist_b[pz, py, px]  # (M,) quantized voxel-to-voxel distance

    nz = inds_b[0, pz, py, px].to(torch.float32)
    ny = inds_b[1, pz, py, px].to(torch.float32)
    nx = inds_b[2, pz, py, px].to(torch.float32)
    nearest = torch.stack([nz, ny, nx], dim=-1)

    v = points_a_zyx.to(torch.float32) - nearest
    dot = (v * normals_a_zyx.to(torch.float32)).sum(dim=-1)
    sign = torch.where(
        dot >= 0.0,
        torch.ones((), dtype=torch.float32, device=dot.device),
        -torch.ones((), dtype=torch.float32, device=dot.device),
    )
    return (d.to(torch.float32) * sign).to(torch.float32)


# Cached quantile tensor keyed on device so we don't reallocate.
_QUANTILE_CACHE: dict = {}


def _quantile_vec(device: torch.device) -> torch.Tensor:
    q = _QUANTILE_CACHE.get(device)
    if q is None:
        q = torch.tensor(
            [v / 100.0 for v in _PCT_VALUES],
            device=device, dtype=torch.float32,
        )
        _QUANTILE_CACHE[device] = q
    return q


def _compute_stats_gpu(signed: torch.Tensor) -> dict:
    """Compute min/max/percentiles on CUDA, flipping sign if median is negative.

    Pulls only ~13 scalars back to host (min, max, sign-flip flag,
    and the 11 percentiles). The raw per-point tensor stays on GPU.
    """
    if signed.numel() == 0:
        return {
            "n_samples": 0,
            "min": 0.0, "max": 0.0,
            "percentiles": {label: 0.0 for label in _PCT_LABELS},
            "sign_flipped": False,
        }

    q = _quantile_vec(signed.device)
    pct = torch.quantile(signed, q)              # (P,) float
    smin = signed.min()
    smax = signed.max()
    median = pct[_PCT_LABELS.index("50")]
    flipped_t = median < 0.0
    # Negate+flip so post-flip percentiles keep the same ordering.
    pct_flipped = -pct.flip(0)
    pct_out = torch.where(flipped_t, pct_flipped, pct)
    smin_out = torch.where(flipped_t, -smax, smin)
    smax_out = torch.where(flipped_t, -smin, smax)

    pct_vals = pct_out.detach().cpu().tolist()
    return {
        "n_samples": int(signed.numel()),
        "min": float(smin_out.item()),
        "max": float(smax_out.item()),
        "percentiles": {
            label: float(pct_vals[i]) for i, label in enumerate(_PCT_LABELS)
        },
        "sign_flipped": bool(flipped_t.item()),
    }


def _candidate_pairs(info: list[dict], seg_idx: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(info)
    for i in range(n):
        for j in range(i + 1, n):
            if info[i].get("chain", -1) != info[j].get("chain", -2):
                continue
            if abs(int(info[i]["pos"]) - int(info[j]["pos"])) != 1:
                continue
            if seg_idx[i] == seg_idx[j]:
                continue
            pairs.append((i, j))
    return pairs


def _surface_desc(i: int, info_entry: dict, patch) -> dict:
    wrap = patch.wraps[info_entry["wrap_idx"]]
    seg = wrap.get("segment")
    seg_path = getattr(seg, "path", None)
    return {
        "surface_slot": int(i),
        "wrap_idx": int(info_entry["wrap_idx"]),
        "label": str(info_entry.get("label", "?")),
        "segment_idx": int(wrap["segment_idx"]),
        "segment_path": str(seg_path) if seg_path is not None else "",
    }


def _pair_score(stats: dict) -> tuple[float, float, float]:
    p = stats["percentiles"]
    return (p["1"], p["0.1"], stats["min"])


def _groups_from_pairs(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find: build groups of original slot indices from merge pairs.

    Every slot in [0, n) is included, singletons for slots that don't
    appear in any pair. Groups are returned in order of first
    occurrence of their representative.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in pairs:
        if 0 <= a < n and 0 <= b < n:
            union(a, b)

    groups_by_root: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups_by_root.setdefault(r, []).append(i)
    ordered_roots = sorted(groups_by_root.keys())
    return [sorted(groups_by_root[r]) for r in ordered_roots]


def _fmt_duration(seconds: float) -> str:
    """Compact duration: `1.2s` / `12.3s` / `4m32s` / `1h23m`."""
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        m = int(seconds // 60)
        s = int(seconds - m * 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    return f"{h}h{m:02d}m"


def run_dataset_overlap(
    train_config: str,
    out_path: str,
    num_samples: int | None = None,
    seed: int = 0,
    patch_size: int | None = None,
    num_workers: int | None = None,
    vis_dir: str | None = None,
    vis_top_k: int | None = None,
    model_path: str | None = None,
    inference_tile_size: int | None = None,
    explicit_indices: list[int] | None = None,
) -> None:
    """Scan patches and write per-patch worst-pair overlap stats to JSONL.

    If ``vis_dir`` is given, also render a `dataset vis`-style JPEG per
    emitted patch — reusing ``dataset_vis.render_batch_figure`` so the
    output is byte-for-byte identical to what `dataset vis` would
    produce. ``model_path`` / ``inference_tile_size`` are forwarded to
    the same ``build_inference_context`` helper the vis uses, so
    prediction + residual rows appear when a checkpoint is given.
    ``vis_top_k`` restricts rendering to the K worst patches (by the
    post-flip ``p1``) after the scan completes; without it, patches
    are rendered inline as they are scanned.
    """
    from torch.utils.data import DataLoader, Subset
    from tifxyz_lasagna_dataset import (
        TifxyzLasagnaDataset,
        collate_variable_surfaces,
    )
    from lasagna3d.dataset_vis import (
        build_inference_context,
        default_vis_filename,
        default_vis_title,
        render_batch_figure,
        _sample_from_batch,
    )

    with open(train_config, "r") as f:
        config = json.load(f)

    vis_out: Path | None = None
    inference_ctx: dict | None = None
    if vis_dir is not None:
        vis_out = Path(vis_dir)
        vis_out.mkdir(parents=True, exist_ok=True)
        # The overlap command always feeds the vis its own detected
        # groups per patch (see `overlap_pairs` below), so threshold
        # resolution here is deliberately fixed at None — the vis
        # shows exactly what the analysis flagged, no separate rule.
        inference_ctx = build_inference_context(
            model_path=model_path,
            config=config,
            patch_size=patch_size,
            inference_tile_size=inference_tile_size,
            same_surface_threshold=None,
        )
    elif model_path is not None:
        print(
            f"{TAG} --model has no effect without --vis-dir; ignoring",
            flush=True,
        )

    # When vis_top_k is set we need to keep the raw batch around so we
    # can render it later; otherwise we render inline and drop it.
    defer_render = vis_out is not None and vis_top_k is not None
    # (batch, idx, ds_name, record, vis_groups)
    deferred: list[tuple[dict, int, str, dict, list[list[int]] | None]] = []

    def _render_one(batch, idx, ds_name, same_surface_groups=None) -> str:
        """Render the vis JPEG and return a short suffix describing the
        outcome (filename or error) so the per-patch summary stays on
        one line. ``same_surface_groups`` is the exact grouping the
        analysis loop just detected for this patch — the vis renders
        that state instead of re-running detection."""
        fname = default_vis_filename(ds_name, idx)
        title = default_vis_title(ds_name, idx, _sample_from_batch(batch))
        try:
            render_batch_figure(
                batch, vis_out / fname, title, seed + idx, inference_ctx,
                same_surface_groups=same_surface_groups,
            )
            return fname
        except Exception as e:
            return f"FAILED ({type(e).__name__}: {e})"

    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        print(f"{TAG} config has no datasets", flush=True)
        return

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = int(num_workers)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    analysis_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if analysis_device.type != "cuda":
        print(
            f"{TAG} WARNING: CUDA unavailable, falling back to CPU "
            "(cupy EDT still required — this will fail on a CPU-only host).",
            flush=True,
        )

    top_records: list[dict] = []
    total_written = 0
    total_overlapping = 0  # patches whose worst-pair p50 <= 1 (same-surface rule)
    # Per-patch stdout columns — min, then percentiles that aren't p50
    # (p50 is already shown as median). Ordered low → high.
    _SUMMARY_PCTS = ("1", "5", "15", "25", "50", "75", "85", "95", "99")
    _HEADER_EVERY = 20
    _HEADER = (
        f"{'dataset':<12} {'idx':>6} {'pair':<5} "
        f"{'min':>5} "
        + " ".join(f"{'p'+lbl:>5}" for lbl in _SUMMARY_PCTS)
        + f" {'overlap':>14} {'mean':>6} {'last':>6} {'eta':>6}"
    )

    with out_file.open("w") as fout:
        for ds_idx, ds_entry in enumerate(datasets_cfg):
            if ds_entry.get("volume_path") is None:
                print(
                    f"{TAG} [{ds_idx}] skipping (volume_path is null)",
                    flush=True,
                )
                continue

            ds_name = _dataset_display_name(ds_entry, ds_idx)
            print(f"{TAG} [{ds_idx}] building dataset '{ds_name}'", flush=True)

            sub_config = dict(config)
            sub_config["datasets"] = [ds_entry]
            dataset = TifxyzLasagnaDataset(
                sub_config,
                apply_augmentation=False,
                include_geometry=True,
                include_patch_ref=True,
            )
            n_total = len(dataset)
            if n_total == 0:
                print(
                    f"{TAG} [{ds_idx}] '{ds_name}' has 0 patches, skipping",
                    flush=True,
                )
                continue

            if explicit_indices is not None:
                indices = [k for k in explicit_indices if 0 <= k < n_total]
                if len(indices) != len(explicit_indices):
                    dropped = sorted(set(explicit_indices) - set(indices))
                    print(
                        f"{TAG} [{ds_idx}] '{ds_name}': dropping out-of-range "
                        f"indices {dropped} (valid range 0..{n_total - 1})",
                        flush=True,
                    )
            else:
                indices = list(range(n_total))
                random.Random(seed + ds_idx).shuffle(indices)
                if num_samples is not None:
                    indices = indices[: min(int(num_samples), n_total)]

            print(
                f"{TAG} [{ds_idx}] '{ds_name}': scanning {len(indices)} / {n_total} "
                f"(num_workers={num_workers})",
                flush=True,
            )

            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset, batch_size=1, shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_variable_surfaces,
                persistent_workers=False,
            )

            ds_scan_start = time.perf_counter()
            ds_last_tick = ds_scan_start
            ds_total_n = len(indices)

            for i, batch in enumerate(loader):
                idx = indices[i]
                n_surfaces = int(batch["num_surfaces"][0])
                info = batch["surface_chain_info"][0]
                geom = batch.get("surface_geometry", [[]])[0]
                masks_t = batch["surface_masks"][0]  # (N, Z, Y, X) float32
                patch_ref = batch["_patch"][0]

                seg_idx_list = [
                    int(patch_ref.wraps[e["wrap_idx"]]["segment_idx"])
                    for e in info
                ]
                pairs = _candidate_pairs(info, seg_idx_list)
                if not pairs:
                    continue

                # Build slot → surface_geometry lookup aligned with `info`.
                geom_by_slot: dict[int, dict] = {}
                for entry in geom:
                    wi = int(entry["wrap_idx"])
                    for slot, ci in enumerate(info):
                        if int(ci["wrap_idx"]) == wi:
                            geom_by_slot[slot] = entry
                            break

                # Move surface masks to GPU **once per patch**; all
                # pair evaluations then share the CUDA tensors.
                masks_cuda = (masks_t.to(analysis_device) > 0.5)

                # Pre-move per-slot points/normals to GPU once per patch.
                pts_by_slot: dict[int, torch.Tensor] = {}
                nrm_by_slot: dict[int, torch.Tensor] = {}
                for slot, entry in geom_by_slot.items():
                    pts = np.asarray(
                        entry.get("points_local", np.zeros((0, 3), np.float32)),
                        dtype=np.float32,
                    )
                    nrm = np.asarray(
                        entry.get("normals_zyx", np.zeros((0, 3), np.float32)),
                        dtype=np.float32,
                    )
                    if pts.shape[0] == 0:
                        continue
                    pts_by_slot[slot] = torch.from_numpy(pts).to(analysis_device)
                    nrm_by_slot[slot] = torch.from_numpy(nrm).to(analysis_device)

                best = None
                num_evaluated = 0
                overlap_pairs: list[tuple[int, int]] = []
                for (a, b) in pairs:
                    pts_a = pts_by_slot.get(a)
                    if pts_a is None:
                        continue
                    nrm_a = nrm_by_slot[a]
                    mask_b_t = masks_cuda[b]
                    if not bool(mask_b_t.any()):
                        continue

                    signed = _signed_distance_samples_gpu(pts_a, nrm_a, mask_b_t)
                    if signed.numel() == 0:
                        continue
                    stats = _compute_stats_gpu(signed)
                    num_evaluated += 1
                    score = _pair_score(stats)
                    if best is None or score < best[0]:
                        best = (score, a, b, stats)
                    # Same rule that drives the stdout overlap counter:
                    # post-flip median ≤ 1 → treat as duplicate.
                    if stats["percentiles"]["50"] <= 1.0:
                        overlap_pairs.append((a, b))

                if best is None:
                    continue

                _, a, b, stats = best
                patch_info = batch["patch_info"][0]
                record = {
                    "dataset": ds_name,
                    "dataset_idx": int(ds_idx),
                    "patch_idx": int(idx),
                    "world_bbox": list(patch_info["world_bbox"]),
                    "num_surfaces": int(n_surfaces),
                    "num_pairs_considered": int(len(pairs)),
                    "num_pairs_evaluated": int(num_evaluated),
                    "worst_pair": {
                        "a": _surface_desc(a, info[a], patch_ref),
                        "b": _surface_desc(b, info[b], patch_ref),
                        **stats,
                    },
                }
                # Invariant: median must be non-negative after sign flip.
                assert record["worst_pair"]["percentiles"]["50"] >= 0.0

                fout.write(json.dumps(record) + "\n")
                fout.flush()
                total_written += 1

                wp = record["worst_pair"]
                perc = wp["percentiles"]
                p50 = perc["50"]
                if p50 <= 1.0:
                    total_overlapping += 1
                overlap_frac = total_overlapping / total_written
                pos_a = int(info[a].get("pos", 0))
                pos_b = int(info[b].get("pos", 0))
                pair_str = f"{pos_a}-{pos_b}"
                overlap_str = (
                    f"{total_overlapping}/{total_written} "
                    f"({100.0 * overlap_frac:.1f}%)"
                )
                stats_str = " ".join(f"{perc[lbl]:5.1f}" for lbl in _SUMMARY_PCTS)
                ds_col = ds_name if len(ds_name) <= 12 else ds_name[:12]

                # Timing: per-row delta and ETA from the dataset's
                # rolling average since ds_scan_start.
                now = time.perf_counter()
                row_dt = now - ds_last_tick
                ds_last_tick = now
                rows_done = i + 1
                mean_dt = (now - ds_scan_start) / max(rows_done, 1)
                remaining = max(ds_total_n - rows_done, 0)
                eta_s = remaining * mean_dt

                line = (
                    f"{ds_col:<12} {int(idx):>6} {pair_str:<5} "
                    f"{wp['min']:5.1f} {stats_str} {overlap_str:>14} "
                    f"{_fmt_duration(mean_dt):>6} {_fmt_duration(row_dt):>6} "
                    f"{_fmt_duration(eta_s):>6}"
                )

                top_records.append(record)

                # The grouping the vis should render = whatever the
                # analysis just flagged for this patch (overlap_pairs,
                # filtered by the `p50 ≤ 1` rule above).
                vis_groups = (
                    _groups_from_pairs(n_surfaces, overlap_pairs)
                    if overlap_pairs else None
                )

                vis_suffix = ""
                if vis_out is not None:
                    if defer_render:
                        deferred.append((batch, idx, ds_name, record, vis_groups))
                    else:
                        vis_suffix = (
                            f"  {_render_one(batch, idx, ds_name, vis_groups)}"
                        )

                if (total_written - 1) % _HEADER_EVERY == 0:
                    print(_HEADER, flush=True)
                print(line + vis_suffix, flush=True)

    print(
        f"{TAG} done — wrote {total_written} records to {out_file}",
        flush=True,
    )

    if defer_render and deferred:
        deferred.sort(
            key=lambda item: (
                item[3]["worst_pair"]["percentiles"]["1"],
                item[3]["worst_pair"]["percentiles"]["0.1"],
                item[3]["worst_pair"].get("min", 0.0),
            )
        )
        k = min(int(vis_top_k), len(deferred))
        print(f"{TAG} rendering top-{k} worst patches into {vis_out}", flush=True)
        for batch, idx, ds_name, _record, vis_groups in deferred[:k]:
            print(
                f"{TAG}   {_render_one(batch, idx, ds_name, vis_groups)}",
                flush=True,
            )

    if top_records:
        top_records.sort(
            key=lambda r: (
                r["worst_pair"]["percentiles"]["1"],
                r["worst_pair"]["percentiles"]["0.1"],
                r["worst_pair"]["min"] if "min" in r["worst_pair"] else 0.0,
            )
        )
        k = min(_TOPK_SUMMARY, len(top_records))
        print(f"{TAG} top-{k} worst patches by p1:", flush=True)
        for r in top_records[:k]:
            la = r["worst_pair"]["a"]["label"]
            lb = r["worst_pair"]["b"]["label"]
            p1 = r["worst_pair"]["percentiles"]["1"]
            p50 = r["worst_pair"]["percentiles"]["50"]
            print(
                f"{TAG}   {r['dataset']} idx={r['patch_idx']} "
                f"{la}↔{lb} p1={p1:.3f} median={p50:.3f}",
                flush=True,
            )
