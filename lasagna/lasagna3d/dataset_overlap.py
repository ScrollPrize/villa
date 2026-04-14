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
from pathlib import Path

import numpy as np

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


def _edt_with_indices(mask_not_b_uint8: np.ndarray):
    """CuPy EDT + feature transform on the complement of B's mask.

    Input: (Z, Y, X) uint8 — 1 where B is absent, 0 on B's surface.
    Returns: (dist, inds) as numpy on CPU. `dist` is (Z, Y, X) float32.
    `inds` is (3, Z, Y, X) int32 — for each voxel, the (z, y, x) index
    of the nearest voxel where the mask is 0 (i.e. nearest point on B).
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    cp_mask = cp.asarray(mask_not_b_uint8)
    dist, inds = cupy_edt(
        cp_mask, return_distances=True, return_indices=True,
    )
    return cp.asnumpy(dist).astype(np.float32), cp.asnumpy(inds).astype(np.int32)


def _signed_distance_samples(
    points_a_zyx: np.ndarray,   # (M, 3) float32
    normals_a_zyx: np.ndarray,  # (M, 3) float32
    mask_b: np.ndarray,         # (Z, Y, X) bool
) -> np.ndarray:
    """Return a length-M array of signed distances from points on A to B."""
    Z, Y, X = mask_b.shape
    if points_a_zyx.shape[0] == 0 or not mask_b.any():
        return np.empty(0, dtype=np.float32)

    complement = (~mask_b).astype(np.uint8)
    dist_b, inds_b = _edt_with_indices(complement)

    pz = np.clip(np.floor(points_a_zyx[:, 0]).astype(np.int64), 0, Z - 1)
    py = np.clip(np.floor(points_a_zyx[:, 1]).astype(np.int64), 0, Y - 1)
    px = np.clip(np.floor(points_a_zyx[:, 2]).astype(np.int64), 0, X - 1)

    d = dist_b[pz, py, px]  # (M,) unsigned distance from voxel of P to B

    nearest_z = inds_b[0, pz, py, px].astype(np.float32)
    nearest_y = inds_b[1, pz, py, px].astype(np.float32)
    nearest_x = inds_b[2, pz, py, px].astype(np.float32)
    nearest = np.stack([nearest_z, nearest_y, nearest_x], axis=-1)

    v = points_a_zyx.astype(np.float32) - nearest  # (M, 3)
    dot = np.sum(v * normals_a_zyx.astype(np.float32), axis=-1)
    sign = np.where(dot >= 0.0, 1.0, -1.0).astype(np.float32)

    return (d.astype(np.float32) * sign).astype(np.float32)


def _compute_stats(signed: np.ndarray) -> dict:
    """Compute min/max/percentiles, flipping sign if median is negative."""
    pct = np.percentile(signed, _PCT_VALUES)
    smin = float(signed.min())
    smax = float(signed.max())
    median = float(pct[_PCT_LABELS.index("50")])
    flipped = False
    if median < 0.0:
        flipped = True
        pct = -pct[::-1]  # negate and reverse so labels still map in order
        smin, smax = -smax, -smin
    return {
        "n_samples": int(signed.shape[0]),
        "min": smin,
        "max": smax,
        "percentiles": {
            label: float(pct[i]) for i, label in enumerate(_PCT_LABELS)
        },
        "sign_flipped": bool(flipped),
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
    same_surface_threshold: float | None = None,
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
        inference_ctx = build_inference_context(
            model_path=model_path,
            config=config,
            patch_size=patch_size,
            inference_tile_size=inference_tile_size,
            same_surface_threshold=same_surface_threshold,
        )
    elif model_path is not None:
        print(
            f"{TAG} --model has no effect without --vis-dir; ignoring",
            flush=True,
        )

    # When vis_top_k is set we need to keep the raw batch around so we
    # can render it later; otherwise we render inline and drop it.
    defer_render = vis_out is not None and vis_top_k is not None
    deferred: list[tuple[dict, int, str, dict]] = []  # (batch, idx, ds_name, record)

    def _render_one(batch, idx, ds_name):
        fname = default_vis_filename(ds_name, idx)
        title = default_vis_title(ds_name, idx, _sample_from_batch(batch))
        try:
            render_batch_figure(
                batch, vis_out / fname, title, seed + idx, inference_ctx,
            )
            print(f"{TAG}   vis → {fname}", flush=True)
        except Exception as e:
            print(
                f"{TAG}   vis render failed for idx={idx}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        print(f"{TAG} config has no datasets", flush=True)
        return

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = int(num_workers)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    top_records: list[dict] = []
    total_written = 0

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

                geom_by_slot: dict[int, dict] = {}
                for entry in geom:
                    wi = int(entry["wrap_idx"])
                    for slot, ci in enumerate(info):
                        if int(ci["wrap_idx"]) == wi:
                            geom_by_slot[slot] = entry
                            break

                masks_np = masks_t.detach().cpu().numpy() > 0.5

                best = None
                num_evaluated = 0
                for (a, b) in pairs:
                    geom_a = geom_by_slot.get(a)
                    if geom_a is None:
                        continue
                    pts_a = np.asarray(
                        geom_a.get("points_local", np.zeros((0, 3), np.float32))
                    )
                    nrm_a = np.asarray(
                        geom_a.get("normals_zyx", np.zeros((0, 3), np.float32))
                    )
                    if pts_a.shape[0] == 0:
                        continue
                    mask_b = masks_np[b]
                    if not mask_b.any():
                        continue

                    signed = _signed_distance_samples(pts_a, nrm_a, mask_b)
                    if signed.size == 0:
                        continue
                    stats = _compute_stats(signed)
                    num_evaluated += 1
                    score = _pair_score(stats)
                    if best is None or score < best[0]:
                        best = (score, a, b, stats)

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

                p1 = record["worst_pair"]["percentiles"]["1"]
                p50 = record["worst_pair"]["percentiles"]["50"]
                la = record["worst_pair"]["a"]["label"]
                lb = record["worst_pair"]["b"]["label"]
                flipped = "*" if record["worst_pair"]["sign_flipped"] else ""
                print(
                    f"{TAG} ds={ds_name} idx={idx} worst={la}↔{lb} "
                    f"p1={p1:.3f}{flipped} median={p50:.3f}",
                    flush=True,
                )

                top_records.append(record)

                if vis_out is not None:
                    if defer_render:
                        deferred.append((batch, idx, ds_name, record))
                    else:
                        _render_one(batch, idx, ds_name)

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
        for batch, idx, ds_name, _record in deferred[:k]:
            _render_one(batch, idx, ds_name)

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
