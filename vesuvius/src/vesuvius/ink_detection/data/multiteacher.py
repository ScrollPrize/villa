"""Explicit flat manifests, conservative holdouts, and resumable sample draws.

Preparation reads masks once. Training streams only the required Zarr chunks.
Mask pooling uses ANY, never nearest-neighbor, so tiny validation marks cannot
disappear when constructing the exclusion map.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
import torch
from torch.utils.data import Dataset, Sampler

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.data.segment import parse_label_asset_path
from vesuvius.ink_detection.volume_io import open_volume


SCROLLS = ("0009b", "0139", "0500p2", "1667", "814", "841", "man5", "phercparis4")
SPACING = dict(zip(SCROLLS, (2.401, 2.399, 2.215, 2.399, 2.399, 2.403, 2.399, 2.4)))
MASK_FACTOR = 16


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def max_pool_mask(mask, factor=MASK_FACTOR):
    """Conservatively represent every marked full-resolution pixel."""
    mask = np.asarray(mask) > 0
    h, w = mask.shape
    padded = np.pad(mask, ((0, (-h) % factor), (0, (-w) % factor)))
    return padded.reshape(math.ceil(h / factor), factor,
                          math.ceil(w / factor), factor).any(axis=(1, 3))


def read_mask(path):
    """Use the lossless source TIFF when available; check its raster geometry."""
    array = open_volume(path, 0)
    tif = Path(path).with_suffix(".tif")
    if tif.exists():
        from vesuvius.ink_detection.preprocessing.create_label_zarrs import load_image
        result = load_image(tif)
        if result.shape != array.shape[-2:]:
            raise ValueError(f"TIFF/Zarr geometry mismatch: {tif}")
        # Verify source/rendered mask correspondence before using TIFF for splits.
        for fy, fx in ((0.25, 0.25), (0.5, 0.5), (0.75, 0.75)):
            y, x = int(result.shape[0] * fy), int(result.shape[1] * fx)
            a = array[array.shape[0] // 2, y:y+128, x:x+128]
            if not np.array_equal(a > 0, result[y:y+128, x:x+128] > 0):
                raise ValueError(f"TIFF/Zarr mask values disagree: {tif}")
        return result
    return np.asarray(array[array.shape[0] // 2])


def select_assets(directory):
    """Resolve a matched version, including the published PHerc1667 prefix."""
    prefixes = (directory.name, directory.name.removesuffix("_2um"))
    candidates = defaultdict(dict)
    for path in sorted(directory.glob("*.zarr")):
        parsed = parse_label_asset_path(path)
        if parsed is not None and parsed["prefix"] in prefixes:
            key = (int(parsed["version_num"]), parsed["prefix"] == directory.name)
            candidates[key][parsed["label_kind"]] = str(path)
    complete = [k for k, v in candidates.items() if {"inklabels", "supervision_mask"} <= v.keys()]
    if not complete:
        raise ValueError(f"No matching label pair in {directory}")
    key = max(complete)
    assets = candidates[key]
    # A corresponding validation asset can retain the original short prefix.
    if "validation_mask" not in assets:
        for k in sorted(candidates, reverse=True):
            if k[0] == key[0] and "validation_mask" in candidates[k]:
                assets["validation_mask"] = candidates[k]["validation_mask"]
                break
    return assets, key[0]


def spatial_holdout(support, seed, identity, block_size=1024):
    cells = block_size // MASK_FACTOR
    blocks = sorted(set(zip(*(np.nonzero(support)[i] // cells for i in (0, 1)))))
    if len(blocks) < 2:
        raise ValueError(f"Not enough spatial blocks to split {identity}")
    blocks.sort(key=lambda p: stable_seed(seed, identity, *p))
    selected = blocks[:max(1, round(len(blocks) * 0.1))]
    mask = np.zeros_like(support)
    for by, bx in selected:
        mask[by*cells:(by+1)*cells, bx*cells:(bx+1)*cells] = True
    return mask, [[int(y), int(x)] for y, x in selected]


def patch_coordinates(support, heldout, shape_yx, patch=256, guard=256):
    """Every training footprint must be outside the full holdout plus guard."""
    size = patch // MASK_FACTOR
    guard_cells = math.ceil(guard / MASK_FACTOR)
    exclusion = maximum_filter(heldout, size=2 * guard_cells + 1, mode="constant")

    def sums(mask, ys, xs):
        integral = np.pad(mask.astype(np.uint32).cumsum(0, dtype=np.uint32)
                          .cumsum(1, dtype=np.uint32), ((1, 0), (1, 0)))
        return (integral[ys+size, xs+size] - integral[ys, xs+size]
                - integral[ys+size, xs] + integral[ys, xs])

    def scan(stride, validation):
        ys, xs = np.meshgrid(np.arange(0, (shape_yx[0]-patch)//MASK_FACTOR+1, stride),
                             np.arange(0, (shape_yx[1]-patch)//MASK_FACTOR+1, stride), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
        if validation:
            keep = sums(heldout & support, ys, xs) > 0
        else:
            keep = ((sums(support, ys, xs) >= 0.05 * size * size)
                    & (sums(exclusion, ys, xs) == 0))
        return np.stack((ys[keep], xs[keep]), axis=1).astype(np.int32) * MASK_FACTOR

    return scan(size // 2, False), scan(size, True)


def aligned_extent(identity, image_shape, label_shapes):
    """Keep the original loader's common-origin coordinates, without resizing.

    The published MAN5 labels stop 59 pixels before the CT's bottom/right
    bounds. Admit that explicit source exception, and keep every patch inside
    both rasters. Other unreviewed geometry differences remain errors.
    """
    shapes = {tuple(s[-2:]) for s in label_shapes}
    if len(shapes) != 1:
        raise ValueError(f"Label geometries disagree: {identity}")
    labels = shapes.pop()
    if labels != tuple(image_shape[-2:]):
        known = (identity == "man5/MAN5_outer_3"
                 and tuple(image_shape) == (65, 24120, 29900)
                 and labels == (24061, 29841))
        if not known:
            raise ValueError(f"Label/CT geometry mismatch: {identity}")
    return labels


def asset_fingerprint(path, *, include_tiff=True):
    path = Path(path)
    result = {"schema_sha256": file_sha256(path / "0" / ".zarray")}
    tif = path.with_suffix(".tif")
    if include_tiff and tif.exists():
        result["tiff_sha256"] = file_sha256(tif)
    return result


def prepare_manifest(root, output, seed=27):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (output/"manifest.json").exists():
        raise FileExistsError(f"Refusing to replace prepared experiment data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    for scroll in SCROLLS:
        for directory in sorted((root / scroll).iterdir()):
            image_path = directory / (directory.name + ".zarr")
            if not image_path.is_dir():
                continue
            assets, version = select_assets(directory)
            image = open_volume(image_path, 0)
            if image.shape[0] < 64 or np.dtype(image.dtype) != np.dtype("uint8"):
                raise ValueError(f"Expected >=64 planes of original uint8 CT: {image_path}")
            identity = scroll + "/" + directory.name
            extent = aligned_extent(identity, image.shape,
                                    [open_volume(p, 0).shape for p in assets.values()])
            support = max_pool_mask(read_mask(assets["supervision_mask"]))
            generated_blocks = []
            if assets.get("validation_mask"):
                heldout = max_pool_mask(read_mask(assets["validation_mask"]))
                val_support = support | heldout
                if not heldout.any():
                    assets.pop("validation_mask")
                    heldout, generated_blocks = spatial_holdout(support, seed, identity)
                    val_support = support
            else:
                heldout, generated_blocks = spatial_holdout(support, seed, identity)
                val_support = support
            train, val = patch_coordinates(val_support, heldout, extent)
            if len(val) == 0:
                raise ValueError(f"Empty train/validation partition: {identity}")
            index = len(records)
            arrays[f"train_{index}"] = train
            arrays[f"val_{index}"] = val
            arrays[f"heldout_{index}"] = heldout
            fingerprints = {name: asset_fingerprint(Path(path), include_tiff=name != "image")
                            for name, path in {"image": str(image_path), **assets}.items()}
            records.append({"scroll": scroll, "segment": directory.name,
                            "physical_segment": identity.removesuffix("_2um"),
                            "image": str(image_path), **assets, "label_version": version,
                            "teacher_id": 0 if scroll == "phercparis4" else 1,
                            "shape": list(image.shape), "spacing_um": SPACING[scroll],
                            "sampling_extent_yx": list(extent),
                            "orientation": "published_canonical_zyx",
                            "source": f"hf://buckets/scrollprize/datasets/ink/{identity}",
                            "fingerprints": fingerprints, "generated_blocks": generated_blocks,
                            "validation_only": len(train) == 0,
                            "train_patches": len(train), "val_patches": len(val)})
            print(json.dumps({"prepared": identity, "version": version,
                              "train": len(train), "val": len(val)}), flush=True)
    if set(r["scroll"] for r in records) != set(SCROLLS):
        raise ValueError("Manifest must contain all eight scrolls")
    physical = [r["physical_segment"] for r in records]
    if len(set(physical)) != len(physical):
        raise ValueError("Duplicate physical representations require shared exclusions")
    np.savez_compressed(output/"patches.npz", **arrays)
    manifest = {"version": 1, "seed": seed, "patch_size": [64, 256, 256],
                "mask_factor": MASK_FACTOR, "guard_pixels": 256,
                "patches_sha256": file_sha256(output/"patches.npz"), "segments": records}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return output/"manifest.json"


def read_oriented_patch(source, y, x, *, reverse_depth=False):
    """Read the central patch of the fully depth-oriented segment.

    Equivalent to source[::-1][middle-32:middle+32] for reverse depth,
    including odd-depth segments, without materializing the entire Zarr.
    All normalization, validity and teacher generation happen after this step.
    """
    depth = source.shape[0]
    middle = depth // 2
    start, stop = middle - 32, middle + 32
    if start < 0 or stop > depth:
        raise ValueError("Segment must contain at least 64 depth planes")
    if reverse_depth:
        start, stop = depth - stop, depth - start
    raw = np.asarray(source[start:stop, y:y+256, x:x+256], dtype=np.float32)
    return raw[::-1].copy() if reverse_depth else raw


class FlatDistillationDataset(Dataset):
    """Each draw is a pure function of seed, draw ID, and a frozen manifest."""

    def __init__(self, manifest_path, teacher_normalizations, *, validation=False,
                 val_patches_per_scroll=128, augment=True, world_size=1,
                 batch_size=1, paris4_per_batch=0, grad_acc_steps=1,
                 paris4_per_rank_update=0, paris4_per_global_update=0,
                 student_normalization=None, valid_depth_margin=0,
                 segment_depth_reversals=None, orientation_calibration_exclusions=None,
                 excluded_segments=None, path_roots=None):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        self.records = self.manifest["segments"]
        self.path_roots = path_roots or {}
        self.excluded_segments = set(excluded_segments or [])
        if self.excluded_segments:
            known = {r['scroll']+'/'+r['segment'] for r in self.records}
            if not self.excluded_segments <= known:
                raise ValueError("Excluded segments must exist in the manifest")
        self.segment_depth_reversals = segment_depth_reversals
        if segment_depth_reversals is not None:
            expected = {r['scroll']+'/'+r['segment'] for r in self.records}
            if (set(segment_depth_reversals) != expected
                    or any(type(v) is not bool for v in segment_depth_reversals.values())):
                raise ValueError("Depth orientation requires an explicit boolean for every segment")
        self.seed = int(self.manifest["seed"])
        self.validation = bool(validation)
        self.augment = bool(augment and not validation)
        self.normalizations = [NormalizationConfig.from_value(v) for v in teacher_normalizations]
        self.student_normalization = (self.normalizations[1] if student_normalization is None
                                      else NormalizationConfig.from_value(student_normalization))
        self.valid_depth_margin = int(valid_depth_margin)
        if not 0 <= self.valid_depth_margin < 32:
            raise ValueError("valid_depth_margin must be between 0 and 31")
        with np.load(self.path.parent/"patches.npz") as arrays:
            self.coordinates = [arrays[f"{'val' if validation else 'train'}_{i}"]
                                for i in range(len(self.records))]
            self.heldout = [arrays[f"heldout_{i}"] for i in range(len(self.records))]
        self.groups = defaultdict(list)
        for i, record in enumerate(self.records):
            if self.excluded_segments and record['scroll']+'/'+record['segment'] in self.excluded_segments:
                continue
            if len(self.coordinates[i]):
                self.groups[record["scroll"]].append(i)
        self.scrolls = sorted(self.groups)
        self.world_size, self.batch_size = int(world_size), int(batch_size)
        self.paris4_per_batch = 0 if validation else int(paris4_per_batch)
        self.grad_acc_steps = int(grad_acc_steps)
        self.paris4_per_rank_update = 0 if validation else int(paris4_per_rank_update)
        self.paris4_per_global_update = 0 if validation else int(paris4_per_global_update)
        if min(self.world_size, self.batch_size, self.grad_acc_steps) <= 0:
            raise ValueError("Sampling dimensions must be positive")
        if sum(bool(q) for q in (self.paris4_per_batch, self.paris4_per_rank_update,
                                 self.paris4_per_global_update)) > 1:
            raise ValueError("Choose only one Paris 4 sampling quota")
        self.other_scrolls = [s for s in self.scrolls if s != "phercparis4"]
        if self.paris4_per_batch and (
            "phercparis4" not in self.groups or not self.other_scrolls
            or not 0 < self.paris4_per_batch < self.batch_size
        ):
            raise ValueError("Paris 4 quota requires both teacher groups in each local batch")
        if self.paris4_per_rank_update and (
            "phercparis4" not in self.groups or not self.other_scrolls
            or not 0 < self.paris4_per_rank_update < self.batch_size*self.grad_acc_steps
        ):
            raise ValueError("Paris 4 update quota requires both teacher groups")
        if self.paris4_per_global_update and (
            "phercparis4" not in self.groups or not self.other_scrolls
            or not 0 < self.paris4_per_global_update < self.world_size*self.batch_size*self.grad_acc_steps
        ):
            raise ValueError("Paris 4 global update quota requires both teacher groups")
        self.validation_draws = []
        if validation:
            exclusions = orientation_calibration_exclusions or {}
            for scroll in self.scrolls:
                candidates = [(i, j) for i in self.groups[scroll]
                              for j in range(len(self.coordinates[i]))
                              if not exclusions or j not in exclusions.get(
                                  self.records[i]['scroll']+'/'+self.records[i]['segment'], [])]
                candidates.sort(key=lambda x: stable_seed(self.seed, "validation", scroll, *x))
                self.validation_draws.extend(candidates[:val_patches_per_scroll])
        self._volumes = {}

    def __len__(self):
        return len(self.validation_draws) if self.validation else 2**40

    def _open(self, path):
        if path not in self._volumes:
            resolved = Path(path)
            for old, new in sorted(self.path_roots.items(), key=lambda kv: -len(kv[0])):
                if resolved.is_relative_to(old):
                    resolved = Path(new) / resolved.relative_to(old)
                    break
            self._volumes[path] = open_volume(resolved, 0)
        return self._volumes[path]

    def locate(self, draw):
        """Inspect draw metadata without reading CT chunks."""
        rng = np.random.default_rng(stable_seed(self.seed, "draw", int(draw)))
        if self.validation:
            index, patch_index = self.validation_draws[draw]
        else:
            if self.paris4_per_global_update:
                local_draw, rank = divmod(int(draw), self.world_size)
                update, slot = divmod(local_draw, self.batch_size*self.grad_acc_steps)
                microstep, batch_slot = divmod(slot, self.batch_size)
                # Spread the exact global quota across microsteps; rotate the
                # participating ranks and local positions deterministically.
                quota = (self.paris4_per_global_update+self.grad_acc_steps-1-microstep)//self.grad_acc_steps
                lane = (rank-update-microstep) % self.world_size
                position = ((batch_slot-update) % self.batch_size)*self.world_size+lane
                scroll = ("phercparis4" if position < quota else
                          self.other_scrolls[int(rng.integers(len(self.other_scrolls)))])
            elif self.paris4_per_rank_update:
                local_draw, rank = divmod(int(draw), self.world_size)
                update, slot = divmod(local_draw, self.batch_size*self.grad_acc_steps)
                # Stagger PH4 across ranks and microsteps. For 4 GPUs, B=2,
                # accumulation=2, quota=1: every microstep has 2 PH4 + 6 others,
                # and every rank receives exactly one PH4 per optimizer update.
                start = ((rank+update) % self.grad_acc_steps)*self.batch_size
                start += (rank//self.grad_acc_steps+update) % self.batch_size
                precise = (slot-start) % (self.batch_size*self.grad_acc_steps) < self.paris4_per_rank_update
                scroll = ("phercparis4" if precise else
                          self.other_scrolls[int(rng.integers(len(self.other_scrolls)))])
            elif self.paris4_per_batch:
                # RankDrawSampler interleaves ranks before advancing local slots.
                slot = (int(draw)//self.world_size) % self.batch_size
                scroll = ("phercparis4" if slot < self.paris4_per_batch else
                          self.other_scrolls[int(rng.integers(len(self.other_scrolls)))])
            else:
                scroll = self.scrolls[int(rng.integers(len(self.scrolls)))]
            indices = self.groups[scroll]
            index = indices[int(rng.integers(len(indices)))]
            patch_index = int(rng.integers(len(self.coordinates[index])))
        return index, patch_index, rng

    def __getitem__(self, draw):
        index, patch_index, rng = self.locate(draw)
        record = self.records[index]
        y, x = map(int, self.coordinates[index][patch_index])
        source = self._open(record["image"])
        reverse_depth = (False if self.segment_depth_reversals is None else
                         self.segment_depth_reversals[record['scroll']+'/'+record['segment']])
        raw = read_oriented_patch(source, y, x, reverse_depth=reverse_depth)
        def plane(key):
            array = self._open(record[key])
            return np.asarray(array[array.shape[0]//2, y:y+256, x:x+256]) > 0
        labels = plane("inklabels")
        support = plane("supervision_mask")
        if self.validation:
            if record.get("validation_mask"):
                support = plane("validation_mask")
            else:
                heldout = self.heldout[index][y//16:y//16+16, x//16:x//16+16]
                support &= heldout.repeat(16, 0).repeat(16, 1)
        valid = np.broadcast_to(np.any(raw != 0, axis=0), raw.shape).copy()
        if self.valid_depth_margin:
            valid[:self.valid_depth_margin] = False
            valid[-self.valid_depth_margin:] = False
        support &= valid.any(axis=0)
        teacher_image = normalize_image(raw.copy(), self.normalizations[record["teacher_id"]])
        image = normalize_image(raw.copy(), self.student_normalization)
        arrays = [raw, teacher_image, image, labels, support, valid]
        if self.augment:
            rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
            arrays = [np.rot90(a, rotation, axes=(-2, -1)) for a in arrays]
            if flip:
                arrays = [np.flip(a, axis=-1) for a in arrays]
        raw, teacher_image, image, labels, support, valid = arrays
        if self.augment:
            image = np.clip(image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        def tensor(a):
            return torch.from_numpy(np.array(a, dtype=np.float32, copy=True)).unsqueeze(0)
        return {"image": tensor(image), "teacher_image": tensor(teacher_image),
                "raw": tensor(raw), "labels_2d": tensor(labels), "mask_2d": tensor(support),
                "valid_3d": tensor(valid), "teacher_id": torch.tensor(record["teacher_id"]),
                "scroll_id": torch.tensor(SCROLLS.index(record["scroll"])),
                "record_id": torch.tensor(index), "draw_id": torch.tensor(draw),
                "reverse_depth": torch.tensor(reverse_depth),
                "yx": torch.tensor([y, x])}


class RankDrawSampler(Sampler):
    """Disjoint rank draws, independent of dataloader prefetch or restart."""

    def __init__(self, start, stop, rank, world_size):
        self.start, self.stop, self.rank, self.world_size = start, stop, rank, world_size

    def __iter__(self):
        return iter(range(self.start + self.rank, self.stop, self.world_size))

    def __len__(self):
        return max(0, (self.stop - self.start - self.rank + self.world_size - 1)//self.world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_manifest(args.root, args.output)
