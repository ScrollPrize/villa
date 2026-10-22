"""Bounded vc_reader CPU/CUDA readers and paired native augmentations."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from vc_reader import transforms as A
from vc_reader.reader import open_zarr

from dinovol_2.dataset.cuda_pipeline import normalization_transforms
from .prepare import FORMAT


class SurfaceData:
    def __init__(self, root, crop_size, *, device="cuda", normalization="robust",
                 max_open=4, min_valid_fraction=0.01):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != FORMAT:
            raise ValueError("Unsupported surface dataset format")
        self.cases = {split: [c for c in self.manifest["cases"] if c["split"] == split]
                      for split in ("train", "val")}
        ids = [{c["id"] for c in self.cases[s]} for s in ("train", "val")]
        if not all(ids) or ids[0] & ids[1]:
            raise ValueError("Both splits must be nonempty and disjoint")
        self.crop_size = tuple(crop_size)
        if len(self.crop_size) != 3 or min(self.crop_size) <= 0:
            raise ValueError("crop_size must contain three positive dimensions")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cuda and cpu (smoke tests)")
        self.backend = self.device.type
        self.max_open, self.min_valid_fraction = max_open, min_valid_fraction
        self.readers = OrderedDict()
        targets = {"image": A.Image(valid=None), "label": A.Label(ignore_value=2)}
        # Only rotate equal-length axes, so anisotropic crops retain their shape.
        planes = ["zyx"[a] + "zyx"[b] for a in range(3) for b in range(a + 1, 3)
                  if self.crop_size[a] == self.crop_size[b]]
        geometry = [A.Flip(axes="zyx", p_axis=0.5)]
        if planes:
            geometry.append(A.Rotate90(planes=planes))
        normalize = normalization_transforms(normalization)
        intensity = [A.BrightnessMultiply(factor=A.Uniform(0.8, 1.2), p=0.15),
                     A.Contrast(factor=A.Uniform(0.75, 1.25), p=0.15),
                     A.Gamma(gamma=A.Uniform(0.7, 1.5), p_retain_stats=1.0, p=0.15),
                     A.GaussianNoise(std=A.Uniform(0, 0.1), p=0.15)]
        medial = [A.MedialSurface(on=["label"], ignore={"label": 2},
                                 tube=True, open=False, close=True)]
        self.pipelines = {
            True: A.Pipeline(geometry + normalize + intensity + medial,
                             targets=targets, backend=self.backend, compact=False),
            False: A.Pipeline(normalize + medial, targets=targets,
                              backend=self.backend, compact=False),
        }

    def _readers_for(self, case):
        key = (case["split"], case["id"])
        if key not in self.readers:
            while len(self.readers) >= self.max_open:
                self._close_pair(self.readers.popitem(last=False)[1])
            pair = []
            try:
                for name in ("image", "label"):
                    cpu = open_zarr(str(self.root / case["path"] / name),
                                    threads=2, memory_cache=32 << 20)
                    pair.append((cpu, None))
                    if self.backend == "cuda":
                        pair[-1] = (cpu, cpu.cuda(cache_bytes=32 << 20,
                                                 device=self.device.index or 0))
            except Exception:
                self._close_pair(pair)
                raise
            self.readers[key] = pair
        self.readers.move_to_end(key)
        return [gpu if gpu is not None else cpu for cpu, gpu in self.readers[key]]

    @staticmethod
    def _close_pair(pair):
        for cpu, gpu in pair:
            if gpu is not None:
                gpu.close()
            cpu.close()

    def close(self):
        for pair in self.readers.values():
            self._close_pair(pair)
        self.readers.clear()

    def _origin(self, case, rng, training):
        # Mix uniform coverage with coarse valid/foreground anchoring. Rejection
        # below checks actual labels; occupancy alone does not guarantee a hit.
        if training and rng.random() < 0.33:
            return tuple(int(rng.integers(max(1, n - c + 1))) if n >= c else (n - c) // 2
                         for n, c in zip(case["shape"], self.crop_size))
        blocks = case["foreground_blocks"] if training and rng.random() < 0.5 else []
        blocks = blocks or case["valid_blocks"]
        low, high = blocks[int(rng.integers(len(blocks)))]
        point = [int(rng.integers(a, b)) for a, b in zip(low, high)]
        return tuple(min(max(p - c // 2, 0), n - c) if n >= c else (n - c) // 2
                     for p, n, c in zip(point, case["shape"], self.crop_size))

    @staticmethod
    def _tensor(value):
        return torch.from_numpy(value) if isinstance(value, np.ndarray) else torch.from_dlpack(value)

    def sample(self, case, rng, *, training):
        image_reader, label_reader = self._readers_for(case)
        reader_kwargs = {"source": "chunks"} if self.backend == "cuda" else {}
        for _ in range(32):
            origin = self._origin(case, rng, training)
            label = label_reader.read_region(origin, self.crop_size, dtype=np.uint8, **reader_kwargs)
            label = label.reshape((1, *self.crop_size))
            if float((self._tensor(label) != 2).float().mean()) < self.min_valid_fraction:
                continue
            image = image_reader.read_region(origin, self.crop_size, dtype=np.float32, **reader_kwargs)
            sample = self.pipelines[training](
                {"image": image.reshape((1, *self.crop_size)), "label": label},
                seed=int(rng.integers(2**63 - 1)), inplace=True)
            tensors = {k: self._tensor(sample[k]) for k in ("image", "label", "label_skel")}
            return tensors, {"id": case["id"], "origin_zyx": list(origin)}
        raise RuntimeError(f"Could not find a valid crop in {case['id']}; lower --min-valid-fraction")

    def training_batch(self, rng, batch_size):
        samples = [self.sample(self.cases["train"][int(rng.integers(len(self.cases["train"])))],
                               rng, training=True)[0] for _ in range(batch_size)]
        return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}

    def validation(self, *, seed, crops_per_case=2, case_limit=None):
        # Fresh RNG each evaluation: same coordinates, no augmentation, no train RNG consumption.
        rng = np.random.default_rng(seed)
        cases = self.cases["val"][:case_limit] if case_limit else self.cases["val"]
        for case in cases:
            for _ in range(crops_per_case):
                sample, metadata = self.sample(case, rng, training=False)
                yield {k: v.unsqueeze(0) for k, v in sample.items()}, metadata
