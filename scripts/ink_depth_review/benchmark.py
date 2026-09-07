"""Exercise the official label writer and time suggestions on the same real crop."""

import argparse
import importlib.util
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from depth_review import load_volume, suggest

p = argparse.ArgumentParser()
p.add_argument("--villa", type=Path, required=True)
p.add_argument("--sample", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
source = a.villa / "vesuvius/src/vesuvius/label_zarr.py"
spec = importlib.util.spec_from_file_location("official_label_zarr", source)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
image = load_volume(a.sample / "image.tif")
prior = load_volume(a.sample / "prior.tif")
with tempfile.TemporaryDirectory() as temp:
    group = o.create_label_group(Path(temp) / "roundtrip.zarr", levels=1)
    array = o.create_label_array(group, "0", shape=prior.shape, dtype=prior.dtype)
    array[o.LABEL_SLICE] = prior[o.LABEL_SLICE]
    equal = np.array_equal(array[:], prior)
    if not equal:
        raise RuntimeError(
            "Official single-slice convention does not reproduce this prior."
        )
times = []
for _ in range(5):
    start = time.perf_counter()
    candidate, report = suggest(image, prior)
    times.append(time.perf_counter() - start)
result = {
    "official_commit": subprocess.check_output(
        ["git", "-C", str(a.villa), "rev-parse", "HEAD"], text=True
    ).strip(),
    "official_module": str(source.relative_to(a.villa)),
    "executed": ["create_label_group", "create_label_array"],
    "official_roundtrip_voxel_equal": bool(equal),
    "official_LABEL_SLICE": o.LABEL_SLICE,
    "sample_voxels": int(image.size),
    "candidate_voxels": int(candidate.sum()),
    "candidate_regions": len(report["components"]),
    "search_support_voxels": report["search_support_voxels"],
    "candidate_fraction_of_search_support": float(
        candidate.sum() / report["search_support_voxels"]
    ),
    "seconds_5_runs": times,
    "seconds_median": float(np.median(times)),
    "precision_recall": "not measured; no verified 3D ground truth",
    "human_time_saved": "not measured",
}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
