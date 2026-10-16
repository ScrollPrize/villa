"""Validate read -> native augmentation on a real local image and optional labels.

This is a correctness harness, not a throughput benchmark. No network or writes
to the input dataset. Optional output is a NumPy archive for visual inspection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from vc_reader import augment as A
from vc_reader.reader import ZarrReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("labels", type=Path, nargs="?")
    parser.add_argument("--start", type=int, nargs="+", required=True)
    parser.add_argument("--shape", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.image.is_dir() or (args.labels and not args.labels.is_dir()):
        parser.error("provide local Zarr array directories")
    if len(args.shape) not in (2, 3) or len(args.start) != len(args.shape) or args.shape[-1] != args.shape[-2]:
        parser.error("use 2D/3D crops with a square YX plane and matching start/shape dimensions")

    image = ZarrReader.open(str(args.image)).read(tuple(args.start), tuple(args.shape), dtype=np.float32)[None]
    sample = {"image": image}
    if args.labels:
        sample["mask"] = ZarrReader.open(str(args.labels)).read(tuple(args.start), tuple(args.shape))[None]
        assert np.count_nonzero(sample["mask"]), "choose a crop containing supervised labels"
    original = {k: v.copy() for k, v in sample.items()}
    geometry = [A.Flip(axes="yx", p_axis=1), A.Rotate90(planes=("yx",), turns=(1,))]
    pipeline = A.Pipeline([*geometry, A.Gamma(gamma=.8), A.GaussianNoise(std=.25)])
    plan = pipeline.sample(sample, seed=args.seed)
    result = pipeline(sample, seed=args.seed)
    replay = plan.apply(sample)
    expected_image = np.rot90(np.flip(image, axis=(-2, -1)), axes=(-2, -1))
    geometry_result = A.Pipeline(geometry)(sample, seed=args.seed)
    np.testing.assert_array_equal(geometry_result["image"], expected_image)
    if args.labels:
        np.testing.assert_array_equal(result["mask"], np.rot90(np.flip(sample["mask"], axis=(-2, -1)), axes=(-2, -1)))
    for key in sample:
        np.testing.assert_array_equal(result[key], replay[key])
        assert np.isfinite(result[key]).all()
        np.testing.assert_array_equal(sample[key], original[key])
    assert image.max() > image.min(), "choose a crop containing CT signal"

    report = dict(image=str(args.image), labels=str(args.labels) if args.labels else None, start=args.start,
                  shape=args.shape, seed=args.seed,
                  image_dtype=str(image.dtype),
                  image_range=[float(image.min()), float(image.max())],
                  geometry_exact=True, replay_exact=True, inputs_unchanged=True)
    if args.labels:
        report.update(labels_exact=True, foreground_voxels=int(np.count_nonzero(sample["mask"])),
                      label_dtype=str(sample["mask"].dtype))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{key}_before": value for key, value in sample.items()}
        arrays.update({f"{key}_after": result[key] for key in sample})
        np.savez_compressed(args.output, **arrays, report=json.dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
