"""Compare native augmentation with the actual Vesuvius Python transforms.

Single-threaded CPU, one channel, fixed real CT input, owning outputs for both
backends. File reads, construction, RNG reset, and correctness checks are untimed.
Python receives a timed clone because several existing transforms mutate inputs;
the native API already copies internally. No augmentation kernels are reimplemented.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch

from vc_reader import augment as A
from vc_reader.reader import ZarrReader

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vesuvius/src/vesuvius"
PREFIX = "vesuvius.models.augmentation.transforms."
DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")


def load(module, name, **kwargs):
    return getattr(importlib.import_module(PREFIX + module), name)(**kwargs)


@dataclass
class Case:
    name: str
    native: list
    legacy: list  # (module, class, keyword args)
    inputs: str = "image"
    comparison: str = "matched"
    fixed: dict | None = None
    note: str = ""


def cases(shape):
    def pair(name, native, module, cls, kwargs, **options):
        return Case(name, [native], [(module, cls, kwargs)], **options)

    result = [
        Case("CopyOnly", [], [], inputs="both"),
        pair("BrightnessMultiply", A.BrightnessMultiply(1.1), "intensity.brightness",
             "MultiplicativeBrightnessTransform", dict(multiplier_range=1.1, synchronize_channels=True)),
        pair("BrightnessAdd", A.BrightnessAdd(.1), "intensity.brightness",
             "BrightnessAdditiveTransform", dict(mu=.1, sigma=0)),
        pair("Contrast", A.Contrast(1.2), "intensity.contrast", "ContrastTransform",
             dict(contrast_range=1.2, preserve_range=True, synchronize_channels=True)),
        pair("Gamma", A.Gamma(.8), "intensity.gamma", "GammaTransform",
             dict(gamma=.8, p_invert_image=0, synchronize_channels=True, p_per_channel=1, p_retain_stats=0)),
        pair("Invert", A.Invert(), "intensity.inversion", "InvertImageTransform", dict(p_invert_image=1)),
        pair("GaussianNoise", A.GaussianNoise(.05), "intensity.gaussian_noise", "GaussianNoiseTransform",
             dict(noise_variance=.05), comparison="independent RNG", note="Both parameters are std=0.05."),
        pair("RicianNoise", A.RicianNoise(.05), "noise.extranoisetransforms", "RicianNoiseTransform",
             dict(noise_variance=(.0025, .0025)), comparison="independent RNG",
             note="Python takes variance=0.0025; native takes std=0.05."),
        pair("ClipOutliers", A.ClipOutliers(1, 99, p_retain_std=0), "intensity.random_clip",
             "CutOffOutliersTransform", dict(percentile_lower=1, percentile_upper=99, p_retain_std=0)),
        pair("GaussianBlur", A.GaussianBlur(1), "noise.gaussian_blur", "GaussianBlurTransform",
             dict(blur_sigma=1, synchronize_axes=True, benchmark=False),
             note="Python direct separable convolution; autotuning disabled, same radius=3."),
        pair("MedianFilter", A.MedianFilter(3), "noise.median_filter", "MedianFilterTransform", dict(filter_size=3)),
        pair("Sharpen", A.Sharpen(.25, p_clamp=0), "noise.sharpen", "SharpeningTransform",
             dict(strength=.25, p_clamp_intensities=0)),
        pair("BrightnessGradient", A.BrightnessGradient(12, loc=.5, strength=.3), "local.brightness_gradient",
             "BrightnessGradientAdditiveTransform", dict(scale=12, loc=.5, max_strength=.3)),
        pair("LocalContrast", A.LocalContrast(12, loc=.5, factor=1.2), "local.local_contrast",
             "LocalContrastTransform", dict(scale=12, loc=.5, new_contrast=1.2)),
        pair("LocalGamma", A.LocalGamma(12, loc=.5, gamma=.8), "local.local_gamma",
             "LocalGammaTransform", dict(scale=12, loc=.5, gamma=.8)),
        pair("LocalSmoothing", A.LocalSmoothing(12, loc=.5, strength=.7, sigma=1), "local.local_smoothing",
             "LocalSmoothingTransform", dict(scale=12, loc=.5, smoothing_strength=.7, kernel_size=1)),
        pair("BlankRectangle", A.BlankRectangle(16, value=0, count=3), "noise.extranoisetransforms",
             "BlankRectangleTransform", dict(rectangle_size=16, rectangle_value=0, num_rectangles=3,
                                             p_per_sample=1, p_per_channel=1), comparison="independent RNG"),
        pair("SliceIllumination", A.SliceIllumination(defects=3, width=4, reduction=.5, base_p=.1, base_reduction=.5),
             "intensity.illumination", "InhomogeneousSliceIlluminationTransform",
             dict(num_defects=3, defect_width=4, mult_brightness_reduction_at_defect=.5,
                  base_p=.1, base_red=(.5, .5), p_per_channel=1), comparison="independent RNG"),
        pair("Smear", A.Smear(shift=(3, 0), alpha=.25), "noise.extranoisetransforms", "SmearTransform",
             dict(shift=(3, 0), alpha=.25)),
        pair("Flip", A.Flip(axes="x", p_axis=1), "spatial.mirroring", "MirrorTransform",
             dict(allowed_axes=(2,)), inputs="both", fixed=dict(axes=[2]),
             note="Python apply with fixed axes; native p_axis=1. No random no-op draws."),
        pair("Rotate90", A.Rotate90(planes=("yx",), turns=(1,)), "spatial.rot90", "Rot90Transform",
             dict(allowed_axes={1, 2}, num_rot_per_combination=(1,)), inputs="both"),
        pair("Transpose", A.Transpose("zxy"), "spatial.transpose", "TransposeAxesTransform",
             dict(allowed_axes={1, 2}), inputs="both", fixed=dict(axis_order=[0, 1, 3, 2]),
             note="Python apply with fixed axis order; native explicit order. No random no-op draws."),
        pair("CropPad", A.CropPad(tuple(s - 8 for s in shape)), "spatial.spatial", "SpatialTransform",
             dict(patch_size=tuple(s - 8 for s in shape), patch_center_dist_from_border=0, random_crop=False), inputs="both"),
        pair("Affine", A.Affine(scale=1 / 1.1), "spatial.spatial", "SpatialTransform",
             dict(patch_size=shape, patch_center_dist_from_border=0, random_crop=False,
                  p_scaling=1, scaling=1.1, p_synchronize_scaling_across_axes=1, mode_seg="nearest"), inputs="both",
             note="Centered scaling only; native forward scale=1/1.1, Python sampling scale=1.1."),
        pair("Elastic", A.Elastic(sigma=3, magnitude=1), "spatial.spatial", "SpatialTransform",
             dict(patch_size=shape, patch_center_dist_from_border=0, random_crop=False,
                  p_elastic_deform=1, elastic_deform_scale=3 / shape[0], elastic_deform_magnitude=1,
                  p_synchronize_def_scale_across_axes=1, mode_seg="nearest"),
             inputs="both", comparison="independent RNG"),
        pair("SheetCompression", A.SheetCompression(strength=.2, intensity_sigma=1, smoothing=3, axes="z"),
             "spatial.sheet_compression", "SheetCompressionTransform",
             dict(compression_strength=.2, intensity_sigma=1, spatial_smoothing=3, compression_axes=(0,)), inputs="both"),
        pair("LowResolution", A.LowResolution(.5), "spatial.low_resolution", "SimulateLowResolutionTransform",
             dict(scale=.5, synchronize_channels=True, synchronize_axes=True, ignore_axes=None)),
        pair("ThickSlice", A.ThickSlice(.5, axes="z"), "spatial.thick_slice", "SimulateThickSliceTransform",
             dict(scale_range=(.5, .5), candidate_axes=(0,)), inputs="both"),
        pair("Morphology", A.Morphology(radius=1, footprint="repeated_cross", on="segmentation"),
             "utils.morphological_closing", "MorphologicalClosingTransform", dict(structure_size=1), inputs="label"),
        pair("RemoveComponents", A.RemoveComponents(channels=(0,), p_fill=0), "nnunet.remove_connected_components",
             "RemoveRandomConnectedComponentFromOneHotEncodingTransform",
             dict(channel_idx=(0,), fill_with_other_class_p=0), inputs="binary", comparison="independent RNG"),
        pair("ReplaceLabel", A.ReplaceLabel(-1, 0, on="segmentation"), "utils.remove_label",
             "RemoveLabelTansform", dict(label_value=-1, set_to=0), inputs="ignore_label"),
        pair("Regions", A.Regions([0, 1], on="segmentation"), "utils.seg_to_regions",
             "ConvertSegmentationToRegionsTransform", dict(regions=[0, 1]), inputs="label"),
        pair("OneHot", A.OneHot([0, 1], on="segmentation", append_to="image"), "nnunet.seg_to_onehot",
             "MoveSegAsOneHotToDataTransform", dict(source_channel_idx=0, all_labels=[0, 1], remove_channel_from_source=False), inputs="both"),
        pair("MaskImage", A.MaskImage(mask="segmentation"), "utils.nnunet_masking", "MaskImageTransform",
             dict(apply_to_channels=[0]), inputs="masked"),
        pair("DeepSupervision", A.DeepSupervision([1, .5, .25], on="segmentation"), "utils.deep_supervision_downsampling",
             "DownsampleSegForDSTransform", dict(ds_scales=[1, .5, .25]), inputs="label"),
        pair("MedialSurface", A.MedialSurface(on="segmentation"), "utils.skeleton_transform",
             "MedialSurfaceTransform", dict(target_keys=["segmentation"]), inputs="label",
             comparison="algorithm comparison", note="Native Zhang-Suen vs scikit-image; exact equivalence not assumed."),
        Case("Pseudo2DRoundTrip", [A.ToPseudo2D(), A.FromPseudo2D()],
             [("utils.pseudo2d", "Convert3DTo2DTransform", {}), ("utils.pseudo2d", "Convert2DTo3DTransform", {})], inputs="both"),
    ]
    # The actual Python morphology callable is supplied after importing its module.
    result.append(pair("RandomBinaryOperator", A.RandomBinaryOperator(channels=(0,), radius=1, operations=("dilate",)),
                       "nnunet.random_binary_operator", "ApplyRandomBinaryOperatorTransform",
                       dict(channel_idx=(0,), strel_size=1), inputs="binary",
                       note="Both forced to dilation, radius=1; Python uses FFT convolution."))
    brightness = result[1]
    gamma = next(c for c in result if c.name == "Gamma")
    blur = next(c for c in result if c.name == "GaussianBlur")
    result.append(Case("IntensityPipeline", brightness.native + gamma.native + blur.native,
                       brightness.legacy + gamma.legacy + blur.legacy))
    return result


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten(sample):
    result = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().numpy()
        elif isinstance(value, (list, tuple)):
            for i, level in enumerate(value):
                result[f"{key}/{i}"] = level.detach().numpy()
    return result


def compare(a, b):
    a, b = flatten(a), flatten(b)
    result = {}
    for key in a.keys() | b.keys():
        if key not in a or key not in b or a[key].shape != b[key].shape:
            result[key] = {"shape_match": False}
            continue
        delta = np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))
        result[key] = dict(shape=list(a[key].shape), exact=bool(np.array_equal(a[key], b[key])),
                           close=bool(np.allclose(a[key], b[key], rtol=2e-5, atol=2e-6)),
                           max_abs=float(delta.max(initial=0)), mean_abs=float(delta.mean()),
                           unequal_fraction=float(np.mean(a[key] != b[key])))
    return result


def summary(values):
    values = np.array(values) * 1000
    return dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(values.min()), max_ms=float(values.max()),
                samples_ms=values.tolist())


def filter_cases(shape):
    """Exercise histogram/fallback median paths and several Gaussian widths."""
    base = {case.name: case for case in cases(shape)}
    result = [base["GaussianBlur"], base["MedianFilter"]]
    for sigma in (.5, 2., 4.):
        result.append(Case(f"GaussianBlurSigma{sigma:g}", [A.GaussianBlur(sigma)],
                           [("noise.gaussian_blur", "GaussianBlurTransform",
                             dict(blur_sigma=sigma, synchronize_axes=True, benchmark=False))]))
    for size in (4, 5, 7):
        result.append(Case(f"MedianFilterSize{size}", [A.MedianFilter(size)],
                           [("noise.median_filter", "MedianFilterTransform", dict(filter_size=size))]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--start", type=int, nargs=3, default=(1280, 2688, 3200))
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--filter-sweep", action="store_true", help="compare only median/blur with several kernel widths")
    parser.add_argument("--input-noise-std", type=float, default=0., help="fixed Gaussian perturbation of the real CT input; tests continuous-valued filters")
    parser.add_argument("--read-each-call", action="store_true", help="time fresh vc_reader float32 reads, normalization and augmentation together")
    parser.add_argument("--inplace", action="store_true", help="reuse freshly read native buffers; requires --read-each-call")
    parser.add_argument("--only", help="comma-separated case names")
    parser.add_argument("--profile", help="comma-separated cases to profile separately after timing")
    parser.add_argument("--output", type=Path, default=Path("/tmp/vc-augment-benchmark.json"))
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/vc-augment-build"), help="CMake metadata for the measured native build")
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 0 or len(set(args.shape)) != 1 or min(args.shape) <= 8:
        parser.error("use a cubic patch larger than 8, positive reps and nonnegative warmup")

    if args.inplace and not args.read_each_call:
        parser.error("--inplace requires --read-each-call so each call receives fresh inputs")
    if not np.isfinite(args.input_noise_std) or args.input_noise_std < 0:
        parser.error("--input-noise-std must be finite and nonnegative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # The Vesuvius root initializer imports volume/download dependencies unrelated
    # to augmentation. Use its source package path; all transform modules and
    # their imports execute unchanged, including real optional dependencies.
    if "vesuvius" not in sys.modules:
        package = types.ModuleType("vesuvius")
        package.__path__ = [str(SOURCE)]
        sys.modules["vesuvius"] = package

    with ZarrReader.open(args.array, threads=1) as reader:
        image = reader.read(tuple(args.start), tuple(args.shape)).astype(np.float32)[None] / 255
    assert image.max() > image.min(), "choose a crop containing CT signal"
    perturbation = None
    if args.input_noise_std:
        perturbation = np.random.default_rng(123).normal(0, args.input_noise_std, image.shape).astype(np.float32)
        image += perturbation
    image = torch.from_numpy(image)
    labels = (image > .5).to(torch.int16)
    ignore = labels.clone()
    ignore[:, :4] = -1
    inputs = {"image": {"image": image}, "both": {"image": image, "segmentation": labels},
              "label": {"segmentation": labels}, "ignore_label": {"segmentation": ignore},
              "masked": {"image": image, "segmentation": ignore}, "binary": {"image": labels.float()}}
    fresh_reader = ZarrReader.open(args.array, threads=1) if args.read_each_call else None

    def fresh_sample(kind):
        patch = fresh_reader.read(tuple(args.start), tuple(args.shape), dtype=np.float32)[None]
        patch /= 255
        if perturbation is not None:
            patch += perturbation
        image = torch.from_numpy(patch)
        if kind == "image":
            return {"image": image}
        labels = (image > .5).to(torch.int16)
        if kind in ("ignore_label", "masked"):
            labels[:, :4] = -1
        if kind == "binary":
            return {"image": labels.float()}
        result = {"segmentation": labels}
        if kind in ("both", "masked"):
            result["image"] = image
        return result

    import vc_reader._augment as extension
    cpu_info = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpu_info.exists() else platform.processor()
    report = dict(command=sys.argv, python=sys.version, platform=platform.platform(), cpu=cpu,
                  logical_cpus=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  torch=torch.__version__, numpy=np.__version__,
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  native_extension=extension.__file__, native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  array=args.array, start=args.start, shape=args.shape, dtype="float32", channels=1,
                  labels="synthetic CT > 0.5 after dividing uint8 by 255; int16; not ground truth",
                  warmup=args.warmup, reps=args.reps, seed=123, ownership="Python timed input clone; native internal copy",
                  cases=[], excluded=["StripGeometry, SurfacePerturb and translate_crop_box: specialized tracing APIs outside this models benchmark",
                                     "OneOf/probability wrappers: no standalone kernel; not separately timed"])
    report["input_noise_std"] = args.input_noise_std
    report["filter_sweep"] = args.filter_sweep
    report["read_each_call"] = args.read_each_call
    report["inplace"] = args.inplace
    if args.read_each_call:
        import vc_reader._native as reader_extension
        report["reader_extension"] = reader_extension.__file__
        report["reader_sha256"] = hashlib.sha256(Path(reader_extension.__file__).read_bytes()).hexdigest()
        report["ownership"] = "Both read fresh vc_reader float32 buffers; Python mutates them; native inplace=" + str(args.inplace)
    report["dependencies"] = {name: importlib.metadata.version(name) for name in
                              ("scipy", "scikit-image", "fft-conv-pytorch", "acvl-utils")}
    flags = args.build_dir / "CMakeFiles/_augment.dir/flags.make"
    cache = args.build_dir / "CMakeCache.txt"
    report["native_build_flags"] = flags.read_text() if flags.exists() else None
    report["native_build_type"] = next((line.split("=", 1)[1] for line in cache.read_text().splitlines()
                                       if line.startswith("CMAKE_BUILD_TYPE:")), None) if cache.exists() else None
    report["native_matches_build"] = any(hashlib.sha256(p.read_bytes()).hexdigest() == report["native_sha256"]
                                          for p in args.build_dir.glob("_augment*.so"))
    selected = set(args.only.split(",")) if args.only else None
    profile_cases = set(args.profile.split(",")) if args.profile else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"CPU: {cpu}; Torch {torch.__version__}; one thread; shape={args.shape}; reps={args.reps}", flush=True)
    print(f"{'transform':24} {'Python ms':>11} {'C++ ms':>11} {'Python/C++':>11}  comparison", flush=True)
    with torch.inference_mode():
        for case in (filter_cases(tuple(args.shape)) if args.filter_sweep else cases(tuple(args.shape))):
            if selected and case.name not in selected:
                continue
            row = dict(name=case.name, input=case.inputs, comparison=case.comparison, note=case.note,
                       native=[repr(t) for t in case.native], python_spec=repr(case.legacy), fixed_python_parameters=case.fixed)
            try:
                legacy = [load(module, name, **kw) for module, name, kw in case.legacy]
                if case.name == "RandomBinaryOperator":
                    legacy[0].any_of_these = (importlib.import_module(PREFIX + "nnunet.random_binary_operator").binary_dilation_torch,)
                sample = inputs[case.inputs]
                native = A.Pipeline(case.native, targets={"image": A.Image(), "segmentation": A.Label()})

                def python_call():
                    value = fresh_sample(case.inputs) if fresh_reader else {key: tensor.clone() for key, tensor in sample.items()}
                    for transform in legacy:
                        value = transform.apply(value, **case.fixed) if case.fixed is not None else transform(**value)
                    return value

                def native_call():
                    value = fresh_sample(case.inputs) if fresh_reader else sample
                    return native(value, seed=123, **({"inplace": True} if args.inplace else {}))

                before = {k: v.clone() for k, v in sample.items()}
                reset_rng(123)
                reference, actual = python_call(), native_call()
                row["difference"] = compare(reference, actual)
                for key, tensor in sample.items():
                    assert torch.equal(tensor, before[key]), "benchmark mutated its input"
                assert all(np.isfinite(v).all() for v in flatten(actual).values()), "nonfinite native output"
                row["outputs_close"] = all(v.get("close", False) for v in row["difference"].values())
                for _ in range(args.warmup):
                    reset_rng(123)
                    python_call()
                    native_call()
                timings = {"python": [], "native": []}
                for iteration in range(args.reps):
                    order = [("python", python_call), ("native", native_call)]
                    if iteration % 2:
                        order.reverse()
                    for name, function in order:
                        reset_rng(123)
                        start = time.perf_counter()
                        output = function()
                        timings[name].append(time.perf_counter() - start)
                        del output
                row.update({name: summary(values) for name, values in timings.items()})
                row["python_over_native"] = row["python"]["p50_ms"] / row["native"]["p50_ms"]
                status = "close" if row["outputs_close"] else ("DIFF" if case.comparison == "matched" else case.comparison)
                print(f"{case.name:24} {row['python']['p50_ms']:11.3f} {row['native']['p50_ms']:11.3f} {row['python_over_native']:11.2f}  {status}", flush=True)
                if case.name in profile_cases:
                    for name, function in [("python", python_call), ("native", native_call)]:
                        profiler = cProfile.Profile()
                        for _ in range(3):
                            reset_rng(123)
                            profiler.runcall(function)
                        destination = args.output.with_name(args.output.stem + f"-{case.name}-{name}.pstats")
                        profiler.dump_stats(destination)
                        with destination.with_suffix(".txt").open("w") as out:
                            pstats.Stats(profiler, stream=out).strip_dirs().sort_stats("cumtime").print_stats(20)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{case.name:24} ERROR {row['error']}", flush=True)
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if fresh_reader:
        report["reader_stats"] = fresh_reader.stats()
        fresh_reader.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
