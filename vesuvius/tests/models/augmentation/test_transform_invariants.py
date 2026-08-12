"""Invariant tests for vesuvius.models.augmentation transforms.

Every transform in the package must either appear in REGISTRY (and pass the
invariants that apply to it) or be listed in SKIPPED with a reason.
test_registry_is_complete enforces this, so new transforms cannot land
untested by accident.

Invariants:
  1. Ownership: a transform may work in place (returned tensor aliases the
     input) or out of place (input untouched), never both.
  2. ImageOnlyTransform subclasses leave segmentation bit-identical.
  3. Image dtype/shape/device are preserved (unless flagged).
  4. RandomTransform(t, apply_probability=0) is a bit-exact identity.
  5. With probabilities pinned to 1 the transform has an observable effect
     (unless flagged may_noop) - keeps the registry params honest.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, NamedTuple

import numpy as np
import pytest
import torch

from vesuvius.models.augmentation.transforms.base.basic_transform import (
    BasicTransform,
    ImageOnlyTransform,
)
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform

SEED = 1337


class Case(NamedTuple):
    id: str
    build: Callable[[], BasicTransform]
    dims: int = 3             # 2 -> (C, X, Y); 3 -> (C, X, Y, Z)
    image_only: bool = False  # derived: subclass of ImageOnlyTransform
    preserves_shape: bool = True
    seg_ok: bool = True       # feed a segmentation tensor alongside the image
    may_noop: str = ""        # non-empty reason exempts the case from invariant 5
    prepare: Callable[[dict], dict] | None = None  # shape the input data the transform needs


def _make_data(dims: int, seed: int = SEED) -> dict:
    g = torch.Generator().manual_seed(seed)
    # Spatial dims deliberately unequal on the leading axis so accidental axis
    # swaps change the shape; the trailing two are equal so in-plane rotations
    # and transposes stay shape-preserving.
    shape = (2, 24, 24) if dims == 2 else (2, 12, 16, 16)
    img = torch.rand(*shape, generator=g, dtype=torch.float32)
    seg = (torch.rand(*shape, generator=g) > 0.5).to(torch.int16)
    return {"image": img, "segmentation": seg}


def _prep_negative_seg_region(data: dict) -> dict:
    # MaskImageTransform masks where seg < 0 (nnU-Net "outside region" label).
    data["segmentation"][:, :4] = -1
    return data


def _prep_component_image(data: dict) -> dict:
    # RemoveRandomConnectedComponent... expects one-hot components in the
    # IMAGE channels (nnU-Net cascade layout): give it two separated blobs.
    img = torch.zeros_like(data["image"])
    img[0, 2:5, 2:5, 2:5] = 1.0
    img[0, 8:11, 8:11, 8:11] = 1.0
    data["image"] = img
    return data


# Constructor params chosen so the transform actually fires on small CPU
# tensors (probabilities pinned to 1 where the API allows it); invariant 5
# keeps these honest. Local-transform `scale` values are Gaussian sigmas in
# PIXELS (see local_transform.py), hence >= 4 for a 12-24 voxel patch.
def _registry() -> list[Case]:
    from vesuvius.models.augmentation.transforms.intensity.brightness import (
        BrightnessAdditiveTransform, MultiplicativeBrightnessTransform)
    from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform
    from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
    from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
    from vesuvius.models.augmentation.transforms.intensity.illumination import (
        InhomogeneousSliceIlluminationTransform)
    from vesuvius.models.augmentation.transforms.intensity.inversion import InvertImageTransform
    from vesuvius.models.augmentation.transforms.intensity.random_clip import CutOffOutliersTransform
    from vesuvius.models.augmentation.transforms.local.brightness_gradient import (
        BrightnessGradientAdditiveTransform)
    from vesuvius.models.augmentation.transforms.local.local_contrast import LocalContrastTransform
    from vesuvius.models.augmentation.transforms.local.local_gamma import LocalGammaTransform
    from vesuvius.models.augmentation.transforms.local.local_smoothing import LocalSmoothingTransform
    from vesuvius.models.augmentation.transforms.nnunet.random_binary_operator import (
        ApplyRandomBinaryOperatorTransform)
    from vesuvius.models.augmentation.transforms.nnunet.remove_connected_components import (
        RemoveRandomConnectedComponentFromOneHotEncodingTransform)
    from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import (
        BlankRectangleTransform, RicianNoiseTransform, SmearTransform)
    from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
    from vesuvius.models.augmentation.transforms.noise.median_filter import MedianFilterTransform
    from vesuvius.models.augmentation.transforms.noise.sharpen import SharpeningTransform
    from vesuvius.models.augmentation.transforms.spatial.low_resolution import (
        SimulateLowResolutionTransform)
    from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
    from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
    from vesuvius.models.augmentation.transforms.spatial.sheet_compression import (
        SheetCompressionTransform)
    from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
    from vesuvius.models.augmentation.transforms.spatial.thick_slice import (
        SimulateThickSliceTransform)
    from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
    from vesuvius.models.augmentation.transforms.utils.morphological_closing import (
        MorphologicalClosingTransform)
    from vesuvius.models.augmentation.transforms.utils.nnunet_masking import MaskImageTransform
    from vesuvius.models.augmentation.transforms.utils.remove_label import RemoveLabelTansform
    from vesuvius.models.augmentation.transforms.utils.seg_to_regions import (
        ConvertSegmentationToRegionsTransform)
    from vesuvius.models.augmentation.transforms.utils.skeleton_transform import (
        MedialSurfaceTransform)

    cases = [
        # --- intensity ---
        Case("BrightnessAdditive", lambda: BrightnessAdditiveTransform(mu=0.5, sigma=0.1)),
        Case("MultiplicativeBrightness",
             lambda: MultiplicativeBrightnessTransform((0.75, 1.25), synchronize_channels=False)),
        Case("Contrast",
             lambda: ContrastTransform((0.6, 0.9), preserve_range=True, synchronize_channels=False)),
        Case("Gamma",
             lambda: GammaTransform((0.7, 1.5), p_invert_image=1.0, synchronize_channels=False,
                                    p_per_channel=1.0, p_retain_stats=1.0)),
        Case("GaussianNoise", lambda: GaussianNoiseTransform(noise_variance=(0.05, 0.1))),
        Case("InhomogeneousSliceIllumination",
             lambda: InhomogeneousSliceIlluminationTransform(
                 num_defects=(1, 3), defect_width=(0.1, 0.3),
                 mult_brightness_reduction_at_defect=(0.3, 0.7),
                 base_p=(0.2, 0.4), base_red=(0.5, 0.9), p_per_channel=1.0)),
        Case("InvertImage", lambda: InvertImageTransform(p_invert_image=1.0)),
        Case("CutOffOutliers", lambda: CutOffOutliersTransform(p_per_channel=1.0)),
        # --- local (scale is a sigma in pixels) ---
        Case("BrightnessGradientAdditive",
             lambda: BrightnessGradientAdditiveTransform(scale=(4.0, 8.0), max_strength=(2.0, 3.0))),
        Case("LocalContrast",
             lambda: LocalContrastTransform(scale=(4.0, 8.0), new_contrast=(2.0, 3.0))),
        Case("LocalGamma", lambda: LocalGammaTransform(scale=(4.0, 8.0), gamma=(2.0, 3.0))),
        Case("LocalSmoothing",
             lambda: LocalSmoothingTransform(scale=(4.0, 8.0), smoothing_strength=(0.8, 1.0),
                                             kernel_size=(3.0, 5.0))),
        # --- noise ---
        Case("BlankRectangle",
             lambda: BlankRectangleTransform(rectangle_size=6, rectangle_value=0.5,
                                             num_rectangles=2, p_per_sample=1.0, p_per_channel=1.0)),
        Case("RicianNoise", lambda: RicianNoiseTransform(noise_variance=(0.05, 0.1), p_per_sample=1.0)),
        # Production params from pipelines/training_transforms.py:394
        Case("Smear3D", lambda: SmearTransform(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3)),
        Case("Smear2D", lambda: SmearTransform(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=1),
             dims=2),
        Case("GaussianBlur", lambda: GaussianBlurTransform(blur_sigma=(1.0, 2.0), benchmark=False)),
        Case("MedianFilter", lambda: MedianFilterTransform(filter_size=3, p_per_channel=1.0)),
        Case("Sharpening", lambda: SharpeningTransform(p_per_channel=1.0, p_clamp_intensities=1.0)),
        # --- spatial ---
        Case("SimulateLowResolution",
             lambda: SimulateLowResolutionTransform(scale=(0.4, 0.6), synchronize_channels=False,
                                                    synchronize_axes=False, ignore_axes=None)),
        Case("Mirror", lambda: MirrorTransform(allowed_axes=(0, 1, 2))),
        # In-plane (equal trailing dims) so the rotation is shape-preserving.
        Case("Rot90", lambda: Rot90Transform(allowed_axes={1, 2}),
             may_noop="num_rot_per_combination may sample a full 360-degree rotation"),
        Case("SheetCompression",
             lambda: SheetCompressionTransform(spatial_smoothing=(1.0, 2.0))),
        Case("Spatial",
             lambda: SpatialTransform(patch_size=(12, 16, 16), patch_center_dist_from_border=0,
                                      random_crop=False, p_rotation=1.0, rotation=(0.3, 0.6))),
        Case("SimulateThickSlice", lambda: SimulateThickSliceTransform()),
        Case("TransposeAxes", lambda: TransposeAxesTransform(allowed_axes={1, 2}),
             may_noop="may sample the identity permutation of the two allowed axes"),
        # --- seg utilities ---
        Case("ApplyRandomBinaryOperator", lambda: ApplyRandomBinaryOperatorTransform(channel_idx=0)),
        Case("RemoveRandomConnectedComponent",
             lambda: RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                 channel_idx=0, dont_do_if_covers_more_than_x_percent=1.0),
             prepare=_prep_component_image),
        Case("MorphologicalClosing", lambda: MorphologicalClosingTransform(structure_size=3)),
        Case("MaskImage", lambda: MaskImageTransform(apply_to_channels=[0], set_outside_to=0.5),
             prepare=_prep_negative_seg_region),
        Case("RemoveLabel", lambda: RemoveLabelTansform(label_value=1, set_to=0)),
        Case("ConvertSegmentationToRegions",
             lambda: ConvertSegmentationToRegionsTransform(regions=((1,),))),
        Case("MedialSurface", lambda: MedialSurfaceTransform()),
    ]
    # image_only flag is derived, not hand-maintained.
    return [c._replace(image_only=isinstance(c.build(), ImageOnlyTransform)) for c in cases]


REGISTRY = _registry()

SKIPPED: dict[str, str] = {
    "transforms.base.basic_transform.ImageOnlyTransform": "abstract base",
    "transforms.base.basic_transform.SegOnlyTransform": "abstract base",
    "transforms.utils.random.RandomTransform": "wrapper - covered by test_random_transform_p0_is_identity",
    "transforms.utils.compose.ComposeTransforms":
        "wrapper - covered by test_compose_and_oneof_preserve_ownership",
    "transforms.utils.oneoftransform.OneOfTransform":
        "wrapper - covered by test_compose_and_oneof_preserve_ownership",
    "transforms.utils.pseudo2d.Convert2DTo3DTransform": "reshapes by design (pseudo-2d plumbing)",
    "transforms.utils.pseudo2d.Convert3DTo2DTransform": "reshapes by design (pseudo-2d plumbing)",
    "transforms.utils.deep_supervision_downsampling.DownsampleSegForDSTransform":
        "returns a list of seg scales by design",
    "transforms.nnunet.seg_to_onehot.MoveSegAsOneHotToDataTransform":
        "moves channels between seg and image by design",
}


def _all_transform_classes():
    import vesuvius.models.augmentation as aug
    found = {}
    for m in pkgutil.walk_packages(aug.__path__, prefix="vesuvius.models.augmentation."):
        if ".pipelines" in m.name:
            continue
        try:
            mod = importlib.import_module(m.name)
        except ImportError:
            continue  # optional heavy deps may be absent locally; CI installs --extra all
        for name, cls in vars(mod).items():
            if (inspect.isclass(cls) and issubclass(cls, BasicTransform)
                    and cls is not BasicTransform and cls.__module__ == m.name):
                key = f"{cls.__module__.removeprefix('vesuvius.models.augmentation.')}.{name}"
                found[key] = cls
    return found


def test_registry_is_complete():
    registered = {type(c.build()).__name__ for c in REGISTRY}
    for key, cls in _all_transform_classes().items():
        assert cls.__name__ in registered or key in SKIPPED, (
            f"{key} is neither in REGISTRY nor SKIPPED - new transforms must be "
            f"covered by the invariant suite or explicitly waived with a reason")


def _build_inputs(case: Case) -> dict:
    data = _make_data(case.dims) if case.seg_ok else {"image": _make_data(case.dims)["image"]}
    if case.prepare is not None:
        data = case.prepare(data)
    return data


def _run(case: Case, data: dict) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return case.build()(**data)


@pytest.mark.parametrize("case", REGISTRY, ids=lambda c: c.id)
def test_ownership_is_coherent(case):
    """If a transform mutates the caller's tensor, it must return that same
    storage (the plain in-place style used across this module) - mutating the
    input while returning a different buffer leaves two diverging copies."""
    data = _build_inputs(case)
    snapshots = {k: v.clone() for k, v in data.items()}
    out = _run(case, data)
    for key, before in snapshots.items():
        caller_tensor = data[key]
        mutated = not torch.equal(caller_tensor, before)
        if mutated:
            returned = out.get(key)
            assert returned is not None, f"{case.id} mutated '{key}' but dropped it from the output"
            assert returned.data_ptr() == caller_tensor.data_ptr() and torch.equal(returned, caller_tensor), (
                f"{case.id} mutated the caller's '{key}' tensor (max abs delta "
                f"{(caller_tensor.to(torch.float32) - before.to(torch.float32)).abs().max().item():.4g}) "
                f"while returning a different buffer - hybrid in-place/out-of-place behaviour")


@pytest.mark.parametrize("case", [c for c in REGISTRY if c.image_only and c.seg_ok],
                         ids=lambda c: c.id)
def test_image_only_transform_leaves_segmentation_identical(case):
    data = _make_data(case.dims)
    seg_before = data["segmentation"].clone()
    out = _run(case, data)
    assert torch.equal(out["segmentation"], seg_before)


@pytest.mark.parametrize("case", [c for c in REGISTRY if c.preserves_shape],
                         ids=lambda c: c.id)
def test_image_dtype_shape_device_preserved(case):
    data = _build_inputs(case)
    img = data["image"]
    shape, dtype, device = img.shape, img.dtype, img.device
    out = _run(case, data)
    assert out["image"].shape == shape
    assert out["image"].dtype == dtype
    assert out["image"].device == device


@pytest.mark.parametrize("case", REGISTRY, ids=lambda c: c.id)
def test_random_transform_p0_is_identity(case):
    data = _build_inputs(case)
    snapshots = {k: v.clone() for k, v in data.items()}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out = RandomTransform(case.build(), apply_probability=0.0)(**data)
    for key, before in snapshots.items():
        assert torch.equal(out[key], before)
        assert torch.equal(data[key], before)


@pytest.mark.parametrize("case", [c for c in REGISTRY if not c.may_noop],
                         ids=lambda c: c.id)
def test_transform_has_effect_when_forced(case):
    """Registry params are chosen to make each transform fire, so the other
    invariants cannot pass vacuously on a no-op."""
    data = _build_inputs(case)
    snapshots = {k: v.clone() for k, v in data.items()}
    out = _run(case, data)
    changed = any(
        isinstance(v, torch.Tensor) and k not in snapshots for k, v in out.items()
    )  # transforms may act by ADDING a key (e.g. MedialSurface's *_skel)
    for key, before in snapshots.items():
        returned = out.get(key)
        if returned is None or returned.shape != before.shape or not torch.equal(returned, before):
            changed = True
            break
    assert changed, (f"{case.id} had no observable effect - registry params are "
                     f"dead, or the transform is broken; flag may_noop with a reason if intended")


def test_compose_and_oneof_preserve_ownership():
    from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
    from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
    from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
    from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform

    data = _make_data(3)
    snapshots = {k: v.clone() for k, v in data.items()}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pipeline = ComposeTransforms([
        OneOfTransform([MirrorTransform(allowed_axes=(0, 1, 2)),
                        Rot90Transform(allowed_axes={1, 2})]),
        MirrorTransform(allowed_axes=(0, 1, 2)),
    ])
    pipeline(**data)
    # Both members are out-of-place, so the caller's tensors must be untouched.
    for key, before in snapshots.items():
        assert torch.equal(data[key], before)
