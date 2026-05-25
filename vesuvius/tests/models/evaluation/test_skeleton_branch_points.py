from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _metric_dependency_stubs() -> dict[str, types.ModuleType]:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.Tensor = object

    skimage_stub = types.ModuleType("skimage")
    morphology_stub = types.ModuleType("skimage.morphology")
    morphology_stub.skeletonize = lambda mask: mask
    skimage_stub.morphology = morphology_stub

    scipy_stub = types.ModuleType("scipy")
    ndimage_stub = types.ModuleType("scipy.ndimage")

    def convolve(values, kernel, mode="constant", cval=0):
        if mode != "constant":
            raise AssertionError(f"unexpected mode: {mode}")
        padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), constant_values=cval)
        result = np.zeros_like(values, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                if kernel[0, dy, dx]:
                    result += padded[:, dy : dy + values.shape[1], dx : dx + values.shape[2]]
        return result

    ndimage_stub.convolve = convolve
    scipy_stub.ndimage = ndimage_stub
    return {
        "torch": torch_stub,
        "skimage": skimage_stub,
        "skimage.morphology": morphology_stub,
        "scipy": scipy_stub,
        "scipy.ndimage": ndimage_stub,
    }

VESUVIUS_SRC = Path(__file__).resolve().parents[3] / "src" / "vesuvius"


def _load_metric_class():
    package_name = "_skeleton_branch_points_metric_under_test"
    private_modules = (
        package_name,
        f"{package_name}.base_metric",
        f"{package_name}.skeleton_branch_points",
    )
    stub_modules = _metric_dependency_stubs()
    saved_modules = {
        name: sys.modules.get(name)
        for name in (*stub_modules.keys(), *private_modules)
    }
    try:
        sys.modules.update(stub_modules)
        evaluation_pkg = types.ModuleType(package_name)
        evaluation_pkg.__path__ = [str(VESUVIUS_SRC / "models" / "evaluation")]
        sys.modules[package_name] = evaluation_pkg

        base_spec = importlib.util.spec_from_file_location(
            f"{package_name}.base_metric",
            VESUVIUS_SRC / "models" / "evaluation" / "base_metric.py",
        )
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[base_spec.name] = base_module
        base_spec.loader.exec_module(base_module)

        metric_spec = importlib.util.spec_from_file_location(
            f"{package_name}.skeleton_branch_points",
            VESUVIUS_SRC / "models" / "evaluation" / "skeleton_branch_points.py",
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        sys.modules[metric_spec.name] = metric_module
        metric_spec.loader.exec_module(metric_module)
        return metric_module.SkeletonBranchPointsMetric
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


SkeletonBranchPointsMetric = _load_metric_class()


class FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)
        self.dtype = np.float32

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def float(self):
        return self


def _branch_logits(class_index: int) -> np.ndarray:
    logits = np.zeros((1, 3, 1, 5, 5), dtype=np.float32)
    logits[:, 0, :, :, :] = 1.0
    for y, x in ((2, 2), (1, 2), (3, 2), (2, 1), (2, 3)):
        logits[0, class_index, 0, y, x] = 5.0
    return logits


class SkeletonBranchPointsMetricTest(unittest.TestCase):
    def test_total_absdiff_sums_per_class_differences_without_class_cancellation(self):
        metric = SkeletonBranchPointsMetric(num_classes=3, ignore_index=0)
        metric._skeletonize_stack_2d = lambda mask: mask.astype(np.uint8)

        result = metric.compute(
            FakeTensor(_branch_logits(class_index=1)),
            FakeTensor(_branch_logits(class_index=2)),
        )

        self.assertEqual(result["branch_points_absdiff_class_1"], 5.0)
        self.assertEqual(result["branch_points_absdiff_class_2"], 5.0)
        self.assertEqual(result["branch_points_pred_total"], 5.0)
        self.assertEqual(result["branch_points_gt_total"], 5.0)
        self.assertEqual(result["branch_points_absdiff_total"], 10.0)

    def test_batched_hard_label_volumes_are_not_argmaxed_as_channel_maps(self):
        metric = SkeletonBranchPointsMetric(num_classes=2, ignore_index=0)
        metric._skeletonize_stack_2d = lambda mask: mask.astype(np.uint8)
        labels = np.zeros((1, 3, 5, 5), dtype=np.float32)
        for y, x in ((2, 2), (1, 2), (3, 2), (2, 1), (2, 3)):
            labels[0, 2, y, x] = 1

        result = metric.compute(FakeTensor(labels), FakeTensor(labels))

        self.assertGreater(result["branch_points_pred_class_1"], 0.0)
        self.assertGreater(result["branch_points_gt_class_1"], 0.0)
        self.assertEqual(result["branch_points_absdiff_class_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
