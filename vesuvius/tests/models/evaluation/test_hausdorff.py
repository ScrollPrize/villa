from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


VESUVIUS_SRC = Path(__file__).resolve().parents[3] / "src" / "vesuvius"


def _load_metric_class():
    package_name = "_hausdorff_metric_under_test"
    private_modules = (
        package_name,
        f"{package_name}.base_metric",
        f"{package_name}.hausdorff",
    )
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.Tensor = object
    scipy_stub = types.ModuleType("scipy")
    scipy_ndimage_stub = types.ModuleType("scipy.ndimage")
    scipy_ndimage_stub.distance_transform_edt = lambda mask: np.asarray(mask, dtype=np.float32)
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("torch", "scipy", "scipy.ndimage", *private_modules)
    }
    try:
        sys.modules["torch"] = torch_stub
        sys.modules["scipy"] = scipy_stub
        sys.modules["scipy.ndimage"] = scipy_ndimage_stub
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
            f"{package_name}.hausdorff",
            VESUVIUS_SRC / "models" / "evaluation" / "hausdorff.py",
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        sys.modules[metric_spec.name] = metric_module
        metric_spec.loader.exec_module(metric_module)
        return metric_module.HausdorffDistanceMetric
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


HausdorffDistanceMetric = _load_metric_class()


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


class HausdorffDistanceMetricTest(unittest.TestCase):
    def test_matching_channel_first_2d_maps_still_use_channel_argmax(self):
        metric = HausdorffDistanceMetric(num_classes=2, ignore_index=0)
        prediction = np.zeros((1, 2, 5, 5), dtype=np.float32)
        ground_truth = np.zeros_like(prediction)
        prediction[:, 0, :, :] = 1.0
        ground_truth[:, 0, :, :] = 1.0
        ground_truth[0, 0, 2, 2] = 0.0
        ground_truth[0, 1, 2, 2] = 1.0

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertTrue(np.isinf(result["hausdorff_distance_class_1"]))
        self.assertTrue(np.isinf(result["hausdorff_distance_95_class_1"]))

    def test_unbatched_hard_label_volumes_are_not_argmaxed_as_channel_maps(self):
        metric = HausdorffDistanceMetric(num_classes=2, ignore_index=0)
        prediction = np.zeros((3, 5, 5), dtype=np.float32)
        ground_truth = prediction.copy()
        ground_truth[2, 2, 2] = 1

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertTrue(np.isinf(result["hausdorff_distance_class_1"]))
        self.assertTrue(np.isinf(result["hausdorff_distance_95_class_1"]))

    def test_batched_hard_label_volumes_are_not_argmaxed_as_channel_maps(self):
        metric = HausdorffDistanceMetric(num_classes=2, ignore_index=0)
        prediction = np.zeros((1, 3, 5, 5), dtype=np.float32)
        ground_truth = prediction.copy()
        ground_truth[0, 2, 2, 2] = 1

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertTrue(np.isinf(result["hausdorff_distance_class_1"]))
        self.assertTrue(np.isinf(result["hausdorff_distance_95_class_1"]))


if __name__ == "__main__":
    unittest.main()
