from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


VESUVIUS_SRC = Path(__file__).resolve().parents[3] / "src" / "vesuvius"


def _metric_dependency_stubs() -> dict[str, types.ModuleType]:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.Tensor = object

    cc3d_stub = types.ModuleType("cc3d")

    def connected_components(mask, connectivity=26):
        labels = np.zeros_like(mask, dtype=np.int32)
        component_id = 0
        for index in zip(*np.nonzero(mask)):
            component_id += 1
            labels[index] = component_id
        return labels

    cc3d_stub.connected_components = connected_components
    return {
        "torch": torch_stub,
        "cc3d": cc3d_stub,
    }


def _load_metric_class():
    package_name = "_connected_components_metric_under_test"
    private_modules = (
        package_name,
        f"{package_name}.base_metric",
        f"{package_name}.connected_components",
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
            f"{package_name}.connected_components",
            VESUVIUS_SRC / "models" / "evaluation" / "connected_components.py",
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        sys.modules[metric_spec.name] = metric_module
        metric_spec.loader.exec_module(metric_module)
        return metric_module.ConnectedComponentsMetric
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


ConnectedComponentsMetric = _load_metric_class()


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


def _labels_with_two_components(class_index: int) -> np.ndarray:
    labels = np.zeros((1, 1, 3, 5, 5), dtype=np.float32)
    labels[0, 0, 1, 1, 1] = class_index
    labels[0, 0, 1, 3, 3] = class_index
    return labels


class ConnectedComponentsMetricTest(unittest.TestCase):
    def test_total_difference_sums_per_class_differences_without_class_cancellation(self):
        metric = ConnectedComponentsMetric(num_classes=3, ignore_index=0)

        result = metric.compute(
            FakeTensor(_labels_with_two_components(class_index=1)),
            FakeTensor(_labels_with_two_components(class_index=2)),
        )

        self.assertEqual(result["connected_components_difference_class_1"], 2.0)
        self.assertEqual(result["connected_components_difference_class_2"], 2.0)
        self.assertEqual(result["connected_components_difference_total"], 4.0)

    def test_matching_channel_first_2d_maps_still_use_channel_argmax(self):
        metric = ConnectedComponentsMetric(num_classes=2, ignore_index=0)
        prediction = np.zeros((1, 2, 5, 5), dtype=np.float32)
        ground_truth = np.zeros_like(prediction)
        prediction[:, 0, :, :] = 1.0
        ground_truth[:, 0, :, :] = 1.0
        ground_truth[0, 0, 2, 2] = 0.0
        ground_truth[0, 1, 2, 2] = 1.0

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertEqual(result["connected_components_difference_class_1"], 1.0)
        self.assertEqual(result["connected_components_difference_total"], 1.0)

    def test_batched_hard_label_predictions_are_not_treated_as_channel_maps(self):
        metric = ConnectedComponentsMetric(num_classes=2, ignore_index=0)
        labels = np.zeros((1, 3, 5, 5), dtype=np.float32)
        labels[0, 2, 2, 2] = 1

        result = metric.compute(FakeTensor(labels), FakeTensor(labels))

        self.assertEqual(result["connected_components_difference_class_1"], 0.0)
        self.assertEqual(result["connected_components_difference_total"], 0.0)

    def test_hard_label_prediction_matches_singleton_channel_ground_truth(self):
        metric = ConnectedComponentsMetric(num_classes=2, ignore_index=0)
        prediction = np.zeros((1, 3, 5, 5), dtype=np.float32)
        ground_truth = prediction[:, np.newaxis, :, :, :].copy()
        prediction[0, 2, 2, 2] = 1
        ground_truth[0, 0, 2, 2, 2] = 1

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertEqual(result["connected_components_difference_class_1"], 0.0)
        self.assertEqual(result["connected_components_difference_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
