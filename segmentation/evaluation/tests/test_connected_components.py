from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


METRIC_PATH = Path(__file__).resolve().parents[1] / "metrics" / "connected_components.py"


def _load_compute():
    cc3d_stub = types.ModuleType("cc3d")

    def connected_components(mask, connectivity=26):
        labels = np.zeros_like(mask, dtype=np.int32)
        component_id = 0
        for index in zip(*np.nonzero(mask)):
            component_id += 1
            labels[index] = component_id
        return labels

    cc3d_stub.connected_components = connected_components
    saved_cc3d = sys.modules.get("cc3d")
    module_name = "_segmentation_connected_components_metric_under_test"
    saved_metric = sys.modules.get(module_name)
    try:
        sys.modules["cc3d"] = cc3d_stub
        spec = importlib.util.spec_from_file_location(module_name, METRIC_PATH)
        metric_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = metric_module
        spec.loader.exec_module(metric_module)
        return metric_module.compute
    finally:
        if saved_cc3d is None:
            sys.modules.pop("cc3d", None)
        else:
            sys.modules["cc3d"] = saved_cc3d
        if saved_metric is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_metric


compute = _load_compute()


def _labels_with_two_components(class_index: int) -> np.ndarray:
    labels = np.zeros((1, 3, 5, 5), dtype=np.int32)
    labels[0, 1, 1, 1] = class_index
    labels[0, 1, 3, 3] = class_index
    return labels


class ConnectedComponentsMetricTest(unittest.TestCase):
    def test_total_difference_sums_per_class_differences_without_class_cancellation(self):
        result = compute(
            label=_labels_with_two_components(class_index=2),
            prediction=_labels_with_two_components(class_index=1),
            num_classes=3,
            ignore_index=0,
        )

        self.assertEqual(result["connected_components_difference_class_1"], 2.0)
        self.assertEqual(result["connected_components_difference_class_2"], 2.0)
        self.assertEqual(result["connected_components_difference_total"], 4.0)

    def test_batched_hard_label_predictions_are_not_treated_as_probability_maps(self):
        labels = np.zeros((1, 4, 5, 5), dtype=np.int32)
        labels[0, 3, 2, 2] = 1
        prediction = labels.copy()

        result = compute(
            label=labels,
            prediction=prediction,
            num_classes=2,
            ignore_index=0,
        )

        self.assertEqual(result["connected_components_difference_class_1"], 0.0)
        self.assertEqual(result["connected_components_difference_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
