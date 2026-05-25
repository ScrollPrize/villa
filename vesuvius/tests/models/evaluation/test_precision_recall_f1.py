from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


VESUVIUS_SRC = Path(__file__).resolve().parents[3] / "src" / "vesuvius"


def _load_metric_class():
    package_name = "_precision_recall_f1_metric_under_test"
    private_modules = (
        package_name,
        f"{package_name}.base_metric",
        f"{package_name}.precision_recall_f1",
    )
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.Tensor = object
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("torch", *private_modules)
    }
    try:
        sys.modules["torch"] = torch_stub
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
            f"{package_name}.precision_recall_f1",
            VESUVIUS_SRC / "models" / "evaluation" / "precision_recall_f1.py",
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        sys.modules[metric_spec.name] = metric_module
        metric_spec.loader.exec_module(metric_module)
        return metric_module.PrecisionRecallF1Metric
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


PrecisionRecallF1Metric = _load_metric_class()


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


class PrecisionRecallF1MetricTest(unittest.TestCase):
    def test_matching_channel_first_2d_maps_still_use_channel_argmax(self):
        metric = PrecisionRecallF1Metric(num_classes=2, ignore_index=0)
        prediction = np.zeros((1, 2, 5, 5), dtype=np.float32)
        ground_truth = np.zeros_like(prediction)
        prediction[:, 0, :, :] = 1
        ground_truth[:, 0, :, :] = 1

        result = metric.compute(FakeTensor(prediction), FakeTensor(ground_truth))

        self.assertLess(result["precision_class_1"], 1e-5)
        self.assertLess(result["recall_class_1"], 1e-5)
        self.assertLess(result["f1_class_1"], 1e-5)

    def test_batched_hard_label_volumes_are_not_argmaxed_as_channel_maps(self):
        metric = PrecisionRecallF1Metric(num_classes=2, ignore_index=0)
        labels = np.zeros((1, 3, 5, 5), dtype=np.float32)
        labels[0, 2, 2, 2] = 1

        result = metric.compute(FakeTensor(labels), FakeTensor(labels))

        self.assertGreater(result["precision_class_1"], 0.99)
        self.assertGreater(result["recall_class_1"], 0.99)
        self.assertGreater(result["f1_class_1"], 0.99)


if __name__ == "__main__":
    unittest.main()
