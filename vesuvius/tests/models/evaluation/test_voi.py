from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


VESUVIUS_SRC = Path(__file__).resolve().parents[3] / "src" / "vesuvius"


def _load_metric_class():
    package_name = "_voi_metric_under_test"
    private_modules = (
        package_name,
        f"{package_name}.base_metric",
        f"{package_name}.voi",
    )
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.Tensor = object
    cc3d_stub = types.ModuleType("cc3d")

    def connected_components(mask, connectivity=26):
        labels = np.zeros_like(mask, dtype=np.int32)
        labels[np.asarray(mask).astype(bool)] = 1
        return labels

    cc3d_stub.connected_components = connected_components
    skimage_stub = types.ModuleType("skimage")
    skimage_metrics_stub = types.ModuleType("skimage.metrics")
    skimage_metrics_stub.variation_of_information = lambda a, b: (0.0, 0.0)
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "cc3d",
            "skimage",
            "skimage.metrics",
            *private_modules,
        )
    }
    try:
        sys.modules["torch"] = torch_stub
        sys.modules["cc3d"] = cc3d_stub
        sys.modules["skimage"] = skimage_stub
        sys.modules["skimage.metrics"] = skimage_metrics_stub
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
            f"{package_name}.voi",
            VESUVIUS_SRC / "models" / "evaluation" / "voi.py",
        )
        metric_module = importlib.util.module_from_spec(metric_spec)
        sys.modules[metric_spec.name] = metric_module
        metric_spec.loader.exec_module(metric_module)
        return metric_module.VOIMetric
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


VOIMetric = _load_metric_class()


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


class VOIMetricTest(unittest.TestCase):
    def test_batched_hard_label_volumes_are_not_argmaxed_as_channel_maps(self):
        metric = VOIMetric(ignore_index=0)
        labels = np.zeros((1, 3, 5, 5), dtype=np.float32)
        labels[0, 2, 2, 2] = 1

        result = metric.compute(FakeTensor(labels), FakeTensor(labels))

        self.assertEqual(result["voi_total"], 0.0)
        self.assertEqual(result["voi_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
