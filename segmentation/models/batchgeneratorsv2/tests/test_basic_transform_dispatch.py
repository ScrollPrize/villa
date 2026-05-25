import pathlib
import sys
import types
import unittest

import numpy as np

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = np.ndarray
sys.modules.setdefault("torch", torch_stub)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class DispatchProbeTransform(BasicTransform):
    def _apply_to_image(self, img, **params):
        return img + 1

    def _apply_to_regr_target(self, regression_target, **params):
        return regression_target + 100

    def _apply_to_segmentation(self, segmentation, **params):
        return segmentation + 10

    def _apply_to_dist_map(self, dist_map, **params):
        return dist_map + 1000

    def _apply_to_geols_labels(self, geols_labels, **params):
        return geols_labels + 10000


class BasicTransformDispatchTests(unittest.TestCase):
    def test_regression_target_uses_regression_dispatch_hook(self):
        sample = {
            "image": np.array([1.0]),
            "regression_target": np.array([1.0]),
            "segmentation": np.array([1.0]),
            "dist_map": np.array([1.0]),
            "geols_labels": np.array([1.0]),
        }

        out = DispatchProbeTransform()(**sample)

        np.testing.assert_allclose(out["image"], np.array([2.0]))
        np.testing.assert_allclose(out["regression_target"], np.array([101.0]))
        np.testing.assert_allclose(out["segmentation"], np.array([11.0]))
        np.testing.assert_allclose(out["dist_map"], np.array([1001.0]))
        np.testing.assert_allclose(out["geols_labels"], np.array([1001.0]))


if __name__ == "__main__":
    unittest.main()
