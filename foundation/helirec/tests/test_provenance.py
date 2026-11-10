"""Run with python -m unittest discover -s tests -p test_provenance.py."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from helirec.recon import DEFAULTS, HelicalRecon


class ProvenanceTests(unittest.TestCase):
    def test_parameters_remain_a_snapshot_across_methods(self):
        r = object.__new__(HelicalRecon)
        r.params = {**DEFAULTS, "backprojector": "fbp",
                    "filters": [{"type": "paganin", "delta_beta": 1000.0}]}
        r.cal = SimpleNamespace(source="measured", fingerprint="test", version=2,
                                rot_center=32.0, vector=(0.0, -8.0), cor_main_detector=24.0)
        r.info = {"grid_size": 65}
        r.main = SimpleNamespace(pixel_size_m=8.64e-6, energy_kev=116.0, distance_m=1.19)
        r._resolved_sld = {"sld_bm18": True, "sld_fractions": [0.12, 0.08], "sld_lengths": [40., 200.]}
        with patch("helirec.recon._build_identity", return_value={}):
            before = r.provenance()
            r.params["filters"][0]["delta_beta"] = 500.0
            r._resolved_sld["sld_fractions"][0] = 0.2
            r._update({"backprojector": "sirt(num_iter=20,init=gridrec)", "filters": []})
            after = r.provenance()
        self.assertEqual(before["params"]["backprojector"], "fbp")
        self.assertEqual(before["params"]["filters"], [{"type": "paganin", "delta_beta": 1000.0}])
        self.assertEqual(after["params"]["backprojector"], "sirt(num_iter=20,init=gridrec)")
        self.assertEqual(after["params"]["filters"], [])
        self.assertEqual(before["resolved_sld"]["sld_fractions"], [0.12, 0.08])
        self.assertEqual(after["resolved_sld"]["sld_fractions"], [0.2, 0.08])


if __name__ == "__main__":
    unittest.main()
