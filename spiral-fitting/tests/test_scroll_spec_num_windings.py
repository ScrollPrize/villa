"""spiral-scroll.json num_windings: parsing, and how it becomes fit defaults."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from config import Config
from fit_session import ScrollSpecError, load_scroll_spec, scroll_spec_config_defaults
from spiral_service import ServiceState


def write_spec(root, **extra):
    document = {"schema_version": 1, "name": "s1", "voxel_size_um": 9.362,
                "spiral_outward_sense": "CW", **extra}
    (Path(root) / "spiral-scroll.json").write_text(json.dumps(document))


class NumWindingsParsingTests(unittest.TestCase):
    def test_absent_means_no_scroll_default(self):
        with tempfile.TemporaryDirectory() as root:
            write_spec(root)
            spec = load_scroll_spec(root)
            self.assertIsNone(spec.num_windings)
            self.assertIsNone(spec.manifest()["num_windings"])
            self.assertEqual(scroll_spec_config_defaults(spec, Config().as_dict()), {})

    def test_integer_is_kept_and_published_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            write_spec(root, num_windings=70)
            spec = load_scroll_spec(root)
            self.assertEqual(spec.num_windings, 70)
            self.assertEqual(spec.manifest()["num_windings"], 70)

    def test_non_integer_or_too_small_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for invalid in (True, 0, 1, -5, 70.0, "70"):
                write_spec(root, num_windings=invalid)
                with self.assertRaisesRegex(ScrollSpecError, "num_windings"):
                    load_scroll_spec(root)


class NumWindingsDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.generic = Config().as_dict()

    def test_replaces_the_generic_scroll_1_count_for_both_settings(self):
        self.assertEqual(self.generic["shell_outer_winding_idx"], 130)
        self.assertEqual(self.generic["model_gap_expander_num_windings"], 130)
        defaults = scroll_spec_config_defaults({"num_windings": 70}, self.generic)
        self.assertEqual(defaults, {"shell_outer_winding_idx": 70,
                                    "model_gap_expander_num_windings": 70})

    def test_capacity_is_raised_only_when_the_count_does_not_fit(self):
        capacity = self.generic["model_gap_expander_capacity_windings"]
        fits = scroll_spec_config_defaults({"num_windings": capacity - 3}, self.generic)
        self.assertNotIn("model_gap_expander_capacity_windings", fits)
        too_big = scroll_spec_config_defaults({"num_windings": capacity}, self.generic)
        self.assertEqual(too_big["model_gap_expander_capacity_windings"], capacity + 3)

    def test_explicit_configuration_still_wins(self):
        # The order every caller uses: Python defaults, then the scroll, then
        # explicit settings (FIT_SPIRAL_CONFIG_OVERRIDES or the panel form).
        config = dict(self.generic)
        config.update(scroll_spec_config_defaults({"num_windings": 70}, config))
        config.update({"shell_outer_winding_idx": 75})
        self.assertEqual(config["shell_outer_winding_idx"], 75)
        self.assertEqual(config["model_gap_expander_num_windings"], 70)

    def test_no_spec_means_no_defaults(self):
        self.assertEqual(scroll_spec_config_defaults(None, self.generic), {})



class ServiceCatalogTests(unittest.TestCase):
    """The panel seeds its form from the served catalog."""

    def catalog_for(self, scroll_spec, config_catalog=None):
        state = SimpleNamespace(
            _base=lambda: {}, scroll_spec=scroll_spec,
            config_catalog=config_catalog or Config.catalog())
        return state, ServiceState.configuration_catalog(state)

    def test_no_scroll_count_serves_the_generic_catalog(self):
        state, served = self.catalog_for({"name": "s1", "num_windings": None})
        self.assertEqual(served["defaults"], state.config_catalog["defaults"])
        self.assertEqual(served["presets"], state.config_catalog["presets"])

    def test_scroll_count_reaches_defaults_and_presets(self):
        state, served = self.catalog_for({"name": "s1", "num_windings": 70})
        self.assertEqual(served["defaults"]["shell_outer_winding_idx"], 70)
        self.assertEqual(served["defaults"]["model_gap_expander_num_windings"], 70)
        self.assertTrue(served["presets"])
        for preset in served["presets"].values():
            self.assertEqual(preset["shell_outer_winding_idx"], 70)

    def test_generic_catalog_is_not_mutated(self):
        # Run requests are validated against the service's own copy.
        generic = Config.catalog()
        before = copy.deepcopy(generic)
        self.catalog_for({"name": "s1", "num_windings": 70}, generic)
        self.assertEqual(generic, before)

    def test_a_preset_that_chose_its_own_count_keeps_it(self):
        catalog = Config.catalog()
        name = next(iter(catalog["presets"]))
        catalog["presets"][name] = dict(catalog["presets"][name],
                                        shell_outer_winding_idx=90)
        _, served = self.catalog_for({"name": "s1", "num_windings": 70}, catalog)
        self.assertEqual(served["presets"][name]["shell_outer_winding_idx"], 90)
        self.assertEqual(served["presets"][name]["model_gap_expander_num_windings"], 70)

    def test_no_dataset_resolved_yet(self):
        state, served = self.catalog_for(None)
        self.assertEqual(served["defaults"], state.config_catalog["defaults"])


if __name__ == "__main__":
    unittest.main()
