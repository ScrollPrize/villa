"""spiral-scroll.json num_windings: parsing, and how it becomes fit defaults."""
import json
from pathlib import Path
import tempfile
import unittest

from config import Config
from fit_session import ScrollSpecError, load_scroll_spec, scroll_spec_config_defaults


def write_spec(root, **extra):
    document = {"schema_version": 1, "name": "s1", "voxel_size_um": 9.362,
                "spiral_outward_sense": "CW", **extra}
    (Path(root) / "spiral-scroll.json").write_text(json.dumps(document))


class NumWindingsTests(unittest.TestCase):
    def test_absent_changes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            write_spec(root)
            spec = load_scroll_spec(root)
            self.assertIsNone(spec.num_windings)
            self.assertEqual(scroll_spec_config_defaults(spec, Config().as_dict()), {})

    def test_count_replaces_the_generic_scroll_1_default(self):
        with tempfile.TemporaryDirectory() as root:
            write_spec(root, num_windings=70)
            spec = load_scroll_spec(root)
            defaults = scroll_spec_config_defaults(spec, Config().as_dict())
            self.assertEqual(defaults["shell_outer_winding_idx"], 70)
            self.assertEqual(defaults["model_gap_expander_num_windings"], 70)

    def test_capacity_is_raised_only_when_the_count_does_not_fit(self):
        spec = {"num_windings": 70}
        self.assertEqual(
            scroll_spec_config_defaults(spec, {"model_gap_expander_capacity_windings": 60})
            ["model_gap_expander_capacity_windings"], 73)
        self.assertNotIn(
            "model_gap_expander_capacity_windings",
            scroll_spec_config_defaults(spec, {"model_gap_expander_capacity_windings": 100}))

    def test_rejected_unless_an_integer_of_at_least_two(self):
        with tempfile.TemporaryDirectory() as root:
            for invalid in (True, 1, -5, 70.0, "70"):
                write_spec(root, num_windings=invalid)
                with self.assertRaisesRegex(ScrollSpecError, "num_windings"):
                    load_scroll_spec(root)


if __name__ == "__main__":
    unittest.main()
