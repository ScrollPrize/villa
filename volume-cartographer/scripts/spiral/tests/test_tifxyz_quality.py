import json
import tempfile
import unittest
from pathlib import Path

from tifxyz_quality import assess_metadata, assess_tifxyz, iter_tifxyz


class TestTifxyzQuality(unittest.TestCase):
    def test_mixed_sheet_is_rejected(self):
        examples = (
            (0.0777519, 0.3963, 3475, 156857),
            (0.1093453, 0.5580, 7090, 279952),
        )
        for thick, trimmed, folded, valid in examples:
            with self.subTest(thick=thick):
                result = assess_metadata(
                    Path("bad.tifxyz"),
                    {
                        "quality": {"thick_cell_frac": thick},
                        "raster": {
                            "trimmed_fraction": trimmed,
                            "fold_masked_vertices": folded,
                            "valid_vertices": valid,
                        },
                    },
                )
                self.assertEqual(result.status, "reject")
                self.assertTrue(
                    any("mixed-sheet" in reason for reason in result.reasons)
                )

    def test_interior_holes_do_not_affect_assessment(self):
        examples = (
            (0.0, 0.0432, 938, 135239),
            (0.0000951, 0.0473, 1272, 233141),
        )
        for thick, trimmed, folded, valid in examples:
            with self.subTest(thick=thick):
                result = assess_metadata(
                    Path("good-with-holes.tifxyz"),
                    {
                        "quality": {"thick_cell_frac": thick},
                        "raster": {
                            "trimmed_fraction": trimmed,
                            "fold_masked_vertices": folded,
                            "valid_vertices": valid,
                        },
                    },
                )
                self.assertEqual(result.status, "accept")

    def test_any_fold_fix_rejection_is_explicit_and_independent(self):
        metadata = {
            "quality": {"thick_cell_frac": 0.0},
            "raster": {
                "trimmed_fraction": 0.01,
                "fold_masked_vertices": 0,
                "valid_vertices": 100000,
                "slim": {"fold_masked_vertices": 1},
            },
        }
        self.assertEqual(
            assess_metadata(Path("fixed.tifxyz"), metadata).status, "accept"
        )
        result = assess_metadata(
            Path("fixed.tifxyz"), metadata, reject_any_fold_fixes=True
        )
        self.assertEqual(result.status, "reject")
        self.assertTrue(any("fold fixes applied" in reason for reason in result.reasons))

    def test_cleanup_damage_is_a_secondary_rejection(self):
        result = assess_metadata(
            Path("damaged.tifxyz"),
            {
                "quality": {"thick_cell_frac": 0.01},
                "raster": {
                    "trimmed_fraction": 0.30,
                    "fold_masked_vertices": 3000,
                    "valid_vertices": 100000,
                },
            },
        )
        self.assertEqual(result.status, "reject")
        self.assertTrue(any("cleanup damage" in reason for reason in result.reasons))

    def test_missing_provenance_is_unknown_not_accepted(self):
        result = assess_metadata(Path("old.tifxyz"), {"raster": {}})
        self.assertEqual(result.status, "unknown")

    def test_reads_and_discovers_tifxyz(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            patch = root / "nested" / "sample.tifxyz"
            patch.mkdir(parents=True)
            (patch / "meta.json").write_text(
                json.dumps({"quality": {"thick_cell_frac": 0.0}})
            )
            self.assertEqual(list(iter_tifxyz([root])), [patch])
            self.assertEqual(assess_tifxyz(patch).status, "accept")


if __name__ == "__main__":
    unittest.main()
