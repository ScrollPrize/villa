"""Check physical sampling groups against the published ink_9um source table.

https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um
Both pherc0139-w028 (aligned) and w044 (native) reference
PHerc0139/segments/20260115000000-w044_2026011522 in different CT scans.
Annotation folder suffixes therefore cannot determine physical identity.
Run directly with Python, or collect with pytest; no model/data imports.
"""

import json
from pathlib import Path
import unittest


_CONFIGS = Path(__file__).parents[2] / "src/vesuvius/ink_detection/configs"


class Aligned21SourceGroupsTest(unittest.TestCase):
    def test_published_source_is_one_group_in_both_configs(self):
        prior = json.loads(
            (_CONFIGS / "aligned21_fixed_scroll_prior.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (_CONFIGS / "aligned21_hybrid_3d2d.json").read_text(encoding="utf-8")
        )
        prior_groups = {
            row["representation_key"]: (row["scroll"], row["physical_segment_key"])
            for row in prior["representations"]
        }
        training_groups = {
            source["sampling_representation_keys"][segment]: (
                source["sampling_scroll"],
                source["sampling_physical_segment_keys"][segment],
            )
            for source in training["datasets"]
            for segment in source["segments"]
        }
        self.assertEqual(prior_groups, training_groups)
        for name, groups in (("prior", prior_groups), ("training", training_groups)):
            with self.subTest(config=name):
                self.assertEqual(len(groups), 29)
                self.assertEqual(len(set(groups.values())), 24)
                self.assertEqual(
                    len({physical for scroll, physical in groups.values() if scroll == "0139"}),
                    9,
                )
                aligned = "public_2p4_level2_zmean4:pherc0139-w028"
                native = "native_9p362_level0:w044"
                self.assertEqual(groups[aligned], ("0139", "0139:w044"))
                self.assertEqual(groups[aligned], groups[native])


if __name__ == "__main__":
    unittest.main()
