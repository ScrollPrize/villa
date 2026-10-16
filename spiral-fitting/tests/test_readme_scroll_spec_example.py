"""The spiral-scroll.json example in README.md must load with parse_scroll_spec (#2105)."""
import json
import re
from pathlib import Path

from fit_session import parse_scroll_spec

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_examples():
    blocks = re.findall(r"```json\n(.*?)```", README.read_text(), re.S)
    return [json.loads(b) for b in blocks if '"spiral_outward_sense"' in b]


def test_readme_scroll_spec_example_parses(tmp_path):
    examples = _readme_examples()
    assert examples, "no spiral-scroll.json example found in README.md"
    for doc in examples:
        parse_scroll_spec(doc, tmp_path)


def test_readme_pherc0826_example_matches_catalog_sense():
    # Open-data catalog, volume 20250821151701: z_direction_is_top_to_bottom=true,
    # left_handed_coordinates=false, which gives ACW under the README rule.
    (doc,) = [d for d in _readme_examples() if d["name"] == "PHerc0826"]
    assert doc["z_direction_is_top_to_bottom"] is True
    assert doc["left_handed_coordinates"] is False
    assert doc["spiral_outward_sense"].upper() == "ACW"
