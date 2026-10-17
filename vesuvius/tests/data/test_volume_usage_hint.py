"""The usage hint Volume prints on failure must be a call that works.

Volume(type="segment", segment_id=...) without scroll_id raises
"Could not determine energy/resolution for scroll None" (issue #1333), so the
hint shown right after one error must not lead to a second.
"""

import re
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "src/vesuvius/data/volume.py"


def test_segment_usage_hint_names_the_scroll():
    source = SOURCE.read_text()
    hint = re.search(r"print\('  segment = (Volume\(type=\"segment\".*?\))'\)", source)
    assert hint, "segment usage hint not found"
    assert "scroll_id=" in hint.group(1)
