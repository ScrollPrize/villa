"""Regenerate tests/data/metadata_subset.json from a full metadata.json.

    python tests/make_fixture.py path/to/metadata.json

The subset keeps real entries for the cases the tests cover and nulls the
opaque provenance fields (``creation_info``, ``copy_info``, scan and volume
``creation.metadata``) to keep it small.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vc_reader._fetch import read_metadata_file  # noqa: E402

KEEP = {
    # Scroll 1: two 2.4um volumes (78/137 keV), labels on the 78 keV volume only.
    "PHercParis4": ["20230702185753", "20231016151002", "20231106155351"],
    # Superseded 2.399um volume; labels native on 9.362um and transferred on 2.399um.
    "PHerc0139": ["20260317000000", "20250831000000", "20250108000005"],
    # Two volumes at 3.24um, no segments kept.
    "PHerc0332": [],
    # Two exports of one 7.91um scan; the second has render_segments false.
    "PHerc0172": [],
    # Two exports of one 2.401um scan, neither flagged.
    "PHerc0009B": [],
    # Fragment with a display name and no volumes.
    "PHercParis1Fr39": [],
}


def _strip(entries):
    for e in entries or ():
        e["creation_info"] = None
        for o in e.get("origins") or ():
            o["copy_info"] = None


def main(src: str) -> None:
    data = read_metadata_file(src)
    out = {"$schema": data.get("$schema"), "samples": {}, "models": data["models"]}
    for sid, seg_ids in KEEP.items():
        tree = data["samples"][sid]
        tree["segments"] = {k: tree["segments"][k] for k in seg_ids}
        _strip(tree["sample"].get("data"))
        for group in ("scans", "volumes", "segments"):
            for ent in tree[group].values():
                _strip(ent.get("data"))
                if group != "segments":
                    ent["creation"]["metadata"] = None
        out["samples"][sid] = tree
    dst = Path(__file__).parent / "data" / "metadata_subset.json"
    dst.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
