from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from dinovol_2.cache_metadata_meshes import ensure_metadata
from dinovol_2.dataset import object_labels
from dinovol_2.dataset.object_labels import _load_metadata_json

CATALOG = {"samples": {"PHerc0332": {"volumes": {}}}}


class GzipMetadataTests(unittest.TestCase):
    def test_load_metadata_json_reads_plain_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.dumps(CATALOG).encode("utf-8")
            plain = Path(tmp) / "plain.json"
            packed = Path(tmp) / "packed.json"
            plain.write_bytes(raw)
            packed.write_bytes(gzip.compress(raw))
            object_labels._METADATA_CACHE.clear()
            self.assertEqual(_load_metadata_json(plain), CATALOG)
            self.assertEqual(_load_metadata_json(packed), CATALOG)

    def test_ensure_metadata_stores_readable_json_when_server_sends_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "served.json"
            source.write_bytes(gzip.compress(json.dumps(CATALOG).encode("utf-8")))
            target = Path(tmp) / "cache" / "metadata.json"
            ensure_metadata(target, url=source.as_uri())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), CATALOG)


if __name__ == "__main__":
    unittest.main()
