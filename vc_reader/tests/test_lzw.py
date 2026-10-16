"""Independent libtiff streams exercise LZW dictionary growth and reset."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("vc_reader._native")

from make_lzw_fixtures import reference_bytes
from test_zarr_reader import _serve
from vc_reader.reader import ZarrReader


def _array(root, kind, codec="imagecodecs_lzw"):
    expected = reference_bytes(kind)
    root.mkdir(parents=True)
    (root / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [len(expected)], "chunks": [len(expected)],
        "dtype": "|u1", "compressor": {"id": codec}, "order": "C", "fill_value": 0,
    }))
    (root / "0").write_bytes((Path(__file__).parent / "data" / f"lzw_{kind}.bin").read_bytes())
    return np.frombuffer(expected, dtype=np.uint8)


@pytest.mark.parametrize("kind", ["repeat", "noise"])
@pytest.mark.parametrize("codec", ["imagecodecs_lzw", "lzw"])
@pytest.mark.parametrize("memory", [0, 1 << 20])
def test_lzw_chunks(tmp_path, kind, codec, memory):
    root = tmp_path / "array"
    expected = _array(root, kind, codec)
    with ZarrReader.open(root, memory_cache=memory, threads=1) as reader:
        np.testing.assert_array_equal(reader[:], expected)
        np.testing.assert_array_equal(reader.read_points([[0], [1024], [-1]]), [expected[0], expected[1024], 0])


def test_lzw_remote_cache(tmp_path):
    root = tmp_path / "www" / "array"
    expected = _array(root, "noise")
    server, base = _serve(root.parent)
    cache = tmp_path / "cache"
    try:
        with ZarrReader.open(f"{base}/array", cache_dir=cache, threads=1) as reader:
            np.testing.assert_array_equal(reader[:], expected)
        with ZarrReader.open(f"{base}/array", cache_dir=cache, offline=True, threads=1) as reader:
            np.testing.assert_array_equal(reader[:], expected)
    finally:
        server.shutdown()


@pytest.mark.parametrize("change", ["truncated", "invalid", "short", "long"])
def test_bad_lzw_raises(tmp_path, change):
    root = tmp_path / "array"
    _array(root, "noise")
    if change in ("truncated", "invalid"):
        payload = (root / "0").read_bytes()
        (root / "0").write_bytes(payload[:-8] if change == "truncated" else b"invalid")
    else:
        meta = json.loads((root / ".zarray").read_text())
        meta["shape"][0] += 1 if change == "long" else -1
        meta["chunks"] = meta["shape"]
        (root / ".zarray").write_text(json.dumps(meta))
    with ZarrReader.open(root, threads=1) as reader:
        with pytest.raises(RuntimeError, match="lzw"):
            reader[:]
