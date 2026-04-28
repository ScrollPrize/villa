import json

from vesuvius.data.zarr_chunk_index import ENV_OVERRIDE_URL, build_chunk_occupancy


def test_sidecar_cache_hit_is_materialized_to_override(tmp_path, monkeypatch):
    array_path = tmp_path / "volume.zarr" / "0"
    array_path.mkdir(parents=True)
    (array_path / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [4, 4, 4],
                "chunks": [2, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ),
        encoding="ascii",
    )
    (array_path / "0.0.0").write_bytes(b"non-empty")

    monkeypatch.setenv("VESUVIUS_CACHE_DIR", str(tmp_path / "cache"))
    first = build_chunk_occupancy(
        str(array_path),
        chunks=(2, 2, 2),
        shape=(4, 4, 4),
        use_cache=True,
    )
    assert first is not None
    assert (array_path / ".chunk_occupancy.npz").exists()

    override = tmp_path / "shared" / "chunk-occupancy.npz"
    monkeypatch.setenv(ENV_OVERRIDE_URL, str(override))
    second = build_chunk_occupancy(
        str(array_path),
        chunks=(2, 2, 2),
        shape=(4, 4, 4),
        use_cache=True,
    )

    assert second is not None
    assert second.shape == first.shape
    assert override.exists()
