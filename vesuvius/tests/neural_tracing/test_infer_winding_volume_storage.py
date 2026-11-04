"""The winding volume script's stores: spill blocks, the native phase cache and merged buckets."""

from types import SimpleNamespace

import numpy as np

from vc_reader.reader import create_group, open_node, open_zarr
from vesuvius.neural_tracing.winding_models import infer_winding_volume as iwv


def test_spill_blocks_round_trip(tmp_path):
    path = tmp_path / "spill.bin"
    first = np.zeros(1000, dtype=iwv._PROB_AGG_RECORD)
    first["key"] = np.arange(1000)
    second = first[:10].copy()
    iwv._append_aggregates(path, first)
    iwv._append_aggregates(path, second)
    blocks = list(iwv._iter_aggregates(path, iwv._PROB_AGG_RECORD))
    assert [len(b) for b in blocks] == [1000, 10]
    np.testing.assert_array_equal(blocks[0], first)
    with open(path, "ab") as handle:
        handle.write(b"short")
    try:
        list(iwv._iter_aggregates(path, iwv._PROB_AGG_RECORD))
    except RuntimeError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("a truncated spill must raise")


def _phase_cache(tmp_path, count=5):
    rays = {
        "seed_xyz": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "direction_xyz": np.ones((count, 3), dtype=np.float32),
        "seed_winding": np.arange(count, dtype=np.int32),
        "z_range": np.array([0, 10]),
        "seed_windings": np.array([1, 2]),
        "dr_per_winding": 2.5,
        "umbilicus_path": "umbilicus.txt",
    }
    args = SimpleNamespace(
        output=str(tmp_path / "cache.zarr"), fit_checkpoint=str(tmp_path / "fit"), model_ckpt=str(tmp_path / "model"),
        reference_zarr=str(tmp_path / "reference.zarr"), volume_scale=0, phase_cache=str(tmp_path / "cache.zarr"))
    config = {"ray_length": 8, "transverse_size": 8, "column_stride": 2, "spacing": 1.5}
    iwv._initialize_native_phase_cache(args, rays, np.array([0, count]), [0], config)
    return args, rays


def test_native_phase_cache_round_trip(tmp_path):
    from vesuvius.neural_tracing.winding_models.volume_slab_extractor import SlabFrame

    args, rays = _phase_cache(tmp_path)
    rng = np.random.default_rng(0)
    phase = rng.random((2, 4, 4, 8), dtype=np.float32)
    valid = rng.random((2, 8, 8, 8)) > 0.5
    frames = [SlabFrame(origin=np.full(3, float(i)), axis_a=np.array([1.0, 0, 0]), axis_b=np.array([0, 1.0, 0]),
                        direction=np.array([0, 0, 1.0]), spacing=1.5) for i in range(2)]
    writer = iwv._NativePhaseCacheWriter(args.output, 0)
    writer.add([1, 3], phase, valid, frames)

    reader = iwv._NativePhaseCacheReader(args.output)
    assert reader.read(0) is None and reader.read(2) is None
    for slab, index in enumerate((1, 3)):
        read_phase, read_valid, frame = reader.read(index)
        np.testing.assert_array_equal(read_phase, phase[slab])
        np.testing.assert_array_equal(read_valid, valid[slab])
        np.testing.assert_array_equal(frame.origin, frames[slab].origin)
        assert frame.spacing == 1.5

    # Bits, in numcodecs' v3 codec name, so caches written before and after read alike.
    meta = open_zarr(f"{args.output}/valid/shard_0").meta
    assert meta.filters == ({"id": "packbits"},)

    create_group(args.output, version=3).attrs.update({"complete": True})
    loaded = iwv._load_native_phase_cache(SimpleNamespace(
        phase_cache=args.output, output=str(tmp_path / "out.zarr"), model_ckpt=args.model_ckpt,
        reference_zarr=args.reference_zarr, volume_scale=0))
    np.testing.assert_array_equal(loaded["seed_xyz"], rays["seed_xyz"])
    assert loaded["dr_per_winding"] == 2.5 and loaded["umbilicus_path"] == "umbilicus.txt"


def test_prob_bucket_writes_only_its_chunk(tmp_path):
    group = create_group(tmp_path / "out.zarr", version=3)
    group.create_array("crossing_prob", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="uint8", fill_value=0, **iwv._OUTPUT_ARRAY)
    records = np.zeros(2, dtype=iwv._PROB_AGG_RECORD)
    records["key"] = [(4 << 42) | (0 << 21) | 1, (5 << 42) | (1 << 21) | 2]
    records["total"] = [90, 200]
    records["count"] = [3, 2]
    records["maximum"] = [40, 150]
    spill = tmp_path / "spill.bin"
    iwv._append_aggregates(spill, records)
    written = iwv._write_prob_bucket((str(tmp_path / "out.zarr"), "crossing_prob", (8, 8, 8), 4, 1, 0, 0, [spill], "mean", 0.0))
    assert written == 2
    data = open_zarr(tmp_path / "out.zarr/crossing_prob")[:]
    expected = np.zeros((8, 8, 8), dtype=np.uint8)
    expected[4, 0, 1] = 30
    expected[5, 1, 2] = 100
    np.testing.assert_array_equal(data, expected)


def test_reference_shape_of_array_and_group(tmp_path):
    group = create_group(tmp_path / "volume.zarr")
    group.create_array("0", shape=(6, 7, 8), chunks=(2, 2, 2), dtype="uint8")
    assert tuple(iwv._reference_shape(tmp_path / "volume.zarr")) == (6, 7, 8)
    assert tuple(iwv._reference_shape(tmp_path / "volume.zarr" / "0")) == (6, 7, 8)
    assert isinstance(open_node(tmp_path / "volume.zarr" / "0").shape, tuple)
