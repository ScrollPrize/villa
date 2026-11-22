"""Confidence-thresholded pseudo-label rules for flat self-training."""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.ink_detection.pseudo_labels import pseudo_label_masks


def _field(shape=(64, 64)):
    papyrus = np.ones(shape, bool)
    mean = np.full(shape, 0.26, np.float32)          # confident background everywhere ...
    mean[20:30, 20:30] = 0.60                         # ... one confident ink block
    mean[40:44, 40:44] = 0.42                         # ... one weak ink candidate
    top = mean.copy()
    return mean, top, papyrus


def test_mean_rule_labels_confident_pixels_and_ignores_the_middle():
    mean, top, papyrus = _field()
    ink, sup = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=0)
    assert ink[25, 25] and sup[25, 25]
    assert not ink[5, 5] and sup[5, 5]
    assert not sup[42, 42]                            # 0.42 is between lo and hi: ignored


def test_max_rule_keeps_a_margin_around_weak_candidates():
    mean, top, papyrus = _field()
    _, sup_mean = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=0)
    ink, sup = pseudo_label_masks(mean, top, papyrus, background="max", margin=8, erode=0)
    assert sup_mean[40, 36] and not sup[40, 36]       # 4 px from the weak candidate: no longer background
    assert sup[5, 5]                                  # far away: still background
    assert ink[25, 25]


def test_max_rule_drops_background_where_any_teacher_disagrees():
    mean, top, papyrus = _field()
    top[5, 5] = 0.35                                  # one teacher above lo
    _, sup = pseudo_label_masks(mean, top, papyrus, background="max", margin=0, erode=0)
    assert not sup[5, 5]


def test_outside_papyrus_and_eroded_edges_are_ignored():
    mean, top, papyrus = _field()
    papyrus[:, :10] = False
    _, sup = pseudo_label_masks(mean, top, papyrus, background="mean", margin=0, erode=4)
    assert not sup[:, :14].any()
    assert sup[5, 30]


def test_rejects_bad_thresholds_and_rules():
    mean, top, papyrus = _field()
    with pytest.raises(ValueError):
        pseudo_label_masks(mean, top, papyrus, hi=0.3, lo=0.5)
    with pytest.raises(ValueError):
        pseudo_label_masks(mean, top, papyrus, background="median")


# ---- export: build_segment / _write_label (need vc_reader's zarr writer and tifffile) ---------------------------------

def _surface_volume(path, depth=7, shape=(300, 260)):
    from vc_reader.reader import create_group

    group = create_group(path, overwrite=True, auth=None)
    array = group.create_array("0", shape=(depth, *shape), chunks=(depth, 128, 128), dtype=np.uint8,
                               cname="zstd", level=5, shuffle=2, fill_value=0)
    vol = np.zeros((depth, *shape), np.uint8)
    vol[:, :, :200] = 100                                  # papyrus on the left, empty render on the right
    array[:, :, :] = vol
    return str(path)


def _teacher(path, shape=(300, 260), ink=0.60, rest=0.26, hole=None):
    import tifffile

    p = np.full(shape, rest, np.float32)
    p[100:140, 60:100] = ink                               # one confident ink block
    if hole is not None:
        p[hole] = 0.0                                      # no prediction here (e.g. outside --mask-path)
    tifffile.imwrite(path, np.round(p * 255).astype(np.uint8))
    return str(path)


def _plane(store, depth):
    from vc_reader.reader import open_node

    arr = np.asarray(open_node(f"{store}/0")[:])
    assert arr.shape[0] == depth
    others = np.delete(arr, depth // 2, axis=0)
    assert not others.any(), "only the middle plane may carry the label"
    return arr[depth // 2]


def test_build_segment_labels_the_middle_plane_and_ignores_pixels_a_teacher_did_not_predict(tmp_path):
    pytest.importorskip("vc_reader.reader")
    pytest.importorskip("tifffile")
    from vesuvius.ink_detection.pseudo_labels import build_segment

    vol = _surface_volume(tmp_path / "sv.zarr")
    hole = (slice(20, 60), slice(20, 60))                  # teacher 2 has no prediction here (written as 0)
    preds = [_teacher(tmp_path / "t1.tif"), _teacher(tmp_path / "t2.tif", hole=hole)]
    build_segment(tmp_path / "out", "s1", vol, preds, background="mean", margin=0, erode=0)
    ink = _plane(tmp_path / "out/s1/s1_inklabels.zarr", 7)
    sup = _plane(tmp_path / "out/s1/s1_supervision_mask.zarr", 7)
    assert ink[120, 80] == 255 and sup[120, 80] == 255     # confident ink
    assert ink[200, 150] == 0 and sup[200, 150] == 255     # confident background
    assert not sup[hole].any()                             # abstention, not background
    assert not sup[:, 200:].any()                          # outside the papyrus


def test_failed_rerun_leaves_no_old_supervision_beside_new_ink(tmp_path, monkeypatch):
    pytest.importorskip("vc_reader.reader")
    pytest.importorskip("tifffile")
    from vesuvius.ink_detection import pseudo_labels as pl

    vol = _surface_volume(tmp_path / "sv.zarr")
    pl.build_segment(tmp_path / "out", "s1", vol, [_teacher(tmp_path / "t1.tif")], background="mean", margin=0, erode=0)
    seg = tmp_path / "out/s1"
    assert (seg / "s1_supervision_mask.zarr").exists()
    real = pl._write_label

    def fail_on_supervision(path, *args, **kwargs):
        if path.name.endswith("_supervision_mask.zarr"):
            raise OSError("simulated failure while writing supervision")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pl, "_write_label", fail_on_supervision)
    with pytest.raises(OSError):
        pl.build_segment(seg.parent, "s1", vol, [_teacher(tmp_path / "t2.tif", ink=0.70)], background="mean",
                         margin=0, erode=0)
    # the trainer pairs <name>_inklabels with <name>_supervision_mask; with the old mask gone it rejects the segment
    # instead of training new ink against stale supervision
    assert (seg / "s1_inklabels.zarr").exists()
    assert not (seg / "s1_supervision_mask.zarr").exists()
