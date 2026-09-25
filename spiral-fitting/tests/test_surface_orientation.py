"""Exported surfaces put the outermost wrap in column 0 and the scroll top in
row 0. Previews are exported from a synthetic spiral through the identity
transform, so each vertex's radius and z say where it belongs."""
import json
import math
from unittest import mock

import click
import numpy as np
from PIL import Image
import pytest
import torch

from fit_session import ScrollSpecError, parse_scroll_spec
from lasagna_publish import _raw_run_diff_rgba
from loss_maps import LossMapRecorder, capture_loss_maps, record_loss_samples
from render_ink import concat_meshes
from sample_spiral import get_spiral_yxs
import spiral_helpers
from surface_orientation import GridLayout, export_metadata, spiral_outward_sense_for
from tifxyz import save_tifxyz

DR, STEP, Z_END, WINDINGS = 10.0, 2, 40, [10, 11, 12]


class IdentityTransform:
    def __call__(self, points):
        return points

    inv = __call__


def export_preview(path, z_direction_is_top_to_bottom):
    cfg = {"shell_outer_winding_idx": WINDINGS[-1], "output_step_size": STEP,
           "model_flow_bounds_z_margin": 0}
    with mock.patch.object(spiral_helpers, "get_spiral_yxs",
                           lambda *a, **k: get_spiral_yxs(*a, **k, device="cpu")):
        return spiral_helpers.save_combined_preview(
            IdentityTransform(), torch.tensor(DR), [], [], path, cfg,
            z_begin=0, z_end=Z_END, voxel_size_um=9.6,
            get_or_build_unattached_pcl_flat=lambda *_: None,
            surface_id="surface",
            z_direction_is_top_to_bottom=z_direction_is_top_to_bottom)


def read_zyx(surface):
    zyx = np.stack([np.asarray(Image.open(f"{surface}/{a}.tif")) for a in "zyx"], -1)
    return zyx, ~np.all(zyx == -1.0, axis=-1)


def parse(**fields):
    return parse_scroll_spec(
        {"schema_version": 1, "name": "s", "voxel_size_um": 9.6, **fields}, "/d")


def test_sense_table():
    # (True, False) is PHerc0139 20260102150214; (False, False) is PHercParis4
    # 20260411134726, fitted CW.
    assert [spiral_outward_sense_for(z, lh) for z, lh in
            [(True, False), (False, False), (False, True), (True, True)]] == \
        ["ACW", "CW", "ACW", "CW"]


def test_scroll_spec_catalog_properties():
    spec = parse(z_direction_is_top_to_bottom=False, left_handed_coordinates=False)
    assert spec.spiral_outward_sense == "CW"
    assert spec.z_direction_is_top_to_bottom is False
    # A stated sense is a cross-check.
    assert parse(z_direction_is_top_to_bottom=True, left_handed_coordinates=False,
                 spiral_outward_sense="acw").spiral_outward_sense == "ACW"
    with pytest.raises(ScrollSpecError, match="contradicts"):
        parse(z_direction_is_top_to_bottom=True, left_handed_coordinates=False,
              spiral_outward_sense="CW")
    with pytest.raises(ScrollSpecError, match="true or false"):
        parse(z_direction_is_top_to_bottom="false", left_handed_coordinates=False)


def test_scroll_spec_without_catalog_properties():
    with pytest.raises(ScrollSpecError, match="spiral_outward_sense"):
        parse(z_direction_is_top_to_bottom=False)
    # The z direction alone is kept, to orient the rows.
    assert parse(z_direction_is_top_to_bottom=False,
                 spiral_outward_sense="CW").z_direction_is_top_to_bottom is False
    assert parse(spiral_outward_sense="CW").z_direction_is_top_to_bottom is None


@pytest.mark.parametrize("top_to_bottom", [True, False])
def test_preview_reads_outer_to_inner_and_top_to_bottom(tmp_path, top_to_bottom):
    manifest = export_preview(tmp_path / "generation", top_to_bottom)
    zyx, valid = read_zyx(manifest["surface_path"])
    radius = np.linalg.norm(zyx[..., 1:], axis=-1)

    # Radius falls along a row; z rises down the grid only if z = 0 is the top.
    columns = np.flatnonzero(valid.any(axis=0))
    assert np.all(np.diff([radius[valid[:, c], c].mean() for c in columns]) < 0)
    rows = np.flatnonzero(valid.any(axis=1))
    heights = np.diff([zyx[r, valid[r], 0].mean() for r in rows])
    assert np.all(heights > 0) if top_to_bottom else np.all(heights < 0)

    # Ranges run left to right, outermost winding first, each on its winding.
    assert manifest["winding_ids"] == WINDINGS[::-1]
    for (begin, end), winding in zip(manifest["winding_column_ranges"],
                                     manifest["winding_ids"]):
        block = radius[:, begin:end][valid[:, begin:end]]
        assert winding * DR - 1e-3 <= block.min() <= block.max() <= (winding + 1) * DR + 1e-3
    meta = json.loads(open(f"{manifest['surface_path']}/meta.json").read())
    assert meta["grid_orientation"] == manifest["grid_orientation"] == export_metadata(top_to_bottom)


def test_loss_map_lands_on_the_cell_holding_the_sample(tmp_path):
    # Near the bottom, so a row mapped the wrong way lands near the top.
    theta, winding, z = 1.0, 11, 8.0
    r = (winding + theta / (2 * math.pi)) * DR
    sample = np.array([[z, math.sin(theta) * r, math.cos(theta) * r]], np.float32)
    root = tmp_path / "generation"
    manifest = export_preview(root, False)
    recorder = LossMapRecorder(manifest, root, z0=0, grid_spacing=STEP,
                               dr_per_winding=DR, weights={"patch_radius": 1})
    with capture_loss_maps(recorder):
        record_loss_samples("patch_radius", sample, np.ones(1, np.float32))
    [entry] = recorder.finish()
    rows, cols = np.nonzero(np.asarray(Image.open(root / entry["path"]))[..., 3])
    zyx, _ = read_zyx(manifest["surface_path"])
    assert np.linalg.norm(zyx[round(rows.mean()), round(cols.mean())] - sample[0]) < 2 * STEP


def test_run_diff_pairs_windings_in_the_export_layout(tmp_path):
    previous = export_preview(tmp_path / "previous", False)
    current = export_preview(tmp_path / "current", False)
    assert _raw_run_diff_rgba(previous, current)[1] == 0

    # Moving the outermost winding marks its columns only.
    zyx, valid = read_zyx(current["surface_path"])
    begin, end = current["winding_column_ranges"][0]
    zyx[:, begin:end, 0][valid[:, begin:end]] += 3.0
    for index, axis in enumerate("zyx"):
        Image.fromarray(zyx[..., index]).save(f"{current['surface_path']}/{axis}.tif")
    marked = _raw_run_diff_rgba(previous, current)[0][..., 3] > 0
    assert marked[:, begin:end].any() and not marked[:, end:].any()
    # A preview in another layout (here, an older spiral-order one) is skipped.
    del previous["grid_orientation"]
    assert _raw_run_diff_rgba(previous, current)[1] == 0


def test_render_ink_concatenates_outermost_winding_first(tmp_path):
    grids = {w: np.full((3, w - 8, 3), float(w), np.float32) for w in (10, 11)}
    for w, grid in grids.items():
        save_tifxyz(GridLayout.export(True).apply(grid), tmp_path, f"w{w}", 1, 9.6, "t",
                    layout_metadata=export_metadata(True))
    combined, layout = concat_meshes([f"{tmp_path}/w10", f"{tmp_path}/w11"])
    assert layout == export_metadata(True)
    np.testing.assert_array_equal(combined, np.concatenate([grids[11], grids[10]], 1))
    save_tifxyz(grids[10], tmp_path, "legacy", 1, 9.6, "t")
    with pytest.raises(click.ClickException):
        concat_meshes([f"{tmp_path}/w11", f"{tmp_path}/legacy"])

