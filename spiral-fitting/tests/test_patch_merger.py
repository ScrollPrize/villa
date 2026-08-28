import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vc_spiral import patch_merger


REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "build" / "patch-merger" / "bin" / "merge_overlapping_patches"


def write_patch(
    root,
    patch_id,
    x,
    y,
    z=None,
    *,
    scale=(1.0, 1.0),
    mask=None,
    erosion=None,
    compression=None,
):
    patch = root / patch_id
    patch.mkdir(parents=True)
    if z is None:
        z = np.zeros_like(x)
    for name, values in (("x", x), ("y", y), ("z", z)):
        tifffile.imwrite(
            patch / f"{name}.tif",
            np.asarray(values, dtype=np.float32),
            compression=compression,
        )
    if mask is not None:
        tifffile.imwrite(patch / "mask.tif", np.asarray(mask, dtype=np.uint8))
    metadata = {"format": "tifxyz", "uuid": patch_id, "scale": list(scale)}
    if erosion is not None:
        metadata["spiral_patch_erode_cells"] = erosion
    (patch / "meta.json").write_text(json.dumps(metadata))


def output_metadata(root, patch_id):
    return json.loads((root / patch_id / "meta.json").read_text())


def test_direct_only_chain_reuses_patches(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:80, :80].astype(np.float32)
    for patch_id, offset in (("A", 0.0), ("B", 50.0), ("C", 100.0)):
        write_patch(inputs, patch_id, col + offset, row)

    options = patch_merger.MergeOptions()
    options.threads = 2
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.accepted_pair_count == 2
    assert report.rejected_pair_count == 1
    assert report.rejections["no_correspondences"] == 1
    assert report.output_count == 2
    assert report.contained_patch_count == 1
    assert output_metadata(tmp_path / "outputs", "A")["member_ids"] == ["A", "B"]
    assert output_metadata(tmp_path / "outputs", "A")["contained_member_ids"] == ["B"]
    assert not (tmp_path / "outputs" / "B").exists()
    assert output_metadata(tmp_path / "outputs", "C")["member_ids"] == ["C", "B"]


def test_partially_discarded_member_is_not_suppressed(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:81, :81].astype(np.float32)
    write_patch(inputs, "A", col, row, erosion=0)
    member_x = col.copy()
    member_y = row.copy()
    member_z = np.zeros_like(row)
    member_x[50:] += 200.0
    member_y[49] = member_x[49] = member_z[49] = -1.0
    write_patch(inputs, "B", member_x, member_y, member_z, erosion=0)

    options = patch_merger.MergeOptions()
    options.erode_cells = 0
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.output_count == 2
    assert report.contained_patch_count == 0
    assert output_metadata(tmp_path / "outputs", "A")["contained_member_ids"] == []
    assert (tmp_path / "outputs" / "B").is_dir()


def test_ninety_percent_covered_direct_neighbor_is_suppressed(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:101, :101].astype(np.float32)
    write_patch(inputs, "A", col, row, erosion=0)
    member_z = np.zeros_like(row)
    member_z[91] = -1.0
    member_z[92:] = 100.0
    member_x = col.copy()
    member_y = row.copy()
    member_x[91] = member_y[91] = -1.0
    write_patch(inputs, "B", member_x, member_y, member_z, erosion=0)

    options = patch_merger.MergeOptions()
    options.erode_cells = 0
    options.output_step = 10.0
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.output_count == 1
    assert report.contained_patch_count == 1
    assert output_metadata(tmp_path / "outputs", "A")["contained_member_ids"] == ["B"]
    assert not (tmp_path / "outputs" / "B").exists()


def test_post_merge_nms_removes_duplicates_created_in_same_round(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:100, :100].astype(np.float32)
    write_patch(inputs, "H", col, row, erosion=0)

    def write_leaf(patch_id, shared_x, unique_x):
        leaf_row, leaf_col = np.mgrid[:100, :156].astype(np.float32)
        mask = np.zeros((100, 156), dtype=np.uint8)
        mask[:, :55] = 1
        mask[:, 101:] = 1
        x = leaf_col.copy()
        x[:, :55] += shared_x
        x[:, 101:] += unique_x - 101.0
        write_patch(inputs, patch_id, x, leaf_row, mask=mask, erosion=0)

    # A and B each overlap a different half of H and have a detached unique
    # component. Their source coverage is only 50%, so neither suppresses the
    # other. Once each absorbs H, largest-component cleanup discards its unique
    # island and both candidate outputs become the same full H surface.
    write_leaf("A", 0.0, 200.0)
    write_leaf("B", 45.0, 400.0)

    options = patch_merger.MergeOptions()
    options.erode_cells = 0
    options.output_step = 10.0
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.output_count == 1
    assert report.output_nms_suppressed_count >= 1
    assert sum(path.is_dir() for path in (tmp_path / "outputs").iterdir()) == 1


@pytest.mark.parametrize(
    ("x", "y", "expected", "reflected"),
    [
        (
            lambda row, col: 70.0 - row,
            lambda row, col: col,
            np.array([[0.0, -1.0, 70.0], [1.0, 0.0, 0.0]]),
            False,
        ),
        (
            lambda row, col: 70.0 - col,
            lambda row, col: row,
            np.array([[-1.0, 0.0, 70.0], [0.0, 1.0, 0.0]]),
            True,
        ),
    ],
)
def test_rotated_and_reflected_pose_recovery(tmp_path, x, y, expected, reflected):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:71, :71].astype(np.float32)
    write_patch(inputs, "A", col, row)
    write_patch(inputs, "B", x(row, col), y(row, col))

    patch_merger.merge_patch_directory(inputs, tmp_path / "outputs")
    pose = output_metadata(tmp_path / "outputs", "A")["fitted_poses"][1]

    np.testing.assert_allclose(np.asarray(pose["matrix"])[:2], expected, atol=1e-5)
    assert pose["reflected"] is reflected


def test_mask_hole_and_fully_eroded_patch(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:41, :41].astype(np.float32)
    mask = np.ones((41, 41), dtype=np.uint8)
    mask[14:27, 14:27] = 0
    write_patch(
        inputs,
        "masked",
        col,
        row,
        mask=mask,
        erosion=0,
        compression="deflate",
    )
    tiny_row, tiny_col = np.mgrid[:3, :3].astype(np.float32)
    write_patch(inputs, "tiny", tiny_col, tiny_row)

    options = patch_merger.MergeOptions()
    options.output_step = 10.0
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.dropped_invalid_count == 1
    x = tifffile.imread(tmp_path / "outputs" / "masked" / "x.tif")
    assert x[2, 2] == -1.0
    assert x[1, 1] == 10.0
    assert not (tmp_path / "outputs" / "tiny").exists()


def test_output_keeps_one_quad_component_and_no_isolated_vertices(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:12, :12].astype(np.float32)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:5, :5] = 1  # dominant component
    mask[8:11, 8:11] = 1  # detached component with a complete output quad
    mask[8:10, :2] = 1  # rasterizes to one isolated output vertex
    write_patch(inputs, "A", col, row, mask=mask, erosion=0)

    options = patch_merger.MergeOptions()
    options.output_step = 1.0
    patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    output = tmp_path / "outputs" / "A"
    xyz = np.stack(
        [tifffile.imread(output / f"{band}.tif") for band in ("x", "y", "z")],
        axis=-1,
    )
    valid = np.any(xyz != -1.0, axis=-1)
    valid_quads = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    vertices_in_quads = np.zeros_like(valid)
    vertices_in_quads[:-1, :-1] |= valid_quads
    vertices_in_quads[:-1, 1:] |= valid_quads
    vertices_in_quads[1:, :-1] |= valid_quads
    vertices_in_quads[1:, 1:] |= valid_quads

    np.testing.assert_array_equal(valid, vertices_in_quads)
    assert valid_quads.sum() == 9
    assert not valid[8, 8]
    assert not valid[8, 0]
    assert output_metadata(tmp_path / "outputs", "A")["area_vx2"] == 9.0


def test_output_with_no_complete_quad_is_skipped(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:2, :2].astype(np.float32)
    write_patch(inputs, "tiny", col, row, erosion=0)

    options = patch_merger.MergeOptions()
    options.output_step = 20.0
    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs", options)

    assert report.output_count == 0
    assert report.dropped_output_count == 1
    assert not (tmp_path / "outputs" / "tiny").exists()
    root_report = json.loads((tmp_path / "outputs" / "report.json").read_text())
    assert root_report["dropped_output_count"] == 1


@pytest.mark.skipif(not CLI.exists(), reason="native merger CLI has not been built")
def test_cli_and_python_outputs_match(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:61, :61].astype(np.float32)
    write_patch(inputs, "A", col, row)
    write_patch(inputs, "B", col + 10.0, row + 5.0)

    options = patch_merger.MergeOptions()
    options.threads = 1
    python_output = tmp_path / "python"
    cli_output = tmp_path / "cli"
    patch_merger.merge_patch_directory(inputs, python_output, options)
    subprocess.run(
        [str(CLI), str(inputs), str(cli_output), "--threads", "1"],
        check=True,
        capture_output=True,
        text=True,
    )

    python_ids = sorted(path.name for path in python_output.iterdir() if path.is_dir())
    cli_ids = sorted(path.name for path in cli_output.iterdir() if path.is_dir())
    assert python_ids == cli_ids
    for patch_id in python_ids:
        assert (python_output / patch_id / "meta.json").read_bytes() == (
            cli_output / patch_id / "meta.json"
        ).read_bytes()
        for band in ("x", "y", "z"):
            np.testing.assert_array_equal(
                tifffile.imread(python_output / patch_id / f"{band}.tif"),
                tifffile.imread(cli_output / patch_id / f"{band}.tif"),
            )


def test_nonempty_output_is_rejected_without_overwrite(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:31, :31].astype(np.float32)
    write_patch(inputs, "A", col, row, erosion=0)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match="new or empty"):
        patch_merger.merge_patch_directory(inputs, output)
    assert sentinel.read_text() == "keep"


def test_anisotropic_scale_and_thread_count_are_deterministic(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fine_row, fine_col = np.mgrid[:81, :81].astype(np.float32)
    coarse_row, coarse_col = np.mgrid[:41, :21].astype(np.float32)
    write_patch(inputs, "fine", fine_col, fine_row, scale=(1.0, 1.0))
    write_patch(
        inputs,
        "coarse",
        coarse_col * 4.0,
        coarse_row * 2.0,
        scale=(0.5, 0.25),
    )

    one = patch_merger.MergeOptions()
    one.threads = 1
    many = patch_merger.MergeOptions()
    many.threads = 4
    patch_merger.merge_patch_directory(inputs, tmp_path / "one", one)
    patch_merger.merge_patch_directory(inputs, tmp_path / "many", many)

    one_ids = sorted(path.name for path in (tmp_path / "one").iterdir() if path.is_dir())
    many_ids = sorted(path.name for path in (tmp_path / "many").iterdir() if path.is_dir())
    assert one_ids == many_ids
    for patch_id in one_ids:
        assert (tmp_path / "one" / patch_id / "meta.json").read_bytes() == (
            tmp_path / "many" / patch_id / "meta.json"
        ).read_bytes()
        for band in ("x", "y", "z"):
            np.testing.assert_array_equal(
                tifffile.imread(tmp_path / "one" / patch_id / f"{band}.tif"),
                tifffile.imread(tmp_path / "many" / patch_id / f"{band}.tif"),
            )


def test_boundary_curl_cannot_form_an_atlas_join(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:81, :81].astype(np.float32)
    write_patch(inputs, "A", col, row)
    curled_x = col.copy()
    curled_y = row.copy()
    curled_z = np.full_like(row, 100.0)
    curled_z[:5] = 0.0
    # A one-vertex invalid seam separates the coincident curled lip from the
    # distant body, avoiding a sloped transition that would itself be close.
    curled_x[5] = curled_y[5] = curled_z[5] = -1.0
    write_patch(inputs, "B", curled_x, curled_y, curled_z)

    no_erosion = patch_merger.MergeOptions()
    no_erosion.erode_cells = 0
    raw_report = patch_merger.merge_patch_directory(
        inputs, tmp_path / "raw", no_erosion
    )
    eroded_report = patch_merger.merge_patch_directory(inputs, tmp_path / "eroded")

    assert raw_report.accepted_pair_count == 0
    assert eroded_report.accepted_pair_count == 0
    assert eroded_report.rejected_pair_count == 1
    assert output_metadata(tmp_path / "raw", "A")["member_ids"] == ["A"]
    assert output_metadata(tmp_path / "eroded", "A")["member_ids"] == ["A"]


def test_nearby_nonoverlap_and_tiny_contact_are_rejected(tmp_path):
    row, col = np.mgrid[:81, :81].astype(np.float32)

    separated = tmp_path / "separated"
    separated.mkdir()
    write_patch(separated, "A", col, row)
    write_patch(separated, "B", col, row, np.full_like(row, 2.01))
    separated_report = patch_merger.merge_patch_directory(
        separated, tmp_path / "separated-output"
    )
    assert separated_report.accepted_pair_count == 0
    assert separated_report.rejected_pair_count == 1
    assert separated_report.rejections["no_correspondences"] == 1

    contact = tmp_path / "contact"
    contact.mkdir()
    write_patch(contact, "A", col, row)
    write_patch(contact, "B", col + 77.0, row)
    contact_report = patch_merger.merge_patch_directory(
        contact, tmp_path / "contact-output"
    )
    assert contact_report.accepted_pair_count == 0
    assert contact_report.rejected_pair_count == 1
    assert any(
        reason.startswith("insufficient_") for reason in contact_report.rejections
    )


def test_conflicting_extension_is_not_admitted_to_atlas(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:121, :121].astype(np.float32)
    write_patch(inputs, "A", col, row)

    target_x = col.copy()
    target_y = row.copy()
    target_y[:, 82:] += 20.0
    target_x[:, 81] = -1.0
    target_y[:, 81] = -1.0
    target_z = np.zeros_like(row)
    target_z[:, 81] = -1.0
    write_patch(inputs, "B", target_x, target_y, target_z)

    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs")
    assert report.accepted_pair_count == 1
    # The dominant overlap still establishes a valid pair fit, but the
    # disconnected shifted extension conflicts with A in the common UV atlas.
    assert output_metadata(tmp_path / "outputs", "A")["member_ids"] == ["A"]


def test_disagreement_clustering_does_not_bridge_inconsistent_sheets(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    row, col = np.mgrid[:81, :81].astype(np.float32)
    for patch_id, height in (("A", 0.0), ("B", 1.5), ("C", 3.0)):
        write_patch(inputs, patch_id, col, row, np.full_like(row, height))

    report = patch_merger.merge_patch_directory(inputs, tmp_path / "outputs")
    merged_z = tifffile.imread(tmp_path / "outputs" / "A" / "z.tif")

    assert report.accepted_pair_count == 2
    assert report.contained_patch_count == 1
    assert not (tmp_path / "outputs" / "B").exists()
    # A and C disagree by 3 vx. Complete-link clustering keeps them apart;
    # A's merge blends only the mutually consistent A/B samples.
    assert merged_z[2, 2] == pytest.approx(0.75, abs=1e-5)
