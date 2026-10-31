import json
import sys
from pathlib import Path

import numpy as np
import torch


SPIRAL_DIR = Path(__file__).resolve().parents[1]
if str(SPIRAL_DIR) not in sys.path:
    sys.path.insert(0, str(SPIRAL_DIR))

from fiber_direction_samples import (FORMAT_VERSION, _cell_argmax, _parse_z_roi,
                                     load_fiber_direction_samples)
from losses import get_fiber_direction_loss


def test_parse_z_roi():
    assert _parse_z_roi("10000,11000") == (10000, 11000)


def test_cell_argmax_keeps_one_highest_presence_voxel_per_cell():
    presence = np.zeros((4, 4, 4), dtype=np.uint8)
    presence[0, 0, 0] = 180
    presence[1, 1, 1] = 220
    presence[2, 0, 0] = 200
    selected, coordinates = _cell_argmax(
        presence, np.zeros(3, dtype=np.int64), 2, 160,
        np.zeros(3, dtype=np.int64), np.full(3, 4, dtype=np.int64))
    assert selected.tolist() == [21, 32]
    assert coordinates.tolist() == [[1, 1, 1], [2, 0, 0]]


def test_load_filters_z_and_decodes_axis(tmp_path):
    np.savez_compressed(
        tmp_path / "fiber_directions.npz",
        position_zyx=np.asarray([[5, 2, 3], [15, 2, 3]], dtype=np.float32),
        nx=np.asarray([128, 255], dtype=np.uint8),
        ny=np.asarray([128, 128], dtype=np.uint8),
        presence=np.asarray([200, 255], dtype=np.uint8),
        metadata_json=np.asarray(json.dumps({
            "artifact_type": "fiber_direction_samples",
            "format_version": FORMAT_VERSION,
        })),
    )
    samples = load_fiber_direction_samples(tmp_path / "fiber_directions.npz", 0, 10)
    assert samples["position_zyx"].tolist() == [[5, 2, 3]]
    assert samples["nx"].tolist() == [128]
    assert samples["ny"].tolist() == [128]
    assert samples["presence"].tolist() == [200]


class _IdentityTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, points):
        return points * self.scale


def test_direction_loss_is_zero_for_sheet_tangent_and_one_for_normal():
    transform = _IdentityTransform()
    base = {"position_zyx": np.asarray([[0, 0, 10]], dtype=np.float32),
            "presence": np.asarray([255], dtype=np.uint8)}
    tangent = {**base, "nx": np.asarray([128], dtype=np.uint8),
               "ny": np.asarray([128], dtype=np.uint8)}
    normal = {**base, "nx": np.asarray([255], dtype=np.uint8),
              "ny": np.asarray([128], dtype=np.uint8)}
    assert torch.isclose(get_fiber_direction_loss(transform, tangent, 1, 1,
                                                  torch.device("cpu")),
                         torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(get_fiber_direction_loss(transform, normal, 1, 1,
                                                  torch.device("cpu")),
                         torch.tensor(1.0), atol=1e-6)


def test_extract_reads_chunks_of_a_store_and_skips_empty_ones(tmp_path):
    from fiber_direction_samples import extract
    from vc_reader.reader import ZarrWriter

    presence = np.zeros((8, 8, 8), dtype=np.uint8)
    presence[5, 1, 2] = 200
    presence[1, 6, 6] = 100  # below the threshold
    nx = np.full((8, 8, 8), 11, dtype=np.uint8)
    ny = np.full((8, 8, 8), 22, dtype=np.uint8)
    for name, values in (("presence", presence), ("nx", nx), ("ny", ny)):
        array = ZarrWriter.create(tmp_path / f"{name}.zarr", shape=values.shape, chunks=(4, 4, 4),
                                  dtype="|u1", skip_empty=True)
        array[...] = values
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source_to_base": 1,
        "groups": {name: {"zarr": f"{name}.zarr", "scaledown": 0}
                   for name in ("presence", "nx", "ny")},
    }))
    output = tmp_path / "samples.npz"
    extract(str(tmp_path / "manifest.json"), (0, 8), output, workers=2)
    with np.load(output) as packed:
        np.testing.assert_array_equal(packed["position_zyx"], [[5, 1, 2]])
        assert packed["presence"].tolist() == [200]
        assert packed["nx"].tolist() == [11] and packed["ny"].tolist() == [22]
