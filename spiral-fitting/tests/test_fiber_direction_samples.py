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


def test_read_chunk_crops_full_sized_boundary_chunk():
    import fiber_direction_samples as module

    stored = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)

    class Response:
        status_code = 200
        content = stored.tobytes()

        @staticmethod
        def raise_for_status():
            return None

    class Session:
        @staticmethod
        def get(*args, **kwargs):
            return Response()

    metadata = {
        "shape": [5, 4, 4],
        "chunks": [4, 4, 4],
        "dtype": "|u1",
        "compressor": None,
        "order": "C",
    }
    result = module._read_chunk(Session(), "https://example.invalid/a", metadata,
                                (1, 0, 0))
    assert result.shape == (1, 4, 4)
    np.testing.assert_array_equal(result, stored[:1])


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


def _write_artifact(path, **metadata):
    np.savez_compressed(
        path,
        position_zyx=np.asarray([[5, 2, 3]], dtype=np.float32),
        nx=np.asarray([128], dtype=np.uint8), ny=np.asarray([128], dtype=np.uint8),
        presence=np.asarray([200], dtype=np.uint8),
        metadata_json=np.asarray(json.dumps({
            "artifact_type": "fiber_direction_samples",
            "format_version": FORMAT_VERSION, **metadata})))


def test_a_window_outside_the_extraction_is_refused(tmp_path):
    import pytest

    _write_artifact(tmp_path / "f.npz", z_roi=[0, 20])
    assert load_fiber_direction_samples(tmp_path / "f.npz", 0, 10) is not None
    with pytest.raises(ValueError, match=r"z \[0, 20\), not the fit's \[10, 30\)"):
        load_fiber_direction_samples(tmp_path / "f.npz", 10, 30)


class _Response:
    def __init__(self, document):
        self.content = json.dumps(document).encode()
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.content)


class _Session:
    def __init__(self, documents):
        self.documents = documents

    def get(self, url, **kwargs):
        return _Response(self.documents[url])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_a_remote_manifest_is_identified_by_its_run():
    import hashlib
    from fiber_direction_samples import remote_manifest_identity

    base = "https://data.example/fibers/"
    manifest = {"version": 2, "groups": {}, "provenance": "inference.json"}
    session = _Session({base + "f.lasagna.json": manifest,
                        base + "inference.json": {"run_uuid": "run-9",
                                                  "manifest": "f.lasagna.json"}})
    assert remote_manifest_identity(session, base + "f.lasagna.json") == (manifest, "run-9")
    bare = {"version": 2, "groups": {}}
    digest = "sha256-" + hashlib.sha256(json.dumps(bare).encode()).hexdigest()
    assert remote_manifest_identity(_Session({base + "g.lasagna.json": bare}),
                                    base + "g.lasagna.json")[1] == digest


def test_an_existing_artifact_is_adopted_into_the_cache(tmp_path, monkeypatch):
    import pytest
    import requests
    import fiber_direction_samples as module

    url = "https://data.example/fibers/f.lasagna.json"
    monkeypatch.setattr(requests, "Session", lambda: _Session({
        url: {"version": 2, "groups": {}, "provenance": "inference.json"},
        "https://data.example/fibers/inference.json": {"run_uuid": "run-9"}}))
    _write_artifact(tmp_path / "f.npz", manifest_url=url, output_scale_base_voxels=4.0,
                    z_roi=[0, 20])
    target = module.adopt(tmp_path / "f.npz", tmp_path / "cache")
    assert target == tmp_path / "cache" / "fiber-directions" / "run-9" / "scale4.npz"
    assert load_fiber_direction_samples(target, 0, 10)["metadata"]["manifest_url"] == url
    with pytest.raises(FileExistsError):
        module.adopt(tmp_path / "f.npz", tmp_path / "cache")
    _write_artifact(tmp_path / "g.npz", manifest_url=url, output_scale_base_voxels=4.0,
                    dataset_identity="run-1")
    with pytest.raises(ValueError, match="extracted from dataset run-1"):
        module.adopt(tmp_path / "g.npz", tmp_path / "cache", overwrite=True)


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
