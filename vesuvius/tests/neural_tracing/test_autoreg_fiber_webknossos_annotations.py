from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vesuvius.data import affine
from vesuvius.neural_tracing.autoreg_fiber import build_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber import webknossos_annotations as wk_inv
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import load_fiber_cache


def _pb_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _pb_key(field_number: int, wire_type: int) -> bytes:
    return _pb_varint((field_number << 3) | wire_type)


def _pb_field_varint(field_number: int, value: int) -> bytes:
    return _pb_key(field_number, 0) + _pb_varint(value)


def _pb_field_bytes(field_number: int, value: bytes) -> bytes:
    return _pb_key(field_number, 2) + _pb_varint(len(value)) + value


def _pb_vec3(x: int, y: int, z: int) -> bytes:
    return _pb_field_varint(1, x) + _pb_field_varint(2, y) + _pb_field_varint(3, z)


def _pb_node(node_id: int, xyz: tuple[int, int, int]) -> bytes:
    return _pb_field_varint(1, node_id) + _pb_field_bytes(2, _pb_vec3(*xyz))


def _pb_edge(source: int, target: int) -> bytes:
    return _pb_field_varint(1, source) + _pb_field_varint(2, target)


def _pb_tree() -> bytes:
    return b"".join(
        [
            _pb_field_varint(1, 7),
            _pb_field_bytes(2, _pb_node(1, (10, 20, 30))),
            _pb_field_bytes(2, _pb_node(2, (11, 21, 31))),
            _pb_field_bytes(3, _pb_edge(1, 2)),
            _pb_field_bytes(7, b"fiber"),
        ]
    )


@dataclass(frozen=True)
class _FakeNode:
    id: int
    position: tuple[float, float, float]


class _FakeTree:
    def __init__(self, *, tree_id: int, name: str, nodes, edges) -> None:
        self.id = tree_id
        self.name = name
        self.nodes = list(nodes)
        self.edges = list(edges)


class _FakeSkeleton:
    def __init__(self, trees) -> None:
        self._trees = list(trees)

    def flattened_trees(self):
        return iter(self._trees)


class _FakeAnnotation:
    def __init__(self, trees) -> None:
        self.skeleton = _FakeSkeleton(trees)

    def save(self, path: Path) -> None:
        Path(path).write_text("fake annotation", encoding="utf-8")


@dataclass(frozen=True)
class _FakeInfo:
    id: str
    name: str
    dataset_id: str = "dataset-id"
    dataset_name: str = "dataset"
    modified: int = 123


@dataclass(frozen=True)
class _FakeUserObj:
    user_id: str
    email: str


class _FakeContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeResponse:
    def __init__(self, payload, headers=None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeApiClient:
    def __init__(self) -> None:
        self.calls = []

    def _get(self, route: str, query=None):
        self.calls.append((route, dict(query or {})))
        if route == "/users/owner-a/annotations":
            return _FakeResponse(
                [
                    {
                        "id": "ua1",
                        "name": "fibers_s3_region",
                        "dataSetName": "s3-with-fibers",
                        "typ": "Explorational",
                        "state": "Active",
                        "modified": 321,
                    },
                    {"id": "skip", "name": "unrelated"},
                ],
                headers={"X-Total-Count": "2"},
            )
        if route == "/users/owner-b/annotations":
            return _FakeResponse([], headers={"X-Total-Count": "0"})
        raise AssertionError(f"unexpected route {route}")


class _FakeAnnotationInfo:
    calls = []

    @classmethod
    def get_remote_annotations(cls, is_finished=False, owner=None):
        cls.calls.append((owner, is_finished))
        if owner == "current" and is_finished is False:
            return [
                _FakeInfo("a1", "fibers_s3_alpha"),
                _FakeInfo("skip", "unrelated"),
            ]
        if owner == "annotator" and is_finished is True:
            return [_FakeInfo("a2", "project_fibers_s1a_beta")]
        return []


class _FakeAnnotationApi:
    downloads = []

    @classmethod
    def download(cls, annotation_id: str, *, skip_volume_data: bool = False):
        cls.downloads.append((annotation_id, skip_volume_data))
        if annotation_id == "a1":
            nodes = [
                _FakeNode(1, (0.0, 0.0, 0.0)),
                _FakeNode(2, (3.0, 0.0, 0.0)),
                _FakeNode(3, (6.0, 0.0, 0.0)),
            ]
            return _FakeAnnotation([_FakeTree(tree_id=11, name="good", nodes=nodes, edges=[(nodes[0], nodes[1]), (nodes[1], nodes[2])])])
        nodes = [
            _FakeNode(1, (0.0, 0.0, 0.0)),
            _FakeNode(2, (1.0, 0.0, 0.0)),
            _FakeNode(3, (0.0, 1.0, 0.0)),
            _FakeNode(4, (0.0, 0.0, 1.0)),
        ]
        return _FakeAnnotation([_FakeTree(tree_id=12, name="branch", nodes=nodes, edges=[(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[0], nodes[3])])])


class _FakeUser:
    @classmethod
    def get_current_user(cls):
        return _FakeUserObj(user_id="current", email="current@example.test")

    @classmethod
    def get_all_managed_users(cls):
        return [_FakeUserObj(user_id="annotator", email="annotator@example.test")]


class _FakeWK:
    AnnotationInfo = _FakeAnnotationInfo
    Annotation = _FakeAnnotationApi
    User = _FakeUser

    @staticmethod
    def webknossos_context(*, url: str, token: str):
        assert url == wk_inv.DEFAULT_WEBKNOSSOS_URL
        assert token == "secret-token"
        return _FakeContext()


def test_marker_inference_maps_target_volumes() -> None:
    assert wk_inv.infer_fiber_marker("my_fibers_s3_trace") == ("fibers_s3", "PHerc0332")
    assert wk_inv.infer_fiber_marker("my_fibers_s1a_trace") == ("fibers_s1a", "PHercParis4")
    assert wk_inv.infer_fiber_marker("non_overlap_preds_s1a-with-fibers_auto") == ("fibers_s1a", "PHercParis4")
    assert wk_inv.infer_fiber_marker("s3-with-fibers_trace") == ("fibers_s3", "PHerc0332")
    assert wk_inv.infer_fiber_marker("fibers_s5_trace") is None
    assert wk_inv.infer_fiber_marker("other") is None
    with pytest.raises(ValueError):
        wk_inv.infer_fiber_marker("fibers_s3_and_fibers_s1a")


def test_token_file_is_read_without_side_effects(tmp_path: Path) -> None:
    token_file = tmp_path / "webknossos-api-token.txt"
    token_file.write_text("  secret-token\n", encoding="utf-8")
    assert wk_inv.resolve_token_path(start_dir=tmp_path) == token_file
    assert wk_inv.read_token_file(token_file) == "secret-token"

    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        wk_inv.read_token_file(empty)


def test_tree_inventory_accepts_only_single_paths(tmp_path: Path) -> None:
    candidate = wk_inv.AnnotationCandidate(
        annotation_id="a1",
        name="fibers_s3_alpha",
        owner="owner",
        marker="fibers_s3",
        target_volume="PHerc0332",
    )
    nodes = [
        _FakeNode(1, (0.0, 0.0, 0.0)),
        _FakeNode(2, (3.0, 0.0, 0.0)),
        _FakeNode(3, (6.0, 0.0, 0.0)),
    ]
    record = wk_inv.tree_inventory_record(
        candidate=candidate,
        tree=_FakeTree(tree_id=1, name="path", nodes=nodes, edges=[(nodes[0], nodes[1]), (nodes[1], nodes[2])]),
        downloaded_path=tmp_path / "a.zip",
    )
    assert record.status == "accepted"
    assert record.reject_reason is None
    assert record.path_length_voxels == pytest.approx(6.0)
    assert record.bbox_xyz_max == [6.0, 0.0, 0.0]

    branch_node = _FakeNode(4, (0.0, 1.0, 0.0))
    branch = wk_inv.tree_inventory_record(
        candidate=candidate,
        tree=_FakeTree(
            tree_id=2,
            name="branch",
            nodes=[*nodes, branch_node],
            edges=[(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[0], branch_node)],
        ),
        downloaded_path=None,
    )
    assert branch.status == "rejected"
    assert branch.reject_reason == "branching"


def test_user_annotation_endpoint_scan_uses_admin_user_route() -> None:
    client = _FakeApiClient()
    candidates, stats = wk_inv.collect_matching_user_endpoint_annotations(
        client,
        [("owner-a", "Owner A"), ("owner-b", "Owner B")],
        endpoint="annotations",
        source="user_annotations",
    )

    assert stats["scanned"] == 2
    assert stats["owners_scanned"] == 2
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.annotation_id == "ua1"
    assert candidate.owner == "owner-a"
    assert candidate.marker == "fibers_s3"
    assert candidate.dataset_name == "s3-with-fibers"
    assert candidate.annotation_type == "Explorational"
    assert candidate.state == "Active"
    assert candidate.source == "user_annotations"


def test_skeleton_tracing_proto_decoder_extracts_trees_nodes_and_edges() -> None:
    tracing = _pb_field_bytes(2, _pb_tree())
    trees = wk_inv.skeleton_trees_from_tracing_proto(tracing)

    assert len(trees) == 1
    assert trees[0].id == 7
    assert trees[0].name == "fiber"
    assert [node.id for node in trees[0].nodes] == [1, 2]
    assert trees[0].nodes[0].position == (10.0, 20.0, 30.0)
    assert [(left.id, right.id) for left, right in trees[0].edges] == [(1, 2)]


def test_build_fiber_cache_converts_accepted_skeleton_proto_inventory(tmp_path: Path, monkeypatch) -> None:
    skeleton_zip = tmp_path / "fibers_s3_test_ann-a.skeleton-v0.zip"
    with zipfile.ZipFile(skeleton_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trace.SkeletonTracing.pb", _pb_field_bytes(2, _pb_tree()))

    inventory = {
        "records": [
            {
                "annotation_id": "ann-a",
                "annotation_name": "fibers_s3_test",
                "owner": "owner",
                "marker": "fibers_s3",
                "target_volume": "PHerc0332",
                "dataset_id": None,
                "dataset_name": "s3-with-fibers",
                "annotation_type": "Explorational",
                "state": "Active",
                "source": "user_annotations",
                "tree_id": 7,
                "tree_name": "fiber",
                "status": "accepted",
                "reject_reason": None,
                "node_count": 2,
                "edge_count": 1,
                "bbox_xyz_min": [10.0, 20.0, 30.0],
                "bbox_xyz_max": [11.0, 21.0, 31.0],
                "path_length_voxels": 1.732,
                "downloaded_path": str(skeleton_zip),
            }
        ]
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    transform_doc = affine.TransformDocument(
        matrix_xyz=np.eye(4, dtype=np.float64),
        fixed_landmarks=[],
        moving_landmarks=[],
        fixed_volume="PHerc0332-fixed",
    )
    monkeypatch.setattr(build_fiber_cache.affine, "read_transform_json", lambda _url: transform_doc)

    manifest = build_fiber_cache.build_fiber_cache(
        inventory_json=inventory_path,
        output_dir=tmp_path / "fiber_cache",
        manifest_json=tmp_path / "manifest.json",
    )

    assert manifest["counts"]["fibers_written"] == 1
    cache_path = Path(manifest["fiber_cache_paths"][0])
    points_zyx, metadata = load_fiber_cache(cache_path)
    np.testing.assert_allclose(points_zyx, np.asarray([[30.0, 20.0, 10.0], [31.0, 21.0, 11.0]]))
    assert metadata["target_volume"] == "PHerc0332"
    assert metadata["marker"] == "fibers_s3"


def test_run_inventory_uses_fake_webknossos_and_writes_sanitized_json(tmp_path: Path) -> None:
    token_file = tmp_path / "webknossos-api-token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    payload = wk_inv.run_inventory(
        cache_dir=tmp_path / "wk-cache",
        token_file=token_file,
        wk_module=_FakeWK,
    )

    assert payload["counts"]["annotations_matched"] == 2
    assert payload["counts"]["owners_scanned"] == 2
    assert payload["counts"]["annotations_downloaded"] == 2
    assert payload["counts"]["trees_accepted"] == 1
    assert payload["counts"]["trees_rejected"] == 1
    assert ("a1", True) in _FakeAnnotationApi.downloads
    assert (tmp_path / "wk-cache" / "inventory.json").exists()
    inventory_text = (tmp_path / "wk-cache" / "inventory.json").read_text(encoding="utf-8")
    assert "secret-token" not in inventory_text
    assert "fibers_s3_alpha" in inventory_text

    summary_csv = tmp_path / "summary.csv"
    wk_inv.write_annotation_summary_csv(payload, summary_csv)
    summary_text = summary_csv.read_text(encoding="utf-8")
    assert "annotation_id,owner,name" in summary_text
    assert "fibers_s3_alpha" in summary_text
