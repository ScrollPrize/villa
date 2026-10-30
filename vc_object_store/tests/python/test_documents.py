"""A tool attaches a results document to a surface; readers get it back."""
import tempfile
from pathlib import Path

import numpy as np

from vc_object_store import ObjectStore
from vc_object_store.documents import attach_document, attached_document
from vc_object_store.surfaces import create_surface


def test_results_document_round_trips_as_attached_data():
    with tempfile.TemporaryDirectory() as directory:
        store = Path(directory) / "objects.vcobj"
        ObjectStore.create(str(store), author="test").close()
        xyz = np.zeros((2, 2, 3), dtype=np.float32) + 1
        surface = create_surface(store, xyz, name="s", scale=1.0, volume="vol", tool="test")
        document = {"collection_avgs": {"7": 0.25},
                    "points_list": [{"point_id": 1, "p": [1, 2, 3], "winding_err": 0.25},
                                    {"point_id": 2, "p": None}, "not a record"]}
        collection = attach_document(store, surface, document, role="results", list_key="points_list",
                                     position_key="p", name="results", volume="vol", author="test")
        read = attached_document(store, surface, "results")
        assert read == {"collection_avgs": {"7": 0.25},
                        "points_list": [{"point_id": 1, "p": [1, 2, 3], "winding_err": 0.25},
                                        {"point_id": 2, "p": None}]}
        assert attached_document(store, surface, "other") is None
        with ObjectStore.open(str(store), read_only=True) as db:
            assert db.neighbours(surface) == []  # attached data, never a graph link
            record = db.object(collection)
            assert record["meta"]["role"] == "results"
            with db.geometry_snapshot() as snapshot:
                points = snapshot.point_collection(collection).points(0, 2)
            assert [list(p) for p in points] == [[1, 2, 3], [-1, -1, -1]]  # a record without a position is -1
