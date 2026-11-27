"""The Neuroglancer link places the fine volume as the unit's picture was drawn."""

import json
from urllib.parse import unquote

import numpy as np

from volreg.cores import dense_correlation as dense
from volreg.review import viewer


class FakeSession:
    fixed_is_coarse = True

    def __init__(self):
        self.state = {"fixed": {"url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/S/coarse.zarr/",
                                "voxel_um": 8.0},
                      "moving": {"url": "https://example.org/fine.zarr/", "voxel_um": 2.0},
                      "current": "f002"}
        self.matrices = {"start": np.diag([0.25, 0.25, 0.25, 1.0]),
                         "f002": np.diag([0.25, 0.25, 0.25, 1.0])}
        self.matrices["f002"][:3, 3] = [1, 2, 3]

    def fine_to_coarse(self, fit_id):
        return self.matrices[fit_id]


def state_of(url):
    return json.loads(unquote(url.split("#!", 1)[1]))


def test_the_unit_centre_lands_on_its_own_measurement():
    held = {"patches": [{"id": "p0001", "measured_under": "start",
                         "centre_zyx": [100, 200, 300], "fine_zyx": [403, 798, 1205],
                         "octants": [{"index": 3, "centre_zyx": [80, 180, 280],
                                      "fine_zyx": [322, 721, 1119]}]}]}
    for unit, entry in (("p0001", held["patches"][0]), ("p0001.3", held["patches"][0]["octants"][0])):
        state = state_of(viewer.unit_link(FakeSession(), held, unit))
        coarse, judged, current = state["layers"]
        assert coarse["source"][0]["url"] == "zarr://https://zarr.aws.ash2txt.org/open-data/S/coarse.zarr/"
        matrix = np.vstack([judged["source"][0]["transform"]["matrix"], [0, 0, 0, 1]])
        assert np.allclose(dense.apply(matrix, np.asarray(entry["fine_zyx"], float)), entry["centre_zyx"])
        assert state["position"] == entry["centre_zyx"]
        assert judged["visible"] and not current["visible"] and "f002" in current["name"]
