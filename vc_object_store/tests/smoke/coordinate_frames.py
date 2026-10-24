"""Compare explicit and metadata scales on a small real Spiral dataset subset.

Run with the build's Python package on PYTHONPATH. The output directory must be
new; input files are read-only. Volume identity here is a local fixture declaration.
"""
import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from vc_object_store import ObjectStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads((args.dataset / "same_windings.json").read_text())
    data["collections"] = dict(list(sorted(data["collections"].items()))[:5])
    for key, collection in data["collections"].items():
        collection["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(args.dataset.resolve()) + "/pcl/" + key))
    subset = args.output / "collections.json"
    subset.write_text(json.dumps(data))
    # Legacy tifxyz uses a display name as UUID. Normalize only a temporary
    # import wrapper; native coordinates and the user's files remain unchanged.
    source_patch = (args.dataset / "verified_patches" / "0000_low_band_final").resolve()
    patch = args.output / "patch"
    patch.mkdir()
    for source in source_patch.iterdir():
        if source.name != "meta.json":
            (patch / source.name).symlink_to(source)
    metadata = json.loads((source_patch / "meta.json").read_text())
    metadata["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(source_patch)))
    (patch / "meta.json").write_text(json.dumps(metadata))
    volume = "coordinate-smoke-local-reference"
    summaries = []
    constraints = []
    for mixed in (False, True):
        destination = args.output / ("mixed.vcobj" if mixed else "explicit.vcobj")
        with ObjectStore.create(str(destination), author="coordinate-smoke") as db:
            for kind, source in (("patch", patch),
                                 ("point_collection", subset)):
                report = db.ingest_geometry_source(kind, str(source),
                    coordinate_context={"volume_uuid": volume, "scale_factor": 1})
                assert not report["problems"], report
            objects = db.select_objects()
            if mixed:
                for i, record in enumerate(objects):
                    uid = record["uuid"]
                    factor = 2.0 if record["kind"] == "patch" else (0.25 if i % 2 else 4.0)
                    with db.geometry_snapshot() as snapshot:
                        geometry = (snapshot.patch(uid) if record["kind"] == "patch"
                                    else snapshot.point_collection(uid)).load()
                    points = np.asarray(geometry.points)
                    valid = points[:, 0] != -1 if record["kind"] == "patch" else np.ones(len(points), dtype=bool)
                    points[valid] *= factor
                    geometry.points = points.tolist()
                    if record["kind"] == "patch":
                        geometry.scale = [s / factor for s in geometry.scale]
                    revision = db.current_geometry(uid)["uuid"]
                    if record["kind"] == "patch":
                        db.put_geometry(uid, geometry, expected_revision=revision)
                    else:
                        db.edit_geometry(uid, [dict(array="points", element_offset=0,
                            data=points.astype("<f8").tobytes())], revision)
                    db.set_meta(uid, "coordinates", {"volume_uuid": volume, "scale_factor": factor})
            options = {"volume_uuid": volume} if mixed else {}
            first = db.link_point_collections_to_patches(**options)
            repeat = db.link_point_collections_to_patches(**options)
            assert first["edges_created"] > 0, first
            assert repeat["edges_created"] == repeat["edges_updated"] == repeat["stale_removed"] == 0, repeat
            names = {obj["uuid"]: obj["name"] for obj in objects}
            materialized = []
            for obj in objects:
                if obj["kind"] != "point_collection":
                    continue
                for edge in db.edges_of(obj["uuid"]):
                    own_a = edge["a"] == obj["uuid"]
                    anchor = edge["a_anchor"] if own_a else edge["b_anchor"]
                    materialized.append((obj["name"], anchor["points"][0]["index"][0],
                        names[edge["b"] if own_a else edge["a"]], edge["meta"]["point_collection_patch"]["ij"]))
            constraints.append(sorted(materialized))
            summaries.append({"mixed": mixed, "objects": len(objects), "first": first, "repeat": repeat})
    assert constraints[0] == constraints[1], "coordinate conversion changed real-data attachments"
    result = {"runs": summaries, "identical_attachments": len(constraints[0])}
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
