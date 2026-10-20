"""Time one-point imports as existing document size grows (five iterations)."""
import json
from pathlib import Path
import statistics
import tempfile
import time
import uuid

from vc_object_store import ObjectStore


def document(collections):
    return json.dumps({"vc_pointcollections_json_version": "1", "collections": collections})


def main():
    rows = []
    for count in (0, 20, 100):
        timings = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            with ObjectStore.create(str(root / "store.vcobj"), author="benchmark") as db:
                if count:
                    source.write_text(document({str(i): {
                        "uuid": str(uuid.uuid4()),
                        "points": {str(i * 1000 + j): {"p": [j, 2, 3]} for j in range(1000)},
                    } for i in range(count)}))
                    report = db.ingest_geometry_source("point_collection", source)
                    assert not report["problems"], report
                for i in range(5):
                    source.write_text(document({str(count + i): {
                        "uuid": str(uuid.uuid4()),
                        "points": {str((count + i) * 1000): {"p": [1, 2, 3]}},
                    }}))
                    start = time.perf_counter()
                    report = db.ingest_geometry_source("point_collection", source)
                    timings.append((time.perf_counter() - start) * 1000)
                    assert not report["problems"], report
        rows.append({
            "existing_collections": count, "existing_points": count * 1000,
            "iterations": 5, "ms": timings, "mean": statistics.mean(timings),
            "min": min(timings), "median": statistics.median(timings), "max": max(timings),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
