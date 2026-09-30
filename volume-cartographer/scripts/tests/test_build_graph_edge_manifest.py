"""Tests for build_graph_edge_manifest on synthetic UUID-converted datasets.

Runs with the standard library: python3 -m unittest scripts/tests/test_build_graph_edge_manifest.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import build_graph_edge_manifest as bgem  # noqa: E402

PATCH_DIR = "0001_some_patch"
PATCH_UUID = "11111111-2222-4333-8444-555555555555"


def cp_position(fiber_index, cp_index):
    return [100.0 * fiber_index + cp_index, 10.0 * cp_index, 1.0 + fiber_index]


class Dataset:
    """A minimal converted copy: fibers/, node manifest, one patch entry."""

    def __init__(self, root: Path, fiber_count=3, cps=4):
        self.root = root
        (root / "fibers").mkdir()
        self.docs = {}
        for i in range(fiber_count):
            name = f"fiber_{i}.json"
            self.docs[name] = {
                "type": "vc3d_fiber",
                "version": 3,
                "uuid": str(uuid.uuid4()),
                "generation": 10 + i,
                "filename": name,
                "tags": ["reviewed"],
                "control_points": [{"position": cp_position(i, c)} for c in range(cps)],
                "line_points": [],
                "branches": [],
            }
        self.patches = {PATCH_DIR: {"old_uuid": PATCH_DIR, "uuid": PATCH_UUID}}

    def name(self, i):
        return f"fiber_{i}.json"

    def record(self, a, ia, b, ib, **extra):
        rec = {
            "control_point_index": ia,
            "branch_fiber_id": 1000 + b,
            "branch_control_point_index": ib,
            "control_point_direction": [1.0, 0.0, 0.0],
            "branch_control_point_direction": [0.0, 1.0, 0.0],
            "control_point_position": cp_position(a, ia),
            "branch_control_point_position": cp_position(b, ib),
            "branch_file": self.name(b),
        }
        rec.update(extra)
        return rec

    def add(self, a, ia, b, ib, array="branches", **extra):
        self.docs[self.name(a)].setdefault(array, []).append(self.record(a, ia, b, ib, **extra))

    def link(self, a, ia, b, ib, array="branches"):
        """Store a consistent reciprocal pair in both fibers."""
        self.add(a, ia, b, ib, array)
        rec = self.record(b, ib, a, ia)
        rec["control_point_direction"] = [0.0, 1.0, 0.0]
        rec["branch_control_point_direction"] = [1.0, 0.0, 0.0]
        self.docs[self.name(b)].setdefault(array, []).append(rec)

    def write(self):
        for name, doc in self.docs.items():
            (self.root / "fibers" / name).write_text(json.dumps(doc, indent=2) + "\n")
        manifest = {
            "version": 1,
            "patches": self.patches,
            "fibers": {n: {"old_uuid": None, "uuid": d["uuid"]} for n, d in self.docs.items()},
            "point_collections": {},
        }
        (self.root / bgem.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    def uuid_of(self, i):
        return self.docs[self.name(i)]["uuid"]


def snapshot(root: Path):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != bgem.EDGE_MANIFEST_NAME:
            result[str(path.relative_to(root))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return result


class EdgeManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Dataset(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, check=True):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "build_graph_edge_manifest.py"), str(self.data.root)],
            capture_output=True, text=True,
        )
        if check:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def build(self):
        self.data.write()
        self.run_script()
        return json.loads((self.data.root / bgem.EDGE_MANIFEST_NAME).read_text())

    def codes(self, out):
        return sorted(problem["code"] for problem in out["problems"])

    def test_reciprocal_pair_is_one_canonical_edge(self):
        self.data.link(1, 2, 0, 3)
        out = self.build()
        self.assertEqual(out["counts"]["edges"], {"fiber_branch": 1})
        self.assertEqual(out["problems"], [])
        edge = out["edges"][0]
        self.assertEqual(edge["status"], "reciprocated")
        self.assertFalse(edge["pending"])
        self.assertEqual(len(edge["sources"]), 2)
        ends = [(e["fiber_uuid"], e["control_point_index"]) for e in edge["endpoints"]]
        self.assertEqual(ends, sorted(ends, key=lambda e: (e[0].lower(), e[1])))
        by_uuid = {e["fiber_uuid"]: e for e in edge["endpoints"]}
        f0, f1 = by_uuid[self.data.uuid_of(0)], by_uuid[self.data.uuid_of(1)]
        self.assertEqual((f0["control_point_index"], f0["fiber_generation"]), (3, 10))
        self.assertEqual((f1["control_point_index"], f1["fiber_generation"]), (2, 11))
        self.assertEqual(f0["position"], cp_position(0, 3))
        self.assertTrue(bgem.is_uuid(edge["uuid"]))

    def test_stale_branch_fiber_id_is_ignored(self):
        self.data.link(0, 0, 1, 1)
        self.data.link(2, 1, 1, 2)
        # Two records naming fiber_1 with different runtime ids; one names fiber_0's id.
        self.data.docs["fiber_0.json"]["branches"][0]["branch_fiber_id"] = 77
        self.data.docs["fiber_2.json"]["branches"][0]["branch_fiber_id"] = 1000
        out = self.build()
        self.assertEqual(out["counts"]["edges"], {"fiber_branch": 2})
        self.assertEqual(out["problems"], [])
        ids = sorted(s["branch_fiber_id"] for e in out["edges"] for s in e["sources"]
                     if s["branch_file"] == "fiber_1.json")
        self.assertEqual(ids, [77, 1000])

    def test_unreciprocated_edge_is_emitted_and_flagged(self):
        self.data.add(0, 1, 2, 3)
        out = self.build()
        self.assertEqual(self.codes(out), ["missing_reciprocal"])
        edge = out["edges"][0]
        self.assertEqual(edge["status"], "unreciprocated")
        by_uuid = {e["fiber_uuid"]: e for e in edge["endpoints"]}
        # The peer's copy is the only one available for the fiber that did not store it.
        self.assertEqual(by_uuid[self.data.uuid_of(2)]["position"], cp_position(2, 3))
        self.assertEqual(out["problems"][0]["missing_from"], "fiber_2.json")

    def test_pending_disagreement_reported(self):
        self.data.link(0, 1, 1, 1)
        self.data.docs["fiber_1.json"]["branches"][0]["pending"] = True
        out = self.build()
        self.assertEqual(self.codes(out), ["reciprocal_disagreement"])
        self.assertIn("pending", out["problems"][0]["differences"])
        self.assertTrue(out["edges"][0]["pending"])

    def test_position_disagreement_reports_both_values(self):
        self.data.link(0, 1, 1, 1)
        self.data.docs["fiber_1.json"]["branches"][0]["branch_control_point_position"] = [9.0, 9.0, 9.0]
        out = self.build()
        self.assertEqual(self.codes(out), ["reciprocal_disagreement"])
        diffs = out["problems"][0]["differences"]
        (field, values), = diffs.items()
        self.assertTrue(field.endswith("_position"))
        self.assertEqual(values, {"fiber_0.json": cp_position(0, 1), "fiber_1.json": [9.0, 9.0, 9.0]})
        self.assertEqual(out["problems"][0]["max_abs_delta"], {field: 9.0 - cp_position(0, 1)[0]})
        by_uuid = {e["fiber_uuid"]: e for e in out["edges"][0]["endpoints"]}
        self.assertEqual(by_uuid[self.data.uuid_of(0)]["position"], cp_position(0, 1))

    def test_out_of_range_indexes(self):
        self.data.add(0, 9, 1, 0)   # local index out of range
        self.data.add(1, 0, 2, 7)   # remote index out of range
        out = self.build()
        self.assertEqual(self.codes(out), ["index_out_of_range", "index_out_of_range"])
        sides = sorted(e["side"] for p in out["problems"] for e in p["endpoints"])
        self.assertEqual(sides, ["local", "remote"])
        self.assertEqual(out["edges"], [])

    def test_missing_target_file(self):
        self.data.add(0, 1, 1, 1, branch_file="gone.json")
        out = self.build()
        self.assertEqual(self.codes(out), ["missing_target_file"])
        self.assertEqual(out["edges"], [])

    def test_adjacent_kind_kept_separate(self):
        self.data.link(0, 0, 1, 0)
        self.data.link(0, 1, 1, 1, array="adjacent_branches")
        out = self.build()
        self.assertEqual(out["counts"]["edges"], {"fiber_adjacent_branch": 1, "fiber_branch": 1})
        self.assertEqual(out["problems"], [])

    def test_kind_mismatch(self):
        self.data.add(0, 0, 1, 0)
        rec = self.data.record(1, 0, 0, 0)
        self.data.docs["fiber_1.json"]["adjacent_branches"] = [rec]
        out = self.build()
        self.assertEqual(self.codes(out), ["kind_mismatch"])
        self.assertEqual(out["edges"], [])

    def test_duplicate_and_self_link(self):
        self.data.link(0, 0, 1, 0)
        self.data.add(0, 0, 1, 0)
        self.data.add(2, 0, 2, 1)
        out = self.build()
        self.assertEqual(self.codes(out), ["duplicate_record", "self_link"])
        self.assertEqual(out["counts"]["edges"], {"fiber_branch": 1})
        self.assertEqual(len(out["edges"][0]["sources"]), 3)

    def test_record_disagreeing_with_own_control_point(self):
        self.data.link(0, 1, 1, 1)
        self.data.docs["fiber_0.json"]["control_points"][1]["position"] = [5.0, 5.0, 5.0]
        out = self.build()
        self.assertEqual(self.codes(out), ["control_point_position_mismatch"])
        self.assertEqual(out["counts"]["edges"], {"fiber_branch": 1})

    def test_explicit_links_resolved_and_unresolved(self):
        tags = self.data.docs["fiber_0.json"]["tags"]
        tags += [f"explicit_link:{PATCH_DIR}", f"explicit_link: {PATCH_UUID} ", "explicit_link:nope"]
        out = self.build()
        self.assertEqual(out["counts"]["edges"], {"fiber_patch_explicit_target": 1})
        self.assertEqual(self.codes(out), ["unresolved_patch_reference"])
        edge = out["edges"][0]
        self.assertTrue(edge["directed"])
        self.assertEqual(edge["endpoints"][0]["fiber_uuid"], self.data.uuid_of(0))
        self.assertEqual(edge["endpoints"][1]["patch_uuid"], PATCH_UUID)
        self.assertEqual(len(edge["sources"]), 2)
        self.assertEqual(out["created_from"]["source_counts"]["explicit_link_tags"], 3)

    def test_rerun_reuses_uuids_and_is_deterministic(self):
        self.data.link(0, 0, 1, 0)
        self.data.link(1, 3, 2, 2)
        self.data.docs["fiber_2.json"]["tags"].append(f"explicit_link:{PATCH_DIR}")
        self.data.write()
        self.run_script()
        first_text = (self.data.root / bgem.EDGE_MANIFEST_NAME).read_text()
        self.run_script()
        second_text = (self.data.root / bgem.EDGE_MANIFEST_NAME).read_text()
        self.assertEqual(first_text, second_text)
        # A new edge gets a fresh UUID; the others keep theirs.
        self.data.link(0, 2, 2, 0)
        self.data.write()
        self.run_script()
        first = {e["key"]: e["uuid"] for e in json.loads(first_text)["edges"]}
        third = {e["key"]: e["uuid"] for e in json.loads(
            (self.data.root / bgem.EDGE_MANIFEST_NAME).read_text())["edges"]}
        self.assertEqual(len(third), 4)
        for key, value in first.items():
            self.assertEqual(third[key], value)

    def test_refuses_when_node_uuids_changed(self):
        self.data.link(0, 0, 1, 0)
        self.build()
        # Fiber file and node manifest disagree.
        path = self.data.root / "fibers" / "fiber_0.json"
        doc = json.loads(path.read_text())
        doc["uuid"] = str(uuid.uuid4())
        path.write_text(json.dumps(doc))
        proc = self.run_script(check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("does not match the node manifest", proc.stderr)
        # Both re-assigned consistently, but an earlier edge manifest names the old UUID.
        self.data.docs["fiber_0.json"]["uuid"] = doc["uuid"]
        self.data.write()
        proc = self.run_script(check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Node UUIDs changed", proc.stderr)

    def test_inputs_never_modified(self):
        self.data.link(0, 0, 1, 0)
        self.data.add(0, 1, 2, 9)
        self.data.docs["fiber_1.json"]["tags"].append("explicit_link:nope")
        self.data.write()
        before = snapshot(self.data.root)
        self.run_script()
        self.run_script()
        self.assertEqual(snapshot(self.data.root), before)
        leftovers = [p.name for p in self.data.root.iterdir() if p.name.startswith(".")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
