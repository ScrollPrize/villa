#!/usr/bin/env python3
"""Ink label RPCs against the real VC3D binary, offscreen.

Builds a temporary project with an empty 64^3 volume and two flat tifxyz
segments, then drives ink_labels.* over the bridge:

  - state before open, and paint without a session (-32007);
  - open, then paint a stroke in label space and a polygon in volume space;
  - paint from a mask PNG and read the pixels back exactly;
  - an expect guard that no longer matches is refused (-32602);
  - history undo and redo;
  - read values, edit them and paint them back exactly; a far-off point is
    refused (-32602);
  - set_tool_state, fill on the labels guide, cleanup, suggest and accept
    without a prediction or suggestions (-32009, -32007), discard, and an
    import of a label PNG (deferred);
  - save is deferred, the TIFFs and sidecar exist, and state is equal after
    the session is reloaded from disk; save as a new version, versions and
    the dataset entry;
  - unsaved labels survive a hard kill as an autosave: the next run's open
    fails with autosave_pending, then succeeds with recover "restore"; close.

Needs numpy and Pillow. Prints one JSON object on the last stdout line and
exits nonzero if a check fails.

    python3 manual_bridge_inklabels.py [--vc3d /path/to/VC3D]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_client import BridgeClient, BridgeError  # noqa: E402
from vc3d_process import VC3DProcess  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VC3D_BIN = REPO_ROOT / "build" / "ci-release-gcc" / "bin" / "VC3D"
TIMEOUT = 20.0

# Segment grid: COLS x ROWS vertices 2 voxels apart (scale 0.5) on the plane
# z = 32, so the label canvas is 2 * COLS x 2 * ROWS px.
COLS, ROWS, Z = 30, 20, 32.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Results:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks[name] = {"pass": bool(ok), "detail": detail}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    @property
    def ok(self) -> bool:
        return all(c["pass"] for c in self.checks.values())


def create_volume(root: Path) -> None:
    volume = root / "volumes" / "vol1"
    level = volume / "0"
    level.mkdir(parents=True)
    (volume / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (volume / ".zattrs").write_text(json.dumps({}))
    (volume / "meta.json").write_text(json.dumps({
        "type": "vol", "uuid": "vol1", "name": "vol1", "format": "zarr",
        "width": 64, "height": 64, "slices": 64, "voxelsize": 1.0, "min": 0.0, "max": 255.0}))
    (level / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [64, 64, 64], "chunks": [64, 64, 64], "dtype": "|u1",
        "compressor": None, "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": "."}))


def create_segment(folder: Path, uuid: str) -> Path:
    folder.mkdir(parents=True)
    cols, rows = np.meshgrid(np.arange(COLS, dtype=np.float32), np.arange(ROWS, dtype=np.float32))
    for name, values in (("x", 2.0 + 2.0 * cols), ("y", 2.0 + 2.0 * rows), ("z", np.full_like(cols, Z))):
        Image.fromarray(values.astype(np.float32), mode="F").save(folder / f"{name}.tif")
    (folder / "meta.json").write_text(json.dumps({
        "type": "seg", "uuid": uuid, "name": folder.name, "format": "tifxyz", "scale": [0.5, 0.5],
        "bbox": [[2.0, 2.0, Z], [2.0 + 2 * (COLS - 1), 2.0 + 2 * (ROWS - 1), Z]]}))
    return folder


def create_project(root: Path) -> Path:
    create_volume(root)
    volpkg = root / "inklabels.vcproj"
    return volpkg


def launch(binary: str, volpkg: Path, config_home: Path) -> tuple[VC3DProcess, BridgeClient]:
    name = f"vc3d-inklabels-{os.getpid()}-{int(time.time() * 1000) % 100000}"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "VC3D.ini").write_text("[viewer]\nusername=Ink label test\n[project]\nshow_open_data_catalog_on_startup=false\nauto_open=false\n")
    proc = VC3DProcess(binary, ["--agent-bridge", "--agent-bridge-name", name],
                       env_overrides={"QT_QPA_PLATFORM": "offscreen", "VC3D_CONFIG_DIR": str(config_home), "OMP_NUM_THREADS": "1"})
    path = proc.wait_for_handshake(timeout=90.0)
    client = BridgeClient(path, connect_timeout=TIMEOUT)
    if not volpkg.exists():
        call(client, "project.create", {"path": str(volpkg), "volume": str(volpkg.parent / "volumes" / "vol1")})
    call(client, "volume.open", {"path": str(volpkg)})
    return proc, client


def call(client: BridgeClient, method: str, params: dict | None = None, timeout: float = TIMEOUT) -> dict:
    return client.call(method, params or {}, timeout=timeout)[0]


def expect_error(client: BridgeClient, method: str, params: dict, code: int, kind: str | None = None) -> tuple[bool, str]:
    try:
        result = call(client, method, params)
        return False, f"expected {code}, got result keys {list(result)[:6]}"
    except BridgeError as e:
        got_kind = e.data.get("kind")
        ok = e.code == code and (kind is None or got_kind == kind)
        return ok, f"code={e.code} kind={got_kind} message={e.message!r}"


_IMPORTED_IDS: dict[Path, str] = {}

def activate(client: BridgeClient, location: Path) -> str:
    if location in _IMPORTED_IDS:
        call(client, "segments.activate", {"segmentId": _IMPORTED_IDS[location]}, timeout=60.0)
        return _IMPORTED_IDS[location]
    started = call(client, "objects.import", {"kind": "surface", "path": str(location),
        "uuidNamespace": "04cbabfe-c4e5-4b2e-9c3a-fbd6ecc70c42", "bind": {"scale": 1}})
    for _ in range(120):
        status = call(client, "job.status", {"jobId": started["jobId"]})
        if status["state"] not in ("queued", "running"):
            if status["state"] != "succeeded" or status["result"]["problems"]:
                raise RuntimeError(str(status))
            break
        time.sleep(.1)
    else:
        raise RuntimeError("surface import did not complete")
    ids = status["result"]["uuids"] or status["result"]["existing"]
    if len(ids) != 1:
        raise RuntimeError(f"expected exactly one imported surface: {status}")
    for _ in range(100):
        listed = call(client, "segments.list")
        if any(segment["id"] == ids[0] for segment in listed["segments"]):
            break
        time.sleep(.1)
    else:
        raise RuntimeError(f"imported surface missing from catalog: {listed}")
    call(client, "segments.activate", {"segmentId": ids[0]}, timeout=60.0)
    _IMPORTED_IDS[location] = ids[0]
    return ids[0]


def run(binary: str) -> Results:
    results = Results()
    with tempfile.TemporaryDirectory(prefix="vc3d-inklabels-") as tmp:
        root = Path(tmp)
        volpkg = create_project(root)
        seg_a = create_segment(root / "segments" / "inkseg_a", "inkseg-a-uuid")
        seg_b = create_segment(root / "segments" / "inkseg_b", "inkseg-b-uuid")
        config = root / "config"
        proc, client = launch(binary, volpkg, config)
        try:
            activate(client, seg_a)
            state = call(client, "ink_labels.state")
            results.record("state_before_open", state["open"] is False, json.dumps({"open": state["open"]}))
            ok, detail = expect_error(client, "ink_labels.paint",
                                      {"class": "ink", "stroke": {"points": [{"x": 5, "y": 5}], "radius": 2}},
                                      -32007, "ink_label_session")
            results.record("paint_without_session", ok, detail)

            opened = call(client, "ink_labels.open")
            canvas = opened["canvas"]
            results.record("open", opened["open"] and canvas["width"] == 2 * COLS and canvas["height"] == 2 * ROWS
                           and opened["store"].endswith("ink_labels/" + opened["segment"]),
                           f"store={opened['store']} canvas={canvas['width']}x{canvas['height']}")

            stroke = call(client, "ink_labels.paint", {
                "class": "ink", "stroke": {"points": [{"x": 10.5, "y": 10.5}, {"x": 30.5, "y": 12.5}], "radius": 3}})
            results.record("paint_stroke_label_space", stroke["changedPixels"] > 0 and stroke["undoDepth"] == 1
                           and stroke["clipped"] is False, json.dumps(stroke))

            # Volume space: grid vertex (c, r) is voxel (2 + 2c, 2 + 2r, Z),
            # label pixel 2c + 1 (grid c sits at the canvas center plus 2 (c - COLS/2)).
            poly_points = [{"x": 2 + 2 * c, "y": 2 + 2 * r, "z": Z} for c, r in ((4, 12), (14, 12), (14, 18), (4, 18))]
            poly = call(client, "ink_labels.paint", {"class": "background", "space": "volume",
                                                     "polygon": {"points": poly_points}})
            results.record("paint_polygon_volume_space", poly["changedPixels"] > 0, json.dumps(poly))
            ok, detail = expect_error(client, "ink_labels.paint", {
                "class": "ink", "space": "volume",
                "stroke": {"points": [{"x": 30, "y": 30, "z": 5}], "radius": 2}}, -32003)
            results.record("volume_point_off_surface", ok, detail)

            # Mask paint and exact read back.
            mask = np.zeros((6, 9), dtype=np.uint8)
            mask[1:5, 2:8] = 255
            mask_path = root / "mask.png"
            Image.fromarray(mask).save(mask_path)
            painted = call(client, "ink_labels.paint", {"class": "validation",
                                                        "mask": {"path": str(mask_path), "origin": {"x": 40, "y": 4}}})
            read_path = root / "read.png"
            read = call(client, "ink_labels.read", {"rect": {"x": 40, "y": 4, "width": 9, "height": 6},
                                                    "format": "validation", "path": str(read_path)})
            back = np.asarray(Image.open(read_path))
            results.record("mask_paint_and_read_back", painted["changedPixels"] == 24 and np.array_equal(back, mask)
                           and read["counts"]["validation"] == 24,
                           f"changed={painted['changedPixels']} counts={read['counts']}")

            serial = call(client, "ink_labels.state")["stateSerial"]
            ok, detail = expect_error(client, "ink_labels.paint", {
                "class": "ink", "stroke": {"points": [{"x": 50, "y": 30}], "radius": 2},
                "expect": {"segment": "inkseg_a", "serial": serial + 7}}, -32602, "expect_mismatch")
            results.record("expect_mismatch_refused", ok, detail)
            after = call(client, "ink_labels.state")
            results.record("expect_mismatch_painted_nothing", after["stateSerial"] == serial, f"serial={after['stateSerial']}")

            undo = call(client, "ink_labels.history", {"action": "undo"})
            redo = call(client, "ink_labels.history", {"action": "redo"})
            results.record("history_undo_redo", undo["applied"] == 1 and redo["applied"] == 1
                           and redo["stateSerial"] == serial, f"undo={undo['applied']} redo={redo['applied']}")

            # Values round trip: read, edit, write back as one entry.
            values_path = root / "values.png"
            call(client, "ink_labels.read", {"rect": {"x": 0, "y": 0, "width": 20, "height": 16},
                                             "format": "values", "path": str(values_path)})
            values = np.asarray(Image.open(values_path)).copy()
            values[0:4, 0:6] = 2   # background
            values[12:16, 14:20] = 5  # ink and validation, unsupervised
            edited_path = root / "edited.png"
            Image.fromarray(values).save(edited_path)
            wrote = call(client, "ink_labels.paint", {"mask": {"path": str(edited_path), "origin": {"x": 0, "y": 0},
                                                               "format": "values"}})
            call(client, "ink_labels.read", {"rect": {"x": 0, "y": 0, "width": 20, "height": 16},
                                             "format": "values", "path": str(values_path)})
            results.record("values_write_back", wrote["changedPixels"] > 0
                           and np.array_equal(np.asarray(Image.open(values_path)), values), json.dumps(wrote))
            ok, detail = expect_error(client, "ink_labels.paint", {
                "class": "ink", "stroke": {"points": [{"x": -1.0e7, "y": 10}, {"x": 10, "y": 10}], "radius": 2}}, -32602)
            results.record("far_point_refused", ok, detail)

            tools = call(client, "ink_labels.set_tool_state", {"class": "background", "radius": 7, "tool": "polygon",
                                                                "assist": {"cleanupSize": 2}})
            results.record("set_tool_state", tools["class"] == "background" and tools["radius"] == 7
                           and tools["tool"] == "polygon" and tools["assist"]["settings"]["cleanupSize"] == 2,
                           json.dumps({k: tools[k] for k in ("class", "radius", "tool")}))
            mode = call(client, "ink_labels.set_tool_state", {"labelMode": True})
            results.record("label_mode_on", mode["labelMode"] is True, f"labelMode={mode['labelMode']}")
            filled = call(client, "ink_labels.fill", {"point": {"x": 55.5, "y": 35.5}, "guide": "labels",
                                                      "class": "validation", "bound": {"x": 50, "y": 30, "width": 10, "height": 10}})
            results.record("fill_labels_guide", filled["changedPixels"] > 0 and filled["reachedBound"] is True,
                           json.dumps(filled))
            grown = call(client, "ink_labels.cleanup", {"kind": "grow", "size": 1})
            results.record("cleanup_grow", grown["changedPixels"] > 0, json.dumps(grown))
            ok, detail = expect_error(client, "ink_labels.suggest", {}, -32009, "no_prediction")
            results.record("suggest_needs_prediction", ok, detail)
            ok, detail = expect_error(client, "ink_labels.accept_suggestions", {"all": True}, -32007, "suggestions")
            results.record("accept_without_suggestions", ok, detail)
            discarded = call(client, "ink_labels.discard_suggestions")
            results.record("discard_nothing", discarded["discarded"] is False, json.dumps(discarded))

            label_png = np.zeros((2 * ROWS, 2 * COLS), dtype=np.uint8)
            label_png[20:30, 2:12] = 255
            label_path = root / "imported.png"
            Image.fromarray(label_png).save(label_path)
            imported = call(client, "ink_labels.import", {"path": str(label_path), "label": "validation"}, timeout=60.0)
            results.record("import_png", imported["setPixels"] == 100 and imported["label"] == "validation_mask",
                           json.dumps({k: imported[k] for k in ("setPixels", "label", "format")}))

            before_save = call(client, "ink_labels.state")
            saved = call(client, "ink_labels.save", timeout=60.0)
            kinds = sorted(f["kind"] for f in saved.get("files", []))
            results.record("save_deferred", saved["saved"] is True
                           and kinds == ["inklabels", "supervision_mask", "validation_mask"],
                           f"kinds={kinds} ms={saved.get('ms')}")
            again = call(client, "ink_labels.save", timeout=60.0)
            results.record("save_clean_is_noop", again["saved"] is False, json.dumps(again))

            # Reload from disk: another segment, then back.
            activate(client, seg_b)
            activate(client, seg_a)
            reopened = call(client, "ink_labels.open")
            results.record("state_equal_after_reload", reopened["counts"] == before_save["counts"]
                           and reopened["dirty"] is False and not reopened["warnings"],
                           f"counts={reopened['counts']} warnings={reopened['warnings']}")

            # A numbered version, listed, and named in the dataset entry.
            v2 = call(client, "ink_labels.save", {"asNewVersion": True}, timeout=60.0)
            versions = call(client, "ink_labels.versions")
            entry = call(client, "ink_labels.dataset_entry")["entry"]
            results.record("save_as_new_version", v2["saved"] is True and v2["version"] == 2
                           and versions == {"versions": [1, 2], "open": 2} and entry.get("label_version") == "v2"
                           and entry["segment_names"] == [opened["segment"]],
                           f"versions={versions} entry={entry}")

            # Unsaved edits, a segment switch (checkpoint), then a hard kill.
            call(client, "ink_labels.paint", {"class": "ink", "stroke": {"points": [{"x": 50.5, "y": 30.5}], "radius": 4}})
            unsaved = call(client, "ink_labels.state")["counts"]
            activate(client, seg_b)
            time.sleep(1.0)
        finally:
            client.close()
            if proc.is_running():
                os.kill(proc.proc.pid, signal.SIGKILL)
                proc.proc.wait(timeout=10)

        proc, client = launch(binary, volpkg, config)
        try:
            activate(client, seg_a)
            ok, detail = expect_error(client, "ink_labels.open", {}, -32602, "autosave_pending")
            results.record("autosave_pending_after_crash", ok, detail)
            restored = call(client, "ink_labels.open", {"recover": "restore"})
            results.record("restore_autosave", restored["restoredAutosave"] is True and restored["counts"] == unsaved
                           and restored["dirty"] is True, f"counts={restored['counts']}")
            closed = call(client, "ink_labels.close")
            results.record("close_keeps_autosave", closed == {"closed": True, "keptAsAutosave": True}
                           and call(client, "ink_labels.state")["open"] is False, json.dumps(closed))
        finally:
            client.close()
            proc.terminate()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vc3d", default=str(DEFAULT_VC3D_BIN), help="path to the VC3D binary")
    args = ap.parse_args()
    if not os.path.exists(args.vc3d):
        print(json.dumps({"ok": False, "error": f"binary not found: {args.vc3d}"}))
        return 2
    results = run(args.vc3d)
    print(json.dumps({"ok": results.ok, "checks": results.checks}))
    return 0 if results.ok else 1


if __name__ == "__main__":
    sys.exit(main())
