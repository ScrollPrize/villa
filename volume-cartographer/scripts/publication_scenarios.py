#!/usr/bin/env python3
"""Reproducible line-annotation publication measurements against a hosted replica.

Drives a real VC3D through the agent bridge, records the publication span log
(``--publication-metrics``) and evaluates it with ``publication_metrics.py``.
One invocation measures ONE build variant (baseline, A, A+B, ...); run it once
per variant with the same arguments so the reports compare like with like.

Scenarios (``--scenario``, default ``all`` = open then new):
  open   an existing fiber opened editable at a control point, N placements
         through left clicks on the arc-length strip at seeded random columns
         (each click places a control point, commits synchronously and lands
         after the solve). The first placements of the fresh process are the
         cold-cache placements; the report splits them out.
  new    a fiber launched from the centre of a main slice viewer, N placements
         the same way. The strip is short, so placements densify the fiber.
  manual the step list below is printed (N = --placements, per step), VC3D
         stays up until Enter is pressed, then the log is evaluated like any
         other run: 2N placements and a tracked-Spiral consumer are required.

Hub isolation. Every variant pushes geometry and leases into the hub it syncs
with, so variants must not share a hub or a replica. Either give an isolated
hub with ``--hub-url`` (a fresh replica is cloned per run) or let the script
copy one server side first:

  --hub-ssh david@dl.ash2txt.org
  --hub-source /home/data/public/other/annotators/test/0139_fibers/0139_fibers.vcobj
  --hub-dest-dir /home/data/public/other/annotators/test/pubmetrics

which makes a seed hub ONCE at ``<dest-dir>/seed/<name>``: the source hub is
cloned under its lock (a consistent snapshot even while others sync with it),
the clone is detached and turned into a hub, and uploaded together with a
manifest (``<name>.ok``: source, size, sha256) that later runs check before
reusing it (a seed without its manifest is an interrupted upload and is
refused; a seed made from another source is refused). Every run then copies
that never-synced seed server side to ``<dest-dir>/<variant>-<stamp>/<name>``,
so every variant starts from identical hub contents, and clones
``sftp://<hub-ssh>/<that path>``. The seed's sha256 is recorded in the run
header, so variants can be checked to have used the same seed. ``--replica`` uses an existing replica as is
(no isolation; only for smoke runs). NEVER point a run at the production hub:
the placements are junk control points.

Manual scenario (iv) steps, performed by hand while the script waits:
  1. Open a fiber that has linked fibers, enable editing, open an intersection
     inspection on one of its links (both inspection sides editable).
  2. Place N control points on the inspection's strip, waiting for each landing.
  3. Switch to the Spiral workspace, load a session that tracks the fiber, go
     back to the line annotation dialog and place N more control points.
  4. Press Enter in the terminal running this script.

Example:
  publication_scenarios.py --vc3d build/bin/VC3D --variant baseline \\
      --volpkg /media/djosey/nvme3/PHerc0139.volpkg.json --out /tmp/pubmetrics \\
      --hub-ssh david@dl.ash2txt.org --hub-source .../0139_fibers.vcobj \\
      --hub-dest-dir .../pubmetrics --placements 50 --sync-interval-ms 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIDGE_TEST_DIR = HERE.parent / "apps" / "VC3D" / "agent_bridge" / "test"
sys.path.insert(0, str(BRIDGE_TEST_DIR))

from bridge_client import BridgeClient, BridgeError  # noqa: E402
from vc3d_process import VC3DProcess  # noqa: E402

STRIP_SUFFIX = "_line_surface"
RPC_TIMEOUT = 120.0


def log(msg: str) -> None:
    print(f"[scenarios {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Span log tailing (the fixture and VC3D share CLOCK_MONOTONIC on Linux)
# ---------------------------------------------------------------------------

class SpanTail:
    """Follows the span log: completed spans by name (with their attributes)
    and the spans still open."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.names: dict[int, str] = {}
        self.completed: dict[str, list[tuple[int, int]]] = {}  # name -> [(begin_ns, end_ns)]
        self.records: dict[str, list[dict]] = {}  # name -> [{begin, end, attrs, unbalanced}]
        self.begins: dict[int, int] = {}
        self.attrs: dict[int, dict[str, str]] = {}
        self.open: dict[int, str] = {}  # id -> name

    def refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self.offset)
            data = fh.read()
            # Keep a partial trailing line for the next refresh.
            cut = data.rfind("\n")
            if cut < 0:
                return
            self.offset += len(data[:cut + 1].encode("utf-8"))
            for line in data[:cut].split("\n"):
                parts = line.split()
                if len(parts) < 7 or parts[0] != "span":
                    continue
                sid, ns, name = int(parts[2]), int(parts[5]), parts[6]
                attrs = dict(p.partition("=")[::2] for p in parts[7:])
                if parts[1] == "B":
                    self.names[sid] = name
                    self.begins[sid] = ns
                    self.attrs[sid] = attrs
                    self.open[sid] = name
                elif parts[1] == "E" and sid in self.names:
                    merged = self.attrs.pop(sid, {})
                    merged.update(attrs)
                    begin = self.begins.pop(sid, ns)
                    self.open.pop(sid, None)
                    record = {"begin": begin, "end": ns, "attrs": merged, "unbalanced": attrs.get("unbalanced") == "1"}
                    self.records.setdefault(self.names[sid], []).append(record)
                    if not record["unbalanced"]:
                        self.completed.setdefault(self.names[sid], []).append((begin, ns))

    def count(self, name: str) -> int:
        self.refresh()
        return len(self.completed.get(name, []))

    def open_count(self, name: str) -> int:
        self.refresh()
        return sum(1 for n in self.open.values() if n == name)

    def applied_landings(self, pane: str, epoch: int | None = None) -> int:
        """Final landings of exactly this pane: a result applied with no
        successor solve running (superseded, stale, failed and pending
        landings do not count), of a solve epoch at least `epoch`."""
        self.refresh()
        return sum(1 for r in self.records.get("landing", [])
                   if not r["unbalanced"] and r["attrs"].get("outcome") == "applied" and
                   r["attrs"].get("surface") == pane and
                   (epoch is None or int(r["attrs"].get("epoch", "0") or 0) >= epoch))

    def wait_for_applied_landing(self, pane: str, count: int, timeout: float, alive, epoch: int | None = None) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.applied_landings(pane, epoch) >= count:
                return True
            if not alive():
                return False
            time.sleep(0.05)
        return False

    def wait_for(self, name: str, count: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.count(name) >= count:
                return True
            time.sleep(0.05)
        return False

    def wait_quiet(self, name: str, quiet_s: float, timeout: float) -> bool:
        """True once no `name` span is open and none completed for quiet_s
        seconds; False when the timeout passed first."""
        deadline = time.monotonic() + timeout
        last = self.count(name)
        since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.1)
            now = self.count(name)
            if now != last or self.open_count(name):
                last, since = now, time.monotonic()
            elif time.monotonic() - since >= quiet_s:
                return True
        return False

    def placement_epoch(self, after_ns: int) -> int | None:
        """The solve epoch the click.place span that ended after `after_ns`
        reported (the landing publishing it carries an epoch >= this)."""
        self.refresh()
        for r in reversed(self.records.get("click.place", [])):
            if r["end"] >= after_ns and r["attrs"].get("epoch"):
                return int(r["attrs"]["epoch"])
        return None


# ---------------------------------------------------------------------------
# Replica preparation
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_replica(args, run_dir: Path) -> tuple[Path, str]:
    """Returns (replica path, description of the hub used)."""
    if args.replica:
        log(f"WARNING: using existing replica {args.replica} without hub isolation")
        return Path(args.replica), f"existing replica {args.replica}"
    hub_url = args.hub_url
    if not hub_url:
        if not (args.hub_ssh and args.hub_source and args.hub_dest_dir):
            sys.exit("give --replica, --hub-url, or --hub-ssh/--hub-source/--hub-dest-dir")
        name = os.path.basename(args.hub_source)
        seed = f"{args.hub_dest_dir.rstrip('/')}/seed/{name}"
        tool = args.hub_tool or shutil.which("vc_object_hub")
        if not tool:
            sys.exit("vc_object_hub not found; pass --hub-tool (vc_object_store build with tools)")
        manifest = subprocess.run(["ssh", args.hub_ssh, f"test -e '{seed}' && cat '{seed}.ok' 2>/dev/null || echo MISSING"],
                                  stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
        exists = subprocess.run(["ssh", args.hub_ssh, f"test -e '{seed}'"]).returncode == 0
        if exists and manifest == "MISSING":
            sys.exit(f"hub seed {seed} exists without its manifest {seed}.ok (an interrupted upload?); remove it first")
        if exists:
            fields = dict(line.split("=", 1) for line in manifest.splitlines() if "=" in line)
            if fields.get("source") != args.hub_source:
                sys.exit(f"hub seed {seed} was made from {fields.get('source')}, not {args.hub_source}; use another --hub-dest-dir")
            remote_size = subprocess.run(["ssh", args.hub_ssh, f"stat -c %s '{seed}'"], stdout=subprocess.PIPE, text=True,
                                         check=True).stdout.strip()
            if remote_size != fields.get("size"):
                sys.exit(f"hub seed {seed} is {remote_size} bytes, its manifest says {fields.get('size')}; remove it first")
            # The content itself is verified, not asserted: a seed changed in
            # place (same size) would silently change the workload.
            seed_sha = fields.get("sha256", "")
            remote_sha = subprocess.run(["ssh", args.hub_ssh, f"sha256sum '{seed}'"], stdout=subprocess.PIPE, text=True,
                                        check=True).stdout.split()[0] if seed_sha else ""
            if not seed_sha or remote_sha != seed_sha:
                sys.exit(f"hub seed {seed} does not match its manifest (sha256 {remote_sha or '?'} vs {seed_sha or 'missing'}); remove it first")
            log(f"reusing the hub seed {seed} (sha256 {seed_sha[:12]}...)")
        else:
            # A consistent snapshot of a live hub: clone it (the clone takes the
            # hub's lock), make the clone a hub of its own, upload it, then its
            # manifest (whose presence marks a complete upload). Nothing on the
            # server is read while another client may be writing it.
            source_url = f"sftp://{args.hub_ssh}{args.hub_source}"
            local = run_dir / "seed.vcobj"
            log(f"cloning {source_url} for the hub seed")
            subprocess.run([tool, "clone", source_url, str(local), "--author", args.author], check=True)
            subprocess.run([tool, "detach", str(local), "--author", args.author], check=True)
            subprocess.run([tool, "init", str(local), "--author", args.author], check=True)
            seed_sha = sha256_of(local)
            manifest = f"source={args.hub_source}\nsize={local.stat().st_size}\nsha256={seed_sha}\n"
            # Published atomically: the upload lands beside the seed and is
            # renamed into place, the manifest last (its presence marks a
            # complete upload). Two first runs at once would still race; make
            # the seed once, then run the variants.
            log(f"uploading the hub seed: {local} -> {args.hub_ssh}:{seed}")
            subprocess.run(["ssh", args.hub_ssh, f"mkdir -p '{os.path.dirname(seed)}'"], check=True)
            subprocess.run(["scp", "-q", str(local), f"{args.hub_ssh}:{seed}.uploading"], check=True)
            subprocess.run(["ssh", args.hub_ssh, f"mv -- '{seed}.uploading' '{seed}'"], check=True)
            subprocess.run(["ssh", args.hub_ssh, f"cat > '{seed}.ok'"], input=manifest, text=True, check=True)
            local.unlink(missing_ok=True)
        args.seed_sha256 = seed_sha
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = f"{args.hub_dest_dir.rstrip('/')}/{args.variant}-{stamp}/{name}"
        # The seed is never synced with, so a plain server-side copy is consistent.
        log(f"copying the hub seed on the server: {seed} -> {dest}")
        subprocess.run(["ssh", args.hub_ssh, f"mkdir -p '{os.path.dirname(dest)}' && cp -- '{seed}' '{dest}'"], check=True)
        hub_url = f"sftp://{args.hub_ssh}{dest}"
    replica = run_dir / "replica.vcobj"
    tool = args.hub_tool or shutil.which("vc_object_hub")
    if not tool:
        sys.exit("vc_object_hub not found; pass --hub-tool (vc_object_store build with tools)")
    log(f"cloning {hub_url} -> {replica}")
    subprocess.run([tool, "clone", hub_url, str(replica)], check=True)
    return replica, hub_url


# ---------------------------------------------------------------------------
# Bridge helpers
# ---------------------------------------------------------------------------

def call(client: BridgeClient, method: str, params: dict | None = None, timeout: float = RPC_TIMEOUT):
    result, _ = client.call(method, params, timeout=timeout)
    return result


def viewers(client: BridgeClient) -> list[dict]:
    return call(client, "state.get").get("viewers", [])


def newest_viewer(client: BridgeClient, suffix: str) -> dict | None:
    matches = [v for v in viewers(client) if str(v.get("surfName", "")).endswith(suffix)]
    if not matches:
        return None
    return max(matches, key=lambda v: int(re.sub(r"\D", "", v["viewerId"]) or 0))


def wait_viewer(client: BridgeClient, suffix: str, known: set[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = newest_viewer(client, suffix)
        if v and v["viewerId"] not in known:
            return v
        time.sleep(0.2)
    listed = ", ".join(f"{v['viewerId']}={v.get('surfName')}" for v in viewers(client))
    raise TimeoutError(f"no new viewer with surface suffix {suffix} within {timeout}s; viewers: {listed}")


def viewer_size(client: BridgeClient, viewer_id: str) -> tuple[int, int]:
    shot = call(client, "screenshot.capture", {"target": viewer_id, "maxDim": 100000})
    return int(shot["width"]), int(shot["height"])


def scene_extent(client: BridgeClient, viewer_id: str) -> tuple[float, float]:
    """Logical scene width/height of a viewer: the viewport size state.get
    reports (bridges that predate it fall back to a screenshot, whose physical
    pixels are divided by a probed device pixel ratio)."""
    for v in viewers(client):
        if v.get("viewerId") == viewer_id and v.get("viewportWidth"):
            return float(v["viewportWidth"]), float(v["viewportHeight"])
    width, height = viewer_size(client, viewer_id)
    for dpr in (1.0, 2.0, 1.5, 3.0, 1.25):
        w, h = width / dpr, height / dpr
        ok = True
        for x in (w / 2.0, w * 0.85):
            try:
                call(client, "canvas.get_cursor_volume_point", {"viewer": viewer_id, "scene": {"x": x, "y": h / 2.0}})
            except BridgeError:
                ok = False
                break
        if ok:
            if dpr != 1.0:
                log(f"viewer {viewer_id}: device pixel ratio {dpr}")
            return w, h
    raise RuntimeError(f"viewer {viewer_id}: no scene point samples the volume (is it visible?)")


def wait_volume(client: BridgeClient, timeout: float) -> dict:
    """Waits for a volume; returns the last state either way (VC3D may have
    restored another project, or none, over --volpkg)."""
    deadline = time.monotonic() + timeout
    state = call(client, "state.get")
    while time.monotonic() < deadline:
        state = call(client, "state.get")
        if state.get("volume"):
            return state
        time.sleep(1.0)
    log("no volume loaded yet; the fibers' volume will be opened explicitly")
    return state


def fiber_rows(client: BridgeClient) -> list[dict]:
    return call(client, "fiber.list").get("fibers", [])


def bound_volume(replica: Path) -> str | None:
    """The volume uuid most of the replica's fibers are bound to."""
    import sqlite3
    from collections import Counter
    counts: Counter = Counter()
    try:
        db = sqlite3.connect(f"file:{replica}?mode=ro", uri=True)
        for (meta,) in db.execute("select meta from objects where kind='fiber'"):
            try:
                uuid = json.loads(meta or "{}").get("coordinates", {}).get("volume_uuid")
            except json.JSONDecodeError:
                continue
            if uuid:
                counts[str(uuid)] += 1
        db.close()
    except sqlite3.Error as error:
        log(f"cannot read the replica's fiber bindings: {error}")
        return None
    return counts.most_common(1)[0][0] if counts else None


def ensure_bound_volume(client: BridgeClient, volpkg: str, volume_id: str | None, timeout: float) -> dict:
    """VC3D may restore another project after --volpkg; the fibers only open
    on the volume they are bound to."""
    state = call(client, "state.get")
    current = (state.get("volume") or {}).get("id")
    if not volume_id or current == volume_id:
        return state
    log(f"current volume {current} is not the fibers' volume {volume_id}: reopening {volpkg}")
    call(client, "volume.open", {"path": volpkg, "volumeId": volume_id}, timeout=600)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = call(client, "state.get")
        if (state.get("volume") or {}).get("id") == volume_id:
            return state
        time.sleep(1.0)
    raise TimeoutError(f"volume {volume_id} did not become current")


# ---------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------

def place(client: BridgeClient, tail: SpanTail, rng: random.Random, args, scenario: str,
          fiber_id: str | None, timeline: list[dict], count: int, alive) -> None:
    known_strip: str | None = None
    for i in range(count):
        if not alive():
            raise RuntimeError("VC3D exited")
        strip = newest_viewer(client, STRIP_SUFFIX)
        if not strip:
            raise RuntimeError("line annotation strip viewer disappeared")
        if strip["viewerId"] != known_strip:
            known_strip = strip["viewerId"]
            width, height = scene_extent(client, known_strip)
            if width < 50 or height < 10:
                raise RuntimeError(f"strip viewer {known_strip} is {width}x{height}; is the dialog visible?")
        # The pane's generated surfaces share the strip's prefix: landings are
        # attributed to exactly this pane, and only final applied ones count.
        # Nothing of an earlier placement may still be in flight.
        pane = str(strip["surfName"])[:-len(STRIP_SUFFIX)]
        if not tail.wait_quiet("landing", 0.5, args.landing_timeout):
            raise RuntimeError(f"placement {i}: landings did not settle before the click")
        if not tail.wait_quiet("click", 0.5, 10.0):
            raise RuntimeError(f"placement {i}: clicks did not settle before the click")
        x = width * (0.15 + 0.7 * rng.random())
        y = height / 2.0
        clicks_before = tail.count("click")
        t_click = time.monotonic_ns()
        try:
            call(client, "canvas.click", {"viewer": known_strip, "space": "scene", "position": {"x": x, "y": y}})
        except BridgeError as error:
            raise RuntimeError(f"placement {i} click failed: {error}") from error
        t_returned = time.monotonic_ns()
        clicked = tail.wait_for("click", clicks_before + 1, 10.0)
        # The placement is published by the landing of its own solve epoch: a
        # click that placed nothing (rejected) has no click.place epoch, and
        # no other landing may stand in for it.
        epoch = tail.placement_epoch(t_click) if clicked else None
        landed = epoch is not None and tail.wait_for_applied_landing(pane, 1, args.landing_timeout, alive, epoch)
        t_landed = time.monotonic_ns()
        entry = {"scenario": scenario, "index": i, "fiberId": fiber_id, "viewer": known_strip, "pane": pane,
                 "epoch": epoch, "scene": [x, y], "t_click_ns": t_click, "t_returned_ns": t_returned,
                 "click_logged": clicked, "landed": landed, "t_landed_ns": t_landed if landed else None}
        timeline.append(entry)
        if not clicked:
            raise RuntimeError(f"placement {i}: no click span was logged (click missed the strip?)")
        if epoch is None:
            raise RuntimeError(f"placement {i}: the click placed no control point (no click.place epoch)")
        if not landed:
            raise RuntimeError(f"placement {i}: no applied landing within {args.landing_timeout}s"
                               + ("" if alive() else " (VC3D exited)"))
        log(f"{scenario} placement {i + 1}/{count}: rpc {(t_returned - t_click) / 1e6:.1f} ms, "
            f"landed after {(t_landed - t_click) / 1e6:.0f} ms")
        time.sleep(args.pace_ms / 1000.0)


def scenario_open(client: BridgeClient, tail: SpanTail, rng: random.Random, args, timeline: list[dict], header: dict, alive):
    rows = fiber_rows(client)
    if not rows:
        raise RuntimeError("the replica lists no fibers")
    if args.fiber_id:
        row = next((r for r in rows if str(r["fiberId"]) == str(args.fiber_id)), None)
        if row is None:
            raise RuntimeError(f"fiber {args.fiber_id} is not listed")
    else:
        row = max(rows, key=lambda r: (int(r.get("controlPointCount", 0)), len(r.get("spans", []))))
    fiber_id = str(row["fiberId"])
    header["open_fiber"] = {"fiberId": fiber_id, "name": row.get("name"),
                            "controlPointCount": row.get("controlPointCount"),
                            "linePointCount": row.get("linePointCount")}
    log(f"open: fiber {fiber_id} ({row.get('name')}, {row.get('controlPointCount')} controls)")
    known = {v["viewerId"] for v in viewers(client)}
    cp = max(0, int(row.get("controlPointCount", 1)) // 2)
    call(client, "fiber.open", {"fiberId": fiber_id, "editable": True, "controlPointIndex": cp})
    wait_viewer(client, STRIP_SUFFIX, known, 120.0)
    if not tail.wait_quiet("landing", 2.0, 120.0):
        raise RuntimeError("the opened fiber's initial landings did not settle")
    place(client, tail, rng, args, "open", fiber_id, timeline, args.placements, alive)


def scenario_new(client: BridgeClient, tail: SpanTail, rng: random.Random, args, timeline: list[dict], header: dict, alive):
    # The main workspace must be frontmost for its viewers to be captured
    # and clicked; a restored session may have left another tab on top.
    try:
        call(client, "workspace.switch", {"name": "main"})
    except BridgeError as error:
        log(f"workspace.switch main: {error}")
    mains = [v for v in viewers(client) if v.get("kind") == "chunked" and
             not str(v.get("surfName", "")).startswith(("line_annotation", "intersection", "__"))]
    order = ("seg xz", "seg yz", "xy plane", "xz plane", "yz plane")
    mains.sort(key=lambda v: order.index(v["surfName"]) if v.get("surfName") in order else len(order))
    main = None
    for candidate in mains:
        try:
            width, height = scene_extent(client, candidate["viewerId"])
        except (BridgeError, RuntimeError) as error:
            log(f"skipping {candidate['viewerId']} ({candidate.get('surfName')}): {error}")
            continue
        main = candidate
        break
    if main is None:
        raise RuntimeError("no visible main slice viewer to launch a fiber from")
    sample = call(client, "canvas.get_cursor_volume_point",
                  {"viewer": main["viewerId"], "scene": {"x": width / 2.0, "y": height / 2.0}})
    point = sample.get("volumePoint") or sample.get("position") or sample
    if not isinstance(point, dict) or "x" not in point:
        raise RuntimeError(f"cannot sample a volume point at the centre of {main['viewerId']}: {sample}")
    before = {str(r["fiberId"]) for r in fiber_rows(client)}
    known = {v["viewerId"] for v in viewers(client)}
    log(f"new: launching from {main['viewerId']} ({main.get('surfName')}) at {point}")
    call(client, "fiber.launch", {"viewer": main["viewerId"], "space": "volume", "replaceOwning": False,
                                  "position": {"x": point["x"], "y": point["y"], "z": point["z"]}})
    wait_viewer(client, STRIP_SUFFIX, known, 120.0)
    if not tail.wait_quiet("landing", 2.0, 120.0):
        raise RuntimeError("the new fiber's seed landings did not settle")
    place(client, tail, rng, args, "new", None, timeline, 1, alive)
    created = [fid for fid in (str(r["fiberId"]) for r in fiber_rows(client)) if fid not in before]
    fiber_id = created[0] if created else None
    header["new_fiber"] = {"fiberId": fiber_id, "launchPoint": point, "viewer": main["viewerId"]}
    for entry in timeline:
        if entry["scenario"] == "new" and entry["fiberId"] is None:
            entry["fiberId"] = fiber_id
    place(client, tail, rng, args, "new", fiber_id, timeline, max(0, args.placements - 1), alive)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vc3d", required=True, help="VC3D binary of the variant under test")
    ap.add_argument("--variant", required=True, help="label: baseline, A, AB, ...")
    ap.add_argument("--volpkg", required=True)
    ap.add_argument("--out", required=True, help="output root; <out>/<variant>-<stamp>/ holds log, replica, reports")
    ap.add_argument("--replica", help="existing replica to use as is (no isolation)")
    ap.add_argument("--hub-url", help="isolated hub to clone a fresh replica from")
    ap.add_argument("--hub-ssh"), ap.add_argument("--hub-source"), ap.add_argument("--hub-dest-dir")
    ap.add_argument("--hub-tool", help="path to vc_object_hub")
    ap.add_argument("--scenario", choices=["all", "open", "new", "manual"], default="all")
    ap.add_argument("--placements", type=int, default=50)
    ap.add_argument("--fiber-id", help="fiber for the open scenario (default: most control points)")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--pace-ms", type=int, default=1000, help="pause after each landing")
    ap.add_argument("--landing-timeout", type=float, default=90.0)
    ap.add_argument("--sync-interval-ms", type=int, default=3000)
    ap.add_argument("--push-delay-ms", type=int, default=500)
    ap.add_argument("--min-overlaps", type=int, default=3, help="commits whose hold must overlap a sync")
    ap.add_argument("--author", default="pubmetrics")
    ap.add_argument("--keep-vc3d", action="store_true", help="leave VC3D running after the run")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"{args.variant}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    replica, hub_desc = prepare_replica(args, run_dir)
    metrics_log = run_dir / "publication.log"
    bridge_name = f"vc3d-pubmetrics-{os.getpid()}"

    env = {"VC3D_HUB_SYNC_INTERVAL_MS": str(args.sync_interval_ms),
           "VC3D_HUB_PUSH_DELAY_MS": str(args.push_delay_ms)}
    vc3d_args = ["--volpkg", args.volpkg, "--object-store", str(replica),
                 "--agent-bridge-name", bridge_name, "--publication-metrics", str(metrics_log)]
    header = {"variant": args.variant, "stamp": stamp, "vc3d": args.vc3d, "vc3d_sha256": sha256_of(Path(args.vc3d)),
              "args": vars(args), "hub": hub_desc, "hub_seed_sha256": getattr(args, "seed_sha256", None),
              "replica": str(replica), "env": env, "git": git_describe(Path(args.vc3d)), "hardware": hardware()}
    log(f"launching {args.vc3d} {' '.join(vc3d_args)}")
    proc = VC3DProcess(args.vc3d, vc3d_args, env_overrides=env)
    timeline: list[dict] = []
    tail = SpanTail(metrics_log)
    status = "ok"
    try:
        sock = proc.wait_for_handshake(timeout=180)
        client = BridgeClient(sock, connect_timeout=30)
        wait_volume(client, 60)
        state = ensure_bound_volume(client, args.volpkg, bound_volume(replica), 600)
        header["volume"] = state.get("volume")
        rng = random.Random(args.seed)
        alive = proc.is_running
        if args.scenario == "manual":
            print(__doc__.split("Manual scenario (iv) steps")[1].split("Example:")[0].replace("N control", f"{args.placements} control"))
            input("Press Enter when the manual steps are done... ")
        else:
            if args.scenario in ("all", "open"):
                scenario_open(client, tail, rng, args, timeline, header, alive)
            if args.scenario in ("all", "new"):
                scenario_new(client, tail, rng, args, timeline, header, alive)
        # Let the delayed Committed poll, consumers and the post-action push
        # land, then require every relevant span (and the deferred consumers)
        # to have closed and stayed quiet; a timeout here is a failed run.
        time.sleep(max(3.0, (args.push_delay_ms + 1500) / 1000.0))
        # (the hub sync itself is periodic and never settles; it is not pending work)
        for name in ("click", "landing", "commit", "committed", "save",
                     "consumer.catalog", "consumer.window_fibers"):
            if not tail.wait_quiet(name, 2.0, 60.0):
                raise RuntimeError(f"{name} spans did not settle before the measurement end")
        if not alive():
            raise RuntimeError("VC3D exited during the run")
    except Exception as error:  # noqa: BLE001
        status = f"failed: {error}"
        log(status)
    finally:
        exit_code = None
        if not args.keep_vc3d:
            # SIGTERM ends VC3D's event loop; its shutdown flushes the span
            # log and writes the end marker the analysis requires. A process
            # that had to be killed, or exited with an error, fails the run.
            proc.terminate(timeout=60)
            exit_code = proc.exit_code()
            if exit_code != 0 and status == "ok":
                status = f"failed: VC3D exited with status {exit_code}"
                log(status)
        (run_dir / "vc3d.log").write_text("\n".join(proc.tail_log(100000)) + "\n")
    header["status"] = status
    header["vc3d_exit_code"] = exit_code
    header["placements"] = len(timeline)
    header["landed"] = sum(1 for t in timeline if t["landed"])
    timeline_path = run_dir / "timeline.json"
    timeline_path.write_text(json.dumps({"header": header, "placements": timeline}, indent=2))

    # Scripted runs must show every placement they made (each matched to its
    # own final landing through the timeline); a manual run the placements it
    # was told to make per step, and the tracked-Spiral consumer of step 3.
    report_args = [sys.executable, "-I", str(HERE / "publication_metrics.py"), str(metrics_log),
                   "--json", str(run_dir / "report.json"), "--min-overlaps", str(args.min_overlaps)]
    if args.scenario == "manual":
        report_args += ["--min-clicks", str(2 * args.placements), "--min-landings", str(2 * args.placements),
                        "--require-span", "consumer.spiral_tracked"]
    else:
        report_args += ["--min-clicks", str(len(timeline)), "--min-landings", str(len(timeline)),
                        "--timeline", str(timeline_path)]
    log("evaluating " + " ".join(report_args[3:]))
    with open(run_dir / "report.txt", "w", encoding="utf-8") as fh:
        result = subprocess.run(report_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(result.stdout)
    print(result.stdout)
    split = cold_warm(timeline, tail)
    # The fixture's own outcome is part of the verdict: a run that failed to
    # complete its scenario is not a measurement, whatever the log says.
    report_path = run_dir / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {"gate": {"verdict": "INCOMPLETE", "pass": False}}
    report["fixture"] = {"status": status, "vc3d_exit_code": exit_code, "cold_warm": split,
                         "variant": args.variant, "vc3d_sha256": header["vc3d_sha256"],
                         "hub_seed_sha256": header["hub_seed_sha256"], "placements": len(timeline)}
    if status != "ok":
        report.setdefault("gate", {})["verdict"] = f"FAIL ({status})"
        report["gate"]["pass"] = False
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"run directory: {run_dir}  status: {status}  verdict: {report['gate'].get('verdict')}")
    if status != "ok":
        return 2
    return result.returncode


def cold_warm(timeline: list[dict], tail: SpanTail, cold: int = 10) -> dict | None:
    """Click span durations of the first `cold` placements of the process vs
    the rest (the first placements pay the cold caches; this is a split by
    order, not a verified cache state). Nearest-rank p95."""
    tail.refresh()
    clicks = sorted(tail.completed.get("click", []))
    if not clicks or not timeline:
        return None
    durations = []
    for entry in timeline:
        hit = [e - b for b, e in clicks if entry["t_click_ns"] - 5_000_000 <= b <= entry["t_returned_ns"] + 5_000_000]
        if hit:
            durations.append(max(hit) / 1e6)
    if len(durations) <= cold:
        return None
    def p95(v):
        v = sorted(v); return v[max(0, math.ceil(0.95 * len(v)) - 1)]  # nearest rank
    split = {"cold": cold, "cold_p95_ms": p95(durations[:cold]), "rest_p95_ms": p95(durations[cold:]), "n": len(durations)}
    print(f"click spans: first {cold} (cold) p95={split['cold_p95_ms']:.1f} ms  rest p95={split['rest_p95_ms']:.1f} ms  n={len(durations)}")
    return split


def git_describe(binary: Path) -> str:
    """The commit of the checkout this script runs from (the binary's own
    build is identified by its sha256 in the header, not by this)."""
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def hardware() -> dict:
    info = {"platform": sys.platform, "cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


if __name__ == "__main__":
    sys.exit(main())
