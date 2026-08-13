#!/usr/bin/env python
"""M5 batch driver: animate + export + render + verify all 31 wraps × 3 variants.

Stages per mesh (manifest-cached in outputs/anim/<stem>/produce_state.json):
  1. animate   — unwrap the DECIMATED mesh (auto axis mode, gates) -> frames.npz + metrics
  2. frames    — per-frame OBJ export (plain + hardlinked ink variant)
  3. videos    — render_video.py per variant × framing (4K masters + 1080p derivatives)
  4. verify    — ffprobe + manifest assembly

CPU stages (1,2) run in a small process pool; GPU stage (3) is strictly serial.
Usage: uv run python scripts/produce.py [--only stem ...] [--stage all|animate|frames|videos]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = ("plain", "ink", "reveal")
FRAMINGS = ("h", "v")

# Bump on any kinematics/renderer change that must invalidate every cached stage —
# stage caches are keyed on CODE_REV + input-file signatures so outputs from
# different code revisions can never silently coexist.
CODE_REV = "r11-final"


def fsig(p: str | Path) -> str:
    st = Path(p).stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def load_plan() -> list[dict]:
    """One production row per wrap mesh, joining M2 (decimated source) and M2.5 (overlay)."""
    m2 = json.loads((ROOT / "reports/m2_decimation.json").read_text())["meshes"]
    if isinstance(m2, dict):
        m2 = list(m2.values())
    ink = json.loads((ROOT / "reports/m25_ink.json").read_text())
    overlays = {}
    for rec in ink["records"]:
        out = (rec.get("output") or {}).get("path")
        if out and (ROOT / out).exists():
            overlays[rec["stem"]] = out
    rows = []
    for rec in m2:
        if rec["group"] not in ("A", "B"):
            continue
        stem = rec["stem"]
        obj = rec["obj"]
        rt_plain = ROOT / "outputs/render_textures" / f"{stem}_plain.png"
        rt_ink = ROOT / "outputs/render_textures" / f"{stem}_ink.png"
        rows.append({
            "stem": stem,
            "group": rec["group"],
            "decimated_obj": str(obj),
            # presentation textures (infilled) preferred for frame exports + videos
            "texture": str(rt_plain) if rt_plain.exists()
                       else (str(Path(rec["obj"]).parent / rec["texture"]) if rec.get("texture") else None),
            "overlay": str(rt_ink) if rt_ink.exists() else overlays.get(stem),
        })
    return rows


def state_path(stem: str) -> Path:
    return ROOT / "outputs/anim" / stem / "produce_state.json"


def get_state(stem: str) -> dict:
    p = state_path(stem)
    return json.loads(p.read_text()) if p.exists() else {}


def set_state(stem: str, **kv) -> None:
    p = state_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    s = get_state(stem)
    s.update(kv)
    p.write_text(json.dumps(s, indent=1))


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-4000:]


ANIMATE_OVERRIDES = {
    # auto_trace: junk-heavy bands need a smoother rigid field (passes at sigma=45)
    "auto_trace_20260121115644348_4": ["--roll-smooth", "45"],
}


def stage_animate(row: dict) -> tuple[str, bool, str]:
    stem = row["stem"]
    insig = f"{CODE_REV}|{fsig(ROOT / row['decimated_obj'])}"
    st = get_state(stem)
    if st.get("animate") == "ok" and st.get("animate_sig") == insig:
        return stem, True, "cached"
    rc, log = run(["uv", "run", "python", "scripts/animate.py", row["decimated_obj"],
                   *ANIMATE_OVERRIDES.get(stem, [])])
    ok = rc == 0
    set_state(stem, animate="ok" if ok else "FAIL", animate_sig=insig if ok else "",
              animate_log=log[-1500:] if not ok else "")
    return stem, ok, "" if ok else log[-400:]


def stage_frames(row: dict) -> tuple[str, bool, str]:
    stem = row["stem"]
    frames_npz = ROOT / "outputs/anim" / stem / "frames.npz"
    if not frames_npz.exists():
        return stem, False, "frames.npz missing (animate stage incomplete)"
    insig = f"{CODE_REV}|{fsig(frames_npz)}|{fsig(ROOT / row['texture'])}" + (
        f"|{fsig(ROOT / row['overlay'])}" if row["overlay"] else "")
    st = get_state(stem)
    if st.get("frames") == "ok" and st.get("frames_sig") == insig:
        return stem, True, "cached"
    cmd = ["uv", "run", "python", "scripts/export_frames.py", f"outputs/anim/{stem}",
           "--texture", row["texture"]]
    if row["overlay"]:
        cmd += ["--overlay", row["overlay"]]
    rc, log = run(cmd)
    ok = rc == 0
    set_state(stem, frames="ok" if ok else "FAIL", frames_sig=insig if ok else "",
              frames_log="" if ok else log[-1500:])
    return stem, ok, "" if ok else log[-400:]


def stage_videos(row: dict) -> tuple[str, bool, str]:
    stem = row["stem"]
    frames_npz = ROOT / "outputs/anim" / stem / "frames.npz"
    if not frames_npz.exists():
        return stem, False, "frames.npz missing (animate stage incomplete)"
    insig = f"{CODE_REV}|{fsig(frames_npz)}|{fsig(ROOT / row['texture'])}" + (
        f"|{fsig(ROOT / row['overlay'])}" if row["overlay"] else "")
    st = get_state(stem)
    variants = [v for v in VARIANTS if v == "plain" or row["overlay"]]
    fails = []
    for variant in variants:
        for framing in FRAMINGS:
            key = f"video_{variant}_{framing}"
            if st.get(key) == "ok" and st.get(f"{key}_sig") == insig:
                continue
            cmd = ["uv", "run", "python", "scripts/render_video.py", f"outputs/anim/{stem}",
                   row["group"], "--variant", variant, "--framing", framing]
            rc, log = run(cmd)
            ok = rc == 0
            set_state(stem, **{key: "ok" if ok else "FAIL",
                               f"{key}_sig": insig if ok else ""})
            st = get_state(stem)
            if not ok:
                fails.append(f"{key}: {log[-300:]}")
                break  # plain failure blocks reveal (rawcache); stop this mesh
        if fails:
            break
    return stem, not fails, "; ".join(fails)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "animate", "frames", "videos"])
    ap.add_argument("--clean", action="store_true",
                    help="delete every derived per-mesh artifact (videos, frame OBJs, "
                         "rawcaches, camera plans, produce_state) before running — the "
                         "only safe way to purge mixed-code-era outputs")
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--gpu-workers", type=int, default=3,
                    help="concurrent video jobs (each = one EGL context + one x264-slow "
                         "encoder; the encoder is the bottleneck, the 4090 multiplexes fine)")
    args = ap.parse_args()

    rows = load_plan()
    if args.only:
        rows = [r for r in rows if r["stem"] in set(args.only)]
    print(f"production plan: {len(rows)} wraps "
          f"({sum(1 for r in rows if r['overlay'])} with ink variants)")

    if args.clean:
        for r in rows:
            d = ROOT / "outputs/anim" / r["stem"]
            if not d.exists():
                continue
            for sub in ("video", "frames_obj", "keyframes_m4"):
                if (d / sub).exists():
                    shutil.rmtree(d / sub)
            for pat in ("produce_state.json", "rawcache_*.json", "rawcache_*.raw",
                        "camera_plan_*.json", "up_choice*.json", "inkflat_*.npy"):
                for f in d.glob(pat):
                    f.unlink()
        print(f"cleaned derived artifacts for {len(rows)} meshes")

    failures: list[str] = []

    def pooled(stage_fn, label):
        nonlocal failures
        todo = rows
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as ex:
            futs = {ex.submit(stage_fn, r): r["stem"] for r in todo}
            for fut in as_completed(futs):
                stem, ok, msg = fut.result()
                print(f"  [{label}] {stem}: {'ok' if ok else 'FAIL'} {msg}")
                if not ok:
                    failures.append(f"{label}:{stem}")

    if args.stage in ("all", "animate"):
        print("== stage 1: animate (CPU pool) ==")
        pooled(stage_animate, "animate")
    if args.stage in ("all", "frames"):
        print("== stage 2: frame OBJ export (CPU pool) ==")
        pooled(stage_frames, "frames")
    if args.stage in ("all", "videos"):
        print(f"== stage 3: videos ({args.gpu_workers} concurrent render+encode jobs) ==")
        # one worker owns one MESH at a time (its variants chain through the rawcache);
        # distinct meshes are fully independent.
        with ProcessPoolExecutor(max_workers=args.gpu_workers) as ex:
            futs = {ex.submit(stage_videos, r): r["stem"] for r in rows}
            for fut in as_completed(futs):
                stem, ok, msg = fut.result()
                print(f"  [videos] {stem}: {'ok' if ok else 'FAIL'} {msg}", flush=True)
                if not ok:
                    failures.append(f"videos:{stem}")
                free_gb = shutil.disk_usage(ROOT).free / 1e9
                if free_gb < 60:
                    print(f"!! low disk: {free_gb:.0f} GB free — cancelling remaining jobs")
                    failures.append("disk")
                    for f2 in futs:
                        f2.cancel()
                    break

    print(f"== production: {'GREEN' if not failures else 'RED'} ({len(failures)} failures) ==")
    if failures:
        print("failures:", failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
