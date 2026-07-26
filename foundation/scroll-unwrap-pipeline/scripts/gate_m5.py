#!/usr/bin/env python
"""Milestone M5 gate — fleet production (docs/QUALITY-GATES.md, M5 section)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import yaml  # noqa: E402

from produce import FRAMINGS, VARIANTS, load_plan  # noqa: E402

_VCFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/global.yaml")
                       .read_text())["variants"]
REVEAL_FRAMES = (240 + int(_VCFG["reveal_hold_in"]) + int(_VCFG["reveal_fade"])
                 + int(_VCFG["reveal_hold_out"]) + int(_VCFG.get("reveal_zoom_hold", 0)))

FAIL: list[str] = []
WAIVED: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAIL.append(msg)
        print("  FAIL  " + msg)


def ffprobe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,pix_fmt,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    s = json.loads(r.stdout)["streams"][0]
    return s


def main() -> int:
    rows = load_plan()
    exceptions = {}
    exc_p = ROOT / "reports/m5_unwrap_exceptions.json"
    if exc_p.exists():
        exceptions = json.loads(exc_p.read_text())

    n_anim = n_videos = 0
    for r in rows:
        stem = r["stem"]
        adir = ROOT / "outputs/anim" / stem
        mp = adir / "metrics.json"
        check(mp.exists(), f"{stem}: metrics.json")
        if mp.exists():
            g = json.loads(mp.read_text())["meta"]["gates"]
            if g.get("pass"):
                n_anim += 1
            elif stem in exceptions and exceptions[stem].get("verdict") == "waived_visually_clean":
                WAIVED.append(stem)
                n_anim += 1
            else:
                check(False, f"{stem}: transit gates fail and no approved waiver")
            # clearance certificate is never waivable (pierce-through/z-fight class)
            check(g.get("clearance_ok") is True,
                  f"{stem}: clearance certificate (worst margin "
                  f"{g.get('clearance_worst_margin')} at frame {g.get('clearance_worst_frame')})")
        # era consistency: every derived artifact must be NEWER than its trajectory,
        # so videos rendered from an older code revision can never ship silently
        # after a partial batch relaunch.
        fnpz = adir / "frames.npz"
        check(fnpz.exists(), f"{stem}: frames.npz")
        if fnpz.exists():
            t_frames = fnpz.stat().st_mtime
            for vid in sorted((adir / "video").glob("*.mp4")) if (adir / "video").exists() else []:
                check(vid.stat().st_mtime > t_frames,
                      f"{stem}: {vid.name} older than frames.npz (stale code-era video)")
        # frame OBJs
        fdir = adir / "frames_obj/plain"
        objs = list(fdir.glob("frame_*.obj")) if fdir.exists() else []
        check(len(objs) == 240, f"{stem}: 240 plain frame OBJs (got {len(objs)})")
        check((fdir / "mesh.mtl").exists(), f"{stem}: plain MTL")
        if r["overlay"]:
            idir = adir / "frames_obj/ink"
            iobjs = list(idir.glob("frame_*.obj")) if idir.exists() else []
            check(len(iobjs) == 240, f"{stem}: 240 ink frame OBJs (got {len(iobjs)})")
            mtl = (idir / "mesh.mtl").read_text() if (idir / "mesh.mtl").exists() else ""
            check("ink_recto" in mtl and "plain_verso" in mtl and mtl.count("map_Kd") == 2,
                  f"{stem}: ink MTL is a two-sided shell (recto ink / verso plain)")
            if iobjs:
                head = iobjs[0].read_text().split("\n", 2)[0]
                check("shell" in head, f"{stem}: ink frame OBJs are shell exports")
        # videos
        variants = [v for v in VARIANTS if v == "plain" or r["overlay"]]
        for variant in variants:
            for framing in FRAMINGS:
                n_expected = REVEAL_FRAMES if variant == "reveal" else 240
                dims = {("h", "4k"): (3840, 2160), ("h", "1080p"): (1920, 1080),
                        ("v", "4k"): (2160, 3840), ("v", "1080p"): (1080, 1920)}
                for res in ("4k", "1080p"):
                    w, h = dims[(framing, res)]
                    vid = adir / "video" / f"{stem}_{variant}_{framing}_{res}.mp4"
                    if not vid.exists():
                        check(False, f"{stem}: missing video {vid.name}")
                        continue
                    s = ffprobe(vid)
                    ok = (s.get("pix_fmt") == "yuv420p"
                          and s.get("r_frame_rate") == "30/1"
                          and int(s.get("nb_frames", 0)) == n_expected
                          and (int(s.get("width", 0)), int(s.get("height", 0))) == (w, h))
                    check(ok, f"{stem}: ffprobe {vid.name} (got {s.get('width')}x{s.get('height')} "
                              f"{s.get('pix_fmt')} {s.get('r_frame_rate')} {s.get('nb_frames')}f, "
                              f"want {w}x{h} {n_expected}f)")
                    n_videos += 1
    # vision verdicts
    vq = ROOT / "reports/m5_vision_qa.json"
    check(vq.exists(), "vision QA verdicts exist (reports/m5_vision_qa.json)")
    if vq.exists():
        v = json.loads(vq.read_text())
        bad = [k for k, r2 in v.get("verdicts", {}).items() if not r2.get("pass")]
        check(not bad, f"vision QA pass for all mesh×variant (failing: {bad[:8]})")
    check((ROOT / "reports/production_report.md").exists(), "production report exists")

    print(f"== M5 summary: {n_anim}/{len(rows)} animations green "
          f"({len(WAIVED)} visually-waived: {WAIVED}), {n_videos} videos checked ==")
    print("== M5:", "GREEN ==" if not FAIL else f"RED ({len(FAIL)} failures) ==")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
