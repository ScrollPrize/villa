#!/usr/bin/env python
"""M4 gate (qa-gates contract): pilots x {plain,ink,reveal} x {h,v}.

Green (exit 0) iff:
  - all 12 pilot videos exist as 4K masters AND 1080p derivatives (24 files) and
    every file passes ffprobe checks: 30 fps, yuv420p, correct resolution,
    frame count per spec (240 morph / 330 reveal), bt709, +faststart;
  - 5 QA keyframes per video exist under reports/m4_keyframes/<stem>_<variant>_<framing>/
    (6 for reveal: + kf_reveal_mid.png), committed copies <= 1280 px;
  - reports/m4_pilot_review.md records TWO consecutive PASSING vision-rubric
    rounds (machine-readable ROUND_RESULT lines: all dimensions >= 4/5, zero
    artifact flags, all 12 videos covered).

Usage: uv run python scripts/gate_m4.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from scrollkit.render.encode import ffprobe_check  # noqa: E402

PILOTS = [
    ("w030_2025122002", "A"),
    ("w011_20260108140509268_merged_00", "B"),
]
VARIANTS = ("plain", "ink", "reveal")
FRAMINGS = ("h", "v")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/global.yaml").read_text())
    fdims = cfg["render"]["framings"]
    vcfg = cfg["variants"]
    morph_frames = int(cfg["unwrap"]["frames"])
    reveal_frames = morph_frames + int(vcfg["reveal_hold_in"]) + int(vcfg["reveal_fade"]) \
        + int(vcfg["reveal_hold_out"])

    failures: list[str] = []
    ok_count = 0
    print(f"{'video':<58} {'4k':>14} {'1080p':>14}")

    # ---- 1) encodes: 12 videos x (4K master + 1080p derivative) ----------------------
    for stem, _group in PILOTS:
        vdir = ROOT / "outputs/anim" / stem / "video"
        for variant in VARIANTS:
            frames = reveal_frames if variant == "reveal" else morph_frames
            for framing in FRAMINGS:
                fkey = "horizontal" if framing == "h" else "vertical"
                w4, h4 = int(fdims[fkey]["width"]), int(fdims[fkey]["height"])
                w1, h1 = (1920, 1080) if framing == "h" else (1080, 1920)
                row = []
                for res, (w, h) in (("4k", (w4, h4)), ("1080p", (w1, h1))):
                    path = vdir / f"{stem}_{variant}_{framing}_{res}.mp4"
                    try:
                        if not path.is_file():
                            raise RuntimeError("missing")
                        st = ffprobe_check(path, width=w, height=h, frames=frames)
                        row.append(f"{st['file_size'] / 2**20:7.1f} MB ok")
                        ok_count += 1
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{path.name}: {exc}")
                        row.append("FAIL")
                print(f"{stem[:30] + '_' + variant + '_' + framing:<58} {row[0]:>14} {row[1]:>14}")

    # ---- 2) QA keyframes ---------------------------------------------------------------
    for stem, _group in PILOTS:
        for variant in VARIANTS:
            for framing in FRAMINGS:
                kdir = ROOT / "reports/m4_keyframes" / f"{stem}_{variant}_{framing}"
                expected = {f"kf_t{t:.2f}.png" for t in (0.0, 0.25, 0.5, 0.75, 1.0)}
                if variant == "reveal":
                    expected.add("kf_reveal_mid.png")
                have = {p.name for p in kdir.glob("*.png")} if kdir.is_dir() else set()
                missing = expected - have
                if missing:
                    failures.append(f"keyframes {kdir.name}: missing {sorted(missing)}")
                    continue
                for name in expected:
                    with Image.open(kdir / name) as im:
                        if max(im.size) > 1280:
                            failures.append(f"keyframe {kdir.name}/{name}: {im.size} > 1280px")

    # ---- 3) vision review: two consecutive passing rounds -----------------------------
    review = ROOT / "reports/m4_pilot_review.md"
    if not review.is_file():
        failures.append("reports/m4_pilot_review.md missing")
    else:
        rounds = []
        for line in review.read_text().splitlines():
            m = re.match(r"^ROUND_RESULT:\s*(\{.*\})\s*$", line.strip())
            if m:
                rounds.append(json.loads(m.group(1)))
        if len(rounds) < 2:
            failures.append(f"review: need >= 2 rubric rounds, found {len(rounds)}")
        else:
            for r in rounds[-2:]:
                label = f"round {r.get('round')}"
                if not r.get("pass"):
                    failures.append(f"review {label}: pass != true")
                if int(r.get("videos", 0)) != len(PILOTS) * len(VARIANTS) * len(FRAMINGS):
                    failures.append(f"review {label}: covers {r.get('videos')} videos, need 12")
                if float(r.get("min_score", 0)) < 4:
                    failures.append(f"review {label}: min_score {r.get('min_score')} < 4")
                if int(r.get("artifact_flags", 1)) != 0:
                    failures.append(f"review {label}: artifact_flags {r.get('artifact_flags')} != 0")
            if not failures:
                print(f"review: rounds {rounds[-2]['round']} and {rounds[-1]['round']} "
                      f"both PASS (min_score >= 4, zero artifact flags)")

    print(f"\nencodes ok: {ok_count}/24")
    if failures:
        print("\nM4 GATE: RED")
        for f in failures:
            print("  -", f)
        return 1
    print("M4 GATE: GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
