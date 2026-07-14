#!/usr/bin/env python
"""Assemble reports/production_report.md from all M5 evidence on disk."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from produce import FRAMINGS, VARIANTS, load_plan  # noqa: E402


def main() -> int:
    rows = load_plan()
    exc = {}
    p = ROOT / "reports/m5_unwrap_exceptions.json"
    if p.exists():
        exc = json.loads(p.read_text())
    vq = {}
    p = ROOT / "reports/m5_vision_qa.json"
    if p.exists():
        vq = json.loads(p.read_text()).get("verdicts", {})
    m2 = {m["stem"]: m for m in json.loads((ROOT / "reports/m2_decimation.json").read_text())["meshes"]}
    infill = json.loads((ROOT / "outputs/render_textures/infill_stats.json").read_text())

    lines = ["# Production report — scroll unwrap fleet\n",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n",
             "\n## Per-mesh summary\n",
             "| mesh | grp | faces (dec) | keep | axis | anim gates | infill | videos | vision |",
             "|---|---|---|---|---|---|---|---|---|"]
    n_vid_total = 0
    disk_video = 0
    for r in rows:
        stem = r["stem"]
        mp = ROOT / "outputs/anim" / stem / "metrics.json"
        g = json.loads(mp.read_text())["meta"]["gates"] if mp.exists() else {}
        anim = "PASS" if g.get("pass") else ("WAIVED" if exc.get(stem, {}).get("verdict") == "waived_visually_clean" else "FAIL")
        vids = list((ROOT / "outputs/anim" / stem / "video").glob("*.mp4"))
        n_vid_total += len(vids)
        disk_video += sum(v.stat().st_size for v in vids)
        inf = (infill.get(stem) or {}).get("plain") or {}
        vq_cell = ",".join(sorted({v.get("verdict", "?") for k, v in vq.items() if k.startswith(stem)})) or "—"
        dec = m2.get(stem, {})
        lines.append(
            f"| {stem} | {r['group']} | {dec.get('faces_out', '—')} | {dec.get('rung_chosen', '—')} "
            f"| {g.get('axis_mode', '—')} | {anim} (sd {g.get('sd_peak', 0):.2f}, "
            f"a95v {g.get('area_p95_vis_max', 0):.3f}/{g.get('area_p95_gate', 0):.3f}) "
            f"| {inf.get('filled_frac_of_papyrus', 0) * 100:.0f}% | {len(vids)} | {vq_cell} |")

    lines += [
        "\n## Fleet accounting\n",
        f"- wraps animated: {len(rows)} (videos on disk: {n_vid_total}, {disk_video / 2**30:.1f} GiB)",
        f"- frame-OBJ exports: {sum(1 for r in rows if (ROOT / 'outputs/anim' / r['stem'] / 'frames_obj/plain').exists())} meshes × 240 frames (+ hardlinked ink sets)",
        f"- unwrap exceptions (vision-disposition): {[k for k in exc]}",
        "\n## Decisions of record\n",
        "- Ink: neutral carbon black #101010, recto-(inner-face)-only via two-sided materials.",
        "- Text rows horizontal in every framing; 9:16 fits by pull-back; reveal = camera push-in then ink fade.",
        "- Render textures: small interior alpha dropout infilled; large genuine lacunae preserved.",
        "- Full gate definitions: docs/QUALITY-GATES.md.",
    ]
    out = ROOT / "reports/production_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
