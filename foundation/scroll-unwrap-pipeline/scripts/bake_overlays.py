#!/usr/bin/env python
"""M2.5 — bake amber ink overlays for the 30 matched wraps (19 march + 11 may).

Reads reports/audit.json (ink matching tables) + configs/global.yaml [ink],
writes outputs/overlays/<march|may>/<stem>_inkoverlay.png, the evidence report
reports/m25_ink.{json,md} and reports/overlay_previews/*.jpg.

Spec: docs/RENDER-STYLE.md ('Ink overlay bake').
Gate:  scripts/gate_m25.py (docs/QUALITY-GATES.md, M2.5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.ink import bake as B  # noqa: E402


def _bake_worker(job: dict, params: dict) -> dict:
    return B.bake_one(job, params, root=ROOT)


# ----------------------------------------------------------------- previews


def _flatten(rgba, bg=(16, 17, 20)):
    """Composite RGBA over the style charcoal background (preview display only)."""
    import numpy as np

    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgba[..., :3].astype(np.float32) * a + np.array(bg, np.float32) * (1.0 - a)
    return out.astype(np.uint8)


def make_previews(records: list[dict], n: int = 6) -> list[dict]:
    """Side-by-side (base | overlay) JPEGs, 1200 px wide, q85, for vision review.

    Selection: densest-ink wrap, sparsest, >=1 march TIF source, >=1 may JPG
    source, rest spread across the coverage range / kinds.
    """
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    out_dir = ROOT / "reports/overlay_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    by_cov = sorted(records, key=lambda r: r["polarity"]["coverage_used"])
    picks: list[dict] = []

    def add(rec, why):
        if rec and all(p["stem"] != rec["stem"] for p, _ in picks):
            picks.append((rec, why))

    add(by_cov[-1], "densest ink")
    add(by_cov[0], "sparsest ink")
    crisp = next((r for r in records if r["stem"] == "w013_20240304141531"), None)
    add(crisp, "march TIF source (2024 vintage, crisp letterforms)")
    if not any(p["kind"] == "may" for p, _ in picks):
        may = sorted((r for r in records if r["kind"] == "may"), key=lambda r: -r["polarity"]["coverage_used"])
        add(may[0] if may else None, "may inverted-JPG source")
    # fill to n with mid-coverage picks alternating kinds
    mids = [r for r in by_cov if all(p["stem"] != r["stem"] for p, _ in picks)]
    want = "may" if sum(p["kind"] == "may" for p, _ in picks) <= len(picks) // 2 else "march"
    while len(picks) < n and mids:
        pool = [r for r in mids if r["kind"] == want] or mids
        rec = pool[len(pool) // 2]
        mids = [r for r in mids if r["stem"] != rec["stem"]]
        add(rec, f"mid-coverage {rec['kind']}")
        want = "may" if want == "march" else "march"

    manifest = []
    for i, (rec, why) in enumerate(picks):
        base = np.asarray(Image.open(ROOT / rec["tex_path"]))
        over = np.asarray(Image.open(ROOT / rec["output"]["path"]))
        panels = [_flatten(base), _flatten(over)]
        files = {}
        for tag, crop in (("sbs", None), ("zoom", rec["dense_window_xywh"])):
            ims = []
            for p in panels:
                q = p if crop is None else p[crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]]
                im = Image.fromarray(q)
                tw = 600  # two panels -> 1200 px wide total
                im = im.resize((tw, max(1, round(im.height * tw / im.width))), Image.Resampling.LANCZOS)
                ims.append(im)
            h = max(im.height for im in ims)
            sheet = Image.new("RGB", (sum(im.width for im in ims), h), (16, 17, 20))
            x = 0
            for im in ims:
                sheet.paste(im, (x, (h - im.height) // 2))
                x += im.width
            fp = out_dir / f"{i:02d}_{rec['stem']}_{tag}.jpg"
            sheet.save(fp, quality=85)
            files[tag] = str(fp.relative_to(ROOT))
        manifest.append({"stem": rec["stem"], "kind": rec["kind"], "why": why,
                         "coverage": rec["polarity"]["coverage_used"], **files})
        del base, over, panels
    return manifest


# ----------------------------------------------------------------- reports


def collect_anomalies(records: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    flipped = [r for r in records if r["polarity"]["flipped_vs_audit"]]
    if flipped:
        anomalies.append(
            {
                "kind": "polarity_label_flipped_vs_audit",
                "severity": "loud",
                "meshes": [r["stem"] for r in flipped],
                "detail": (
                    f"{len(flipped)}/{len(records)} wraps: the audit polarity label failed the "
                    "per-image sparse-tail verification and was flipped. All 19 March TIFs are "
                    "labelled 'positive (ink bright)' in reports/audit.json but the rasters are "
                    "ink-DARK on a light sheet (background ~208/255, letterforms dark; visually "
                    "confirmed Greek letterforms in w013 render dark, identical convention to the "
                    "May 'predictions_inv' JPGs). With the audit label the in-mask ink fraction "
                    "is wildly out of band (papyrus counted as ink — see per-mesh evidence); "
                    "inverted it lands in the [0.5%,35%] band for every wrap."
                ),
                "evidence": [
                    {
                        "stem": r["stem"],
                        "coverage_as_audit_label": r["polarity"]["coverage_as_audit_label"],
                        "coverage_flipped_used": r["polarity"]["coverage_used"],
                    }
                    for r in flipped
                ],
                "explanation": (
                    "The audit's March polarity was an unverified assumption (the May entries carry "
                    "polarity_verified_mean_gt_half, the March entries do not). Pixel data win: ink "
                    "is normalized as 1 - v/255 for March as well, so ink=high before grading."
                ),
            }
        )
    flipped_reg = [r for r in records if r["registration"]["variant_used"] != "identity"]
    if flipped_reg:
        anomalies.append(
            {
                "kind": "registration_flip_applied",
                "severity": "loud",
                "meshes": [r["stem"] for r in flipped_reg],
                "detail": (
                    f"{len(flipped_reg)} wraps violated the same-pixel-grid assumption and were "
                    "decisively fixed by a flip variant. All 11 May predictions are FLIPUD "
                    "relative to their PLY composite textures: the predictions are pixel-aligned "
                    "to the 'max comp 5' composites (footprint IoU 0.97 at identity on the native "
                    "10724x9021 canvas), and those composites are stored vertically flipped vs "
                    "the textures (zero-shift content correlation ~0.87-0.94 for flipud vs "
                    "~0.24-0.43 for identity/fliplr/rot180)."
                ),
                "evidence": [
                    {
                        "stem": r["stem"],
                        "variant_used": r["registration"]["variant_used"],
                        "identity_outside_frac": r["registration"]["identity"]["outside_frac"],
                        "fixed_outside_frac": r["registration"]["outside_frac"],
                        "iou_identity_vs_used": [r["registration"]["identity"]["iou"], r["registration"]["iou"]],
                        "content_check": (r["registration"]["anomaly"] or {}).get("content_check"),
                    }
                    for r in flipped_reg
                ],
                "explanation": (
                    "Residual outside-mask ink after the flip (~31-43%) is NOT misregistration: "
                    "the prediction/mc5 canvas carries sheet regions that are alpha-0 holes in "
                    "this texture (the mc5 render covers more papyrus than the mesh chart). Those "
                    "texels are invisible under alpha-MASK cutout rendering, so they cannot leak "
                    "into the videos."
                ),
            }
        )
    halo_reg = [r for r in records
                if r["registration"]["anomaly"] and r["registration"]["variant_used"] == "identity"]
    if halo_reg:
        anomalies.append(
            {
                "kind": "registration_edge_halo",
                "severity": "loud",
                "meshes": [r["stem"] for r in halo_reg],
                "detail": (
                    f"{len(halo_reg)} wraps exceed the 3% outside-mask budget with identity "
                    "remaining the decisively best variant (flips score far worse IoU) — the "
                    "outside ink is a sheet-edge halo: the prediction render's silhouette is "
                    "slightly fatter than the texture's alpha cutout, plus resampling blur of "
                    "edge gradients and (2026-vintage TIFs) soft shading at the sheet tips."
                ),
                "evidence": [
                    {
                        "stem": r["stem"],
                        "outside_frac": r["registration"]["outside_frac"],
                        "iou_identity": r["registration"]["identity"]["iou"],
                        "best_flip_iou": max(
                            v["iou"] for k, v in r["registration"]["anomaly"]["variant_scores"].items()
                            if k != "identity"
                        ),
                        "halo": r["registration"]["anomaly"].get("halo_identity"),
                    }
                    for r in halo_reg
                ],
                "explanation": (
                    "Outside-mask texels are invisible at render time (alpha MASK cutout, "
                    "threshold 0.5), so the halo cannot appear in any output; in-mask "
                    "registration is verified by the IoU dominance of identity."
                ),
            }
        )
    for r in records:
        cov = r["polarity"]["coverage_used"]
        if not (B.COVERAGE_BAND[0] <= cov <= B.COVERAGE_BAND[1]):
            anomalies.append(
                {
                    "kind": "coverage_out_of_band",
                    "stem": r["stem"],
                    "coverage": cov,
                    "explanation": r["coverage"].get("explanation", ""),
                }
            )
    return anomalies


def write_md(report: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# M2.5 — Ink overlay bake report\n")
    A(f"Generated {report['generated_at']} — {report['n_baked']}/30 overlays baked "
      f"in {report['wall_runtime_s']:.0f}s wall ({report['workers']} workers).\n")
    A("## Parameters (configs/global.yaml [ink])\n")
    A("```json\n" + json.dumps(report["params"], indent=2) + "\n```\n")
    A(f"- Normalization: `{report['records'][0]['normalization']}`")
    A(f"- Ink fraction (primary coverage metric): in-mask fraction with normalized ink past the "
      f"smoothstep midpoint (lo+hi)/2 — visibly inked, opacity > half max. Band: "
      f"[{B.COVERAGE_BAND[0]:.1%}, {B.COVERAGE_BAND[1]:.0%}]. The smoothstep-onset fraction "
      f"(ink > lo) is recorded as a secondary stat.")
    A(f"- Registration: ink>0.3·max must keep <{B.REG_OUTSIDE_MAX:.0%} of its pixels outside alpha>127\n")
    A("## Matching table (from reports/audit.json: ink)\n")
    A("| mesh stem | kind | ink source | status |")
    A("|---|---|---|---|")
    for r in report["records"]:
        A(f"| {r['stem']} | {r['kind']} | `{r['ink_path']}` | matched |")
    for u in report["matching"]["unmatched_meshes"]:
        A(f"| {Path(u).stem} | march | — | **UNMATCHED** (no same-stem TIF; auto-trace segment traced "
          f"2026-01-21, after the ink-detection batch — plain-texture renders only) |")
    for o in report["matching"]["orphan_inks"]:
        A(f"| — | march | `{o}` | **ORPHAN** (no w010 mesh was delivered; kept for provenance) |")
    A("")
    A("## Per-mesh results\n")
    A("| stem | kind | tex W×H | resample | polarity used | pol flip? | ink frac | onset frac | mean op | reg outside | reg variant | runtime |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report["records"]:
        steps = " + ".join(
            f"block_mean/{s['factor']}" if s["op"] == "block_mean" else "lanczos" for s in r["resample_steps"]
        )
        pol = r["polarity"]
        reg = r["registration"]
        cov = r["coverage"]
        flag = "" if B.COVERAGE_BAND[0] <= pol["coverage_used"] <= B.COVERAGE_BAND[1] else " ⚠"
        A(f"| {r['stem']} | {r['kind']} | {r['tex_wh'][0]}×{r['tex_wh'][1]} | {steps} "
          f"| {pol['used'].split(' ')[0]} | {'YES' if pol['flipped_vs_audit'] else 'no'} "
          f"| {pol['coverage_used']:.2%}{flag} | {cov['onset_frac_in_mask']:.2%} "
          f"| {cov['mean_opacity_in_mask']:.4f} "
          f"| {reg['outside_frac']:.3%} | {reg['variant_used']} | {r['runtime_s']:.0f}s |")
    A("")
    A("## Anomalies\n")
    if not report["anomalies"]:
        A("None.")
    for an in report["anomalies"]:
        A(f"### {an['kind']}" + (f" — {an['stem']}" if "stem" in an else ""))
        A(an.get("detail", ""))
        if an.get("explanation"):
            A(f"\n*Explanation:* {an['explanation']}")
        if an["kind"] == "polarity_label_flipped_vs_audit":
            A("\n| stem | coverage @ audit label | coverage flipped (used) |")
            A("|---|---|---|")
            for e in an["evidence"]:
                A(f"| {e['stem']} | {e['coverage_as_audit_label']:.2%} | {e['coverage_flipped_used']:.2%} |")
        A("")
    A("## Previews (reports/overlay_previews/)\n")
    for p in report["previews"]:
        A(f"- **{p['stem']}** ({p['kind']}, coverage {p['coverage']:.2%}, {p['why']}): "
          f"`{p['sbs']}`, zoom `{p['zoom']}`")
    A("")
    (ROOT / "reports/m25_ink.md").write_text("\n".join(L))


# ----------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="bake only these stems (debug)")
    ap.add_argument("--no-previews", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    params = B.load_params(ROOT)
    audit = B.load_audit(ROOT)
    jobs = B.build_jobs(audit, ROOT)
    if args.only:
        jobs = [j for j in jobs if j["stem"] in set(args.only)]
    # largest ink first → better parallel packing, big TIFs never queue at the end
    jobs.sort(key=lambda j: -j["ink_size"])
    print(f"baking {len(jobs)} overlays with {args.workers} workers")

    records: list[dict] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_bake_worker, j, params): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                failures.append(f"{j['stem']}: {e}")
                print(f"  FAIL {j['stem']}: {e}")
                continue
            records.append(rec)
            print(f"  ok {rec['stem']:45s} cov={rec['polarity']['coverage_used']:7.2%} "
                  f"flip={'Y' if rec['polarity']['flipped_vs_audit'] else 'n'} "
                  f"reg_out={rec['registration']['outside_frac']:.3%} {rec['runtime_s']:6.1f}s")

    records.sort(key=lambda r: (r["kind"], r["stem"]))
    previews = [] if (args.no_previews or args.only) else make_previews(records)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "workers": args.workers,
        "n_baked": len(records),
        "failures": failures,
        "matching": {
            "n_march_matched": len(audit["ink"]["march_matches"]),
            "n_may_matched": len(audit["ink"]["may_matches"]),
            "unmatched_meshes": audit["ink"]["unmatched_meshes"],
            "orphan_inks": [o["tif"] for o in audit["ink"]["orphans"]],
        },
        "coverage_band": list(B.COVERAGE_BAND),
        "registration_outside_max": B.REG_OUTSIDE_MAX,
        "records": records,
        "anomalies": collect_anomalies(records),
        "previews": previews,
        "wall_runtime_s": round(time.time() - t0, 1),
    }
    if not args.only:
        (ROOT / "reports/m25_ink.json").write_text(json.dumps(report, indent=1))
        write_md(report)
        print(f"wrote reports/m25_ink.json + .md + {len(previews)} preview pairs")
    print(f"done: {len(records)}/{len(jobs)} in {report['wall_runtime_s']:.0f}s wall")
    return 0 if not failures and len(records) == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
