# Quality Gates (binding contract)

Each milestone has `scripts/gate_mX.py` → exit 0 = green. Commits land only on
green. Thresholds are never weakened to pass; a failing gate is investigated
with the failing report in hand.

## M0 — Bootstrap + audit
- `pytest` green (convention-trap + analytic cylinder fixtures included).
- `reports/audit.json` covers every mesh + the ink inventory with zero
  unexplained anomalies (each anomaly carries an `explanation`).
- Headless GPU render smoke test; sparse-solver benchmark recorded.

## M1 — Conversion (per mesh)
- OBJ reload: V/N/UV float32 bit-equal, faces identical, texture SHA256 equal.
- Render-parity SSIM(PLY render, OBJ render) ≥ 0.99 at identical camera.

## M2 — Decimation (keep-ladder [5, 8, 12, 20, 35]%; most aggressive green rung wins)
- Render SSIM vs full-res ≥ 0.97 at production framing; ≥ 0.95 on two
  close-up crops (max-curvature and max-text-density regions).
- Two-sided Hausdorff ≤ 1.0 × source mean edge; no new flipped UV triangles;
  boundary length within 2%.
- Group C (full-scroll photogrammetry) extends the ladder with [0.5, 0.65,
  0.8]; if no rung passes, the mesh ships undecimated with a pinned rationale
  (`no_safe_decimation`) — these models are not oversampled relative to their
  textures, so any reduction is perceptually lossy.

## M2.5 — Ink overlays (per matched mesh)
- Matching table complete (every mesh matched or explained); overlay dims ==
  base texture dims; alpha byte-identical to base.
- Polarity sanity: ink coverage within the papyrus mask ∈ [0.5%, 35%] or
  flagged with an explanation.

## M3 — Unwrap core (every frame)
- Topology: V/F identity, all finite. Flipped/degenerate ≤ 0.01% (target 0).
- Area (HARD): per-triangle vs endpoint blend, quantiles over VISIBLE faces
  (texture alpha at the face's UV centroid) — P95 ≤ max(2%, 3.0 × the mesh's
  de-normalization rel. residual). Edge lengths: same form. Real wraps are
  non-developable, so transit distortion is bounded below by the chart's own
  non-isometry; thresholds scale with the measured residual while the failure
  modes they exist to catch (rotation seams, noisy axis fields, integration
  drift) still fail by an order of magnitude.
- Symmetric Dirichlet (3-frame rolling minimum = the SUSTAINED level)
  ≤ 1.15 × max(E0, E1) + 0.05; raw peak reported.
- Temporal smoothness (popping): per-frame P99 displacement sequence must
  have no discontinuity — max |Δ| ≤ max(0.7 × median, 0.30 × peak).
  Calibrated for the flap-bridge kinematics: visually-clean reference runs
  measure 0.33–0.56 med-ratio / ≤0.26 peak-ratio; true junction snaps measure
  2.9–3.8 / 0.35–0.47 — the thresholds sit inside that gap.
- Clearance certificate (HARD, every frame, non-waivable): no rolled vertex
  within 0.05 × median-edge of (or below) the settled sheet at the same plan
  cell, landing flap exempt; exact-state mode for rolling kinematics.
  Quantile gates are provably blind to sub-percent vertex populations; this
  certificate is the guard for the interpenetration class.
- Chirality: numeric `mirror_check > 0` per mesh; the on-screen reading
  direction is a camera decision verified visually.

## M4 — Cinematics
- ffprobe: 30 fps, yuv420p, exact resolution and frame count, faststart.
- QA keyframes exist per video; vision rubric (RENDER-STYLE) all dimensions
  ≥ 4/5 with zero artifact flags.

## M5 — Batch (fleet)
- M3 gates green per mesh (numeric, automated); a metrics-exception ships
  only with a recorded visually-clean waiver.
- ffprobe checks for every deliverable video (4K + 1080p, both framings).
- Frame-OBJ folders complete (240 OBJs + MTL + texture); ink variant is a
  two-sided shell (recto ink / verso plain).
- Era consistency: every video newer than its trajectory (`frames.npz`) —
  artifacts from different code revisions can never silently coexist.
- Vision-review verdicts pass for every mesh × variant; production report
  written.

## Vision review interface
The M5 gate reads `reports/m5_vision_qa.json`. Reviews may be produced by any
process (human or model) applying the rubric in `docs/RENDER-STYLE.md` to the
QA keyframes under `reports/m4_keyframes/<stem>_<variant>_<framing>/`. Schema:

```json
{
  "verdicts": {
    "<stem>|<variant>": {
      "pass": true,
      "scores": {"artifact_free": 5, "texture_integrity": 5,
                  "framing": 4, "lighting": 5, "motion": 5},
      "flags": [],
      "summary": "pixel-level evidence for the verdict"
    }
  }
}
```

`pass` requires every score ≥ 4 and zero artifact flags.
