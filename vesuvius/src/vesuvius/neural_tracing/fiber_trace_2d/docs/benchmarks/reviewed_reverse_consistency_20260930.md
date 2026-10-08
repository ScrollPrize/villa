# Periodic reverse-consistency replay: reviewed91

## Protocol and provenance

Same frozen91 approved annotations /182 directions /1386.175605 directed mm
as reviewed_fibers_corrected_20260930.md. Gaussian bestEMA999 u17700 first,
then lowres212757 bestEMA999 u22000. Original checkpoints, cohort hashes,
source bindings, prediction namespaces and192-worker scheduling unchanged.
GPU0–5 only on demand. These are experimental evaluations, not new defaults.

- Check every256base voxels of forward traced travel; reset window after each
  restart, and also check the final partial window once. Reverse trace uses
  the same production beam and20normal/40in-sheet base-voxel acceptance radii.
- Shared native verifier extracted from dense-crop tracing, preserving its
  monotone local matching, same beam settings, and prediction-only comparison
  to the forward span. The checker itself does not access GT.
- On rejection discard the whole unverified window, retain geometry through
  its previous successful checkpoint, and record a **filtered stop**, not a
  normal GT failure. An earlier ordinary GT threshold failure takes precedence.
- Search strictly later along the same eligible reference span, starting one
  nominal forward step beyond detection (8base Gaussian,32base lowres).
  At each location sample the transverse lattice at prediction resolution
  (2base Gaussian,8base lowres), within10normal/20in-sheet ellipse.
- Require positive presence, >= all8 transverse neighbors and >at least one,
  unsigned predicted/reference tangent alignment<=25deg. Choose closest
  eligible peak by normalized ellipse distance; orient predicted direction
  along requested reference direction. No fallback seed; if none exists,
  explicitly skip the tail. Completely flat3x3 neighborhoods are not accepted.
- Ordinary restarts retain80base episode suppression. Filtered stops are
  separate, unsuppressed. Artifacts store [rejected begin, detection, reseed/end]
  reference arcs and actual retained segment/seed positions. This is a
  **GT-assisted evaluation restart policy**, not blind deployment tracing.

## Results

| Model | Baseline ordinary | New ordinary | Filtered stops | All interruptions | Retained % | Filter-skipped mm | Checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 32 | 12 | 62 | 74 | 97.2928 | 37.5082 | 2319 |
| lowres | 29 | 15 | 51 | 66 | 97.7023 | 31.8499 | 2301 |

| Model | Scroll | Baseline ordinary | New ordinary | Filtered | Retained % | Filter-skipped mm |
|---|---|---:|---:|---:|---:|---:|
| gaussian | PHerc0139 | 12 | 4 | 20 | 95.8918 | 11.7176 |
| gaussian | PHerc0332 | 0 | 0 | 0 | 100.0000 | 0.0000 |
| gaussian | PHerc1299 | 4 | 5 | 10 | 95.7045 | 6.3328 |
| gaussian | PHerc1451 | 0 | 0 | 0 | 100.0000 | 0.0000 |
| gaussian | PHercParis4 | 16 | 3 | 32 | 97.9101 | 19.4579 |
| lowres | PHerc0139 | 10 | 5 | 22 | 95.2357 | 13.5889 |
| lowres | PHerc0332 | 0 | 0 | 0 | 100.0000 | 0.0000 |
| lowres | PHerc1299 | 4 | 4 | 1 | 99.5682 | 0.6366 |
| lowres | PHerc1451 | 0 | 0 | 0 | 100.0000 | 0.0000 |
| lowres | PHercParis4 | 15 | 6 | 28 | 98.1088 | 17.6243 |

Fewer ordinary GT failures are accompanied by more interruptions overall;
this is not an unconditional improvement in continuity. Gaussian loses2.7072%
of reference coverage; lowres loses2.2977%. Gaussian has0.018108mm additional
untraced ordinary-reset advances; other missing coverage is filtering/reseeding.
Retained coverage does not mean every point is within the GT ellipse: ordinary
excursion grace and failures remain part of the evaluation.

Legacy mean-distance fields elsewhere in summary.json still partition the
entire reference including skips: **do not interpret these as uninterrupted
retained tracing lengths**. The new consistency block reports retained
length/(ordinary episodes+filtered stops+directed cases):5.268161mm Gaussian,
5.460991mm lowres. This is a separately labeled piece-length diagnostic,
not the historical global reference/(restarts+1) metric.

## Outputs and commands

Root: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reverse256_v1

Each model has workload.json, full/summary.json (including consistency totals
and per-scroll breakdown), full/inference_summary.json and full/cases/*/trace.json.
No visualization requested or run. Older results/cache exports remain unchanged;
Gaussian acquired additional coverage, not included in the earlier frozen link
inventory.

Set scripts/training_failure_mining_control.json to
{"action":"reviewed_reverse","reviewed_model":"gaussian"} or model lowres, then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Native option: `--replay-reverse-interval-base 256`.
Zero config default leaves replay unchanged.
Build/test uses same runner with action build. No installs.

## Verification and timings

Workspace RelWithDebInfo, one run/model: all182directions complete, no
infrastructure errors. Gaussian coordinator76.246s/native65.302s with863new
inference tiles; lowres coordinator33.927s/native23.289s with0newtiles. These
are mixed-cache end-to-end observations, not a controlled speed comparison.
Independent plan and implementation reviews completed.
Native tests:23replay,88trace,25reference,86crop;4CLIcoordinate tests.
Python:8reviewed/consistency tests and15existing replay-metric tests.
Default-path/crop regression tests retained; new tests cover partial-window
checks, rejection rollback, no-seed tail, later reseed, peak/angle/ellipse
criteria, source sign orientation and straight reverse verification.

Limitations: sample-based reverse check, not continuous proof; transverse
peak search is lattice-resolution, not subpixel optimization; no peak-height
threshold beyond positive presence. Shared reverse helper currently samples
normals in tracing coordinates, matching these evaluated source bindings.
