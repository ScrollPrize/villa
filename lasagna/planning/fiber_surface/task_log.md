# Fixed-UV Lasagna surface task log

- 2026-10-13: Isolated branch global-fiber-sgd-surface from 9425ddab6.
- User confirmed explicit ruled initialization with extrapolation, followed by
  ordinary Lasagna regularization and a weak-to-strong positional data ramp.
- Initial weight must remain positive. UV correspondences are frozen, XYZ is
  optimized; this is positional supervision, not normal-only motion.
- Task documents moved under lasagna/planning/fiber_surface/ at user request.
- Independent plan review approved. Validate fixed UV across scales, finite
  extrapolation domain and unsupported-column behavior, boundary samples, and
  actual fit drift (positive weight alone cannot guarantee spiral preservation).
- Implementation: fixed_uv optimizer term, positive geometric ramp, shared AFV
  PolylineBank, explicit H-only ruled initialization and LZW tifxyz outputs.
- Initial tests: 40 passed; existing atlas speculative-config test expects
  debug=true/vis_interval=10 but committed config has false/1. Unrelated files
  untouched; document this separately, do not silently change its expectations.
- First real pilot: 56 pieces, 81x201 grid, 2490 targets, 300 CPU iterations in
  11.34s. Found pathological extrapolation from nearly coincident V supports:
  XYZ reached -1.1 million base voxels. This pilot is rejected, not a usable
  surface. Bound extrapolation speed to 1 base voxel per UV voxel; preserve
  supported interpolation/targets. Add regression and rerun same fixed patch.
- Imported endpoints can translate to identical float64 U coordinates despite
  distinct source arc roundoff; remove exact duplicate translated U samples.
- Bounded v2 rerun: initialization 0.050s, solve 10.93s; mean error 118.17 ->
  61.34 base vx, RMS 331.14 -> 208.95. Local opposing-triangle count 103 -> 276:
  not an acceptable topology claim. Recorded all data rather than hiding this.
- Stronger existing geometry terms, LR5, 600 steps (v3): mean error 11.86,
  median 3.59, p95 10.39, RMS 68.45; 19.80s solve. 177 opposing-triangle quads
  remain. Fixed-UV constraints may conflict inside a coarse quad; no automatic
  target rejection and no new intersection repair introduced.
- Independent implementation review approved loss/mapping/reuse, identified
  single-H row-dependent slope discontinuity. Fixed by integrating bounded
  donor increments and translating to the observed H; added regression.
- Added manager-controlled CUDA launcher using existing launch_managed;
  CPU pilots do not reserve GPUs. GPU timing remains unmeasured until run.
- Final v4: 600 steps in 19.326s, mean XYZ error116.35→11.06 base vx,
  median25.10→2.68, p95766.12→9.77, RMS318.82→68.41. Remaining maximum841.60
  and 181/16000 local folded quads explicitly reported; not silently filtered.
- Final regression selection47passed, 1known-unrelated deselected, 2.66s;
  git diff --check clean. Exact commands/results in results.md.
- No full-scroll solve or actual CUDA benchmark performed. CPU pilot is the
  first implementation validation; CUDA launch path tested with a stub manager.
  This is not an anatomical/topology validation and does not guarantee no
  self-intersections. Upstream solve and all villa2 files untouched.
