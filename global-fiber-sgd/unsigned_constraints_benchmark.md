# Unsigned local-event constraints from Qual Paris4 AFV

2026-10-08. Completed; no whole-volume run. No normal sign alignment, BP solve,
tracing, inference, fiber cutting or optimizer. Historical dense alignment was
stopped and removed from this adapter following user clarification.

## Method

Native C++ shared `extractFiberTraceConstraints`, `localEvents=true`,
`localKeepWholeFibers=true`, aligned-normal pointer null. Sheet-normal axes and
Lasagna winding density are sampled on demand along connector probes. Distance
is the unsigned integral of density times absolute normal projection. Two-sided
orientation/distance/category consistency remains enabled, but no signed-order
certification is attempted. Input curves remain whole; no continuation links.

AFV: `data/Paris4_v1.afv`, SHA256 and native Paris4 L0 frame in README.md.
Sheet normals: PHercParis4-20260411134726-las-sd2-5b17ff6c, level4 (spacing16).
Defaults: event spacing64, tangent span16, parallel probe span64, candidate
distance128, integration step8 base voxels; angular tolerance30degrees,
distance tolerance0.25windings, maximum distance strictly below1.5windings.

All selections span full XY `[0,32693]²`, centered at Z38000. Thickness below is
in native base voxels, not normal-grid voxels. Exact intersection selects source
IDs, then **whole original fibers** are loaded. Extraction is not clipped to the
slab; there is no duplicate fiber within a run. Nonintersecting neighboring fibers
are absent, especially near the ends of selected curves outside the slab.

Build: GCC13 RelWithDebInfo, 128 workers configured, same binary throughout.
Three extraction repetitions in each native process. First repetition has a
fresh process chunk cache; filesystem cache is uncontrolled. Remaining two reuse
process and filesystem caches. Runner wall time includes all three repetitions.

## Results

| Z thickness | Fibers | Arc (million base vx) | Constraints | Query + load | First extraction | Warm mean | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 6,612 | 12.521 | 19,889 | 0.410s | 1.609s | 1.001s | 1.16GiB |
| 1,024 | 12,389 | 20.962 | 45,806 | 0.759s | 3.046s | 2.470s | 1.69GiB |
| 4,096 | 36,389 | 55.454 | 147,927 | 2.229s | 10.060s | 8.604s | 4.61GiB |

| Z thickness | Extraction mean / median | Extraction min–max | Parallel / orthogonal events | Candidates |
|---:|---:|---:|---:|---:|
| 256 | 1.204 / 1.004s | 0.999–1.609s | 12,444 / 7,445 | 248,943 |
| 1,024 | 2.662 / 2.471s | 2.470–3.046s | 25,631 / 20,175 | 612,538 |
| 4,096 | 9.090 / 8.615s | 8.593–10.060s | 76,334 / 71,593 | 2,052,212 |

Largest slab phase means: preparation0.050s, candidate search2.116s,
orientation0.549s, winding integration/certification5.943s. Total9.090s also
includes remapping/normalization/final sorting. Winding scoring dominates (~65%).
Configured128 workers does not mean all phases utilize128 cores: process CPU
time296.1s / wall29.84s averages about9.9 cores across loading and all repeats.

All counts repeat identically, all events unsigned, no invalid winding
integrations, and native regression suite137cases passed. Adapter asserts one
whole piece per input and no cuts/continuation. These are correctness/repeatability
checks, not a claim that measured constraints are anatomical ground truth.

## Whole-volume estimate

Full AFV:512,698fibers,719.132Mbase-voxel arc. Largest subset contains7.71% of
that arc. Linear arc scaling gives **112s warm / 130s initial extraction**, plus
roughly28s query/loading by fiber-count scaling: approximately **2–3minutes**.
This is a working estimate, not a measured ETA or upper bound. Smaller slab
estimates are lower (58–104s), reflecting lower candidate density/boundary loss.
Global density, spatial-index costs, memory and source-cache misses can increase
runtime. A whole-volume pass must avoid duplicate ownership if tiled; simply
repeating overlapping slab selections would repeat whole fibers and events.

## Reproduction and artifacts

Build/run commands are in README.md. Use `run_benchmark_unsigned.json` for tests
and z256; `run_benchmark_unsigned_scaling.json` for z1024/z4096, via the same
approved `run_classic_fiber_evaluation.py crop-benchmark --config ...` command.
Despite the historical runner name, the native command is exclusively
`vc_fiber_trace_chunk afv-constraints CONFIG.json`, with no trace or BP solve.

Raw summaries: `data/unsigned_z_slab_v1/z{256,1024,4096}.json`.
Logs/resources/binary and input hashes/source patches: sibling `runner/`.
Compact tracked aggregate: `benchmark_results.json`, recreated with
`python summarize_benchmark.py --output benchmark_results.json` from this directory.
Adapter source is separately hashed as an input because it is a new file.
The adapter currently saves benchmark counts/timings, not the full event graph.
