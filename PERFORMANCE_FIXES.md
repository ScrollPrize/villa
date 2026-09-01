# Three performance fixes for `volume-cartographer`

These three commits speed up the two tools that dominate a segmentation run —
`vc_grow_seg_from_seed` (surface tracing) and `vc_render_tifxyz` (surface
rendering) — without changing what either of them computes. Every change was
held to a byte-identical-output bar against a build of the unmodified tree.

| # | Fix | Tool(s) | Effect |
|---|-----|---------|--------|
| 1 | `omp_set_dynamic(0)` in `main()` | both | up to **4.7x**; OpenMP teams were silently collapsing to a single thread |
| 2 | Band-loop scheduling and per-band allocation | `vc_render_tifxyz` | removes an 8-thread ceiling and the serial blocks around it |
| 3 | Byte-budgeted normal-grid cache | `vc_grow_seg_from_seed` | **1.88x wall and 1.90x fewer CPU-seconds** — strictly cheaper |

The three are independent. Fix 1 is a prerequisite for fix 2 being visible at
all, but nothing in fix 2 or fix 3 depends on the others being applied.

---

## Fix 1 — OpenMP teams silently collapse to one thread

### The bug

OpenCV's `libopencv_core` carries a static initializer — `parallel.cpp`,
symbol `_GLOBAL__sub_I_parallel.cpp` — that calls `omp_set_dynamic(1)`
unconditionally, before `main()` runs.

With dyn-var enabled, libgomp does not give a parallel region the number of
threads you asked for. It calls `gomp_dynamic_max_threads()`, which:

1. takes `n = min(online CPUs, nthreads-var)`;
2. reads the **one-minute load average** via `getloadavg()`;
3. returns **exactly 1** if `loadavg >= n`, otherwise `n - loadavg`.

Two consequences, both bad for batch work:

* On a busy machine every OpenMP region in these tools runs single-threaded.
  A host running several tracing or rendering jobs at once is *always* in that
  state — that is the normal operating condition, not an edge case.
* Because the cap is `min(online CPUs, nthreads-var)`, **asking for fewer
  threads makes the collapse easier to trigger, not harder.** On a 32-core host
  at load 19, an unrestricted run still gets about 13 threads; a run that
  politely asks for 8 gets **1**.

### Why it was invisible

Every documented control sets *nthreads-var*, which dyn-var then overrides:

* `vc_grow_seg_from_seed`'s `"thread_limit"` params key → `omp_set_num_threads()`
* `OMP_NUM_THREADS`
* any per-tool thread option

`OMP_DYNAMIC=FALSE` does not help either, because libgomp parses that during
its own initialization and OpenCV's constructor runs *after* it.

Two independent investigations found the same thing from opposite directions:

* Instrumenting the OpenMP internal control variables inside a running
  `vc_grow_seg_from_seed` showed a **team size of 1** at `thread_limit`
  1, 2, 4, 8, 16 and 32 alike. 77 OS threads existed; 72 of them had zero
  accumulated user time.
* Running `vc_render_tifxyz` with `OMP_NUM_THREADS` unset, 1 and 32 gave
  19.0 / 22.1 / 18.5 s at ~100–112% CPU — three settings, one behaviour.

### The fix

`main()` runs after every static initializer, so it gets the last word. A small
shared helper in `vc_core` (`vc::core::util::disableOpenMPDynamicTeams()`) calls
`omp_set_dynamic(0)`, and both tools call it as their first statement.
`VC_OMP_DYNAMIC=1` restores the old behaviour for bisecting.

This changes how many threads execute a parallel region, never what is computed
in one.

### Measured

Isolated from every other change by `LD_PRELOAD`-ing a library that calls
`omp_set_dynamic(0)` into the **unmodified** binary, so the number is
attributable to this alone. 32-core host, load ~22, one real 12.6 cm² segment,
65 layers, warm page cache:

```
stock                        20.03 s wall   113% CPU
dynamic teams disabled        4.22 s wall  1234% CPU     4.7x
```

Reproduced from source in the clean tree below (see "Combined results").

Growth, same host, same seed, same params, `"thread_limit": 8`: **4.48x**,
peak observed 5.55x.

The gain is load-dependent *by construction*. On an idle host libgomp's dynamic
sizing already hands back nearly the full team and there is little to win. The
fix matters exactly where these tools are actually run.

---

## Fix 2 — render band loop: an 8-thread ceiling, and allocation churn

`vc_render_tifxyz`'s band path (`--tif-output` without `--zarr-output`) renders
the output in horizontal bands, 128 rows tall by default. Three problems:

1. **`schedule(dynamic, 16)` over a 128-row band is 8 work units.** However
   large the pool, the sampling loop could never occupy more than 8 threads.
   Measured on a 32-core host: 119 threads in the pool, ~3.1x scaling, and
   CPU-seconds *rising* with thread count (18.3 → 42.3). Changed to
   `schedule(dynamic, 1)` — one row per unit, 128 units per band. Rows still
   vary a lot in cost (pixels that miss the surface are nearly free) so dynamic
   still beats static, and a row of `width × nSlices` trilinear samples dwarfs
   the scheduling atomic.

2. **A 16384-slot `ChunkSampler` constructed per thread per band.** Every
   thread built one on entering the parallel region — including the threads the
   `dynamic, 16` schedule then gave no rows to. Each is ~0.6 MB of
   value-initialised slots holding `shared_ptr`s. `ChunkSampler` now supports
   default construction plus `bind()`, and the caller keeps one per worker
   thread across bands. `bind()` releases only the slots that have actually
   held a chunk — tracked in a small `touched` list — so it stays O(chunks
   visited); and it *does* release them, because a slot pins decoded chunk
   bytes through a `shared_ptr`.

3. **The whole per-band working set was reallocated every band.** `base`,
   `dirs`, the raw plane vector and the slice vector. Every band but the last
   has identical dimensions, so they are now hoisted and reused, and
   `prepareBaseAndDirs` uses `copyTo` instead of `clone` so the `create()`
   inside it becomes a no-op. On a 9040-pixel-wide render that removes about
   75 MB of malloc/free per band. Peak RSS is unchanged (12.34 → 12.35 GB on
   the largest segment measured).

Three regions that were serial per band, and are per-pixel or per-slice
independent, are now parallel:

* **`writeTifBand`** — each output slice has its own `TiffWriter`, `TIFF*`
  handle and tile buffer, so LZW encoding of the slices is independent. This
  was the largest serial block in the band loop. `writeTile()` throws on
  libtiff failure and an exception must not leave an OpenMP region, so the
  first one is captured and rethrown after the join: a write error still fails
  the render loudly.
* **`normalizeNormals`** — per-pixel, in place.
* **the prefetch bounding-box scan** — min/max over floats is associative,
  commutative and rounds nothing, so the reduction is bit-identical to the
  serial scan.

A `--threads` option (and `VC_RENDER_NUM_THREADS`) was added, since with
dynamic teams disabled the value now actually takes effect. `0` keeps
libgomp's default of one thread per hardware thread.

### Where each part of this fix shows up

Worth stating precisely, because the two halves have different regimes:

* The `dynamic, 16` → `dynamic, 1` change only matters **above 8 threads**.
  At exactly 8 threads the old schedule already produced 8 units and filled
  the pool, so this half of the fix is a no-op there.
* Parallelising `writeTifBand` / `normalizeNormals` / the bbox scan, and
  removing the per-band allocations, help at **any** thread count.

Measured on a small segment at 8 threads the second group alone gives 1.45x;
the first group is what unlocks scaling past 8 threads.

---

## Fix 3 — the normal-grid cache was budgeted in the wrong unit

`NormalGridVolume` caches `GridStore` objects — one per normal-grid slice,
each an mmapped `.grid` file plus its decoded polylines. The cap was
`max_cache_size = 512` **entries**.

An entry count is the wrong unit. A cached store costs the `.grid` file it maps
— whose size varies by an order of magnitude between slices — plus up to 2 MiB
of decoded seglists. Any entry cap is therefore simultaneously too loose (a
worst case of *entries* × 2 MiB) and too tight (it evicts small stores that
cost almost nothing to keep).

And 512 was far below a real working set. Measured with `strace` on one resume
round of a large patch, whose grid working set was **7,424 distinct slices
totalling 4.00 GB**: the tracer performed **110,631 `.grid` opens** — it
reopened, remapped and re-parsed every file about **15 times over**, discarding
each store's decoded polylines with it.

The fix replaces the entry count with a global **byte** budget, tunable via
`VC_GRID_CACHE_BYTES`, default 4 GiB, with an entry backstop
(`VC_GRID_CACHE_ENTRIES`, default 65536) so a volume of unusually tiny slices
cannot grow the map without bound. `GridStore` gained a `residentBytes()`
accessor for the accounting. The eviction path also stopped constructing a
fresh `std::mt19937` from `std::random_device` on *every single eviction* while
holding the exclusive cache lock.

Budget sweep on that same round:

| budget | GridStore constructions | resident entries |
|--------|------------------------:|-----------------:|
| 256 MiB | 112,447 | 659 |
| 1 GiB | 50,426 | 2,162 |
| 2 GiB | 32,423 | 3,730 |
| **4 GiB** | **7,427** | **7,427** |
| 8 GiB | 7,427 | 7,427 |

4 GiB is the knee: it holds the whole working set with zero evictions — one
construction per distinct slice — and 8 GiB buys nothing. **All six budgets
produced byte-identical output**, which is the point: the cache is transparent,
so an evicted store is simply reloaded from the same immutable file.

The accounting charges each store its whole mapped file size while only touched
pages are actually resident, so real RSS stays well under the budget (1.8 GB at
the 4 GiB default in that run). It over-charges rather than under-charges,
which is the safe direction. The mappings are file-backed, so concurrent
tracers working the same grids share those pages through the page cache.

This one is **not** a CPU-for-wall trade. It is strictly cheaper: the win is
almost entirely *system* time, ~104,000 fewer `mmap`/`munmap`/`stat` round
trips per round.

---

## Combined results, reproduced from source in a clean clone

Everything below was re-measured from a fresh clone of upstream `main`, built
four times — unpatched, then with each commit added in turn — so every figure
is attributable to one change. 32-core host, `/usr/bin/time`, warm page cache
after a discarded warm-up pass, load average given per arm. All arms verified
byte-identical (md5 of the complete output).

### Rendering, small segment (1.67 cm², 65 layers), `OMP_NUM_THREADS=8`

| tree | wall | %CPU | CPU-s | load |
|------|-----:|-----:|------:|-----:|
| unpatched | 15.20 s | 102% | 15.5 | 18.2 |
| + fix 1 | 3.39 s | 483% | 16.4 | 18.2 |
| + fix 2 | **2.34 s** | 763% | 17.8 | 16.1 |

**6.50x wall for +15% CPU-seconds.** The unpatched arm's 102% CPU is the bug
itself: asking for 8 threads on a host at load 18 got exactly one.

### Rendering, medium segment (12.63 cm², 6860×5080, 65 layers)

At `OMP_NUM_THREADS=8`, fix 1 alone (interleaved, two passes):

| tree | wall | %CPU | CPU-s | load |
|------|-----:|-----:|------:|-----:|
| unpatched | 143.78 s / 138.99 s | 141% / 146% | 202.8 / 203.4 | 23.9 / 31.9 |
| + fix 1 | 37.69 s | 573% | 216.0 | 22.3 |

**3.81x wall for +6.5% CPU-seconds.**

At `OMP_NUM_THREADS=32`, fix 2 on top of fix 1 (two interleaved passes):

| tree | wall | %CPU | CPU-s | load |
|------|-----:|-----:|------:|-----:|
| + fix 1 | 43.11 s / 39.28 s | 611% / 663% | 263.8 / 260.8 | 14.7 / 20.9 |
| + fix 2 | **16.37 s / 22.38 s** | 2260% / 1616% | 370.1 / 362.0 | 18.6 / 18.7 |

**2.0–2.4x wall.** Note the %CPU of the fix-1-only arm: ~650% on a 32-thread
pool is precisely the 8-work-unit ceiling described above, and it does not
move until fix 2 removes it. CPU-seconds rise 40%, which is the cost of
actually using the extra cores on an already-loaded box — the reason the
`--threads` cap exists.

At 8 threads on this segment fix 2 is within noise, exactly as its mechanism
predicts: the old schedule already produced 8 work units, so only the
serial-block parallelisation is in play.

### Growth, one resume round of a real segment, `thread_limit 1`

| tree | CPU-s | user | sys | wall | maxRSS | load |
|------|------:|-----:|----:|-----:|-------:|-----:|
| unpatched | 64.4 | 38.53 | 25.89 | 65.37 s | 196 MB | 31.9 |
| + fix 1 | 70.8 | 40.50 | 30.31 | 74.69 s | 189 MB | 16.4 |
| + fix 3 | **44.2** | 33.50 | **10.69** | **51.94 s** | 1563 MB | 20.6 |

**1.60x fewer CPU-seconds and 1.44x less wall** from fix 3, almost entirely
in system time (2.84x less). Fix 1 is neutral at `thread_limit 1` by
construction — there is no team to un-collapse — and the two arms differ only
by load noise. With the grid files served over a network rather than local
disk the same comparison measured 1.88x wall and 1.90x fewer CPU-seconds.

All three growth arms produced `md5(x,y,z,generations.tif)` identical and
`area_cm2 = 10.66587924071438` to every digit.

---

## Measurement methodology

* **CPU-seconds are the primary figure.** On the benchmark host they repeat to
  within 3% across runs; wall-clock does not, because the machine is shared.
  Wall-clock speedups are reported alongside, with the load average of every
  arm stated, and arms interleaved so load drift affects both sides.
* **Load average is stated for every arm.** For fix 1 this is not incidental —
  the size of the bug is a function of load, so a number without a load figure
  is meaningless.
* **Cold vs warm cache is controlled.** A warm page cache is worth about +55%
  on a re-run of identical inputs, so each benchmark begins with a discarded
  warm-up pass and all measured arms are warm. This is stated per table.
* **Exit codes are checked before any ratio is computed.** A killed process
  still produces a timing number.
* **Growth is nondeterministic by default.** `VC_GROWPATCH_RNG_SEED` is unset
  in normal operation and the tracer seeds from `std::random_device`. Every
  comparison here pins it.
* Each fix was built and benchmarked **in isolation**, against a build of the
  immediately preceding tree, so each commit message carries a number that is
  defensible for that change alone.

---

## Correctness evidence

A build of the **unmodified** clone was made first and verified to reproduce
the previously deployed binary exactly. That is the reference every patched
build is held to.

* **Reference established before patching.** The clean-clone build's render of
  a real segment is md5-identical to the deployed production binary's, and its
  growth output at `thread_limit 1` is md5-identical in `x/y/z/generations.tif`
  with the same `area_cm2`.
* **Render: byte-identical.** 65/65 layers identical on real segments, at
  `--threads` 1, 8 and 32; identical on eight synthetic segments including a
  sentinel-torture case and a 3×3 grid with 8 valid points; 168/168 files
  identical on the `--zarr-output` path.
* **Growth: byte-identical at `thread_limit 1`,** in both seed and resume mode,
  with `area_cm2` equal to 16 significant digits, and identical across every
  cache budget from 256 MiB to 8 GiB.

### One honest caveat: thread count changes growth geometry

Growth output depends on the thread count. This is **pre-existing and unrelated
to these changes**: the tracer's `random_perturbation()` draws from a
`thread_local` RNG, so which thread claims a point decides that point's draw.
Consequences:

* The **unpatched** build diverges identically once it is given real teams.
* `thread_limit 4` is not even self-reproducible at a pinned seed, for the same
  reason.
* `generations.tif` is identical in every case — the same points, in the same
  generations. Only solved coordinates move. `thread_limit 1` vs 8 leaves
  83.6% of points bit-identical, mean displacement 1.42 voxels.

Fix 1 *exposes* this, because before it the teams silently collapsed to one
thread. It does not cause it. This is why **`thread_limit 1` is the recommended
adoption for growth**, where fix 3 is a pure cache win with byte-identical
output, and why fix 1's growth benefit should be taken with the understanding
that a multi-threaded grow was already non-reproducible.

---

## Portability

The patches are ordinary, toolchain-agnostic C++ source changes. Nothing in
them depends on a compiler version, a libc version or a platform.

Packaging is worth a note, though, because it bit us: a binary built with a
current system GCC on Ubuntu 26.04 requires **glibc 2.43** and will not even
load on Ubuntu/Pop!_OS 20.04 (glibc 2.31) — `GLIBC_2.38 not found`. Building
the same source against an older sysroot solves it completely:

* built with the conda `x86_64-conda-linux-gnu` toolchain against its
  **glibc 2.17 sysroot**, with `-DCMAKE_SYSROOT=<prefix>/x86_64-conda-linux-gnu/sysroot`
  and `-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'`, the patched binary's maximum
  glibc requirement drops to **2.14**;
* that single artifact was verified to launch on all three test machines —
  Ubuntu 26.04 / glibc 2.43, and two Pop!_OS 20.04 / glibc 2.31 hosts — with no
  container.

This is packaging guidance, not part of the patches.

---

## What is deliberately *not* here

Several other avenues were measured and did not earn a place. Recording them so
nobody spends the time again:

* **GPU render prototype.** The sampling kernel is genuinely 16.5–24.6 G
  samples/s against 19.4 M/s for a whole 32-core box, and end-to-end 8.3–14.2x
  in-process. But rendering turned out to be an I/O job wearing a compute job's
  clothes: on an idle dual-GPU box with the volume over 1 GbE, a render took
  470.98 s of which **458.52 s (97.4%) was disk read**, while the sampling
  kernel itself took 0.09 s. The right move is to integrate the kernel inside
  the existing band loop, not to ship a separate renderer — and the scheduling
  lesson ("run the render where the volume is on local disk", not "on whichever
  box is idle") matters more than the kernel.
* **A morphological dilate in the growth inner loop.** Amdahl ceiling of
  1.00002x, confirmed by three independent instruments.
* **Enabling the `SURFACE_SDT` residual.** Changing its weight by 1000x moved
  the resulting area by 0.06%.
* **Passing the surface prediction as the growth volume instead of raw CT.**
  −1.9% area, no other benefit.
* **Coarse-to-fine growth.** −26% throughput, −38% area, and wrong-winding
  failures.
