# Fitting on VRAM-limited GPUs: GC policy, host-RAM brick pool, export-OOM guard, memory model

Three switches and one model, all in this folder, all off or safe by default:

| What | Switch | Default | What it fixes |
|---|---|---|---|
| Per-step garbage collection of the reference cycles that hold gap-logit and flow copies | `FIT_SPIRAL_GC_POLICY=freeze` (also `every-step`, `pressure`, `off`) | `freeze` (see `GC_CYCLE_LEAK.md`) | device memory that grows with the iteration count until the card is full |
| Lasagna brick pool in page-locked host RAM, read by the card over PCIe | `FIT_SPIRAL_POOL_OFFLOAD=1` (optional `FIT_SPIRAL_POOL_OFFLOAD_MIN_GIB`) | off | the pool (5 to 21 GiB per scroll) no longer sits on the card |
| Optimiser state, gradients and cached blocks released after the final checkpoint, before the mesh export | `FIT_SPIRAL_EXPORT_FREE=0` disables it | on | `torch.OutOfMemoryError` in `save_mesh` after all iterations had run |
| Per-scroll memory model and table | `pool_offload/rb_mem_table.py`, `pool_offload/make_vram_chart.py` | n/a | predicting what a fit needs before it is launched |

## Use

```sh
FIT_SPIRAL_GC_POLICY=freeze FIT_SPIRAL_POOL_OFFLOAD=1 \
  .venv/bin/python fit_spiral.py --inputs MANIFEST --object-store STORE ...
```
The offload needs host RAM at least the size of the pool (`pool GiB` column of `pool_offload/vram_table_public.md`) and a CUDA driver that
allows `cuMemHostRegister` (the pool maps its page-locked memory into the card's address space with it). If that registration fails the fit fails at pool creation; there is no silent fallback.

## What is measured, and where (provenance, stated plainly)

* **What in the table is measured and what is modelled** (2026-11-16 audit): MEASURED are the lasagna brick pool sizes (the on-disk size of each scroll's respooled pool, `channel_0.u8` + `channel_1.u8`, 20 scrolls; an earlier version of this table used an estimate for 11 of them, which was off by up to 25x for PHerc0490B: 5.1 GiB estimated, 0.21 GiB measured), the scroll heights, and the five whole-height fits in `measured.json`. MODELLED are every per-scroll peak (the coefficients are calibrated on one scroll, PHerc0343), the track term of every scroll but PHerc0343 (cached points scaled from a 13,000-slice log; not re-measured), the shell index of the scrolls marked "not measured", and the number of iterations until a fit without the GC fix runs out of memory. Rows for scrolls without a measured pool (PHerc0306B, PHerc1447) are modelled from the PHerc0343 density and say so in `notes`.
* **Per-scroll tables and chart** (`pool_offload/vram_table_public.md`, `vram_chart_public.png`): the memory model
  `peak = pool + tracks + flow*(H+320)/1000 + 7 gap copies + act*H/1000 + 1.5 GiB headroom`, calibrated on one scroll at full height
  (20.5 / 28.8 GiB predicted vs 20.1 / 28.2 GiB measured on a V100-32, 32- / 16-voxel flow grid, about 2 %). It is a model, not a
  measurement, for every other scroll. Diamonds in the chart are nvidia-smi samples (2 s) of real whole-height fits (`pool_offload/measured.json`):
  32-voxel grid, GC, no offload: 20.1 GiB steady, 24.0 GiB max (one scroll); GC + offload: 9.3 / 13.2, 14.65 / 25.1 and 13.25 / 20.1 GiB
  steady / max (three scrolls). The 25.1 GiB max includes the mesh export, which is what the export guard below addresses.
* **Where those fits ran:** the local fitter copy of 2026-08-31 with the garbage-collection and offload code loaded by an import hook (and
  the freeze policy ported by hand into a copy of that fitter), on V100-SXM2-32GB. They are end-to-end whole-height fits. They were NOT run
  with this branch's `fit_spiral.py`, because the current upstream fitter needs the object-store input format.
* **This branch's code** is tested with `tests/test_pool_offload.py` (bit-identical gather versus the resident pool, factory selection;
  4 passed, 1 skipped, run on the hub with a V100 visible). The hand-ported freeze policy and the in-tree offload are the same logic as the hook versions.
* **Upstream test suite on this branch** (2026-11-16, unified branch, hub; 2026-11-14 on the fit-only branch: 1062 passed, 90 skipped, 2 failed):
  1,125 passed, 30 skipped, 2 failed in 590 s. The two failures (`test_service_commits.py::test_only_a_preview_of_this_session_is_committed`,
  `test_service_object_store.py::test_clients_cannot_choose_the_binding_or_host_input_paths`) fail identically on unmodified upstream `main`
  (`2d3fb38d3`, same environment), so they are not caused by this branch; whether they depend on this machine is not established.
* **This branch's own `fit_spiral.py` on an ingested upstream object store** (hub V100-16GB, torch 2.11.0+cu126; PHerc0490B, z 7000-7200, 557,248 tracks ingested with `vc_object_ingest`, 32-voxel flow grid, 401 iterations, CLI `--inputs ... --object-store ...`):
  three fits, one run each. A: `FIT_SPIRAL_GC_POLICY=freeze`, offload on, export guard on: exit 0, final checkpoint, export-guard step, 438 windings written, offload engaged. B: all three off, on a fresh copy of the same store: exit 0, same structure
  (401 iterations, same model-state keys, shapes and dtypes, same 438 windings, identical satisfaction output). E: the A configuration again on A's store: exit 0, and its link refresh was skipped by the store's link marker ("unchanged since commit 557249").
  The parameter values differ between runs (the fit is not deterministic): median relative difference over the 6 parameter tensors A vs E (same switches) 0.074, A vs B (switches differ) 0.058, E vs B 0.076, so the switches change nothing beyond the run-to-run noise at this size.
  This window's pool is 0.04 GiB, so these fits check function and compatibility, not the memory saving. Its startup with upstream's
  store module was about 95 min before iteration 0 (input read about 55 min, track-crossing link refresh about 27-62 min). Upstream's #2564 has
  since moved the track crossings out of the store read, so those two timings are no longer claimed for current `main` (see claims 11 and 12).
* **Export guard** (`FIT_SPIRAL_EXPORT_FREE`): one end-to-end observation. A 16-voxel fit of one scroll that had finished all 30,000 iterations
  failed twice in `save_mesh` (torch.cat of 1.79 GiB with 1.42 GiB, then 362 MiB, free). With the guard on, allocated memory before the export dropped
  from 21.73 to 16.78 GiB and the export wrote 500 of 500 windings (480 tifxyz folders, 3.6 GB) on the same shared card. One scroll, one run, on the
  2026-08-31 fitter with the same function patched in. Larger export spikes (up to 10.5 GiB measured earlier) are untested with the guard.
* **Speed:** the offload costs about 1 % per iteration (measured on one scroll). GC policy A/B on this branch's own `fit_spiral.py`
  (2026-11-16; PHerc0490B z 7000-7200 from an ingested object store, 557,248 tracks, 32-voxel, 1,000-step configuration, one dedicated
  hub V100-16GB; collector timed with `gc.callbacks`): after startup the Python heap holds 50.8 M objects and one full collection of it
  costs 22-30 s. Seconds per iteration (it/s): `freeze` 0.058 / 0.060 (17.2 / 16.6, two runs); a collection every step (the old
  hook) 28.9 / 27.2 (0.035 / 0.037, two runs; 75 collections of 27-30 s); every 10 steps 2.81 (0.36); `pressure` 0.058 (17.4); `off` 0.050 / 0.052
  (20.1 / 19.4, two runs). After `gc.freeze()` the 1,001 per-step collections took 0.06-0.18 s in total. Chart:
  `pool_offload/time_chart_public.png` (right panel; the other panels are production whole-height fits run with the old every-step hook
  on the older fitter copy, and the controlled pairs).

## Limits, not hidden

* A 16-voxel flow grid on the largest scroll in the table (24.3 k slices, shell 275) still does not fit a 32 GiB card even with the offload
  (model 32.5 GiB). The model also marks the fits that need the offload at 32-voxel (`GC` column).
* The host RAM that holds the pool is not free: it is pinned, and the card reads it over PCIe (about 1 % slower in the one measurement).
* The memory model's gap-copy count (7) and activation coefficients were measured on one scroll; other scrolls are predictions.

## Files

`brick_pool_offload.py`, `host_pool_cuda.py` (the pool), `sparse_cuda_cache.resident_brick_pool_class()` (the switch), `lasagna_data.py`
(uses it), `fit_spiral.py` (GC policy hook, `_free_fit_state_for_export`), `GC_CYCLE_LEAK.md`, `tests/test_pool_offload.py`,
`pool_offload/` (model, chart script, measured points, public table and chart).

## Every claim this branch makes, and its evidence

| # | claim | evidence | strength |
|---|---|---|---|
| 1 | `FIT_SPIRAL_GC_POLICY=freeze` (default) bounds the device-memory growth of a long fit | torch peak 2.81 GiB against 8.44 GiB with `off` (one scroll, z 9000-10500, 16-voxel, 600 steps, V100-32); without it a whole-height fit died at iteration 18 | one scroll, older fitter copy |
| 2 | the freeze policy makes the per-step collection cheap | branch fitter, object-store inputs: 0.06-0.18 ms per collection after the freeze against 22-30 s for a full collection of the 50.8 M-object heap; 16.6-17.2 it/s against 0.035-0.037 it/s with a collection every step and 0.36 it/s every 10 steps; `off` (no explicit collection, runs out of memory on long z ranges) 19.4-20.1 it/s. Older fitter copy: 13.71 against 13.56 it/s (freeze / off, idle card) | two runs per arm for freeze, off and every-step, one for the others; one window |
| 3 | `FIT_SPIRAL_POOL_OFFLOAD=1` (opt-in) returns exactly what the resident pool returns | 4,101,001 random and edge lookups on the real pool of the smoke fit: 0 differences; `tests/test_pool_offload.py` 4 passed | strong, on this branch's code |
| 4 | the offload takes the pool off the card | card median 20,582 against 9,522 MiB (whole-height, 32-voxel, one scroll); whole-height fits of the two largest scrolls ran on one V100-32 | one run per arm, older fitter copy |
| 5 | the offload costs about 1 % per iteration | 0.754 against 0.747 it/s (one scroll) | one scroll, older fitter copy |
| 6 | `FIT_SPIRAL_EXPORT_FREE` (default on) lets the final mesh export finish | 21.73 -> 16.78 GiB allocated before the export; 500 of 500 windings written where the same step died twice; without it the largest scroll's export peaked at 30.75 of 31.73 GiB | one observation with the guard |
| 7 | the per-scroll memory model predicts what a fit needs | calibrated on one scroll at full height within 2 %; 21 scrolls predicted, 5 measured points | a model, not a measurement, for most scrolls |
| 8 | the three switches work on upstream's current object-store input path | this branch's own `fit_spiral.py` ran to completion (exit 0) with all switches on, all off, and on again, on an ingested store; same structure, parameter differences inside the run-to-run noise | one window, one run each |
| 9 | nothing else in upstream's behaviour is changed | the upstream spiral-fitting suite on the unified branch merged with upstream main f29334cd8: see the PR description for the final counts; `vc_object_store` ctest (adapters, linking, tools, tests): all pass, `object_store_ssh` skipped (needs an ssh target) | strong |
| 10 | `vc_object_ingest --bulk` builds the same tracks much faster | 557,248 tracks in 19-46 s against 23 min per track; 11,364,609 tracks in 394 s (8 shards); identical track objects (metadata, geometry bytes, attributes; regression test `object_store_bulk_tracks`) | one window plus one whole scroll |
| 11 | (measured before upstream #2564; NOT claimed for current `main`, which no longer reads track crossings from the store) the fit's input read was faster with identical output | "Reading object-store inputs" 31 min 38 s -> 8 min 40 s on the same window, store, fitter and configuration (only `vc_object_store` swapped); `selected_inputs` 1,737 s -> 368 s, every one of 557,248 objects and 678,404 relationships identical (SHA-256 of canonical JSON), byte-identical unsorted JSON on 2,000 references | one window, full known-answer check |
| 12 | (measured before upstream #2564; NOT claimed for current `main`, whose link refresh no longer runs the track-track detector) the first link write was faster with identical links | 678,404 crossings, two concurrent pairs on the same unlinked store: submit 3,338.6 s (default 8,000 KiB page cache) -> 1,996.2 s (`VC_OBJECT_STORE_SQLITE_CACHE_MB=4096`) -> 1,586.1 s (+ edge SQL text cached per connection); the same multiset of 678,404 relationship contents and the same object versions in every pair | one window, one run per arm |

**Time:** besides the memory switches, the branch builds an ingested store in seconds to minutes (`--bulk`, claim 10). The time the switches cost: about 1 % per iteration for the offload, and for `freeze` a sub-millisecond collection per step (claim 2).

## Measured on 16 GB cards, this branch's `fit_spiral.py` (2026-11-16)
Configuration: PHerc0343 at full height (z 0-17,998, shell 250, 32-voxel flow grid), its real 10.81 GiB lasagna pool, 60 iterations followed by the mesh export. Inputs: 45,007 tracks (0.24 % of the scroll's tracks) plus a 3.65 GiB tensor held on the card in place of the device memory of the remaining tracks (the full-store track term was measured at 3.66 GiB on this scroll), so the card sees the full-store footprint. One run per row.

| card | switches | outcome |
|---|---|---|
| V100-SXM2-16GB (15.77 GiB; 1.77 GiB used by a desktop process) | `FIT_SPIRAL_GC_POLICY=freeze`, `FIT_SPIRAL_POOL_OFFLOAD=1`, `FIT_SPIRAL_EXPORT_FREE=1` | completed 60 iterations, wrote 492 of 492 windings; torch peak 12.31 GiB, nvidia-smi peak 15,072 MiB |
| V100-SXM2-16GB | all off | `torch.OutOfMemoryError` while loading the pool (10.81 GiB requested, 10.08 GiB free) |
| V100-SXM2-16GB | `freeze` only | the same error at pool load (the pool is resident) |
| V100-SXM2-16GB | offload on, GC policy off | completed; same peaks as the first row (with this heap size the GC policy made no difference) |
| RTX 4060 Ti (16,380 MiB) | `freeze`, offload, export guard | completed and wrote 490 of 490 windings; torch peak 12.25 GiB, nvidia-smi peak 14,069 MiB; 63 full collections, 2.07 s in total; the freeze was released at exit |

On a 16 GB card the offload is what allows the pool to load and the fit to run. The GC policy matters when the Python heap is large; see `GC_CYCLE_LEAK.md`.
Not covered: the full-store input read, the 16-voxel grid, and runs longer than 60 iterations.

## Runtime requirements
- A CUDA-capable NVIDIA driver and a PyTorch build that matches the card (for example the cu126 wheel for V100).
- A C compiler (`gcc`) and the Python development headers for the interpreter in use, including the architecture-specific `pyconfig.h`: Triton compiles a small C helper at the first CUDA call. On Debian and Ubuntu: `apt install gcc python3-dev` (or `libpython3.X-dev` for a non-system interpreter). If the headers live outside the default search path, set `C_INCLUDE_PATH`.
- The native `vc_object_store` module and its shared libraries; `ldd` on the built `_native*.so` should report none as `not found`. Build prerequisites are in `vc_reader/README.md`.
- 16 GB of GPU memory is enough for the configuration above with `FIT_SPIRAL_POOL_OFFLOAD=1`; without it the 10.81 GiB pool must fit on the card next to the model.

A host can be checked before a fit with a short script: import `torch`, run one CUDA kernel, and import `vc_object_store`, `vc_spiral`, `vc_reader` and `fit_spiral`. Failures at these steps name the missing component.

## Driver requirement (review of #2588)

The gather kernel is compiled at run time with the NVRTC of the environment. villa's locked NVRTC 12.8.93 emits PTX ISA 8.7 for every card from compute capability 7.0 to 8.9 (checked by compiling for sm_70, sm_80, sm_86 and sm_89), and the driver JITs that PTX when the pool loads: it needs an NVIDIA driver of the **R570 series or newer**. Older drivers run ordinary CUDA 12 programs under minor-version compatibility and still reject this PTX with `cuModuleLoadData failed with 222` (CUDA_ERROR_UNSUPPORTED_PTX_VERSION); the error now says so and names the NVRTC version. Fix: update the driver, or pin `nvidia-cuda-nvrtc-cu12` to the CUDA release of the installed driver (`nvidia-cuda-nvrtc-cu12==12.6.*` for an R560 driver). The offload is opt-in, so a fit without `FIT_SPIRAL_POOL_OFFLOAD=1` is not affected.

## Teardown (review of #2588)

`close()` unregisters the page-locked memory on the pool's own card (after that card's outstanding gathers finished) and frees the host array only after the driver released the registration. If the driver refuses (`cuMemHostUnregister` returns an error), `close()` raises, the pool keeps its memory and stays registered, and a later `close()` retries; a pool garbage-collected in that state parks its registration (memory kept alive; `brick_pool_offload.retry_parked_registrations()` retries them). A headless fit closes its store too (`FitContext.close()`), not only an interactive one.
