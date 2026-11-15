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
* **Upstream test suite on this branch** (2026-11-14, hub, CPU only, `tests/` without `test_interactive_e2e.py` and without
  `test_mps_track_sampling.py`, which cannot be collected on upstream `main` either: it imports `load_tracks_from_dbm`, which `tracks.py` lacks):
  1062 passed, 90 skipped, 2 failed in 126 s. The two failures (`test_service_commits.py::test_only_a_preview_of_this_session_is_committed`,
  `test_service_object_store.py::test_clients_cannot_choose_the_binding_or_host_input_paths`) fail identically on unmodified upstream `main`
  (`2d3fb38d3`, same environment), so they are not caused by this branch; whether they depend on this machine is not established.
* **This branch's own `fit_spiral.py` on an ingested upstream object store** (hub V100-16GB, torch 2.11.0+cu126; PHerc0490B, z 7000-7200, 557,248 tracks ingested with `vc_object_ingest`, 32-voxel flow grid, 401 iterations, CLI `--inputs ... --object-store ...`):
  three fits, one run each. A: `FIT_SPIRAL_GC_POLICY=freeze`, offload on, export guard on: exit 0, final checkpoint, export-guard step, 438 windings written, offload engaged. B: all three off, on a fresh copy of the same store: exit 0, same structure
  (401 iterations, same model-state keys, shapes and dtypes, same 438 windings, identical satisfaction output). E: the A configuration again on A's store: exit 0, and its link refresh was skipped by the store's link marker ("unchanged since commit 557249").
  The parameter values differ between runs (the fit is not deterministic): median relative difference over the 6 parameter tensors A vs E (same switches) 0.074, A vs B (switches differ) 0.058, E vs B 0.076, so the switches change nothing beyond the run-to-run noise at this size.
  What this does not show: any memory saving (this window's pool is 0.04 GiB), any speed difference, or anything at whole-height scale. The fit's startup on an ingested store is slow (about 95 min for this window before iteration 0, of which about 62 min is the track-crossing link refresh; a second fit skips the refresh but still needs about 55 min of input reading); that is upstream's object-store input path, not this branch.
* **Export guard** (`FIT_SPIRAL_EXPORT_FREE`): one end-to-end observation. A 16-voxel fit of one scroll that had finished all 30,000 iterations
  failed twice in `save_mesh` (torch.cat of 1.79 GiB with 1.42 GiB, then 362 MiB, free). With the guard on, allocated memory before the export dropped
  from 21.73 to 16.78 GiB and the export wrote 500 of 500 windings (480 tifxyz folders, 3.6 GB) on the same shared card. One scroll, one run, on the
  2026-08-31 fitter with the same function patched in. Larger export spikes (up to 10.5 GiB measured earlier) are untested with the guard.
* **Speed:** the offload costs about 1 % per iteration (measured on one scroll); the GC policies' per-iteration cost and the freeze policy's
  effect are in `GC_CYCLE_LEAK.md`. A controlled freeze A/B on a dedicated card is not done yet and no time chart is shipped for that reason.

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
| 2 | the freeze policy costs nothing measurable | 13.71 against 13.56 it/s on an idle card; a collection after the freeze takes 0.2-0.3 ms against 0.4-2 s for a full collection every step | one scroll, older fitter copy |
| 3 | `FIT_SPIRAL_POOL_OFFLOAD=1` (opt-in) returns exactly what the resident pool returns | 4,101,001 random and edge lookups on the real pool of the smoke fit: 0 differences; `tests/test_pool_offload.py` 4 passed | strong, on this branch's code |
| 4 | the offload takes the pool off the card | card median 20,582 against 9,522 MiB (whole-height, 32-voxel, one scroll); whole-height fits of the two largest scrolls ran on one V100-32 | one run per arm, older fitter copy |
| 5 | the offload costs about 1 % per iteration | 0.754 against 0.747 it/s (one scroll) | one scroll, older fitter copy |
| 6 | `FIT_SPIRAL_EXPORT_FREE` (default on) lets the final mesh export finish | 21.73 -> 16.78 GiB allocated before the export; 500 of 500 windings written where the same step died twice; without it the largest scroll's export peaked at 30.75 of 31.73 GiB | one observation with the guard |
| 7 | the per-scroll memory model predicts what a fit needs | calibrated on one scroll at full height within 2 %; 21 scrolls predicted, 5 measured points | a model, not a measurement, for most scrolls |
| 8 | the three switches work on upstream's current object-store input path | this branch's own `fit_spiral.py` ran to completion (exit 0) with all switches on, all off, and on again, on an ingested store; same structure, parameter differences inside the run-to-run noise | one window, one run each |
| 9 | nothing else in upstream's behaviour is changed | the upstream spiral-fitting suite: 1062 passed, 2 failed identically on unmodified main | strong |

**What this branch does NOT do:** it does not speed up upstream. In particular it does nothing about upstream's slow start of a fit on an ingested store (reading the inputs and the track-crossing link refresh took about 95 minutes for a 557 k-track window; a second fit on the same store skips the refresh but still read inputs for about 55 minutes). The only time costs it adds are the about 1 % of the offload and the 0.2-0.3 ms per step of the freeze policy. A separate change to `vc_object_ingest` (a bulk mode) speeds up building the store (557 k tracks in 19-46 s instead of 23 min), not the fit's own startup.
