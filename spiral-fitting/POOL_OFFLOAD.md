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
