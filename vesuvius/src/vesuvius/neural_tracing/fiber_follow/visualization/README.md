# Model interpretation atlas

`interpret` captures an actual decision and writes seven paired-view pages, an
HTML index, PNGs/previews, individual and combined PDFs, raw activation arrays,
intervention metrics, provenance, and a validation report. It supports only the
current `axial_patch4_overlap_tokens_fiber_slabs_v11` model (EMA weights).

Run from `fiber_follow` using the existing project environment:

```bash
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/fiber-atlas-mpl NUMBA_CACHE_DIR=/tmp/fiber-atlas-numba
/home/sean/Documents/villa4/vesuvius/.venv/bin/python \
  -m vesuvius.neural_tracing.fiber_follow.visualization.interpret \
  --checkpoint output/patch4_run3/ckpt_006000.pt \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --replay output/axial_patch4_tokens_memory_v9_run1/dagger/decisions_002000_mmap_v5 \
  --row 478 --fiber-index 377 \
  --compare-atlas output/model_interpretation_v9_fiber377 \
  --output-dir output/model_interpretation_patch4_run3_step6000_fiber377
```

Volume and fiber paths default to the checkpoint's values. This is the run used
for the new atlas (those three path arguments were omitted because they match
the checkpoint). Choose a fresh output directory on reruns. Use numbered
checkpoints while training is active; the checkpoint hash is checked again at
the end. Inference is CPU FP32 with four threads by default, seed zero, and no
augmentation or compilation. It does not allocate GPU memory.

To select a new clean annotated decision, replace `--replay`, `--row`, and
`--compare-atlas` with `--fiber-name NAME.json --arc-position 100`, or use
`--fiber-index N --arc-position 100`. Distances are trace-grid voxels measured
from the traversal start; `--reverse` reverses the traversal. This mode explicitly
uses the annotation prefix as synthetic observed history. Replay mode uses only
the saved observations as inputs; annotations are plotted as references.

Indices belong to `--fiber-split train` by default. `--fiber-split val` and `all`
are available, with `--val-z LOW HIGH` in base voxels (default: checkpoint band).
Replay fiber manifests must match the entire selected split. A fiber selector
with replay is a consistency check. `--ct-level`, `--fiber-level`,
`--ct-grid-scale`, and `--grid-scale` override the checkpoint's volume geometry.
Other controls include `--confidence-threshold`, `--n-commit`, and `--threads`.
Use `--help` for all arguments.

The original replay is v5, which stores resampled history rather than a complete
committed polyline. The read-only compatibility adapter accepts it only when the
saved history reaches within 1.5 history steps of the original seed and covers
its saved age. It joins that small residual gap to the saved seed and records the
approximation in provenance. It refuses truncated remote histories. Current v6
replay uses its saved complete prefix. This data adapter does not support the old
model architecture. `--compare-atlas` checks fiber identity, position, frame,
history and mask exactly, and reports per-channel input differences after
resampling with the current preprocessing code.

The pages cover the actual 6-cube stride-4 patch convolution, axial encoder
updates, live CT/path slabs, separate history reads, controlled history
interventions, trajectory proposals, and causal survival scoring. History reads
have independent softmax distributions; their mass must not be compared with
image attention mass. Disabling a scorer read can change retry/selection as well
as confidence. Fixed-baseline-curve scores separate scoring changes from path
changes. These single-decision interventions are sensitivity measurements, not
accuracy evaluations.

Attention diagnostics retain FP32 Q/K logits but compute softmax and head means
in FP64 before storing FP32 weights. This avoids accumulated normalization error
over the roughly 20,000 image keys without changing model inference or relaxing
the attention validation tolerance.

Rendering and validation can be repeated without accessing volumes:

```bash
python -m vesuvius.neural_tracing.fiber_follow.visualization.render OUTPUT_DIR
python -m vesuvius.neural_tracing.fiber_follow.visualization.validate OUTPUT_DIR
```

Validation requires the original checkpoint. It checks baseline equality,
policy counts, survival, attention masks and normalization, marginal probability
mass, history masks, the full overlapping convolution, checkpoint hash and
readable local assets. The extraction also checks every instrumented output
against an uninstrumented inference and reruns the restored intervention model.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python -m pytest \
  tests/test_visualization.py -q -o cache_dir=/tmp/fiber-atlas-pytest
```

No model, sampler or training code is modified by the visualization pipeline.
