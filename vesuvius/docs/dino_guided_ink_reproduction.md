# DINO-guided 3D ink reproduction

This guide reconstructs the checkpoint lineage behind W&B runs `76ezvyks`,
`6az0505g`, `af3pga3d`, `29ulc4ly`, and `4b07qv8p`. The recipes use the current
`vesuvius.ink_detection` implementation. They intentionally contain only local
`/path/to/...` placeholders: training does not download, rename, or resolve any
data or checkpoint.

## Required local inputs

Edit every placeholder in the selected JSON before launching it. The inputs
have distinct roles:

- `datasets[].segments_path` is a directory of tifxyz segment directories.
  Each usable segment contains `x.tif`, `y.tif`, `z.tif`,
  `<segment>_inklabels.zarr`, and `<segment>_supervision_mask.zarr`. A
  `<segment>_validation_mask.zarr` is optional.
- `datasets[].volume_path` is the matching raw 3D CT Zarr. The tifxyz
  coordinates index this volume. It is also the source of v3's coordinate-
  centered extra patches.
- `dynamic_label.dino_ckpt` is the frozen Dinovol step-352500 checkpoint used
  by v1 and v2. `dynamic_label.ref_embedding` is its 864-value average ink
  reference embedding.
- `checkpoint` is the full student training state to resume. A dynamic-label
  U-Net path names a separately frozen teacher; in v1/v2 it intentionally
  points to the same checkpoint as the student initializer.
- `out_dir` must be a new or intentionally reusable run directory with enough
  space for approximately 1.7 GB per full checkpoint.

The teacher recipe uses the eight dataset roots recorded by `76ezvyks`:
PHerc. Paris 4, 0139, 0500P2, 0814, 0841, 1667, MAN5, and 0009B. All later
phases use only PHerc. Paris 4. Directory names in the example are descriptive;
the corresponding label and volume bytes are what determine reproducibility.

The Hugging Face bucket tutorial segment at
`ink/phercparis4/w00_20231016151002` has the expected asset layout, but the
audit did not establish that it is byte-identical to the April 2026 training
snapshot. W&B records the historical paths and CT URLs, not immutable hashes
for the label trees or every raw volume. Consequently these recipes make the
pipeline executable, but exact optimization-trajectory reproducibility still
requires a pinned corpus manifest with per-file hashes.

## Environment and launch

The supported environment is the one locked by current `main`. From
`villa/vesuvius`:

```bash
uv sync --extra models --extra tests
```

Before a long GPU launch, inspect `nvidia-smi`, host memory, and free disk. Do
not start a phase on devices already serving another job. A single-process
launch is:

```bash
uv run --extra models accelerate launch --num_processes 1 --module \
  vesuvius.ink_detection.training.train \
  src/vesuvius/ink_detection/configs/dino_guided_teacher.json
```

For a from-scratch chain, run `dino_guided_teacher.json`, then
`dino_guided_v1.json`, then `dino_guided_v2.json`. After v2, run
`dino_guided_v3.json` and `dino_guided_v3_fullsup.json` as independent sibling
experiments. The latter two do not consume one another. Use the same command
with the corresponding config path; increase `--num_processes` only after
checking resource availability.

All transitions are full-state continuations (`weights_only: false`): model,
EMA, optimizer, scheduler, step, and EMA optimizer-step state are restored.
The frozen teachers prefer EMA weights from their checkpoint.

Each phase keeps `max_steps: 250000`, the historical cosine-scheduler horizon,
while `num_iterations` stops that recipe at its required transition snapshot.
Changing `max_steps` to the earlier stopping point would change the learning-
rate trajectory and would not reproduce the archived state.

## Phase lineage and checkpoint names

Current training filenames use completed-iteration counts, while the archived
trainer used the zero-based `step` stored inside the checkpoint. The recipes
preserve current naming and request the exact transition iterations:

| Recipe | Input | Completed iteration(s) saved | Current filename | Stored `step` | Archived equivalent |
|---|---|---:|---|---:|---|
| teacher | none | 60,001 | `ckpt_060001.pth` | 60,000 | `teacher_unet_ckpt_060000.pth` |
| v1 | teacher 60k | 63,001 | `ckpt_063001.pth` | 63,000 | `ckpt_063000.pth` |
| v2 | v1 63k | 64,001 and 77,001 | `ckpt_064001.pth`, `ckpt_077001.pth` | 64,000, 77,000 | `ckpt_064000.pth`, `ckpt_077000.pth` |
| v3 | v2 77k; v2 64k ensemble | 79,001 | `ckpt_079001.pth` | 79,000 | `ckpt_079000.pth` |
| v3 fullsup | v2 77k; v2 64k ensemble | 78,001 | `ckpt_078001.pth` | 78,000 | `ckpt_078000.pth` / `ckpt_78k_fullsup.pth` |

Do not rename a current checkpoint to imitate an archived filename. For a
direct archived resume, replace the relevant `checkpoint` and frozen-teacher
paths with the archived file. For example, v2 accepts archived v1
`ckpt_063000.pth` as both `checkpoint` and `dynamic_label.unet_ckpt`; v3 accepts
archived v2 `ckpt_077000.pth` and `ckpt_064000.pth` as its primary and ensemble
teachers. The embedded step makes the next training iteration identical.

The recovered behavior by phase is:

- v1: `(sigmoid(frozen EMA U-Net) * minmax(DINO cosine)) > 0.5`. There is no
  raw-intensity gate.
- v2: the v1 intersection with a strict `raw > 50` foreground gate.
- both v3 siblings: eight mirror TTAs of the v2 77k primary teacher. Samples
  with strict `raw_mean > 105` and `raw_std < 30` instead average the 77k and
  64k TTA probabilities 50/50. Primary and ensemble thresholds are `0.17647`
  and `0.15686`; labels are gated by strict `raw > 50`.
- v3 fullsup alone sets `force_full_supervision: true`; ordinary v3 retains
  the stored supervision mask. Both use the same eight recorded XYZ centers,
  jitter 1024, and 25% extra-patch sampling mass.

## Archived artifact identity

Verify supplied files before a direct resume:

| Artifact | SHA-256 | Availability established by the audit |
|---|---|---|
| `teacher_unet_ckpt_060000.pth` | `e906c0b08f373e6e5f52a8d303b9221d37bac3fd6b24571e77523b824ae53edd` | private experiment storage |
| `ckpt_063000.pth` (v1) | `9ce03bad9657494583edbef64fed569f0d5dbd2235a9f418801d9b8a759c6738` | private experiment storage |
| `checkpoint_step_352500_paris4.pt` | `5fb1cf4bd831275bdea28ba522ee76641aa1e42f194a61a7fdb25b0c6670083a` | Hugging Face `scrollprize/dinovol_v2_ps8_with_paris4_352500`, revision `6a8cccbafef191a966da815e22ff5c6eae075aae` |
| `avg_ref_embedding.npy` | `61bdf93bc5e3fd956eebdbed52618985d27264b8a9e5cb043087d0f234507a81` | Hugging Face `scrollprize/ink_3d_dino_guided`, revision `73a79525466037432191284dfa237baf830c49ec` |
| `ckpt_064000.pth` (v2) | `71f78c0a4e9e3daa2b86608c0be0416d051879b27e68a73b4bfeda1775421825` | same ink repository and revision |
| `ckpt_077000.pth` (v2) | `20d9b54824bc23af7a0d4b06da2f40de9fa445cbe6f210b7304b281669137f51` | same ink repository and revision |
| `ckpt_079000.pth` (ordinary v3) | `ccab8bc727e4adc0e380c3a29ec95bbc41c7f7bedb0c88b00b598d361ead7117` | private experiment storage |
| `ckpt_78k_fullsup.pth` | `5a148c2c1bb730bfa683f2b3e3cdfc2000424003605e42300b527ff90118b303` | same ink repository and revision |

The current Hugging Face organization inventory contained no byte-identical
alias for the teacher 60k, v1 63k, or ordinary-v3 79k files. This table is
provenance information, not an implicit download mechanism.
