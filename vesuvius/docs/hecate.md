# Hecate training and checkpoint export

This is the training counterpart of [scrollprize/hecate](https://huggingface.co/scrollprize/hecate).
It reuses the canonical architecture in this repository; it does not copy the standalone Hub runtime.
The existing multi-teacher trainer remains available for human 2D supervision plus frozen 3D teachers.
The new resolution-distillation trainer uses a frozen Hecate EMA for **both** targets.

| Model | Input Z,Y,X | Normalization | Early XY reduction | Depth used |
|---|---|---|---|---|
| Hecate 2.4 µm | 64,256,256 | divide by 200, no clipping | 4× | central 62 planes |
| Hecate 9.6 µm | 16,64,64 | divide by 255, no clipping | none | all 16 planes |

Both use a ResNet-152 encoder, a 3D decoder, one shared ink classifier, and learned attention that combines the voxel logits along depth. The 2D loss backpropagates through the encoder, classifier and attention. There is no fixed Gaussian depth penalty. The 2.4 µm model's interpolated 3D output does not add resolved XY detail.

For 9.6 µm distillation, average-pool the fine CT and teacher probabilities by four in Z,Y,X, preserving the physical input extent. For same-resolution 2.4 µm distillation, keep the original grid and division by 200. Neither operation guarantees an identical theoretical receptive field. Zero teacher probabilities below each scroll's raw-CT background cutoff **before** reducing targets or augmenting student inputs; valid background stays a supervised negative. Exclude the teacher's artificial edge planes using fractional validity during reduction. Both losses are masked soft BCE, defaulting to weights 0.5/0.5; no binarization, min–max normalization or soft-target Dice is used.

## Data and preparation

Use the original audited manifest and its adjacent `patches.npz` to preserve the split. These follow `data.multiteacher.FlatDistillationDataset`; the source labels/renders are available in the [ScrollPrize dataset bucket](https://huggingface.co/buckets/scrollprize/datasets/tree/ink). Creating a new manifest with `python -m vesuvius.ink_detection.data.multiteacher DATA_DIR OUTPUT_DIR` creates a new split, not a reproduction of an archived run.

Supply a private JSON `DATA_CONFIG` containing:

- `manifest`: path to the audited manifest.
- `segment_depth_reversals`: an explicit boolean for every `scroll/segment`; reversal applies to the full segment before cropping.
- `background_thresholds`: measured raw uint8 CT cutoffs, keyed by scroll.
- Optional `excluded_segments`, `orientation_calibration_exclusions`, `path_roots`, and `val_patches_per_scroll` use the existing loader's semantics.
- Optional `paired_native` uses the recorded registration manifest and CT-quality scores consumed by `data.paired_native.PairedInputs`. Preserve the fixed per-segment transforms, split coordinates, and recorded appearance settings. Eligible training patches sample native CT with `native_probability`; failed gates retain fine-derived inputs. Native and fine variants share the same frozen targets and holdout. Registration estimation is not part of training.

Run from `vesuvius/` in its existing models environment, with `PYTHONPATH=src`. Paths below are caller-supplied variables; no downloads or production training happen during preparation.

```bash
python -m vesuvius.ink_detection.training.resolution_distillation prepare \
  "$TEACHER" "$RUN" --data-config "$DATA_CONFIG" --factor 4
```

Use `--factor 1` for the released 2.4 µm architecture. Add `--student-checkpoint "$INITIAL_EMA"` to initialize from a same-resolution Hub release or trusted historical EMA checkpoint with a **fresh optimizer and LR ramp**. Without it, student weights are random. Full training checkpoints contain Python objects and must come from a trusted source.

Edit `RUN/config.json` before training: world size, batch size, accumulation, LR, duration, validation/calibration budget, loss weights and W&B settings are explicit. Defaults describe a new distillation run, not the complete history that produced the published weights. Frozen teachers and data manifests are fingerprinted. Source-specific native blur/contrast augmentation changes student inputs only; teacher CT remains unaugmented.

```bash
python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  --module vesuvius.ink_detection.training.resolution_distillation smoke "$RUN/config.json"

python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  --module vesuvius.ink_detection.training.resolution_distillation train "$RUN/config.json" --production
```

Smoke mode runs two updates in a separate `smoke/` directory without W&B. Production logs losses, per-scroll validation, image previews and recipe artifacts to W&B. Checkpoints include optimizer, scheduler, per-rank RNG and calibrated EMA. BatchNorm recalibration uses **training draws only**, before EMA validation/export. Human labels are evaluation-only in this trainer. CT registration quality is not an ink-accuracy metric.

Resume with `train ... --production --resume "$CHECKPOINT"`; this restores the optimizer/LR schedule and sampling position. Sampling/RNG state is restored, but CUDA/BF16 distributed arithmetic is not guaranteed bitwise reproducible. Changes require an explicit checkpoint-bound transition. Historical 3D-weight, batch-regrouping and native-only transitions are supported. Use student initialization instead of resume when intentionally restarting the schedule.

## Export and verification

```bash
python -m vesuvius.ink_detection.training.resolution_distillation export \
  "$CHECKPOINT" "$INFERENCE_CHECKPOINT"
```

The exported file loads directly with the Hub's `hecate.load_model`. It contains EMA tensors and whitelisted architecture metadata only: no optimizer, private paths, dataset manifests or service credentials. The standalone runtime handles 2D and optional 3D inference; use the actual render sampling and an explicit orientation.

`tests/ink_detection/test_hecate_training.py` checks exact 2D/3D agreement against the downloaded Hub runtime, gradients through the 2D objective, and export/reload. Set `HECATE_TEST_ASSETS` to a directory containing `hecate.py`, `hecate_{2.4,9.6}um.pth`, and real uint8 CT fixtures `test_{2.4,9.6}.npy`. Without these external assets only the local loss/schema tests run. The validation fixtures are not bundled in Git.
