# Cross-scroll ink detection on native scans: scan-type transfer, self-training, and mirror TTA

The flat 9 um ink recipe (`configs/aligned21_hybrid_3d2d.json`) reads the scrolls it was trained on almost perfectly, but
the scrolls that are still unread are scanned natively at 8.6-9.4 um, while most labelled training data are renders pooled
from 2.4 um scans: PHerc 1667 and PHerc Paris 4 have no 9.362 um scan at all. This page adds three things that measurably
improve reading an **unseen** scroll from its **native** scan, and the evaluation that shows it.

## Evaluation: leave one scroll out

Train on PHerc 1667 + Paris 4 + 0814 (recipe batch shares, 25k steps), score per-pixel ROC-AUC on PHerc 0139, which the
model never saw: its 5 native 9.362 um segments (w035 w039 w040 w041 w044, open-data labels inside their supervision masks)
and its 9 pooled renders. A model trained *with* 0139 scores 0.99+ on the same pixels, so the gap below is the scroll gap.
All numbers are forward direction (the reverse direction is ~0.5 throughout).

| model (seed 42 unless noted) | native, plain | native, mirror TTA | native, mirror + 3 depth windows | pooled, 9 segments |
|---|---|---|---|---|
| baseline recipe | 0.743 | 0.779 | 0.790 | 0.821 |
| baseline, 3 seeds (42 / 43 / 44) | 0.743 / 0.783 / 0.763 | 0.779 / 0.800 / 0.792 | | 0.821 / 0.822 / 0.831 |
| + scan transfer (`scan_transfer`) | 0.785 | 0.797 | | 0.826 |
| + self-training, round 1 (`background="mean"`) | 0.786 | 0.796 | 0.803 | 0.844 |
| + self-training, safer background (`background="max"`) | **0.804** | **0.813** | **0.818** | 0.835 |
| 4-model average (3 seeds + a 78k-step run), the teacher | 0.799 | 0.808 | | 0.850 |
| + self-training on 3 segments only (split test, see below) | **0.812** | **0.820** | | 0.838 |

Stroke-centreline recall (share of the true strokes' skeleton recovered at 75% pixel precision, a readability-oriented
companion to AUC): baseline 0.447, safer-background self-training 0.590 (mirror + depth windows).

## 1. Mirror TTA (no code change)

`vesuvius.ink_detection.inference.infer --tta-mirror` lifts every model on the unseen scroll (+0.017 to +0.036 native,
single-model seed mean 0.763 -> 0.790) at ~4x inference cost. Averaging three depth windows shifted by -2/0/+2 planes
(`--layer-start/--layer-end`, within the recipe's training jitter) adds +0.005 to +0.011. Use mirror TTA by default.

## 2. Scan-type transfer augmentation (`training/scan_transfer.py`)

A registered pooled/native pair of the same surface (PHerc 0814 segment 46527) shows the native render is, to first
order, the pooled one blurred over ~3 depth planes and slightly in-plane, plus smooth low-frequency intensity differences
(`scan_transfer_fit.py`: held-out R^2 0.740 vs 0.623 for resampling alone; three PHerc 0846A pairs give the same
picture). The augmentation applies that kernel plus a matching smooth random field to a share of the training samples:

```json
"scan_transfer": {"enabled": true, "kernel": "builtin:pooled_to_native_pherc0814", "probability": 0.5,
                  "noise_amplitude": 0.5, "noise_sigma_xy": 4.0, "noise_sigma_z": 1.5}
```

Leave-one-scroll-out: native 0.743 -> 0.785 for the same seed, higher on all 5 segments (+0.024 to +0.059), with no loss
on pooled renders. It needs no data from the target scroll. With mirror TTA the same-seed gain is +0.019 (0.779 -> 0.797);
against the three-seed mean it is smaller (0.790 -> 0.797), so treat it as a consistent but modest gain.

## 3. Self-training on the target scroll (`pseudo_labels.py`)

Teachers (e.g. the four leave-one-out models with mirror TTA) predict the target segments; only confident pixels become
labels (ink if the mean >= 0.50; background if every teacher <= 0.30 and the pixel is > 8 px from any weak candidate with
mean >= 0.40); everything else is ignored by the loss. The labelled segments are added as one dataset group with 16 of 64
batch slots. A second round with the same rule adds nothing (0.785 vs 0.786); the safer background rule is what helped:
it halves the true ink wrongly taught as background (22% -> 10% of true ink) at equal ink precision (0.87).

```bash
python -m vesuvius.ink_detection.pseudo_labels OUT_ROOT \
  --segment w035p /data/native/w035/surface-volume.zarr t1/w035.tif t2/w035.tif t3/w035.tif t4/w035.tif
```

**First Letters rule.** Ink outputs of a submitted region must not overlap any training data. Pseudo-label some regions of
the target scroll and read others. In the table above all 5 native segments were adapted (transductive); on the 4 pooled
segments that were never pseudo-labelled the round-1 student matches its teacher (0.850 vs 0.846) while the safer-background
student is lower (0.830), so check transfer to unadapted regions before relying on it.

**Split test (the First Letters setting).** Self-training on native w035, w040 and w044 only (safer-background labels from
the same four teachers), then reading w039 and w041, which never contributed a label:

| native AUC, mirror TTA | w039 (unadapted) | w041 (unadapted) | mean of the two |
|---|---|---|---|
| baseline seed 42 | 0.735 | 0.803 | 0.769 |
| baseline seeds 42 / 43 / 44, mean | 0.757 | 0.815 | 0.786 |
| 4-model teacher | 0.778 | 0.828 | 0.803 |
| **split self-trained (seed 42)** | **0.770** | **0.841** | **0.806** |

Adapting on some regions of a scroll improves reading of its other regions (+0.037 over the same seed, +0.020 over the
seed mean), so one model matches the four-model ensemble there. Stroke-centreline recall over all five segments: 0.580
(teacher 0.544, baseline 0.416). One seed, one held-out scroll.

## Reproduce

- Configs: the recipe restricted to the training scrolls (`sampling_scroll` in 1667 / Paris4 / 0814), `seed` 42, 25k
  steps; for self-training add the pseudo-labelled group (`sampling_scroll` "0139p", physical keys per segment) with
  16 slots; for scan transfer add the `scan_transfer` block above.
- Native 0139 labels: open-data S3 `PHerc0139/segments/*/` (identical to the ink_9um w035 labels pixel for pixel).
- Scoring: ROC-AUC of the forward prediction inside the supervision mask, per segment, mean over segments.
- Hardware: one RTX 4070 (12 GB), ~2.5 h per 25k-step run at batch 64.
- Checkpoints, configs, the kernel and a PHerc 0846A 2.403 um -> 9.362 um transform:
  https://huggingface.co/TequilaYuen/vesuvius-ink-9um-cross-scroll (CC BY-NC 4.0, as the training data).

## Data and credit

Vesuvius Challenge open data (CC BY-NC 4.0). Cite: G. Angelotti, S. Parsons, S. Johnson, E. R. Dal Pra, J. Rudolph,
P. Tafforeau, A. Mirone, P. Henderson, H. Schilling, F. McDonald, D. Josey, Y. Nader, C. S. Parker, W. B. Seales,
*Vesuvius Challenge - CT Scans of Herculaneum Papyri*; and for Scrolls 1-4 / Fragments 1-6 scanned at DLS before 2025,
Parsons et al. (2023), *EduceLab-Scrolls*, arXiv:2304.02084. The flat 9 um recipe, its labels and the inference code are
the Vesuvius Challenge team's; this page adds the scan transfer, the self-training procedure and their evaluation.
