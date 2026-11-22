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
| + scan transfer, 3 seeds (42 / 43 / 44) | 0.785 / 0.789 / 0.786 | 0.797 / 0.800 / 0.806 | | 0.826 / 0.825 / 0.832 |
| + self-training, round 1 (`background="mean"`), transductive | 0.786 | 0.796 | 0.803 | 0.844 |
| + self-training, safer background (`background="max"`), transductive | 0.804 | 0.813 | 0.818 | 0.835 |
| 4-model average (3 seeds + a 78k-step run), the teacher | 0.799 | 0.808 | | 0.850 |
| + self-training on 3 segments only (3 of the 5 scored segments adapted) | 0.812 | 0.820 | | 0.838 |
| **+ self-training, unadapted segments only, 3 seeds (42 / 43 / 44; split test, section 3)** | **0.796 / 0.797 / 0.803** | **0.808 / 0.809 / 0.809** | | |

Rows marked transductive were scored on the same native segments they were self-trained on; the last row scores every
segment with a model that never got a label, true or pseudo, on it (section 3).

Stroke-centreline recall (share of the true strokes' skeleton recovered at 75% pixel precision, a readability-oriented
companion to AUC; each segment's threshold is chosen on the other four segments and applied blind, realised pixel
precision 0.75-0.76), mirror TTA, seed 42: baseline 0.446, scan transfer 0.563, safer-background self-training 0.628
(three-seed means: baseline 0.532, scan transfer 0.586).

## 1. Mirror TTA (no code change)

`vesuvius.ink_detection.inference.infer --tta-mirror` lifts every model on the unseen scroll (+0.017 to +0.036 native,
single-model seed mean 0.763 -> 0.790) at ~4x inference cost. Averaging three depth windows shifted by -2/0/+2 planes
(`--layer-start/--layer-end`, within the recipe's training jitter) adds +0.005 to +0.011. Use mirror TTA by default.

## 2. Scan-type transfer augmentation (`training/scan_transfer.py`)

A registered pooled/native pair of the same surface (PHerc 0814 segment 46527) shows the native render is, to first
order, the pooled one blurred over ~3 depth planes and slightly in-plane, plus smooth low-frequency intensity differences
(R^2 0.707 vs 0.580 for resampling alone on held-out 256 px spatial blocks with a guard band, a check outside
`scan_transfer_fit.py`; 0.711 vs 0.590 on randomly held-out pixels, the split `scan_transfer_fit.py` uses, so
neighbouring-pixel leakage is negligible; three PHerc 0846A pairs give the same picture). The augmentation applies
that kernel plus a matching smooth random field to a share of the training samples:

```json
"scan_transfer": {"enabled": true, "kernel": "builtin:pooled_to_native_pherc0814", "probability": 0.5,
                  "noise_amplitude": 0.5, "noise_sigma_xy": 4.0, "noise_sigma_z": 1.5, "skip_datasets": ["native9"]}
```

`skip_datasets` lists substrings of a dataset group's `segments_path`; samples from matching groups (scans that are
already native, such as the ink_9um `native9-*` labels) are never converted. An earlier revision of this pull request
converted every training sample, native ones included. The leave-one-scroll-out runs below are unaffected (all their
training groups are pooled renders); the production models were retrained with the fix (see "Other unseen scrolls").

Leave-one-scroll-out, three seeds, each run paired with the baseline of the same seed: plain native AUC 0.763 -> 0.787
(+0.024, SE 0.003 across segments; higher on all 5 segments and in 13 of 15 segment-seed pairs; per seed +0.042 / +0.005
/ +0.023), with no loss on pooled renders (0.824 -> 0.828). With mirror TTA the gain shrinks to +0.011 (SE 0.005;
0.790 -> 0.801; per seed +0.019 / +0.000 / +0.014; 9 of 15 pairs), and the three-seed averages differ by +0.008
(0.803 -> 0.811). Blind stroke-centreline recall, mirror TTA, seed mean: 0.532 -> 0.586. It needs no data from the
target scroll. Seed 42, used in the first rows of the table, is the most favourable pairing: treat the transfer as a
consistent gain under plain inference and a small one once mirror TTA is used.

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

**Split test (the First Letters setting).** Self-train on some native segments, read the others. Split A adapts on
w035, w040 and w044 and reads w039 and w041; split B adapts on w039 and w041 and reads w035, w040 and w044 (the same
safer-background labels from the same four teachers, 16 of 64 batch slots). Together they score every native segment
with a model that never got a label, true or pseudo, on it. Three seeds, each paired with the baseline of the same seed:

| native AUC, unadapted segments only | seed 42 | seed 43 | seed 44 | paired gain (SE across segments) |
|---|---|---|---|---|
| baseline, mirror TTA | 0.779 | 0.800 | 0.792 | |
| self-trained, mirror TTA | 0.808 | 0.809 | 0.809 | +0.018 (0.006); 13 of 15 segment-seed pairs |
| baseline, plain | 0.743 | 0.783 | 0.763 | |
| self-trained, plain | 0.796 | 0.797 | 0.803 | +0.035 (0.006); 13 of 15 |

Adapting on some regions of a scroll improves reading of its other regions: one self-trained model matches the
four-model teacher there (0.808 with mirror TTA), and the three self-trained models averaged reach 0.815 (the three
baselines averaged: 0.803). Seed 42 is the most favourable seed (+0.029; seeds 43 and 44: +0.009 and +0.017).
Stroke-centreline recall, blind thresholds as above, seed means: 0.626 against 0.532 for the baselines (teacher 0.588).
The reverse direction (the other face) stays at 0.51-0.55 for every model. On the four pooled 0139 renders that no run
pseudo-labelled (w016 w017 w029 w043), the six self-trained runs score 0.833-0.851 against 0.805-0.831 for the
baselines. One held-out scroll.

## Other unseen scrolls

Two scrolls that no model here was trained on, read with the production models (trained on all four labelled scrolls,
25k steps, mirror TTA), ROC-AUC:

- PHerc 0841, human labels, the inkbench `p0841` protocol (three overlapping segments, compat label transfer, tie-aware
  AUC inside validation AND supervision, mean over segments). Our renders of the published 9.366 um meshes, read in
  reverse (the text side on these renders). Anchor: the public ink_9um seed-42 checkpoint scores 0.763 here (inkbench
  reports 0.739 for it).
- PHerc 0343P, IR-photograph letter truth on the 8.64 um / 116 keV scan (masks from DomRusso2/ir-truth-9um, segment
  20250511003658), forward.

| production models, seeds 42 / 43 / 44 (ensemble) | PHerc 0841 | PHerc 0343P |
|---|---|---|
| recipe, no scan transfer | 0.767 / 0.748 / 0.755 (0.769) | 0.746 / 0.746 / 0.767 (0.766) |
| + scan transfer, earlier revision (native groups converted too) | 0.758 / 0.769 / 0.777 (0.780) | 0.732 / 0.719 / 0.742 (0.740) |
| + scan transfer, `skip_datasets: ["native9"]` | 0.753 / 0.774 / 0.781 (0.782) | 0.748 / 0.741 / 0.759 (0.759) |

For reference, dense_native (Nieuwlaar, MIT) scores 0.814 / 0.768; inkbench publishes 0.867 for Reader V3 on 0841.
With the earlier revision the conversion did not carry over consistently (seed mean / ensemble against the recipe:
+0.011 / +0.011 on 0841, within that benchmark's noise; -0.022 / -0.026 on 0343P, a 116 keV scan unlike the 0814 pair
the kernel was fitted on). With `skip_datasets` the 0343P loss is gone (0.749 / 0.759 against 0.753 / 0.766) and 0841
keeps the small gain (0.769 / 0.782 against 0.757 / 0.769, within the benchmark's noise). Seed 44 of the fixed row was
trained on a different GPU with the same code and config.

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
