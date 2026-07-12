---
id: ink_detection_problem
slug: /open_problems/ink_detection
title: "Ink Detection"
sidebar_label: "Ink Detection"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="An introduction to the open ink-detection problem: making models generalize across Herculaneum scrolls while requiring far fewer labels."
  />
  <meta property="og:type" content="article" />
  <meta property="og:url" content="https://scrollprize.org/open_problems/ink_detection" />
  <meta property="og:title" content="Ink Detection | Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Ink detection works, but not reliably across scrolls and not yet without extensive scroll-specific labeling."
  />
  <meta property="og:image" content="https://scrollprize.org/img/social/opengraph.jpg" />
  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:title" content="Ink Detection | Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="Ink detection works, but not reliably across scrolls and not yet without extensive scroll-specific labeling."
  />
  <meta property="twitter:image" content="https://scrollprize.org/img/social/opengraph.jpg" />
</head>

import ChatCallout from '@site/src/components/ChatWidget/ChatCallout';

*Last updated: July 12, 2026*

<ChatCallout prefill="Explain the open ink detection problem and how I could work on it" />

**We have ink detection. We do not yet have ink detection that works reliably across the collection.**

Current models can recover readable text from some Herculaneum scroll data. Iterative labeling and ink detection were central to the complete virtual unwrapping and reading of PHerc. 1667. On other scrolls, however, the same models may produce weak, misleading, or no convincing ink. Making a model work well often requires creating many labels for that particular scroll and repeatedly retraining it.

The open problem is to build ink detection that **generalizes to new scrolls and scan conditions while using far less labeled data**.

<figure>
  <div className="flex flex-wrap justify-center">
    <div className="w-[31%] mr-[2%]">
      <img src="/img/tutorials/ink-modality-color.webp" />
      <div className="text-center text-sm text-dim">Color photograph</div>
    </div>
    <div className="w-[31%] mr-[2%]">
      <img src="/img/tutorials/ink-modality-infrared.webp" />
      <div className="text-center text-sm text-dim">Infrared photograph</div>
    </div>
    <div className="w-[31%]">
      <img src="/img/tutorials/ink-modality-xray.webp" />
      <div className="text-center text-sm text-dim">X-ray CT</div>
    </div>
  </div>
  <figcaption className="mt-0">The same detached fragment in visible light, infrared, and X-ray CT. The writing is clear in infrared but almost disappears in the CT data used to see inside a sealed scroll.</figcaption>
</figure>

## What ink detection does

Carbon ink and carbonized papyrus have very similar X-ray attenuation, so the ink can be nearly invisible in a raw CT slice. An ink-detection model looks for subtler local differences in texture, morphology, or material response and predicts where ink is likely to be.

Most current workflows operate on a **surface volume**: a stack of CT samples taken just above and below a virtually unwrapped papyrus surface. The model receives this local 3D neighborhood and produces a 2D ink-probability image. Where the signal and surface geometry are favorable, ink can also be segmented directly in the original 3D volume.

This is signal recovery, not OCR. The model is not asked to guess Greek words or draw plausible letters; a papyrologist reads the recovered evidence.

## Where the labels come from

Detached fragments provide the cleanest ground truth. Their exposed writing can be photographed in infrared, aligned with a CT scan, and traced into ink labels. But a fragment's exposed surface is not the same domain as the damaged interior of every sealed scroll.

Inside a scroll there is no infrared photograph. Labels are usually bootstrapped through **iterative labeling**:

1. Run an existing model on a segmented surface.
2. Conservatively label the clearest predicted strokes and nearby non-ink background.
3. Retrain the model on the expanded labels.
4. Repeat while measuring performance on a region that is never used for labeling.

This loop works, but it is labor-intensive and scroll-specific. It also creates a risk of reinforcing a model's own mistakes, which is why conservative labeling and genuinely held-out validation are essential.

## Why current models do not generalize well

Different scrolls—and even different regions of one scroll—can vary in preservation, compression, ink chemistry, surface morphology, scanning resolution, and reconstruction quality. Fragment labels provide only a 2D location even though the usable ink signal may lie at an uncertain depth in the 3D scan. A slightly misplaced surface can hide the signal from an otherwise capable model.

This makes failures hard to diagnose. A blank prediction could mean that the ink signal is absent, that the scan did not capture it strongly enough, that the surface is misplaced, or that the labels, representation, architecture, or training distribution are wrong. Better diagnostics are therefore part of the ink-detection problem, not an afterthought.

## What progress looks like

The target is not another model that performs well only where it was heavily labeled. Strong progress would include:

- cross-scroll evaluation with strict separation between training and test scrolls;
- strong few-shot adaptation from a small number of trusted labels;
- self-supervised or large-scale pretraining on unlabeled CT volumes;
- representations and augmentations robust to scan, resolution, and surface-placement changes;
- direct 3D segmentation where the signal supports trustworthy voxel-level targets;
- uncertainty estimates that distinguish confident negatives from unfamiliar data;
- tools that identify the few regions most valuable for a human to label;
- diagnostics that isolate scanning, geometry, label, and model failures.

Ultimately, a useful system should arrive at a newly scanned scroll with a strong prior for what ink looks like, improve from a small amount of expert input, and tell us when its answer should not be trusted.

## Start here

- [Learn how current models, labels, training, and inference work](/tutorial5)
- [Read the in-depth discussion of ink recovery and its failure modes](/2026_open_problems#3-ink-recovery-reading-the-scrolls)
- [Browse the current `ink-labels` dataset](/data_datasets#ink-labels-2026-07)
- [Read the ink-label dataset documentation](pathname:///data/datasets/ink-labels-README.md)
- [Inspect the ink-detection code](https://github.com/ScrollPrize/villa/tree/main/ink-detection)
- [Browse released model checkpoints](https://huggingface.co/scrollprize)

Join the [Discord](https://discord.gg/V4fJhvtaQn) to coordinate experiments, validation splits, and labeling work with the team.
