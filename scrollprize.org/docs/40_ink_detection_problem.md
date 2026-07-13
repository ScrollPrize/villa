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

Current models can recover readable text from some scrolls. Iterative labeling and ink detection were central to the complete virtual unwrapping and reading of PHerc. 1667. On other scrolls, however, the same models may produce weak, misleading, or no convincing ink. Making a model work well often requires creating many labels for that particular scroll and repeatedly retraining it.

The goal is to build ink detection that **generalizes to new scrolls and scan conditions while using far less labeled data**.

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

## What a good ink detection model should look like

The target is not another model that performs well only where it was heavily labeled. Strong progress would include:

- cross-scroll evaluation with strict separation between training and test scrolls;
- strong few-shot adaptation from a small number of trusted labels;
- self-supervised or large-scale pretraining on unlabeled CT volumes;
- representations and augmentations robust to scan, resolution, and surface-placement changes;
- direct 3D segmentation where the signal supports trustworthy voxel-level targets;
- tools that identify the few regions most valuable for a human to label;
- diagnostics that isolate scanning, geometry, label, and model failures.

## Some ideas we've been considering, in no particular order
Taken directly from our last brainstorming session on improving ink detection. Not intended as a comprehensive list, and the exclusion of any idea is not an indication that it is not worth considering.

<div className="vc-card my-8">

### image data

- train 2d model slicewise, compute loss as max over channels
    - model forced to localise, only sees one slice at a time
- train on 2D “depth-maps” (somehow generated from segments, looking toward the recto)
    - could be gotten via a mean downsample > resize (gives a sort of depth map)
    - could smooth the segmentation surface → step along normal until some minimum intensity value is hit → write out as essentially a distance transform image
- finer stepping along normals during rendering (*vc_render_tifxyz supports this already*)
    - simply write more layers or ,
    - accum partial steps — take some substep of a full step, accum n steps into a max/avg/something and write out at the full step length
- multi-channel input with different energies
    - likely should be done , but do not have multi-energy data for all scrolls
- multi-resolution input (native scan res, not downsampled, probably scaled?)
    - other resolutions show differences in the scan data which is different from just a simple “one is more blurry” , sometimes features show up more at one than other
    - difficult to match shapes w/o some complex feature fusion kind of shit. could pad but then you have zero convs in the entire network all the time.

### label data

- skeletonize labels
    - sort of topo-adjacent, forces model to learn the “shape” of a letter in a sense
        - very difficult to get clean skeletons
        - what is the “right” skeleton for some letters?
            - is there some other form of “minimal annotation” which can represent a letter?
- persistent affine/tps transform of labels, optimised jointly with model
    - could help with “bad” labels
    - also in 3D? allowing the gt to ‘bend’ to find the sheet?

### supervised losses

- 2D, ‘stitched’ (i.e. operating over large areas, probably several independent forward passes)
    - was attempted, slow and model had artifacts at the stitch borders
        - maybe from zero-pad? unsure
    - should possibly be revisited
- spatially-hinged losses — *likely clear win since we know alignment is problematic*
    - don’t require matching the exact stroke thickness; instead chamfer or something
    - keep receptive field small; apply loss over multiple RFs
- topological losses (loops etc.) — *likely to correlate strongly with legibility*
    - almost all topo based losses are prohibitively expensive in full 3d, loss likely should be computed in 2d (either from flattened or in the 3d projected case)
    - betti matching loss
        - *was tried in march ink sprint, results were positive on the 2.5d case. much less noise, similar ink but hard to put an exact metric to it.*
    - skel recall / medial surface
    - keep receptive field small; apply loss over multiple RFs
- for 3D, ‘slab aware’ loss (instead of trying to restrict ‘lifted’ labels to surface)
    - i.e. max-pool along normals
    - *tried very briefly in march ink sprint, impl was slow and ran out of time to properly eval*
- “is this the right letter?” loss, based on similarity of embeddings of predictions/gt → *might give a better learning signal than topo etc while capturing same info*
- local supervised contrastive — to get more info out of the labels

### semi-supervised / transfer / out-of-distribution / domain-shift

- transfer learning from **self-supervised** models…
    - scroll-DINO-pretrained model → sota for images →
        - https://arxiv.org/pdf/2512.00872 TAP-CT
        - https://arxiv.org/pdf/2511.17209 SPECTRE
    - NEPA → simple and with few design decisions →
    - *JEPA-pretrained model (maybe LeJEPA)  → architecture agnostic but more hyperparameters re view selection etc
- IRM-ish stuff — to encourage ‘smooth’ transfer across domains (BayesianIRM ?)
    - …or Vanilla GroupDRO or Group Distributionally Robust Machine Learning
- alignment of features across scrolls (i.e. domain generalisation / robustness, smart than augmentations / lots-of-data)
    - adversarial (DANN; DGAFL) / variational / other
    - relevant in semi-supervised case, but also pretraining
- mean teacher / cross-teaching / etc.
    - pseudo-labels automatic weighting https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09643.pdf
- Adversarial Style Augmentation ( AdvStyle-ish https://arxiv.org/pdf/2207.04892)
- more experiments on transfer from natural-image-pretrained DINO
- **3d dino transfer**
    - pretrained model produces embeddings from which simple cosine similarity among an average of selected embeddings can directly resolve ink.
        - giorgio trained a sort of student/teacher distillation with this by taking a 3d ink model i trained and this averaged embedding from the 3d dino and multiplying the prediction from the 3d label by the embedding produced from the pretrained model to create a sort of pseudo-label
        - embedding is somewhat weak,  could likely be reinforced
            - easiest path for this is look for ink in segments, go to that 3d location, and place a point directly on the ink rather than the segmentation line. should be fast, segmentation line might be good enough as-is?
            - could add some contrastive-ness to this (this embedding is fiber, this embedding is *not* ink, etc)
            - could enforce same-embedding across distinct letters over a longer range, and contrast between different ones.
    - if we want to continue with flattened ink detection, we should pretrain a model on flattened segments
        - will help ink “imbalance” problem somewhat present in 3d data
            - could also fix this by just oversampling from chunks with ink
        - can also do *much* larger crop sizes due to shortened z dim

### unsupervised / unpaired losses

- 2D, ‘stitched’
    - letter-recall (c.f. letter-recall unpaired metric) → *seems like the unpaired loss that’ll give the cleanest learning signal*
    - score-distillation to use a diffusion prior for guidance, starting from a pretrained ckpt → *likely hard to learn*

### other ideas
- 3d model conditioned on input seg mask , predict ink on this segment only, pool loss across normals
- 2.5 flattened space seg mask conditioning (show the model which sheet we are looking for ink on)

</div>

## Start here

- [Learn how current models, labels, training, and inference work](/tutorial5)
- [Read the in-depth discussion of ink recovery and its failure modes](/2026_open_problems#3-ink-recovery-reading-the-scrolls)
- [Browse the current `ink-labels` dataset](/data_datasets#ink-labels-2026-07)
- [Read the ink-label dataset documentation](pathname:///data/datasets/ink-labels-README.md)
- [Inspect the ink-detection code](https://github.com/ScrollPrize/villa/tree/main/ink-detection)
- [Browse released model checkpoints](https://huggingface.co/scrollprize)

Join the [Discord](https://discord.gg/V4fJhvtaQn) to coordinate experiments, validation splits, and labeling work with the team.
