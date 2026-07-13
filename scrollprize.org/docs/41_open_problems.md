---
title: "Open Problems"
slug: /open_problems
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="The two open problems standing between us and reading every Herculaneum scroll: virtual unwrapping (winding annotations and spiral fitting) and ink detection."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="The two open problems standing between us and reading every Herculaneum scroll: virtual unwrapping (winding annotations and spiral fitting) and ink detection."
  />
  <meta property="og:image" content="https://scrollprize.org/img/social/opengraph.jpg" />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="The two open problems standing between us and reading every Herculaneum scroll: virtual unwrapping (winding annotations and spiral fitting) and ink detection."
  />
  <meta property="twitter:image" content="https://scrollprize.org/img/social/opengraph.jpg" />
</head>

import BeforeAfter from "@site/src/components/BeforeAfter";

# Open Problems

We have read one scroll. Reading the rest — faster, cheaper, and at scale — comes down to two problems: recovering the wound-up writing surface from the CT volume (virtual unwrapping), and recovering the ink on it (ink detection).

## Virtual Unwrapping

<div className="border border-solid border-[var(--vc-line)] rounded-lg bg-[var(--vc-surface)] p-3 mb-6">
  <BeforeAfter
    beforeImage="/img/data/raw_pred.png"
    afterImage="/img/data/patches.png"
  />
  <figcaption className="mt-2 text-sm text-[var(--vc-text-faint)]">Raw surface predictions (left) vs. traced surface patches (right) — drag the slider.</figcaption>
</div>

A CT scan yields voxels, not pages: the writing surface must be segmented, meshed, and flattened. A scroll is a single papyrus sheet wound into a spiral around a central axis (the *umbilicus*). The hardest part of unwrapping is not finding papyrus, it is deciding **which wrap (*winding*) of the sheet each piece of papyrus belongs to** where sheets press together, tear, or drift apart.

Our current pipeline uses **winding annotations** made in VC3D: manually-verified surface patches, line annotations traced across the surface (often following fibers), and point collections — each carrying same-, relative-, or absolute-winding relationships. A **spiral fitting** optimization then fits a global winding solution for the whole scroll that is consistent with these annotations, producing a single parametrized surface that can be flattened and rendered for ink detection.

Producing annotations is manual, skilled work, and each scroll needs many of them: the [`spiral-input`](/data_datasets) dataset for PHercParis4 alone contains tens of thousands of patches and annotations. To scale to hundreds of scrolls we need winding annotations that are **more plentiful, faster to produce, and ultimately automated** — better tooling, smarter interpolation, and models that propose windings for humans to verify rather than draw from scratch.

<div className="flex flex-wrap gap-2 mb-4" aria-label="Related skills">
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">geometry processing</span>
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">3D computer vision</span>
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">optimization</span>
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">C++</span>
</div>

<div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-0 border-t border-solid border-[var(--vc-line)] pt-2.5 mb-6 text-[0.8125rem]">
  <span className="vc-label">Solve it, win</span>
  <a href="/prizes">2027 Grand Prize + monthly Progress Prizes&nbsp;→</a>
</div>

<div className="vc-label mb-1">Go deeper</div>

- [Winding Annotations](/open_problems/winding_annotations) — the problem in depth
- [Tutorial: Volume Cartographer 3D](/tutorial_VC3D) — the tool used to create annotations
- [Tutorial: Spiral Fitting](/tutorial_spiral) — fitting a whole-scroll surface to the annotations
- [`spiral-input` dataset](/data_datasets) — the annotations themselves, ready to download

## Ink Detection

<div className="border border-solid border-[var(--vc-line)] rounded-lg bg-[var(--vc-surface)] p-3 mb-6">
  <BeforeAfter
    beforeImage="/img/ink/51002_crop/32.jpg"
    afterImage="/img/ink/51002_crop/prediction.jpg"
  />
  <figcaption className="mt-2 text-sm text-[var(--vc-text-faint)]">Surface volume (left) vs. model ink prediction (right) — drag the slider.</figcaption>
</div>

Carbon ink is nearly indistinguishable from papyrus in X-ray CT — it has almost the same density, so it is invisible to direct inspection. Machine-learning models learn to detect it from subtle 3D texture instead. On detached fragments, infrared photography of exposed writing provides ground truth; inside sealed scrolls there is no such reference. Labels begin as hand-annotated ink strokes and are refined by **iterative pseudo-labeling**. Models trained on early labels reveal fainter strokes, which become the next round of training data.

This works — [we have read one scroll](/firstscroll) — but it does not yet generalize. Ink has surfaced on 9 of the 45 scanned scrolls and fragments, not always legibly, and a model trained on one scroll often fails on another scanned at different resolution or energy. **Robust ink detection across scrolls and scan conditions is the open problem**, and it needs both better models and more labeled ink.

<div className="flex flex-wrap gap-2 mb-4" aria-label="Related skills">
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">machine learning</span>
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">computer vision</span>
  <span className="text-xs leading-none px-2 py-[5px] border border-solid border-[var(--vc-line)] rounded-md text-[var(--vc-text-faint)] whitespace-nowrap">domain generalization</span>
</div>

<div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-0 border-t border-solid border-[var(--vc-line)] pt-2.5 mb-6 text-[0.8125rem]">
  <span className="vc-label">Solve it, win</span>
  <a href="/prizes">First Letters + First Title prizes&nbsp;→</a>
</div>

<div className="vc-label mb-1">Go deeper</div>

- [Ink Detection](/open_problems/ink_detection) — the problem in depth
- [Tutorial: Ink Detection](/tutorial5) — train your first ink-detection model
- [`ink-labels` dataset](/data_datasets) — binary ink masks for training
