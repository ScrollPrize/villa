---
id: prizes
title: "Open Prizes"
sidebar_label: "Open Prizes"
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,500,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,500,000+ machine learning and computer vision competition"
  />
  <meta
    property="og:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="A $1,500,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

import TOCInline from '@theme/TOCInline';

Vesuvius Challenge is ongoing and **YOU** can win the below prizes and help us make history!

***

<TOCInline
  toc={toc}
/>

***

## Unwrapping at Scale Prize

We’re awarding **$200,000** (single winner) for delivering **automated virtual unwrapping at production scale on two different scrolls**.

**Scope (two-scroll generalization).** Your pipeline must pass all requirements on **two distinct scrolls** from our official datasets. **At least one must be Scroll 1 _or_ Scroll 5.** The second can be **a different** scroll from the set.

Ink detection is **not required**; the focus is high-quality segmentation, continuity, and flattening suitable for later reading.

We accept submissions for this prize until 11:59pm Pacific, December 31st, 2025!

<div className="mb-4">
  <img src="/img/landing/scroll.webp" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">A carbonized scroll.</figcaption>
</div>

<details class="submission-details">
<summary>Submission criteria and requirements</summary>

### What to deliver (for each of the two scrolls)
1. **Segmented sheet manifold(s).** A continuous 3D mesh (or meshes) of the papyrus sheet(s) suitable for flattening. If the submission includes multiple meshes, the minimum size of each mesh should be larger than a full papyrus sheet wrap in the scroll volume.
2. **Flattened 2D sheets.** UV maps / atlases covering the **accepted area** you claim as correct.
3. **Accepted-area mask.** Binary mask(s) over the flattened sheets indicating regions you assert are error-free.
4. **Video record (if humans involved).** A screen-capture timelapse or periodic snapshots covering all interactive steps.
5. **Reproducible pipeline.** Container with a one-click script that regenerates meshes, maps, masks, and sheets from organizer-supplied volumes.

### Pass/fail gates (must be met on **both** scrolls)
- **Coverage: ≥ 70% (per scroll)** of the (to be estimated) total scroll surface after exclusions of areas masked as errors in the binary masks.
- **Sheet-switch rate: ≤ 0.5% per scroll** over the audited area.
    - *Definition:* It's the rate of triangles or quads in the delivered meshes that are marked as error-free in the binary masks but actually contain errors.
- **Human effort cap: ≤ 72 human-hours per scroll** (i.e., **≤ 144 hours total** across the two).
    - *Counts:* any human touch specific to processing the evaluation scrolls (seed placement, parameter tweaks, manual stitching/edits, quality control passes, mask painting, bookkeeping).
    - *Doesn’t count:* general R&D prior to evaluation, model training on public data, writing docs, idle waiting while jobs run.
- **Reproducibility:** Organizers must be able to re-run your container end-to-end and reproduce deliverables and metrics.

> **No compute cap.** We do not limit hardware or cloud cost.

### Data & generalization
- Two distinct scrolls from our official set. **At least one must be Scroll 1 or Scroll 5.** The second can be a **different** scroll.  
- You may use public volumes/fragments for development. For prize evaluation, organizers provide the exact evaluation volumes for the two scrolls.  

### Submission package
- **GitHub repository** with code, container (e.g. Docker, conda, etc.), and run scripts.  
- **Deliverables:** 3D meshes, flattened sheets (TIFF/PNG), UV maps, accepted-area masks.  
- **Logs:** timing CSVs per scroll; screen-capture or snapshots; CLI logs; environment/hardware info; container image digest.  
- **Method note:** 2–6 pages describing pipeline, assumptions, and known failure modes.  
- **Licensing:** if you win, you agree to open-source the method under the CC BY-NC 4.0 License.

### Winner determination & tie-breakers
- **Single winner:** the **first team** to pass all gates on **both** scrolls during organizer verification receives **$200,000**.  
- If two teams pass within **96 hours**, tie-breakers:  
  1) Lower **sheet-switch rate** across the two;
  2) Higher **coverage** across the two scrolls;    
  3) Fewer **total human-hours** (sum over both);  
  4) Earlier submission timestamp.

### Deadline
- **11:59pm Pacific, December 31st, 2025**

</details>

[Submission Form](https://forms.gle/MqP3XQGX7o2ZFfZW6)

***

## 3D Surface Detection Challenge (hosted on Kaggle)

We are holding a $100,000 prize pool competition for detection of the recto papyrus surfaces in the 3D CT scans.
The contest is currently hosted on Kaggle on [this page](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection).

### Prizes
- 1st place: $25,000
- 2nd place: $20,000
- 3rd place: $15,000
- 4th place: $10,000
- 5th–10th: $5,000

### Timeline
- November 13, 2025 - Start Date.
- February 6, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
- February 6, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
- February 13, 2026 - Final Submission Deadline.

***

## First Letters and Title Prizes

One of the frontiers of Vesuvius Challenge is finding techniques that work across multiple scrolls.
While we've discovered text in some of our scrolls, others have not yet produced legible findings.
Finding the first letters inside one of these scrolls is a major step forward.

Additionally, finding the title of any scroll is a huge and exciting discovery that helps scholars contextualize the rest of the work!

**First Letters: $60,000 to the first team that uncovers 10 letters within a single 4cm^2 area of any of Scrolls 2-4.**

**First Title: $60,000 to the first team to discover the title in any of Scrolls 1-4.**

<div className="mb-4">
  <img src="/img/data/title_example.webp" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">Visible Title in a Scroll Fragment.</figcaption>
</div>

<details>
<summary>Submission criteria and requirements</summary>

* **Image.** Submissions must be an image of the virtually unwrapped segment, showing visible and legible text.
  * Submit a single static image showing the text region. Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text. This includes annotations that were then used as training data and memorized by a machine learning ink model. Ink model outputs of this region should not overlap with any training data used.
  * For the First Title Prize, please illustrate the ink predictions in spatial context of the title search, similar to what is [shown here](https://scrollprize.substack.com/p/30k-first-title-prize). You **do not** have to read the title yourself, but just have to produce an image of it that our team of papyrologists are able to read.
  * Specify which scroll the image comes from. For multiple scrolls, please make multiple submissions.
  * Include a scale bar showing the size of 1 cm on the submission image.
  * Specify the 3D position of the text within the scroll. The easiest way to do this is to provide the segmentation file (or the segmentation ID, if using a public segmentation).
* **Methodology.** A detailed technical description of how your solution works. We need to be able to reproduce your work, so please make this as easy as possible:
  * For fully automated software, consider a Docker image that we can easily run to reproduce your work, and please include system requirements.
  * For software with a human in the loop, please provide written instructions and a video explaining how to use your tool. We’ll work with you to learn how to use it, but we’d like to have a strong starting point.
  * Please include an easily accessible link from which we can download it.
* **Hallucination mitigation.** If there is any risk of your model hallucinating results, please let us know how you mitigated that risk. Tell us why you are confident that the results you are getting are real.
  * We strongly discourage submissions that use window sizes larger than 0.5x0.5 mm to generate images from machine learning models. This corresponds to 64x64 pixels for 8 µm scans. If your submission uses larger window sizes, we may reject it and ask you to modify and resubmit.
  * In addition to hallucination mitigation, do not include overlap between training and prediction regions. This leads to the memorization of annotated labels.
* **Other information.** Feel free to include any other things we should know.

Your submission will be reviewed by the review teams to verify technical validity and papyrological plausibility and legibility.
Just as with the Grand Prize, please **do not** make your discovery public until winning the prize. We will work with you to announce your findings.
</details>

[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform?usp=dialog)

***

> ⚠️ The previous prizes are too ambitious? You can still contribute!

## Progress Prizes

In addition to milestone-based prizes, we offer monthly prizes for open source contributions that help read the scrolls.
These prizes are more open-ended, and we have a wishlist to provide some ideas.
If you are new to the project, this is a great place to start.
Progress prizes will be awarded at a range of levels based on the contribution:

* Gold Aureus: \$20,000 (estimated 4-8 per year) – for major contributions
* Denarius: \$10,000 (estimated 10-15 per year)
* Sestertius: \$2,500 (estimated 25 per year)
* Papyrus: \$1,000 (estimated 50 per year)

We favor submissions that:
* Are **released or open-sourced early**. Tools released earlier have a higher chance of being used for reading the scrolls than those released the last day of the month.
* Actually **get used**. We’ll look for signals from the community: questions, comments, bug reports, feature requests. Our Annotation Team will publicly provide comments on tools they use.
* Are **well documented**. It helps a lot if relevant documentation, walkthroughs, images, tutorials or similar are included with the work so that others can use it!

We maintain a [public wishlist](https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) of ideas that would make excellent progress prize submissions.
[Improvements to VC3D](https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3AVC3D) can be also considered for progress prizes!
Some are additionally labeled as [good first issues](https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) for newcomers!

Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, November 30th, 2025!

<details>
<summary>Submission criteria and requirements</summary>

**Core Requirements:**
1. Problem Identification and Solution
   * Address a specific challenge using Vesuvius Challenge scroll data
   * Provide clear implementation path and a demonstration of its use
   * Demonstrate significant advantages over existing solutions
2. Documentation
   * Include comprehensive documentation
   * Provide usage examples
3. Technical Integration
   * Accept standard community formats (e.g. OME-Zarr or Zarr arrays, quadmeshes, triangular meshes)
   * Maintain consistent output formats
   * Designed for modular integration
</details>

[Submission Form](https://forms.gle/G4QenHtuasL2vUeh7)

***

## Terms and Conditions

Prizes are awarded at the sole discretion of Curious Cases, Inc. and are subject to review by our Technical Team, Annotation Team, and Papyrological Team. We may issue more or fewer awards based on the spirit of the prize and the received submissions. You agree to make your method open source if you win a prize. It does not have to be open source at the time of submission, but you have to make it open source under a permissive license to accept the prize. Submissions for milestone prizes will close once the winner is announced and their methods are open sourced. Curious Cases, Inc. reserves the right to modify prize terms at any time in order to more accurately reflect the spirit of the prize as designed. Prize winner must provide payment information to Curious Cases, Inc. within 30 days of prize announcement to receive prize.
