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
    content="A $1,000,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,000,000+ machine learning and computer vision competition"
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
    content="A $1,000,000+ machine learning and computer vision competition"
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

## Read Entire Scroll Prize ($200,000)

We’re awarding $200,000 to the first team to unwrap an entire scroll and produce readable letters within.

We've made great strides by uncovering hidden text from inside a sealed Herculaneum scroll (now multiple!), but we're still reading 5% or less of full scrolls.
Segmenting and reading an entire scroll, probably using more automated methods, is the next major milestone for our technical endeavor.
Reach this bar, and $200,000 is yours!

<div className="mb-4">
  <img src="/img/landing/scroll.webp" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">An Entire Scroll.</figcaption>
</div>

<details class="submission-details">
<summary>Submission criteria and requirements</summary>

1. **Segmentation**
* Estimate the total area of the scroll’s written surface (in cm2)
* Compute the total surface area of the segmented mesh (in cm2) in your submission. **You must segment 90% or more of the total estimated area**
* Segments should be flattened and shown in 2D as if the scroll were unwrapped.
* The scroll should ideally be captured by a single segmentation (or each connected component of the scroll) rather than separate overlapping segmentations.
* Segments should pass geometric sanity checks; for example, no self-intersections

2. **Ink detection**
* Your submission must contain ink predictions for the entire flattened mesh, of the same shape as the flattened surface
* The entire submission is too large to transcribe quickly, so the papyrological team will evaluate each line as:
    * ✅ **readable** (could read 85% of the characters),
    * ❌ **not readable** (couldn't),
    * 🟡 **maybe** (would have to stop and actually do the transcription to determine), or
    * 🔷 **incomplete** (line incomplete due to the physical boundaries of the scroll)
* 80% of the total complete lines (incomplete lines will not be judged) must be either 🟡 **maybe** or ✅ **readable**. Multiple papyrologists may review each line, in which case ties will be broken favorably towards the submission.

As a baseline, here's how the 2023 Grand Prize banner would have scored:

<div className="mb-4">
  <img src="/img/2024-prizes/GP_scores_sample.webp" className="w-[80%]"/>
  <figcaption className="mt-[-6px]">Visible Ink on Scroll 5 (<a href="/img/ink/2023_GP_banner_lines_score.webp">full banner</a>).</figcaption>
</div>

Total lines: 240. Complete lines: 206. Passing lines: 137. Pass rate: 137 / 206 = **67% (needs to be 80%)**.

We may reward partial work - if your unrolling works but the ink detection isn't all the way there yet, go ahead and submit!

</details>

[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSed2vRgW3HECXW3E6-llYPOST550yr4L0T5AIQp45GAYRcnGQ/viewform?usp=dialog)

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
Some are additionally labeled as [good first issues](https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) for newcomers!

Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, April 30th, 2025!

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
   * Accept standard community formats (multipage TIFs, on-disk numpy arrays)
   * Maintain consistent output formats
   * Designed for modular integration
</details>

[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSc0yrAetoErg3BPt9FYblj3Emhg9eZcdJYJ15M7K63izO9ICQ/viewform?usp=dialog)

***

## Referral Prize

Refer a successful hire - earn a $5,000 prize.

<details>
<summary>Submission criteria and requirements</summary>

1. **The Offer:** Earn $5,000 for referring a candidate we hire for our research team!
2. **Referrer Eligibility:** You must be legally eligible to receive the prize payment.
3. **Hire Eligibility:** Candidates must be new contacts and legally authorized to receive payment from Vesuvius Challenge.
4. **How to Refer:** Email jobs@scrollprize.org with subject: "Referral: [Candidate Name]." Include: your name & Discord username (if applicable), your contact information, candidate's name & contact information (confirming their permission), and why they're a good fit.
5. **Terms:** Hires and prizes are determined at the sole discretion of Curious Cases, Inc. Referral must be submitted per the above instructions. Referral must be received before the candidate applies directly. Your referral must be the first one received for that candidate. The candidate must acknowledge your referral during hiring. The candidate must be hired for a full-time research team role.
</details>

***

## Terms and Conditions

Prizes are awarded at the sole discretion of Curious Cases, Inc. and are subject to review by our Technical Team, Annotation Team, and Papyrological Team. We may issue more or fewer awards based on the spirit of the prize and the received submissions. You agree to make your method open source if you win a prize. It does not have to be open source at the time of submission, but you have to make it open source under a permissive license to accept the prize. Submissions for milestone prizes will close once the winner is announced and their methods are open sourced. Curious Cases, Inc. reserves the right to modify prize terms at any time in order to more accurately reflect the spirit of the prize as designed. Prize winner must provide payment information to Curious Cases, Inc. within 30 days of prize announcement to receive prize.
