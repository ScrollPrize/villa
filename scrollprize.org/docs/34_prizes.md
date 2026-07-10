---
id: prizes
title: "Open Prizes"
sidebar_label: "Open Prizes"
# Structured mirror of the open prizes described in the body of this page.
# Read at build time by plugins/prizes-data.js and rendered on the landing
# page (usePluginData("prizes-data")). KEEP IN SYNC with the body copy:
# editing amounts/tiers here updates the landing automatically.
prizes:
  - id: grand-prize-2027
    title: "2027 Grand Prize"
    amount: 1000000
    cadence: "Deadline June 25th, 2027"
    href: "/prizes#2027-grand-prize"
    hook: "Fully unroll and read one of 23 sealed scrolls."
    featured: true
    tiers:
      - name: "1st"
        amount: 800000
      - name: "2nd"
        amount: 100000
      - name: "3rd"
        amount: 50000
      - name: "4th"
        amount: 50000
  - id: first-letters
    title: "First Letters"
    amount: 500000
    cadence: "Max 10 scrolls · Deadline June 25th, 2027"
    href: "/prizes#first-letters-prizes"
    hook: "$50,000 per scroll across the 2027 Grand Prize volumes: uncover 10 letters within a single 4 cm² area."
  - id: first-title
    title: "PHerc. Paris 4's Title"
    amount: 50000
    cadence: "Deadline June 25th, 2027"
    href: "/prizes#first-title-prize"
    hook: "Discover the title of PHerc. Paris 4 (Scroll 1) — any scan, including the 2.4 µm volumes."
  - id: progress-prizes
    title: "Progress Prizes"
    amount: 590000
    unit: "per year"
    cadence: "Awarded monthly"
    href: "/prizes#progress-prizes"
    hook: "Open-ended awards for open source contributions — including $20,000 every month for the best submission."
    tiers:
      - name: "Best of the month"
        amount: 20000
      - name: "Gold Aureus"
        amount: 20000
      - name: "Denarius"
        amount: 10000
      - name: "Sestertius"
        amount: 2500
      - name: "Papyrus"
        amount: 1000
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Open Vesuvius Challenge prizes: win cash for finding the first letters or title in a Herculaneum scroll, plus monthly progress prizes for open source work."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Open Vesuvius Challenge prizes: win cash for finding the first letters or title in a Herculaneum scroll, plus monthly progress prizes for open source work."
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
    content="Open Vesuvius Challenge prizes: win cash for finding the first letters or title in a Herculaneum scroll, plus monthly progress prizes for open source work."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

import PrizePoolBanner from '@site/src/components/PrizePoolBanner';

Vesuvius Challenge is ongoing and **YOU** can win the below prizes and help us make history!

<PrizePoolBanner />

***

## 2027 Grand Prize {#2027-grand-prize}

**\$800,000 to the first team or individual to fully unroll and read a scroll from the set below.** We also have prizes for **second place (\$100,000)**, **third place (\$50,000)**, and **fourth place (\$50,000)** — \$1,000,000 in total. Prizes are awarded in the order that qualifying submissions are made.

Prizes will be awarded to any team or individual that fully digitally unrolls and makes readable (according to the conditions and requirements specified below) one of the eligible CT scans of carbonized scrolls from Herculaneum:

<details>
<summary>Eligible scroll volumes (23)</summary>

1. [PHerc. 125](/data_browser/PHerc0125) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22125_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
2. [PHerc. 175A](/data_browser/PHerc0175A) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22175A_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
3. [PHerc. 175B](/data_browser/PHerc0175B) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0175B/volumes/20250521125822-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22175B_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
4. [PHerc. 191](/data_browser/PHerc0191) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0191/volumes/20250821151635-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22191_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
5. [PHerc. 211](/data_browser/PHerc0211) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22211_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
6. [PHerc. 257](/data_browser/PHerc0257) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0257/volumes/20250821151750-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22257_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
7. [PHerc. 268](/data_browser/PHerc0268) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0268/volumes/20251110183117-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22268_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
8. [PHerc. 306B](/data_browser/PHerc0306B) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22306B_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
9. [PHerc. 343](/data_browser/PHerc0343) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22343_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
10. [PHerc. 358](/data_browser/PHerc0358) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22358_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
11. [PHerc. 483A](/data_browser/PHerc0483A) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0483A/volumes/20250521140913-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22483A_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
12. [PHerc. 483B](/data_browser/PHerc0483B) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0483B/volumes/20251124083638-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22483B_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
13. [PHerc. 490A](/data_browser/PHerc0490A) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0490A/volumes/20250521151210-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22490A_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
14. [PHerc. 490B](/data_browser/PHerc0490B) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0490B/volumes/20250521151215-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22490B_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
15. [PHerc. 800](/data_browser/PHerc0800) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%22800_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
16. [PHerc. 813](/data_browser/PHerc0813) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22813_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
17. [PHerc. 826](/data_browser/PHerc0826) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22826_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
18. [PHerc. 846A](/data_browser/PHerc0846A) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22846A_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
19. [PHerc. 846B](/data_browser/PHerc0846B) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%22846B_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
20. [PHerc. 1203](/data_browser/PHerc1203) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%221203_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
21. [PHerc. 1218](/data_browser/PHerc1218) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%221218_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
22. [PHerc. 1447](/data_browser/PHerc1447) — [8.64µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr%22%2C%22name%22%3A%221447_8.64um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)
23. [PHerc. 1545](/data_browser/PHerc1545) — [9.362µm](https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22zarr2%3A//https%3A//vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1545/volumes/20250821151648-9.362um-1.2m-113keV-masked.zarr%22%2C%22name%22%3A%221545_9.362um%22%2C%22shader%22%3A%22%23uicontrol%20invlerp%20normalized%5Cnvoid%20main%28%29%20%7B%20emitGrayscale%28normalized%28%29%29%3B%20%7D%22%7D%5D%2C%22layout%22%3A%224panel%22%7D)

</details>

**Deadline: June 25th, 2027 (11:59pm Pacific)**

<details>
<summary>Submission criteria and requirements</summary>

**General conditions**

* Pipeline fully reproducible and code shared under an open source license (e.g. MIT), published publicly on GitHub. It does not have to be open source at the time of submission, but you have to make it open source under a permissive license, publicly on GitHub, to accept the prize.
* Pipeline should be seamlessly integrated in the [VC3D](https://github.com/ScrollPrize/villa/tree/main/volume-cartographer) software.
* 100% of the papyrus recto surface unrolled. If the scroll possesses flakes or detached patches, they should also be segmented and unrolled either in a single or in a sequence of separate tifxyz meshes. It is permissible to skip disconnected outer patches if they constitute less than 10% of the total scroll surface.
* Ink detection or renders with ink should be produced from or on top of the flattened images. Columns of text should be visible everywhere. Verify that at least 70% of each counted column's preserved characters are legible. Legible characters only count as legible when identified on a letter-by-letter basis without papyrological interpolation. If text is not displayed, that area will be counted as "non legible" unless a valid explanation is provided for the lack of ink.
* No overlap between training and prediction regions. Overlap leads to the memorization of annotated labels — ink model outputs should not overlap with any training data used.
* In case multiple eligible submissions are concurrently evaluated, a ranking will be determined counting the number of legible characters in the submitted images, abiding by the legibility criteria defined before.
* You may use any information or resource that is publicly available (published scholarship, other scrolls' data, segments, or models, third-party pretrained models, etc.), provided that use is permitted by that resource's own license or terms. Data derived from higher resolution scans of the submitted scroll volume cannot be used.

**Compute and data conditions**

* Unlimited time, compute, and manual effort may be spent creating generic ML datasets to train ML models.
* Any dataset created or used to train ML models must be published publicly, under a CC-BY-NC 4.0 license.
* If pseudo-labeling (or iterative labeling) is used to improve and train ink detection models, the datasets and checkpoints at every stage must be released under a CC-BY-NC 4.0 license.
* If any part of training or inference is stochastic, random seeds must be fixed and reported, for both training and inference.
* For any trained model, the full experiment-tracking run (e.g. Weights & Biases) must be shared publicly, for both training and inference.
* The unrolling pipeline should be fully automated; up to 8 documented hours of human annotation / input are tolerated.

**Submitting your result**

If you have a qualifying result, submit it for consideration by sending an email to [grandprize@scrollprize.org](mailto:grandprize@scrollprize.org) and provide the following:

* **Images.** Submissions must include images of the virtually unwrapped papyrus, showing visible and legible text.
  * Submit a single static image for each column or sequence of consecutive wraps. Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text.
  * Specify which scroll each image came from.
  * Specify where in the scroll they were found: include information about the position of the text vertically as well as radially within the scroll. One easy way to do this is to provide images showing the 3D position of the text surface inside the scroll, or the segmentation file/ID.
  * Include scale bars showing the size of 1cm on each submission image.
* **Methodology.** A detailed technical description of how your solution works. We need to be able to reproduce your work, so please make this as easy as possible:
  * Please create a Docker image that we can easily run to reproduce your work, and please include system requirements.
  * Attach your code/video directly to the email, or include an easily accessible link from which we can download it.
* **Other information.** Feel free to include any other things we should know.

If you're competing as a team, please have your team leader submit your results. We will communicate with the team leader exclusively, and any prize money will be distributed according to the instructions of the team leader. You'd have to sort out within your team how to split any prizes.

**Review process**

All submissions will be assessed by the Review Team, which consists of a Technical Team to review your methodology, and an independent Papyrology Team to review your results.

**1. Technical assessment.** The Technical Team will look at your method, and try to reproduce your results independently. We may also try to apply your techniques to other scrolls to see if they are able to generate new results there.

* We will work with you on reproducing your solution. We might have questions, such as how your code works, how to use your manual tools (if applicable), and so on. Please make it as easy for us to run your code as reasonably possible, but also don't wait until your solution is perfect. If you have any questions, or if you're wondering if you're ready to submit, just reach out!
* We will acknowledge having received your submission within a week. Depending on the difficulty of verifying your methodology, it might take longer until we are able to make our final assessment.
* In case there are multiple teams that submit qualifying results, the team that submitted first will win (independent of how long our assessment takes).

**2. Papyrological assessment.** Once we are reasonably confident that your solution is technically valid and appears to meet the qualifications, we will share your results with the Papyrology Team, who will judge if the text is legitimate and meets the required legibility standards.

**Additional terms**

* To qualify, you must have registered on the [Vesuvius Challenge Discord](https://discord.gg/V4fJhvtaQn) at the time of the submission.
* Do not make your discovery public until winning the prize is officially announced. We will work with you to announce your findings.
* If no team meets the criteria by the deadline, we reserve the right to award the prizes to the teams that came closest. This is not a guarantee — we will only award prizes if we believe the spirit of the prize has substantially been met and if a submission comes very close to the objective threshold. This is entirely at our discretion.
* We will work with the winners to verify their results, put them in a historical context, and co-publish them in academic venues where applicable.
* The general [Terms and Conditions](#terms-and-conditions) at the bottom of this page also apply.

</details>

***

## First Letters Prizes {#first-letters-prizes}

One of the frontiers of Vesuvius Challenge is finding techniques that work across multiple scrolls.
While we’ve discovered text in some of our scrolls, others have not yet produced legible findings.
These prizes bridge ink detection on fragments to the much harder problem of reading intact scrolls: we want to prove that ink detection works on scrolls where nothing has been read yet. The review bar is deliberately high — we’d rather be slow than wrong.

**First Letters: \$50,000 per scroll, for any of the [scroll volumes eligible for the 2027 Grand Prize](#2027-grand-prize).** \$50,000 to the first team that uncovers 10 letters within a single 4 cm² area of that scroll — and open sources their methods and results (after winning the prize). First Letters prizes will be awarded for a maximum of 10 scrolls — up to \$500,000 in total.

**Deadline: June 25th, 2027 (11:59pm Pacific)**

<details>
<summary>Submission criteria and requirements</summary>

* **Image.** Submissions must be an image of the virtually unwrapped segment, showing at least 10 visible and legible letters within a single 4 cm² area.
  * Submit a single static image showing the text region. Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text. This includes annotations that were then used as training data and memorized by a machine learning ink model. Ink model outputs of this region should not overlap with any training data used.
  * Specify which scroll the image comes from. For multiple scrolls, please make multiple submissions.
  * Include a scale bar showing the size of 1 cm on the submission image, and the pixel and millimeter dimensions of a few representative letters.
  * Specify the 3D position of the text within the scroll: where it sits, whether the surface faces inward or outward, and which way is "up". A 3D orientation image is the easiest way to show this — or provide the segmentation file (or the segmentation ID, if using a public segmentation).
  * Annotate the rows of text. Letters in read samples run overwhelmingly parallel to the horizontal papyrus fibers — where possible, overlay your ink predictions on a fiber-visible rendering. Misaligned text or text without clear rows does not immediately disqualify a submission, but it does make it less likely that you found valid text.
* **Methodology.** A detailed technical description of how your solution works. We need to be able to reproduce your work, so please make this as easy as possible:
  * For fully automated software, consider a Docker image that we can easily run to reproduce your work, and please include system requirements.
  * For software with a human in the loop, please provide written instructions and a video explaining how to use your tool. We’ll work with you to learn how to use it, but we’d like to have a strong starting point.
  * Please include an easily accessible link from which we can download it.
* **Hallucination mitigation.** If there is any risk of your model hallucinating results, please let us know how you mitigated that risk. Tell us why you are confident that the results you are getting are real.
  * We strongly discourage submissions that use window sizes larger than 0.5x0.5 mm to generate images from machine learning models. This corresponds to 64x64 pixels for 8 µm scans. If your submission uses larger window sizes, we may reject it and ask you to modify and resubmit.
  * In addition to hallucination mitigation, do not include overlap between training and prediction regions. This leads to the memorization of annotated labels.
* **Held-out validation.** Run your method on the public scroll fragments with known ground truth (using k-fold validation if you trained on them) and include the results. We may also run your method, following your instructions, on held-out data with known ground truth.
* **Other information.** Feel free to include any other things we should know.

Your submission will be reviewed by the review teams to verify technical validity and papyrological plausibility and legibility.
Submissions remain open until the prize is won: if we discover months from now that your method was right all along, you will then win.
Just as with the Grand Prize, please **do not** make your discovery public until winning the prize. We will work with you to announce your findings.
</details>

[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform?usp=dialog)

***

## PHerc. Paris 4’s Title Prize {#first-title-prize}

Discovering a scroll’s title tells scholars what — and whom — they have been reading, and helps contextualize the entire work. We have already recovered the titles of [PHerc. 172](/data_browser/PHerc0172) and [PHerc. 139](/data_browser/PHerc0139) — but the title of Scroll 1, the scroll where the first passages were read, is still missing.

**PHerc. Paris 4’s Title: \$50,000 to the first team to discover the title of [PHerc. Paris 4](/data_browser/PHercParis4) (Scroll 1), using any of its scans — including the 2.4 µm volumes.** Scroll 1 is one of our most-read scrolls: substantial continuous Greek text of an Epicurean prose work has been recovered, yet its author and title remain unknown. The expected title region has shown no detectable ink so far — possibly a different ink, and the top rows are physically missing — so finding it may take better methods, higher resolution, or looking somewhere new.

**Deadline: June 25th, 2027 (11:59pm Pacific)**

<div className="mb-4">
  <img src="/img/data/title_example.webp" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">Visible Title in a Scroll Fragment.</figcaption>
</div>

<details>
<summary>Submission criteria and requirements</summary>

* **Image.** Submissions must be an image of the virtually unwrapped region, showing the title visibly and legibly.
  * Illustrate the ink predictions in the spatial context of the title search, similar to what is [shown here](https://scrollprize.substack.com/p/30k-first-title-prize). You **do not** have to read the title yourself — you have to produce an image of it that our team of papyrologists is able to read.
  * Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text. Ink model outputs of this region should not overlap with any training data used.
  * Specify which scan the image comes from — any of Scroll 1’s published volumes qualifies, including the 2.4 µm ones.
  * Include a scale bar showing the size of 1 cm on the submission image, and the pixel and millimeter dimensions of a few representative letters.
  * Specify the 3D position of the title within the scroll: where it sits, whether the surface faces inward or outward, and which way is "up". A 3D orientation image is the easiest way to show this — or provide the segmentation file (or the segmentation ID, if using a public segmentation).
* **Methodology.** A detailed technical description of how your solution works. We need to be able to reproduce your work, so please make this as easy as possible:
  * For fully automated software, consider a Docker image that we can easily run to reproduce your work, and please include system requirements.
  * For software with a human in the loop, please provide written instructions and a video explaining how to use your tool. We’ll work with you to learn how to use it, but we’d like to have a strong starting point.
  * Please include an easily accessible link from which we can download it.
* **Hallucination mitigation.** If there is any risk of your model hallucinating results, please let us know how you mitigated that risk. Tell us why you are confident that the results you are getting are real.
  * We strongly discourage submissions that use window sizes larger than 0.5x0.5 mm to generate images from machine learning models. If your submission uses larger window sizes, we may reject it and ask you to modify and resubmit.
  * In addition to hallucination mitigation, do not include overlap between training and prediction regions. This leads to the memorization of annotated labels.
* **Held-out validation.** Run your method on the public scroll fragments with known ground truth (using k-fold validation if you trained on them) and include the results. We may also run your method, following your instructions, on held-out data with known ground truth.
* **Other information.** Feel free to include any other things we should know.

Your submission will be reviewed by the review teams to verify technical validity and papyrological plausibility and legibility.
Submissions remain open until the prize is won: if we discover months from now that your method was right all along, you will then win.
Just as with the Grand Prize, please **do not** make your discovery public until winning the prize. We will work with you to announce your findings.
</details>

[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform?usp=dialog)

***

:::warning
The previous prizes are too ambitious? You can still contribute!
:::

## Progress Prizes

In addition to milestone-based prizes, we offer monthly prizes for open source contributions that help read the scrolls.
These prizes are more open-ended, and we have a wishlist to provide some ideas.
If you are new to the project, this is a great place to start.

**Best Submission of the Month: \$20,000, guaranteed every month, to the single best submission — selected by the Vesuvius Challenge team.**

Beyond that, progress prizes will be awarded at a range of levels based on the contribution:

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

Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, July 31st, 2026!

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

[Submission Form](https://forms.gle/Sy6mW5cfJS2U7E9F7)

***

## Terms and Conditions

Prizes are awarded at the sole discretion of Scroll Prize, Inc. and are subject to review by our Technical Team, Annotation Team, and Papyrological Team. We may issue more or fewer awards based on the spirit of the prize and the received submissions. You agree to make your method open source if you win a prize. It does not have to be open source at the time of submission, but you have to make it open source under a permissive license to accept the prize. Submissions for milestone prizes will close once the winner is announced and their methods are open sourced. Scroll Prize, Inc. reserves the right to modify prize terms at any time in order to more accurately reflect the spirit of the prize as designed. Prize winner must provide payment information to Scroll Prize, Inc. within 30 days of prize announcement to receive prize.
