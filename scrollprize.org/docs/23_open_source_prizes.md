---
title: "$12k Open Source Prizes (Dec 31)"
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

In our tradition of progress prizes, we’re awarding four \$3,000 prizes for qualifying submissions by Dec 31st. This time, we do not award prizes specific to segmentation or ink detection. Anything that increases the probability of reading the scrolls this year qualifies.

<div>There are some conditions:</div>

* Your submission must substantially increase the probability of reading the scrolls this year, as judged by the Review Team. We may award more or fewer prizes at our discretion, depending on the number of qualifying submissions.
* Your submission must be open source.
* We are heavily favoring submissions that:
  * Are released <span className="underline">early</span>. Tools released tomorrow have a higher chance of being used for reading the scrolls than those released a day before the deadline.
  * Actually <span className="underline">get used</span>. We’ll look for signals from the community: questions, comments, bug reports, feature requests. Our Segmentation Team will publicly provide comments on tools they use.
* Submissions are closed on December 31st 11:59pm PT, after which the Review Team will select winners.
* Make your submission using [this form](https://forms.gle/A8eDKEJ8Xtm2J9u38).

[Segmentation](data_segments) is still a big focus, since for the Grand Prize we need multiple large segments. Large enough to contain continuous passages of at least 140 characters.

<div>Some ideas to get you started:</div>

* The Segmentation Team has put together a list of [feature requests](https://docs.google.com/document/d/1YFILhWVHyijU_Yky3lKPvGAjmYm2QnRTYzMM7VqcogA/edit) — be sure to tag @Hari_Seldon on Discord if you’re going to work on this. These include:
  * Many feature requests for Volume Cartographer (we’ve been using [@spacegaier’s fork](https://github.com/spacegaier/volume-cartographer/))
  * Integration of ink detection models to help see if an area was segmented correctly
  * Real time papyrus visualization like Khartes
  * Segmenting “at an angle” for tilted papyrus surfaces
* Tools for detecting overlap in segments, to avoid unwittingly running training and inference on the same data.
* Tools for merging segments or laying out multiple related segments in 2D (like Marzia D’Angelo’s [fragment map](faq#books); more explanation [here](https://discord.com/channels/1079907749569237093/1085972686158712892/1115918553602855002)).
* Auto-segmentation of “mushy” areas: algorithms / models that can detect and follow the papyrus fibers, even where they are bunched up with other layers of papyrus.
* Comparison of various ink detection model architectures.
* Anything that comes up in Discord that the community broadly deems useful.

<figure className="max-w-[600px]">
  <img src="/img/faq/marzia.webp" className="w-[100%]"/>
  <figcaption className="mt-0">Marzia D’Angelo showing her fragment map — can we make something like this for our segments?</figcaption>
</figure>
