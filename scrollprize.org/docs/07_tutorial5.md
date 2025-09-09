---
title: "Tutorial: Ink Detection"
sidebar_label: "Ink Detection"
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

import { TutorialsTop } from '@site/src/components/TutorialsTop';

<TutorialsTop highlightId={5} />

This tutorial gives a high-level overview on ink detection methods.
Since this tutorial was written, ink detection has successfully recovered text from inside the rolled scrolls.
The technical principles remain the same as what is described here.

The tutorial can be followed by the more hands-on notebooks [on Kaggle](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial) and [a more recent version on Colab](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example2_ink_detection.ipynb).

Ink detection is the task of taking data from a 3D X-ray scan around a papyrus surface, and identifying the locations of the inked parts of the papyrus.

This is where one of the difficulties of the Herculaneum Papyri comes in: the ink and the papyrus have very similar densities, making it hard to detect ink in 3D X-ray scans.

* **Campfire & En-Gedi scrolls:** Dense ink shows up as brighter voxels in 3D X-ray scans, so ink detection can be done by taking the brightest pixel in some voxel region.
* **Herculaneum scrolls & fragments:** Ink is less directly visible in 3D X-ray scans, but there does seem to be data there. Machine learning models can detect it, and humans can sometimes see subtle textural patterns in the data.

In the video below Dr. Seales talks about how ink detection got started for the Herculaneum scrolls:

<iframe className="w-[100%] aspect-video mb-4" src="https://www.youtube.com/embed/g-7-Xg75CCI?start=5930" title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen/>

Not only can machine learning models detect the ink, on occasion we can see the ink directly in the 3D X-ray volumes. Here are some examples, with slices from the 3D surface volumes on the left, and infrared photos showing ink on the right (from the [data paper](https://arxiv.org/abs/2304.02084)):

<figure className="">
  <img src="/img/tutorials/ink2-alpha.webp" />
</figure>

<figure className="">
  <img src="/img/tutorials/ink1-alpha.webp" />

  <figcaption className="mt-0">Ink visible in 3D surface volumes (left: 3D volume slice; right: infrared photo), found by Stephen Parsons</figcaption>
</figure>

You have to look closely, but the shapes are visible!

Discoveries from the community have found widespread examples of visible ink like this inside the intact scrolls. In particular the [“crackle pattern”](https://caseyhandmer.wordpress.com/2023/08/05/reading-ancient-scrolls/) discovered by Casey Handmer has proven useful, and inspired a number of labeling approaches that successfully produce models capable of detecting ink in the scrolls.

For the purposes of the tutorial, we will use the fragment datasets, which contain ground truth ink labels made using infrared photography of the exposed writing on the surface.

At a high level, training on a fragment works like this:

<figure className="">
  <img src="/img/tutorials/ml-overview-alpha.webp" />
</figure>

From a fragment (a) we obtain a 3D volume (b), from which we segment a mesh (c), around which we sample a surface volume (d). We also take an infrared photo (e) of the fragment, which we align (f) with the surface volume, and then manually turn into a binary label image (g).

We train this model by picking a pixel in the binary label image, and sampling a subvolume around the same coordinates from the surface volume. We then backpropagate the known label data to update the model weights:

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] " poster="/img/tutorials/ink-training-anim3-dark.webp">
    <source src="/img/tutorials/ink-training-anim3-dark.webm" type="video/webm"/>
  </video>
</figure>

We can then use the model to predict what a label image would have looked like, from different input data than you have trained on.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/ink-detection-anim3-dark.webp">
    <source src="/img/tutorials/ink-detection-anim3-dark.webm" type="video/webm"/>
  </video>
</figure>

Of course, in reality the label image on the right doesn’t come out perfectly. Stephen Parsons’ [ink-id](https://github.com/educelab/ink-id/) program is one example of an ML-based approach. It produces outputs like this (showing different training epochs in k-fold training/prediction):

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] max-w-[238px]" poster="/img/landing/fragment-training2.webp">
    <source src="/img/landing/fragment-training2.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-0">A model learning to detect ink on a fragment, showing different training epochs</figcaption>
</figure>

When running ink-id on all the public fragments, the results look like this (prediction left, infrared right):

<figure className="">
  <table className="w-[100%] max-w-[400px]">
    <tr><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f1_composite.webp" /></td><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f1_ir.webp" /></td></tr>
    <tr><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f2_composite.webp" /></td><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f2_ir.webp" /></td></tr>
    <tr><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f3_composite.webp" /></td><td className="w-50%"><img className="w-[100%]" src="/img/tutorials/f3_ir.webp" /></td></tr>
  </table>
  <figcaption className="mt-0">Predicted label images from ink-id (left); infrared photos (right)</figcaption>
</figure>

As you can see, some letters can be clearly seen, others not at all, and a lot of letters are somewhere in between. All fragments also have “hidden layers”: pieces of papyrus that are fused to the backs of the fragments. Running the machine model on those reveals some previously unseen letters:

<div className="flex flex-wrap items-end max-w-[500px] mb-4">
  <figure className="w-[33%]">
    <img src="/img/tutorials/f1_hidden_composite.webp"/>
  </figure>
  <figure className="w-[33%]">
    <img src="/img/tutorials/f2_hidden_composite.webp"/>
  </figure>
  <figure className="w-[33%]">
    <img src="/img/tutorials/f4_hidden_composite_partially_redacted.webp"/>
  </figure>
  <figcaption className="mt-0">“Hidden layers” of papyrus, partially revealed by machine learning.</figcaption>
</div>

The [Ink Detection Progress Prize on Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/) was all about creating the best possible machine learning model for detecting ink within the fragments. Since then newer models have successfully uncovered ink in full scrolls (the [First Letters Prize](firstletters), and then the [2023 Grand Prize](grandprize)).

So how can a machine learning model detect ink? In the electron microscope images below (from the paper [From invisibility to readability: Recovering the ink of Herculaneum](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0215775&type=printable)), you can clearly see the difference between the inked and non-inked regions. We suspect that machine learning models are able to learn some of these features from the 3D X-ray scans.

<figure>
  <a href="/img/tutorials/sem.webp" target="_blank"><img src="/img/tutorials/sem-alpha.webp"  className="w-[100%]"/></a>
  <figcaption className="mt-0">Electron microscope pictures from the top (A and B) and the side (C) <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775">(source)</a></figcaption>
</figure>

<div>The main challenges for ink detection are:</div>

* Model performance, getting more letters to be legible.
* Applying these models to the full scrolls.
* Reverse engineering the models to better understand the kind of patterns they are using to detect ink.
* Creating more ground truth data (e.g. “campfire scrolls” or synthetic data).

Now let’s create a model! This part of the tutorial is over [on Kaggle as a notebook](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial).

To run more advanced models on the [scroll segments](data_segments), check out the winning code from the [First Letters Prize](firstletters) and the [2023 Grand Prize](grandprize).
