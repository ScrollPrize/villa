---
title: "$700k/$100k/$50k Grand Prize (Dec 31)"
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


The \$700,000 Grand Prize will go to the first team to read four passages of text from the inside of the two intact scrolls. We also have prizes for second place (\$100,000) and third place (\$50,000). More details on the qualifying criteria are available [here](#qualifying-criteria).

Here are the scrolls in question:

<div className="flex w-[100%]">
  <div className="w-[100%] mb-2 mr-2"><img src="/img/overview/scroll1-small-actual.webp" className="w-[100%]"/></div>
  <div className="w-[100%] mb-2"><img src="/img/overview/scroll2-small-actual.webp" className="w-[100%]"/></div>
</div>
<div className="flex w-[100%]">
  <div className="w-[39.2%] mb-2 mr-2"><img src="/img/overview/PHerc332.webp" className="w-[100%]"/></div>
  <div className="w-[60.8%] mb-2"><img src="/img/overview/PHerc1667.webp" className="w-[100%]"/></div>
</div>

We have provided you with 3D X-ray scans of each of these scrolls, which you can find [here](/data). Your job is to extract the text from these scans.

You can approach this challenge through any means necessary: machine learning, computer vision, or machine-assisted tools operated by humans.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/overview/scroll-inside-animation-4.webp">
    <source src="/img/overview/scroll-inside-animation-4.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-0">3D X-ray scan of a scroll <a href="https://www.youtube.com/watch?v=PpNq2cFotyY">(source)</a></figcaption>
</figure>

#### Why is this difficult?

As you will read in the [tutorials](/tutorial), advanced tools and techniques exist for virtually unwrapping papyrus scrolls. This was demonstrated in 2015 when Dr. Seales's team [unwrapped the En-Gedi scroll](https://www2.cs.uky.edu/dri/the-scroll-from-en-gedi/), and in their recent result [identifying ink from 3D X-ray scans in the Herculaneum scrolls](https://arxiv.org/abs/2304.02084).

<div>But the Herculaneum scrolls have proved more challenging. The remaining challenges include:</div>

- Segmenting the scrolls. The Herculaneum scrolls are especially long, tightly wrapped, damaged, and distorted. To date, no one has successfully done a large-scale segmentation of these scrolls to identify the surfaces of all the rolled layers.
- Finding the ink. The ink used in the Herculaneum scrolls is [radiolucent](https://en.wikipedia.org/wiki/Radiodensity), making it difficult to see in the scans. Recently, Dr. Seales's team has trained a machine learning model which can detect the ink from subtle patterns in the 3D X-rays. This works in the fragments, but these models are not yet perfect and will probably need to be improved to work at the scale of an entire scroll.
- Putting it all together. Applying the ink detection models to the segmented scroll has not yet been successfully demonstrated.

Based on the [landmark results](https://arxiv.org/abs/2304.02084) that Dr. Seales and his team have recently produced, we believe that it is possible to read the Herculaneum scrolls using the [scans](/data) that we already have and the tools and techniques that they have developed. And that is Vesuvius Challenge!

The Grand Prize deadline is 11:59pm Pacific, December 31st, 2023.

### Qualifying criteria

<div>A <a href="#review-process">Review Team</a> made up of technical experts and <a href="https://en.wikipedia.org/wiki/Papyrology">papyrologists</a> will assess all Grand Prize submissions to ensure that they can:</div>

- Read at least 4 passages from the available full-scroll data, each containing at least 140 characters of contiguous text (e.g. within the same column)
- Verify that each passage contains no more than 15% of characters which are missing or illegible
- The 140 characters per passage include the 15% of characters which may be missing or illegible, so 119 characters must be legible. Legible characters only count as legible when identified on a letter-by-letter basis without papyrological interpolation.
- Confirm that submissions contain legitimate and linguistically plausible text.
- Independently reproduce and verify your results using your code and documented techniques.

If no team meets the criteria by the deadline, we reserve the right to award the prizes to the teams that came closest. **This is not a guarantee** — we will only award prizes if we believe the spirit of the prize has substantially been met and if a submission comes very close to the objective threshold. This is entirely at our discretion. If you are very very close to meeting the bar, we encourage you to submit your work before the deadline.

<div>Prizes are awarded in the order that qualifying submisisons are made:</div>

- 1st place: \$700,000
- 2nd place: \$100,000
- 3rd place: \$50,000

To qualify for the Grand Prize, you must have registered at the time you downloaded the data. The registration form can be found on the [data page](/data).

We will work with the Grand Prize winners to verify their results, put them in a historical context, and co-publish them in academic venues. The winning code will be made public under a permissive open source license, so that others can reproduce and build on your work.

### Submitting your result

<div>If you have a qualifying result, submit it for consideration by sending an email to <a href="mailto:grandprize@scrollprize.org">grandprize@scrollprize.org</a> and provide the following:</div>

* **Images.** Submissions must be in the form of images of the virtually unwrapped papyrus, showing visible and legible text.
    * Submit a single static image for each text region. Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text.
    * Specify which scroll each image came from.
    * Specify where in the scroll they were found: include information about the position of the text vertically as well as radially within the scroll. One easy way to do this is to provide images showing the 3D position of the text surface inside the scroll.
    * Include scale bars showing the size of 1cm on each submission image.
* **Methodology.** A detailed technical description of how your solution works. We need to be able to reproduce your work, so please make this as easy as possible:
    * For fully automated software, please create a Docker image that we can easily run to reproduce your work, and please include system requirements.
    * For software with a human in the loop, please provide written instructions and a video explaining how to use your tool. We’ll work with you to learn how to use it, but we’d like to have a strong starting point. It should take us no more than an hour to perform the human tasks to get your software working.
    * Either attach your code/video directly to the email, or include an easily accessible link from which we can download it.
* **Hallucination mitigation.**
    * If there is any risk of your model hallucinating results, please let us know how you mitigated that risk. Tell us why you are confident that the results you are getting are real.
    * We strongly discourage submissions that use window sizes larger than 0.5x0.5 mm to generate images from machine learning models. This corresponds to 64x64 pixels for 8 µm scans. If your submission uses larger window sizes, we may reject it and ask you to modify and resubmit.
* **Other information.** Feel free to include any other things we should know.

If you’re competing as a team, please have your team leader submit your results. We will communicate with the team leader exclusively, and any prize money will be distributed according to the instructions of the team leader. You’d have to sort out within your team how to split any prizes.

### Review process

All submissions will be assessed by the Review Team, which consists of a technical team to review your methodology, and an independent team of papyrologists to review your results. Note that the technical team also judges the Open Source Prizes.

<div className="font-bold">Technical team</div>

- Dr. Brent Seales
- Nat Friedman
- Dr. Stephen Parsons
- Seth Parker
- JP Posma
- Daniel Havíř

<div className="font-bold">Papyrology Team</div>

- Robert Fowler, Fellow of the British Academy; Professor Emeritus of Classics, Bristol University
- Tobias Reinhardt, Corpus Christi Professor of the Latin Language and Literature, Oxford
- Richard Janko, Professor of Classics, University of Michigan
- Federica Nicolardi, Professor of Classics, University of Naples Federico II
- Gianluca Del Mastro, Professor of Papyrology, l’Università della Campania «L. Vanvitelli»
- Daniel Delattre, Emeritus Research Director and Papyrologist, CNRS and IRHT

We will process your submission as follows:

<div><strong>1. Technical assessment.</strong> The technical team will look at your method, and try to reproduce your results independently. We may also try to apply your techniques to the held-back part of the scrolls to see if they are able to generate new results there.</div>

  * We want to make sure that your method doesn’t hallucinate, so please let us know how you mitigated this risk.
  * We will work with you on reproducing your solution. We might have questions, such as how your code works, how to use your manual tools (if applicable), and so on. Please make it as easy for us to run your code as reasonably possible, but also don’t wait until your solution is perfect. If you have any questions, or if you’re wondering if you’re ready to submit, just reach out!
  * We will acknowledge having received your submission within a week. Depending on the difficulty of verifying your methodology, it might take longer until we are able to make our final assessment.
  * In case there are multiple teams that submit qualifying results, the team that submitted first will win (independent of how long our assessment takes).

<div><strong>2. Papyrological assessment.</strong> Once we are reasonably confident that your solution is technically valid and appears to meet the qualifications, we will share your results with the papyrology team, who will judge if the text is plausible and legitimate.</div>
