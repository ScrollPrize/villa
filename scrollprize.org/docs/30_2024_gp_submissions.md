---
title: "2024 Grand Prize Submission Instructions"
hide_table_of_contents: true
slug: "2024_gp_submissions"
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


### Submitting your result

<div>If you have a qualifying result, submit it for consideration by sending an email to <a href="mailto:grandprize@scrollprize.org">grandprize@scrollprize.org</a> and provide the following:</div>

* **Images.** Submissions must be in the form of images of the virtually unwrapped papyrus, showing visible and legible text.
    * Submit a single static image for each scroll. If necessary, the image can be tiled to reasonable dimensions. Images must be generated programmatically, as direct outputs of CT data inputs, and should not contain manual annotations of characters or text.
    * Specify which scroll each image came from.
    * Specify where in the scroll they were found: include information about the position of the text vertically as well as radially within the scroll. One easy way to do this is to provide images showing the 3D position of the text surface inside the scroll.
    * Include metric scale bars on each submission image.
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

All submissions will be assessed by the Review Team, which consists of a technical team to review your methodology, and an independent team of papyrologists to review your results. Note that the Review Team also judges the progress prizes.

We will process your submission as follows:

<div><strong>1. Technical assessment.</strong> The technical team will look at your method, and try to reproduce your results independently.</div>

  * We want to make sure that your method doesn’t hallucinate, so please let us know how you mitigated this risk.
  * We will work with you on reproducing your solution. We might have questions, such as how your code works, how to use your manual tools (if applicable), and so on. Please make it as easy for us to run your code as reasonably possible, but also don’t wait until your solution is perfect. If you have any questions, or if you’re wondering if you’re ready to submit, just reach out!
  * We will acknowledge having received your submission within a week. Depending on the difficulty of verifying your methodology, it might take longer until we are able to make our final assessment.

<div><strong>2. Papyrological assessment.</strong> Once we are reasonably confident that your solution is technically valid and appears to meet the qualifications, we will share your results with the papyrology team, who will proceed with papyrological evaluation.</div>
