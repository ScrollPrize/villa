---
title: "Jobs"
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

# Jobs

We are laser focused on our mission of reading the Herculaneum scrolls.
If you can help us achieve this goal, please reach out to jobs@scrollprize.org.
In addition, we are hiring for the specific roles listed below.

***

<TOCInline
  toc={toc}
/>

***

## Geometry and Computer Vision Applied Researcher

The Vesuvius Challenge applied researchers advance the frontier of the most promising research directions from our community, furthering progress on the most promising research avenues in virtual unwrapping and related problems.

#### Overview
If the Vesuvius Challenge community performs a breadth-first search of research ideas to help us read the scrolls, this role performs depth-first search. We (and you!) are laser focused on solving the remaining hurdles to extract these hidden texts from their scroll confines.

We process volumetric CT data of scrolls, and are working to solve a challenging segmentation problem (mapping the surface of the crumpled scroll within the volume) as well as ink detection (using machine learning to detect the subtle presence of ink within the scan). This role will pursue the most promising methods on the data, adapt to follow emerging directions, and share the results along the way with our technical community.

We are looking for individuals who can review the [Virtual Unwrapping](unwrapping) problem overview and contribute concrete and actionable research ideas.

#### Example tasks might include:
- Implement optimization algorithms to fit 3D surfaces to complex data
- Process high-dimensional images using classical and machine learning methods
- Optimize data formats for storage space, accuracy and accessibility
- Implement and train neural networks for segmentation of 3D image data
- Process non-watertight manifold triangular meshes and other geometrical objects
- Optimize existing tools to run faster or with fewer resources on large image datasets
- Polish and publish community-developed proofs of concept
- Perform deformable registration of 3D objects
- Analyze and validate pipeline steps with complex metrics, and define those metrics

#### Qualifications:
- Image/geometry processing experience, ideally with large or high dimensional datasets
- Experience implementing and training neural networks
- Generalist/excited to pursue flexible directions
- Previous research experience in a related domain (PhD-level, but PhD not required)

#### To Apply
Send your resume, cover letter, and examples of relevant projects to jobs@scrollprize.org. Include "Applied Researcher" in the subject line.

***

## Synchrotron Tomography Reconstruction Expert

We seek an expert in synchrotron tomography reconstruction to help us scan and process hundreds of Herculaneum scrolls. This role will be critical in developing and maintaining our acquisition protocol and reconstruction pipeline to handle the petabytes of CT scan data we expect to process.

#### Overview
At Vesuvius Challenge we aim to scan the full collection of Herculaneum papyri found thus far. This includes hundreds of scrolls, whose images need to be acquired at beamline synchrotron facilities and reconstructed via computed tomography.

We need help scanning and reconstructing the images of the scrolls. Specifically, we seek a Synchrotron Tomography Reconstruction Expert to develop and maintain a robust, high-throughput, and facility-agnostic tomographic reconstruction pipeline. The expert will integrate open-source toolkits with in-house automation scripts to streamline data processing—from raw projections through phase retrieval to final reconstructions.

#### Responsibilities include:
- Develop and optimize an automated pipeline for X-ray tomography reconstruction (including phase contrast imaging) using open-source libraries
- Run the pipeline to reconstruct the images of hundreds of Herculaneum papyri in a short time-frame
- Ensure synchrotron/beamline-invariant compatibility: adapt metadata and geometry definitions from multiple facilities into a uniform pipeline
- Collaborate with beamline scientists and domain experts to validate reconstruction quality and optimize parameters (e.g., phase retrieval, artifact correction, center-of-rotation finding)
- Integrate HPC and GPU acceleration
- Deploy and maintain the pipeline on cloud infrastructure
- Participate in scan sessions to ensure the acquisition of high-quality data
- Write comprehensive documentation and provide user training/support both for internal researchers and our online community

#### Ideal qualifications:
- Direct synchrotron beamline experience
- MSc or PhD in Experimental Physics, Applied Mathematics, CS or a related field
- Experience with tomographic software frameworks
- Experience in scientific computing, HPC, and/or GPU programming
- Exposure to cloud computing

#### To Apply
Send your resume, cover letter, and examples of relevant projects to jobs@scrollprize.org. Include "Reconstruction Expert" in the subject line.

***

## Platform Engineer

The Vesuvius Challenge platform team builds the tools that make researchers and contestants more productive. This includes libraries, visualizations, annotation tools, and data organization & accessibility.

#### Overview
Vesuvius Challenge produces and maintains multi-terabyte CT scans of ancient scrolls, with the goal of multiplying this dataset significantly in the coming year. We serve this data to a technical community around the world, and also support an in-house research team.

Presently, data is generated and organized on an ad hoc basis, and requires tedious and large downloads to access. We aim to make the datasets more easily accessible to our team and community via strengthened organizational schemas and streaming them via libraries.

#### Responsibilities include:
- Develop automated tools for:
  - Existing photogrammetry workflows
  - 3D scroll case design
  - CT data reconstruction and image post-processing
  - Data transfer and organization
- Maintain libraries to stream and access data in C++, C and Python
- Perform semantic and instance segmentation of 3D images
- Provide easy access to trained models (semantic segmentation and ink detection)
  - Available via HuggingFace and/or our libraries
- Enable easy and accessible visualization of raw data and derived formats
- Enhance annotation tools/GUIs to support data annotation team 
- Integrate components of software pipelines into cloud infrastructure, enabling on-demand execution

#### Qualifications:
- Appreciates well-structured schemas in multi-terabyte datasets 
- Experience shipping finished products
- Fluent in basic cloud infrastructure and comfortable with basic image/geometry processing
- Experienced in C++ and/or Python
- Experienced building web frontends

#### To Apply
Send your resume, cover letter, and examples of relevant projects to jobs@scrollprize.org. Include "Platform Engineer" in the subject line.

***

## Annotation Specialist 

Help us unravel the mysteries of the only remaining library from antiquity! 

#### Role
As an Annotation Specialist you will analyze and process volumetric CT scan data from ancient scrolls, which were carbonized and buried at Herculaneum two thousand years ago during the eruption of Mt. Vesuvius. 

#### Overview
The most important requirement is the capacity to sustain a high level of engagement in complex repetitive tasks. Computer literacy and experience looking at data are strongly suggested, and direct experience in medical annotation or similar is a bonus. Scroll annotation has many flavors in 2D and 3D, and will be an evolving landscape. 

A good fit comes down to personality as much as skill sets. Maybe you have strong gaming skills, or rebuild vintage watches as a hobby. If fast-paced repetitive complex tasks sound relaxing, this may be for you! 

This is not a programming position, and the majority of your time will be spent in the trenches annotating papyrus. However, this is active research, so experience writing code, beta-testing and debugging software, algorithm development, graphics, general data science, visual art and other creative skills will all be of great benefit. 

This is a full-time position (30+ hours/week). Pay rate up to $40 per hour.

#### Key Responsibilities
- Master several techniques for annotating X-ray CT scans of scrolls with evolving software
- Collaborate with researchers to develop and test new annotation tooling 
- Document processes and maintain consistent standards
- Ensure access to a fast internet connection

#### Required Skills 
- Capacity to sustain a high level of engagement in complex repetitive tasks
- Attention to detail
- Excellent communication skills
- Familiarity with basic Linux command line operations 

#### Preferred Qualifications
- Experience with volumetric image segmentation and annotation
- Experience beta-testing and debugging software
- Graphics and algorithm development
- C/C++, Python

#### Impact
Not only will you unwrap ancient scrolls and be the first human to lay eyes on the resulting text in two millennia, your efforts will directly contribute to automation of unwrapping, which is a key ingredient to help catalyze further excavation of Herculaneum and unearth the only remaining library from antiquity before it is too late! 

#### To Apply
Send your resume, cover letter, and examples of relevant projects to jobs@scrollprize.org. Include "Annotation Specialist" in the subject line.

***
