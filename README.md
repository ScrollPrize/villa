# 🏛️ villa

[![MIT licensed][licence-badge]][licence-url]
[![Stars][stars-badge]][stars-url]
[![Discord chat][discord-badge]][discord-url]

Vesuvius Challenge is a machine learning and computer vision competition to read the Herculaneum scrolls.  
This repository contains the source code for Vesuvius Challenge: scroll tools, libraries, our website, data processing scripts, and [more](https://github.com/ScrollPrize/villa/blob/main/scrollprize.org/docs/20_community_projects.md).

---

## Libraries for Accessing Scrolls

### [vesuvius](vesuvius/)
A Python library for accessing CT scans of ancient scrolls.

---

## Dataset & Rendering Tools

### [foundation](foundation/)
Tools to build/manage scroll datasets and cloud infrastructure.

### [ink-detection](ink-detection/)
Training and inference tools for Vesuvius Challenge ink detection models, built on the model that won the 2023 Grand Prize.  
Originally developed by [Youssef Nader](https://github.com/younader) and [Luke Farritor](https://github.com/lukeboi), extended by the Vesuvius Challenge team.

---

## Automatic Unwrapping (Segmentation) Pipelines

### [VC3D (surface tracer)](volume-cartographer)
A semi-automatic segmentation pipeline to extract papyrus sheets from CT scans of ancient scrolls.
As of September 2025, it is the approach used by the Vesuvius Challenge team.
Developed by [Hendrik Schilling](https://github.com/hendrikschilling) and [Sean Johnson](https://github.com/bruniss) as a fork of [Volume Cartographer](https://github.com/educelab/volume-cartographer).

### [lasagna](lasagna/)
A PyTorch-based optimization framework for growing and refining papyrus surface meshes, offered as an alternative to VC3D's GrowPatch. It can jointly optimize several stacked sheets so they stay consistent with each other, and also drives fiber tracing.

### [spiral fitting](volume-cartographer/scripts/spiral)
A fully automatic unwrapping pipeline that fits a single, globally coherent spiral to an entire scroll by deforming an ideal spiral to match traced patches, fiber skeletons, and winding annotations.
Originally developed by [Paul Henderson](https://github.com/pmh47).

---

## Supporting Resources

### [scrollprize.org](scrollprize.org/)
Source for the [Vesuvius Challenge website](https://scrollprize.org).

---

## Deprecated

### [thaumato-anakalyptor](deprecated/thaumato-anakalyptor/)
A semi-automatic segmentation pipeline to extract papyrus sheets from CT scans of ancient scrolls.  
Originally developed by [Julian Schilliger](https://github.com/schillij95). Superseded by [VC3D](volume-cartographer) and [lasagna](lasagna/).

### [vesuvius-c](deprecated/vesuvius-c/)
A single-header C library for accessing CT scans of ancient scrolls.

### [crackle-viewer](deprecated/crackle-viewer/)
A GUI tool to inspect and label ink on virtually unwrapped segments of ancient scrolls.  
Originally developed by [Julian Schilliger](https://github.com/schillij95).

[licence-badge]: https://img.shields.io/github/license/ScrollPrize/villa?color=blue
[licence-url]: https://github.com/ScrollPrize/villa/blob/main/LICENSE
[stars-badge]: https://img.shields.io/github/stars/ScrollPrize/villa?style=social
[stars-url]: https://github.com/ScrollPrize/villa/stargazers
[discord-badge]: https://img.shields.io/discord/1079907749569237093.svg?logo=discord&style=flat-square
[discord-url]: https://discord.com/invite/V4fJhvtaQn
