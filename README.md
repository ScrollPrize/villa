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

### [vesuvius-c](vesuvius-c/)
A single-header C library for accessing CT scans of ancient scrolls.

---

## Dataset & Rendering Tools

### [foundation](foundation/)
Tools to build/manage scroll datasets and cloud infrastructure.

### [crackle-viewer](crackle-viewer)
A GUI tool to inspect and label ink on virtually unwrapped segments of ancient scrolls.  
Originally developed by [Julian Schilliger](https://github.com/schillij95).

### [ink-detection](ink-detection/)
The ink detection machine learning model used to win the Vesuvius Challenge 2023 Grand Prize.  
Originally developed by [Youssef Nader](https://github.com/younader) and [Luke Farritor](https://github.com/lukeboi).

---

## Automatic Unwrapping (Segmentation) Pipelines

### [VC3D (surface tracer)](volume-cartographer)
A semi-automatic segmentation pipeline to extract papyrus sheets from CT scans of ancient scrolls.
As of September 2025, it is the approach used by the Vesuvius Challenge team.
Developed by [Hendrik Schilling](https://github.com/hendrikschilling) and [Sean Johnson](https://github.com/bruniss) as a fork of [Volume Cartographer](https://github.com/educelab/volume-cartographer).

### [spiral fitting](https://github.com/pmh47/spiral-fitting)
A fully automatic unwrapping pipeline that fits a spiral to the CT scan.
Originally developed by [Paul Henderson](https://github.com/pmh47).

### [thaumato-anakalyptor](thaumato-anakalyptor/)
A semi-automatic segmentation pipeline to extract papyrus sheets from CT scans of ancient scrolls.  
Originally developed by [Julian Schilliger](https://github.com/schillij95).

---

## Supporting Resources

### [scrollprize.org](scrollprize.org/)
Source for the [Vesuvius Challenge website](https://scrollprize.org).

[licence-badge]: https://img.shields.io/github/license/ScrollPrize/villa?color=blue
[licence-url]: https://github.com/ScrollPrize/villa/blob/main/LICENSE
[stars-badge]: https://img.shields.io/github/stars/ScrollPrize/villa?style=social
[stars-url]: https://github.com/ScrollPrize/villa/stargazers
[discord-badge]: https://img.shields.io/discord/1079907749569237093.svg?logo=discord&style=flat-square
[discord-url]: https://discord.com/invite/V4fJhvtaQn
