**Thank you for your interest in contributing to villa!** 

This codebase is an open research project and thus the code within it is highly research focused and/or experimental. We try and share as much of our experiment-related code as we can, and as a result there are many things in this 
codebase which may be infrequently used or never intended to be "productionized". 

We expect as people interact with our codebase in pursuit of the goal of unrolling the scrolls, they are going to encounter bugs or other issues, or have ideas of their own which would improve the codebase. 
We appreciate attempts to implement fixes or improvements in the form of PRs.

The types of PRs we love to see, and those most likely to get merged, have some (or all) of these properties: 
- They target _actively used_ portions of the codebase, either by the team or by members of the community
- They add needed features or ergonomics which meaningfully improve the user experience our applications like VC3D or our terminal-run scripts
- They contain examples before / after on real scroll data on real pipelines, both in metric form and also in the form of images or videos 
- Where appropriate, they compare results against current SOTA pipelines (for ex: you propose an ink detection change, and you show the current prediction vs your proposed one)
- They address real bugs which a user has actively encountered while working on scroll data or which harm task performance of our tooling

## Contribution Guidelines
- PRs for fixes or improvements must come as a result of you running the tool on real scroll data in persuit of one of the goals of this project. Given the experimental and constantly changing nature of this codebase we do not intend to support
every script or line of code ever commited. We are only interested in supporting things which are actually used by humans.
- Any bugfix PR must be accompanied by a screenshot of the error (either terminal or within the tool), and the script/tool running without error afterward
- Bugfixes or improvements must be run on real scroll data. Synthetic or toy examples are not accepted
- PRs must contain a motivation section, detailing what it is you were attempting to do when this issue arose (ex: "i was attempting to use `vc_grow_seg_from_seed` in this volume to do some ink exploration when this error popped up")
- Where applicable, images or videos must be provided comparing results against current methods (ie. ink detection, segmentation, etc)

## Pull Request and Issue Templates

When opening a pull request, please follow the [pull request template](.github/pull_request_template.md). It asks for a concise explanation, one real example, and direct before/after evidence before any additional detail.

When reporting a problem or making a request, please follow the [issue template](.github/ISSUE_TEMPLATE/issue.md). It asks what you were trying to do, what happened, and the shortest evidence or reproduction needed to understand it.

## AI Guidelines
We support the use of LLMs as coding assistants, and we make broad use of them ourselves. However, because reviewing PRs takes a significant amount of time away from the goal of unrolling the scrolls, we have a few rules regarding llm assisted PRs:
- PRs for bugfixes or improvements must come as a result of a human interacting with the codebase in an attempt to work on the scroll data. 
- Any LLM generated PR must be accompanied by human-written commentary explaining why this PR is relevant or useful
- The description of what you propose to change and why should be as concise and jargon-free as possible.
- Usefulness of any change must be calculated on real scroll data. Synthetic or toy examples are not allowed. 
- We expect that you have reviewed the code yourself for simplicity/accuracy.
- We may close PRs which appear to be simple "fishing expeditions" for llms (ie. "claude find bugs in this codebase") unrelated to humans actually using the code in pursuit of our goals. 

## Developer installation 
The codebase is constantly changing, and the python focused portions have complex dependencies, it is common to run into some issues here. For the most part we have 
tried to settle on using `uv` lockfiles with `pyproject.toml` in the primary subprojected (`ink-detection`, `volume-cartographer`, `vesuvius`, and `volume-cartographer/scripts/spiral`). 
it is highly recommended to use `uv`. 

For these projects, moving into the dir with the `pyproject.toml` and using `uv sync` should get you up and running with a proper venv. in the case of `vesuvius` you
may have to use `uv sync --all-extras`.

for VC3D `build_from_src_debian.sh` should handle the build, otherwise once you have the deps installed you can use cmake/Ninja to install from source

```bash
cmake -S . -B build -GNinja
ninja -C build
```
