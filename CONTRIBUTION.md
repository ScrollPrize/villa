**Thank you for your interest in contributing to villa!** 

This codebase is an open research project and thus the code within it is highly research focused and/or experimental. We try and share as much of our experiment-related code as we can, and as a result there are many things in this 
codebase which may be infrequently used or never intended to be "productionized". 

We expect as people interact with our codebase in pursuit of the goal of unrolling the scrolls, they are going to encounter bugs or other issues, or have ideas of their own which would improve the codebase. 
We appreciate attempts to implement fixes or improvements in the form of PRs. 
___

## Contribution Guidelines
- PRs for fixes or improvements must come as a result of you running the tool on real scroll data in persuit of one of the goals of this project. Given the experimental and constantly changing nature of this codebase we do not intend to support
every script or line of code ever commited. We are only interested in supporting things which are actually used by humans.
- Any bugfix PR must be accompanied by a screenshot of the error (either terminal or within the tool), and the script/tool running without error afterward
- Bugfixes or improvements must be run on real scroll data. Synthetic or toy examples are not accepted
- PRs must contain a motivation section, detailing what it is you were attempting to do when this issue arose (ex: "i was attempting to use `vc_grow_seg_from_seed` in this volume to do some ink exploration when this error popped up")


## AI Guidelines** 
We support the use of LLMs as coding assistants, and we make broad use of them ourselves. However, because reviewing PRs takes a significant amount of time away from the goal of unrolling the scrolls, we have a few rules regarding llm assisted PRs:
- PRs for bugfixes or improvements must come as a result of a human interacting with the codebase in an attempt to work on the scroll data. 
- Any LLM generated PR must be accompanied by human-written commentary explaining why this PR is relevant or useful
- Usefulness of any change must be calculated on real scroll data. Synthetic or toy examples are not allowed. 
- We expect that you have reviewed the code yourself for simplicity/accuracy
- We may close PRs which appear to be simple "fishing expeditions" for llms (ie. "claude find bugs in this codebase") unrelated to humans actually using the code in pursuit of our goals. 
