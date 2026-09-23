# Task

Add a `las_manager snapshot import` command which copies a training checkpoint
into persistent manager-owned storage and binds the copied bytes to a required
external model identifier (for example `scrollprize/lasagna-fiber`). Snapshot
identity must be keyed by checkpoint SHA-256 rather than a mutable path.
Inference must refuse to launch any checkpoint for which no trusted model
identifier can be resolved.
