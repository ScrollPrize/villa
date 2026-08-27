# Task: Give Fiberlet crop lookahead full boundary context

Crop tracing must not rank lookahead routes against a graph truncated at the
requested output crop. Materialize a search graph expanded from the requested
half-open crop by the configured lookahead distance on every face, while
retaining only requested-crop anchors as seeds and clipping accepted output at
the original requested crop.

Record staged Fiberlet filtering as the next, separate speed/quality
experiment; do not enable filtering in this task.
