# Python Native 3D Trace2CP CP Label Placement

Change native 3D whole-fiber Trace2CP visualization so CP labels no longer
cover the point markers.

Requirements:

- draw CP distance labels at the bottom of the respective strip instead of next
  to/on top of the CP marker;
- include the CP index in the label so the user can pass it back through
  `--start-cp-index` / `--target-cp-index`;
- keep the CP marker itself visible;
- do not change tracing, metrics, inference, or output selection behavior.
