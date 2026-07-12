# Joint Top-View Direction And Distance-Transform Model

Train a second `fiber_trace_2d` model jointly with the existing side-strip
direction model. The second model uses top-view strip slices constructed the
same way as the top-strip visualization.

- The top-view model outputs the same Lasagna ambiguous two-channel direction
  representation as the side model.
- Instead of sheet/fiber presence, the top-view model also outputs a sigmoid
  distance-transform channel:
  - target is `1.0` at the fiber center;
  - target falls linearly to `0.0` at distance `30 px`;
  - pixels beyond `30 px` on the supervised line are also explicitly
    supervised as `0.0`.
- Distance-transform supervision is only along the rounded normal/cross-fiber
  line through each transformed CP in the top-view patch.
- Add top-view losses, metrics, TensorBoard scalars, and TensorBoard images:
  - top-view GT line plus estimated direction overlay;
  - top-view distance-transform output map.
- Include top-view slice dependencies in training prefetch, so chunks touched
  only by the top-view slices are fetched before training.
