# VC3D Native Tracing For New Fibers

- A newly created fiber already defaults to the native fiber tracer in the
  GUI and persisted global mode. Its initial generated geometry must also use
  the native tracer when fiber-inference data is configured and usable.
- Reuse the existing single-control-point native extrapolation path. The
  Lasagna seed solve may provide its required reference line and tangent, but
  that intermediate geometry must not be displayed or saved as the completed
  new fiber.
- When no fiber-inference dataset is configured, retain the existing Lasagna
  seed behavior. Do not force a dataset picker merely because the default mode
  is native tracing.
- Failures while opening configured fiber-inference data must remain visible;
  they must not be silently treated as successful native tracing.
