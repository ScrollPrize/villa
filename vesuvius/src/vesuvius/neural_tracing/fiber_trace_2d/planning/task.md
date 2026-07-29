# Native 3D Trace2CP Target Plane Normals

Fix Trace2CP target-plane termination and endpoint-error calculation so CP-to-CP
chord normals are not used.

Requirements:

- Do not use the straight chord from start CP to target CP as a target-plane
  normal.
- For each target CP, consider three target planes:
  - plane through the CP with normal from the CP to the next loaded fiber line
    point;
  - plane through the CP with normal from the CP to the previous loaded fiber
    line point;
  - plane through the CP with normal from the sampled fiber-direction inference
    at that CP.
- Continue tracing until all three target planes have been crossed, or until
  the existing failure conditions apply.
- When all three planes are crossed, compute the in-plane CP error for each
  crossing and use the smaller/best error for segment success and metric
  reporting.
- Apply the same behavior to Python native 3D Trace2CP and the native VC3D
  fiber tracer/metric path so their target-plane semantics remain aligned.
- Keep existing trace scoring, beam-search, restart, and visualization behavior
  unchanged except for displaying/reporting the new selected crossing/error
  information where relevant.
