# Native 3D Whole-Fiber Trace2CP Eight-Row Visualization

Update native 3D Trace2CP whole-fiber visualization so each visual span has
eight rows:

- initial side volume;
- initial side 3D presence;
- initial top volume;
- initial top 3D presence;
- regenerated/fused side volume;
- regenerated/fused side 3D presence;
- regenerated/fused top volume;
- regenerated/fused top 3D presence.

The regenerated strip should be built from the traced/fused whole-fiber span,
matching the single-pair debug visualization behavior.

Additionally, draw the control points for the displayed span into the full
fiber visualization panels, including regenerated side/top rows.

Overlay each displayed control point's Trace2CP distance to the trace at that
CP plane.
