---
name: vc3d-agent-bridge
description: Compatibility entry point for driving VC3D through MCP. Load before vc3d_* calls when an older prompt names vc3d-agent-bridge; it routes to the current session and workflow skills.
---

# VC3D bridge compatibility entry point

Use `vc3d-bridge-session` as the primary session skill. It owns build identity,
live state, job sources, waits, cleanup, and bridge failures.

Then load the focused skill for the requested workflow: Open Data, segments,
editing, points/winding, fiber tracing, Lasagna, Atlas, Spiral, seeding,
rendering, flattening, or visual evidence. Tool descriptions and the generated
RPC contract remain authoritative for individual parameters.

For preserved cross-call behavior that does not fit one tool description, read
[`references/cross-call-footguns.md`](references/cross-call-footguns.md).
