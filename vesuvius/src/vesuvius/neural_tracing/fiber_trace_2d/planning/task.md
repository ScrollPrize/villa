# Graceful cooperative maintenance under allocation changes

Fix cooperative Fiber3D training so reviewed-trace maintenance requests are
handled correctly regardless of when manager pause/reallocation directives
arrive.

- A maintenance request must remain pending across allocation-driven cohort
  drain, release, and replacement.
- Starting maintenance while its cohort is temporarily unable to hand off must
  wait rather than fail.
- A grant revoked during the reviewed-trace child must stop that GPU work,
  release cleanly, and retry the same maintenance request after a new grant.
- Training must resume from the last complete checkpoint without losing or
  duplicating the requested periodic trace boundary.
