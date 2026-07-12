# Trace2CP Vertical Range Increase

Increase the vertical range used by Trace2CP segment visualization/tracing.

- The Trace2CP segment strip should provide more vertical room before traces hit
  RF/edge margins.
- Keep the existing segment construction, z-search behavior, metrics, and
  normal/VC3D coordinate semantics unchanged.
- Update regression tests and docs/specs that describe the Trace2CP segment
  height multiplier.
