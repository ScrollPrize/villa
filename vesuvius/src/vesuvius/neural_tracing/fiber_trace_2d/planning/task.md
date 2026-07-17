# Native 3D Trace Smoothness Weight

Increase native 3D Trace2CP search smoothness so abrupt branch switches are
penalized by default.

Required behavior:

- Raise the default smoothing loss weight used by native 3D Trace2CP.
- Keep the CLI override so experiments can still set a different value.
- Update specs/docs/tests to reflect the new default.
