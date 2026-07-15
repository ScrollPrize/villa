# 3D Fiber Follow-Up Plan

Plan the parts left out of the initial 3D fiber CP implementation:

- smooth displacement augmentation in 1D, 2D, and full 3D;
- anisotropic directional 3D blur;
- full Trace2CP metric wiring for 3D checkpoints;
- keep shear and ringing explicitly out of scope;
- confirm current regular affine augmentation support, especially rotation and
  shift.

The current 2D fiber path must stay supported. The 3D path should keep ordinary
CP-centered 3D block loading, not fiber-aligned strip loading.
