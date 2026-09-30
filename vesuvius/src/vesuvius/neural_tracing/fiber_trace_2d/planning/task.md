# Task: anisotropic Gaussian fiber-presence supervision

Resume high-resolution five-scroll training from the refinement run's best
EMA-0.999 snapshot, keeping online fiber refinement enabled. Replace the
selected run's point/centerline-only presence supervision with a soft Gaussian
tube around each supervised fiber line. The tube is anisotropic in the local
fiber/sheet frame: its in-sheet width is four times its surface-normal
thickness. Apply direction supervision over the complete Gaussian support,
weighted by the Gaussian falloff. Preserve all legacy training behavior unless
the new target mode is explicitly selected. Run the replacement only on
physical GPUs 0-5; GPUs 6 and 7 are reserved for other work.

Continue the stopped Gaussian run from its latest optimizer checkpoint with
half the previous distribution sigma (normal sigma 1 voxel, in-sheet sigma 4
voxels), retaining the 4:1 anisotropy and all refined-fiber state. Run this
continuation on physical GPUs 0-6.

Also allow online refinement to rotate the anisotropic distribution around the
local fiber tangent. Keep rotation updates small and weakly tether the refined
orientation to the original native Lasagna normal. Persist this orientation
state together with refined fiber geometry and use it for subsequent targets.
