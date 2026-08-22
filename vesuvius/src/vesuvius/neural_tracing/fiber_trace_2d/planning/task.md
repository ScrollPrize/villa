# Task: adopt the compact float-position fiberlet default

Make the accepted float-position, compact-direction, fixed nonlinear `uint16`
cost profile the default for cache-backed fiberlet replay and future compact
storage work. Keep the all-float profile available explicitly as the correctness
oracle. Do not change the unpublished persistent compact payload schema as part
of this default-selection task.

The accepted profile is:

- exact float endpoint positions;
- compact two-byte fitted directions;
- fixed sqrt-density `uint16` edge costs with density ceiling 256;
- float path length and float join costs.

Commit this default together with the completed one-eighth-base-voxel
quantization experiment.
