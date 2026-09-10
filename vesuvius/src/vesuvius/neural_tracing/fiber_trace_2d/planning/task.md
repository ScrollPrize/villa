# Task: local Fiber3D metric embeddings

Add opt-in U-Net embedding output and within-patch supervision. Defaults when
enabled: 16 normalized embedding channels, 65,536 pairs for a 128-wide patch,
pair count linear in patch width, half positive and half negative. Positive
endpoints are sampled along the annotated trace independently of presence.
Negative pairs join a GT point to a random output-grid point sampled in direct
proportion to detached predicted presence. Zero presence has zero probability.
Exclude the tracer's Lasagna-normal ellipse: 20 original-volume voxels normal,
80 tangential. Require dataset Lasagna normals and preserve geometry through
scale/rotation/warp. Positive loss is 1-cosine; negative loss is relu(cosine).
Average positive/negative means equally. No cross-patch pairs. Preserve old
configs, inference direction/presence interpretation, and effective-batch/DDP
normalization. Document and test; avoid unapproved GPU use.
