# Fit surface geometry to regularized UV attachments

Keep smooth100/anchor0.1 and all existing fitting settings. Add optional
geometry_tangent_weight (default0): geometry at detached current UV receives
normal squared error plus weight times tangential squared error, using detached
surface normals. UV continues receiving its existing tangent-only error and
regularizers. H targets stay actual XYZ; V keeps its weaker offset targets.
Weight1 is full positional geometry fitting. No gradient to UV from the new term.

Run matched fresh800-step candidates0.1,0.3,1 via approved manager-backed runner,
nonexclusive8GiB GPU grants on0–5. Compare existing weight0 baseline: tangential
and full positional RMS per family, normal RMS, UV offsets, folds and exact locks.
Render useful candidate with identical crop,CT JPG and8px H/V TIFF overlays.

## Spec update
Geometry normal-only baseline remains default; opt-in tangent fitting is now
supported. Save the weight with attachments and restore during continuation.
No upstream changes, no fixed UV guarantee or global nonintersection claim.

## Tests and docs
Independent plan review; regression tests for geometry-only tangent gradients,
weight0 compatibility, weight1 full-XYZ equivalence, invalid weights, restoration.
Record full configs, metrics, commands and limitations; archive prior active log.
