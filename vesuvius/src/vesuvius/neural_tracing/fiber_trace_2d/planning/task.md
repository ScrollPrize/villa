# Parallel post-trace extraction

Commit current work, parallelize saved-trace auditing and downstream exports. Preserve finished native trials, scientific selection, producer settings and GPU 0–5 allocation.

User approved correction of the discovered run-local/full-fiber arc bug: discard old reproduction results (not inference), add independently expected start XYZ validation with a small tolerance, and rerun. Old incorrect trial results must not be recovered.
