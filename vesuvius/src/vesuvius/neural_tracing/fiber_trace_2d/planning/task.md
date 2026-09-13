# Current task: embedding consistency before tracing integration

Measure the running combined-data embedding model along held-out GT fibers:
within inference tiles, the same physical points predicted by overlapping tiles,
and along fibers crossing blended inference tile and stored-chunk boundaries.
Freeze and identify the checkpoint. Use existing inference input preparation,
tile geometry/blend functions and native VC3D CT loading. Do not change training
or tracer decisions. Run GPU analysis under exclusive manager ownership;
training resumes automatically. Review findings with the user first.

## Next task: embedding-guided live tracer (not implemented now)
1. Expose 16 embedding channels from the same inference forward through existing
   coherent chunk cache/readiness handling, not a separate loader.
2. Batch C++ embedding reads with candidate sampling: trilinear vectors then L2 normalize.
3. Compare candidates against distance-spaced recent accepted trace history
   within approximately the training patch range, not the original seed forever.
   Each beam hypothesis owns its reference history; no GT assistance.
4. Add embedding_weight * mean(1-cosine) to existing candidate cost, normalized
   for sampling density; retain direction, presence, normals and beam search.
5. Default weight zero preserves behavior; requested but missing embeddings error.
6. Check overlapping-tile embedding consistency before choosing vector blending.
   Review alternatives if necessary; no silently imposed embedding alignment.
7. Manager-exclusive 50-fiber evaluation: embedding off versus small weight
   sweep; per-scroll trace metrics, runtime, memory/cache cost and regression tests.
