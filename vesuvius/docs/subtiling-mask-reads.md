# Reuse mask surfaces during subtiling patch discovery

`find_segment_patches` in `ink_detection/data/patch_finding_subtiling.py`
already reads the complete inklabel and supervision surfaces to build its
index. It previously discarded those arrays, then read overlapping Zarr
windows from both stores again for each candidate. The change retains those
two arrays and slices them in memory. Tile selection, patch order, coverage,
held-out filtering and validation-mask access are unchanged.

The same idea has been proposed for the default finder in
[#2197](https://github.com/ScrollPrize/villa/pull/2197). This change covers the
separate subtiling finder; it is not a new caching algorithm.

## Measured example

Data: published `scrollprize/datasets` bucket,
`ink/man5/MAN5_outer_3/{MAN5_outer_3_inklabels,MAN5_outer_3_supervision_mask}.zarr`,
level 2, middle slice 32. The actual label planes are 6016 × 7461 uint8.
A complete listing of that level was used to copy its 147 stored chunks
(58,606 compressed bytes); omitted fill-value chunks are left omitted.

Configuration: `patch_size=[17,64,64]`, overlap 0.5, subtiling tile size 64,
stride 32, empty-tile filtering on, minimum labeled coverage 0.1, no validation
mask. The data has 270,947 nonzero label pixels and 701,075 supervised pixels.
The baseline is villa `37d1897fb3e7a02ecba21de6efb6912d73cee5af`.

Each row compares the full `find_segment_patches` call, including local Zarr
reads/decompression, index building, coverage checks and Patch construction.
The OS file cache was not cleared. Both versions produce the same 237 training
patches in the same order. Neither run trains a model.

| Measurement | Baseline | Reuse surfaces |
| --- | --- | --- |
| In-process median, 9 alternating runs | 1.930 s | 1.198 s |
| Independent in-process median, 5 alternating runs | 2.008 s | 1.267 s |
| Fresh-process work median, 5 pairs after one warmup pair | 2.191 s | 1.408 s |
| Benchmark-process launch-to-exit median, same fresh pairs | 2.650 s | 1.839 s |
| Median peak process RSS, same fresh pairs | 224,292 KiB | 223,020 KiB |
| Inklabel/supervision array reads, counted separately | 263 / 263 | 1 / 1 |

The fresh-process work reduction is 35.7%; benchmark-process time is 30.6%
lower. The small RSS difference is not evidence of a memory improvement.
These figures describe one segment and one supported non-default preparation
mode, not training/inference throughput, unread-text quality, all scrolls, or
concurrent multi-segment memory use. The known image/label mapping question in
[#2369](https://github.com/ScrollPrize/villa/issues/2369) is not corrected here.

Regression checks cover patch order, held-out overrides, unchanged input data,
one read per already-needed mask surface, and no new validation reads for an
empty index. Validation is deliberately left on its existing per-patch path.
Data provenance/terms: https://scrollprize.org/data (CC-BY-NC 4.0 unless noted).
