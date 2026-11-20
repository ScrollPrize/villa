# Sheet strips for fibers with gaps

`sheet_strip/infer_all.py` grows one strip per fiber of a store, along all of its line points. Three things make
that wrong or wasteful for hand-annotated fibers:

- a fiber can carry spans tagged `gap` or `damaged`, which a strip must not cross;
- its line points run past the outermost control points (tails), which are not part of the fiber;
- strip nodes within `--trim` (75 patch voxels) of either fiber end are not kept, and fibers shorter than
  `--min-length-mm` (2.5 mm) are not traced, so a fiber cut at its gaps loses both ends of every piece or is not
  traced at all.

This folder derives fiber stores that `infer_all --fibers` can trace instead, and turns the strips into patches for
the spiral fit.

| Mode | What it does |
|---|---|
| `pieces` | One fiber, and so one strip, per valid piece. Simple; loses 75 patch voxels at both ends of every piece. |
| `mirror` | Pieces of a fiber closer than 75 patch voxels are joined into a chain (the gap points stay cut out). Each chain end is extended by 80 patch voxels with a mirrored copy of the line, so the tracer's end trim eats the copy instead of the fiber. After tracing, the chain strips are cut back into one strip per piece and the joint and mirrored columns are dropped. |

No strip from either mode has a node on a gap, a damaged span or a tail.

## Modules

| Module | Step |
|---|---|
| `spans.py` | read a store's fibers with their control points and span tags; the valid pieces |
| `pieces.py` | `SOURCE.vcobj` -> `PIECES.vcobj` |
| `mirror.py` | `SOURCE.vcobj` -> `PIECES.vcobj` and `CHAINS.vcobj` (+ `CHAINS.vcobj.joined.json`) |
| `retry.py` | retry fibers for chains the tracer could not start |
| `split.py` | chain strips -> one strip per piece |
| `fit_patches.py` | strips -> tifxyz patches with fit weights |
| `grow_strips.sh` | all of the above in order |

Each module's docstring gives its command line and output format. The source store is only read. A piece or chain
has a UUID derived from its source fiber's UUID, so a rerun gives the same strip folders, and its `meta.gap_strips`
names the source fiber and version.

## Inputs

- A fiber store whose fibers are bound to one volume, and optionally a text file of the fiber UUIDs or names to use.
- The CT as an OME-Zarr (local path or URL).
- `umbilicus.json` in patch voxels.
- A GPU for the `trace` step. The other steps run on the CPU.

The strip checkpoint defaults to `scrollprize/fiber_strip_small_v1` on Hugging Face (see the `sheet_strip` section of
`fibers/README.md`). `--join`, `--ext` and the patch radius are in patch voxels; `--level-scale` (default 4) is the
number of volume voxels per patch voxel.

## Run

With the vesuvius environment active:

```bash
export SOURCE=/path/to/fibers.vcobj NAMES=/path/to/names.txt   # NAMES is optional
export CT=/path/to/ct.zarr UMBILICUS=/path/to/umbilicus.json OUT=/path/to/output
sheet_strip/gap_strips/grow_strips.sh pieces all
sheet_strip/gap_strips/grow_strips.sh mirror all
```

The steps can be run one at a time: `fibers`, `trace`, `split` (mirror mode), `patches`. The result is
`$OUT/patches`: one tifxyz per strip, named by the piece's UUID, with `loss_weight_scale` 1.0 in `meta.json` (0.2
when the strip's median radius from the umbilicus is under 230 patch voxels), every 2nd row and column kept. Strips
with fewer than 500 nodes are not written. `grow_strips.sh` lists the optional settings in its header.

### Chains the tracer cannot start

A chain's strip starts on the mirrored copy, which can lie outside the scanned material. Retry such chains without
the leading copy and add their strips before the `patches` step:

```bash
PKG=vesuvius.neural_tracing.tracers.fibers.sheet_strip
python -m $PKG.gap_strips.retry $OUT/chains.vcobj $OUT/retry.vcobj UUID1 UUID2 ...
python -m $PKG.infer_all --fibers $OUT/retry.vcobj --out $OUT/strips_retry --ct $CT --ct-level 2 --umbilicus $UMBILICUS
python -m $PKG.gap_strips.split $OUT/strips_retry $OUT/retry.vcobj $OUT/pieces.vcobj $OUT/strips_split_retry
cp -r $OUT/strips_split_retry/strips/* $OUT/strips_split/strips/
```

## Tests

`fibers/tests/test_gap_strips.py` covers the piece cutting, the derived stores, the chain joining and mirroring, the
retry, the column labelling of the split and the patch weights, on synthetic data.
