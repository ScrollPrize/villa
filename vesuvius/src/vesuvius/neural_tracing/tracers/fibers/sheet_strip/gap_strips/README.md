# Sheet strips for fibers with gaps

`sheet_strip/infer_all.py` grows one strip per fiber of a store, along all of its line points. Three things make
that wrong or wasteful for hand-annotated fibers:

- a fiber can carry spans tagged `gap` or `damaged`, which a strip must not cross;
- its line points run past the outermost control points (tails), which are not part of the fiber;
- strip nodes within `--trim` (75 patch voxels) of either fiber end are not kept, and fibers shorter than
  `--min-length-mm` (2.5 mm) are not traced, so a fiber cut at its gaps loses both ends of every piece or is not
  traced at all.

This folder derives fiber stores that `infer_all --fibers` can trace instead, and thins the strips into patches for
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
| `mirror.py` | `SOURCE.vcobj` -> `PIECES.vcobj` and `CHAINS.vcobj` |
| `retry.py` | retry fibers for chains the tracer could not start |
| `split.py` | chain strips -> one strip per piece |
| `fit_patches.py` | strips -> thinned tifxyz patches |
| `grow_strips.sh` | all of the above in order |

Each module's docstring gives its command line and output format. The source store is only read. A piece or chain
has a UUID derived from its source fiber's UUID, so a rerun gives the same strip folders. Its `meta.gap_strips`
names the source fiber and version and, for a chain, its piece versions and where they and the mirrored parts lie;
`split.py` reads that from the chain version a strip was traced on, and refuses a strip traced on another chain
version or a pieces store other than the one the chain was made with.

## Inputs

- A fiber store whose fibers are bound to one volume, and optionally a text file of the fiber UUIDs or names to use.
  A fiber's control points must be in the order of its line points (either direction).
- The CT as an OME-Zarr (local path or URL).
- `umbilicus.json` in patch voxels, for the tracer.
- A GPU for the `trace` step. The other steps run on the CPU.

The strip checkpoint defaults to `scrollprize/fiber_strip_small_v1` on Hugging Face (see the `sheet_strip` section of
`fibers/README.md`). `--join` and `--ext` are in patch voxels. `mirror.py --level-scale` (default 4) is the number of
volume voxels per patch voxel and must be the scale the tracer runs at (`infer_all --ct-grid-scale`);
`grow_strips.sh` passes its `LEVEL_SCALE` to both, with `CT_LEVEL` (default 2) as the CT pyramid level read, and
gives the tracer the matching voxel sizes (`BASE_VOXEL_UM` x `LEVEL_SCALE` per patch voxel).

## Run

With the vesuvius environment active:

```bash
export SOURCE=/path/to/fibers.vcobj NAMES=/path/to/names.txt   # NAMES is optional
export CT=/path/to/ct.zarr UMBILICUS=/path/to/umbilicus.json OUT=/path/to/output
sheet_strip/gap_strips/grow_strips.sh pieces all
sheet_strip/gap_strips/grow_strips.sh mirror all
```

The steps can be run one at a time: `fibers`, `trace`, `split` (mirror mode), `patches`. The result is
`$OUT/patches`: one tifxyz per strip, named `strip-net-<piece UUID>`, every 2nd row and column kept. Strips with
fewer than 500 nodes are not written. `grow_strips.sh` lists the optional settings in its header.

The patches carry no weights. The spiral fit weighs patches by name prefix and by place, with its settings
`patch_sampling_name_weights` (e.g. `{"strip-net-": 1.0}`), `patch_sampling_core_radius` and
`patch_sampling_core_weight`.

### Chains the tracer cannot start

A chain's strip starts on the mirrored copy, which can lie outside the scanned material. Retry such chains without
the leading copy and add their strips before the `patches` step:

```bash
PKG=vesuvius.neural_tracing.tracers.fibers.sheet_strip
python -m $PKG.gap_strips.retry $OUT/chains.vcobj $OUT/retry.vcobj UUID1 UUID2 ...
python -m $PKG.infer_all --fibers $OUT/retry.vcobj --out $OUT/strips_retry --ct $CT --ct-level 2 --ct-grid-scale 4 \
    --base-voxel-um 2.4 --voxel-size-um 9.6 --umbilicus $UMBILICUS
python -m $PKG.gap_strips.split $OUT/strips_retry $OUT/retry.vcobj $OUT/pieces.vcobj $OUT/strips_split_retry
cp -r $OUT/strips_split_retry/strips/* $OUT/strips_split/strips/
```

`fit_patches.py` takes every strip folder under `strips/`, so the copied ones are staged with the rest; the index
and report files of `strips_split` still list the first run only.

## Tests

`fibers/tests/test_gap_strips.py` covers the piece cutting, the fiber selection, the derived stores of a store fiber
with a tagged gap, the chain joining and mirroring, the retry, the split (both strip layouts, columns beside a piece
end, a strip of another chain version or scale) and the patch thinning, on synthetic data.
