# growpatch_blend

Refines a GrowPatch segment so it sits more squarely on the papyrus sheet it is already following. It takes a
tifxyz segment (VC3D's surface format) and the surface prediction it was grown on, and writes a new tifxyz segment
with the same files GrowPatch writes. Here "sheet" means one labeled papyrus layer, as in the Vesuvius Challenge instance labels.

```sh
python -m segmentation.growpatch_blend /path/to/scroll.volpkg/paths/<segment> <prediction.zarr>/0 --workers 8
```

The new segment is written next to the input as `<segment>_blend`. It also holds `move.npy` (how far each point
moved, in voxels) and `blend_report.json`. Its grid is 4× denser than an 8-voxel GrowPatch grid.

## What it does

Tiles are laid every 32 voxels over the segment. In each tile, a second, independent trace of the same prediction
is grown from the segment (a ridge-following graph trace). Both surfaces are then compared along short rays across
the sheet:

- Where both see exactly one sheet and are within 8 voxels of each other, the segment point moves halfway toward
  the trace.
- Where they disagree by more, or either sees several sheets, the point stays where GrowPatch put it.
- A tile whose trace finds no ridge near the segment changes nothing.

Each point's move is a weighted average of the tiles that see it, so no point moves more than 4 voxels. The tool
never looks at labels.

## Evidence

Tested against the Vesuvius Challenge hand labels of PHerc. Paris. 4 (Scroll 1, 7.91 µm). The settings were fixed
on 16 labeled regions first, then the tool ran once on 16 other regions:

| GrowPatch cutoff | Target-sheet area within 128 voxels of the seed | Regions better / tied / worse | Share of labeled hits on a wrong sheet |
|---|---|---|---|
| 1 | 30.48 → 33.98 mm² (+11.5%) | 15 / 1 / 0 | 2.94% → 3.73% |
| 128 | 25.02 → 30.22 mm² (+20.8%) | 15 / 1 / 0 | 1.66% → 4.14% |
| 192 | 19.44 → 25.96 mm² (+33.5%) | 15 / 1 / 0 | 5.22% → 6.91% |

"Target-sheet area" counts only surface on the hand-labeled sheet that is joined to the seed.

The wrong-sheet share rises. Almost nothing already on the target sheet is moved off it. The rise comes from
GrowPatch surface that sat on unlabeled voxels and was pulled onto a neighbouring labeled sheet. At cutoffs 1 and
128, most of it is in regions where GrowPatch never connects the target sheet to the seed. Stricter versions that required the overlapping
tiles to agree did not remove the rise on the development regions, and they lost most of the gain.

On a full official 160-generation GrowPatch segment from PHerc. 125 it ran in about two to three minutes on 20
CPU workers (the 41% of that segment inside its prediction cube is refined), and it gives the same x, y and z
values as the benchmark. The benchmark, the label replay and an
interactive viewer of all 16 test regions are at <https://github.com/udemex10/vesuvius-surface-patches>.

## Limits

- It tightens the sheet GrowPatch is already on. It does not repair a place where GrowPatch has already jumped to
  a neighbouring sheet.
- It was tested against labels on one scroll. On PHerc. 125 it runs, but there are no labels there to measure
  accuracy.
- The tool reads the prediction in a box around the segment. Parts of a segment that GrowPatch extrapolated
  beyond the prediction get no ridge and stay as they are.

## Tests

```sh
python -m pytest segmentation/growpatch_blend/tests
```

The tests use a small synthetic prediction (one flat sheet) and need only numpy and scipy. Reading and writing
segments and Zarr arrays goes through `vc_reader`.
