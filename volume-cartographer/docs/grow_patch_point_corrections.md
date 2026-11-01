# Point Based correction workflow

vc_grow_seg_from_seed can incorporate point based annotations to constrain the solver to move the patch surface through those points.
Together with resuming from an existing surface this can be used to grow larger patches, fixing errors as they appear.

Inside VC3D the same corrections are part of the segmentation editor (see [extending_and_modifying_segmentations.md](extending_and_modifying_segmentations.md)); there they are stored with the surface as linked point collections. This document covers the command line, which works on the surfaces of an object store.

## Workflow

1. Grow some patch into the store with vc_grow_seg_from_seed (`--store`, `--volume-uuid`)
2. Annotate corrections using the point collection feature in VC3D and save them as JSON (Point Collections → Save)
3. optional: mask areas that should be removed before continuing patch growth (the segmentation editor's mask brush commits them to the surface)
4. re-run vc_grow_seg_from_seed with `--surface <uuid>` and `--correct <corrections.json>`; the result is a new surface whose provenance names the one it resumed from

```shell
vc_grow_seg_from_seed -v <volume.zarr> -p <params.json> \
  --store <objects.vcobj> --volume-uuid <volume id> \
  --surface <surface uuid> --correct <corrections.json>
```

## Details

### Annotation

Every point collection in vc3d should be one broken area. The first point placed for each collection is special and should be a point close the problem but still on the correct surface (best place it in the flattened segmentation view so it is actually on the correct surface). The remaining points can be on or off the current surface but should be where the surface is supposed to go. They will pull in the surface from close to them so place them where the surface in principle could go but for some reason drifted off or is offset by some amount along its normal. Easiest is to place the points by focusing (ctrl-click) into the broken area and then placing a chain of points along the right path, ideally with the first ones being on the correct surface then some where the surface went off and then again on the corect surface, so the edges of the correction touch the existing surface. Save the annotation using some sensible name.

### Masking

The correction feature will solve an area around each collection of points, Often tracing errors start at a certain location and then spread, so to minimize the annotation effort simply annotate a small area around the initial error and then mask out the full error area. The solver will keep the area around the convex hull of the correction points, hence after correction the surface can now grow along the correct surface.

## Useful Parameters

### config json for vc_grow_seg_from_seed
- z_min,z_max - limits z range -> useful to solve a long spiral, can be quite thin (just a few quads)
- "snapshot-interval" - in tifxyz mode (`--target-dir`) saves snapshot_gen_XXX snapshots at the specified interval; store mode writes no snapshots

### vc_grow_seg_from_seed
- `--surface` - the stored surface from which to resume processing (correct & grow)
- `--correct` - pass the json with the correction points to apply them
- `--resume` - tifxyz mode only: the tifxyz directory to resume from
