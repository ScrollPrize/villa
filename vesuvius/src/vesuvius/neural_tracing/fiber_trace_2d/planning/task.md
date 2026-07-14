# Replace Dense Disk Strip-Coordinate Cache With Compact In-RAM Fiber-Line Geometry

Remove the current per-control-point dense strip-coordinate disk cache. It is
too large because it stores full `H x W x 3` coordinate fields for every CP,
even though neighboring CPs overlap heavily and the strip grid is derived from
an interpolated fiber centerline plus frame axes.

Instead, build one shared compact full-fiber geometry store at loader startup
and keep it in RAM:

- one shared cache object per configured dataset/loader process, not duplicated
  per worker/isolated loader;
- progress output while startup preprocesses the fibers;
- direct CP lookup into preprocessed records, with no search in the hot path;
- compact per-column center/axis data that reconstructs side/top strip grids
  quickly by broadcast math at sample time;
- preserve current strip geometry semantics, Lasagna normal ambiguity handling,
  deterministic sample order, prefetch behavior, training, and visualization
  outputs.
