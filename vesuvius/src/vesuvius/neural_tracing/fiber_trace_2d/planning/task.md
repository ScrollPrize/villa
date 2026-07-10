# Task: Cache Source Line Coordinates And Split Coord Profiling

Extend the strip-coordinate cache.

- Version-bump the strip-coordinate cache so old entries are ignored.
- Include source-space line coordinates and source-space CP pixel coordinates
  in the cached source entry.
- Use the cached source line/CP coordinates for unaugmented patches.
- Keep random augmented line coordinates computed per patch, because they
  depend on each augmentation draw.
- Split training profile output so strip-coordinate cache load time is reported
  separately from the broader coordinate stage.
