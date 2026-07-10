# Task: Strip Coordinate Cache

Implement the todo item `cache strip coord`.

- Add a separate configurable cache for CP-local strip patch coordinates.
- Use the cache from the shared strip source path so training,
  visualizations, runner loading, and prefetch all benefit.
- Cache identity must include the CP 3D coordinate, selected volume path/scale,
  and strip pixel scale/step. Because strip coordinates also depend on local
  fiber geometry and Lasagna normals, include a fiber-line identity to prevent
  incorrect cross-fiber hits. Store source patch size in metadata so larger
  entries can satisfy smaller requests.
- A cached entry with source height/width greater than or equal to the requested
  source size is a valid hit and should be center-cropped to the requested
  source size.
- If a larger source patch is generated later, it should replace the smaller
  cached source for that identity family.
