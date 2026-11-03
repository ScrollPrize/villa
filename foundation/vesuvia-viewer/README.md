# vesuvia

Use vesuvia to view Herculaneum scroll CT in a browser. It has three slice panes and one flattened surface pane. It draws one `.tifxyz` surface as outlines in the slice panes, overlays prediction volumes on the CT, and shows one 2D ink image on the surface.

```
uv sync
uv run tools/serve.py   # http://localhost:8080/
```

You can serve the files from any static host instead, and the browser code needs no build step. Click **Open…** and enter the URLs. Press `?` to list the keys. The URL hash stores the view, so a copied link reproduces the view. See `ui/hash.js` for the hash format.

To open PHerc. Paris 4 from S3 with its CT, its 3D ink prediction, one segment, and that segment's 2D ink image, use this link:

```
http://localhost:8080/#ct=https%3A%2F%2Fvesuvius-challenge-open-data.s3.us-east-1.amazonaws.com%2FPHercParis4%2Fvolumes%2F20260411134726-2.400um-0.2m-78keV-masked.zarr%2F&layer=https%3A%2F%2Fvesuvius-challenge-open-data.s3.us-east-1.amazonaws.com%2FPHercParis4%2Frepresentations%2Fpredictions%2Fink-3d%2F20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr%2F&surface=https%3A%2F%2Fvesuvius-challenge-open-data.s3.us-east-1.amazonaws.com%2FPHercParis4%2Fsegments%2F20231221180251%2Fmesh%2F20231221180251-on-20260411134726-2.4um.tifxyz%2F&image=https%3A%2F%2Fvesuvius-challenge-open-data.s3.us-east-1.amazonaws.com%2FPHercParis4%2Fsegments%2F20231221180251%2Fink-detection%2FPHercParis4-20231221180251-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif&focus=44193.53,13822.27,20409.01&zoom=8
```

To browse the published scrolls, open `http://localhost:8080/#ext=extensions/catalog.js`.

For an S3 URL that names a region, the viewer fetches from the four hostnames for the bucket. Add `&atlas=1024` to give the CT 1024 MiB of GPU memory.

## Extensions

Write one module. Its default export is `create(viewer)`, and the function returns an object with `panel`, `draw(pane)`, and `dispose()`. You can add `drawSurface(pane)` and `keys` when the extension needs them.

Load the module with `&ext=<url>`. If the module has another origin and that origin is not in `EXTENSION_ORIGINS` in `config.js`, the page asks before it loads the module. `main.js` builds `viewer`. `extensions/catalog.js` is a complete example.

An extension can draw over any pane with `pane.drawSegments()` and `pane.drawPoints()`. It can take the left button with `scene.tool` or `scene.surfaceTool`. It can read voxels with `readRegion()` in `volume/region.js`. To show a volume that it computes or paints, it uses `viewer.addLayer()` and `createMemoryVolume()` in `volume/memory.js`. `labelColor()` in `gpu/layer.glsl.js` returns the colour a labels layer draws for an id.

## Checks

```
git config core.hooksPath tools/hooks   # node tests before a commit, and browser checks when a staged file affects the page
node --test
uv run tools/e2e.py
uv run tools/perf.py '<hash>'           # on the GPU; tools/compare.py against HEAD
```
