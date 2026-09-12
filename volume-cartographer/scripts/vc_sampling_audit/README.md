# Final sampling-ray audit (report only)

This optional Python diagnostic reads the final rays exported by
`vc_render_tifxyz --sampling-grid-output FRESH_DIRECTORY`. It reports
continuous-depth triangle-sweep minima and deterministic failure locations.
It never reads CT, adjusts rays, repairs geometry, or declares ink or letters.

## Run without installing Villa

The Python diagnostic needs only an existing Python with NumPy. It does not
need the renderer binary, model weights, internet, GPU, or a C++ build.
Tested here on macOS arm64 with Python 3.14.6 / NumPy 2.4.6. Other Python/NumPy
versions and platforms have not been tested for this extracted package.

From a Villa checkout:

```sh
python volume-cartographer/scripts/vc_sampling_grid_audit.py \
  --capture /path/to/fresh-capture/grid.json \
  --reference 0.7750637572267833 -0.39282152542956317 0.49494183637341965 \
  --out /path/to/new-report.json
PYTHONPATH=volume-cartographer/scripts python -m unittest discover \
  -s volume-cartographer/scripts/vc_sampling_audit/tests -v
```

The reference above belongs to the recorded PHerc0800 research chart, **not a
universal direction**. Supply the oriented reference for your own base chart.
It is not a fitted normal correction. Reversing all normals reverses depth,
not the base chart; do not negate the reference just for `--flip-normals`.
Transposing the base chart can change its orientation.

To run the standalone source archive, replace
`volume-cartographer/scripts/` above with `scripts/`.
Alternatively, set `PYTHONPATH` to that scripts directory and run
`python -m vc_sampling_audit` with the same arguments. No pip install or
top-level Villa build is required. Existing reports are refused, not replaced.
A successful process exit means a report was produced, **not that the geometry
passed**; inspect both diagonal summaries.

## Capture contract and supported renderer modes

The companion C++ patch is opt-in. Use the existing renderer command for your
own data, with an additional `--sampling-grid-output FRESH_DIRECTORY`.
At the pinned revision, capture accepts TIFF-only, single-part, unrotated,
non-composite output without output flips, resume or preallocation. The grid
area must be at most 262,144 pixels and offset count at most 65,536.
The reader additionally requires a grid at least 2 x 2 and a JSON file no
larger than 64 MiB; a valid one-pixel-wide renderer crop is not auditable here.

The export copies final `base`, `dirs`, and actual offsets immediately before
`readMultiSlice`, after transforms. It combines multiple bands and retains
invalid rays as null. A complete `grid.json` is promoted only after the render
and capture finish. Unsupported modes and existing capture directories fail.
The reader rejects invalid rays entirely; it does not silently mask or fill.

Decimal values are returned to producer float32 before the float64 audit,
preserving signed zero. The interval uses actual offset extrema, including
accumulation, without recentering. The report records capture/offset and shared
code hashes. Keep source and binary hashes, producing command and render logs
separately: a reader cannot attest that the renderer actually ran.

## Mathematics and limits

Positions and directions interpolate affinely inside each triangle:
`F(w,d) = sum_i w_i (q_i + d*n_i)`; directions are not renormalized inside
triangles. The two fixed cell diagonals are reported separately. Cross-product
area coefficients are quadratic in depth. The audit considers endpoints and
interior stationary minima, and cross-checks those values using displaced
vertices. Numerical precision, zero threshold, residual limit (1e-8), triangle
ordering and tie handling are unchanged from the retained research diagnostic.

This is a **surrogate between-pixel map**, not a claim about QuadSurface's
interpolation. It is a float64 diagnostic, not interval-arithmetic certification.
Positive base orientation is required. Zero local failures do not establish
global injectivity, nonintersection, anatomical sheet identity, CT support, ink,
or legible letters. The tool neither gates nor changes renderer output.

## Provenance and evidence scope

The shared `geometry.py`, `slab.py`, and `capture.py` were mechanically
extracted from the existing research modules. Workspace compatibility callers
import this same implementation; there is no second active numerical solver.
Archives are frozen distribution snapshots, not development copies.

The frozen C++ patch was built and tested in the full Linux amd64 renderer.
Seven fixed real PHerc0800 CT cases produced 35 byte-identical TIFF pairs
(940,800 pixels) with capture off/on in the same patched binary. That comparison
is instrumentation identity, not a speed benchmark or an unpatched-vs-patched
comparison. The research chart is not independently validated anatomy.
The source archive contains code/tests/docs, not CT data or model weights.
Full upstream CI, full macOS renderer build, independent adoption and recovered
letters are not established. See the accompanying review evidence for exact
input hashes, earlier failures, and the bounded August-overlap assessment.
