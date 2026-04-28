# TIFXYZ format (Volume Cartographer)

TIFXYZ is a directory-based surface format representing a quad-grid (2D lattice) of 3D points stored as 3 single-channel TIFF images (`x.tif`, `y.tif`, `z.tif`) plus JSON metadata (`meta.json`). Additional per-vertex channels may be stored as extra `*.tif` files in the same directory.

This document is intended to be sufficient to implement compatible readers/writers in other languages.

## 1. Container layout

TIFXYZ is a folder (not a single file) containing at least:

- `x.tif` (single-channel)
- `y.tif` (single-channel)
- `z.tif` (single-channel)
- `meta.json`

Optional common files:

- `mask.tif` (single- or multi-channel; channel 0 used as validity mask)
- `generations.tif` (usually `uint16`)

Any other `*.tif` files are interpreted as additional channels (by filename stem) and are not required for core compatibility.

## 2. Grid geometry & indexing

- `x.tif`, `y.tif`, `z.tif` must have identical width/height.
- Pixel coordinate `(row=y, col=x)` corresponds to one grid vertex.
- The corresponding 3D point is `P = (X, Y, Z)` where:

	- `X = x.tif[row, col]`
	- `Y = y.tif[row, col]`
	- `Z = z.tif[row, col]`

The grid is a *regular* 2D lattice in index space; physical spacing of the grid in “surface units” is given by `meta.json.scale`.

## 3. Data types (TIFF)

### 3.1 Coordinate TIFFs (`x.tif`, `y.tif`, `z.tif`)

- Must be **single-sample** (SamplesPerPixel = 1).
- Can be stored as any of the following sample formats:

	- IEEE float: 32-bit or 64-bit
	- Unsigned int: 8/16/32-bit
	- Signed int: 8/16/32-bit

- Readers should convert samples to `float32`/`float64` internally.

Writers should prefer `float32` for coordinates.

### 3.2 Extra channel TIFFs

Extra `*.tif` files may be any type supported by the toolchain; readers should not assume a specific dtype unless they recognize a channel.

### 3.3 Tiling

TIFFs may be tiled or scanline-based. Readers must support either.

## 4. Validity conventions

Each grid vertex can be “valid” (present) or “invalid” (absent).

### 4.1 Invalid sentinel in point grid

Internally, invalid points are represented as:

- `(-1, -1, -1)`

This sentinel is what many tools write/expect in memory.

### 4.2 Load-time invalidation rule

When loading, the implementation invalidates points with `Z <= 0` (sets them to `(-1, -1, -1)`), regardless of `X` and `Y`.

Implications:

- If you write valid points, ensure `Z > 0`.
- If you need valid points at non-positive Z, this convention will currently drop them.

### 4.3 Optional `mask.tif`

If `mask.tif` exists and is not explicitly ignored by the caller, it further invalidates vertices.

- Conceptually, `mask.tif` is a “validity mask”. Values >= 255 are treated as “keep”; values < 255 invalidate.
- Only **channel 0** is considered for validity. If the TIFF is multi-channel, channel 0 of each pixel is the validity value.

Resolution / scaling:

- `mask.tif` may be the same resolution as `x/y/z`, or an integer multiple in each axis.
- If `mask` is higher resolution by a factor `scaleX = maskW / W` and `scaleY = maskH / H` (each must be an integer), then a mask pixel at `(srcX, srcY)` maps to point `(dstX=srcX/scaleX, dstY=srcY/scaleY)`.
- Any mask pixel marked “invalid” invalidates its mapped point.

Note:

- Non-integer scaling is ignored by the loader (mask may effectively be skipped).

## 5. Metadata (`meta.json`)

`meta.json` is a JSON object. At minimum, writers must provide:

- `format`: string, must be `"tifxyz"`
- `scale`: array of 2 numbers `[sx, sy]`

Common fields produced by tools:

- `uuid`: string identifier
- `type`: e.g. `"seg"`
- `bbox`: `[[minX, minY, minZ], [maxX, maxY, maxZ]]`

### 5.1 `scale` meaning

`scale = [sx, sy]` describes the grid spacing in surface-parameter space.

- Many tools interpret the parametric coordinate of a vertex `(row, col)` as `(u = col * sx, v = row * sy)`.
- Some operations may also use `1/sx` and `1/sy` as a “pixels-per-unit” scaling.

To stay compatible:

- Preserve `scale` when copying/transforming surfaces.
- If you resample the grid resolution, update `scale` consistently.

## 6. Writing rules (recommended)

To produce surfaces that load consistently:

- Write `x.tif`, `y.tif`, `z.tif` as single-channel float32 images of identical size.
- For invalid vertices, either:

	- write `Z <= 0` (and any X/Y), or
	- write `(-1, -1, -1)` and ensure `Z <= 0` (recommended to match loader rule).

- Write `meta.json` with `format` and `scale`.
- If writing `mask.tif`, use `uint8` with 255 meaning “keep” and 0 meaning “invalidate”.

## 7. Reading rules (must-haves)

A compatible reader should:

- Read `meta.json` first and parse `scale`.
- Load `x.tif`, `y.tif`, `z.tif` and convert samples to float.
- Invalidate any vertex with `Z <= 0`.
- If `mask.tif` exists, and mask scaling is an integer multiple, invalidate vertices wherever mask channel 0 < 255.
- Treat any other `*.tif` file as an optional named per-vertex channel.

## 8. Reference implementation pointers

- Writer populates `meta.json.format = "tifxyz"` & `meta.json.scale = [sx, sy]`: [`core/src/QuadSurface.cpp:729`](core/src/QuadSurface.cpp:729)
- Reader loads `x/y/z.tif`, invalidates `Z <= 0`, then applies optional `mask.tif`: [`core/src/QuadSurface.cpp:848`](core/src/QuadSurface.cpp:848)

