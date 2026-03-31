# VC3D Web - Volume Viewer in the Browser

## Architecture

```
Browser
+-------------------------------------------+
|  index.html                                |
|  +---------------------------------------+ |
|  | JavaScript                            | |
|  | - Canvas/WebGL display                | |
|  | - Mouse/keyboard events               | |
|  | - fetch() zarr chunks from S3/HTTP    | |
|  | - Blosc/zstd decompression (JS/WASM)  | |
|  | - Pass raw voxel data to WASM         | |
|  +------------------+--------------------+ |
|                     | loadChunk / render   |
|  +------------------v--------------------+ |
|  | WASM Module (C++)                     | |
|  | - ChunkCache: in-memory LRU           | |
|  | - Renderer: trilinear plane sampling  | |
|  | - Output: RGBA pixel buffer           | |
|  +------------------+--------------------+ |
|                     | pixel data           |
|  +------------------v--------------------+ |
|  | WebGL                                 | |
|  | - Fullscreen quad + texture           | |
|  | - Upload RGBA pixels from WASM        | |
|  +---------------------------------------+ |
+-------------------------------------------+
```

## Data flow

1. User specifies a zarr volume URL (e.g., `https://s3.amazonaws.com/bucket/volume.zarr`)
2. JS fetches `/{level}/.zarray` metadata for each pyramid level
3. JS computes which chunks are visible given the current slice Z, zoom, and pan
4. JS fetches chunk files via `fetch()` (e.g., `/{level}/{iz}.{iy}.{ix}`)
5. JS decompresses chunks (blosc/zstd) into raw uint8 voxel data
6. JS passes raw data to WASM via `vc3d_load_chunk()`
7. WASM renders the slice plane via trilinear interpolation into RGBA buffer
8. JS uploads RGBA pixels to a WebGL texture and draws a fullscreen quad

## What's ported from VC3D

The WASM renderer is a simplified version of `Slicing.cpp::samplePlaneImpl`:
- Same plane parameterization: origin + vx_step + vy_step
- Same trilinear interpolation math
- Same chunk addressing: ZYX row-major, chunk_idx = voxel / chunk_dim
- Stripped: no OpenMP (single-threaded in WASM), no SIMD (could add later),
  no tricubic, no prefetch intrinsics

The chunk cache is a simplified version of `TieredChunkCache`:
- In-memory only (no disk tier, no warm/compressed tier)
- No IO threads (JS fetch is already async)
- Simple hash map with byte-budget eviction
- Chunks arrive pre-decompressed from JS

## Key design decisions

### Decompression in JS, not WASM
Zarr chunks are typically blosc-compressed (lz4 inner codec). Rather than
compiling c-blosc to WASM (adds ~200KB), we decompress in JavaScript.
Options:
- **blosc.js**: Pure JS blosc decoder (~15KB)
- **c-blosc WASM**: Compile blosc to a separate WASM module
- **Server-side**: Use CloudFront/Lambda@Edge to serve pre-decompressed chunks
- **Video codec**: Chunks may be VC3D video-compressed (H.264/AV1); use
  browser's native `VideoDecoder` API for zero-copy decode

### No H.264/video codec in initial prototype
The native VC3D app uses OpenH264 for chunk compression. For the web version:
- Browser's `WebCodecs VideoDecoder` can decode H.264 natively (very fast)
- Need to extract raw Y plane from decoded frames (grayscale volume data)
- This is a Phase 2 feature

### WebGL for display, not rendering
WebGL is only used to display the final RGBA pixel buffer (texture upload +
fullscreen quad). The actual volume sampling happens in WASM C++ code.
Future: could move sampling to a WebGL fragment shader for GPU acceleration.

### Single-threaded WASM (for now)
Emscripten supports pthreads via SharedArrayBuffer, but this requires:
- COOP/COEP headers on the server
- More complex build
- Phase 2: add Web Workers for parallel chunk sampling

## Exported WASM API

```c
void vc3d_init(int width, int height);
void vc3d_resize(int width, int height);
void vc3d_add_level(int sz, int sy, int sx, int cz, int cy, int cx);
void vc3d_reset_levels();
void vc3d_set_plane(float ox, float oy, float oz,
                    float vx_x, float vx_y, float vx_z,
                    float vy_x, float vy_y, float vy_z);
void vc3d_set_slice(float z, float scale, float panX, float panY);
void vc3d_set_level(int level);
void vc3d_set_cache_max(int maxBytes);
void vc3d_load_chunk(int level, int iz, int iy, int ix,
                     const uint8_t* data, int size);
const uint8_t* vc3d_render();
int vc3d_width();
int vc3d_height();
int vc3d_cache_count();
int vc3d_cache_bytes();
```

## Build

```bash
# Install Emscripten SDK (one time)
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source emsdk_env.sh

# Build
cd vc3d-web
./build.sh

# Run
cp index.html build/
cd build && python3 -m http.server 8080
# Open http://localhost:8080
```

## File structure

```
vc3d-web/
  index.html          - UI: canvas, toolbar, status bar, all JS
  CMakeLists.txt      - Emscripten build configuration
  build.sh            - Build script
  PLAN.md             - This file
  src/
    main.cpp          - WASM entry point, exported C functions
    renderer.hpp      - Renderer class declaration
    renderer.cpp      - Plane sampling + RGBA output
    chunk_cache.hpp   - In-memory chunk cache (header-only)
```

## Phase 2 roadmap

1. **Blosc decompression**: Integrate blosc.js or compile c-blosc to WASM
2. **Video codec support**: Use WebCodecs VideoDecoder for H.264-compressed chunks
3. **Multi-threaded rendering**: Emscripten pthreads for parallel sampling
4. **GPU rendering**: Move plane sampling to WebGL fragment shader
5. **Oblique slicing**: Arbitrary plane orientation (not just axis-aligned)
6. **Smooth scrolling**: Interpolate between Z slices during scroll
7. **Progressive loading**: Show low-resolution (high pyramid level) first,
   refine with higher resolution as chunks arrive
8. **URL state**: Encode view state (Z, zoom, pan, volume URL) in URL hash
9. **Touch support**: Pinch-to-zoom, two-finger pan on mobile
10. **Chunk prefetch**: Predict scroll direction and prefetch adjacent Z slices
