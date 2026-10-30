# GPU sampling for vc_render_tifxyz

Opt-in tool that moves the trilinear sampling of `vc_render_tifxyz` (`sampleTileSlices`)
to a CUDA or ROCm GPU. Surface generation, normals, flip and normalize stay in the
unchanged C++ binary. Without `RENDER_GPU=1`, or without PyTorch and a GPU, `render_band.sh` runs the normal
CPU render.

## Files

| File | Purpose |
| --- | --- |
| `render_shim.cpp` | `LD_PRELOAD` shim. Interposes `sampleTileSlices<u8>` and `VcDataset::writeChunkSkipEmpty`, dumps each tile's base points and unit step directions to `RENDER_SHIM_DIR`, and skips the CPU sampling. |
| `render_gpu.py` | Reads the dumps, keeps volume chunks in a GPU pool, samples, writes level 0 and pyramid levels 1-5 into the zarr the C++ run created. |
| `rgpu_kernel.py` | Fused sampling kernel, compiled on first use with `torch.utils.cpp_extension`. |
| `render_band.sh` | Renders a list of surfaces; with `RENDER_GPU=1` runs the dumps and one `render_gpu.py` process side by side so the chunk pool is shared across surfaces. |
| `cmp_zarr.py` | Compares one level of two zarrs. |

## Build

The shim is built by hand, not by the CMake build:

```bash
g++ -O2 -std=c++17 -shared -fPIC render_shim.cpp -o render_shim.so $(pkg-config --cflags opencv4) -ldl
```

It only needs the OpenCV headers: the `cv::Mat` symbols it uses are resolved from
`vc_render_tifxyz`, which is already linked against OpenCV.

`render_gpu.py` needs PyTorch with CUDA or ROCm and numcodecs; `cmp_zarr.py` needs zarr.

## Usage

```bash
# list.txt: one "SURFACE_DIR OUT_ZARR" per line, no spaces in paths
RENDER_GPU=1 RGPU_POOL_GB=10 VC_RENDER_TIFXYZ=/path/to/vc_render_tifxyz \
  ./render_band.sh list.txt /path/to/volume.zarr
```

Other env: `RENDER_N` (31), `RENDER_FLIP` (1), `RGPU_DEVICE` (`cuda:0`), `PYTHON` (`python3`).

Run by hand, the shim also takes `RENDER_SHIM_MODE=both`: the CPU sampler runs as well, so
one `vc_render_tifxyz` run writes the CPU reference zarr and the dump.

## Limits

- Local OME-Zarr volume with 128³ u8 chunks; a remote volume falls back to the CPU.
- `-g 0`, u8 output, no composite, no accumulate offsets, no rotate or flip-axis options.
- A pool smaller than the chunks one surface touches (about 7 GB for a 6,000 px wide
  surface) means re-reading chunks from disk for every surface.
- The shim matches `sampleTileSlices` and `VcDataset::writeChunkSkipEmpty` by mangled
  name, so `vc_core` must be a shared library (the default, `BUILD_SHARED_LIBS=ON`).
  It relies on each sample call being followed by one write on the same thread.
  Other programs that inherit `LD_PRELOAD` (`nice`, `bash -c`) are left alone. If the
  renderer's signatures change, the dump comes out empty and `render_gpu.py` stops with an
  error. A native hook in `vc_render_tifxyz` (dump or sampler callback) would be cleaner.

## Parity

The kernel reproduces `ChunkSampler::sampleTrilinear`: the same fp32 `fma` chain
(built with `--fmad=false` on CUDA, `-ffp-contract=off` on ROCm), bounds test,
truncation and clamping. All-zero chunks are skipped like `writeChunkSkipEmpty`, and
levels 1-5 use the same rounding as `renderTiles`. The output is intended to be
byte-identical to the CPU render at every level; check it with:

```bash
RENDER_GPU=0 ./render_band.sh cpu_list.txt VOLUME
RENDER_GPU=1 ./render_band.sh gpu_list.txt VOLUME
for l in 0 1 2 3 4 5; do python3 cmp_zarr.py CPU_OUT.zarr GPU_OUT.zarr $l; done   # expect "identical": true
```
