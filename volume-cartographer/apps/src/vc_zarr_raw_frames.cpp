// vc_zarr_raw_frames: stream raw u8 z-slice frames from a (sharded) zarr
// dataset to stdout.  Pair with ffmpeg for on-the-fly video encoding:
//
//   vc_zarr_raw_frames <zarr-root> <level> [stride] \
//     | ffmpeg -f rawvideo -pix_fmt gray -s WxH -r 30 -i - \
//              -c:v libx264 -pix_fmt yuv420p out.mp4
//
// Frame dims are printed on stderr at startup.
//
// To avoid redecoding each 256³ inner chunk once per z-slice, the reader
// processes one full z-chunk-layer at a time: decode every (iy, ix) chunk
// in that layer in parallel, then emit every stride-th z within that
// layer directly from the decoded buffers.  Each chunk is decoded exactly
// once regardless of stride.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/c3d_codec.hpp"
#include "utils/zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

namespace {

struct Codec {
    std::function<std::vector<std::byte>(std::span<const std::byte>, std::size_t)> decompress;
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
            "usage: %s <zarr-root> <level> [stride]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path inRoot(argv[1]);
    const int level  = std::atoi(argv[2]);
    const int stride = (argc > 3) ? std::atoi(argv[3]) : 1;
    if (stride < 1) {
        std::fprintf(stderr, "stride must be >= 1\n");
        return 2;
    }

    // Use VcDataset only to parse metadata / expose shape; then bypass its
    // single-threaded readRegion and drive the underlying ZarrArray
    // ourselves so chunk decodes can go in parallel.
    auto ds = std::make_unique<vc::VcDataset>(inRoot / std::to_string(level));
    const auto& shape = ds->shape();
    if (shape.size() != 3) {
        std::fprintf(stderr, "expected 3D dataset, got %zuD\n", shape.size());
        return 1;
    }
    if (ds->getDtype() != vc::VcDtype::uint8) {
        std::fprintf(stderr, "only uint8 supported\n");
        return 1;
    }
    const size_t Z = shape[0], Y = shape[1], X = shape[2];

    // Inner chunk shape — we expect a sharded c3d array here with 256³
    // inner chunks.  Fall back to whatever VcDataset reports for unsharded
    // input.
    const auto chunkShape = ds->defaultChunkShape();
    if (chunkShape.size() != 3) {
        std::fprintf(stderr, "expected 3D chunk shape\n");
        return 1;
    }
    const size_t CZ = chunkShape[0], CY = chunkShape[1], CX = chunkShape[2];
    if (Y % CY != 0 || X % CX != 0) {
        std::fprintf(stderr,
            "volume X/Y (%zu/%zu) not a multiple of chunk X/Y (%zu/%zu)\n",
            X, Y, CX, CY);
        return 1;
    }
    const size_t nIY = Y / CY;
    const size_t nIX = X / CX;
    const size_t chunk_voxels = CZ * CY * CX;

    std::fprintf(stderr,
        "vc_zarr_raw_frames: X=%zu Y=%zu Z=%zu chunk=%zux%zux%zu stride=%d "
        "-> %zu frames\n",
        X, Y, Z, CZ, CY, CX, stride, (Z + stride - 1) / stride);

    // Decode one chunk from its c3d/raw bytes.  VcDataset exposes
    // readChunk(iz, iy, ix, void*) which handles both the sharded-codec
    // dispatch and fill-value when the chunk is absent.
    auto decode_chunk = [&](size_t iz, size_t iy, size_t ix,
                            std::vector<uint8_t>& buf)
    {
        buf.assign(chunk_voxels, 0);
        ds->readChunkOrFill(iz, iy, ix, buf.data());
    };

    const size_t nThreads = std::max<size_t>(1, std::thread::hardware_concurrency());

    // Output slice buffer (single writer, reused per frame).
    std::vector<uint8_t> slice(Y * X);

    for (size_t iz0 = 0; iz0 * CZ < Z; ++iz0) {
        // Decode the (nIY × nIX) chunks covering z-chunk-layer iz0, in
        // parallel across nThreads worker threads.
        std::vector<std::vector<uint8_t>> chunks(nIY * nIX);
        std::atomic<size_t> next{0};
        const size_t total = nIY * nIX;
        std::vector<std::thread> workers;
        workers.reserve(nThreads);
        for (size_t t = 0; t < nThreads; ++t) {
            workers.emplace_back([&]{
                for (;;) {
                    size_t idx = next.fetch_add(1);
                    if (idx >= total) break;
                    const size_t iy = idx / nIX;
                    const size_t ix = idx % nIX;
                    decode_chunk(iz0, iy, ix, chunks[idx]);
                }
            });
        }
        for (auto& w : workers) w.join();

        // Emit every stride-th z within [iz0*CZ, iz0*CZ+CZ).
        for (size_t dz = 0; dz < CZ; ++dz) {
            const size_t z = iz0 * CZ + dz;
            if (z >= Z) break;
            if (z % stride != 0) continue;

            // Assemble a full Y×X slice by copying chunkY × chunkX rows
            // from each (iy, ix) chunk at depth dz.
            for (size_t iy = 0; iy < nIY; ++iy) {
                for (size_t ix = 0; ix < nIX; ++ix) {
                    const auto& c = chunks[iy * nIX + ix];
                    // Source row-start: dz plane within the chunk.
                    const uint8_t* base = c.data() + dz * CY * CX;
                    uint8_t* dst = slice.data() + (iy * CY) * X + ix * CX;
                    for (size_t dy = 0; dy < CY; ++dy) {
                        std::memcpy(dst + dy * X, base + dy * CX, CX);
                    }
                }
            }

            if (std::fwrite(slice.data(), 1, slice.size(), stdout) != slice.size()) {
                std::fprintf(stderr, "\nshort write to stdout\n");
                return 1;
            }
        }
        std::fprintf(stderr, "\r[z-layer %zu/%zu]", iz0 + 1, (Z + CZ - 1) / CZ);
    }
    std::fprintf(stderr, "\n");
    return 0;
}
