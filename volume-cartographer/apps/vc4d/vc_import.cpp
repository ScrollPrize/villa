// vc_import — convert volume data into the canonical libvc shard format.
//
// Inputs: directory of TIFF slices, zarr v2 array, or single multi-page TIFF.
// Output: volume.zarr/ with 1024^3 shards of 128^3 H265 chunks, padded.
//
// Usage:
//   vc_import --input /path/to/tiffs --output volume.zarr [--qp 26] [--voxel-size 7.91]
//   vc_import --input /path/to/zarr  --output volume.zarr [--qp 26]

#include "codec.hpp"
#include "json.hpp"
#include "shard.hpp"
#include "thread_pool.hpp"
#include "types.hpp"

#include <tiffio.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Read a single TIFF slice as a flat u8 buffer. Returns {width, height}.
static vc::Vec2i read_tiff_slice(const fs::path& path, std::vector<uint8_t>& out) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    uint32_t w = 0, h = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

    uint16_t bps = 8, spp = 1, fmt = SAMPLEFORMAT_UINT;
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);

    out.resize(size_t(w) * h);

    if (bps == 8 && spp == 1) {
        // Direct u8 read
        for (uint32_t y = 0; y < h; ++y)
            TIFFReadScanline(tif, out.data() + y * w, y);
    } else if (bps == 16 && spp == 1) {
        // u16 → u8 (divide by 257)
        std::vector<uint16_t> row16(w);
        for (uint32_t y = 0; y < h; ++y) {
            TIFFReadScanline(tif, row16.data(), y);
            for (uint32_t x = 0; x < w; ++x)
                out[y * w + x] = uint8_t(row16[x] / 257);
        }
    }

    TIFFClose(tif);
    return {int(w), int(h)};
}

// Collect and sort TIFF files from a directory.
static std::vector<fs::path> collect_tiffs(const fs::path& dir) {
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".tif" || ext == ".tiff")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Round up to next multiple of 1024.
static int pad1024(int v) { return (v + 1023) & ~1023; }

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::println("usage: vc_import --input <path> --output <zarr_dir> [--qp 26] [--voxel-size 7.91]");
        return 1;
    }

    fs::path input, output;
    int qp = 26;
    double voxel_size = 1.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else if (arg == "--qp" && i + 1 < argc) qp = std::stoi(argv[++i]);
        else if (arg == "--voxel-size" && i + 1 < argc) voxel_size = std::stod(argv[++i]);
    }

    // Collect input slices
    auto tiffs = collect_tiffs(input);
    if (tiffs.empty()) { std::println(stderr, "no tiff files found in {}", input.string()); return 1; }

    // Read first slice to get dimensions
    std::vector<uint8_t> slice_buf;
    auto [W, H] = read_tiff_slice(tiffs[0], slice_buf);
    int Z = int(tiffs.size());
    std::println("input: {}x{}x{} ({} slices)", W, H, Z, tiffs.size());

    // Padded dimensions (always multiple of 1024)
    int pW = pad1024(W), pH = pad1024(H), pZ = pad1024(Z);
    std::println("padded: {}x{}x{}", pW, pH, pZ);

    // Write meta.json
    fs::create_directories(output);
    auto meta = vc::Json::object();
    meta.set("shape", vc::Json::array());
    // shape is [Z, Y, X]
    auto shape = vc::Json::array();
    shape.push(pZ); shape.push(pH); shape.push(pW);
    meta.set("shape", std::move(shape));
    meta.set("voxel_size", voxel_size);
    meta.set("levels", 1);  // TODO: build pyramid
    meta.dump_to_file(output / "meta.json");

    // Process shard by shard
    int nsx = pW / vc::SHARD_DIM;
    int nsy = pH / vc::SHARD_DIM;
    int nsz = pZ / vc::SHARD_DIM;
    std::println("shards: {}x{}x{} = {} total", nsx, nsy, nsz, nsx * nsy * nsz);

    // Create level 0 directory
    fs::create_directories(output / "0");

    // Read all slices into memory (simple approach — could stream for huge volumes)
    std::println("reading {} slices...", Z);
    std::vector<uint8_t> volume(size_t(pW) * pH * pZ, 0);
    for (int z = 0; z < Z; ++z) {
        read_tiff_slice(tiffs[size_t(z)], slice_buf);
        // Copy into padded volume
        for (int y = 0; y < H; ++y)
            memcpy(volume.data() + size_t(z) * pW * pH + size_t(y) * pW,
                   slice_buf.data() + size_t(y) * W, size_t(W));
        if ((z + 1) % 100 == 0) std::println("  read {}/{}", z + 1, Z);
    }

    // Encode shards
    vc::ThreadPool pool(std::thread::hardware_concurrency());
    std::atomic<int> shards_done{0};
    int total_shards = nsx * nsy * nsz;

    for (int sz = 0; sz < nsz; ++sz)
    for (int sy = 0; sy < nsy; ++sy)
    for (int sx = 0; sx < nsx; ++sx) {
        pool.enqueue([&, sz, sy, sx] {
            std::array<std::vector<uint8_t>, vc::INDEX_COUNT> chunks;

            for (int cz = 0; cz < vc::CHUNKS_PER; ++cz)
            for (int cy = 0; cy < vc::CHUNKS_PER; ++cy)
            for (int cx = 0; cx < vc::CHUNKS_PER; ++cx) {
                // Extract 128^3 chunk from volume
                std::vector<uint8_t> raw(vc::CHUNK_DIM * vc::CHUNK_DIM * vc::CHUNK_DIM);
                int gz0 = sz * vc::SHARD_DIM + cz * vc::CHUNK_DIM;
                int gy0 = sy * vc::SHARD_DIM + cy * vc::CHUNK_DIM;
                int gx0 = sx * vc::SHARD_DIM + cx * vc::CHUNK_DIM;

                bool all_zero = true;
                for (int lz = 0; lz < vc::CHUNK_DIM; ++lz)
                for (int ly = 0; ly < vc::CHUNK_DIM; ++ly) {
                    auto* src = volume.data() +
                        size_t(gz0 + lz) * pW * pH +
                        size_t(gy0 + ly) * pW +
                        gx0;
                    auto* dst = raw.data() +
                        size_t(lz) * vc::CHUNK_DIM * vc::CHUNK_DIM +
                        size_t(ly) * vc::CHUNK_DIM;
                    memcpy(dst, src, vc::CHUNK_DIM);
                    if (all_zero) {
                        for (int x = 0; x < vc::CHUNK_DIM; ++x)
                            if (dst[x]) { all_zero = false; break; }
                    }
                }

                int idx = vc::ShardIndex::linear(cz, cy, cx);
                if (!all_zero)
                    chunks[size_t(idx)] = vc::h265_encode(raw, qp);
                // else: leave empty → offset=0, length=0 in index
            }

            auto path = vc::shard_path(output, 0, sz, sy, sx);
            vc::write_shard(path, std::span<const std::vector<uint8_t>, vc::INDEX_COUNT>(chunks));

            int done = shards_done.fetch_add(1) + 1;
            if (done % 10 == 0 || done == total_shards)
                std::println("  encoded {}/{} shards", done, total_shards);
        });
    }

    pool.wait_idle();
    std::println("done: {}", output.string());
    return 0;
}
