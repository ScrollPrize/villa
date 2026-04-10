// vc_import — convert TIFF stacks into canonical shard format.
//
// Shard = 8 slab videos of 1024x1024 × 128 frames H.265.
// Volumes are padded to 1024 multiples.
//
// Usage:
//   vc_import --input /path/to/tiffs --output volume/ [--qp 26] [--voxel-size 7.91]

#include "codec.hpp"
#include "json.hpp"
#include "shard.hpp"
#include "thread_pool.hpp"

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

static vc::Vec2i read_tiff_slice(const fs::path& path, std::vector<uint8_t>& out) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    uint32_t w = 0, h = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    uint16_t bps = 8;
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    out.resize(size_t(w) * h);
    if (bps == 8) {
        for (uint32_t y = 0; y < h; ++y)
            TIFFReadScanline(tif, out.data() + y * w, y);
    } else if (bps == 16) {
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

static std::vector<fs::path> collect_tiffs(const fs::path& dir) {
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".tif" || ext == ".tiff") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static int pad1024(int v) { return (v + 1023) & ~1023; }

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::println("usage: vc_import --input <tiff_dir> --output <volume_dir> [--qp 26] [--voxel-size 7.91]");
        return 1;
    }

    fs::path input, output;
    int qp = 26;
    double voxel_size = 1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i+1 < argc) input = argv[++i];
        else if (a == "--output" && i+1 < argc) output = argv[++i];
        else if (a == "--qp" && i+1 < argc) qp = std::stoi(argv[++i]);
        else if (a == "--voxel-size" && i+1 < argc) voxel_size = std::stod(argv[++i]);
    }

    auto tiffs = collect_tiffs(input);
    if (tiffs.empty()) { std::println(stderr, "no tiffs in {}", input.string()); return 1; }

    std::vector<uint8_t> slice_buf;
    auto [W, H] = read_tiff_slice(tiffs[0], slice_buf);
    int Z = int(tiffs.size());
    int pW = pad1024(W), pH = pad1024(H), pZ = pad1024(Z);
    std::println("input: {}x{}x{} → padded: {}x{}x{}", W, H, Z, pW, pH, pZ);

    // Write meta.json
    fs::create_directories(output);
    auto meta = vc::Json::object();
    auto shape = vc::Json::array();
    shape.push(pZ); shape.push(pH); shape.push(pW);
    meta.set("shape", std::move(shape));
    meta.set("voxel_size", voxel_size);
    meta.set("levels", 1);
    meta.dump_to_file(output / "meta.json");

    // Read all slices into padded volume buffer
    // Layout: volume[z * pH * pW + y * pW + x]
    std::println("reading {} slices...", Z);
    std::vector<uint8_t> volume(size_t(pW) * pH * pZ, 0);
    for (int z = 0; z < Z; ++z) {
        read_tiff_slice(tiffs[size_t(z)], slice_buf);
        for (int y = 0; y < H; ++y)
            memcpy(volume.data() + size_t(z) * pW * pH + size_t(y) * pW,
                   slice_buf.data() + size_t(y) * W, size_t(W));
        if ((z+1) % 100 == 0) std::println("  read {}/{}", z+1, Z);
    }

    // Encode shards. Each shard = 8 slab videos.
    // For a single-shard-wide volume (pW=1024, pH=1024), there's one shard
    // per z-block of 1024. For wider volumes, we'd need multiple shards
    // in x/y — but each shard's video is always 1024x1024.
    // For now: assume pW=pH=1024 (single shard column). TODO: multi-shard x/y.

    int nsz = pZ / vc::SHARD_DIM;
    std::println("encoding {} shards ({} slabs each)...", nsz, vc::SLABS_PER);
    fs::create_directories(output / "0");

    vc::ThreadPool pool(std::thread::hardware_concurrency());
    std::atomic<int> done{0};

    for (int sz = 0; sz < nsz; ++sz) {
        pool.enqueue([&, sz] {
            std::array<std::vector<uint8_t>, vc::SLABS_PER> slabs;

            for (int slab = 0; slab < vc::SLABS_PER; ++slab) {
                int z_base = sz * vc::SHARD_DIM + slab * vc::SLAB_FRAMES;

                // Check if entire slab is zeros
                bool all_zero = true;
                size_t slab_size = size_t(vc::SLAB_FRAMES) * pW * pH;
                auto* slab_data = volume.data() + size_t(z_base) * pW * pH;
                for (size_t i = 0; i < slab_size && all_zero; ++i)
                    if (slab_data[i]) all_zero = false;

                if (!all_zero)
                    slabs[size_t(slab)] = vc::h265_encode_slab(
                        std::span(slab_data, slab_size), qp);
            }

            auto path = vc::shard_path(output, 0, sz, 0, 0);
            vc::write_shard(path, std::span<const std::vector<uint8_t>, vc::SLABS_PER>(slabs));
            std::println("  shard {}/{}", done.fetch_add(1) + 1, nsz);
        });
    }

    pool.wait_idle();
    std::println("done: {}", output.string());
}
