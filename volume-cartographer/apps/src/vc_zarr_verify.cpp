// vc_zarr_verify: Verify H265 shard quality against raw source chunks.
//
// Downloads random shards from the output, decodes each inner chunk,
// fetches the corresponding raw source chunk, and computes error stats.
//
// Usage:
//   vc_zarr_verify <input> <output> [--samples N] [--level L]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "utils/Json.hpp"
#include "utils/video_codec.hpp"
#include "utils/http_fetch.hpp"
#include "utils/zarr.hpp"

using Json = utils::Json;

static constexpr size_t SHARD_DIM = 1024;
static constexpr size_t CHUNK_DIM = 128;
static constexpr size_t CHUNKS_PER_SHARD = SHARD_DIM / CHUNK_DIM;
static constexpr size_t INNER_CHUNKS = CHUNKS_PER_SHARD * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD;
static constexpr size_t CHUNK_VOXELS = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;
static constexpr size_t INDEX_BYTES = INNER_CHUNKS * 16;

// ---- I/O backend (same as recompress) ----

struct IOBackend {
    virtual ~IOBackend() = default;
    virtual std::vector<std::byte> read(const std::string& key) = 0;
    virtual std::string read_string(const std::string& key) = 0;
    virtual bool exists(const std::string& key) = 0;
};

struct S3Backend : IOBackend {
    std::string base_url;
    std::unique_ptr<utils::HttpClient> client;

    S3Backend(const std::string& s3_url) {
        auto parsed = utils::parse_s3_url(s3_url);
        if (!parsed) throw std::runtime_error("Invalid S3 URL: " + s3_url);
        base_url = utils::s3_to_https(*parsed);
        while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();
        utils::HttpClient::Config cfg;
        cfg.aws_auth = utils::AwsAuth::load();
        if (!parsed->region.empty()) cfg.aws_auth.region = parsed->region;
        if (cfg.aws_auth.region.empty()) cfg.aws_auth.region = "us-east-1";
        cfg.transfer_timeout = std::chrono::seconds(120);
        cfg.max_retries = 3;
        client = std::make_unique<utils::HttpClient>(std::move(cfg));
    }

    std::string url(const std::string& key) { return base_url + "/" + key; }

    std::vector<std::byte> read(const std::string& key) override {
        auto resp = client->get(url(key));
        if (!resp.ok())
            throw std::runtime_error("S3 GET failed (" + std::to_string(resp.status_code) + "): " + url(key));
        return std::move(resp.body);
    }

    std::string read_string(const std::string& key) override {
        auto data = read(key);
        return std::string(reinterpret_cast<const char*>(data.data()), data.size());
    }

    bool exists(const std::string& key) override {
        auto resp = client->head(url(key));
        return resp.ok();
    }
};

static uint64_t read_le64(const std::byte* src) {
    uint64_t val = 0;
    for (int i = 0; i < 8; ++i)
        val |= static_cast<uint64_t>(static_cast<uint8_t>(src[i])) << (8 * i);
    return val;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: vc_zarr_verify <input-zarr> <output-zarr> [--samples N] [--level L]\n");
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int num_samples = 10;
    int level = 0;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--samples" && i + 1 < argc) num_samples = std::atoi(argv[++i]);
        else if (arg == "--level" && i + 1 < argc) level = std::atoi(argv[++i]);
    }

    auto src = std::make_unique<S3Backend>(input_path);
    auto dst = std::make_unique<S3Backend>(output_path);

    // Read source metadata
    std::string zarray_str = src->read_string(std::to_string(level) + "/.zarray");
    Json zarray = Json::parse(zarray_str);

    std::vector<size_t> shape;
    for (auto& v : zarray["shape"]) shape.push_back(v.get_size_t());

    std::string dim_sep = "/";
    if (zarray.contains("dimension_separator"))
        dim_sep = zarray["dimension_separator"].get_string();

    size_t out_nz = (shape[0] + CHUNK_DIM - 1) / CHUNK_DIM;
    size_t out_ny = (shape[1] + CHUNK_DIM - 1) / CHUNK_DIM;
    size_t out_nx = (shape[2] + CHUNK_DIM - 1) / CHUNK_DIM;

    size_t shard_nz = (shape[0] + SHARD_DIM - 1) / SHARD_DIM;
    size_t shard_ny = (shape[1] + SHARD_DIM - 1) / SHARD_DIM;
    size_t shard_nx = (shape[2] + SHARD_DIM - 1) / SHARD_DIM;

    printf("Level %d: shape [%zu, %zu, %zu]\n", level, shape[0], shape[1], shape[2]);
    printf("Shard grid: %zux%zux%zu, chunk grid: %zux%zux%zu\n",
           shard_nz, shard_ny, shard_nx, out_nz, out_ny, out_nx);
    printf("Sampling %d shards...\n\n", num_samples);

    // Pick random shard coords
    std::mt19937 rng(42);
    struct ShardPos { size_t sz, sy, sx; };
    std::vector<ShardPos> all_shards;
    for (size_t sz = 0; sz < shard_nz; sz++)
        for (size_t sy = 0; sy < shard_ny; sy++)
            for (size_t sx = 0; sx < shard_nx; sx++)
                all_shards.push_back({sz, sy, sx});

    std::shuffle(all_shards.begin(), all_shards.end(), rng);
    if ((int)all_shards.size() > num_samples)
        all_shards.resize(num_samples);

    // Aggregate error stats
    std::vector<int> all_errors;
    size_t total_compared = 0;
    double total_abs_error = 0;
    int chunks_compared = 0;
    int chunks_skipped = 0;

    for (auto& [sz, sy, sx] : all_shards) {
        std::string shard_key = std::to_string(level) + "/c/" +
            std::to_string(sz) + "/" + std::to_string(sy) + "/" + std::to_string(sx);

        printf("Shard %s: ", shard_key.c_str());
        fflush(stdout);

        std::vector<std::byte> shard_data;
        try {
            shard_data = dst->read(shard_key);
        } catch (...) {
            printf("not found, skipping\n");
            continue;
        }

        if (shard_data.size() < INDEX_BYTES) {
            printf("too small (%zu bytes), skipping\n", shard_data.size());
            continue;
        }

        // Parse index (first 8192 bytes)
        int shard_chunks = 0;
        int shard_zero = 0;
        int shard_missing = 0;
        double shard_abs_error = 0;
        size_t shard_voxels = 0;
        std::vector<int> shard_errors;

        for (size_t iz = 0; iz < CHUNKS_PER_SHARD; iz++) {
            for (size_t iy = 0; iy < CHUNKS_PER_SHARD; iy++) {
                for (size_t ix = 0; ix < CHUNKS_PER_SHARD; ix++) {
                    size_t inner_idx = iz * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD +
                                       iy * CHUNKS_PER_SHARD + ix;

                    uint64_t offset = read_le64(shard_data.data() + inner_idx * 16);
                    uint64_t nbytes = read_le64(shard_data.data() + inner_idx * 16 + 8);

                    // Missing chunk
                    if (offset == ~uint64_t(0) && nbytes == ~uint64_t(0)) {
                        shard_missing++;
                        continue;
                    }

                    // Zero sentinel
                    if (offset == (~uint64_t(0) - 1) && nbytes == 0) {
                        shard_zero++;
                        continue;
                    }

                    // Decode H265
                    if (offset + nbytes > shard_data.size()) {
                        fprintf(stderr, "  chunk %zu/%zu/%zu: offset+size out of bounds\n", iz, iy, ix);
                        continue;
                    }

                    utils::VideoCodecParams params;
                    params.depth = CHUNK_DIM;
                    params.height = CHUNK_DIM;
                    params.width = CHUNK_DIM;

                    std::vector<std::byte> decoded;
                    try {
                        decoded = utils::video_decode(
                            std::span<const std::byte>(shard_data.data() + offset, nbytes),
                            CHUNK_VOXELS, params);
                    } catch (std::exception& e) {
                        fprintf(stderr, "  chunk %zu/%zu/%zu: decode failed: %s\n", iz, iy, ix, e.what());
                        continue;
                    }

                    // Fetch raw source chunk
                    size_t cz = sz * CHUNKS_PER_SHARD + iz;
                    size_t cy = sy * CHUNKS_PER_SHARD + iy;
                    size_t cx = sx * CHUNKS_PER_SHARD + ix;

                    if (cz >= out_nz || cy >= out_ny || cx >= out_nx) {
                        chunks_skipped++;
                        continue;
                    }

                    std::string src_key = std::to_string(level) + "/" +
                        std::to_string(cz) + dim_sep +
                        std::to_string(cy) + dim_sep +
                        std::to_string(cx);

                    std::vector<std::byte> raw;
                    try {
                        raw = src->read(src_key);
                    } catch (...) {
                        // Source chunk doesn't exist — should be all zeros
                        raw.resize(CHUNK_VOXELS, std::byte{0});
                    }

                    // Pad raw to 128³ if needed
                    if (raw.size() < CHUNK_VOXELS) {
                        std::vector<std::byte> padded(CHUNK_VOXELS, std::byte{0});
                        std::memcpy(padded.data(), raw.data(), raw.size());
                        raw = std::move(padded);
                    }

                    // Compare
                    size_t n = std::min(decoded.size(), raw.size());
                    for (size_t i = 0; i < n; i++) {
                        int diff = std::abs((int)static_cast<uint8_t>(decoded[i]) -
                                            (int)static_cast<uint8_t>(raw[i]));
                        shard_abs_error += diff;
                        if (diff > 0) shard_errors.push_back(diff);
                    }
                    shard_voxels += n;
                    shard_chunks++;
                    chunks_compared++;
                }
            }
        }

        double mae = shard_voxels > 0 ? shard_abs_error / shard_voxels : 0;
        printf("%d chunks, %d zero, %d missing, MAE=%.3f",
               shard_chunks, shard_zero, shard_missing, mae);

        if (!shard_errors.empty()) {
            std::sort(shard_errors.begin(), shard_errors.end());
            int p90 = shard_errors[(size_t)(shard_errors.size() * 0.90)];
            int p95 = shard_errors[(size_t)(shard_errors.size() * 0.95)];
            int p99 = shard_errors[(size_t)(shard_errors.size() * 0.99)];
            int pmax = shard_errors.back();
            printf(", p90=%d p95=%d p99=%d max=%d", p90, p95, p99, pmax);
        }
        printf("\n");

        total_abs_error += shard_abs_error;
        total_compared += shard_voxels;
        all_errors.insert(all_errors.end(), shard_errors.begin(), shard_errors.end());
    }

    // Global stats
    printf("\n=== SUMMARY ===\n");
    printf("Chunks compared: %d\n", chunks_compared);
    printf("Total voxels: %zu (%.1f M)\n", total_compared, total_compared / 1e6);

    if (total_compared > 0) {
        double mae = total_abs_error / total_compared;
        printf("Global MAE: %.4f\n", mae);

        if (!all_errors.empty()) {
            std::sort(all_errors.begin(), all_errors.end());
            size_t n = all_errors.size();
            printf("Non-zero error voxels: %zu (%.2f%%)\n", n, 100.0 * n / total_compared);
            printf("p90=%d  p95=%d  p99=%d  max=%d\n",
                   all_errors[(size_t)(n * 0.90)],
                   all_errors[(size_t)(n * 0.95)],
                   all_errors[(size_t)(n * 0.99)],
                   all_errors.back());
        } else {
            printf("All voxels exact match!\n");
        }
    }

    return 0;
}
