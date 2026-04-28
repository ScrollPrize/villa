// vc_zarr_recompress: Recompress zarr v2 volumes to zarr v3 with c3d sharding.
//
// Reads zarr v2 chunks (blosc/zstd/raw) from S3 or local filesystem,
// recompresses with c3d into zarr v3 shards (4096³ shards, 256³ inner chunks),
// writes zarr v3 output to S3 or local filesystem.
//
// Each 256³ inner chunk is c3d-encoded individually (C3DC header + c3d
// bitstream). Shards have a fixed-size index at the start (16 bytes per
// entry: u64 offset + u64 size, little-endian).
//
// Shard index encoding:
//   (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF) = missing chunk (not present)
//   (0xFFFFFFFFFFFFFFFE, 0)                  = zero chunk (all-zero data)
//   (offset, nbytes)                          = compressed chunk data
//
// Work is partitioned by output shard — each thread gets exclusive shards,
// so no two threads ever read/write the same input chunks or output shards.
//
// All pyramid levels (0-5) are processed in a single invocation.
//
// Usage:
//   vc_zarr_recompress <input> <output> [options]
//
// Input/output can be:
//   /path/to/volume.zarr          (local filesystem)
//   s3://bucket/path/volume.zarr  (S3, uses AWS env credentials)
//   s3+us-east-1://bucket/...     (S3 with explicit region)
//
// Options:
//   --target-ratio R  c3d target compression ratio [default: 50]
//                     (50 ≈ 40 dB PSNR on scroll CT)
//   --verify          Verify roundtrip (decode after encode)
//   --jobs N          Outer worker threads (shards in flight) [default: 8]
//   --inner-jobs K    Inner worker threads per shard (chunks in flight)
//                     [default: hardware_concurrency]
//   --log FILE        Log completed shards to this file [default: none]
//   --stats-pct N     Sample N% of encoded chunks and compute lossy-codec
//                     quality metrics (MAE, RMSE, PSNR, percentiles).
//                     Default 0 (disabled). 1-5 is cheap and representative.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <array>
#include <condition_variable>
#include <deque>
#include <random>
#include <set>
#include <sstream>
#include <future>
#include <map>
#include <unordered_set>
#include <chrono>

#include "utils/Json.hpp"
#include <blosc.h>

#include "utils/c3d_codec.hpp"
#include "utils/http_fetch.hpp"
#include "utils/zarr.hpp"

namespace fs = std::filesystem;
using Json = utils::Json;

// c3d canonical geometry: 4096³ shards with 256³ inner chunks.
static constexpr size_t SHARD_DIM = 4096;
static constexpr size_t CHUNK_DIM = 256;
static constexpr size_t CHUNKS_PER_SHARD = SHARD_DIM / CHUNK_DIM;                 // 16
static constexpr size_t INNER_CHUNKS = CHUNKS_PER_SHARD * CHUNKS_PER_SHARD
                                     * CHUNKS_PER_SHARD;                          // 4096
static constexpr size_t CHUNK_VOXELS = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;

// ============================================================================
// I/O abstraction: local filesystem or S3
// ============================================================================

struct IOBackend {
    virtual ~IOBackend() = default;
    virtual std::vector<std::byte> read(const std::string& key) = 0;
    virtual void write(const std::string& key, const std::vector<std::byte>& data) = 0;
    virtual void write_string(const std::string& key, const std::string& data) = 0;
    virtual std::string read_string(const std::string& key) = 0;
    virtual bool exists(const std::string& key) = 0;
    virtual std::vector<std::string> list_chunks(const std::string& prefix) = 0;
    virtual void write_from_file(const std::string& key, const std::string& file_path) = 0;
};

// --- Local filesystem backend ---
struct LocalBackend : IOBackend {
    fs::path root;
    explicit LocalBackend(const fs::path& r) : root(r) {}

    std::vector<std::byte> read(const std::string& key) override {
        auto p = root / key;
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open: " + p.string());
        auto sz = f.tellg();
        f.seekg(0);
        std::vector<std::byte> buf(sz);
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        return buf;
    }

    void write(const std::string& key, const std::vector<std::byte>& data) override {
        auto p = root / key;
        fs::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot write: " + p.string());
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    void write_string(const std::string& key, const std::string& data) override {
        auto p = root / key;
        fs::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot write: " + p.string());
        f << data;
    }

    std::string read_string(const std::string& key) override {
        auto p = root / key;
        std::ifstream f(p);
        if (!f) throw std::runtime_error("Cannot open: " + p.string());
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }

    void write_from_file(const std::string& key, const std::string& file_path) override {
        auto p = root / key;
        fs::create_directories(p.parent_path());
        std::error_code ec;
        fs::rename(file_path, p, ec);
        if (ec) {
            fs::copy_file(file_path, p, fs::copy_options::overwrite_existing);
            fs::remove(file_path);
        }
    }

    bool exists(const std::string& key) override {
        return fs::exists(root / key);
    }

    std::vector<std::string> list_chunks(const std::string& prefix) override {
        std::vector<std::string> result;
        auto dir = root / prefix;
        if (!fs::is_directory(dir)) return result;
        for (auto& entry : fs::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            auto fname = entry.path().filename().string();
            if (fname[0] == '.' || fname.find(".json") != std::string::npos) continue;
            result.push_back(fs::relative(entry.path(), root).string());
        }
        return result;
    }
};

// --- S3 backend ---
struct S3Backend : IOBackend {
    std::string base_url;  // https://bucket.s3.region.amazonaws.com/prefix
    std::unique_ptr<utils::HttpClient> client;

    S3Backend(const std::string& s3_url) {
        auto parsed = utils::parse_s3_url(s3_url);
        if (!parsed) throw std::runtime_error("Invalid S3 URL: " + s3_url);

        base_url = utils::s3_to_https(*parsed);
        // Remove trailing slash
        while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();

        // Configure client with AWS auth
        utils::HttpClient::Config cfg;
        cfg.aws_auth = utils::AwsAuth::load();
        if (!parsed->region.empty()) cfg.aws_auth.region = parsed->region;
        if (cfg.aws_auth.region.empty()) cfg.aws_auth.region = "us-east-1";
        cfg.transfer_timeout = std::chrono::seconds(120);
        cfg.max_retries = 3;
        client = std::make_unique<utils::HttpClient>(std::move(cfg));
    }

    std::string url(const std::string& key) {
        return base_url + "/" + key;
    }

    std::vector<std::byte> read(const std::string& key) override {
        auto resp = client->get(url(key));
        if (!resp.ok()) {
            throw std::runtime_error("S3 GET failed (" + std::to_string(resp.status_code) +
                                     "): " + url(key));
        }
        return std::move(resp.body);
    }

    void write(const std::string& key, const std::vector<std::byte>& data) override {
        auto resp = client->put(url(key), std::span<const std::byte>(data));
        if (!resp.ok()) {
            throw std::runtime_error("S3 PUT failed (" + std::to_string(resp.status_code) +
                                     "): " + url(key));
        }
    }

    void write_string(const std::string& key, const std::string& data) override {
        std::vector<std::byte> bytes(data.size());
        std::memcpy(bytes.data(), data.data(), data.size());
        write(key, bytes);
    }

    std::string read_string(const std::string& key) override {
        auto data = read(key);
        return std::string(reinterpret_cast<const char*>(data.data()), data.size());
    }

    void write_from_file(const std::string& key, const std::string& file_path) override {
        auto resp = client->put_file(url(key), file_path);
        if (!resp.ok()) {
            fs::remove(file_path);
            throw std::runtime_error("S3 PUT failed (" + std::to_string(resp.status_code) +
                                     "): " + url(key));
        }
        fs::remove(file_path);
    }

    bool exists(const std::string& key) override {
        auto resp = client->head(url(key));
        return resp.ok();
    }

    std::vector<std::string> list_chunks(const std::string& prefix) override {
        // S3 ListObjectsV2 via REST API
        std::vector<std::string> result;
        std::string continuation_token;

        auto after_scheme = base_url.substr(base_url.find("//") + 2);
        auto dot = after_scheme.find('.');
        auto bucket = after_scheme.substr(0, dot);
        auto host_end = after_scheme.find('/');
        auto host = after_scheme.substr(0, host_end);
        std::string root_prefix;
        if (host_end != std::string::npos) {
            root_prefix = after_scheme.substr(host_end + 1);
        }

        // Percent-encode a string for use as an S3 query parameter value.
        // S3's continuation-token contains raw '+', '/', '=' bytes that
        // collide with URL syntax; SigV4 canonicalization expects them
        // percent-encoded so the canonical query string matches the wire form.
        auto pct_encode = [](const std::string& s) {
            std::string out;
            out.reserve(s.size() * 3);
            for (unsigned char c : s) {
                bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
                                  || (c >= '0' && c <= '9') || c == '-' || c == '_'
                                  || c == '.' || c == '~';
                if (unreserved) {
                    out.push_back(static_cast<char>(c));
                } else {
                    static const char hex[] = "0123456789ABCDEF";
                    out.push_back('%');
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0xF]);
                }
            }
            return out;
        };

        std::string full_prefix = root_prefix.empty() ? prefix : root_prefix + "/" + prefix;
        std::string list_url_base =
            "https://" + host + "/?list-type=2&prefix=" + pct_encode(full_prefix);

        do {
            std::string list_url = list_url_base;
            if (!continuation_token.empty()) {
                list_url += "&continuation-token=" + pct_encode(continuation_token);
            }
            list_url += "&max-keys=10000";

            auto resp = client->get(list_url);
            if (!resp.ok()) {
                std::string body{reinterpret_cast<const char*>(resp.body.data()),
                                 std::min<size_t>(resp.body.size(), 1024)};
                throw std::runtime_error("S3 list failed: " + std::to_string(resp.status_code)
                                         + " url=" + list_url + " body=" + body);
            }

            auto body = std::string(resp.body_string());

            // Simple XML parsing for <Key>...</Key> and <NextContinuationToken>
            continuation_token.clear();
            size_t pos = 0;
            while (true) {
                auto key_start = body.find("<Key>", pos);
                if (key_start == std::string::npos) break;
                key_start += 5;
                auto key_end = body.find("</Key>", key_start);
                if (key_end == std::string::npos) break;
                auto full_key = body.substr(key_start, key_end - key_start);

                // Make relative to our root
                if (!root_prefix.empty() && full_key.starts_with(root_prefix + "/")) {
                    full_key = full_key.substr(root_prefix.size() + 1);
                }

                // Skip metadata files
                auto fname = full_key.substr(full_key.rfind('/') + 1);
                if (!fname.empty() && fname[0] != '.' &&
                    fname.find(".json") == std::string::npos &&
                    fname.find(".zarray") == std::string::npos &&
                    fname.find(".zattrs") == std::string::npos &&
                    fname.find(".zgroup") == std::string::npos) {
                    result.push_back(full_key);
                }
                pos = key_end + 6;
            }

            auto nct_start = body.find("<NextContinuationToken>");
            if (nct_start != std::string::npos) {
                nct_start += 23;
                auto nct_end = body.find("</NextContinuationToken>", nct_start);
                if (nct_end != std::string::npos) {
                    continuation_token = body.substr(nct_start, nct_end - nct_start);
                }
            }
        } while (!continuation_token.empty());

        return result;
    }
};

static std::unique_ptr<IOBackend> make_backend(const std::string& path) {
    if (utils::is_s3_url(path)) {
        return std::make_unique<S3Backend>(path);
    }
    return std::make_unique<LocalBackend>(path);
}

// ============================================================================
// Blosc decompression
// ============================================================================

static std::vector<std::byte> decompress_blosc(const std::vector<std::byte>& compressed,
                                                 size_t expected_size) {
    // Check blosc header magic (first byte 0x02)
    if (compressed.size() < 16 ||
        static_cast<uint8_t>(compressed[0]) != 0x02) {
        // Not blosc — assume raw
        return compressed;
    }

    // Read nbytes from blosc header (bytes 4-7, little-endian)
    size_t nbytes = 0;
    std::memcpy(&nbytes, reinterpret_cast<const char*>(compressed.data()) + 4, 4);

    std::vector<std::byte> output(nbytes);
    int ret = blosc_decompress(
        reinterpret_cast<const void*>(compressed.data()),
        reinterpret_cast<void*>(output.data()),
        nbytes);
    if (ret < 0) {
        throw std::runtime_error("blosc_decompress failed: " + std::to_string(ret));
    }
    output.resize(ret);
    return output;
}

// ============================================================================
// Zarr v3 metadata generation
// ============================================================================

static std::string make_zarr_v3_metadata(const std::vector<size_t>& shape,
                                         float target_ratio) {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = shape;
    meta.chunks = {SHARD_DIM, SHARD_DIM, SHARD_DIM};  // shard shape
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";  // "/" separator
    meta.node_type = "array";

    // Sharding config: SHARD_DIM³ shards with CHUNK_DIM³ inner chunks.
    utils::ShardConfig sc;
    sc.sub_chunks = {CHUNK_DIM, CHUNK_DIM, CHUNK_DIM};

    utils::ZarrCodecConfig codec_cfg;
    codec_cfg.name = "c3d";
    codec_cfg.configuration = std::make_shared<utils::JsonValue>(
        utils::JsonValue{{"target_ratio", Json((double)target_ratio)}});
    sc.sub_codecs.push_back(codec_cfg);
    meta.shard_config = sc;

    return utils::detail::serialize_zarr_json(meta);
}

static std::string make_zarr_v3_group() {
    Json root;
    root["zarr_format"] = 3;
    root["node_type"] = "group";
    return root.dump(2) + "\n";
}

static std::string make_zarr_v3_group_with_multiscales(
    const std::vector<int>& levels,
    const std::vector<std::vector<size_t>>& shapes)
{
    Json root;
    root["zarr_format"] = 3;
    root["node_type"] = "group";

    // OME-Zarr multiscales attribute
    Json axes = Json::array();
    axes.push_back(Json({{"name", "z"}, {"type", "space"}}));
    axes.push_back(Json({{"name", "y"}, {"type", "space"}}));
    axes.push_back(Json({{"name", "x"}, {"type", "space"}}));

    Json datasets = Json::array();
    for (size_t i = 0; i < levels.size(); i++) {
        double scale = std::pow(2.0, levels[i]);
        Json ds;
        ds["path"] = std::to_string(levels[i]);
        Json scale_arr = Json::array();
        scale_arr.push_back(scale); scale_arr.push_back(scale); scale_arr.push_back(scale);
        Json ct = Json::array();
        ct.push_back(Json({{"type", "scale"}, {"scale", scale_arr}}));
        ds["coordinateTransformations"] = ct;
        datasets.push_back(ds);
    }

    Json ms;
    ms["version"] = "0.4";
    ms["name"] = "/";
    ms["axes"] = axes;
    ms["datasets"] = datasets;

    Json ms_arr = Json::array();
    ms_arr.push_back(ms);
    root["attributes"] = Json({{"multiscales", ms_arr}});
    return root.dump(2) + "\n";
}

// ============================================================================
// Shard building: write_le64 for shard index
// ============================================================================

static void write_le64(std::byte* dst, uint64_t val) {
    for (int i = 0; i < 8; ++i)
        dst[i] = static_cast<std::byte>((val >> (8 * i)) & 0xFF);
}

static bool is_all_zero(const std::vector<std::byte>& data) {
    auto* p = reinterpret_cast<const uint64_t*>(data.data());
    size_t n64 = data.size() / 8;
    for (size_t i = 0; i < n64; i++)
        if (p[i]) return false;
    for (size_t i = n64 * 8; i < data.size(); i++)
        if (data[i] != std::byte{0}) return false;
    return true;
}

// Zero chunk sentinel: offset = 0xFFFFFFFFFFFFFFFE, nbytes = 0
static void write_zero_sentinel(std::byte* dst) {
    write_le64(dst, ~uint64_t(0) - 1);
    write_le64(dst + 8, 0);
}

// ============================================================================
// Occupancy: which source chunks exist on storage
// ============================================================================

// Fast path: a single S3 LIST (paginated) returns every existing key under
// the level's prefix.  We parse the trailing "<cz>/<cy>/<cx>" out of each
// key and mark the corresponding chunk as occupied.  No per-chunk HEAD/GET,
// no separate-level mask scan — one round-trip per 10K chunks.
// Load a coordinator-built occupancy bitmap from /dev/shm or any file.
// Format: 12-byte header (u32 nz, u32 ny, u32 nx, little-endian), then
// packed bitset of nz*ny*nx bits (ceil(N/8) bytes).
// Dims must match the level's (nz, ny, nx) or we throw.
static std::vector<bool> load_occupancy_file(const std::string& path,
                                              size_t nz, size_t ny, size_t nx)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("occupancy-file: cannot open " + path);
    uint32_t header[3];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f) throw std::runtime_error("occupancy-file: short read on header");
    if (header[0] != nz || header[1] != ny || header[2] != nx) {
        throw std::runtime_error("occupancy-file: dims mismatch (expected "
            + std::to_string(nz) + "/" + std::to_string(ny) + "/"
            + std::to_string(nx) + ", got "
            + std::to_string(header[0]) + "/" + std::to_string(header[1]) + "/"
            + std::to_string(header[2]) + ")");
    }
    const size_t bits = nz * ny * nx;
    const size_t bytes = (bits + 7) / 8;
    std::vector<uint8_t> packed(bytes);
    f.read(reinterpret_cast<char*>(packed.data()), bytes);
    if (!f) throw std::runtime_error("occupancy-file: short read on bitmap");
    std::vector<bool> mask(bits, false);
    for (size_t i = 0; i < bits; ++i)
        mask[i] = (packed[i >> 3] >> (i & 7)) & 1;
    return mask;
}

static std::vector<bool> build_occupancy_from_listing(
    IOBackend& io, int level,
    const std::vector<size_t>& shape, const std::vector<size_t>& chunks,
    int parallelism = 32)
{
    if (shape.size() < 3) return {};
    size_t nz = (shape[0] + chunks[0] - 1) / chunks[0];
    size_t ny = (shape[1] + chunks[1] - 1) / chunks[1];
    size_t nx = (shape[2] + chunks[2] - 1) / chunks[2];
    std::vector<bool> mask(nz * ny * nx, false);

    auto t0 = std::chrono::steady_clock::now();

    // Fan out one paginated LIST per cz prefix (e.g. "0/123/").  Each LIST
    // returns up to ~ny*nx keys (16K for L0 cy×cx = 256×256 = 65K worst case
    // → ~7 pages).  With `parallelism` workers in flight we hide RTT.
    std::atomic<size_t> next_cz{0};
    std::atomic<size_t> parsed{0};
    std::mutex mask_mtx;

    auto worker = [&]() {
        for (;;) {
            size_t cz = next_cz.fetch_add(1);
            if (cz >= nz) break;
            std::string prefix = std::to_string(level) + "/" + std::to_string(cz) + "/";
            std::vector<std::string> keys;
            try {
                keys = io.list_chunks(prefix);
            } catch (...) {
                continue;
            }
            std::vector<std::pair<size_t,size_t>> local;  // (cy, cx)
            local.reserve(keys.size());
            for (const auto& k : keys) {
                size_t s3 = k.rfind('/');
                if (s3 == std::string::npos) continue;
                size_t s2 = k.rfind('/', s3 - 1);
                if (s2 == std::string::npos) continue;
                try {
                    size_t cy = std::stoul(k.substr(s2 + 1, s3 - s2 - 1));
                    size_t cx = std::stoul(k.substr(s3 + 1));
                    if (cy < ny && cx < nx) local.emplace_back(cy, cx);
                } catch (...) {}
            }
            if (!local.empty()) {
                std::lock_guard lk(mask_mtx);
                for (auto [cy, cx] : local) {
                    mask[cz * ny * nx + cy * nx + cx] = true;
                }
                parsed.fetch_add(local.size());
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(parallelism);
    for (int i = 0; i < parallelism; ++i) workers.emplace_back(worker);
    for (auto& t : workers) t.join();

    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t p = parsed.load();
    printf("  Occupancy LIST: %zu / %zu chunks present (%.1f%% sparse) in %.1fs\n",
           p, mask.size(), 100.0 * (1.0 - (double)p / mask.size()), dt);
    return mask;
}

static std::vector<bool> build_occupancy_mask(
    IOBackend& io,
    int level,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& chunks,
    int mask_level,
    const std::vector<size_t>& mask_shape,
    const std::vector<size_t>& mask_chunks)
{
    if (shape.size() < 3 || mask_shape.size() < 3) return {};

    int scale = 1 << (mask_level - level);

    size_t nz = (shape[0] + chunks[0] - 1) / chunks[0];
    size_t ny = (shape[1] + chunks[1] - 1) / chunks[1];
    size_t nx = (shape[2] + chunks[2] - 1) / chunks[2];
    size_t total = nz * ny * nx;

    std::vector<bool> mask(total, false);

    size_t mask_nz = (mask_shape[0] + mask_chunks[0] - 1) / mask_chunks[0];
    size_t mask_ny = (mask_shape[1] + mask_chunks[1] - 1) / mask_chunks[1];
    size_t mask_nx = (mask_shape[2] + mask_chunks[2] - 1) / mask_chunks[2];

    printf("  Building occupancy mask from level %d (%zu chunks to scan)...\n",
           mask_level, mask_nz * mask_ny * mask_nx);

    std::string mask_prefix = std::to_string(mask_level) + "/";
    std::string dim_sep = "/";

    for (size_t mz = 0; mz < mask_nz; mz++) {
        for (size_t my = 0; my < mask_ny; my++) {
            for (size_t mx = 0; mx < mask_nx; mx++) {
                std::string key = mask_prefix + std::to_string(mz) + dim_sep +
                                  std::to_string(my) + dim_sep + std::to_string(mx);

                std::vector<std::byte> raw;
                try {
                    auto data = io.read(key);
                    raw = decompress_blosc(data, mask_chunks[0] * mask_chunks[1] * mask_chunks[2]);
                } catch (...) {
                    continue;
                }

                size_t cz = mask_chunks[0], cy = mask_chunks[1], cx = mask_chunks[2];
                size_t z0 = mz * cz, y0 = my * cy, x0 = mx * cx;
                size_t z1 = std::min(z0 + cz, mask_shape[0]);
                size_t y1 = std::min(y0 + cy, mask_shape[1]);
                size_t x1 = std::min(x0 + cx, mask_shape[2]);

                for (size_t z = z0; z < z1; z++) {
                    for (size_t y = y0; y < y1; y++) {
                        for (size_t x = x0; x < x1; x++) {
                            size_t local_z = z - z0, local_y = y - y0, local_x = x - x0;
                            size_t idx = local_z * cy * cx + local_y * cx + local_x;
                            if (idx < raw.size() && raw[idx] != std::byte{0}) {
                                size_t tz = (z * scale) / chunks[0];
                                size_t ty = (y * scale) / chunks[1];
                                size_t tx = (x * scale) / chunks[2];
                                if (tz < nz && ty < ny && tx < nx) {
                                    mask[tz * ny * nx + ty * nx + tx] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    size_t occupied = 0;
    for (auto b : mask) if (b) occupied++;
    printf("  Mask: %zu / %zu chunks occupied (%.1f%% sparse)\n",
           occupied, total, 100.0 * (1.0 - (double)occupied / total));

    return mask;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    setlinebuf(stdout);
    if (argc < 3) {
        std::cerr << "Usage: vc_zarr_recompress <input> <output> [options]\n"
                  << "\n"
                  << "Input/output: local path or s3://bucket/path\n"
                  << "\n"
                  << "Options:\n"
                  << "  --target-ratio R  c3d target compression ratio (>1.0). [50]\n"
                  << "  --verify         Verify roundtrip after encoding\n"
                  << "  --jobs N         Outer workers (shards in flight) [8]\n"
                  << "  --inner-jobs K   Inner workers per shard (chunks in flight)\n"
                  << "                   [default: hardware_concurrency]\n"
                  << "  --log FILE       Log completed shards to file\n"
                  << "  --stats-pct N    Sample N%% of chunks for quality metrics\n"
                  << "                   (MAE, RMSE, PSNR, percentiles).  [0 = off]\n"
                  << "  --levels CSV     Process only listed levels (e.g. 4,5). [default: all]\n"
                  << "  --rank N         VM index in fanout (0..world-1). [default: 0]\n"
                  << "  --world N        Total VM count for horizontal scaling. [default: 1]\n"
                  << "                   With --world 4, VM 0 handles sz%%4==0, VM 1 sz%%4==1, etc.\n"
                  << "  --one-shard L/sz/sy/sx  Process exactly one shard + exit.\n"
                  << "                          Skips occupancy and resume lists. Coordinator\n"
                  << "                          handles both. Use for per-shard process fanout.\n"
                  << "  --shard-file PATH       File of 'L/sz/sy/sx' lines — batch many shards\n"
                  << "                          in one process (amortize encoder init + TCP reuse).\n"
                  << "  --encode-jobs N  Encode pool threads per worker [default: 2*cores]\n"
                  << "  --occupancy-file PATH  Binary bitmap of input chunk existence\n"
                  << "                          (nz*ny*nx bits). Skips S3 LIST entirely.\n"
                  << "                          {L} in path is replaced by level number.\n";
        return 1;
    }

    blosc_init();

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    bool verify = false;
    // Outer workers process shards in parallel. Since the per-shard work is
    // now mostly parallel internally (see --inner-jobs), a modest outer count
    // is enough to hide per-shard upload latency. Default 8.
    int jobs = 8;
    // Inner workers do concurrent chunk download + decode + encode within
    // a single shard. Scale with hardware_concurrency to saturate network
    // RTT (one chunk fetch ~30-80 ms over S3) and CPU on encode.
    int inner_jobs = std::max(1, (int)std::thread::hardware_concurrency());
    std::string log_path;
    int stats_pct = 0;
    std::string levels_arg;    // empty = all discovered; else CSV of levels
    int rank = 0;              // this VM's index in the fanout (0..world-1)
    int world = 1;             // total VM count for horizontal scaling
    std::string one_shard_arg; // "L/sz/sy/sx" — process exactly one shard + exit
    std::string shard_file;    // file of "L/sz/sy/sx" lines — batch in one process
    int encode_jobs = 0;       // 0 = 2*hw_concurrency (default)
    std::string occupancy_file; // external occupancy bitmap (coordinator-built)
    float target_ratio = 50.0f; // c3d target compression ratio (>1.0)

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verify") verify = true;
        else if (arg == "--jobs" && i + 1 < argc) jobs = std::atoi(argv[++i]);
        else if (arg == "--inner-jobs" && i + 1 < argc) inner_jobs = std::atoi(argv[++i]);
        else if (arg == "--log" && i + 1 < argc) log_path = argv[++i];
        else if (arg == "--stats-pct" && i + 1 < argc) stats_pct = std::atoi(argv[++i]);
        else if (arg == "--levels" && i + 1 < argc) levels_arg = argv[++i];
        else if (arg == "--rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
        else if (arg == "--world" && i + 1 < argc) world = std::atoi(argv[++i]);
        else if (arg == "--one-shard" && i + 1 < argc) one_shard_arg = argv[++i];
        else if (arg == "--shard-file" && i + 1 < argc) shard_file = argv[++i];
        else if (arg == "--encode-jobs" && i + 1 < argc) encode_jobs = std::atoi(argv[++i]);
        else if (arg == "--occupancy-file" && i + 1 < argc) occupancy_file = argv[++i];
        else if (arg == "--target-ratio" && i + 1 < argc) target_ratio = (float)std::atof(argv[++i]);
        else if (arg == "--codec" || arg == "--qp" || arg == "--air-clamp" || arg == "--bit-shift") {
            // Legacy h265 flags: removed in the c3d-only switch. Fail fast
            // so orchestration scripts that still pass them don't silently
            // inherit --target-ratio defaults and produce mis-encoded data.
            fprintf(stderr,
                "Error: flag %s was removed along with the H.265 codec path.\n"
                "       The recompress tool is c3d-only now; use --target-ratio "
                "instead of --qp, and drop --codec / --air-clamp / --bit-shift.\n",
                arg.c_str());
            return 1;
        }
        else {
            fprintf(stderr, "Error: unknown argument %s\n", arg.c_str());
            return 1;
        }
    }
    if (!(target_ratio > 1.0f)) {
        fprintf(stderr, "--target-ratio must be > 1.0, got %g\n", (double)target_ratio);
        return 1;
    }
    if (stats_pct < 0) stats_pct = 0;
    if (stats_pct > 100) stats_pct = 100;
    // In --one-shard mode, force the level filter to the shard's level.
    if (!one_shard_arg.empty()) {
        int ol = -1;
        std::sscanf(one_shard_arg.c_str(), "%d", &ol);
        if (ol >= 0 && ol <= 5) levels_arg = std::to_string(ol);
    }
    // Parse --shard-file into a map of level -> list of (sz, sy, sx).
    // Drives both levels_arg (so only listed levels are processed) and the
    // shard_positions filter inside each level.
    struct ShardPos { size_t sz, sy, sx; };
    std::map<int, std::vector<ShardPos>> shard_file_entries;
    if (!shard_file.empty()) {
        std::ifstream sf(shard_file);
        if (!sf) {
            fprintf(stderr, "Cannot open --shard-file: %s\n", shard_file.c_str());
            return 1;
        }
        std::string line;
        std::set<int> seen_levels;
        while (std::getline(sf, line)) {
            if (line.empty()) continue;
            int L = -1;
            size_t sz = 0, sy = 0, sx = 0;
            if (std::sscanf(line.c_str(), "%d/%zu/%zu/%zu", &L, &sz, &sy, &sx) != 4) {
                fprintf(stderr, "bad shard-file line: %s\n", line.c_str());
                return 1;
            }
            shard_file_entries[L].push_back({sz, sy, sx});
            seen_levels.insert(L);
        }
        // Override levels_arg to cover exactly the levels in the file.
        std::string csv;
        for (int L : seen_levels) {
            if (!csv.empty()) csv += ",";
            csv += std::to_string(L);
        }
        levels_arg = csv;
    }
    if (world < 1) world = 1;
    if (rank < 0 || rank >= world) {
        fprintf(stderr, "Invalid --rank %d (must be 0..%d)\n", rank, world - 1);
        return 1;
    }

    // Open shared log file for shard completion logging
    std::mutex log_mtx;
    std::ofstream log_file;
    if (!log_path.empty()) {
        log_file.open(log_path, std::ios::app);
        if (!log_file) {
            fprintf(stderr, "Cannot open log file: %s\n", log_path.c_str());
            return 1;
        }
    }

    auto input = make_backend(input_path);
    auto output = make_backend(output_path);

    printf("Input:  %s\n", input_path.c_str());
    printf("Output: %s\n", output_path.c_str());
    printf("Codec: c3d (target-ratio %.2f), shard: %zu³, chunk: %zu³\n",
           (double)target_ratio, SHARD_DIM, CHUNK_DIM);
    printf("Outer jobs: %d  |  Inner jobs/shard: %d\n", jobs, inner_jobs);
    if (world > 1) printf("Fanout: rank %d / world %d (sz %% %d == %d)\n",
                           rank, world, world, rank);
    printf("\n");

    // Discover pyramid levels (all 6 by default, or a --levels CSV subset).
    // Preserve the CSV order so callers can request e.g. "5,4,3,2,1,0" to
    // finish small levels first (available for testing) before L0.
    std::vector<int> levels_order;
    std::set<int> levels_filter;
    if (!levels_arg.empty()) {
        std::stringstream ss(levels_arg);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) {
                int l = std::atoi(tok.c_str());
                if (levels_filter.insert(l).second) levels_order.push_back(l);
            }
    }
    std::vector<int> levels;
    std::vector<std::vector<size_t>> shapes;
    std::vector<Json> zarrays; // parallel with levels/shapes; cached from discovery

    for (int l = 0; l < 6; l++) {
        if (!levels_filter.empty() && !levels_filter.count(l)) continue;
        std::string zarray_key = std::to_string(l) + "/.zarray";
        std::string zarr_json_key = std::to_string(l) + "/zarr.json";

        Json zarray;
        try {
            if (input->exists(zarray_key)) {
                zarray = Json::parse(input->read_string(zarray_key));
            } else if (input->exists(zarr_json_key)) {
                zarray = Json::parse(input->read_string(zarr_json_key));
            } else {
                continue;
            }
        } catch (...) { continue; }

        std::vector<size_t> shape;
        if (zarray.contains("shape")) {
            for (auto& v : zarray["shape"]) shape.push_back(v.get_size_t());
        }

        // Pad each dimension to next multiple of 1024 (minimum 1024)
        std::vector<size_t> padded_shape = shape;
        for (auto& d : padded_shape)
            d = std::max(SHARD_DIM, ((d + SHARD_DIM - 1) / SHARD_DIM) * SHARD_DIM);

        levels.push_back(l);
        shapes.push_back(padded_shape);
        zarrays.push_back(zarray);
        printf("Found level %d: shape [%zu, %zu, %zu] -> padded [%zu, %zu, %zu]\n",
               l,
               shape.size() > 0 ? shape[0] : 0,
               shape.size() > 1 ? shape[1] : 0,
               shape.size() > 2 ? shape[2] : 0,
               padded_shape.size() > 0 ? padded_shape[0] : 0,
               padded_shape.size() > 1 ? padded_shape[1] : 0,
               padded_shape.size() > 2 ? padded_shape[2] : 0);
    }

    if (levels.empty()) {
        std::cerr << "No pyramid levels found\n";
        return 1;
    }

    // Write zarr v3 root metadata (coordinator writes it once in one-shard
    // mode, so workers skip this).
    if (one_shard_arg.empty() && shard_file.empty()) {
        output->write_string("zarr.json",
            make_zarr_v3_group_with_multiscales(levels, shapes));
    }

    // Reorder levels for processing based on --levels CSV (user may request
    // e.g. 5,4,3,2,1,0 to see small levels first).
    if (!levels_order.empty()) {
        std::vector<int> new_levels;
        std::vector<std::vector<size_t>> new_shapes;
        std::vector<Json> new_zarrays;
        for (int l : levels_order) {
            auto it = std::find(levels.begin(), levels.end(), l);
            if (it == levels.end()) continue;
            size_t idx = it - levels.begin();
            new_levels.push_back(levels[idx]);
            new_shapes.push_back(shapes[idx]);
            new_zarrays.push_back(zarrays[idx]);
        }
        levels = std::move(new_levels);
        shapes = std::move(new_shapes);
        zarrays = std::move(new_zarrays);
    }

    for (size_t li = 0; li < levels.size(); li++) {
        int l = levels[li];
        auto& shape = shapes[li];

        printf("\n=== Level %d ===\n", l);

        // Write per-level zarr v3 metadata (skipped in one-shard mode — the
        // coordinator writes it once before fanning out workers).
        if (one_shard_arg.empty() && shard_file.empty()) {
            output->write_string(std::to_string(l) + "/zarr.json",
                                  make_zarr_v3_metadata(shape, target_ratio));
        }

        // Reuse cached .zarray from discovery phase (saves a GET per worker).
        std::string level_prefix = std::to_string(l) + "/";
        Json zarray = zarrays[li];

        std::string compressor_id;
        if (zarray.contains("compressor") && !zarray["compressor"].is_null()) {
            compressor_id = zarray["compressor"].value("id", "");
        }

        std::vector<size_t> src_chunks = {128, 128, 128};
        if (zarray.contains("chunks")) {
            src_chunks.clear();
            for (auto& v : zarray["chunks"]) src_chunks.push_back(v.get_size_t());
        }

        std::string dim_sep = "/";
        if (zarray.contains("dimension_separator")) {
            dim_sep = zarray["dimension_separator"].get_string();
        }

        // Detect dtype
        bool is_u16 = false;
        if (zarray.contains("dtype") && zarray["dtype"].is_string()) {
            std::string dt = zarray["dtype"].get_string();
            std::string_view sv = dt;
            if (!sv.empty() && (sv[0] == '<' || sv[0] == '>' || sv[0] == '|'))
                sv.remove_prefix(1);
            is_u16 = (sv == "u2");
        }

        // Source chunk voxel count
        size_t src_chunk_voxels = 1;
        for (auto d : src_chunks) src_chunk_voxels *= d;
        size_t src_raw_bytes = is_u16 ? src_chunk_voxels * 2 : src_chunk_voxels;

        // Source chunks must tile the output chunk evenly.  Output chunks
        // larger than source (common: 128³ source → 256³ c3d output) are
        // assembled by the downloader; sources larger than output are not
        // supported by this tool.
        for (int d = 0; d < 3; ++d) {
            if (src_chunks[d] == 0 || CHUNK_DIM % src_chunks[d] != 0
                || src_chunks[d] > CHUNK_DIM) {
                fprintf(stderr,
                    "level %d: incompatible source chunk shape [%zu,%zu,%zu] "
                    "for output chunk dim %zu (must divide evenly and be <= output)\n",
                    l, src_chunks[0], src_chunks[1], src_chunks[2], CHUNK_DIM);
                return 1;
            }
        }

        // Shard grid dimensions
        size_t shard_nz = (shape[0] + SHARD_DIM - 1) / SHARD_DIM;
        size_t shard_ny = (shape[1] + SHARD_DIM - 1) / SHARD_DIM;
        size_t shard_nx = (shape[2] + SHARD_DIM - 1) / SHARD_DIM;
        size_t total_shards = shard_nz * shard_ny * shard_nx;

        // Build a global per-chunk occupancy bitmap up front via parallel
        // per-cz LISTs.  Workers will look up bits instead of doing 8 LISTs
        // per shard.  Skipped in --one-shard mode: the coordinator already
        // filtered to this shard and the inner download loop falls back to
        // per-chunk GETs with 404 being cheap (~50 ms each, 128 parallel).
        // Occupancy mask is indexed by SOURCE chunk grid (that's what the
        // S3 listing returns).  When source chunk size == output chunk
        // size these two grids coincide; when src is smaller (e.g. 128³
        // source → 256³ c3d output), one output chunk is "present" if
        // any of its covering source chunks exists — checked below.
        std::vector<bool> occ_mask;
        size_t occ_nz = (shape[0] + src_chunks[0] - 1) / src_chunks[0];
        size_t occ_ny = (shape[1] + src_chunks[1] - 1) / src_chunks[1];
        size_t occ_nx = (shape[2] + src_chunks[2] - 1) / src_chunks[2];
        if (!occupancy_file.empty()) {
            // Substitute the level number if path contains {L}
            std::string p = occupancy_file;
            auto pos = p.find("{L}");
            if (pos != std::string::npos) p.replace(pos, 3, std::to_string(l));
            occ_mask = load_occupancy_file(p, occ_nz, occ_ny, occ_nx);
        } else if (one_shard_arg.empty() && shard_file.empty()) {
            occ_mask = build_occupancy_from_listing(*input, l, shape, src_chunks, 64);
        }

        printf("  Source: chunks [%zu,%zu,%zu], compressor: %s, sep: '%s', dtype: %s\n",
               src_chunks[0], src_chunks[1], src_chunks[2],
               compressor_id.empty() ? "none" : compressor_id.c_str(),
               dim_sep.c_str(), is_u16 ? "uint16" : "uint8");
        printf("  Output: %zu shards (%zux%zux%zu), %zu inner chunks each\n",
               total_shards, shard_nz, shard_ny, shard_nx, INNER_CHUNKS);

        // Number of 128³ chunks in the output volume
        size_t out_nz = (shape[0] + CHUNK_DIM - 1) / CHUNK_DIM;
        size_t out_ny = (shape[1] + CHUNK_DIM - 1) / CHUNK_DIM;
        size_t out_nx = (shape[2] + CHUNK_DIM - 1) / CHUNK_DIM;

        // Build list of shard positions to process.  With --world > 1 we
        // only take sz planes where (sz % world == rank).  Interleaving by
        // sz gives each VM a mix of dense and empty regions so wall times
        // stay balanced.
        std::vector<ShardPos> shard_positions;
        if (!one_shard_arg.empty()) {
            // Parse "L/sz/sy/sx" — L already filtered via levels_arg.
            size_t sz = 0, sy = 0, sx = 0;
            int parsed_L = -1;
            std::sscanf(one_shard_arg.c_str(), "%d/%zu/%zu/%zu",
                        &parsed_L, &sz, &sy, &sx);
            if (parsed_L != l) continue;  // this level isn't our target
            if (sz >= shard_nz || sy >= shard_ny || sx >= shard_nx) {
                fprintf(stderr, "--one-shard %s: out of range\n", one_shard_arg.c_str());
                return 1;
            }
            shard_positions.push_back({sz, sy, sx});
        } else if (!shard_file.empty()) {
            // Only process shards listed in the file for this level.
            auto it = shard_file_entries.find(l);
            if (it != shard_file_entries.end()) {
                for (auto& sp : it->second) {
                    if (sp.sz < shard_nz && sp.sy < shard_ny && sp.sx < shard_nx)
                        shard_positions.push_back(sp);
                }
            }
        } else {
            for (size_t sz = 0; sz < shard_nz; sz++) {
                if (world > 1 && (sz % (size_t)world) != (size_t)rank) continue;
                for (size_t sy = 0; sy < shard_ny; sy++)
                    for (size_t sx = 0; sx < shard_nx; sx++)
                        shard_positions.push_back({sz, sy, sx});
            }
            if (world > 1) {
                printf("  Fanout: this VM owns %zu of %zu shards\n",
                       shard_positions.size(), shard_nz * shard_ny * shard_nx);
            }
        }

        // List existing shards for resume (skipped in one-shard mode: the
        // coordinator handles resume by skipping this whole invocation).
        std::set<std::string> existing_shards;
        if (one_shard_arg.empty() && shard_file.empty()) {
            std::string shard_prefix = std::to_string(l) + "/c/";
            auto keys = output->list_chunks(shard_prefix);
            for (auto& k : keys) {
                auto rel = k.substr(shard_prefix.size());
                existing_shards.insert(shard_prefix + rel);
            }
            printf("  Resume LIST: %zu existing shards in S3 (will skip on resume)\n",
                   existing_shards.size());
            fflush(stdout);
        }

        const size_t INDEX_BYTES = INNER_CHUNKS * 16;

        std::atomic<size_t> total_raw{0}, total_compressed{0};
        std::atomic<int> processed_shards{0}, processed_chunks{0};
        std::atomic<int> errs{0}, skipped_chunks{0}, zero_chunks{0};
        std::atomic<int> verify_ok{0}, verify_fail{0};
        std::mutex print_mtx;

        // Lossy-codec quality sampling. Histogram is 256 bins — one per
        // possible abs(uint8 - uint8) value — summed across sampled chunks.
        // Aggregating this way avoids per-voxel storage; percentiles fall
        // out of a cumulative walk at end-of-level.
        std::array<std::atomic<uint64_t>, 256> stats_err_hist{};
        std::atomic<uint64_t> stats_voxels_total{0};
        std::atomic<uint64_t> stats_chunks_sampled{0};

        auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next_shard{0};

        // Async upload queue. Outer workers hand assembled shards to this
        // queue and immediately begin the next shard's downloads, so the
        // S3 PUT of shard N overlaps with the S3 GETs of shard N+1.
        // Bounded to (jobs + up_jobs) items so we don't let shard bytes pile
        // up in memory faster than we can drain them.
        struct UploadJob {
            std::string key;
            std::vector<std::byte> bytes;
        };
        const int up_jobs = std::max(1, jobs);
        const size_t up_queue_max = static_cast<size_t>(jobs) + up_jobs;
        std::deque<UploadJob> upload_queue;
        std::mutex up_mtx;
        std::condition_variable up_not_empty;
        std::condition_variable up_not_full;
        bool up_done = false;

        auto enqueue_upload = [&](std::string key, std::vector<std::byte>&& bytes) {
            std::unique_lock lk(up_mtx);
            up_not_full.wait(lk, [&]{ return upload_queue.size() < up_queue_max; });
            upload_queue.push_back({std::move(key), std::move(bytes)});
            up_not_empty.notify_one();
        };

        auto upload_worker = [&]() {
            auto u_output = make_backend(output_path);
            for (;;) {
                UploadJob job;
                {
                    std::unique_lock lk(up_mtx);
                    up_not_empty.wait(lk, [&]{ return up_done || !upload_queue.empty(); });
                    if (upload_queue.empty()) return;
                    job = std::move(upload_queue.front());
                    upload_queue.pop_front();
                    up_not_full.notify_one();
                }
                try {
                    u_output->write(job.key, job.bytes);
                } catch (const std::exception& e) {
                    std::lock_guard lk(print_mtx);
                    fprintf(stderr, "  UPLOAD FAIL: %s (%s)\n", job.key.c_str(), e.what());
                    errs.fetch_add(1);
                }
            }
        };

        std::vector<std::thread> upload_threads;
        upload_threads.reserve(up_jobs);
        for (int i = 0; i < up_jobs; i++) upload_threads.emplace_back(upload_worker);

        // ============================================================
        // Per-level pipeline: download pool + encode pool + upload pool.
        // Each worker lives for the whole level so backend/curl/decoder
        // setup is paid once. Chunks from multiple shards are interleaved
        // in the queues, so downloads of shard N+1 overlap with encodes
        // of shard N.
        // ============================================================

        enum ResultKind : uint8_t {
            RESULT_NONE = 0,         // slot empty (chunk missing/failed read)
            RESULT_ZERO = 1,         // all-zero chunk, write zero sentinel
            RESULT_COMPRESSED = 2,   // newly h265-encoded bytes in result_data
            RESULT_PASSTHROUGH = 3,  // already-encoded VC3D bytes in result_data
        };

        struct ShardState {
            std::string shard_key;
            std::vector<std::byte> index_bytes;
            // One entry per job slot (parallel arrays, no shared writers per j).
            std::vector<std::vector<std::byte>> result_data;
            std::vector<uint8_t> result_kind;
            std::vector<size_t> result_inner_idx;
            std::atomic<int> remaining{0};
            std::atomic<bool> any_data{false};
        };

        struct DownloadTask {
            std::shared_ptr<ShardState> shard;
            size_t job_idx;
            // Base source-chunk coords that anchor this output chunk, plus
            // the per-axis ratio (CHUNK_DIM / src_chunk_dim) of source
            // chunks along each axis.  When ratio is {1,1,1} the worker
            // reads exactly one source chunk (original 1:1 path); when it
            // is {2,2,2} (typical: 128³ source → 256³ c3d output) the
            // worker fetches 8 source chunks and assembles them into the
            // output-chunk voxel grid.
            size_t src_base_z, src_base_y, src_base_x;
            size_t rz, ry, rx;
        };

        struct EncodeTask {
            std::shared_ptr<ShardState> shard;
            size_t job_idx;
            std::vector<std::byte> raw;      // 128^3 bytes ready to encode
        };

        std::deque<DownloadTask> dl_q;
        std::mutex dl_mtx;
        std::condition_variable dl_cv;
        bool dl_done = false;

        std::deque<EncodeTask> enc_q;
        std::mutex enc_mtx;
        std::condition_variable enc_cv;
        bool enc_done = false;
        // Bound enc_q so downloaders can't race ahead of the encode pool and
        // pile up CHUNK_VOXELS-sized raw buffers (16 MiB each at 256³).
        // Cap = 4 × encode workers keeps each encoder fed through normal
        // pop/encode cycles while capping peak raw RAM at ~ENC_Q_CAP*16 MiB.
        const size_t ENC_Q_CAP = std::max(8, encode_jobs * 4);

        // In-flight shard limiter: bounds how many ShardStates are live in
        // RAM at once (each holds up to 512 compressed chunk buffers ~ 30-40
        // MB). Defaults to 2*jobs so pipelining has slack without blowing RAM.
        const int max_in_flight_shards = std::max(1, jobs * 2);
        int in_flight_shards = 0;
        std::mutex in_flight_mtx;
        std::condition_variable in_flight_cv;

        auto report_progress = [&](int s_count) {
            if (s_count % 10 == 0 || s_count == (int)total_shards) {
                auto now = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(now - t0).count();
                double mins = secs / 60.0;
                int pc = processed_chunks.load();
                double shards_per_min = mins > 0 ? s_count / mins : 0;
                double chunks_per_min = mins > 0 ? pc / mins : 0;
                int remaining = (int)total_shards - s_count;
                double eta_min = shards_per_min > 0 ? remaining / shards_per_min : 0;
                std::lock_guard lk(print_mtx);
                printf("  %d/%zu shards (%.0f/min), %d chunks (%.0f/min), %d zero | ETA %.0fm\n",
                       s_count, total_shards, shards_per_min,
                       pc, chunks_per_min, zero_chunks.load(), eta_min);
            }
        };

        // Called when a shard's remaining count hits 0. Assembles the shard
        // bytes, hands it to the upload pool, and releases the in-flight
        // shard slot.
        auto finalize_shard = [&](std::shared_ptr<ShardState> shard) {
            // Always write the shard, even when it has no inner-chunk content:
            // index_bytes is pre-filled with 0xFF (missing-sentinel for all 512
            // inner chunks), so an empty shard is a valid 8KB-only object.
            // Downstream readers expect every shard slot in the grid to exist.
            std::vector<std::byte> shard_data;
            for (size_t j = 0; j < shard->result_kind.size(); j++) {
                uint8_t kind = shard->result_kind[j];
                if (kind == RESULT_NONE) continue;
                size_t inner_idx = shard->result_inner_idx[j];
                if (kind == RESULT_ZERO) {
                    write_zero_sentinel(shard->index_bytes.data() + inner_idx * 16);
                } else {
                    const auto& bytes = shard->result_data[j];
                    uint64_t offset = INDEX_BYTES + shard_data.size();
                    uint64_t nbytes = bytes.size();
                    write_le64(shard->index_bytes.data() + inner_idx * 16, offset);
                    write_le64(shard->index_bytes.data() + inner_idx * 16 + 8, nbytes);
                    shard_data.insert(shard_data.end(), bytes.begin(), bytes.end());
                }
            }
            std::vector<std::byte> shard_bytes(INDEX_BYTES + shard_data.size());
            std::memcpy(shard_bytes.data(), shard->index_bytes.data(), INDEX_BYTES);
            std::memcpy(shard_bytes.data() + INDEX_BYTES,
                        shard_data.data(), shard_data.size());
            enqueue_upload(shard->shard_key, std::move(shard_bytes));

            int s_count = processed_shards.fetch_add(1) + 1;
            if (log_file.is_open()) {
                std::lock_guard lk(log_mtx);
                log_file << shard->shard_key << "\n";
                log_file.flush();
            }
            report_progress(s_count);

            std::lock_guard lk(in_flight_mtx);
            in_flight_shards--;
            in_flight_cv.notify_one();
        };

        // Download worker: pop task, fetch from S3, decompress/pad, detect
        // all-zero, and either mark RESULT_ZERO directly or push a raw
        // buffer into the encode queue.
        auto dl_fn = [&]() {
            auto t_input = make_backend(input_path);
            for (;;) {
                DownloadTask task;
                {
                    std::unique_lock lk(dl_mtx);
                    dl_cv.wait(lk, [&]{ return !dl_q.empty() || dl_done; });
                    if (dl_q.empty()) return;
                    task = std::move(dl_q.front());
                    dl_q.pop_front();
                }

                auto finalize_slot = [&](uint8_t kind,
                                         std::vector<std::byte> bytes) {
                    if (kind != RESULT_NONE) {
                        task.shard->any_data.store(true);
                        if (kind == RESULT_COMPRESSED || kind == RESULT_PASSTHROUGH) {
                            task.shard->result_data[task.job_idx] = std::move(bytes);
                        }
                        task.shard->result_kind[task.job_idx] = kind;
                    }
                    if (task.shard->remaining.fetch_sub(1) == 1) {
                        finalize_shard(task.shard);
                    }
                };

                // Fast 1:1 passthrough when the source is already stored
                // in our output codec and the output chunk size matches
                // (ratio {1,1,1}). Matching input magic to the configured
                // output codec avoids misinterpreting h265 bytes as c3d
                // (or vice versa) as raw voxels downstream.
                if (task.rz == 1 && task.ry == 1 && task.rx == 1) {
                    std::string src_key = level_prefix +
                        std::to_string(task.src_base_z) + dim_sep +
                        std::to_string(task.src_base_y) + dim_sep +
                        std::to_string(task.src_base_x);
                    try {
                        auto data = t_input->read(src_key);
                        if (utils::is_c3d_compressed(
                                std::span<const std::byte>(data))) {
                            total_raw.fetch_add(CHUNK_VOXELS);
                            total_compressed.fetch_add(data.size());
                            processed_chunks.fetch_add(1);
                            finalize_slot(RESULT_PASSTHROUGH, std::move(data));
                            continue;
                        }
                    } catch (...) {
                        // fall through to the assemble path, which tolerates
                        // missing source chunks via zero-fill.
                    }
                }

                // Assemble the output chunk by reading each covering source
                // chunk and copying it into the right sub-region of a
                // CHUNK_VOXELS buffer.  Missing source chunks (404 / throw
                // on read) are left as the pre-zeroed default.
                std::vector<std::byte> raw(CHUNK_VOXELS, std::byte{0});
                const size_t sx = src_chunks[2];
                const size_t sy = src_chunks[1];
                const size_t sz = src_chunks[0];
                for (size_t dz = 0; dz < task.rz; ++dz)
                for (size_t dy = 0; dy < task.ry; ++dy)
                for (size_t dx = 0; dx < task.rx; ++dx) {
                    std::string src_key = level_prefix +
                        std::to_string(task.src_base_z + dz) + dim_sep +
                        std::to_string(task.src_base_y + dy) + dim_sep +
                        std::to_string(task.src_base_x + dx);
                    std::vector<std::byte> src_raw;
                    try {
                        auto data = t_input->read(src_key);
                        if (!compressor_id.empty()) {
                            src_raw = decompress_blosc(data, src_raw_bytes);
                        } else {
                            src_raw = std::move(data);
                        }
                    } catch (...) {
                        continue;   // missing source chunk → zero sub-block
                    }

                    if (is_u16) {
                        size_t n = src_raw.size() / 2;
                        auto* s16 = reinterpret_cast<const uint16_t*>(src_raw.data());
                        for (size_t i = 0; i < n; ++i) {
                            src_raw[i] = static_cast<std::byte>(
                                static_cast<uint8_t>(s16[i] / 257));
                        }
                        src_raw.resize(n);
                    }
                    // Copy src_raw (sz × sy × sx) into raw at sub-offset
                    // (dz*sz, dy*sy, dx*sx).  A row is sx bytes contiguous
                    // in both src and dst; outer strides differ (dst row
                    // stride is CHUNK_DIM, dst z-stride is CHUNK_DIM²).
                    const size_t have_z = std::min(sz, src_raw.size() / (sy * sx));
                    for (size_t z = 0; z < have_z; ++z) {
                        for (size_t y = 0; y < sy; ++y) {
                            const std::byte* src_row =
                                src_raw.data() + (z * sy + y) * sx;
                            std::byte* dst_row = raw.data()
                                + ((dz * sz + z) * CHUNK_DIM
                                   + (dy * sy + y)) * CHUNK_DIM
                                + dx * sx;
                            std::memcpy(dst_row, src_row, sx);
                        }
                    }
                }

                // Air-clamp is applied inside the codec (and the threshold is
                // stored in the chunk header so decode auto-zeros). We do not
                // pre-snap here.

                if (is_all_zero(raw)) {
                    zero_chunks.fetch_add(1);
                    finalize_slot(RESULT_ZERO, {});
                    continue;
                }

                {
                    std::unique_lock elk(enc_mtx);
                    enc_cv.wait(elk, [&]{
                        return enc_q.size() < ENC_Q_CAP || enc_done;
                    });
                    enc_q.push_back({task.shard, task.job_idx, std::move(raw)});
                }
                enc_cv.notify_one();
            }
        };

        // Encode worker: pop raw buffer, c3d-encode, store, decrement
        // shard remaining; finalize on last chunk.
        auto enc_fn = [&]() {
            for (;;) {
                EncodeTask task;
                {
                    std::unique_lock lk(enc_mtx);
                    enc_cv.wait(lk, [&]{ return !enc_q.empty() || enc_done; });
                    if (enc_q.empty()) return;
                    task = std::move(enc_q.front());
                    enc_q.pop_front();
                }
                // Wake any downloader blocked on queue-full.
                enc_cv.notify_one();

                total_raw.fetch_add(CHUNK_VOXELS);

                std::vector<std::byte> compressed;
                auto decode_cb = [&](std::vector<std::byte>& enc) {
                    utils::C3dCodecParams p;
                    p.target_ratio = target_ratio;
                    p.depth = (int)CHUNK_DIM;
                    p.height = (int)CHUNK_DIM;
                    p.width = (int)CHUNK_DIM;
                    return utils::c3d_decode(std::span<const std::byte>(enc),
                                             CHUNK_VOXELS, p);
                };

                {
                    utils::C3dCodecParams p;
                    p.target_ratio = target_ratio;
                    p.depth = (int)CHUNK_DIM;
                    p.height = (int)CHUNK_DIM;
                    p.width = (int)CHUNK_DIM;
                    compressed = utils::c3d_encode(
                        std::span<const std::byte>(task.raw), p);
                }
                total_compressed.fetch_add(compressed.size());

                if (verify) {
                    auto decoded = decode_cb(compressed);
                    // c3d is lossy; --verify's strict memcmp will almost
                    // always fail. Kept for parity with the legacy flag;
                    // --stats-pct is the meaningful quality check.
                    if (decoded.size() != task.raw.size() ||
                        std::memcmp(decoded.data(), task.raw.data(),
                                    task.raw.size()) != 0) {
                        verify_fail.fetch_add(1);
                    } else {
                        verify_ok.fetch_add(1);
                    }
                }

                // Quality sampling: decode a random subset of encoded chunks
                // and fold the error histogram into the per-level aggregate.
                if (stats_pct > 0) {
                    thread_local std::mt19937 rng{std::random_device{}()};
                    thread_local std::uniform_int_distribution<int> roll(1, 100);
                    if (roll(rng) <= stats_pct) {
                        auto decoded = decode_cb(compressed);
                        size_t n = std::min(decoded.size(), task.raw.size());
                        std::array<uint64_t, 256> local_hist{};
                        for (size_t i = 0; i < n; i++) {
                            int diff = std::abs(
                                (int)static_cast<uint8_t>(decoded[i]) -
                                (int)static_cast<uint8_t>(task.raw[i]));
                            local_hist[diff]++;
                        }
                        for (int i = 0; i < 256; i++) {
                            if (local_hist[i])
                                stats_err_hist[i].fetch_add(local_hist[i],
                                                            std::memory_order_relaxed);
                        }
                        stats_voxels_total.fetch_add(n, std::memory_order_relaxed);
                        stats_chunks_sampled.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                task.shard->result_data[task.job_idx] = std::move(compressed);
                task.shard->result_kind[task.job_idx] = RESULT_COMPRESSED;
                task.shard->any_data.store(true);
                processed_chunks.fetch_add(1);
                if (task.shard->remaining.fetch_sub(1) == 1) {
                    finalize_shard(task.shard);
                }
            }
        };

        // Start worker pools. --inner-jobs sizes the download pool; the
        // encode pool defaults to hardware_concurrency (we can saturate CPU
        // independently of how many downloads are in flight).
        const int n_dl = std::max(1, inner_jobs);
        // 2x hardware_concurrency: x265's internal pool already adds parallelism
        // so we don't need a 1:1 thread-to-core ratio. The extra encode threads
        // hide x265 internal sync stalls and let CPU saturate.
        const int n_enc = std::max(1,
            encode_jobs > 0 ? encode_jobs
                            : (int)std::thread::hardware_concurrency() * 2);
        std::vector<std::thread> dl_threads;
        dl_threads.reserve(n_dl);
        for (int i = 0; i < n_dl; i++) dl_threads.emplace_back(dl_fn);
        std::vector<std::thread> enc_threads;
        enc_threads.reserve(n_enc);
        for (int i = 0; i < n_enc; i++) enc_threads.emplace_back(enc_fn);

        // Submitter: single-threaded, but cheap — LIST calls are the only
        // per-shard cost here and each shard has at most 8 of them. Runs
        // ahead of the workers, throttled by max_in_flight_shards.
        auto submitter_input = make_backend(input_path);

        for (size_t si = 0; si < shard_positions.size(); si++) {
            auto [sz, sy, sx] = shard_positions[si];
            std::string shard_key = std::to_string(l) + "/c/" +
                std::to_string(sz) + "/" +
                std::to_string(sy) + "/" +
                std::to_string(sx);

            if (existing_shards.count(shard_key)) {
                int s_count = processed_shards.fetch_add(1) + 1;
                if (s_count % 100 == 0) {
                    std::lock_guard lk(print_mtx);
                    printf("  %d/%zu shards (skipping existing)\n",
                           s_count, total_shards);
                }
                continue;
            }

            {
                std::unique_lock lk(in_flight_mtx);
                in_flight_cv.wait(lk, [&]{
                    return in_flight_shards < max_in_flight_shards;
                });
                in_flight_shards++;
            }

            size_t base_cz = sz * CHUNKS_PER_SHARD;
            size_t base_cy = sy * CHUNKS_PER_SHARD;
            size_t base_cx = sx * CHUNKS_PER_SHARD;

            // Per-shard LIST removed: occ_mask was built up front with one
            // parallel LIST per cz prefix.  Workers consult occ_mask
            // directly, so existing_input is no longer needed.
            std::unordered_set<std::string> existing_input;
            if (false) {
                std::vector<std::future<std::vector<std::string>>> list_futs;
                list_futs.reserve(CHUNKS_PER_SHARD);
                for (size_t iz = 0; iz < CHUNKS_PER_SHARD; iz++) {
                    size_t cz = base_cz + iz;
                    if (cz >= out_nz) break;
                    std::string cz_prefix = level_prefix + std::to_string(cz) + dim_sep;
                    list_futs.push_back(std::async(std::launch::async,
                        [in = submitter_input.get(), prefix = std::move(cz_prefix)]() {
                            try {
                                return in->list_chunks(prefix);
                            } catch (const std::exception& e) {
                                static std::atomic<int> warn_count{0};
                                int n = warn_count.fetch_add(1);
                                if (n < 20)
                                    fprintf(stderr, "[warn] per-cz LIST failed "
                                            "(prefix=%s): %s\n", prefix.c_str(), e.what());
                                return std::vector<std::string>{};
                            }
                        }));
                }
                for (auto& f : list_futs)
                    for (auto& k : f.get()) existing_input.insert(std::move(k));
            }

            auto shard = std::make_shared<ShardState>();
            shard->shard_key = shard_key;
            shard->index_bytes.assign(INDEX_BYTES, std::byte{0xFF});

            std::vector<DownloadTask> tasks;
            tasks.reserve(INNER_CHUNKS);
            for (size_t iz = 0; iz < CHUNKS_PER_SHARD; iz++) {
                for (size_t iy = 0; iy < CHUNKS_PER_SHARD; iy++) {
                    for (size_t ix = 0; ix < CHUNKS_PER_SHARD; ix++) {
                        size_t cz = base_cz + iz;
                        size_t cy = base_cy + iy;
                        size_t cx = base_cx + ix;
                        size_t inner_idx = iz * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD
                                         + iy * CHUNKS_PER_SHARD + ix;
                        if (cz >= out_nz || cy >= out_ny || cx >= out_nx) continue;
                        size_t rz = CHUNK_DIM / src_chunks[0];
                        size_t ry = CHUNK_DIM / src_chunks[1];
                        size_t rx = CHUNK_DIM / src_chunks[2];
                        size_t src_base_z = cz * rz;
                        size_t src_base_y = cy * ry;
                        size_t src_base_x = cx * rx;
                        // Occupancy mask is indexed by source chunk grid;
                        // output chunk is "present" if any covering source
                        // chunk exists in the mask.
                        if (!occ_mask.empty()) {
                            bool any = false;
                            for (size_t dz = 0; dz < rz && !any; ++dz)
                            for (size_t dy = 0; dy < ry && !any; ++dy)
                            for (size_t dx = 0; dx < rx && !any; ++dx) {
                                size_t sz0 = src_base_z + dz;
                                size_t sy0 = src_base_y + dy;
                                size_t sx0 = src_base_x + dx;
                                if (sz0 >= occ_nz || sy0 >= occ_ny || sx0 >= occ_nx)
                                    continue;
                                if (occ_mask[sz0 * occ_ny * occ_nx
                                             + sy0 * occ_nx + sx0])
                                    any = true;
                            }
                            if (!any) {
                                skipped_chunks.fetch_add(1);
                                continue;
                            }
                        }
                        // existing_input is built from a listing of source
                        // keys; when source chunks are smaller than output
                        // chunks an output chunk exists if any covering
                        // source chunk exists.
                        if (!existing_input.empty()) {
                            bool any = false;
                            for (size_t dz = 0; dz < rz && !any; ++dz)
                            for (size_t dy = 0; dy < ry && !any; ++dy)
                            for (size_t dx = 0; dx < rx && !any; ++dx) {
                                std::string sk = level_prefix +
                                    std::to_string(src_base_z + dz) + dim_sep +
                                    std::to_string(src_base_y + dy) + dim_sep +
                                    std::to_string(src_base_x + dx);
                                if (existing_input.count(sk)) any = true;
                            }
                            if (!any) continue;
                        }
                        size_t job_idx = shard->result_kind.size();
                        shard->result_kind.push_back(RESULT_NONE);
                        shard->result_data.emplace_back();
                        shard->result_inner_idx.push_back(inner_idx);
                        tasks.push_back({shard, job_idx,
                                         src_base_z, src_base_y, src_base_x,
                                         rz, ry, rx});
                    }
                }
            }

            if (tasks.empty()) {
                // All inner chunks skipped via occupancy.  Finalize so the
                // 8KB sentinel shard still gets uploaded — we want every
                // grid slot to exist as an object downstream.
                finalize_shard(shard);
                continue;
            }

            shard->remaining.store(static_cast<int>(tasks.size()));
            {
                std::lock_guard lk(dl_mtx);
                for (auto& t : tasks) dl_q.push_back(std::move(t));
                dl_cv.notify_all();
            }
        }

        // Wait for all outstanding shards to drain through both pools.
        {
            std::unique_lock lk(in_flight_mtx);
            in_flight_cv.wait(lk, [&]{ return in_flight_shards == 0; });
        }

        // Shut down download and encode pools.
        { std::lock_guard lk(dl_mtx); dl_done = true; dl_cv.notify_all(); }
        for (auto& t : dl_threads) t.join();
        { std::lock_guard lk(enc_mtx); enc_done = true; enc_cv.notify_all(); }
        for (auto& t : enc_threads) t.join();

        // All shards enqueued. Signal the upload pool to drain and exit.
        {
            std::lock_guard lk(up_mtx);
            up_done = true;
            up_not_empty.notify_all();
        }
        for (auto& t : upload_threads) t.join();

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        size_t tr = total_raw.load(), tc = total_compressed.load();
        double ratio = tc > 0 ? (double)tr / tc : 0;
        double mb_s = tc > 0 ? (double)tc / (1024 * 1024) / elapsed : 0;

        printf("  Processed: %d shards, %d chunks (zero: %d, skipped: %d, errors: %d)\n",
               processed_shards.load(), processed_chunks.load(),
               zero_chunks.load(), skipped_chunks.load(), errs.load());
        printf("  Raw: %zu MB -> Compressed: %zu MB (ratio: %.2f:1)\n",
               tr / 1024 / 1024, tc / 1024 / 1024, ratio);
        printf("  Time: %.1fs (%.0f chunks/s, %.1f MB/s)\n",
               elapsed, processed_chunks.load() / elapsed, mb_s);
        if (verify) {
            printf("  Verify: %d OK, %d FAIL\n", verify_ok.load(), verify_fail.load());
        }

        // Dump lossy-codec quality stats for this level if sampling was on.
        uint64_t sv = stats_voxels_total.load();
        if (sv > 0) {
            uint64_t sum = 0, sum_sq = 0;
            int pmax = 0;
            for (int i = 0; i < 256; i++) {
                uint64_t c = stats_err_hist[i].load();
                sum += uint64_t(i) * c;
                sum_sq += uint64_t(i) * i * c;
                if (c > 0) pmax = i;
            }
            auto pct_bin = [&](double p) -> int {
                uint64_t target = uint64_t(double(sv) * p);
                uint64_t acc = 0;
                for (int i = 0; i < 256; i++) {
                    acc += stats_err_hist[i].load();
                    if (acc >= target) return i;
                }
                return pmax;
            };
            double mae = double(sum) / double(sv);
            double mse = double(sum_sq) / double(sv);
            double rmse = std::sqrt(mse);
            double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 1e9;
            int p50 = pct_bin(0.50), p90 = pct_bin(0.90),
                p95 = pct_bin(0.95), p99 = pct_bin(0.99);
            printf("  Quality (%%%d sample, %lu chunks, %lu voxels):\n",
                   stats_pct,
                   (unsigned long)stats_chunks_sampled.load(),
                   (unsigned long)sv);
            printf("    MAE=%.3f  RMSE=%.3f  PSNR=%.2f dB\n",
                   mae, rmse, psnr);
            printf("    percentiles: p50=%d  p90=%d  p95=%d  p99=%d  max=%d\n",
                   p50, p90, p95, p99, pmax);

            // Reset histogram for next level.
            for (int i = 0; i < 256; i++) stats_err_hist[i].store(0);
            stats_voxels_total.store(0);
            stats_chunks_sampled.store(0);
        }
    }

    blosc_destroy();
    return 0;
}
