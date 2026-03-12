// vc_zarr_recompress: Recompress zarr v2 volumes to zarr v3 with C3D + sharding.
//
// Reads zarr v2 chunks (blosc/zstd/raw) from S3 or local filesystem,
// recompresses with C3D into C3T containers (128³ shards, 32³ blocks),
// writes zarr v3 output to S3 or local filesystem.
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
//   --shift N     C3D quality preset (0=lossless, 1=1/2, ..., 5=1/32) [default: 0]
//   --verify      Verify lossless roundtrip (only meaningful with --shift 0)
//   --level N     Process only this pyramid level (-1=all) [default: -1]
//   --jobs N      Worker threads [default: half hardware threads]
//   --in-place    Write output back to input path (local only)

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
#include <chrono>

#include <nlohmann/json.hpp>
#include <blosc.h>

#include "utils/video_codec.hpp"
#include "utils/http_fetch.hpp"
#include "utils/zarr.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

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
        cfg.aws_auth = utils::AwsAuth::from_env();
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

    bool exists(const std::string& key) override {
        auto resp = client->head(url(key));
        return resp.ok();
    }

    std::vector<std::string> list_chunks(const std::string& prefix) override {
        // S3 ListObjectsV2 via REST API
        std::vector<std::string> result;
        std::string continuation_token;

        // We need the bucket and key_prefix from base_url
        // base_url is like https://bucket.s3.region.amazonaws.com/root_prefix
        // We need to call: https://bucket.s3.region.amazonaws.com?list-type=2&prefix=root_prefix/level_prefix
        // Parse bucket from base_url
        auto after_scheme = base_url.substr(base_url.find("//") + 2);
        auto dot = after_scheme.find('.');
        auto bucket = after_scheme.substr(0, dot);
        auto host_end = after_scheme.find('/');
        auto host = after_scheme.substr(0, host_end);
        std::string root_prefix;
        if (host_end != std::string::npos) {
            root_prefix = after_scheme.substr(host_end + 1);
        }

        std::string full_prefix = root_prefix.empty() ? prefix : root_prefix + "/" + prefix;
        std::string list_url_base = "https://" + host + "/?list-type=2&prefix=" + full_prefix;

        do {
            std::string list_url = list_url_base;
            if (!continuation_token.empty()) {
                list_url += "&continuation-token=" + continuation_token;
            }
            list_url += "&max-keys=10000";

            auto resp = client->get(list_url);
            if (!resp.ok()) {
                throw std::runtime_error("S3 list failed: " + std::to_string(resp.status_code));
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

static std::string make_zarr_v3_metadata(
    const std::vector<size_t>& shape,
    int shift)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = shape;
    meta.chunks = {128, 128, 128};  // shard shape
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";  // "/" separator
    meta.node_type = "array";

    // Sharding config: 128³ shards with 32³ inner chunks
    utils::ShardConfig sc;
    sc.sub_chunks = {32, 32, 32};
    sc.index_at_end = true;

    // Sub-chunk codec: C3D
    utils::ZarrCodecConfig c3d_codec;
    c3d_codec.name = "c3d";
    c3d_codec.configuration = json{{"shift", shift}};
    sc.sub_codecs.push_back(c3d_codec);

    // Index codec: bytes (little-endian)
    utils::ZarrCodecConfig bytes_codec;
    bytes_codec.name = "bytes";
    bytes_codec.configuration = json{{"endian", "little"}};
    sc.index_codecs.push_back(bytes_codec);

    meta.shard_config = sc;

    return utils::detail::serialize_zarr_json(meta);
}

static std::string make_zarr_v3_group() {
    json root;
    root["zarr_format"] = 3;
    root["node_type"] = "group";
    return root.dump(2) + "\n";
}

static std::string make_zarr_v3_group_with_multiscales(
    const std::vector<int>& levels,
    const std::vector<std::vector<size_t>>& shapes)
{
    json root;
    root["zarr_format"] = 3;
    root["node_type"] = "group";

    // OME-Zarr multiscales attribute
    json multiscales = json::array();
    json ms;
    ms["version"] = "0.4";
    ms["name"] = "/";

    json axes = json::array();
    axes.push_back({{"name", "z"}, {"type", "space"}});
    axes.push_back({{"name", "y"}, {"type", "space"}});
    axes.push_back({{"name", "x"}, {"type", "space"}});
    ms["axes"] = axes;

    json datasets = json::array();
    for (size_t i = 0; i < levels.size(); i++) {
        double scale = std::pow(2.0, levels[i]);
        json ds;
        ds["path"] = std::to_string(levels[i]);
        ds["coordinateTransformations"] = json::array({
            {{"type", "scale"}, {"scale", {scale, scale, scale}}}
        });
        datasets.push_back(ds);
    }
    ms["datasets"] = datasets;
    multiscales.push_back(ms);

    root["attributes"] = {{"multiscales", multiscales}};
    return root.dump(2) + "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: vc_zarr_recompress <input> <output> [options]\n"
                  << "\n"
                  << "Input/output: local path or s3://bucket/path\n"
                  << "\n"
                  << "Options:\n"
                  << "  --shift N     C3D preset (0=lossless, 1=1/2, ..., 5=1/32) [0]\n"
                  << "  --verify      Verify lossless roundtrip\n"
                  << "  --level N     Single pyramid level (-1=all) [-1]\n"
                  << "  --jobs N      Worker threads [half HW threads]\n"
                  << "  --in-place    Recompress input in-place (ignores output arg)\n";
        return 1;
    }

    blosc_init();

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int shift = 0;
    bool verify = false;
    int target_level = -1;
    int jobs = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    bool in_place = false;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--shift" && i + 1 < argc) shift = std::atoi(argv[++i]);
        else if (arg == "--verify") verify = true;
        else if (arg == "--level" && i + 1 < argc) target_level = std::atoi(argv[++i]);
        else if (arg == "--jobs" && i + 1 < argc) jobs = std::atoi(argv[++i]);
        else if (arg == "--in-place") in_place = true;
    }

    auto input = make_backend(input_path);
    auto output = in_place ? make_backend(input_path) : make_backend(output_path);

    printf("Input:  %s\n", input_path.c_str());
    printf("Output: %s%s\n", in_place ? input_path.c_str() : output_path.c_str(),
           in_place ? " (in-place)" : "");
    printf("C3D shift: %d (%s)\n", shift,
           shift == 0 ? "lossless" : ("1/" + std::to_string(1 << shift) + " lossless").c_str());
    printf("Jobs: %d\n\n", jobs);

    // Discover pyramid levels
    std::vector<int> levels;
    std::vector<std::vector<size_t>> shapes;

    for (int l = 0; l < 10; l++) {
        if (target_level != -1 && l != target_level) continue;

        std::string zarray_key = std::to_string(l) + "/.zarray";
        std::string zarr_json_key = std::to_string(l) + "/zarr.json";

        json zarray;
        try {
            if (input->exists(zarray_key)) {
                zarray = json::parse(input->read_string(zarray_key));
            } else if (input->exists(zarr_json_key)) {
                zarray = json::parse(input->read_string(zarr_json_key));
            } else {
                continue;
            }
        } catch (...) { continue; }

        std::vector<size_t> shape;
        if (zarray.contains("shape")) {
            for (auto& v : zarray["shape"]) shape.push_back(v.get<size_t>());
        }

        levels.push_back(l);
        shapes.push_back(shape);
        printf("Found level %d: shape [%zu, %zu, %zu]\n",
               l, shape.size() > 0 ? shape[0] : 0,
               shape.size() > 1 ? shape[1] : 0,
               shape.size() > 2 ? shape[2] : 0);
    }

    if (levels.empty()) {
        std::cerr << "No pyramid levels found\n";
        return 1;
    }

    // Write zarr v3 root metadata
    if (!in_place) {
        output->write_string("zarr.json", make_zarr_v3_group_with_multiscales(levels, shapes));
    }

    // C3D params
    utils::VideoCodecParams params;
    params.type = utils::VideoCodecType::C3D;
    params.qp = shift;
    params.depth = 128;
    params.height = 128;
    params.width = 128;

    const size_t raw_chunk_size = 128 * 128 * 128;

    for (size_t li = 0; li < levels.size(); li++) {
        int l = levels[li];
        auto& shape = shapes[li];

        printf("\n=== Level %d ===\n", l);

        // Write per-level zarr v3 metadata
        if (!in_place) {
            auto meta_json = make_zarr_v3_metadata(shape, shift);
            output->write_string(std::to_string(l) + "/zarr.json", meta_json);
        }

        // Read input .zarray to determine compressor and chunk layout
        std::string level_prefix = std::to_string(l) + "/";
        json zarray;
        try {
            zarray = json::parse(input->read_string(level_prefix + ".zarray"));
        } catch (...) {
            // Maybe zarr v3 or no metadata at level — try to proceed raw
        }

        std::string compressor_id;
        if (zarray.contains("compressor") && !zarray["compressor"].is_null()) {
            compressor_id = zarray["compressor"].value("id", "");
        }

        std::vector<size_t> src_chunks = {128, 128, 128};
        if (zarray.contains("chunks")) {
            src_chunks.clear();
            for (auto& v : zarray["chunks"]) src_chunks.push_back(v.get<size_t>());
        }

        std::string dim_sep = "/";
        if (zarray.contains("dimension_separator")) {
            dim_sep = zarray["dimension_separator"].get<std::string>();
        }

        // Compute chunk keys from shape and chunk size (avoids S3 listing)
        std::vector<std::string> chunk_keys;
        if (shape.size() >= 3 && src_chunks.size() >= 3) {
            size_t nz = (shape[0] + src_chunks[0] - 1) / src_chunks[0];
            size_t ny = (shape[1] + src_chunks[1] - 1) / src_chunks[1];
            size_t nx = (shape[2] + src_chunks[2] - 1) / src_chunks[2];
            for (size_t iz = 0; iz < nz; iz++) {
                for (size_t iy = 0; iy < ny; iy++) {
                    for (size_t ix = 0; ix < nx; ix++) {
                        chunk_keys.push_back(
                            level_prefix + std::to_string(iz) + dim_sep +
                            std::to_string(iy) + dim_sep + std::to_string(ix));
                    }
                }
            }
        }

        printf("  Source: chunks [%zu,%zu,%zu], compressor: %s, sep: '%s'\n",
               src_chunks[0], src_chunks[1], src_chunks[2],
               compressor_id.empty() ? "none" : compressor_id.c_str(),
               dim_sep.c_str());
        printf("  Computed %zu chunk keys\n", chunk_keys.size());

        std::atomic<size_t> total_raw{0}, total_compressed{0};
        std::atomic<int> processed{0}, errs{0}, skipped{0};
        std::atomic<int> verify_ok{0}, verify_fail{0};
        std::mutex print_mtx;

        auto t0 = std::chrono::steady_clock::now();

        // Process chunks in parallel
        // Each thread gets its own S3 client (HttpClient is not thread-safe due to mutex)
        std::atomic<size_t> next_idx{0};

        auto worker = [&]() {
            // Thread-local I/O backends for S3 (each needs its own curl handle)
            auto t_input = make_backend(input_path);
            auto t_output = in_place ? make_backend(input_path) : make_backend(output_path);

            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= chunk_keys.size()) break;

                const auto& key = chunk_keys[idx];
                try {
                    auto data = t_input->read(key);

                    // Skip if already C3T
                    if (utils::is_c3d_compressed(std::span<const std::byte>(data))) {
                        skipped.fetch_add(1);
                        processed.fetch_add(1);
                        continue;
                    }

                    // Decompress if needed
                    std::vector<std::byte> raw;
                    if (!compressor_id.empty()) {
                        raw = decompress_blosc(data, raw_chunk_size);
                    } else {
                        raw = std::move(data);
                    }

                    // Pad undersized boundary chunks to 128³
                    if (raw.size() < raw_chunk_size) {
                        std::vector<std::byte> padded(raw_chunk_size, std::byte{0});
                        std::memcpy(padded.data(), raw.data(), raw.size());
                        raw = std::move(padded);
                    } else if (raw.size() > raw_chunk_size) {
                        std::lock_guard lk(print_mtx);
                        fprintf(stderr, "  WARN: oversized chunk %s (%zu bytes), skipping\n",
                                key.c_str(), raw.size());
                        skipped.fetch_add(1);
                        processed.fetch_add(1);
                        continue;
                    }

                    total_raw.fetch_add(raw_chunk_size);

                    // Compress with C3D
                    auto compressed = utils::video_encode(
                        std::span<const std::byte>(raw), params);
                    total_compressed.fetch_add(compressed.size());

                    // Verify roundtrip
                    if (verify) {
                        auto decoded = utils::video_decode(
                            std::span<const std::byte>(compressed), raw_chunk_size, params);
                        if (decoded.size() != raw.size() ||
                            std::memcmp(decoded.data(), raw.data(), raw.size()) != 0) {
                            verify_fail.fetch_add(1);
                            std::lock_guard lk(print_mtx);
                            fprintf(stderr, "  VERIFY FAIL: %s\n", key.c_str());
                        } else {
                            verify_ok.fetch_add(1);
                        }
                    }

                    // Determine output key
                    // Input: level/iz/iy/ix (zarr v2 with / separator)
                    // Output: same path (C3T shard replaces the chunk)
                    // For zarr v3 with default key encoding, chunk keys are c/iz/iy/ix
                    std::string out_key = key;
                    if (!in_place) {
                        // Convert zarr v2 chunk path to v3: prepend "c/" to the chunk indices
                        // Input key: "0/12/34/56" -> output: "0/c/12/34/56"
                        auto slash = key.find('/');
                        if (slash != std::string::npos) {
                            out_key = key.substr(0, slash) + "/c" + key.substr(slash);
                        }
                    }

                    t_output->write(out_key, compressed);

                    int p_count = processed.fetch_add(1) + 1;
                    if (p_count % 500 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double secs = std::chrono::duration<double>(now - t0).count();
                        std::lock_guard lk(print_mtx);
                        printf("  %d/%zu (%.0f/s) ratio: %.2f:1\n",
                               p_count, chunk_keys.size(), p_count / secs,
                               (double)total_raw / total_compressed);
                    }
                } catch (const std::exception& e) {
                    errs.fetch_add(1);
                    processed.fetch_add(1);
                    std::lock_guard lk(print_mtx);
                    fprintf(stderr, "  ERROR: %s: %s\n", key.c_str(), e.what());
                }
            }
        };

        std::vector<std::thread> threads;
        for (int j = 0; j < jobs; j++) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        size_t tr = total_raw.load(), tc = total_compressed.load();
        double ratio = tc > 0 ? (double)tr / tc : 0;

        printf("  Processed: %d (skipped: %d, errors: %d)\n",
               processed.load(), skipped.load(), errs.load());
        printf("  Raw: %zu MB -> Compressed: %zu MB (ratio: %.2f:1)\n",
               tr / 1024 / 1024, tc / 1024 / 1024, ratio);
        printf("  Time: %.1fs (%.0f chunks/s)\n", elapsed, chunk_keys.size() / elapsed);
        if (verify) {
            printf("  Verify: %d OK, %d FAIL\n", verify_ok.load(), verify_fail.load());
        }
    }

    blosc_destroy();
    return 0;
}
