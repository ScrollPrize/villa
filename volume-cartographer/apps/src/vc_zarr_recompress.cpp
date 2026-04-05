// vc_zarr_recompress: Recompress zarr v2 volumes to zarr v3 with H265 + sharding.
//
// Reads zarr v2 chunks (blosc/zstd/raw) from S3 or local filesystem,
// recompresses with H265 into zarr v3 shards (1024³ shards, 128³ inner chunks),
// writes zarr v3 output to S3 or local filesystem.
//
// Each 128³ inner chunk is H265-encoded individually (VC3D header + H265
// bitstream). Shards use the standard zarr v3 sharding_indexed format with
// the index at end.
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
//   --qp N        H265 quantization parameter (0-51) [default: 15]
//   --verify      Verify roundtrip (decode after encode)
//   --level N     Process only this pyramid level (-1=all) [default: -1]
//   --jobs N      Worker threads [default: half hardware threads]

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
#include <set>
#include <chrono>

#include "utils/Json.hpp"
#include <blosc.h>

#include "utils/video_codec.hpp"
#include "utils/http_fetch.hpp"
#include "utils/zarr.hpp"

namespace fs = std::filesystem;
using Json = utils::Json;

static constexpr size_t SHARD_DIM = 1024;  // shard shape per axis
static constexpr size_t CHUNK_DIM = 128;   // inner chunk shape per axis
static constexpr size_t CHUNKS_PER_SHARD = SHARD_DIM / CHUNK_DIM;  // 8
static constexpr size_t INNER_CHUNKS = CHUNKS_PER_SHARD * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD;  // 512
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

static std::string make_zarr_v3_metadata(const std::vector<size_t>& shape, int qp) {
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = shape;
    meta.chunks = {SHARD_DIM, SHARD_DIM, SHARD_DIM};  // shard shape
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0;
    meta.chunk_key_encoding = "default";  // "/" separator
    meta.node_type = "array";

    // Sharding config: 1024³ shards with 128³ inner chunks
    utils::ShardConfig sc;
    sc.sub_chunks = {CHUNK_DIM, CHUNK_DIM, CHUNK_DIM};
    sc.index_at_end = true;

    // Sub-chunk codec: h265 (VC3D video codec)
    utils::ZarrCodecConfig h265_codec;
    h265_codec.name = "h265";
    h265_codec.configuration = std::make_shared<utils::JsonValue>(utils::JsonValue{{"qp", Json(qp)}});
    sc.sub_codecs.push_back(h265_codec);

    // Index codec: bytes (little-endian)
    utils::ZarrCodecConfig bytes_codec;
    bytes_codec.name = "bytes";
    bytes_codec.configuration = std::make_shared<utils::JsonValue>(utils::JsonValue{{"endian", Json("little")}});
    sc.index_codecs.push_back(bytes_codec);

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

// ============================================================================
// Mask: use lowest-resolution level to skip empty chunks
// ============================================================================

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
                  << "  --qp N        H265 quantization (0-51, lower=better) [15]\n"
                  << "  --verify      Verify roundtrip after encoding\n"
                  << "  --level N     Single pyramid level (-1=all) [-1]\n"
                  << "  --jobs N      Worker threads [half HW threads]\n";
        return 1;
    }

    blosc_init();

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int qp = 15;
    bool verify = false;
    int target_level = -1;
    int jobs = std::max(1, (int)std::thread::hardware_concurrency() / 2);

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--qp" && i + 1 < argc) qp = std::atoi(argv[++i]);
        else if (arg == "--verify") verify = true;
        else if (arg == "--level" && i + 1 < argc) target_level = std::atoi(argv[++i]);
        else if (arg == "--jobs" && i + 1 < argc) jobs = std::atoi(argv[++i]);
    }

    auto input = make_backend(input_path);
    auto output = make_backend(output_path);

    printf("Input:  %s\n", input_path.c_str());
    printf("Output: %s\n", output_path.c_str());
    printf("H265 QP: %d, shard: %zu³, chunk: %zu³\n", qp, SHARD_DIM, CHUNK_DIM);
    printf("Jobs: %d\n\n", jobs);

    // Discover pyramid levels
    std::vector<int> levels;
    std::vector<std::vector<size_t>> shapes;

    for (int l = 0; l < 10; l++) {
        if (target_level != -1 && l != target_level) continue;

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
    output->write_string("zarr.json", make_zarr_v3_group_with_multiscales(levels, shapes));

    for (size_t li = 0; li < levels.size(); li++) {
        int l = levels[li];
        auto& shape = shapes[li];

        printf("\n=== Level %d ===\n", l);

        // Write per-level zarr v3 metadata
        output->write_string(std::to_string(l) + "/zarr.json",
                              make_zarr_v3_metadata(shape, qp));

        // Read input .zarray to determine compressor, chunks, dtype, dim_sep
        std::string level_prefix = std::to_string(l) + "/";
        Json zarray;
        try {
            zarray = Json::parse(input->read_string(level_prefix + ".zarray"));
        } catch (...) {}

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

        // Shard grid dimensions
        size_t shard_nz = (shape[0] + SHARD_DIM - 1) / SHARD_DIM;
        size_t shard_ny = (shape[1] + SHARD_DIM - 1) / SHARD_DIM;
        size_t shard_nx = (shape[2] + SHARD_DIM - 1) / SHARD_DIM;
        size_t total_shards = shard_nz * shard_ny * shard_nx;

        // Build occupancy mask at 128³ chunk granularity
        std::vector<bool> occ_mask;
        int mask_level = -1;
        for (int mi = (int)levels.size() - 1; mi >= 0; mi--) {
            if (levels[mi] > l) {
                mask_level = levels[mi];
                break;
            }
        }
        if (mask_level >= 0) {
            for (size_t mi = 0; mi < levels.size(); mi++) {
                if (levels[mi] == mask_level) {
                    // Build mask at 128³ granularity
                    std::vector<size_t> chunk128 = {CHUNK_DIM, CHUNK_DIM, CHUNK_DIM};
                    std::vector<size_t> mask_src_chunks = {128, 128, 128};
                    // Read mask level's actual chunk size
                    try {
                        Json mask_zarray = Json::parse(
                            input->read_string(std::to_string(mask_level) + "/.zarray"));
                        if (mask_zarray.contains("chunks")) {
                            mask_src_chunks.clear();
                            for (auto& v : mask_zarray["chunks"])
                                mask_src_chunks.push_back(v.get_size_t());
                        }
                    } catch (...) {}
                    occ_mask = build_occupancy_mask(
                        *input, l, shape, chunk128,
                        mask_level, shapes[mi], mask_src_chunks);
                    break;
                }
            }
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

        // Build list of shard positions to process
        struct ShardPos { size_t sz, sy, sx; };
        std::vector<ShardPos> shard_positions;
        for (size_t sz = 0; sz < shard_nz; sz++)
            for (size_t sy = 0; sy < shard_ny; sy++)
                for (size_t sx = 0; sx < shard_nx; sx++)
                    shard_positions.push_back({sz, sy, sx});

        // List existing shards for resume (one S3 list instead of per-shard HEAD)
        std::set<std::string> existing_shards;
        {
            std::string shard_prefix = std::to_string(l) + "/c/";
            auto keys = output->list_chunks(shard_prefix);
            for (auto& k : keys) {
                // Extract shard key: "level/c/sz/sy/sx" from "level/c/sz/sy/sx/..."
                // Shard files are just "level/c/sz/sy/sx" (no subdirectory)
                auto rel = k.substr(shard_prefix.size());
                // rel is like "0/0/0" — reconstruct full shard key
                existing_shards.insert(shard_prefix + rel);
            }
            if (!existing_shards.empty()) {
                printf("  Resume: found %zu existing shards, will skip\n",
                       existing_shards.size());
            }
        }

        std::atomic<size_t> total_raw{0}, total_compressed{0};
        std::atomic<int> processed_shards{0}, processed_chunks{0};
        std::atomic<int> errs{0}, skipped_chunks{0};
        std::atomic<int> verify_ok{0}, verify_fail{0};
        std::mutex print_mtx;

        auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next_shard{0};

        auto worker = [&]() {
            auto t_input = make_backend(input_path);
            auto t_output = make_backend(output_path);

            while (true) {
                size_t si = next_shard.fetch_add(1);
                if (si >= shard_positions.size()) break;

                auto [sz, sy, sx] = shard_positions[si];

                // Skip shards that already exist (resume support)
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

                // Base 128³ chunk coords for this shard
                size_t base_cz = sz * CHUNKS_PER_SHARD;
                size_t base_cy = sy * CHUNKS_PER_SHARD;
                size_t base_cx = sx * CHUNKS_PER_SHARD;

                // Stream shard to temp file: encode each inner 128³ chunk
                std::string temp_path = "/tmp/shard_" + std::to_string(l) + "_" +
                    std::to_string(sz) + "_" + std::to_string(sy) + "_" +
                    std::to_string(sx) + ".tmp";
                std::ofstream shard_file(temp_path, std::ios::binary | std::ios::trunc);
                size_t shard_pos = 0;
                // Index: INNER_CHUNKS entries, 16 bytes each (offset + nbytes)
                std::vector<std::byte> index_bytes(INNER_CHUNKS * 16);
                // Initialize all index entries to missing (0xFFFFFFFFFFFFFFFF)
                std::memset(index_bytes.data(), 0xFF, index_bytes.size());

                bool any_data = false;

                for (size_t iz = 0; iz < CHUNKS_PER_SHARD; iz++) {
                    for (size_t iy = 0; iy < CHUNKS_PER_SHARD; iy++) {
                        for (size_t ix = 0; ix < CHUNKS_PER_SHARD; ix++) {
                            size_t cz = base_cz + iz;
                            size_t cy = base_cy + iy;
                            size_t cx = base_cx + ix;

                            // Out of bounds?
                            if (cz >= out_nz || cy >= out_ny || cx >= out_nx) continue;

                            // Check occupancy mask
                            if (!occ_mask.empty()) {
                                size_t flat = cz * out_ny * out_nx + cy * out_nx + cx;
                                if (!occ_mask[flat]) {
                                    skipped_chunks.fetch_add(1);
                                    continue;
                                }
                            }

                            // Map output 128³ chunk to source chunk(s)
                            // If source chunks are 128³, it's 1:1
                            // If source chunks differ, we need to remap
                            // For now, assume source chunks are 128³ (most common)
                            std::string src_key = level_prefix +
                                std::to_string(cz) + dim_sep +
                                std::to_string(cy) + dim_sep +
                                std::to_string(cx);

                            std::vector<std::byte> raw;
                            try {
                                auto data = t_input->read(src_key);

                                // Already VC3D? Use as-is
                                if (utils::is_video_compressed(
                                        std::span<const std::byte>(data))) {
                                    // Write directly into shard
                                    size_t inner_idx = iz * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD +
                                                       iy * CHUNKS_PER_SHARD + ix;
                                    uint64_t offset = shard_pos;
                                    uint64_t nbytes = data.size();
                                    write_le64(index_bytes.data() + inner_idx * 16, offset);
                                    write_le64(index_bytes.data() + inner_idx * 16 + 8, nbytes);
                                    shard_file.write(reinterpret_cast<const char*>(data.data()),
                                                     data.size());
                                    shard_pos += data.size();
                                    any_data = true;
                                    processed_chunks.fetch_add(1);
                                    total_compressed.fetch_add(data.size());
                                    total_raw.fetch_add(CHUNK_VOXELS);
                                    continue;
                                }

                                // Decompress blosc/zstd
                                if (!compressor_id.empty()) {
                                    raw = decompress_blosc(data, src_raw_bytes);
                                } else {
                                    raw = std::move(data);
                                }
                            } catch (...) {
                                continue;  // chunk doesn't exist
                            }

                            // Convert uint16 -> uint8
                            if (is_u16) {
                                size_t n_voxels = raw.size() / 2;
                                auto* src = reinterpret_cast<const uint16_t*>(raw.data());
                                for (size_t i = 0; i < n_voxels; i++) {
                                    raw[i] = static_cast<std::byte>(
                                        static_cast<uint8_t>(src[i] / 257));
                                }
                                raw.resize(n_voxels);
                            }

                            // Pad to 128³
                            if (raw.size() < CHUNK_VOXELS) {
                                std::vector<std::byte> padded(CHUNK_VOXELS, std::byte{0});
                                std::memcpy(padded.data(), raw.data(), raw.size());
                                raw = std::move(padded);
                            } else if (raw.size() > CHUNK_VOXELS) {
                                raw.resize(CHUNK_VOXELS);
                            }

                            total_raw.fetch_add(CHUNK_VOXELS);

                            // H265 encode
                            utils::VideoCodecParams params;
                            params.qp = qp;
                            params.depth = CHUNK_DIM;
                            params.height = CHUNK_DIM;
                            params.width = CHUNK_DIM;

                            auto compressed = utils::video_encode(
                                std::span<const std::byte>(raw), params);
                            total_compressed.fetch_add(compressed.size());

                            // Verify
                            if (verify) {
                                auto decoded = utils::video_decode(
                                    std::span<const std::byte>(compressed),
                                    CHUNK_VOXELS, params);
                                if (decoded.size() != raw.size() ||
                                    std::memcmp(decoded.data(), raw.data(), raw.size()) != 0) {
                                    verify_fail.fetch_add(1);
                                    std::lock_guard lk(print_mtx);
                                    fprintf(stderr, "  VERIFY FAIL: %s\n", src_key.c_str());
                                } else {
                                    verify_ok.fetch_add(1);
                                }
                            }

                            // Record in shard index
                            size_t inner_idx = iz * CHUNKS_PER_SHARD * CHUNKS_PER_SHARD +
                                               iy * CHUNKS_PER_SHARD + ix;
                            uint64_t offset = shard_pos;
                            uint64_t nbytes = compressed.size();
                            write_le64(index_bytes.data() + inner_idx * 16, offset);
                            write_le64(index_bytes.data() + inner_idx * 16 + 8, nbytes);

                            shard_file.write(reinterpret_cast<const char*>(compressed.data()),
                                             compressed.size());
                            shard_pos += compressed.size();
                            any_data = true;
                            processed_chunks.fetch_add(1);
                        }
                    }
                }

                if (any_data) {
                    // Append index at end of shard
                    shard_file.write(reinterpret_cast<const char*>(index_bytes.data()),
                                     index_bytes.size());
                    shard_file.close();
                    t_output->write_from_file(shard_key, temp_path);
                } else {
                    shard_file.close();
                    fs::remove(temp_path);
                }

                int s_count = processed_shards.fetch_add(1) + 1;
                if (s_count % 10 == 0 || s_count == (int)total_shards) {
                    auto now = std::chrono::steady_clock::now();
                    double secs = std::chrono::duration<double>(now - t0).count();
                    size_t tc = total_compressed.load();
                    double mb_s = tc > 0 ? (double)tc / (1024 * 1024) / secs : 0;
                    std::lock_guard lk(print_mtx);
                    printf("  %d/%zu shards, %d chunks (%.0f/s, %.1f MB/s)\n",
                           s_count, total_shards, processed_chunks.load(),
                           processed_chunks.load() / secs, mb_s);
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
        double mb_s = tc > 0 ? (double)tc / (1024 * 1024) / elapsed : 0;

        printf("  Processed: %d shards, %d chunks (skipped: %d, errors: %d)\n",
               processed_shards.load(), processed_chunks.load(),
               skipped_chunks.load(), errs.load());
        printf("  Raw: %zu MB -> Compressed: %zu MB (ratio: %.2f:1)\n",
               tr / 1024 / 1024, tc / 1024 / 1024, ratio);
        printf("  Time: %.1fs (%.0f chunks/s, %.1f MB/s)\n",
               elapsed, processed_chunks.load() / elapsed, mb_s);
        if (verify) {
            printf("  Verify: %d OK, %d FAIL\n", verify_ok.load(), verify_fail.load());
        }
    }

    blosc_destroy();
    return 0;
}
