// vc_zarr_recompress: Recompress zarr v2 volumes with H264-encoded chunks.
//
// Reads zarr v2 chunks (blosc/zstd/raw) from S3 or local filesystem,
// recompresses with H264 (VC3D video codec), writes zarr v2 output
// to S3 or local filesystem. Each output chunk is a VC3D H264 blob
// that VC3D's VcDecompressor can decode directly.
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
//   --qp N        H264 quantization parameter (0-51) [default: 40]
//   --verify      Verify roundtrip (decode after encode)
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
// Zarr v2 metadata generation
// ============================================================================

static std::string make_zgroup() {
    return "{\"zarr_format\": 2}\n";
}

static std::string make_zarray(const std::vector<size_t>& shape,
                                const std::vector<size_t>& chunks,
                                const std::string& dim_sep) {
    json root;
    root["zarr_format"] = 2;

    root["shape"] = shape;
    root["chunks"] = chunks;
    root["dtype"] = "|u1";
    root["compressor"] = nullptr;
    root["fill_value"] = 0;
    root["order"] = "C";
    root["filters"] = nullptr;
    root["dimension_separator"] = dim_sep;

    return root.dump(2) + "\n";
}

static std::string make_zattrs_multiscales(
    const std::vector<int>& levels,
    const std::vector<std::vector<size_t>>& shapes)
{
    json axes = json::array();
    axes.push_back({{"name", "z"}, {"type", "space"}});
    axes.push_back({{"name", "y"}, {"type", "space"}});
    axes.push_back({{"name", "x"}, {"type", "space"}});

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

    json ms;
    ms["version"] = "0.4";
    ms["name"] = "/";
    ms["axes"] = axes;
    ms["datasets"] = datasets;

    json root;
    root["multiscales"] = json::array({ms});
    return root.dump(2) + "\n";
}

// ============================================================================
// Mask: use lowest-resolution level to skip empty chunks
// ============================================================================

// Build a 3D boolean mask indicating which chunks at `level` have nonzero data,
// using the lowest-resolution pyramid level as a guide.
// Returns an empty vector if mask cannot be built (caller should process all).
static std::vector<bool> build_occupancy_mask(
    IOBackend& io,
    int level,                           // target level (e.g. 0)
    const std::vector<size_t>& shape,    // target level shape
    const std::vector<size_t>& chunks,   // target level chunk size
    int mask_level,                      // lowest-res level to use as mask
    const std::vector<size_t>& mask_shape,
    const std::vector<size_t>& mask_chunks)
{
    if (shape.size() < 3 || mask_shape.size() < 3) return {};

    int scale = 1 << (mask_level - level);  // e.g. 2^5 = 32 for level 5 vs level 0

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

    // Read each mask-level chunk and check which target chunks have data
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
                    continue;  // chunk doesn't exist = all zero
                }

                // For each voxel in this mask chunk, determine which target chunk it maps to
                size_t cz = mask_chunks[0], cy = mask_chunks[1], cx = mask_chunks[2];
                // Clamp to actual mask shape
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
                                // This mask voxel is nonzero. Mark the corresponding target chunk.
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
    if (argc < 3) {
        std::cerr << "Usage: vc_zarr_recompress <input> <output> [options]\n"
                  << "\n"
                  << "Input/output: local path or s3://bucket/path\n"
                  << "\n"
                  << "Options:\n"
                  << "  --qp N        H264 quantization (0-51, lower=better) [40]\n"
                  << "  --verify      Verify roundtrip after encoding\n"
                  << "  --level N     Single pyramid level (-1=all) [-1]\n"
                  << "  --jobs N      Worker threads [half HW threads]\n"
                  << "  --in-place    Recompress input in-place (ignores output arg)\n";
        return 1;
    }

    blosc_init();

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int qp = 40;
    bool verify = false;
    int target_level = -1;
    int jobs = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    bool in_place = false;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--qp" && i + 1 < argc) qp = std::atoi(argv[++i]);
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
    printf("H264 QP: %d\n", qp);
    printf("Jobs: %d\n\n", jobs);

    // Discover pyramid levels
    std::vector<int> levels;
    std::vector<std::vector<size_t>> shapes;
    std::vector<std::vector<size_t>> all_chunks;  // per-level chunk sizes
    std::vector<std::string> all_dim_seps;        // per-level dimension separators
    std::vector<bool> all_is_u16;                  // per-level dtype

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

        std::vector<size_t> chunks = {128, 128, 128};
        if (zarray.contains("chunks")) {
            chunks.clear();
            for (auto& v : zarray["chunks"]) chunks.push_back(v.get<size_t>());
        }

        std::string dim_sep = "/";
        if (zarray.contains("dimension_separator")) {
            dim_sep = zarray["dimension_separator"].get<std::string>();
        }

        // Detect dtype — check for uint16 (">u2", "<u2", "|u2")
        bool is_u16 = false;
        if (zarray.contains("dtype") && zarray["dtype"].is_string()) {
            std::string dt = zarray["dtype"].get<std::string>();
            // Strip byte-order prefix
            std::string_view sv = dt;
            if (!sv.empty() && (sv[0] == '<' || sv[0] == '>' || sv[0] == '|'))
                sv.remove_prefix(1);
            is_u16 = (sv == "u2");
        }

        levels.push_back(l);
        shapes.push_back(shape);
        all_chunks.push_back(chunks);
        all_dim_seps.push_back(dim_sep);
        all_is_u16.push_back(is_u16);
        printf("Found level %d: shape [%zu, %zu, %zu] chunks [%zu, %zu, %zu] %s\n",
               l, shape.size() > 0 ? shape[0] : 0,
               shape.size() > 1 ? shape[1] : 0,
               shape.size() > 2 ? shape[2] : 0,
               chunks[0], chunks[1], chunks[2],
               is_u16 ? "uint16" : "uint8");
    }

    if (levels.empty()) {
        std::cerr << "No pyramid levels found\n";
        return 1;
    }

    // Write zarr v2 root metadata
    if (!in_place) {
        output->write_string(".zgroup", make_zgroup());
        output->write_string(".zattrs", make_zattrs_multiscales(levels, shapes));
    }

    for (size_t li = 0; li < levels.size(); li++) {
        int l = levels[li];
        auto& shape = shapes[li];
        auto& src_chunks = all_chunks[li];
        auto& dim_sep = all_dim_seps[li];
        bool is_u16 = all_is_u16[li];

        printf("\n=== Level %d ===\n", l);

        // Write per-level zarr v2 metadata
        if (!in_place) {
            std::string level_str = std::to_string(l);
            output->write_string(level_str + "/.zarray",
                                  make_zarray(shape, src_chunks, dim_sep));
            output->write_string(level_str + "/.zgroup", make_zgroup());
        }

        // Read input .zarray to determine compressor
        std::string level_prefix = std::to_string(l) + "/";
        json zarray;
        try {
            zarray = json::parse(input->read_string(level_prefix + ".zarray"));
        } catch (...) {}

        std::string compressor_id;
        if (zarray.contains("compressor") && !zarray["compressor"].is_null()) {
            compressor_id = zarray["compressor"].value("id", "");
        }

        // Chunk voxel count (for decompression buffer sizing)
        size_t chunk_voxels = 1;
        for (auto d : src_chunks) chunk_voxels *= d;
        size_t raw_bytes = is_u16 ? chunk_voxels * 2 : chunk_voxels;

        // Compute chunk keys from shape and chunk size (avoids S3 listing)
        std::vector<std::string> chunk_keys;
        if (shape.size() >= 3 && src_chunks.size() >= 3) {
            size_t nz = (shape[0] + src_chunks[0] - 1) / src_chunks[0];
            size_t ny = (shape[1] + src_chunks[1] - 1) / src_chunks[1];
            size_t nx = (shape[2] + src_chunks[2] - 1) / src_chunks[2];

            // Build occupancy mask from lowest-res level to skip empty chunks
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
                        occ_mask = build_occupancy_mask(
                            *input, l, shape, src_chunks,
                            mask_level, shapes[mi], all_chunks[mi]);
                        break;
                    }
                }
            }

            size_t skipped_by_mask = 0;
            for (size_t iz = 0; iz < nz; iz++) {
                for (size_t iy = 0; iy < ny; iy++) {
                    for (size_t ix = 0; ix < nx; ix++) {
                        if (!occ_mask.empty()) {
                            size_t flat = iz * ny * nx + iy * nx + ix;
                            if (!occ_mask[flat]) {
                                skipped_by_mask++;
                                continue;
                            }
                        }
                        chunk_keys.push_back(
                            level_prefix + std::to_string(iz) + dim_sep +
                            std::to_string(iy) + dim_sep + std::to_string(ix));
                    }
                }
            }
            if (!occ_mask.empty()) {
                printf("  Mask skipped %zu empty chunk positions\n", skipped_by_mask);
            }
        }

        printf("  Source: chunks [%zu,%zu,%zu], compressor: %s, sep: '%s', dtype: %s\n",
               src_chunks[0], src_chunks[1], src_chunks[2],
               compressor_id.empty() ? "none" : compressor_id.c_str(),
               dim_sep.c_str(), is_u16 ? "uint16" : "uint8");
        printf("  Computed %zu chunk keys\n", chunk_keys.size());

        std::atomic<size_t> total_raw{0}, total_compressed{0};
        std::atomic<int> processed{0}, errs{0}, skipped{0};
        std::atomic<int> verify_ok{0}, verify_fail{0};
        std::mutex print_mtx;

        auto t0 = std::chrono::steady_clock::now();

        std::atomic<size_t> next_idx{0};

        auto worker = [&]() {
            // Thread-local I/O backends (HttpClient is not thread-safe)
            auto t_input = make_backend(input_path);
            auto t_output = in_place ? make_backend(input_path) : make_backend(output_path);

            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= chunk_keys.size()) break;

                const auto& key = chunk_keys[idx];
                try {
                    auto data = t_input->read(key);

                    // Skip if already VC3D-encoded
                    if (utils::is_video_compressed(std::span<const std::byte>(data))) {
                        skipped.fetch_add(1);
                        processed.fetch_add(1);
                        continue;
                    }

                    // Decompress blosc/zstd if needed
                    std::vector<std::byte> raw;
                    if (!compressor_id.empty()) {
                        raw = decompress_blosc(data, raw_bytes);
                    } else {
                        raw = std::move(data);
                    }

                    // Convert uint16 -> uint8 if needed (divide by 257)
                    if (is_u16) {
                        size_t n_voxels = raw.size() / 2;
                        auto* src = reinterpret_cast<const uint16_t*>(raw.data());
                        for (size_t i = 0; i < n_voxels; i++) {
                            raw[i] = static_cast<std::byte>(
                                static_cast<uint8_t>(src[i] / 257));
                        }
                        raw.resize(n_voxels);
                    }

                    // Pad undersized boundary chunks to full chunk size
                    if (raw.size() < chunk_voxels) {
                        std::vector<std::byte> padded(chunk_voxels, std::byte{0});
                        std::memcpy(padded.data(), raw.data(), raw.size());
                        raw = std::move(padded);
                    } else if (raw.size() > chunk_voxels) {
                        // Truncate to expected size (boundary rounding)
                        raw.resize(chunk_voxels);
                    }

                    total_raw.fetch_add(chunk_voxels);

                    // Encode with H264
                    utils::VideoCodecParams params;
                    params.type = utils::VideoCodecType::H264;
                    params.qp = qp;
                    params.depth = static_cast<int>(src_chunks[0]);
                    params.height = static_cast<int>(src_chunks[1]);
                    params.width = static_cast<int>(src_chunks[2]);

                    auto compressed = utils::video_encode(
                        std::span<const std::byte>(raw), params);
                    total_compressed.fetch_add(compressed.size());

                    // Verify roundtrip
                    if (verify) {
                        auto decoded = utils::video_decode(
                            std::span<const std::byte>(compressed), chunk_voxels, params);
                        if (decoded.size() != raw.size() ||
                            std::memcmp(decoded.data(), raw.data(), raw.size()) != 0) {
                            verify_fail.fetch_add(1);
                            std::lock_guard lk(print_mtx);
                            fprintf(stderr, "  VERIFY FAIL: %s\n", key.c_str());
                        } else {
                            verify_ok.fetch_add(1);
                        }
                    }

                    // Write chunk at same path (zarr v2 -> zarr v2)
                    t_output->write(key, compressed);

                    int p_count = processed.fetch_add(1) + 1;
                    if (p_count % 500 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double secs = std::chrono::duration<double>(now - t0).count();
                        double mb_s = (double)total_compressed / (1024 * 1024) / secs;
                        std::lock_guard lk(print_mtx);
                        printf("  %d/%zu (%.0f chunks/s, %.1f MB/s) ratio: %.2f:1\n",
                               p_count, chunk_keys.size(), p_count / secs, mb_s,
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
        double mb_s = tc > 0 ? (double)tc / (1024 * 1024) / elapsed : 0;

        printf("  Processed: %d (skipped: %d, errors: %d)\n",
               processed.load(), skipped.load(), errs.load());
        printf("  Raw: %zu MB -> Compressed: %zu MB (ratio: %.2f:1)\n",
               tr / 1024 / 1024, tc / 1024 / 1024, ratio);
        printf("  Time: %.1fs (%.0f chunks/s, %.1f MB/s)\n",
               elapsed, chunk_keys.size() / elapsed, mb_s);
        if (verify) {
            printf("  Verify: %d OK, %d FAIL\n", verify_ok.load(), verify_fail.load());
        }
    }

    blosc_destroy();
    return 0;
}
