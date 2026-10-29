// Chunks compressed with zlib, gzip and volcomp read back through the native reader,
// from Zarr v2 compressors and Zarr v3 codec pipelines (plain and sharded).
#include "vc_reader/reader.hpp"

#include <volcomp_lib.h>
#include <zlib.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<uint8_t> deflate_bytes(const std::vector<uint8_t>& data, bool gzip)
{
    z_stream z{};
    require(deflateInit2(&z, 6, Z_DEFLATED, gzip ? 15 + 16 : 15, 8, Z_DEFAULT_STRATEGY) == Z_OK, "deflate init");
    std::vector<uint8_t> out(deflateBound(&z, data.size()));
    z.next_in = const_cast<Bytef*>(data.data());
    z.avail_in = static_cast<uInt>(data.size());
    z.next_out = out.data();
    z.avail_out = static_cast<uInt>(out.size());
    require(deflate(&z, Z_FINISH) == Z_STREAM_END, "deflate");
    out.resize(z.total_out);
    deflateEnd(&z);
    return out;
}

std::vector<uint8_t> volcomp_lossless(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(volcomp_lib_encode_bound());
    size_t n = 0;
    require(volcomp_lib_encode(data.data(), VOLCOMP_LIB_Q_LOSSLESS, out.data(), out.size(), &n) == VOLCOMP_LIB_OK,
            "volcomp encode");
    out.resize(n);
    return out;
}

void write_file(const fs::path& path, const std::vector<uint8_t>& bytes)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

void write_text(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

// Two 128^3 chunks along x; the second is absent and reads as the fill value.
constexpr int64_t kEdge = 128;

void check(const fs::path& array, const std::vector<uint8_t>& data, const std::string& what)
{
    vcr::CtReader reader(array.string());
    require(reader.shape() == std::array<int64_t, 3>{kEdge, kEdge, 2 * kEdge}, what + ": shape");
    std::vector<uint8_t> out(kEdge * kEdge * 2 * kEdge);
    reader.readRegion({0, 0, 0}, {kEdge, kEdge, 2 * kEdge}, out);
    for (int64_t z = 0; z < kEdge; ++z)
        for (int64_t y = 0; y < kEdge; ++y)
            for (int64_t x = 0; x < 2 * kEdge; ++x) {
                const uint8_t got = out[(z * kEdge + y) * 2 * kEdge + x];
                const uint8_t want = x < kEdge ? data[(z * kEdge + y) * kEdge + x] : 9;
                if (got != want)
                    throw std::runtime_error(what + ": voxel " + std::to_string(z) + "," + std::to_string(y) + "," +
                                             std::to_string(x) + " is " + std::to_string(got) + ", expected " +
                                             std::to_string(want));
            }
}

std::string v2_zarray(const std::string& compressor)
{
    return R"({"zarr_format":2,"shape":[128,128,256],"chunks":[128,128,128],"dtype":"|u1","compressor":)" +
           compressor + R"(,"filters":null,"fill_value":9,"order":"C"})";
}

std::string v3_zarr_json(const std::string& codec, bool sharded)
{
    const std::string inner = R"([{"name":"bytes","configuration":{"endian":"little"}},)" + codec + "]";
    const std::string codecs = sharded
        ? R"([{"name":"sharding_indexed","configuration":{"chunk_shape":[128,128,128],"codecs":)" + inner +
              R"(,"index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"index_location":"end"}}])"
        : inner;
    return R"({"zarr_format":3,"node_type":"array","shape":[128,128,256],"data_type":"uint8",)"
           R"("chunk_grid":{"name":"regular","configuration":{"chunk_shape":[128,128,)" +
           std::string(sharded ? "256" : "128") +
           R"(]}},"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},)"
           R"("fill_value":9,"codecs":)" + codecs + "}";
}

// One shard holding chunk (0,0,0) and leaving (0,0,1) empty; index at the end.
std::vector<uint8_t> shard(const std::vector<uint8_t>& chunk)
{
    std::vector<uint8_t> out(chunk);
    const uint64_t entries[4] = {0, chunk.size(), ~uint64_t(0), ~uint64_t(0)};
    const auto* p = reinterpret_cast<const uint8_t*>(entries);
    out.insert(out.end(), p, p + sizeof(entries));
    return out;
}
}  // namespace

int main()
{
    const auto root = fs::temp_directory_path() /
                      ("vc-reader-codecs-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};
    try {
        std::vector<uint8_t> data(kEdge * kEdge * kEdge);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = uint8_t(128 + 100 * std::sin(double(i % 977) * 0.05) + double(i / (kEdge * kEdge)) * 0.1);

        const auto zlib = deflate_bytes(data, false), gzip = deflate_bytes(data, true), volcomp = volcomp_lossless(data);
        struct Case {
            std::string name, v2, v3;
            std::vector<uint8_t> chunk;
        };
        const std::vector<Case> cases{
            {"zlib", R"({"id":"zlib","level":6})", R"({"name":"numcodecs.zlib","configuration":{"level":6}})", zlib},
            {"gzip", R"({"id":"gzip","level":6})", R"({"name":"gzip","configuration":{"level":6}})", gzip},
            {"volcomp", R"({"id":"volcomp","q":0})", R"({"name":"volcomp","configuration":{"q":0}})", volcomp},
        };
        for (const auto& c : cases) {
            const auto v2 = root / ("v2-" + c.name);
            write_text(v2 / ".zarray", v2_zarray(c.v2));
            write_file(v2 / "0.0.0", c.chunk);
            check(v2, data, c.name + " v2");

            const auto v3 = root / ("v3-" + c.name);
            write_text(v3 / "zarr.json", v3_zarr_json(c.v3, false));
            write_file(v3 / "c/0/0/0", c.chunk);
            check(v3, data, c.name + " v3");

            const auto sharded = root / ("v3-sharded-" + c.name);
            write_text(sharded / "zarr.json", v3_zarr_json(c.v3, true));
            write_file(sharded / "c/0/0/0", shard(c.chunk));
            check(sharded, data, c.name + " v3 sharded");
        }

        // A truncated stream is an error, not silently short data.
        const auto broken = root / "v2-broken";
        write_text(broken / ".zarray", v2_zarray(R"({"id":"zlib","level":6})"));
        write_file(broken / "0.0.0", std::vector<uint8_t>(zlib.begin(), zlib.begin() + zlib.size() / 2));
        bool failed = false;
        try {
            check(broken, data, "truncated zlib");
        } catch (const std::runtime_error& e) {
            failed = std::string(e.what()).find("zlib") != std::string::npos;
        }
        require(failed, "truncated zlib chunk must fail to decode");
        std::cout << "zlib, gzip and volcomp chunks passed (v2, v3, v3 sharded)\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
