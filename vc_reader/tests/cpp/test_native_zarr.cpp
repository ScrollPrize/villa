// The native opener: metadata parsing, local arrays and OME-Zarr groups, and
// remote (sftp://) arrays with decoded and VC3D disk caches, online and offline.
#include "vc_reader/zarr.hpp"
#include "sftp_fixture.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vcr::zarr::Json;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class F>
void must_fail(F&& f, const std::string& message)
{
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
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

// 4^3 uint16 little-endian volume in 2^3 chunks; chunk (1,1,1) is absent (fill 7).
std::vector<uint16_t> volume()
{
    std::vector<uint16_t> v(64);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = uint16_t(1000 + i);
    return v;
}

std::vector<uint8_t> chunk_bytes(const std::vector<uint16_t>& v, int cz, int cy, int cx, uint16_t add = 0)
{
    std::vector<uint8_t> out;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                const uint16_t value = v[((cz * 2 + z) * 4 + cy * 2 + y) * 4 + cx * 2 + x] + add;
                out.push_back(uint8_t(value & 255));
                out.push_back(uint8_t(value >> 8));
            }
    return out;
}

const std::string kZarray = R"({"zarr_format":2,"shape":[4,4,4],"chunks":[2,2,2],"dtype":"<u2",)"
                            R"("compressor":null,"filters":null,"fill_value":7,"order":"C"})";

// An OME-Zarr group with levels "0" (the volume) and "1" (2^3, every value 9).
void write_group(const fs::path& group, const std::vector<uint16_t>& v)
{
    write_text(group / ".zgroup", R"({"zarr_format":2})");
    write_text(group / ".zattrs", R"({"multiscales":[{"datasets":[{"path":"0"},{"path":"1"}]}]})");
    write_text(group / "0/.zarray", kZarray);
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx)
                if (cz + cy + cx < 3)
                    write_file(group / "0" / (std::to_string(cz) + "." + std::to_string(cy) + "." + std::to_string(cx)),
                               chunk_bytes(v, cz, cy, cx));
    write_text(group / "1/.zarray", R"({"zarr_format":2,"shape":[2,2,2],"chunks":[2,2,2],"dtype":"<u2",)"
                                    R"("compressor":null,"filters":null,"fill_value":7,"order":"C"})");
    write_file(group / "1/0.0.0", std::vector<uint8_t>(16, 0));
    std::vector<uint8_t> nines;
    for (int i = 0; i < 8; ++i) {
        nines.push_back(9);
        nines.push_back(0);
    }
    write_file(group / "1/0.0.0", nines);
}

std::vector<uint16_t> read_all(vcr::Array& array)
{
    std::vector<uint16_t> out(64);
    array.read_many({{0, 0, 0}}, {{4, 4, 4}}, {reinterpret_cast<uint8_t*>(out.data())});
    return out;
}

void expect_volume(const std::vector<uint16_t>& got, const std::vector<uint16_t>& v, const std::string& what,
                   uint16_t add = 0)
{
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) {
                const size_t i = (z * 4 + y) * 4 + x;
                const uint16_t want = z >= 2 && y >= 2 && x >= 2 ? 7 : uint16_t(v[i] + add);
                require(got[i] == want, what + ": voxel " + std::to_string(i) + " is " + std::to_string(got[i]) +
                                            ", expected " + std::to_string(want));
            }
}

void metadata()
{
    auto m = vcr::zarr::parseArrayMeta(Json::parse(kZarray));
    require(m.dtype == "<u2" && m.itemsize() == 2 && !m.type.swap && m.chunkBytes() == 16, "v2 dtype");
    require(m.fill == std::vector<uint8_t>{7, 0} && m.codec().empty() && !m.v3, "v2 fill and codec");
    require(m.decodedZarray()["compressor"].is_null(), "decoded layout");
    require(m.delta3dZarray()["compressor"]["id"] == "vc-delta3d", "delta3d layout");

    auto big = vcr::zarr::parseArrayMeta(Json::parse(
        R"({"zarr_format":2,"shape":[2],"chunks":[2],"dtype":">f4","compressor":{"id":"zlib"},"filters":null,"fill_value":"NaN","order":"C"})"));
    require(big.type.swap && big.type.kind == 'f' && std::isnan(big.type.load(big.fill.data())), "big-endian NaN fill");
    require(big.codec() == "zlib", "zlib codec id");

    auto v3 = vcr::zarr::parseArrayMeta(Json::parse(
        R"({"zarr_format":3,"node_type":"array","shape":[4,4],"data_type":"float32",)"
        R"("chunk_grid":{"name":"regular","configuration":{"chunk_shape":[4,4]}},)"
        R"("chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":"0x3f800000",)"
        R"("codecs":[{"name":"sharding_indexed","configuration":{"chunk_shape":[2,2],)"
        R"("codecs":[{"name":"bytes","configuration":{"endian":"little"}},{"name":"gzip","configuration":{"level":1}}],)"
        R"("index_codecs":[{"name":"bytes","configuration":{"endian":"little"}},{"name":"crc32c"}]}}],)"
        R"("attributes":{"note":"x"},"dimension_names":["y","x"]})"));
    require(v3.v3 && v3.v3->shards == vcr::Index{2, 2} && v3.v3->indexCrc && v3.v3->prefix == "c/", "v3 shards");
    require(v3.type.load(v3.fill.data()) == 1.0 && v3.codec() == "gzip", "v3 hex fill and codec");

    auto unsupported = vcr::zarr::parseArrayMeta(Json::parse(
        R"({"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"|u1","compressor":{"id":"lz4"},"filters":null,"fill_value":0,"order":"C"})"));
    must_fail([&] { (void)unsupported.codec(); }, "unsupported compressor must be refused when decoding");
    must_fail([] { vcr::zarr::parseArrayMeta(Json::parse(R"({"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"|O","fill_value":0})")); },
              "object dtype");
}

void local(const fs::path& root, const std::vector<uint16_t>& v)
{
    const auto group = root / "local.zarr";
    write_group(group, v);
    auto g = vcr::zarr::Group::open(group.string(), {});
    require(g.levels() == std::vector<std::string>{"0", "1"}, "multiscales levels");
    auto level0 = g.array("0");
    expect_volume(read_all(*level0.array), v, "local level 0");
    require(level0.meta.shape == vcr::Index{2 * 2, 4, 4}, "level shape");
    auto level1 = vcr::zarr::openArray((group / "1").string(), {});
    uint16_t one[8];
    level1.array->read_many({{0, 0, 0}}, {{2, 2, 2}}, {reinterpret_cast<uint8_t*>(one)});
    require(one[0] == 9 && one[7] == 9, "direct level open");
    require(!g.node("2") && g.node("1")->first == vcr::zarr::Group::NodeKind::Array, "child lookup");
    must_fail([&] { g.node("../x"); }, "path traversal refused");

    // Without multiscales, levels are the consecutive arrays "0", "1", ...
    write_text(group / ".zattrs", "{}");
    require(vcr::zarr::Group::open(group.string(), {}).levels().size() == 2, "plain numbered levels");
}

void remote(const fs::path& root, const std::vector<uint16_t>& v)
{
    const auto group = root / "remote.zarr";
    write_group(group, v);
    const auto groupUrl = sftp_fixture::url(group);
    const auto cache = root / "cache";

    // Online with a decoded cache: chunks are fetched and appended.
    vcr::zarr::OpenOptions options;
    options.cacheDirs = {cache.string()};
    {
        auto g = vcr::zarr::Group::open(groupUrl, options);
        auto a = g.array("0");
        require(a.layers.size() == 1 && a.layers[0].kind == "decoded" && a.writer == 0, "new decoded cache");
        expect_volume(read_all(*a.array), v, "remote online");
        a.array->flush();
        require(a.array->stats().remote_fetches.load() > 0, "chunks were fetched");
    }
    const auto decoded = vcr::zarr::decodedCachePath(cache, sftp_fixture::url(group / "0"));
    require(fs::exists(decoded / ".zarray") && fs::exists(decoded / "0.0.0") && fs::exists(decoded / "1.1.1.missing"),
            "decoded cache files and missing markers written");
    require(fs::exists(vcr::zarr::decodedCachePath(cache, groupUrl) / ".zattrs"), "group metadata cached");

    // Offline: metadata and chunks only from the caches, even with the source gone.
    fs::rename(group, root / "moved.zarr");
    options.offline = true;
    {
        auto g = vcr::zarr::Group::open(groupUrl, options);
        require(g.levels() == std::vector<std::string>{"0", "1"}, "offline group levels");
        auto a = g.array("0");
        expect_volume(read_all(*a.array), v, "remote offline");
        require(a.array->stats().remote_fetches.load() == 0, "offline never fetches");
        must_fail([&] { vcr::zarr::openArray(sftp_fixture::url(group / "1"), options); },
                  "offline array without a cache is a cache miss");
    }
    fs::rename(root / "moved.zarr", group);

    // A VC3D mirror cache of the group is found by its URL-derived directory, read
    // first, and appended to. Its stored chunk differs from the source on purpose.
    const auto vc3d = root / "vc3d";
    const auto vdir = vc3d / vcr::zarr::vc3dVolumeId(groupUrl);
    write_text(vdir / "0/.zarray", kZarray);
    write_file(vdir / "0/0.0.0", chunk_bytes(v, 0, 0, 0, 5000));
    vcr::zarr::OpenOptions vc3dOptions;
    vc3dOptions.cacheDirs = {vc3d.string()};
    {
        auto a = vcr::zarr::Group::open(groupUrl, vc3dOptions).array("0");
        require(a.layers.size() == 1 && a.layers[0].kind == "vc3d-mirror" && a.writer == 0, "VC3D mirror layer");
        std::vector<uint16_t> first(8);
        a.array->read_many({{0, 0, 0}}, {{2, 2, 2}}, {reinterpret_cast<uint8_t*>(first.data())});
        require(first[0] == uint16_t(v[0] + 5000), "the mirror's chunk is read, not the source's");
        read_all(*a.array);
        a.array->flush();
    }
    require(fs::exists(vdir / "0/0.1.0") && fs::exists(vdir / "0/1.1.1.empty"), "VC3D mirror appended in its format");

    // An owner-named VC3D directory is used as given (VC3D names caches by volume ID).
    const auto named = root / "named" / "my-volume";
    fs::create_directories(named);
    vcr::zarr::OpenOptions namedOptions;
    namedOptions.vc3dVolumeDir = named;
    {
        auto a = vcr::zarr::Group::open(groupUrl, namedOptions).array("0");
        require(a.layers.size() == 1 && a.layers[0].kind == "vc3d-mirror" && a.layers[0].root == (named / "0").string(),
                "owner-named VC3D layer");
        expect_volume(read_all(*a.array), v, "owner-named VC3D cache");
        a.array->flush();
    }
    require(fs::exists(named / "0/.zarray") && fs::exists(named / "0/0.0.0"), "owner-named cache filled");

    // A changed source refuses a stale decoded cache instead of mixing arrays.
    write_text(group / "0/.zarray", R"({"zarr_format":2,"shape":[4,4,6],"chunks":[2,2,2],"dtype":"<u2",)"
                                    R"("compressor":null,"filters":null,"fill_value":7,"order":"C"})");
    options.offline = false;
    must_fail([&] { vcr::zarr::Group::open(groupUrl, options).array("0"); }, "stale decoded cache refused");
}
}  // namespace

int main(int argc, char** argv)
{
    if (const int status = sftp_fixture::run_as_ssh(argc, argv); status >= 0)
        return status;
    const auto root = fs::temp_directory_path() /
                      ("vc-reader-zarr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};
    try {
        fs::create_directories(root);
        const auto v = volume();
        metadata();
        local(root, v);
        if (!sftp_fixture::install(root)) {
            std::cout << "metadata and local groups passed; no OpenSSH sftp-server for remote cases\n";
            return 0;
        }
        remote(root, v);
        std::cout << "metadata, local groups, remote decoded/VC3D caches online and offline passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
