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
        R"({"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"|u1","compressor":{"id":"bz2"},"filters":null,"fill_value":0,"order":"C"})"));
    must_fail([&] { (void)unsupported.codec(); }, "unsupported compressor must be refused when decoding");
    must_fail([] { vcr::zarr::parseArrayMeta(Json::parse(R"({"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"|O","fill_value":0})")); },
              "object dtype");
}

void localShards(const fs::path& root, bool remote = false)
{
    for (bool end : {false, true}) for (bool crc : {false, true}) {
        const auto path = root / (std::string("shard-") + (end ? "end" : "start") + (crc ? "-crc" : ""));
        const vcr::ShardLayout layout({1, 1, 2}, end, crc, 2);
        auto fixture = [&](bool firstPresent) {
            std::vector<uint8_t> index(layout.indexBytes, 255), body;
            auto put = [&](size_t at, uint64_t value, int count) {
                for (int i = 0; i < count; ++i) index[at + i] = uint8_t(value >> (8 * i));
            };
            if (firstPresent) {
                put(0, end ? 0 : layout.indexBytes, 8);
                put(8, 2, 8);
                body = {1, 2};
            }
            put(16, body.size() + (end ? 0 : layout.indexBytes), 8);
            put(24, 2, 8);
            body.insert(body.end(), firstPresent ? std::initializer_list<uint8_t>{9, 10}
                                                : std::initializer_list<uint8_t>{7, 8});
            if (crc) put(index.size() - 4, vcr::crc32c(index.data(), index.size() - 4), 4);
            if (end) body.insert(body.end(), index.begin(), index.end());
            else body.insert(body.begin(), index.begin(), index.end());
            return std::string(reinterpret_cast<const char*>(body.data()), body.size());
        };
        vcr::zarr::atomicWrite(path, fixture(false));
        vcr::LocalShard snapshot(path, layout);
        vcr::zarr::atomicWrite(path, fixture(true));
        std::vector<uint8_t> got;
        require(!snapshot.readEntry(0, got), "snapshot retains missing entry");
        require(snapshot.readEntry(1, got) && got == std::vector<uint8_t>{7, 8},
                "index and payload use one file snapshot across replacement");
        require(vcr::LocalShard(path, layout).readEntry(1, got) && got == std::vector<uint8_t>{9, 10},
                "new shard reader sees replacement");
        std::atomic<uint64_t> requests{0}, bytes{0};
        std::unique_ptr<vcr::HttpClient> client;
        if (remote) client = std::make_unique<vcr::HttpClient>(vcr::HttpOptions{});
        vcr::ZarrStorage storage(remote ? sftp_fixture::url(root) : root.string(), "", ".", {1, 1, 2},
                                 end, crc, 2, client.get(), requests, bytes);
        vcr::zarr::atomicWrite(root / "0.0.0", fixture(true));
        require(storage.read("0.0.1", got) && got == std::vector<uint8_t>{9, 10}, "storage shard read");
        require(storage.read("0.0.0", got) && got == std::vector<uint8_t>{1, 2}, "second shard entry read");
        if (remote) require(requests == 3, "remote shard reuses its validated index");
        auto broken = fixture(true);
        broken.resize(3);
        vcr::zarr::atomicWrite(path, broken);
        must_fail([&] { vcr::LocalShard(path, layout); }, "truncated index refused");
        if (crc) {
            broken = fixture(true);
            broken[end ? broken.size() - 1 : layout.indexBytes - 1] ^= 1;
            vcr::zarr::atomicWrite(path, broken);
            must_fail([&] { vcr::LocalShard(path, layout); }, "corrupt index CRC refused");
        }
    }
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

void sparseLevels(const fs::path& root)
{
    const auto path = root / "sparse.zarr";
    write_text(path / "3/.zarray", kZarray);
    write_text(path / "10/.zarray", kZarray);
    auto group = vcr::zarr::Group::open(path.string(), {});
    require(group.levels() == std::vector<std::string>{"3", "10"}, "implicit sparse group numeric order");
    auto selected = vcr::zarr::openPyramidLevel(path.string(), 10, {});
    require(selected.location == (path / "10").string(), "sparse physical selection");
    must_fail([&] { vcr::zarr::openPyramidLevel(path.string(), 0, {}); }, "sparse levels must not rebase");
    must_fail([&] { vcr::zarr::openPyramidLevel(path.string(), -1, {}); }, "negative level refused");
    require(vcr::zarr::openPyramidLevel((path / "3").string(), 0, {}).meta.shape == vcr::Index{4,4,4},
            "root array selection");
    must_fail([&] { vcr::zarr::openPyramidLevel((path / "3").string(), 1, {}); }, "root has only level zero");

    write_text(path / ".zattrs", R"({"multiscales":[{"datasets":[{"path":"10"},{"path":"3"}]}]})");
    require(vcr::zarr::openPyramidLevel(path.string(), 3, {}).location == (path / "3").string(),
            "numeric OME paths retain physical level");
    write_text(path / ".zattrs", R"({"multiscales":[{"datasets":[{"path":"3"},{"path":"03"}]}]})");
    must_fail([&] { vcr::zarr::openPyramidLevel(path.string(), 3, {}); }, "duplicate numeric levels refused");
    write_text(path / "coarse/.zarray", kZarray);
    write_text(path / ".zattrs", R"({"multiscales":[{"datasets":[{"path":"3"},{"path":"coarse"}]}]})");
    require(vcr::zarr::openPyramidLevel(path.string(), 1, {}).location == (path / "coarse").string(),
            "named OME levels use advertised order");

    // A remote metadata hook sees the same sparse topology without needing a
    // directory listing. Only the selected level is opened, never its siblings.
    std::vector<std::string> requested;
    vcr::zarr::OpenOptions options;
    options.metadata = [&](const std::string& location, const std::string& name) -> std::optional<std::string> {
        requested.push_back(location + "/" + name);
        if (name == ".zgroup" && location == "https://example.invalid/sparse") return R"({"zarr_format":2})";
        if (name == ".zarray" && location == "https://example.invalid/sparse/3") return kZarray;
        return {};
    };
    auto remote = vcr::zarr::openPyramidLevel("https://example.invalid/sparse", 3, options, 5);
    require(remote.meta.shape == vcr::Index{4,4,4}, "remote bounded sparse probing");
    require(std::find(requested.begin(), requested.end(), "https://example.invalid/sparse/5/.zarray") == requested.end(),
            "remote probing respects limit");
    options.metadata = [&](const std::string& location, const std::string& name) -> std::optional<std::string> {
        if (location != "https://example.invalid/sparse") throw std::runtime_error("unexpected child metadata fetch");
        if (name == ".zgroup") return R"({"zarr_format":2})";
        if (name == ".zmetadata") return Json{{"metadata", {{"3/.zarray", Json::parse(kZarray)},
                                                           {"10/.zarray", Json::parse(kZarray)},
                                                           {"10/.zattrs", Json::object()}}}}.dump();
        return {};
    };
    auto indexed = vcr::zarr::Group::open("https://example.invalid/sparse", options);
    require(indexed.levels() == std::vector<std::string>{"3", "10"}, "consolidated sparse discovery");

    const auto v3path = root / "sparse-v3.zarr";
    vcr::zarr::CreateOptions create;
    create.version = 3; create.shape = {4,4,4}; create.chunks = {2,2,2};
    auto writer = vcr::zarr::Writer::create(v3path / "4", create);
    require(vcr::zarr::openPyramidLevel(v3path.string(), 4, {}).meta.v3.has_value(), "v3 sparse level");
    require(vcr::zarr::openPyramidLevel(writer.path().string(), 0, {}).meta.v3.has_value(), "v3 root array");
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

    // A bounded mirror stays the write layer and adopts pre-existing chunks.
    vc3dOptions.cacheMaxBytes=32;
    {
        auto a=vcr::zarr::Group::open(groupUrl,vc3dOptions).array("0");
        require(a.writer==0 && a.layers[0].kind=="vc3d-mirror" && !a.layers[0].budgetRoot.empty(),
                "bounded VC3D cache remains the writer");
        expect_volume(read_all(*a.array),v,"bounded mirror read");
        a.array->flush();
        require(a.array->last_write_error().empty(),"bounded mirror writes succeed");
        std::uintmax_t bytes=0;
        for(const auto& file:fs::directory_iterator(vdir/"0"))
            if(file.is_regular_file() && file.path().filename().string().find_first_not_of("0123456789.")==std::string::npos)
                bytes+=file.file_size();
        require(bytes<=32,"existing and newly fetched mirror chunks share the byte budget");
        require(fs::exists(vdir/"0/.zarray") && fs::exists(vdir/"0/1.1.1.empty"),"budget preserves metadata and fill markers");
    }

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

    // Budgeted Delta3D caches keep their format and reuse both data and fill.
    const auto delta=root/"delta-cache"/"my-volume";
    write_text(delta/".vc_delta3d_cache","D3D1\n");
    vcr::zarr::OpenOptions deltaOptions;
    deltaOptions.vc3dVolumeDir=delta;
    deltaOptions.cacheMaxBytes=8192;
    {
        auto a=vcr::zarr::Group::open(groupUrl,deltaOptions).array("0");
        require(a.layers[0].kind=="vc3d-delta3d" && a.writer==0,"bounded Delta3D writer");
        expect_volume(read_all(*a.array),v,"bounded Delta3D read");
        a.array->flush();
        require(a.array->last_write_error().empty(),"bounded Delta3D writes succeed");
    }
    {
        auto a=vcr::zarr::Group::open(groupUrl,deltaOptions).array("0");
        expect_volume(read_all(*a.array),v,"reused Delta3D read");
        require(a.array->stats().remote_fetches==0,"Delta3D data and fill reuse without downloading");
    }
    require(fs::exists(delta/".zattrs") && fs::exists(delta/".zgroup"),"VC3D group metadata mirrored");
    deltaOptions.offline=true;
    {
        auto g=vcr::zarr::Group::open(groupUrl,deltaOptions);
        require(g.levels()==std::vector<std::string>{"0","1"},"named cache offline multiscales metadata");
        auto a=g.array("0");
        expect_volume(read_all(*a.array),v,"offline named Delta3D cache");
        require(a.array->stats().remote_fetches==0,"offline named cache never downloads");
    }
    deltaOptions.offline=false;
    deltaOptions.cacheMaxBytes=0;
    {
        auto a=vcr::zarr::Group::open(groupUrl,deltaOptions).array("0");
        require(!fs::exists(delta/"0/0.0.0") && fs::exists(delta/"0/.zarray"),"Delta3D budget eviction preserves metadata");
    }

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
        localShards(root);
        local(root, v);
        sparseLevels(root);
        if (!sftp_fixture::install(root)) {
            std::cout << "metadata and local groups passed; no OpenSSH sftp-server for remote cases\n";
            return 0;
        }
        remote(root, v);
        localShards(root, true);
        std::cout << "metadata, local groups, remote decoded/VC3D caches online and offline passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
