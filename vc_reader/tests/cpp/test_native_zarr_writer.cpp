// Native Zarr writes: every codec round-trips through the reader, empty chunks are
// skipped, regions are read-modified-written, and attributes and metadata are kept.
#include "vc_reader/zarr.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/zarr_write.hpp"
#include "vc_reader/native/rounding.hpp"
#include "vc_reader/native/sigv4.hpp"

#include <algorithm>
#include <barrier>
#include <future>
#include <chrono>
#include <cstring>
#include <cmath>
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

std::vector<uint8_t> read(vcr::Array& array, const vcr::Index& start, const vcr::Index& shape, std::size_t itemsize)
{
    std::size_t n = itemsize;
    for (const auto s : shape)
        n *= std::size_t(s);
    std::vector<uint8_t> out(n);
    array.read_many({start}, {shape}, {out.data()});
    return out;
}

void codecs(const fs::path& root, int version, bool sharded = false)
{
    struct Case {
        std::string compressor, dtype;
        vcr::Index chunks;
    };
    const std::vector<Case> cases{
        {"none", "<u2", {8, 8, 8}},  {"blosc", "<u2", {8, 8, 8}},  {"zstd", "|u1", {8, 8, 8}},
        {"zlib", "<f4", {8, 8, 8}},  {"gzip", "|u1", {8, 8, 8}},   {"lz4", "<u2", {8, 8, 8}},
        {"vc-delta3d", "<u2", {8, 8, 8}}, {"volcomp", "|u1", {128, 128, 128}},
    };
    for (const auto& c : cases) {
        vcr::zarr::CreateOptions options;
        options.version = version;
        options.chunks = c.chunks;
        options.shape = {c.chunks[0], c.chunks[1], 2 * c.chunks[2]};
        if (sharded) options.shardShape = options.shape;
        options.dtype = c.dtype;
        options.compressor = c.compressor;
        options.level = c.compressor == "volcomp" ? 0 : -1;  // volcomp q 0: lossless
        options.fillValue = 3;
        auto writer = vcr::zarr::Writer::create(root / c.compressor, options);
        const auto& meta = writer.meta();
        std::vector<uint8_t> chunk(meta.chunkBytes());
        for (std::size_t i = 0; i < chunk.size(); ++i)
            chunk[i] = uint8_t((i * 7 + i / 977) % 251);
        if (c.dtype == "<f4")  // finite floats
            for (std::size_t i = 0; i < chunk.size(); i += 4) {
                const float v = float(i) * 0.25f;
                std::memcpy(chunk.data() + i, &v, 4);
            }
        writer.writeChunk({0, 0, 0}, chunk.data(), chunk.size());
        const double zyx[]{0, 0, 0};
        std::vector<std::uint8_t> sampled(meta.itemsize());
        writer.reader().sample_nearest(zyx, 1, sampled.data());
        require(std::equal(sampled.begin(), sampled.end(), chunk.begin()), "the written array samples its chunk");
        // The second chunk is never written: it reads as fill.
        auto reopened = vcr::zarr::openArray((root / c.compressor).string(), {});
        const auto all = read(*reopened.array, {0, 0, 0}, meta.shape, meta.itemsize());
        const auto row = std::size_t(c.chunks[2]) * meta.itemsize();
        for (int64_t zy = 0; zy < c.chunks[0] * c.chunks[1]; ++zy) {
            require(std::equal(chunk.begin() + zy * row, chunk.begin() + (zy + 1) * row, all.begin() + 2 * zy * row),
                    c.compressor + ": stored chunk round trip");
            const auto fill = all.begin() + (2 * zy + 1) * row;
            require(meta.itemsize() != 1 || std::all_of(fill, fill + row, [](uint8_t b) { return b == 3; }),
                    c.compressor + ": absent chunk reads as fill");
        }
    }
}

void editing(const fs::path& root, int version)
{
    vcr::zarr::CreateOptions options;
    options.version = version;
    options.shape = {5, 6, 7};  // edge chunks are partial
    options.chunks = {2, 4, 4};
    options.dtype = "<u2";
    options.compressor = "zstd";
    options.fillValue = 9;
    options.separator = "/";
    auto w = vcr::zarr::Writer::create(root / "group" / "edit", options);
    require(fs::exists(root / "group" / (version == 3 ? "zarr.json" : ".zgroup")), "parent group marker");
    require(vcr::zarr::Group::open((root / "group").string(), {}).version() == version, "parent group version");

    // A region straddling chunks; partially covered chunks keep their other values.
    std::vector<uint16_t> full(5 * 6 * 7);
    for (std::size_t i = 0; i < full.size(); ++i)
        full[i] = uint16_t(100 + i);
    w.writeRegion({0, 0, 0}, {5, 6, 7}, reinterpret_cast<const uint8_t*>(full.data()));
    std::vector<uint16_t> patch(2 * 3 * 3, 7777);
    w.writeRegion({1, 2, 3}, {2, 3, 3}, reinterpret_cast<const uint8_t*>(patch.data()));
    for (int z = 0; z < 5; ++z)
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 7; ++x)
                if (z >= 1 && z < 3 && y >= 2 && y < 5 && x >= 3 && x < 6)
                    full[(z * 6 + y) * 7 + x] = 7777;
    auto got = read(w.reader(), {0, 0, 0}, {5, 6, 7}, 2);
    require(std::memcmp(got.data(), full.data(), got.size()) == 0, "region read-modify-write");
    require(fs::exists(w.chunkPath({0, 0, 0})) && w.chunkPath({0, 0, 0}).filename() == "0" &&
            w.chunkPath({1, 1, 1}).parent_path().filename() == "1", "nested chunk keys");

    // An all-fill chunk is not stored, and removes what was there.
    std::vector<uint16_t> fills(2 * 4 * 4, 9);
    require(!w.writeChunkSkipEmpty({0, 0, 0}, reinterpret_cast<const uint8_t*>(fills.data()), fills.size() * 2) &&
            !w.chunkExists({0, 0, 0}), "empty chunk skipped and removed");
    got = read(w.reader(), {0, 0, 0}, {1, 1, 1}, 2);
    require(got[0] == 9 && got[1] == 0, "removed chunk reads as fill");
    require(!w.removeChunk({0, 0, 0}), "removing an absent chunk");

    must_fail([&] { w.writeRegion({4, 0, 0}, {2, 1, 1}, nullptr); }, "region outside the array refused");
    must_fail([&] { w.writeChunk({0, 0, 0}, nullptr, 3); }, "wrong chunk size refused");
    must_fail([&] { w.chunkPath({3, 0, 0}); }, "chunk index outside the grid refused");

    vcr::zarr::writeAttributes(root / "group", {{"multiscales", Json::array()}});
    require(vcr::zarr::readAttributes(root / "group").contains("multiscales"), "attributes round trip");
    require(vcr::zarr::readAttributes(root / "missing").empty(), "absent attributes");

    auto reopened = vcr::zarr::Writer::open(root / "group" / "edit");
    require(reopened.meta().separator == "/" && reopened.meta().codec() == "zstd", "reopened metadata");
    must_fail([&] { reopened.writeChunkSkipEmpty({1, 1, 1}, nullptr, 0); }, "empty buffer is not a fill chunk");
    require(reopened.chunkExists({1, 1, 1}), "invalid skip-empty leaves stored data intact");
    auto differentFormat = options;
    differentFormat.version = version == 2 ? 3 : 2;
    must_fail([&] { vcr::zarr::Writer::create(reopened.path(), differentFormat); }, "format replacement refused");

    vcr::zarr::CreateOptions bad = options;
    bad.compressor = "volcomp";
    must_fail([&] { vcr::zarr::Writer::create(root / "bad", bad); }, "volcomp needs 128^3 uint8");
    bad.compressor = "c3d";
    must_fail([&] { vcr::zarr::Writer::create(root / "bad", bad); }, "unknown compressor refused");
    require(!fs::exists(root / "bad" / ".zarray"), "refused arrays write no metadata");
    require(!fs::exists(root / "bad" / "zarr.json"), "refused v3 arrays write no metadata");
}

void v3Metadata(const fs::path& root)
{
    vcr::zarr::CreateOptions options;
    options.version = 3;
    options.shape = {3, 4};
    options.chunks = {2, 2};
    options.dtype = ">u2";
    options.fillValue = 513;
    options.compressor = "none";
    auto w = vcr::zarr::Writer::create(root / "v3-metadata" / "0", options);
    require(w.meta().v3->prefix == "c." && w.chunkPath({1, 1}).filename() == "c.1.1", "default dot keys");
    std::vector<uint8_t> chunk{0, 3, 1, 4, 2, 5, 3, 6};
    w.writeChunk({0, 0}, chunk.data(), chunk.size());
    require(read(w.reader(), {0, 0}, {2, 2}, 2) == chunk, "big endian bytes round trip");
    require(read(w.reader(), {2, 2}, {1, 1}, 2) == std::vector<uint8_t>{2, 1}, "big endian fill");
    vcr::zarr::writeAttributes(w.path(), {{"units", "labels"}});
    const auto before = w.meta().v3->metadata;
    const auto disk = Json::parse(std::ifstream(w.path() / "zarr.json"));
    require(disk.at("data_type") == "uint16" && disk.at("codecs")[0]["configuration"]["endian"] == "big",
            "v3 datatype and bytes endian");
    auto expected = before;
    expected["attributes"] = {{"units", "labels"}};
    require(disk == expected && !fs::exists(w.path() / ".zattrs"), "attributes preserve v3 metadata");
    must_fail([&] { vcr::zarr::writeAttributes(w.path(), Json::array()); }, "nonobject attributes refused");

    // Edit an existing array using v2 chunk key encoding within a v3 node.
    expected["chunk_key_encoding"]["name"] = "v2";
    vcr::zarr::atomicWrite(w.path() / "zarr.json", expected.dump());
    auto v2keys = vcr::zarr::Writer::open(w.path());
    require(v2keys.chunkPath({0, 0}).filename() == "0.0", "v3 node with v2 keys");
    v2keys.writeChunk({0, 0}, chunk.data(), chunk.size());
    require(read(v2keys.reader(), {0, 0}, {2, 2}, 2) == chunk, "v2-key v3 readback");

    expected["codecs"].push_back({{"name", "zstd"}, {"configuration", {{"level", 5}, {"checksum", true}}}});
    vcr::zarr::atomicWrite(w.path() / "zarr.json", expected.dump());
    auto checksum = vcr::zarr::Writer::open(w.path());
    checksum.writeChunk({0, 0}, chunk.data(), chunk.size());
    std::ifstream encoded(checksum.chunkPath({0, 0}), std::ios::binary);
    std::array<uint8_t, 5> header{};
    encoded.read(reinterpret_cast<char*>(header.data()), header.size());
    require((header[4] & 4) != 0, "zstd writer honors frame checksum metadata");
    require(read(checksum.reader(), {0, 0}, {2, 2}, 2) == chunk, "checksummed zstd readback");
}

void mutableChunks(const fs::path& root, int version, bool sharded = false)
{
    for (const auto* codec : {"none", "gzip"}) {
        vcr::zarr::CreateOptions options;
        options.version = version;
        options.shape = {3, 3, 3};
        if (sharded) options.shardShape = {4, 4, 4};
        options.chunks = {2, 2, 2};
        options.dtype = "<u2";
        options.compressor = codec;
        options.fillValue = 513;
        const auto path = root / (std::string("mutable-") + codec);
        auto writer = vcr::zarr::Writer::create(path, options);
        std::vector<uint16_t> chunk(8, 42), got(8, 0);
        auto* bytes = reinterpret_cast<uint8_t*>(got.data());
        writer.writeChunk({1, 1, 1}, reinterpret_cast<uint8_t*>(chunk.data()), 16);
        require(writer.reader().read_chunk({1, 1, 1}, bytes, 16) && got == chunk,
                "full edge chunk retains stored padding");
        require(!writer.reader().read_chunk({0, 0, 0}, bytes, 16) && got == std::vector<uint16_t>(8, 513),
                "missing chunk materializes a multi-byte fill");
        must_fail([&] { writer.reader().read_chunk({0, 0, 0}, bytes, 15); }, "wrong destination size refused");
        must_fail([&] { writer.reader().read_chunk({2, 0, 0}, bytes, 16); }, "invalid chunk index refused");
        const auto untouched = got;
        must_fail([&] {
            writer.reader().read_many({{0, 0, 0}, {2, 2, 2}}, {{1, 1, 1}, {2, 1, 1}},
                                     {bytes, bytes + 2}, vcr::Array::Bounds::Strict);
        }, "strict batch refuses an out-of-bounds region");
        require(got == untouched, "strict refusal leaves all outputs untouched");
        must_fail([&] {
            writer.reader().read_many({{0, 0, 0}}, {{INT64_MAX, 1, 1}}, {bytes}, vcr::Array::Bounds::Strict);
        }, "strict bounds check avoids overflow");
        must_fail([&] {
            writer.reader().read_many({{0, 0, 0}}, {}, {bytes});
        }, "mismatched batch counts refused");
        writer.reader().read_many({{-1}}, {{0}}, {nullptr}, vcr::Array::Bounds::Strict);
        writer.reader().read_many({{0, 0, 0}}, {{2, 2, 2}}, {bytes}, vcr::Array::Bounds::Strict);
        require(got == std::vector<uint16_t>(8, 513), "strict reads still fill missing chunks");

        // Read, replace through another writer, then patch twice. A retained
        // mmap of the first file loses edits or hides the replacement/removal.
        auto other = vcr::zarr::Writer::open(path);
        read(writer.reader(), {2, 2, 2}, {1, 1, 1}, 2);
        std::fill(chunk.begin(), chunk.end(), 77);
        other.writeChunk({1, 1, 1}, reinterpret_cast<uint8_t*>(chunk.data()), 16);
        auto region = read(writer.reader(), {2, 2, 2}, {1, 1, 1}, 2);
        require(region[0] == 77 && region[1] == 0, "reader sees another writer's atomic replacement");
        uint16_t a = 1234, b = 5678;
        writer.writeRegion({0, 0, 0}, {1, 1, 1}, reinterpret_cast<uint8_t*>(&a));
        writer.writeRegion({0, 0, 1}, {1, 1, 1}, reinterpret_cast<uint8_t*>(&b));
        writer.reader().read_chunk({0, 0, 0}, bytes, 16);
        require(got[0] == a && got[1] == b && got[2] == 513, "repeated partial writes keep previous edits");
        other.removeChunk({1, 1, 1});
        region = read(writer.reader(), {2, 2, 2}, {1, 1, 1}, 2);
        require(region[0] == 1 && region[1] == 2, "reader sees removal after a read");

        const auto before = writer.meta().raw;
        vcr::zarr::writeAttributes(path, {{"purpose", "labels"}});
        writer.setFillValue(1027);
        auto expected = before;
        expected["fill_value"] = 1027;
        if (version == 3) {
            expected["_vc_reader_v3"]["metadata"]["fill_value"] = 1027;
            expected["_vc_reader_v3"]["metadata"]["attributes"] = {{"purpose", "labels"}};
        }
        auto reopened = vcr::zarr::openArray(path.string(), {});
        require(writer.meta().raw == expected && reopened.meta.raw == expected,
                "fill update preserves all other array metadata");
        require(vcr::zarr::readAttributes(path)["purpose"] == "labels", "fill update preserves attributes");
        require(!writer.reader().read_chunk({1, 1, 1}, bytes, 16) && got == std::vector<uint16_t>(8, 1027),
                "fill update reaches the writer reader");
        writer.reader().read_chunk({0, 0, 0}, bytes, 16);
        require(got[0] == a && got[1] == b && got[2] == 513, "fill update leaves stored chunks unchanged");
        must_fail([&] { writer.setFillValue(Json::object()); }, "invalid fill refused before writing");
        require(vcr::zarr::openArray(path.string(), {}).meta.raw == expected,
                "failed fill update leaves metadata unchanged");
    }
}

std::string fileBytes(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void shardEditing(const fs::path& root)
{
    for (bool end : {false, true}) for (bool crc : {false, true}) for (auto sep : {".", "/"}) {
        vcr::zarr::CreateOptions o;
        o.version = 3;
        o.shape = {5, 7, 9}; o.chunks = {2, 3, 4}; o.shardShape = {4, 6, 8};
        o.shardIndexEnd = end; o.shardIndexCrc = crc; o.separator = sep;
        o.compressor = "gzip"; o.fillValue = 3;
        const auto path = root / (std::string("edit-") + (end ? "end" : "start") +
                                 (crc ? "-crc" : "") + (sep[0] == '.' ? "-dot" : "-slash"));
        auto w = vcr::zarr::Writer::create(path, o);
        require(w.meta().chunks == o.chunks && w.meta().v3->shards == vcr::Index{2, 2, 2}, "logical and shard geometry");
        std::vector<uint8_t> expected(5 * 7 * 9);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = uint8_t(i % 251);
        w.writeRegion({0, 0, 0}, o.shape, expected.data());
        std::vector<uint8_t> patch(3 * 3 * 4, 253);
        w.writeRegion({1, 2, 3}, {3, 3, 4}, patch.data());
        for (int z = 1; z < 4; ++z) for (int y = 2; y < 5; ++y) for (int x = 3; x < 7; ++x)
            expected[(z * 7 + y) * 9 + x] = 253;
        require(read(w.reader(), {0, 0, 0}, o.shape, 1) == expected, "region patch crosses chunks and shards");
        require(w.chunkPath({0, 0, 0}) == w.chunkPath({1, 1, 1}), "chunks share physical shard path");
        const auto& v3 = *w.meta().v3;
        const vcr::ShardLayout layout(v3.shards, v3.indexEnd, v3.indexCrc, w.meta().chunkBytes());
        std::vector<uint8_t> before, after;
        require(vcr::LocalShard(w.chunkPath({0, 0, 1}), layout).readEntry(1, before), "encoded neighbor exists");
        std::vector<uint8_t> fill(w.meta().chunkBytes(), 3);
        require(!w.writeChunkSkipEmpty({0, 0, 0}, fill.data(), fill.size()), "fill chunk omitted within shard");
        require(!w.chunkExists({0, 0, 0}) && w.chunkExists({0, 0, 1}), "chunk existence uses index");
        require(vcr::LocalShard(w.chunkPath({0, 0, 1}), layout).readEntry(1, after) && after == before,
                "neighbor's encoded bytes survive editing unchanged");
        require(read(w.reader(), {0, 0, 0}, o.chunks, 1) == fill, "removed entry reads fill");
        require(!w.removeChunk({0, 0, 0}), "absent entry removal is a no-op");

        // Corrupt input cannot be silently overwritten or damage other chunks.
        const auto file = w.chunkPath({0, 0, 1});
        const auto valid = fileBytes(file);
        auto broken = valid;
        if (crc) broken[end ? broken.size() - 1 : layout.indexBytes - 1] ^= 1;
        else {
            const size_t indexStart = end ? broken.size() - layout.indexBytes : 0;
            // A present entry with an impossible offset.
            std::fill_n(broken.begin() + indexStart + 16, 8, char(255));
        }
        vcr::zarr::atomicWrite(file, broken);
        must_fail([&] { w.writeChunk({0, 0, 0}, fill.data(), fill.size()); }, "corrupt shard update refused");
        require(fileBytes(file) == broken, "failed update preserves original shard");
        must_fail([&] { w.removeChunk({0, 0, 1}); }, "corrupt shard deletion refused");
        vcr::zarr::atomicWrite(file, valid);
        for (int z = 0; z < 3; ++z) for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x)
            w.removeChunk({z, y, x});
        require(!fs::exists(file), "last chunk removal deletes shard");
        require(read(w.reader(), {0, 0, 0}, o.shape, 1) == std::vector<uint8_t>(expected.size(), 3), "all removed shards read fill");
        for (const auto& entry : fs::recursive_directory_iterator(path))
            require(entry.path().filename().string().find(".tmp.") == std::string::npos, "no temporary shard files left");
    }
    vcr::zarr::CreateOptions bad;
    bad.version = 3; bad.shape = {5, 5, 5}; bad.chunks = {2, 2, 2};
    for (const auto& shards : {vcr::Index{4, 4}, vcr::Index{4, 4, 5}, vcr::Index{0, 4, 4},
                              vcr::Index{INT64_MAX - 1, 4, 4}}) {
        bad.shardShape = shards;
        must_fail([&] { vcr::zarr::Writer::create(root / "bad-shards", bad); }, "invalid shard geometry refused");
        require(!fs::exists(root / "bad-shards"), "invalid geometry writes no files");
    }
    bad.version = 2; bad.shardShape = {4, 4, 4};
    must_fail([&] { vcr::zarr::Writer::create(root / "bad-shards", bad); }, "v2 shards refused");
}

void concurrentShards(const fs::path& root)
{
    vcr::zarr::CreateOptions o;
    o.version = 3; o.shape = {4, 4, 4}; o.chunks = {2, 2, 2}; o.shardShape = o.shape;
    o.compressor = "gzip";
    auto w = vcr::zarr::Writer::create(root / "concurrent", o);
    std::barrier ready(8);
    std::vector<std::future<void>> workers;
    for (int t = 0; t < 8; ++t) workers.push_back(std::async(std::launch::async, [&, t] {
        auto writer = vcr::zarr::Writer::open(t % 2 ? fs::relative(w.path()) : w.path());
        const vcr::Index key{t / 4, (t / 2) % 2, t % 2};
        std::vector<uint8_t> chunk(8);
        ready.arrive_and_wait();
        for (int repeat = 0; repeat < 20; ++repeat) {
            std::fill(chunk.begin(), chunk.end(), uint8_t(1 + t + 8 * repeat));
            writer.writeChunk(key, chunk.data(), chunk.size());
        }
    }));
    for (auto& worker : workers) worker.get();
    for (int t = 0; t < 8; ++t) {
        std::vector<uint8_t> chunk(8);
        require(w.reader().read_chunk({t / 4, (t / 2) % 2, t % 2}, chunk.data(), chunk.size()) &&
                chunk == std::vector<uint8_t>(8, uint8_t(1 + t + 8 * 19)), "concurrent distinct chunks retain every update");
    }
}
void pyramids(const fs::path& root)
{
    // A 2x2x1-style pyramid (Z kept) of an odd-shaped volume spanning output chunks.
    const auto dir = root / "pyramid";
    vcr::zarr::CreateOptions create;
    create.shape = {2, 7, 9};
    create.chunks = {2, 4, 4};
    create.compressor = "zstd";
    auto level0 = vcr::zarr::Writer::create(dir / "0", create);
    std::vector<std::uint8_t> voxels(2 * 7 * 9);
    for (std::size_t i = 0; i < voxels.size(); ++i)
        voxels[i] = std::uint8_t(i * 3);
    level0.writeRegion({0, 0, 0}, {2, 7, 9}, voxels.data());
    vcr::zarr::PyramidOptions options;
    options.levels = 2;
    options.factors = {1, 2, 2};
    vcr::zarr::createPyramidLevels(dir, create, options);
    require(vcr::zarr::Writer::open(dir / "1").meta().shape == vcr::Index({2, 4, 5}), "level 1 shape rounds up");
    require(vcr::zarr::Writer::open(dir / "2").meta().chunks == vcr::Index({2, 2, 3}), "chunks capped to the level");
    require(vcr::zarr::Writer::open(dir / "1").meta().codec() == level0.meta().codec(), "levels keep the codec");

    // Two disjoint chunk ranges write what one call would.
    vcr::zarr::writePyramidLevel(dir, 1, options, vcr::zarr::ChunkRange{{0, 0, 0}, {1, 1, 99}});
    vcr::zarr::writePyramidLevel(dir, 1, options, vcr::zarr::ChunkRange{{0, 1, 0}, {1, 99, 99}});
    vcr::zarr::writePyramidLevel(dir, 2, options);
    const auto at = [&](int z, int y, int x) { return int(voxels[std::size_t((z * 7 + y) * 9 + x)]); };
    const auto mean = [](int sum, int count) { return int(vcr::divide_round_half_even(std::uint64_t(sum), std::uint64_t(count))); };
    auto level1 = vcr::zarr::Writer::open(dir / "1");
    const auto one = read(level1.reader(), {0, 0, 0}, {2, 4, 5}, 1);
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 5; ++x) {
                int sum = 0, count = 0;
                for (int j = 2 * y; j < std::min(2 * y + 2, 7); ++j)
                    for (int i = 2 * x; i < std::min(2 * x + 2, 9); ++i) {
                        sum += at(z, j, i);
                        ++count;
                    }
                require(one[std::size_t((z * 4 + y) * 5 + x)] == mean(sum, count), "block mean, clipped at the edge");
            }
    require(vcr::divide_round_half_even(5, 2) == 2 && vcr::divide_round_half_even(7, 2) == 4 &&
                vcr::divide_round_half_even(11, 4) == 3,
            "integer means round to nearest, ties to even");
    require([&] { auto l2 = vcr::zarr::Writer::open(dir / "2"); return read(l2.reader(), {0, 0, 0}, {1, 1, 1}, 1).size() == 1; }(), "level 2 written");

    // Other dtypes, and chunks of pure fill are not stored.
    for (const std::string dtype : {"<u2", "<f4"}) {
        const auto other = root / ("pyramid" + dtype.substr(1));
        vcr::zarr::CreateOptions c;
        c.shape = {8, 8, 8};
        c.chunks = {2, 2, 4};
        c.dtype = dtype;
        c.compressor = "";
        auto base = vcr::zarr::Writer::create(other / "0", c);
        vcr::zarr::PyramidOptions cubes;
        cubes.levels = 1;
        vcr::zarr::createPyramidLevels(other, c, cubes);
        if (dtype == "<u2") {
            const std::vector<std::uint16_t> values(16, 1000);
            base.writeRegion({0, 0, 0}, {2, 2, 4}, reinterpret_cast<const std::uint8_t*>(values.data()));
        } else {
            const std::vector<float> values(16, 0.5f);
            base.writeRegion({0, 0, 0}, {2, 2, 4}, reinterpret_cast<const std::uint8_t*>(values.data()));
        }
        vcr::zarr::writePyramidLevel(other, 1, cubes);
        auto reduced = vcr::zarr::Writer::open(other / "1");
        require(reduced.chunkExists({0, 0, 0}), "a chunk with data is stored");
        const auto bytes = read(reduced.reader(), {0, 0, 0}, {1, 1, 1}, dtype == "<u2" ? 2 : 4);
        if (dtype == "<u2") {
            std::uint16_t value;
            std::memcpy(&value, bytes.data(), 2);
            require(value == 1000, "uint16 means");
        } else {
            float value;
            std::memcpy(&value, bytes.data(), 4);
            require(value == 0.5f, "float32 means");
        }
    }
    {
        auto reduced = vcr::zarr::Writer::open(root / "pyramidu2" / "1");
        require(!reduced.chunkExists({1, 1, 0}) && !reduced.chunkExists({0, 1, 0}), "chunks of fill stay absent");
    }

    // Multiscales attributes: the reader places the levels where the means sit.
    vcr::zarr::MultiscalesOptions ome;
    ome.levels = 2;
    ome.factors = {1, 2, 2};
    ome.scale = {3.0, 0.5, 0.5};
    ome.unit = "micrometer";
    vcr::zarr::writeAttributes(dir, {{"multiscales", Json::array({vcr::zarr::omeMultiscales(ome)})}});
    auto group = vcr::zarr::Group::open(dir.string(), {});
    require(group.levels() == std::vector<std::string>({"0", "1", "2"}), "multiscales list the levels");
    const auto placement = group.placement("2");
    require(placement.scale[0] == 1 && placement.scale[1] == 4 && placement.offset[0] == 0 && placement.offset[1] == 1.5,
            "a level 2 voxel covers 4 voxels in Y/X, centred on them, and keeps Z");
    require(group.multiscales().at(0).at("axes").at(1).at("unit") == "micrometer", "axes carry the unit");
}

}  // namespace

static void generalOptions(const fs::path& root)
{
    using namespace vcr::zarr;
    CreateOptions options;
    options.shape = {4, 4};
    options.chunks = {2, 2};
    options.cname = "zstd";
    options.shuffle = 2;
    auto writer = Writer::create(root / "general" / "a", options);
    if (writer.meta().compressor.value("cname", "") != "zstd" || writer.meta().compressor.value("shuffle", 0) != 2)
        throw std::runtime_error("blosc options are kept");
    const std::vector<std::uint8_t> zeros(4, 0), ones(4, 1);
    writer.writeRegion({0, 0}, {2, 2}, ones.data());
    writer.writeRegion({2, 2}, {2, 2}, zeros.data(), true);
    if (!writer.chunkExists({0, 0}) || writer.chunkExists({1, 1}))
        throw std::runtime_error("skipEmpty does not store an all-fill chunk");
    options.overwrite = true;
    auto replaced = Writer::create(root / "general" / "a", options);
    if (replaced.chunkExists({0, 0}))
        throw std::runtime_error("overwrite removes stored chunks");
    createGroup(root / "general" / "g", Json{{"k", 1}});
    createGroup(root / "general" / "g");
    if (readAttributes(root / "general" / "g").value("k", 0) != 1)
        throw std::runtime_error("an existing group keeps its attributes");
    if (nodeChildren(root / "general") != std::vector<std::string>{"a", "g"})
        throw std::runtime_error("node children");
    if (!removeNode(root / "general" / "g") || removeNode(root / "general" / "g"))
        throw std::runtime_error("remove a node once");
}

// AWS's published example of signing a PUT ("Example: PUT Object", S3 API reference).
void signsAwsPutExample()
{
    const vcr::S3Key key{"AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "", "us-east-1"};
    const auto headers = vcr::sigv4_headers(
        "PUT", "https://examplebucket.s3.amazonaws.com/test%24file.text", key, "20130524T000000Z",
        {{"date", "Fri, 24 May 2013 00:00:00 GMT"}, {"x-amz-storage-class", "REDUCED_REDUNDANCY"}},
        vcr::to_hex(vcr::Sha256::of("Welcome to Amazon S3.")));
    require(headers.back().first == "authorization" &&
                headers.back().second.ends_with("Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"),
            "AWS PUT signature: " + headers.back().second);
}

int main()
{
    const auto root = fs::temp_directory_path() /
                      ("vc-reader-writer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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
        generalOptions(root);
        for (int version : {2, 3}) {
            const auto dir = root / std::to_string(version);
            codecs(dir, version);
            editing(dir, version);
            mutableChunks(dir, version);
        }
        signsAwsPutExample();
        v3Metadata(root);
        codecs(root / "sharded-codecs", 3, true);
        mutableChunks(root / "sharded-mutable", 3, true);
        shardEditing(root);
        concurrentShards(root);
        pyramids(root);
        std::cout << "every codec round-trips; skip-empty, regions, attributes and validation passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
