// Local zarr volume creation + chunk I/O. Exercises Volume / VcDataset /
// Zarr.cpp paths that the live S3 test never hits (writes, downsample,
// fill-value, removeChunk, readChunkOrFill).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vol_local_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

Volume::ZarrCreateOptions makeOpts(std::array<size_t, 3> shape = {64, 64, 64},
                                   std::array<size_t, 3> chunks = {32, 32, 32},
                                   size_t numLevels = 2)
{
    Volume::ZarrCreateOptions opts;
    opts.shapeZYX = shape;
    opts.chunkShapeZYX = chunks;
    opts.numLevels = numLevels;
    opts.fillValue = 0.0;
    opts.voxelSize = 1.0;
    opts.uuid = "test-vol-uuid";
    opts.name = "test-vol";
    opts.overwriteExisting = true;
    // The build-bot blosc support is optional; force uncompressed so the
    // test runs regardless of whether libblosc was linked.
    opts.compressor = "none";
    return opts;
}

} // namespace

TEST_CASE("Volume::New(path, opts): creates a fresh local zarr pyramid")
{
    auto d = tmpDir("create");
    auto opts = makeOpts();
    auto v = Volume::New(d, opts);
    REQUIRE(v);
    CHECK_FALSE(v->isRemote());
    CHECK(v->id() == "test-vol-uuid");
    CHECK(v->numScales() == 2);

    auto s0 = v->levelShape(0);
    CHECK(s0[0] == 64);
    CHECK(s0[1] == 64);
    CHECK(s0[2] == 64);

    auto s1 = v->levelShape(1);
    CHECK(s1[0] == 32);
    CHECK(s1[1] == 32);
    CHECK(s1[2] == 32);

    fs::remove_all(d);
}

TEST_CASE("Volume: reopen existing local zarr without options")
{
    auto d = tmpDir("reopen");
    auto opts = makeOpts();
    {
        auto v = Volume::New(d, opts);
        REQUIRE(v);
    }
    // Second open without options finds the existing pyramid on disk.
    auto v = Volume::New(d);
    REQUIRE(v);
    CHECK_FALSE(v->isRemote());
    CHECK(v->numScales() == 2);
    fs::remove_all(d);
}

TEST_CASE("Volume::checkDir: true for created volume, false for arbitrary")
{
    auto d = tmpDir("checkdir");
    {
        auto v = Volume::New(d, makeOpts());
        REQUIRE(v);
    }
    CHECK(Volume::checkDir(d));
    CHECK_FALSE(Volume::checkDir("/__nonexistent__/x"));
    fs::remove_all(d);
}

TEST_CASE("Volume: chunkByteSize and chunkExists on a fresh volume")
{
    auto d = tmpDir("chunkexist");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    // No chunks written yet — none should exist.
    CHECK_FALSE(v->chunkExists(0, {0, 0, 0}));
    // chunkByteSize = chunk_volume * dtype_size = 32*32*32 * 1 (uint8) = 32768
    CHECK(v->chunkByteSize(0) == 32768);
    fs::remove_all(d);
}

TEST_CASE("Volume: writeZYX then readZYX round-trips a small buffer")
{
    auto d = tmpDir("writeread");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);

    // Write a small region of known values.
    Array3D<uint8_t> in({16, 16, 16}, /*fill=*/0);
    // Set a deterministic gradient.
    for (size_t z = 0; z < 16; ++z)
        for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x)
                in(z, y, x) = static_cast<uint8_t>((z * 31 + y * 13 + x) & 0xFF);

    v->writeZYX(in, {0, 0, 0}, /*level=*/0);

    Array3D<uint8_t> out({16, 16, 16});
    bool ok = v->readZYX(out, {0, 0, 0}, 0);
    CHECK(ok);
    // Spot-check a few values.
    CHECK(int(out(0, 0, 0)) == int(in(0, 0, 0)));
    CHECK(int(out(15, 15, 15)) == int(in(15, 15, 15)));
    CHECK(int(out(7, 3, 11)) == int(in(7, 3, 11)));
    fs::remove_all(d);
}

TEST_CASE("Volume: readChunkOrFill returns fill when chunk is absent")
{
    auto d = tmpDir("orfill");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    auto bytes = v->readChunkOrFill(0, {0, 0, 0});
    CHECK(bytes.size() == v->chunkByteSize(0));
    // Fill value = 0; every byte should be 0.
    for (auto b : bytes) CHECK(std::to_integer<int>(b) == 0);
    fs::remove_all(d);
}

TEST_CASE("Volume: chunkExists turns true after writeChunk")
{
    auto d = tmpDir("wc");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0));
    // Fill with something non-zero so we don't get write_empty_chunks=false elided.
    std::memset(chunk.data(), 0xAB, chunk.size());
    v->writeChunk(0, {0, 0, 0}, chunk);
    CHECK(v->chunkExists(0, {0, 0, 0}));
    auto opt = v->readChunk(0, {0, 0, 0});
    REQUIRE(opt.has_value());
    CHECK(opt->size() == chunk.size());
    CHECK(std::to_integer<int>((*opt)[0]) == 0xAB);
    fs::remove_all(d);
}

TEST_CASE("Volume: removeChunk drops the chunk file")
{
    auto d = tmpDir("rm");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0));
    std::memset(chunk.data(), 0x55, chunk.size());
    v->writeChunk(0, {0, 0, 0}, chunk);
    CHECK(v->chunkExists(0, {0, 0, 0}));
    CHECK(v->removeChunk(0, {0, 0, 0}));
    CHECK_FALSE(v->chunkExists(0, {0, 0, 0}));
    // Second remove is a no-op (chunk already gone).
    CHECK_FALSE(v->removeChunk(0, {0, 0, 0}));
    fs::remove_all(d);
}

TEST_CASE("Volume: empty-chunk write with writeEmptyChunks=false is elided")
{
    auto d = tmpDir("empty");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    std::vector<std::byte> zero(v->chunkByteSize(0));
    // all zero == fill value
    Volume::ChunkWriteOptions o;
    o.writeEmptyChunks = false;
    v->writeChunk(0, {0, 0, 0}, zero, o);
    CHECK_FALSE(v->chunkExists(0, {0, 0, 0}));
    fs::remove_all(d);
}

TEST_CASE("Volume: writeXYZ and readXYZ obey UI-order offsets")
{
    auto d = tmpDir("xyz");
    auto v = Volume::New(d, makeOpts());
    REQUIRE(v);
    Array3D<uint8_t> in({4, 4, 4}, /*fill=*/77);
    v->writeXYZ(in, /*offsetXYZ=*/{0, 0, 0}, 0);
    Array3D<uint8_t> out({4, 4, 4});
    bool ok = v->readXYZ(out, {0, 0, 0}, 0);
    CHECK(ok);
    CHECK(int(out(0, 0, 0)) == 77);
    fs::remove_all(d);
}

TEST_CASE("Volume: shape() / sliceWidth() / sliceHeight() / numSlices() agree")
{
    auto d = tmpDir("shape");
    auto v = Volume::New(d, makeOpts({100, 50, 200}, {25, 25, 25}, 1));
    REQUIRE(v);
    auto sh = v->shape();
    CHECK(sh[0] == v->numSlices());
    CHECK(sh[1] == v->sliceHeight());
    CHECK(sh[2] == v->sliceWidth());
    fs::remove_all(d);
}
