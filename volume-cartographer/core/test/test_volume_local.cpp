// Local zarr volume reads and the chunk cache. Exercises Volume / VcDataset /
// Zarr.cpp paths that the live S3 test never hits (fill values,
// readChunkOrFill, cache retention and reconfiguration).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/util/LoadJson.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
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

vc::test::ZarrVolumeSpec makeOpts(std::array<size_t, 3> shape = {64, 64, 64},
                                   std::array<size_t, 3> chunks = {32, 32, 32},
                                   size_t numLevels = 2)
{
    vc::test::ZarrVolumeSpec opts;
    opts.shapeZYX = shape;
    opts.chunkShapeZYX = chunks;
    opts.numLevels = numLevels;
    opts.fillValue = 0.0;
    opts.voxelSize = 1.0;
    opts.uuid = "test-vol-uuid";
    opts.name = "test-vol";
    return opts;
}

void writeTextFile(const fs::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << text;
}

} // namespace

TEST_CASE("Volume: reopen existing local zarr without options")
{
    auto d = tmpDir("reopen");
    auto opts = makeOpts();
    {
        auto v = vc::test::createZarrVolume(d, opts);
        REQUIRE(v);
    }
    // Second open without options finds the existing pyramid on disk.
    auto v = Volume::New(d);
    REQUIRE(v);
    CHECK_FALSE(v->isRemote());
    CHECK(v->numScales() == 2);
    fs::remove_all(d);
}

TEST_CASE("Volume: chunkByteSize and chunkExists on a fresh volume")
{
    auto d = tmpDir("chunkexist");
    auto v = vc::test::createZarrVolume(d, makeOpts());
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
    auto v = vc::test::createZarrVolume(d, makeOpts());
    REQUIRE(v);

    // Write a small region of known values.
    Array3D<uint8_t> in({16, 16, 16}, /*fill=*/0);
    // Set a deterministic gradient.
    for (size_t z = 0; z < 16; ++z)
        for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x)
                in(z, y, x) = static_cast<uint8_t>((z * 31 + y * 13 + x) & 0xFF);

    vc::test::writeZYX(*v, in, {0, 0, 0}, /*level=*/0);

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
    auto v = vc::test::createZarrVolume(d, makeOpts());
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
    auto v = vc::test::createZarrVolume(d, makeOpts());
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0));
    // Fill with something non-zero so we don't get write_empty_chunks=false elided.
    std::memset(chunk.data(), 0xAB, chunk.size());
    vc::test::writeChunk(*v, 0, {0, 0, 0}, chunk);
    CHECK(v->chunkExists(0, {0, 0, 0}));
    auto opt = v->readChunk(0, {0, 0, 0});
    REQUIRE(opt.has_value());
    CHECK(opt->size() == chunk.size());
    CHECK(std::to_integer<int>((*opt)[0]) == 0xAB);
    fs::remove_all(d);
}

TEST_CASE("Volume: regular cache retains decoded chunks")
{
    auto d = tmpDir("cache-client");
    auto v = vc::test::createZarrVolume(d, makeOpts({32, 32, 32}, {32, 32, 32}, 1));
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0), std::byte{0x5a});
    vc::test::writeChunk(*v, 0, {0, 0, 0}, chunk);

    auto service = vc::render::processChunkCacheService();
    service->configureDecodedByteCapacity(1);
    service->configureDecodedByteCapacity(64 * 1024);
    auto cache = v->sharedChunkCache();
    REQUIRE(cache);
    CHECK(v->chunkedCache() == cache.get());
    CHECK(cache->getChunkBlocking(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);
    CHECK(service->decodedByteBudget()->stats().decodedBytes == chunk.size());
    CHECK(cache->stats().decodedBytes == chunk.size());

    std::weak_ptr<vc::render::ChunkCache> weakCache = cache;
    cache.reset();
    CHECK_FALSE(weakCache.expired());
    CHECK(v->sharedChunkCache() == weakCache.lock());
    fs::remove_all(d);
}

TEST_CASE("Volume: process service keeps decoded chunks across handles")
{
    auto d = tmpDir("cache-service-client");
    auto firstVolume = vc::test::createZarrVolume(
        d, makeOpts({32, 32, 32}, {32, 32, 32}, 1));
    REQUIRE(firstVolume);
    std::vector<std::byte> chunk(
        firstVolume->chunkByteSize(0), std::byte{0x3c});
    vc::test::writeChunk(*firstVolume, 0, {0, 0, 0}, chunk);

    auto service = vc::render::processChunkCacheService();
    service->configureDecodedByteCapacity(1);
    service->configureDecodedByteCapacity(64 * 1024);
    const auto sourcesBefore = service->sourceCount();
    auto first = firstVolume->sharedChunkCache();
    REQUIRE(first->getChunkBlocking(0, 0, 0, 0).status ==
            vc::render::ChunkStatus::Data);
    const auto sourceId = first->sourceId();
    first.reset();
    firstVolume.reset();

    CHECK(service->decodedByteBudget()->stats().decodedBytes == chunk.size());
    auto secondVolume = Volume::New(d);
    REQUIRE(secondVolume);
    auto reacquired = secondVolume->sharedChunkCache();
    CHECK(reacquired->sourceId() == sourceId);
    CHECK(service->sourceCount() == sourcesBefore + 1);
    CHECK(reacquired->getChunkIfCached(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);
    fs::remove_all(d);
}

TEST_CASE("Volume: service concurrency reconfigures without eviction")
{
    auto d = tmpDir("cache-service-threads");
    auto v = vc::test::createZarrVolume(d, makeOpts({32, 32, 32}, {32, 32, 32}, 1));
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0), std::byte{0x4d});
    vc::test::writeChunk(*v, 0, {0, 0, 0}, chunk);

    auto service = vc::render::processChunkCacheService();
    service->configureFetchConcurrency(2, false);

    auto cache = v->sharedChunkCache();
    REQUIRE(cache);
    auto loaded = cache->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(loaded.status == vc::render::ChunkStatus::Data);
    REQUIRE(loaded.bytes);
    service->configureFetchConcurrency(3, false);
    const auto concurrency = service->fetchConcurrency();
    CHECK(concurrency.maxConcurrentReads == 3);
    CHECK_FALSE(concurrency.adaptive);
    CHECK(v->sharedChunkCache() == cache);
    const auto warm = cache->getChunkIfCached(0, 0, 0, 0);
    REQUIRE(warm.status == vc::render::ChunkStatus::Data);
    REQUIRE(warm.bytes);
    CHECK(warm.bytes->front() == std::byte{0x4d});

    fs::remove_all(d);
}

TEST_CASE("Volume: cache budget reconfigures without replacing source state")
{
    auto d = tmpDir("cache-service-budget");
    auto v = vc::test::createZarrVolume(d, makeOpts({32, 32, 32}, {32, 32, 32}, 1));
    REQUIRE(v);
    std::vector<std::byte> chunk(v->chunkByteSize(0), std::byte{0x5a});
    vc::test::writeChunk(*v, 0, {0, 0, 0}, chunk);

    auto service = vc::render::processChunkCacheService();
    service->configureDecodedByteCapacity(1);
    service->configureDecodedByteCapacity(64 * 1024);
    const auto sourcesBefore = service->sourceCount();
    auto cache = v->sharedChunkCache();
    REQUIRE(cache);
    const auto source = cache->sourceId();
    REQUIRE(cache->getChunkBlocking(0, 0, 0, 0).status ==
            vc::render::ChunkStatus::Data);

    service->configureDecodedByteCapacity(128 * 1024);
    CHECK(v->sharedChunkCache() == cache);
    CHECK(cache->sourceId() == source);
    CHECK(service->sourceCount() == sourcesBefore + 1);
    CHECK(service->decodedByteBudget()->maximumBytes() == 128 * 1024);
    CHECK(cache->getChunkIfCached(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::Data);

    service->configureDecodedByteCapacity(16 * 1024);
    CHECK(v->sharedChunkCache() == cache);
    CHECK(cache->sourceId() == source);
    CHECK(service->sourceCount() == sourcesBefore + 1);
    CHECK(service->decodedByteBudget()->maximumBytes() == 16 * 1024);
    CHECK(service->decodedByteBudget()->stats().decodedBytes == 0);
    CHECK(cache->getChunkIfCached(0, 0, 0, 0).status ==
          vc::render::ChunkStatus::MissQueued);
    fs::remove_all(d);
}

TEST_CASE("Volume: writeXYZ and readXYZ obey UI-order offsets")
{
    auto d = tmpDir("xyz");
    auto v = vc::test::createZarrVolume(d, makeOpts());
    REQUIRE(v);
    Array3D<uint8_t> in({4, 4, 4}, /*fill=*/77);
    vc::test::writeZYX(*v, in, /*offsetXYZ=*/{0, 0, 0}, 0);
    Array3D<uint8_t> out({4, 4, 4});
    bool ok = v->readXYZ(out, {0, 0, 0}, 0);
    CHECK(ok);
    CHECK(int(out(0, 0, 0)) == 77);
    fs::remove_all(d);
}

TEST_CASE("Volume: shape() / sliceWidth() / sliceHeight() / numSlices() agree")
{
    auto d = tmpDir("shape");
    auto v = vc::test::createZarrVolume(d, makeOpts({100, 50, 200}, {25, 25, 25}, 1));
    REQUIRE(v);
    auto sh = v->shape();
    CHECK(sh[0] == v->numSlices());
    CHECK(sh[1] == v->sliceHeight());
    CHECK(sh[2] == v->sliceWidth());
    fs::remove_all(d);
}

TEST_CASE("Volume: an unusable voxelsize field reports 0 instead of throwing")
{
    // voxelSize() promises 0 for "unknown". A document may state the key as
    // null, as text, or not at all, and each of those used to make the
    // accessor throw from deep inside whatever asked for a physical area.
    auto d = tmpDir("unusable_voxelsize");
    auto opts = makeOpts({32, 32, 32}, {16, 16, 16}, 1);
    vc::test::createZarrVolume(d, opts);
    const auto meta = vc::json::load_json_file(d / "meta.json");
    const auto metaWithVoxelSize = [&](const std::string& voxelsizeJson) {
        std::string text = R"({"type": "vol", "uuid": ")" + meta["uuid"].get_string() +
                           R"(", "width": 32, "height": 32, "slices": 32)";
        if (!voxelsizeJson.empty()) {
            text += R"(, "voxelsize": )" + voxelsizeJson;
        }
        return text + "}";
    };

    SUBCASE("meta.json without a voxelsize key")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize(""));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with a null voxelsize")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize("null"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with a non-positive voxelsize")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize("-1"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    SUBCASE("meta.json with voxelsize as unknown text")
    {
        writeTextFile(d / "meta.json", metaWithVoxelSize(R"("unknown")"));
        auto v = Volume::New(d);
        CHECK(v->voxelSize() == 0.0);
    }

    fs::remove_all(d);
}
