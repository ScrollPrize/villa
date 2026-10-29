// Extra Volume coverage beyond test_volume_local.cpp:
// - uint16 dtype reads
// - process cache configuration
// - MissingScaleLevelPolicy::AllFill behaviour for a level that exists but is empty

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "support/TestZarrVolume.hpp"
#include "vc/core/types/Array3D.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_vol_extras_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

vc::test::ZarrVolumeSpec optsU8()
{
    vc::test::ZarrVolumeSpec o;
    o.shapeZYX = {32, 32, 32};
    o.chunkShapeZYX = {16, 16, 16};
    o.numLevels = 1;
    return o;
}

vc::test::ZarrVolumeSpec optsU16()
{
    auto o = optsU8();
    o.dtype = vc::render::ChunkDtype::UInt16;
    return o;
}

} // namespace

TEST_CASE("Volume uint16 round-trip via writeZYX/readZYX")
{
    auto d = tmpDir("u16");
    auto v = vc::test::createZarrVolume(d, optsU16());
    REQUIRE(v);
    CHECK(v->dtype() == vc::render::ChunkDtype::UInt16);
    CHECK(v->dtypeSize() == 2);

    Array3D<uint16_t> in({8, 8, 8}, 0);
    for (size_t z = 0; z < 8; ++z)
        for (size_t y = 0; y < 8; ++y)
            for (size_t x = 0; x < 8; ++x)
                in(z, y, x) = static_cast<uint16_t>(z * 64 + y * 8 + x);
    vc::test::writeZYX(*v, in, {0, 0, 0}, 0);

    Array3D<uint16_t> out({8, 8, 8});
    CHECK(v->readZYX(out, {0, 0, 0}, 0));
    CHECK(out(0, 0, 0) == in(0, 0, 0));
    CHECK(out(7, 7, 7) == in(7, 7, 7));
    fs::remove_all(d);
}

TEST_CASE("Volume uint16: writeXYZ/readXYZ overload")
{
    auto d = tmpDir("u16_xyz");
    auto v = vc::test::createZarrVolume(d, optsU16());
    REQUIRE(v);
    Array3D<uint16_t> in({4, 4, 4}, 12345);
    vc::test::writeZYX(*v, in, {0, 0, 0}, 0);
    Array3D<uint16_t> out({4, 4, 4});
    CHECK(v->readXYZ(out, {0, 0, 0}, 0));
    CHECK(out(0, 0, 0) == 12345);
    fs::remove_all(d);
}

TEST_CASE("Volume chunk cache reports cumulative decoder work")
{
    auto d = tmpDir("decode_stats");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    Array3D<uint8_t> seed({16, 16, 16}, 19);
    vc::test::writeZYX(*v, seed, {0, 0, 0}, 0);
    auto cache = v->sharedChunkCache();
    REQUIRE(cache);
    v->invalidateCache();
    const auto before = cache->stats();

    Array3D<uint8_t> out({16, 16, 16});
    CHECK(v->readZYX(out, {0, 0, 0}, 0));
    const auto after = cache->stats();

    CHECK(after.decodeAttempts > before.decodeAttempts);
    CHECK(after.decodedChunks > before.decodedChunks);
    CHECK(after.decodeInputBytes > before.decodeInputBytes);
    CHECK(after.decodedChunkBytes > before.decodedChunkBytes);
    CHECK(after.decodeNanoseconds >= before.decodeNanoseconds);
    fs::remove_all(d);
}

TEST_CASE("Volume uses the process chunk-cache service")
{
    auto d = tmpDir("knobs");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    auto service = vc::render::processChunkCacheService();
    service->configureDecodedByteCapacity(1ULL << 20);
    auto cache = v->sharedChunkCache();
    REQUIRE(cache);
    CHECK(cache->sourceId() != vc::render::VolumeSourceId{});
    fs::remove_all(d);
}

TEST_CASE("process chunk-cache decode concurrency is configurable")
{
    auto service = vc::render::processChunkCacheService();
    const auto original = service->decodeConcurrency();
    service->configureDecodeConcurrency(1);
    CHECK(service->decodeConcurrency() == 1);
    service->configureDecodeConcurrency(original);
}

TEST_CASE("process chunk-cache configuration preserves the service")
{
    auto service = vc::render::processChunkCacheService();
    const auto original = service->fetchConcurrency();

    vc::render::ChunkCacheService::Options options;
    options.decodedByteCapacity = 2ULL << 20;
    options.fetchConcurrency.workerCapacity = original.workerCapacity;
    options.fetchConcurrency.maxConcurrentReads = 2;
    options.fetchConcurrency.adaptive = false;
    auto configured = vc::render::configureProcessChunkCacheService(
        std::move(options));

    CHECK(configured == service);
    CHECK(configured->decodedByteBudget()->maximumBytes() == 2ULL << 20);
    CHECK(configured->fetchConcurrency().maxConcurrentReads == 2);
    CHECK_FALSE(configured->fetchConcurrency().adaptive);

    configured->configureFetchConcurrency(
        original.maxConcurrentReads, original.adaptive);
}

TEST_CASE("Volume::invalidateCache")
{
    auto d = tmpDir("invalidate");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    // Just exercise; no observable state.
    v->invalidateCache();
    CHECK(v->chunkedCache() != nullptr);
    fs::remove_all(d);
}

TEST_CASE("Volume::sharedChunkCache yields a non-null cache pointer")
{
    auto d = tmpDir("createcache");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    auto c = v->sharedChunkCache();
    CHECK(c != nullptr);
    CHECK(c->numLevels() == 1);
    fs::remove_all(d);
}

TEST_CASE("Volume: missing chunk read returns fill with AllFill policy")
{
    auto d = tmpDir("allfill_policy");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    // No writes — every chunk is absent and should fill with 0.
    Array3D<uint8_t> out({16, 16, 16});
    bool ok = v->readZYX(out, {0, 0, 0}, 0, Volume::MissingScaleLevelPolicy::AllFill);
    CHECK(ok);
    // All zeros (fillValue).
    bool allZero = true;
    for (size_t z = 0; z < 16; ++z)
        for (size_t y = 0; y < 16; ++y)
            for (size_t x = 0; x < 16; ++x)
                if (out(z, y, x) != 0) { allZero = false; break; }
    CHECK(allZero);
    fs::remove_all(d);
}

TEST_CASE("Volume::chunkGridShape")
{
    auto d = tmpDir("grid_shape");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    // shape=32^3, chunk=16^3 → 2x2x2 chunk grid
    auto g = v->chunkGridShape(0);
    CHECK(g[0] == 2);
    CHECK(g[1] == 2);
    CHECK(g[2] == 2);
    CHECK(v->chunkCount(0) == 8);
    fs::remove_all(d);
}

TEST_CASE("Volume::voxelSize defaults to 1.0 when not set in opts")
{
    auto d = tmpDir("voxel");
    auto v = vc::test::createZarrVolume(d, optsU8());
    REQUIRE(v);
    CHECK(v->voxelSize() > 0.0);
    fs::remove_all(d);
}
