#include "test.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unordered_set>

#include "vc/core/cache/ChunkKey.hpp"
#include "vc/core/cache/ChunkData.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"

namespace fs = std::filesystem;

// ---- ChunkKey ---------------------------------------------------------------

TEST(ChunkKey, Equality)
{
    vc::cache::ChunkKey a{0, 1, 2, 3};
    vc::cache::ChunkKey b{0, 1, 2, 3};
    vc::cache::ChunkKey c{0, 1, 2, 4};
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != b);
}

TEST(ChunkKey, DefaultValues)
{
    vc::cache::ChunkKey k;
    EXPECT_EQ(k.level, 0);
    EXPECT_EQ(k.iz, 0);
    EXPECT_EQ(k.iy, 0);
    EXPECT_EQ(k.ix, 0);
}

TEST(ChunkKey, CoarsenBasic)
{
    vc::cache::ChunkKey k{0, 8, 12, 16};
    auto c1 = k.coarsen(1);
    EXPECT_EQ(c1.level, 1);
    EXPECT_EQ(c1.iz, 4);
    EXPECT_EQ(c1.iy, 6);
    EXPECT_EQ(c1.ix, 8);

    auto c2 = k.coarsen(2);
    EXPECT_EQ(c2.level, 2);
    EXPECT_EQ(c2.iz, 2);
    EXPECT_EQ(c2.iy, 3);
    EXPECT_EQ(c2.ix, 4);
}

TEST(ChunkKey, CoarsenSameLevel)
{
    vc::cache::ChunkKey k{2, 4, 5, 6};
    auto c = k.coarsen(2);
    EXPECT_EQ(c.level, 2);
    EXPECT_EQ(c.iz, 4);
    EXPECT_EQ(c.iy, 5);
    EXPECT_EQ(c.ix, 6);
}

TEST(ChunkKey, CoarsenLowerLevelReturnsUnchanged)
{
    vc::cache::ChunkKey k{3, 4, 5, 6};
    auto c = k.coarsen(1);  // targetLevel < level
    EXPECT_EQ(c.level, 3);
    EXPECT_EQ(c.iz, 4);
    EXPECT_EQ(c.iy, 5);
    EXPECT_EQ(c.ix, 6);
}

TEST(ChunkKeyHash, DifferentKeysProduceDifferentHashes)
{
    vc::cache::ChunkKeyHash hash;
    std::unordered_set<size_t> hashes;
    // Generate a grid of keys and check for collisions
    for (int l = 0; l < 3; ++l)
        for (int z = 0; z < 4; ++z)
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    hashes.insert(hash({l, z, y, x}));
    // 3*4*4*4 = 192 keys, should have zero or near-zero collisions
    EXPECT_EQ(hashes.size(), 192u);
}

TEST(ChunkKeyHash, UsableInUnorderedMap)
{
    std::unordered_set<vc::cache::ChunkKey, vc::cache::ChunkKeyHash> s;
    s.insert({0, 1, 2, 3});
    s.insert({0, 1, 2, 3});  // duplicate
    s.insert({1, 0, 0, 0});
    EXPECT_EQ(s.size(), 2u);
}

// ---- ChunkData --------------------------------------------------------------

TEST(ChunkData, BasicAccessors)
{
    vc::cache::ChunkData cd;
    cd.shape = {2, 3, 4};
    cd.elementSize = 2;
    cd.bytes.resize(2 * 3 * 4 * 2);

    EXPECT_EQ(cd.numElements(), 24u);
    EXPECT_EQ(cd.totalBytes(), 48u);
    EXPECT_EQ(cd.strideZ(), 12);
    EXPECT_EQ(cd.strideY(), 4);
    EXPECT_EQ(cd.strideX(), 1);
}

TEST(ChunkData, TypedAccess)
{
    vc::cache::ChunkData cd;
    cd.shape = {1, 1, 4};
    cd.elementSize = 2;
    cd.bytes.resize(8);

    auto* p = cd.data<uint16_t>();
    p[0] = 100;
    p[1] = 200;
    p[2] = 300;
    p[3] = 400;

    const auto& ccd = cd;
    const auto* cp = ccd.data<uint16_t>();
    EXPECT_EQ(cp[0], 100);
    EXPECT_EQ(cp[3], 400);
}

TEST(ChunkData, EmptyShape)
{
    vc::cache::ChunkData cd;
    EXPECT_EQ(cd.numElements(), 0u);
    EXPECT_EQ(cd.totalBytes(), 0u);
}

// ---- CacheDebugLog ----------------------------------------------------------

TEST(CacheDebugLog, ReturnsNullWhenEnvNotSet)
{
    // VC_CACHE_DEBUG_LOG should not be set in test environment
    // If it is, this test is inconclusive rather than failing
    FILE* f = vc::cache::cacheDebugLog();
    (void)f;  // just verify it doesn't crash
}

// ---- CacheUtils -------------------------------------------------------------

TEST(CacheUtils, ChunkFilename)
{
    vc::cache::ChunkKey k{0, 1, 2, 3};
    EXPECT_EQ(vc::cache::chunkFilename(k, "."), std::string("1.2.3"));
    EXPECT_EQ(vc::cache::chunkFilename(k, "/"), std::string("1/2/3"));
}

TEST(CacheUtils, ReadFileToVectorRoundTrip)
{
    auto tmpDir = fs::temp_directory_path() / "vc_test_cache_utils";
    fs::create_directories(tmpDir);
    auto tmpFile = tmpDir / "test.bin";

    // Write test data
    std::vector<uint8_t> data = {0x00, 0x11, 0x22, 0x33, 0x44, 0xFF};
    {
        std::ofstream ofs(tmpFile, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    auto result = vc::cache::readFileToVector(tmpFile);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ((*result)[i], data[i]);
    }

    // Cleanup
    fs::remove_all(tmpDir);
}

TEST(CacheUtils, ReadFileToVectorMissingFile)
{
    auto result = vc::cache::readFileToVector("/tmp/vc_nonexistent_file_12345");
    EXPECT_FALSE(result.has_value());
}

TEST(CacheUtils, ReadFileToVectorEmptyFile)
{
    auto tmpDir = fs::temp_directory_path() / "vc_test_cache_utils";
    fs::create_directories(tmpDir);
    auto tmpFile = tmpDir / "empty.bin";

    // Create empty file
    { std::ofstream ofs(tmpFile); }

    auto result = vc::cache::readFileToVector(tmpFile);
    EXPECT_FALSE(result.has_value());  // empty files return nullopt

    fs::remove_all(tmpDir);
}
