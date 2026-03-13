#include "test.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <vector>

#include "vc/core/util/SparseChunkSpool.hpp"

namespace {

namespace fs = std::filesystem;

class ScopedTempDir {
public:
    ScopedTempDir()
    {
        static std::atomic<unsigned long long> counter{0};
        const auto suffix = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count())
            + "_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
        path_ = fs::temp_directory_path() / ("vc_test_sparse_spool_" + suffix);
        fs::create_directories(path_);
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

size_t linearIndex(const vc::core::util::Shape3& shape, size_t z, size_t y, size_t x)
{
    return (z * shape[1] + y) * shape[2] + x;
}

} // namespace

TEST(SparseChunkSpool, TracksTouchedChunksAndMaterializesDenseReference)
{
    ScopedTempDir tempDir;
    const vc::core::util::Shape3 chunkShape{4, 4, 4};
    const vc::core::util::Shape3 volumeShape{9, 9, 9};

    vc::core::util::SparseChunkSpool spool(tempDir.path(), chunkShape, volumeShape, /*inMemoryMaxBytes=*/32);
    vc::core::util::SparseChunkSpoolBuffer thread0(spool);
    vc::core::util::SparseChunkSpoolBuffer thread1(spool);

    thread0.emit(0, 0, 0, {1, 2, 3, 4, 5, 6, 7});
    thread0.emit(3, 3, 3, {11, 12, 13, 14, 15, 16, 17});
    thread1.emit(4, 4, 4, {21, 22, 23, 24, 25, 26, 27});
    thread1.emit(8, 8, 8, {31, 32, 33, 34, 35, 36, 37});
    thread0.flushAll();
    thread1.flushAll();

    const auto touched = spool.touchedChunks();
    const vc::core::util::SparseChunkIndex chunk000{0, 0, 0};
    const vc::core::util::SparseChunkIndex chunk111{1, 1, 1};
    const vc::core::util::SparseChunkIndex chunk222{2, 2, 2};
    EXPECT_EQ(touched.size(), static_cast<size_t>(3));
    EXPECT_TRUE(touched[0] == chunk000);
    EXPECT_TRUE(touched[1] == chunk111);
    EXPECT_TRUE(touched[2] == chunk222);

    const auto stats = spool.stats();
    EXPECT_EQ(stats.appendedRecords, static_cast<uint64_t>(4));
    EXPECT_GT(stats.spillFiles, 0u);

    std::array<std::vector<uint8_t>, 7> dense;
    for (int c = 0; c < 7; ++c) {
        dense[c].assign(volumeShape[0] * volumeShape[1] * volumeShape[2], (c < 3) ? 128 : 0);
    }
    const std::array<std::array<uint8_t, 7>, 4> expectedValues{{
        {1, 2, 3, 4, 5, 6, 7},
        {11, 12, 13, 14, 15, 16, 17},
        {21, 22, 23, 24, 25, 26, 27},
        {31, 32, 33, 34, 35, 36, 37},
    }};
    const std::array<std::array<size_t, 3>, 4> expectedCoords{{
        {0, 0, 0},
        {3, 3, 3},
        {4, 4, 4},
        {8, 8, 8},
    }};
    for (size_t i = 0; i < expectedCoords.size(); ++i) {
        const auto& coord = expectedCoords[i];
        const auto lin = linearIndex(volumeShape, coord[0], coord[1], coord[2]);
        for (int c = 0; c < 7; ++c) {
            dense[c][lin] = expectedValues[i][c];
        }
    }

    for (const auto& chunk : touched) {
        std::vector<vc::core::util::SparseChunkRecordU8x7> records;
        EXPECT_TRUE(spool.readChunkRecords(chunk, records));

        std::array<std::vector<uint8_t>, 7> chunkBufs;
        for (int c = 0; c < 7; ++c) {
            chunkBufs[c].assign(chunkShape[0] * chunkShape[1] * chunkShape[2], (c < 3) ? 128 : 0);
        }
        for (const auto& record : records) {
            const auto lin = linearIndex(chunkShape, record.z, record.y, record.x);
            for (int c = 0; c < 7; ++c) {
                chunkBufs[c][lin] = record.values[c];
            }
        }

        for (size_t z = 0; z < chunkShape[0]; ++z) {
            for (size_t y = 0; y < chunkShape[1]; ++y) {
                for (size_t x = 0; x < chunkShape[2]; ++x) {
                    const size_t globalZ = static_cast<size_t>(chunk.z) * chunkShape[0] + z;
                    const size_t globalY = static_cast<size_t>(chunk.y) * chunkShape[1] + y;
                    const size_t globalX = static_cast<size_t>(chunk.x) * chunkShape[2] + x;
                    if (globalZ >= volumeShape[0] || globalY >= volumeShape[1] || globalX >= volumeShape[2]) {
                        continue;
                    }
                    const auto globalLin = linearIndex(volumeShape, globalZ, globalY, globalX);
                    const auto localLin = linearIndex(chunkShape, z, y, x);
                    for (int c = 0; c < 7; ++c) {
                        EXPECT_EQ(chunkBufs[c][localLin], dense[c][globalLin]);
                    }
                }
            }
        }
    }
}
