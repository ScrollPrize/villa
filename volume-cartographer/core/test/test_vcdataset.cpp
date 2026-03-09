#include "test.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/Zarr.hpp"

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
        path_ = fs::temp_directory_path() / ("vc_test_vcdataset_" + suffix);
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

nlohmann::json readJson(const fs::path& path)
{
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

std::vector<uint8_t> readBytes(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

size_t linearIndex(const std::vector<size_t>& shape, size_t z, size_t y, size_t x)
{
    return (z * shape[1] + y) * shape[2] + x;
}

} // namespace

TEST(VcDataset, BloscDatasetWritesCompressedChunksAndReadsBack)
{
    ScopedTempDir tempDir;
    const std::vector<size_t> shape = {32, 32, 32};
    const std::vector<size_t> chunks = {32, 32, 32};

    auto ds = vc::createZarrDataset(tempDir.path(), "0", shape, chunks, vc::VcDtype::uint8, "blosc");
    ASSERT_TRUE(ds != nullptr);

    const auto zarray = readJson(tempDir.path() / "0" / ".zarray");
    ASSERT_TRUE(zarray.contains("compressor"));
    ASSERT_TRUE(zarray["compressor"].is_object());
    EXPECT_EQ(zarray["compressor"]["id"].get<std::string>(), std::string("blosc"));
    EXPECT_EQ(zarray["compressor"]["cname"].get<std::string>(), std::string("zstd"));
    EXPECT_EQ(zarray["compressor"]["clevel"].get<int>(), 3);
    EXPECT_EQ(zarray["compressor"]["shuffle"].get<int>(), 1);
    EXPECT_EQ(zarray["compressor"]["blocksize"].get<int>(), 0);

    std::vector<uint8_t> chunk(shape[0] * shape[1] * shape[2], 0);
    chunk[0] = 255;
    chunk[17] = 64;
    chunk[511] = 7;
    chunk.back() = 23;

    EXPECT_TRUE(ds->writeChunk(0, 0, 0, chunk.data(), chunk.size() * sizeof(uint8_t)));

    const fs::path chunkPath = tempDir.path() / "0" / "0.0.0";
    ASSERT_TRUE(fs::exists(chunkPath));
    EXPECT_LT(fs::file_size(chunkPath), static_cast<uintmax_t>(chunk.size() * sizeof(uint8_t)));

    vc::VcDataset reopened(tempDir.path() / "0");

    std::vector<uint8_t> roundTrip(chunk.size(), 0);
    EXPECT_TRUE(reopened.readChunk(0, 0, 0, roundTrip.data()));
    ASSERT_EQ(roundTrip.size(), chunk.size());
    for (size_t i = 0; i < chunk.size(); ++i) {
        EXPECT_EQ(roundTrip[i], chunk[i]);
    }

    const auto rawBytes = readBytes(chunkPath);
    ASSERT_FALSE(rawBytes.empty());
    std::vector<uint8_t> decompressed(chunk.size(), 0);
    reopened.decompress(rawBytes, decompressed.data(), decompressed.size());
    for (size_t i = 0; i < chunk.size(); ++i) {
        EXPECT_EQ(decompressed[i], chunk[i]);
    }
}

TEST(VcDataset, CreateZarrDatasetWritesConfiguredFillValue)
{
    ScopedTempDir tempDir;
    const std::vector<size_t> shape = {8, 9, 10};
    const std::vector<size_t> chunks = {3, 4, 5};

    auto ds = vc::createZarrDataset(
        tempDir.path(), "0", shape, chunks, vc::VcDtype::uint8, "none", ".", 128);
    ASSERT_TRUE(ds != nullptr);

    const auto zarray = readJson(tempDir.path() / "0" / ".zarray");
    ASSERT_TRUE(zarray.contains("fill_value"));
    EXPECT_EQ(zarray["fill_value"].get<int>(), 128);
}

TEST(VcDataset, WriteZarrRegionU8ByChunkPreservesSubregionAndFillPadding)
{
    ScopedTempDir tempDir;
    const std::vector<size_t> shape = {8, 9, 10};
    const std::vector<size_t> chunks = {3, 4, 5};
    const std::vector<size_t> offset = {1, 2, 3};
    const std::vector<size_t> regionShape = {3, 3, 3};

    auto ds = vc::createZarrDataset(
        tempDir.path(), "0", shape, chunks, vc::VcDtype::uint8, "none", ".", 128);
    ASSERT_TRUE(ds != nullptr);

    std::vector<uint8_t> region(regionShape[0] * regionShape[1] * regionShape[2], 0);
    std::iota(region.begin(), region.end(), static_cast<uint8_t>(1));

    writeZarrRegionU8ByChunk(ds.get(), offset, regionShape, region.data(), 128);

    EXPECT_FALSE(ds->chunkExists(2, 2, 1));

    std::vector<uint8_t> full(shape[0] * shape[1] * shape[2], 0);
    EXPECT_TRUE(ds->readRegion({0, 0, 0}, shape, full.data()));

    for (size_t z = 0; z < shape[0]; ++z) {
        for (size_t y = 0; y < shape[1]; ++y) {
            for (size_t x = 0; x < shape[2]; ++x) {
                uint8_t expected = 128;
                if (z >= offset[0] && z < offset[0] + regionShape[0] &&
                    y >= offset[1] && y < offset[1] + regionShape[1] &&
                    x >= offset[2] && x < offset[2] + regionShape[2]) {
                    expected = region[linearIndex(
                        regionShape, z - offset[0], y - offset[1], x - offset[2])];
                }
                EXPECT_EQ(full[linearIndex(shape, z, y, x)], expected);
            }
        }
    }
}
