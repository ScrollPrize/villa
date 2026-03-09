#include "test.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/VcDataset.hpp"

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
