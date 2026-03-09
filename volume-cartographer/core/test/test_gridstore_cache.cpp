#include "test.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include "vc/core/util/GridStore.hpp"

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
        path_ = fs::temp_directory_path() / ("vc_test_gridstore_" + suffix);
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

std::string pathSignature(const std::vector<std::shared_ptr<std::vector<cv::Point>>>& paths)
{
    std::multiset<std::string> encoded;
    for (const auto& path : paths) {
        std::stringstream ss;
        for (const auto& pt : *path) {
            ss << pt.x << "," << pt.y << ";";
        }
        encoded.insert(ss.str());
    }

    std::stringstream out;
    for (const auto& entry : encoded) {
        out << entry << "|";
    }
    return out.str();
}

void populateWriter(vc::core::util::GridStore& grid)
{
    grid.add({{10, 10}, {18, 18}, {26, 26}, {34, 34}});
    grid.add({{20, 60}, {28, 62}, {36, 64}, {44, 66}});
    grid.add({{80, 80}, {88, 90}, {96, 100}, {104, 110}});
    grid.add({{48, 12}, {52, 16}, {56, 20}, {60, 24}, {64, 28}});
}

} // namespace

TEST(GridStoreCache, ReusesDecodedPathsOnRepeatedROIQuery)
{
    ScopedTempDir tempDir;
    const auto gridPath = tempDir.path() / "sample.grid";

    vc::core::util::GridStore writer(cv::Rect(0, 0, 256, 256), 16);
    populateWriter(writer);
    writer.save(gridPath.string());

    vc::core::util::GridStore reader(gridPath.string());
    reader.resetCacheStats();

    const cv::Rect roi(0, 0, 96, 96);
    const auto first = reader.get(roi);
    const auto firstStats = reader.cacheStats();
    const auto second = reader.get(roi);
    const auto secondStats = reader.cacheStats();

    EXPECT_EQ(pathSignature(first), pathSignature(second));
    EXPECT_GT(firstStats.decodedPathMisses, 0);
    EXPECT_GT(secondStats.decodedPathHits, 0);

    std::set<const void*> firstPtrs;
    std::set<const void*> secondPtrs;
    for (const auto& path : first) firstPtrs.insert(path.get());
    for (const auto& path : second) secondPtrs.insert(path.get());
    EXPECT_EQ(firstPtrs, secondPtrs);
}

TEST(GridStoreCache, ConcurrentRepeatedROIQueriesAreStable)
{
    ScopedTempDir tempDir;
    const auto gridPath = tempDir.path() / "sample.grid";

    vc::core::util::GridStore writer(cv::Rect(0, 0, 256, 256), 16);
    populateWriter(writer);
    writer.save(gridPath.string());

    vc::core::util::GridStore reader(gridPath.string());
    reader.resetCacheStats();

    const cv::Rect roi(0, 0, 128, 128);
    const auto reference = pathSignature(reader.get(roi));

    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                if (pathSignature(reader.get(roi)) != reference) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_TRUE(ok.load(std::memory_order_relaxed));
    const auto stats = reader.cacheStats();
    EXPECT_GT(stats.decodedPathHits, 0);
}

TEST(GridStoreCache, SaveWithoutVerificationStillReloadsIdentically)
{
    ScopedTempDir tempDir;
    const auto gridPath = tempDir.path() / "sample.grid";

    vc::core::util::GridStore writer(cv::Rect(0, 0, 256, 256), 16);
    populateWriter(writer);
    writer.save(gridPath.string(), vc::core::util::GridStore::SaveOptions{.verify_reload = false});

    vc::core::util::GridStore reader(gridPath.string());
    EXPECT_EQ(pathSignature(writer.get_all()), pathSignature(reader.get_all()));
}
