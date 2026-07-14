// Regression test for the approval-mask autosave crash: a worker-thread
// saveOverwrite() (directory swap) racing a main-thread saveChannel() (in-place
// rename) on the SAME segment dir used to delete the tmp out from under the
// rename -> uncaught filesystem_error -> std::terminate. The per-directory
// write lock (QuadSurface::dirWriteMutex) must serialize them.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_save_conc_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> grid(int rows = 8, int cols = 8, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(float(c), float(r), z);
    return m;
}

cv::Mat_<uint8_t> mask(uint8_t v) { return cv::Mat_<uint8_t>(8, 8, v); }

} // namespace

TEST_CASE("concurrent saveChannel + saveOverwrite on same dir does not crash or throw")
{
    auto root = tmpDir("race");
    auto seg = root / "seg";
    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.save(seg.string(), "uuid-a", false);
    }

    constexpr int kIters = 200;
    std::atomic<bool> failed{false};
    std::atomic<int> done{0};
    std::mutex errorMutex;
    std::string errorMessage;

    auto recordFailure = [&](const char* worker, const std::exception* error) {
        failed = true;
        std::lock_guard<std::mutex> lock(errorMutex);
        errorMessage = worker;
        errorMessage += error ? std::string(": ") + error->what() : ": unknown exception";
    };

    // Worker A: approval channel writes (the main-thread equivalent).
    std::thread a([&] {
        try {
            QuadSurface qs(seg);
            qs.ensureLoaded();
            for (int i = 0; i < kIters; ++i) {
                qs.setChannel("approval", mask(static_cast<uint8_t>(i)));
                qs.saveChannel("approval");
            }
        } catch (const std::exception& e) {
            recordFailure("saveChannel worker", &e);
        } catch (...) {
            recordFailure("saveChannel worker", nullptr);
        }
        ++done;
    });

    // Worker B: full-dir overwrites (the autosave-worker equivalent), a separate
    // QuadSurface object pointing at the same on-disk dir.
    std::thread b([&] {
        try {
            QuadSurface qs(seg);
            qs.ensureLoaded();
            for (int i = 0; i < kIters; ++i) {
                *qs.rawPointsPtr() = grid(8, 8, static_cast<float>(i));
                qs.invalidateCache();
                qs.saveOverwrite();
            }
        } catch (const std::exception& e) {
            recordFailure("saveOverwrite worker", &e);
        } catch (...) {
            recordFailure("saveOverwrite worker", nullptr);
        }
        ++done;
    });

    a.join();
    b.join();
    CHECK(done == 2);
    CHECK_MESSAGE(!failed.load(), errorMessage);

    // Surface remains loadable + intact after the contention.
    QuadSurface reloaded(seg);
    reloaded.ensureLoaded();
    CHECK(fs::exists(seg / "x.tif"));
    CHECK(fs::exists(seg / "approval.tif"));

    fs::remove_all(root);
}
