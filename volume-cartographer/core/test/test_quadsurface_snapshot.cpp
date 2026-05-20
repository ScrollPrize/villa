// Exercises QuadSurface::saveSnapshot's rotation-when-full branch by saving
// past maxBackups so older slots get rotated out.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_qs_snap_" + tag + "_" + std::to_string(rng()));
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

} // namespace

TEST_CASE("saveSnapshot: empty path errors, no on-disk state is no-op")
{
    QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
    // No path set yet → throws.
    CHECK_THROWS(qs.saveSnapshot(3));
    // Set path but no x.tif on disk → silent no-op.
    auto d = tmpDir("nodir");
    qs.path = d / "paths" / "seg";
    qs.id = "seg";
    qs.saveSnapshot(3);
    fs::remove_all(d);
}

TEST_CASE("saveSnapshot rotates when existing backups exceed maxBackups")
{
    auto vol = tmpDir("rot");
    auto paths = vol / "paths";
    fs::create_directories(paths);
    auto segDir = paths / "seg1";

    // First save creates the segment.
    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg1";
        qs.save(segDir);
        qs.path = segDir; // hook up for subsequent snapshots
    }

    auto reload = [&](){
        auto qs = std::make_unique<QuadSurface>(segDir);
        qs->id = "seg1";
        qs->path = segDir;
        return qs;
    };

    // Take 3 snapshots with maxBackups=2 — third call must rotate.
    {
        auto qs = reload();
        qs->saveSnapshot(/*maxBackups=*/2);
    }
    {
        auto qs = reload();
        qs->saveSnapshot(2);
    }
    {
        auto qs = reload();
        qs->saveSnapshot(2);
    }

    auto backupsRoot = vol / "backups" / "seg1";
    REQUIRE(fs::exists(backupsRoot));
    // After 3 calls with maxBackups=2 we still have slots 0 and 1.
    CHECK(fs::exists(backupsRoot / "0"));
    CHECK(fs::exists(backupsRoot / "1"));
    fs::remove_all(vol);
}

TEST_CASE("saveSnapshot copies all regular files (not just tifs)")
{
    auto vol = tmpDir("allfiles");
    auto paths = vol / "paths";
    fs::create_directories(paths);
    auto segDir = paths / "seg2";

    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg2";
        qs.save(segDir);
    }
    // Add a side-channel file.
    {
        std::ofstream f(segDir / "extra.txt");
        f << "hello";
    }

    auto qs = std::make_unique<QuadSurface>(segDir);
    qs->id = "seg2";
    qs->path = segDir;
    qs->saveSnapshot(3);

    auto slot = vol / "backups" / "seg2" / "0";
    REQUIRE(fs::exists(slot));
    CHECK(fs::exists(slot / "x.tif"));
    CHECK(fs::exists(slot / "extra.txt"));
    fs::remove_all(vol);
}
