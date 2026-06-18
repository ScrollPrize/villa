// Exercises QuadSurface::saveSnapshot's rotation-when-full branch by saving
// past maxBackups so older slots get rotated out.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
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

    // Take 3 snapshots with maxBackups=2 — third call must rotate. force=true
    // bypasses the per-minute throttle so the rapid calls actually snapshot.
    {
        auto qs = reload();
        qs->saveSnapshot(/*maxBackups=*/2, /*force=*/true);
    }
    {
        auto qs = reload();
        qs->saveSnapshot(2, /*force=*/true);
    }
    {
        auto qs = reload();
        qs->saveSnapshot(2, /*force=*/true);
    }

    // Backups are a sibling of the segment dir, not under the volpkg root.
    auto backupsRoot = paths / "backups" / "seg1";
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

    auto slot = paths / "backups" / "seg2" / "0";
    REQUIRE(fs::exists(slot));
    CHECK(fs::exists(slot / "x.tif"));
    CHECK(fs::exists(slot / "extra.txt"));
    fs::remove_all(vol);
}

TEST_CASE("saveSnapshot throttles rapid calls; force bypasses")
{
    auto vol = tmpDir("throttle");
    auto paths = vol / "paths";
    fs::create_directories(paths);
    auto segDir = paths / "seg3";

    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg3";
        qs.save(segDir);
    }
    auto reload = [&](){
        auto qs = std::make_unique<QuadSurface>(segDir);
        qs->id = "seg3"; qs->path = segDir; return qs;
    };

    auto backupsRoot = paths / "backups" / "seg3";

    // First snapshot creates slot 0.
    reload()->saveSnapshot(10);
    REQUIRE(fs::exists(backupsRoot / "0"));

    // Immediate second snapshot is throttled (within the per-minute window):
    // no new slot appears.
    reload()->saveSnapshot(10);
    CHECK_FALSE(fs::exists(backupsRoot / "1"));

    // force=true bypasses the throttle and creates the next slot.
    reload()->saveSnapshot(10, /*force=*/true);
    CHECK(fs::exists(backupsRoot / "1"));

    fs::remove_all(vol);
}

TEST_CASE("saveSnapshot: no backupRoot falls back to the segment's parent dir")
{
    // With backupRoot unset (e.g. standalone CLI tools), backups land in a
    // backups/ dir beside the segment directory.
    auto base = tmpDir("fallback");
    auto segDir = base / "paths" / "seg1";
    fs::create_directories(base / "paths");

    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg1";
        qs.save(segDir);
    }

    auto qs = std::make_unique<QuadSurface>(segDir);
    qs->id = "seg1";
    qs->path = segDir;
    REQUIRE(qs->backupRoot.empty());
    qs->saveSnapshot(3, /*force=*/true);

    CHECK(fs::exists(segDir.parent_path() / "backups" / "seg1" / "0" / "x.tif"));

    fs::remove_all(base);
}

TEST_CASE("saveSnapshot: backupRoot anchors backups regardless of segment location")
{
    // VolumePkg sets backupRoot to the volpkg.json's directory. Backups must go
    // under <backupRoot>/backups/<id>/ even when the segment lives in a
    // subdirectory (paths/) or somewhere else entirely.
    auto volpkgDir = tmpDir("volpkg");
    auto segDir = volpkgDir / "paths" / "seg1";
    fs::create_directories(volpkgDir / "paths");

    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg1";
        qs.save(segDir);
    }

    auto qs = std::make_unique<QuadSurface>(segDir);
    qs->id = "seg1";
    qs->path = segDir;
    qs->backupRoot = volpkgDir;  // as VolumePkg would set it
    qs->saveSnapshot(3, /*force=*/true);

    // Backup is a sibling of the volpkg dir's contents, NOT next to the segment.
    CHECK(fs::exists(volpkgDir / "backups" / "seg1" / "0" / "x.tif"));
    CHECK_FALSE(fs::exists(segDir.parent_path() / "backups"));

    fs::remove_all(volpkgDir);
}

TEST_CASE("saveSnapshot: backupRoot with a segment outside the volpkg dir")
{
    // The volpkg.json may point at an explicit segment path anywhere on disk.
    // Backups still go under backupRoot, not beside the far-flung segment.
    auto volpkgDir = tmpDir("volpkg_explicit");
    auto segParent = tmpDir("elsewhere");
    auto segDir = segParent / "seg1";

    {
        QuadSurface qs(grid(), cv::Vec2f(1.f, 1.f));
        qs.id = "seg1";
        qs.save(segDir);
    }

    auto qs = std::make_unique<QuadSurface>(segDir);
    qs->id = "seg1";
    qs->path = segDir;
    qs->backupRoot = volpkgDir;
    qs->saveSnapshot(3, /*force=*/true);

    CHECK(fs::exists(volpkgDir / "backups" / "seg1" / "0" / "x.tif"));
    CHECK_FALSE(fs::exists(segParent / "backups"));

    fs::remove_all(volpkgDir);
    fs::remove_all(segParent);
}
