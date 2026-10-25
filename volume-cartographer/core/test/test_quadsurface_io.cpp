// More QuadSurface coverage: the path-only save() overload.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>

namespace fs = std::filesystem;

namespace {

struct TmpSegDir {
    fs::path root;
    fs::path segDir;
    std::string segId;
    TmpSegDir() {
        std::mt19937_64 rng(std::random_device{}());
        root = fs::temp_directory_path() /
               ("vc_qs_io_" + std::to_string(rng()));
        segId = "seg_" + std::to_string(rng());
        segDir = root / "paths" / segId;
        fs::create_directories(root / "paths");
    }
    ~TmpSegDir() { std::error_code ec; fs::remove_all(root, ec); }
};

cv::Mat_<cv::Vec3f> grid(int rows = 16, int cols = 16)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), 50.f);
    return m;
}

} // namespace

TEST_CASE("save(path, force_overwrite=false) writes a segment dir")
{
    TmpSegDir t;
    auto pts = grid();
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = t.segId;
    qs.save(t.segDir);
    CHECK(fs::exists(t.segDir));
    CHECK(fs::exists(t.segDir / "meta.json"));
    CHECK(fs::exists(t.segDir / "x.tif"));
    CHECK(fs::exists(t.segDir / "y.tif"));
    CHECK(fs::exists(t.segDir / "z.tif"));
}

TEST_CASE("save() with force_overwrite overwrites an existing dir")
{
    TmpSegDir t;
    auto pts = grid();
    {
        QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
        qs.id = t.segId;
        qs.save(t.segDir);
    }
    // Second save with force_overwrite=true should succeed.
    QuadSurface qs2(pts, cv::Vec2f(1.f, 1.f));
    qs2.id = t.segId;
    qs2.save(t.segDir, /*force_overwrite=*/true);
    CHECK(fs::exists(t.segDir / "x.tif"));
}

TEST_CASE("save() on a path-less surface still writes when path passed explicitly")
{
    TmpSegDir t;
    auto pts = grid();
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = t.segId;
    // Note: qs.path is empty here.
    qs.save(t.segDir);
    CHECK(fs::exists(t.segDir / "x.tif"));
}

TEST_CASE("saveSnapshot: copies prior on-disk state into backups/<id>/0")
{
    TmpSegDir t;
    auto pts = grid();
    {
        QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
        qs.id = t.segId;
        qs.path = t.segDir;
        qs.save(t.segDir);
    }
    // saveSnapshot stores into <segDir-parent>/backups/<id>/0
    QuadSurface qs2(t.segDir);
    qs2.id = t.segId;
    qs2.path = t.segDir;
    qs2.saveSnapshot(/*maxBackups=*/3);

    auto backupsRoot = t.segDir.parent_path() / "backups" / t.segId;
    CHECK(fs::exists(backupsRoot / "0"));
}
