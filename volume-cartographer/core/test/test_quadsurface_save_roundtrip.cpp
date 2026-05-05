// Round-trip + recovery tests for QuadSurface::saveOverwrite().
//
// Pre-fix scenario the annotator hit: a tall-skinny segment came back
// containing only the small Shift-E hole-fill patches because the
// in-memory points had (-1,-1,-1) outside the patch when save ran;
// load + trim then cropped to the patch and the next save wrote that
// crop as the entire segment, permanently losing the rest.
//
// Defenses verified here:
//  1) M2 aggressive-trim guard: a sparse on-disk surface still loads
//     at original size and saveOverwrite preserves it.
//  2) M3 saveSnapshot before saveOverwrite: the rotating backup at
//     <volpkg>/backups/<seg>/0/ contains the prior state.
//  3) M3 atomic TIFF writes (already provided by save's directory
//     swap on Linux): a stray .tmp file does not break reload.

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

struct TmpVolpkg {
    fs::path root;        // <root>
    fs::path pathsDir;    // <root>/paths
    fs::path segDir;      // <root>/paths/<segName>
    fs::path backupsDir;  // <root>/backups/<segName>
    std::string segName;

    explicit TmpVolpkg(const std::string& tag)
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        root = fs::temp_directory_path() /
               ("vc_qs_test_" + tag + "_" + std::to_string(rng()));
        pathsDir = root / "paths";
        segName = "seg_" + std::to_string(rng());
        segDir = pathsDir / segName;
        backupsDir = root / "backups" / segName;
        fs::create_directories(pathsDir);
    }

    ~TmpVolpkg()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

cv::Mat_<cv::Vec3f> makeSparseGrid(int rows, int cols, int patchH, int patchW)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols, cv::Vec3f(-1.f, -1.f, -1.f));
    const int r0 = (rows - patchH) / 2;
    const int c0 = (cols - patchW) / 2;
    for (int r = r0; r < r0 + patchH; ++r) {
        for (int c = c0; c < c0 + patchW; ++c) {
            pts(r, c) = cv::Vec3f(static_cast<float>(c),
                                  static_cast<float>(r),
                                  100.f);
        }
    }
    return pts;
}

}  // namespace

TEST_CASE("saveOverwrite round-trip preserves sparse-but-large surface")
{
    TmpVolpkg pkg("roundtrip_sparse");

    // 200x200 with a 30x30 patch. keepFraction = 900/40000 = 2.25% which
    // is well under the M2 guard threshold (40%). With the guard the
    // load-side trim refuses to crop, so the on-disk x/y/z.tif stay
    // 200x200 across save+load cycles.
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(200, 200, 30, 30);

    // First save creates the segment dir.
    {
        QuadSurface surf(pts, cv::Vec2f(1.f, 1.f));
        surf.path = pkg.segDir;
        surf.id = pkg.segName;
        surf.save(pkg.segDir.string(), pkg.segName, /*force_overwrite=*/false);
    }

    // saveOverwrite path: load, then save again.
    {
        QuadSurface loaded(pkg.segDir);
        loaded.ensureLoaded();
        REQUIRE(loaded.rawPointsPtr() != nullptr);
        // M2 guard kept the size at 200x200.
        CHECK(loaded.rawPointsPtr()->size() == cv::Size(200, 200));
        loaded.saveOverwrite();
    }

    // Re-read after the second save.
    {
        QuadSurface reloaded(pkg.segDir);
        reloaded.ensureLoaded();
        CHECK(reloaded.rawPointsPtr()->size() == cv::Size(200, 200));
    }
}

TEST_CASE("saveOverwrite creates a backup of the prior state")
{
    TmpVolpkg pkg("backup");

    cv::Mat_<cv::Vec3f> pts(64, 64, cv::Vec3f(0.f, 0.f, 50.f));

    {
        QuadSurface surf(pts, cv::Vec2f(1.f, 1.f));
        surf.path = pkg.segDir;
        surf.id = pkg.segName;
        surf.save(pkg.segDir.string(), pkg.segName, /*force_overwrite=*/false);
    }
    REQUIRE_FALSE(fs::exists(pkg.backupsDir));  // no backups yet

    // saveOverwrite must take a snapshot before replacing on-disk files.
    {
        QuadSurface loaded(pkg.segDir);
        loaded.ensureLoaded();
        loaded.saveOverwrite();
    }

    // backups/<seg>/0/ should now contain the prior x.tif.
    REQUIRE(fs::exists(pkg.backupsDir));
    CHECK(fs::exists(pkg.backupsDir / "0" / "x.tif"));
    CHECK(fs::exists(pkg.backupsDir / "0" / "y.tif"));
    CHECK(fs::exists(pkg.backupsDir / "0" / "z.tif"));
}

TEST_CASE("stale .tmp file in segment dir does not break reload")
{
    TmpVolpkg pkg("stale_tmp");

    cv::Mat_<cv::Vec3f> pts(64, 64, cv::Vec3f(0.f, 0.f, 50.f));

    {
        QuadSurface surf(pts, cv::Vec2f(1.f, 1.f));
        surf.path = pkg.segDir;
        surf.id = pkg.segName;
        surf.save(pkg.segDir.string(), pkg.segName, /*force_overwrite=*/false);
    }

    // Simulate a crash mid-write: some file got written as .tmp but
    // never renamed. Reload must ignore it and use the real x.tif.
    {
        std::ofstream stale(pkg.segDir / "x.tif.tmp");
        stale << "garbage";
    }
    REQUIRE(fs::exists(pkg.segDir / "x.tif"));
    REQUIRE(fs::exists(pkg.segDir / "x.tif.tmp"));

    QuadSurface reloaded(pkg.segDir);
    reloaded.ensureLoaded();
    CHECK(reloaded.rawPointsPtr() != nullptr);
    CHECK(reloaded.rawPointsPtr()->size() == cv::Size(64, 64));
}
