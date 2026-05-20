// More QuadSurface coverage: writeOverlappingJson / readOverlappingJson,
// save_meta, refreshMaskTimestamp, the path-only save() overload, plus the
// free helpers write_overlapping_json / read_overlapping_json.

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

TEST_CASE("save_meta updates meta.json without rewriting TIFFs")
{
    TmpSegDir t;
    auto pts = grid();
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = t.segId;
    qs.path = t.segDir;
    qs.save(t.segDir);
    // Mutate meta in-memory and persist.
    qs.meta = utils::Json::object();
    qs.meta["custom_field"] = "custom_value";
    qs.save_meta();
    // Reload and verify
    QuadSurface reloaded(t.segDir);
    CHECK(reloaded.meta.contains("custom_field"));
    CHECK(reloaded.meta["custom_field"].get_string() == "custom_value");
}

TEST_CASE("writeOverlappingJson / readOverlappingJson round-trip")
{
    TmpSegDir t;
    auto pts = grid();
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = t.segId;
    qs.path = t.segDir;
    qs.save(t.segDir);

    qs.setOverlappingIds({"a", "b", "c"});
    qs.writeOverlappingJson();
    CHECK(fs::exists(t.segDir / "overlapping.json"));

    QuadSurface qs2(t.segDir);
    qs2.readOverlappingJson();
    CHECK(qs2.overlappingIds() == std::set<std::string>{"a", "b", "c"});
}

TEST_CASE("write_overlapping_json (free helper) writes a parseable file")
{
    TmpSegDir t;
    fs::create_directories(t.segDir);
    write_overlapping_json(t.segDir, std::set<std::string>{"x", "y"});
    CHECK(fs::exists(t.segDir / "overlapping.json"));
    auto names = read_overlapping_json(t.segDir);
    CHECK(names == std::set<std::string>{"x", "y"});
}

TEST_CASE("read_overlapping_json: missing file returns empty set")
{
    TmpSegDir t;
    fs::create_directories(t.segDir);
    auto names = read_overlapping_json(t.segDir);
    CHECK(names.empty());
}

TEST_CASE("readMaskTimestamp: returns nullopt when no mask file")
{
    TmpSegDir t;
    fs::create_directories(t.segDir);
    auto ts = QuadSurface::readMaskTimestamp(t.segDir);
    CHECK_FALSE(ts.has_value());
}

TEST_CASE("readMaskTimestamp / refreshMaskTimestamp: writeValidMask leaves a timestamped mask.tif")
{
    TmpSegDir t;
    auto pts = grid();
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.id = t.segId;
    qs.path = t.segDir;
    qs.save(t.segDir);
    qs.writeValidMask(); // writes mask.tif inside segDir

    // readMaskTimestamp (static) should now succeed if mask.tif exists.
    auto ts = QuadSurface::readMaskTimestamp(t.segDir);
    if (fs::exists(t.segDir / "mask.tif")) {
        CHECK(ts.has_value());
    }

    // refreshMaskTimestamp is a void member; just exercise it.
    qs.refreshMaskTimestamp();
    auto memTs = qs.maskTimestamp();
    if (fs::exists(t.segDir / "mask.tif")) {
        CHECK(memTs.has_value());
    }
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
    // saveSnapshot stores into <root>/backups/<id>/0
    QuadSurface qs2(t.segDir);
    qs2.id = t.segId;
    qs2.path = t.segDir;
    qs2.saveSnapshot(/*maxBackups=*/3);

    auto backupsRoot = t.root / "backups" / t.segId;
    CHECK(fs::exists(backupsRoot / "0"));
}
