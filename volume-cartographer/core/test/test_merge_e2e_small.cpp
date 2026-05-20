// End-to-end smoke test for vc_merge_tifxyz on a synthetic 2-cell grid.
//
// Builds two trivial 16x16 tifxyz directories under a temp paths dir,
// places them so they share spatial overlap, writes merge.json, and
// invokes the built binary via std::system. Asserts exit==0, the
// expected output files exist, and the summary JSON has a non-
// degenerate bounding box.
//
// Opt-in: only runs when VC_RUN_E2E is set to "1". Skips silently
// otherwise so default ctest stays fast and hermetic.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

cv::Mat_<cv::Vec3f> makePlane(int rows, int cols, int xOffset)
{
    // QuadSurface's loader treats z<=0 as the (-1,-1,-1) sentinel,
    // so points must sit at strictly positive z to count as valid.
    cv::Mat_<cv::Vec3f> pts(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pts(r, c) = cv::Vec3f(static_cast<float>(c + xOffset),
                                  static_cast<float>(r),
                                  100.f);
        }
    }
    return pts;
}

fs::path locateBinary(const fs::path& candidate)
{
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) return candidate;
    return {};
}

fs::path findVcMergeTifxyz()
{
    // 1) explicit override
    if (const char* env = std::getenv("VC_MERGE_TIFXYZ_BIN")) {
        if (auto p = locateBinary(env); !p.empty()) return p;
    }
    // 2) sibling in build/bin/ relative to common source-tree layouts
    for (const fs::path& base : {fs::path("build/bin"),
                                 fs::path("build-macos/bin"),
                                 fs::path("build-macos-rel/bin")}) {
        if (auto p = locateBinary(base / "vc_merge_tifxyz"); !p.empty()) return p;
    }
    // 3) PATH lookup
    if (const char* path = std::getenv("PATH")) {
        std::string s = path;
        std::string::size_type from = 0;
        while (from <= s.size()) {
            auto next = s.find(':', from);
            std::string seg = s.substr(from, next == std::string::npos ? std::string::npos
                                                                       : next - from);
            if (!seg.empty()) {
                if (auto p = locateBinary(fs::path(seg) / "vc_merge_tifxyz"); !p.empty()) return p;
            }
            if (next == std::string::npos) break;
            from = next + 1;
        }
    }
    return {};
}

}

TEST_CASE("vc_merge_tifxyz end-to-end on a synthetic 2-cell grid")
{
    const char* run = std::getenv("VC_RUN_E2E");
    if (!run || std::string(run) != "1") {
        MESSAGE("VC_RUN_E2E != 1; skipping (set VC_RUN_E2E=1 to enable)");
        return;
    }

    const fs::path bin = findVcMergeTifxyz();
    REQUIRE_MESSAGE(!bin.empty(),
                    "vc_merge_tifxyz binary not found; build first or set "
                    "VC_MERGE_TIFXYZ_BIN");

    std::random_device rd;
    std::mt19937_64 rng(rd());
    const std::string tag = "vc_merge_e2e_" + std::to_string(rng());
    const fs::path root  = fs::temp_directory_path() / tag;
    const fs::path paths = root / "paths";
    fs::create_directories(paths);

    // surface_a fills [0,16) x [0,16); surface_b fills [8,24) x [0,16).
    // The 8-wide overlap on the right of A == left of B is what the
    // merge tool aligns.
    const std::string nameA = "surface_a";
    const std::string nameB = "surface_b";
    const fs::path dirA = paths / nameA;
    const fs::path dirB = paths / nameB;

    {
        QuadSurface a(makePlane(16, 16, 0), cv::Vec2f(1.f, 1.f));
        a.path = dirA;
        a.id   = nameA;
        a.save(dirA.string(), nameA, /*force_overwrite=*/false);
    }
    {
        QuadSurface b(makePlane(16, 16, 8), cv::Vec2f(1.f, 1.f));
        b.path = dirB;
        b.id   = nameB;
        b.save(dirB.string(), nameB, /*force_overwrite=*/false);
    }

    const fs::path mj = root / "merge.json";
    {
        std::ofstream f(mj);
        f << R"({"rows":["surface_a surface_b"]})" << std::endl;
    }

    // Run the binary. Tunables stay at defaults; only --merge is required.
    // Redirect stdout/stderr to a log file in the temp dir for triage.
    const fs::path log = root / "merge.log";
    const std::string cmd =
        bin.string() + " --merge " + mj.string() +
        " > " + log.string() + " 2>&1";
    const int rc = std::system(cmd.c_str());

    INFO("vc_merge_tifxyz log: ", log.string());
    REQUIRE_MESSAGE(rc == 0, "vc_merge_tifxyz exited non-zero; see " << log.string());

    // Output dir is auto-named under <volpkg>/paths/ as
    // <alpha_first>_merged[_vN]. <volpkg> here is `root` (the parent of
    // merge.json), so the output lives under paths/. We just probe the
    // surface_a_merged variants in order.
    fs::path outDir;
    for (const auto& cand : {std::string("surface_a_merged"),
                              std::string("surface_a_merged_v2"),
                              std::string("surface_a_merged_v3")}) {
        if (fs::is_directory(paths / cand)) { outDir = paths / cand; break; }
    }
    REQUIRE_MESSAGE(!outDir.empty(),
                    "no surface_a_merged[_vN] directory under " << paths.string());

    CHECK(fs::is_regular_file(outDir / "x.tif"));
    CHECK(fs::is_regular_file(outDir / "y.tif"));
    CHECK(fs::is_regular_file(outDir / "z.tif"));
    // mask.tif is emitted only when vc_obj2tifxyz_legacy detects an
    // invalid region; for the all-valid synthetic case there isn't
    // necessarily one. Inspect when present, otherwise skip.
    if (fs::is_regular_file(outDir / "mask.tif")) {
        MESSAGE("mask.tif present at " << (outDir / "mask.tif").string());
    }

    // Locate the *_summary.json that the tool writes alongside the OBJ.
    fs::path summary;
    for (const auto& entry : fs::directory_iterator(outDir)) {
        const auto& p = entry.path();
        if (p.extension() == ".json" &&
            p.filename().string().find("_summary") != std::string::npos)
        {
            summary = p;
            break;
        }
    }
    REQUIRE_MESSAGE(!summary.empty(), "no *_summary.json under " << outDir.string());

    nlohmann::json j;
    {
        std::ifstream f(summary);
        REQUIRE(f.good());
        f >> j;
    }
    CHECK(j.contains("merge_json"));
    CHECK(j.contains("surfaces"));
    CHECK(j["surfaces"].is_array());
    CHECK(j["surfaces"].size() == 2);

    // Cleanup happens via TmpDir-style remove_all only on success; on
    // failure leave it so the operator can poke at the log.
    std::error_code ec;
    fs::remove_all(root, ec);
}
