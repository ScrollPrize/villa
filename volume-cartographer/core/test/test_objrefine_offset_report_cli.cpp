// End-to-end contract for vc_objrefine's opt-in offset report, on a
// synthetic volume with a known answer: a flat tifxyz sheet and a bright slab
// exactly 10 voxels away along the sheet normal, covering only part of the
// sheet. The summary math has its own fixtures in
// test_objrefine_offset_report.cpp; this file covers what only the binary
// does: a report-only run writes the maps on the segment grid and no
// surface, it refuses an output that is or holds a segment, the maps match
// the move refine actually makes, the refined surface is the same with and
// without the report, the default run writes no report, a later save of the
// folder drops an old report, and bad report parameters are refused.
//
// The binary path arrives from CMake as VC_OBJREFINE_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "utils/Json.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
TEST_CASE("cli tests are POSIX-only") {}
#else

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr int ROWS = 12;
constexpr int COLS = 12;
constexpr float SHEET_Z = 20.f;
constexpr int SLAB_DIST = 10;   // voxels from the sheet to the slab face
constexpr int SLAB_X0 = 40;     // the slab covers x >= SLAB_X0 only
constexpr float BORDER_OFF = 1.f;
constexpr int VOL_Z = 64;
constexpr int VOL_XY = 96;
// The probe is a (2r+1)^2 patch at 1/reader_scale voxel spacing: with the
// defaults (r = 3, reader_scale = 0.5) it spans 12 voxels along each in-plane
// axis, plus one voxel of trilinear support. The patch currently starts at
// the vertex rather than centering on it; this bound holds either way.
constexpr float REACH = 13.f;

// Grid cell (v, u) sits at x = 16 + 4u, y = 16 + 4v, z = SHEET_Z, so every
// probe stays inside the volume. Columns with x + REACH < SLAB_X0 see no
// papyrus; columns with x - REACH >= SLAB_X0 see only slab.
float gridX(int u) { return 16.f + 4.f * u; }

void write_tifxyz(const fs::path& dir)
{
    cv::Mat x(ROWS, COLS, CV_32F), y(ROWS, COLS, CV_32F), z(ROWS, COLS, CV_32F);
    for (int v = 0; v < ROWS; ++v)
        for (int u = 0; u < COLS; ++u) {
            x.at<float>(v, u) = gridX(u);
            y.at<float>(v, u) = 16.f + 4.f * v;
            z.at<float>(v, u) = SHEET_Z;
        }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), z));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"scale\": [1.0, 1.0], \"uuid\": \"objrefine-fixture\"}\n";
    REQUIRE(meta.good());
}

// Slab of 255 on the side the sheet normal points to, starting exactly
// SLAB_DIST voxels away, so an exact probe sample lands on its face.
void write_volume(const fs::path& dir, float normalZ)
{
    Volume::ZarrCreateOptions opts;
    opts.shapeZYX = {VOL_Z, VOL_XY, VOL_XY};
    opts.chunkShapeZYX = {32, 32, 32};
    opts.numLevels = 1;
    opts.uuid = "objrefine-fixture-vol";
    opts.name = "objrefine-fixture-vol";
    opts.overwriteExisting = true;
    opts.compressor = "none";
    auto vol = Volume::New(dir, opts);
    REQUIRE(vol);
    Array3D<uint8_t> data({VOL_Z, VOL_XY, VOL_XY}, 0);
    for (int z = 0; z < VOL_Z; ++z) {
        if (normalZ * (static_cast<float>(z) - SHEET_Z) < SLAB_DIST - 0.5f)
            continue;
        for (int y = 0; y < VOL_XY; ++y)
            for (int x = SLAB_X0; x < VOL_XY; ++x)
                data(z, y, x) = 255;
    }
    vol->writeZYX(data, {0, 0, 0}, 0);
}

void write_params(const fs::path& path, bool refine, bool report)
{
    std::ofstream p(path);
    p << "{\"refine\": " << (refine ? "true" : "false")
      << ", \"offset_report\": " << (report ? "true" : "false")
      << ", \"border_off\": " << BORDER_OFF
      << ", \"scale_group\": 0, \"cache_mb\": 64}\n";
    REQUIRE(p.good());
}

std::string sh(const std::string& s)
{
    std::string q = "'";
    for (char c : s)
        if (c == '\'') q += "'\\''"; else q += c;
    return q + "'";
}

int run_cli(const std::vector<std::string>& args, const fs::path& cwd = {})
{
    std::string cmd = cwd.empty() ? std::string() : "cd " + sh(cwd.string()) + " && ";
    cmd += sh(VC_OBJREFINE_BIN);
    for (const auto& a : args) cmd += " " + sh(a);
    cmd += " >/dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    REQUIRE(rc != -1);
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -2;
}

// Sorted names of the entries directly in `dir`.
std::vector<std::string> listing(const fs::path& dir)
{
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(dir))
        names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

const std::vector<std::string> REPORT_FILES{"offset.tif", "opacity.tif", "report.json"};

// Top-level <name>.tif files: what the tifxyz loader registers as channels.
std::vector<std::string> tif_stems(const fs::path& dir)
{
    std::vector<std::string> stems;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".tif")
            stems.push_back(e.path().stem().string());
    std::sort(stems.begin(), stems.end());
    return stems;
}

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

cv::Mat read_f32(const fs::path& p)
{
    cv::Mat m = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!m.empty());
    REQUIRE(m.type() == CV_32FC1);
    return m;
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        path = fs::temp_directory_path()
             / ("vc_objrefine_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

}  // namespace

TEST_CASE("offset report: known offset, grid-aligned maps, surface unchanged")
{
    TempDir tmp;
    const fs::path src = tmp.path / "sheet.tifxyz";
    write_tifxyz(src);

    float normalZ = 0.f;
    {
        auto surf = load_quad_from_tifxyz(src.string());
        const cv::Vec3f n = surf->gridNormal(ROWS / 2, COLS / 2);
        REQUIRE(std::fabs(std::fabs(n[2]) - 1.f) < 1e-5f);
        normalZ = n[2];
    }
    const fs::path vol = tmp.path / "vol.zarr";
    write_volume(vol, normalZ);

    // Report only: measure without moving anything. The output is a plain
    // folder with the report and no surface.
    const fs::path pReport = tmp.path / "report.json";
    write_params(pReport, false, true);
    const fs::path reportDir = tmp.path / "measured";
    REQUIRE(run_cli({vol.string(), src.string(), reportDir.string(),
                     pReport.string()}) == 0);
    CHECK(listing(reportDir) == REPORT_FILES);
    CHECK_FALSE(fs::exists(tmp.path / ".measured.tmp"));

    const cv::Mat off = read_f32(reportDir / "offset.tif");
    const cv::Mat alpha = read_f32(reportDir / "opacity.tif");
    CHECK(off.rows == ROWS);
    CHECK(off.cols == COLS);
    CHECK(alpha.size() == off.size());

    int measured = 0, onSlab = 0, offSlab = 0;
    for (int v = 0; v < ROWS; ++v)
        for (int u = 0; u < COLS; ++u) {
            const float o = off.at<float>(v, u);
            const float a = alpha.at<float>(v, u);
            CHECK(std::isfinite(o) == std::isfinite(a));
            if (!std::isfinite(o)) continue;
            ++measured;
            if (gridX(u) - REACH >= SLAB_X0) {
                ++onSlab;
                CHECK(o == doctest::Approx(SLAB_DIST).epsilon(1e-4));
                CHECK(a == doctest::Approx(1.0).epsilon(1e-4));
            } else if (gridX(u) + REACH < SLAB_X0) {
                // No papyrus in range: the raw offset is 0, the same as a
                // vertex on the papyrus face; only the opacity tells them apart.
                ++offSlab;
                CHECK(a == 0.f);
                CHECK(o == 0.f);
            }
        }
    REQUIRE(onSlab > 0);
    REQUIRE(offSlab > 0);

    utils::Json j = utils::Json::parse_file(reportDir / "report.json");
    CHECK(j["tool"].get_string() == "vc_objrefine");
    CHECK(j["schema_version"].get_int() == 1);
    CHECK(j["offset_map"].get_string() == "offset.tif");
    CHECK(j["opacity_map"].get_string() == "opacity.tif");
    CHECK(j["inputs"]["volume"].get_string() == fs::weakly_canonical(vol).string());
    CHECK(j["inputs"]["source_surface"].get_string() == fs::weakly_canonical(src).string());
    CHECK(j["inputs"]["source_uuid"].get_string() == "objrefine-fixture");
    CHECK(!j["parameters"]["refine"].get_bool());
    utils::Json& s = j["summary"];
    CHECK(s["rows"].get_int() == ROWS);
    CHECK(s["cols"].get_int() == COLS);
    // Every point is valid, but refine skips the last row and column.
    CHECK(s["valid_vertices"].get_int() == ROWS * COLS);
    CHECK(measured < ROWS * COLS);
    CHECK(s["measured_vertices"].get_int() == measured);
    CHECK(s["no_sheet_vertices"].get_int() >= offSlab);
    CHECK(s["no_sheet_vertices"].get_int() + s["summarized_vertices"].get_int()
          == measured);
    // Every vertex that saw the slab saw it at 10 voxels or more.
    CHECK(s["beyond_threshold_vertices"].get_int()
          == s["summarized_vertices"].get_int());
    CHECK(s["offset"]["p05"].get_double()
          == doctest::Approx(SLAB_DIST).epsilon(1e-4));

    // The flag must not change the refined surface, and must agree with it.
    const fs::path pPlain = tmp.path / "plain.json";
    const fs::path pBoth = tmp.path / "both.json";
    write_params(pPlain, true, false);
    write_params(pBoth, true, true);
    const fs::path outPlain = tmp.path / "refined.tifxyz";
    const fs::path outBoth = tmp.path / "refined_report.tifxyz";
    REQUIRE(run_cli({vol.string(), src.string(), outPlain.string(),
                     pPlain.string()}) == 0);
    REQUIRE(run_cli({vol.string(), src.string(), outBoth.string(),
                     pBoth.string()}) == 0);

    CHECK_FALSE(fs::exists(outPlain / "alphacomp_offset"));
    CHECK(tif_stems(outPlain) == std::vector<std::string>{"x", "y", "z"});
    for (const char* c : {"x.tif", "y.tif", "z.tif"})
        CHECK(slurp(outPlain / c) == slurp(outBoth / c));
    // Measuring does not depend on whether the vertices are then moved.
    CHECK(slurp(reportDir / "offset.tif")
          == slurp(outBoth / "alphacomp_offset" / "offset.tif"));
    // With refine the report sits in its own folder, not as surface
    // channels: the loader would register any top-level <name>.tif.
    CHECK(tif_stems(outBoth) == std::vector<std::string>{"x", "y", "z"});
    CHECK(listing(outBoth / "alphacomp_offset") == REPORT_FILES);

    // The refined vertex moved by offset + border_off along the normal.
    const cv::Mat zIn = read_f32(src / "z.tif");
    const cv::Mat zOut = read_f32(outBoth / "z.tif");
    for (int v = 0; v < ROWS; ++v)
        for (int u = 0; u < COLS; ++u) {
            const float o = off.at<float>(v, u);
            const float dz = zOut.at<float>(v, u) - zIn.at<float>(v, u);
            if (std::isfinite(o))
                CHECK(dz == doctest::Approx(normalZ * (o + BORDER_OFF))
                                .epsilon(1e-4));
            else
                CHECK(dz == 0.f);
        }

    // A later run into the same folder (here without the flag) drops the old
    // report, so it cannot outlive the geometry it measured.
    REQUIRE(run_cli({vol.string(), src.string(), outBoth.string(),
                     pPlain.string()}) == 0);
    CHECK(fs::exists(outBoth / "x.tif"));
    CHECK_FALSE(fs::exists(outBoth / "alphacomp_offset"));

    // A report-only run never writes into a segment: not the source, not a
    // folder inside it, not another segment. Each is refused untouched.
    const std::vector<std::string> srcFiles = listing(src);
    const std::string srcX = slurp(src / "x.tif");
    CHECK(run_cli({vol.string(), src.string(), src.string(),
                   pReport.string()}) == 1);
    CHECK(run_cli({vol.string(), src.string(), (src / "report").string(),
                   pReport.string()}) == 1);
    CHECK(listing(src) == srcFiles);
    CHECK(slurp(src / "x.tif") == srcX);
    const std::vector<std::string> plainFiles = listing(outPlain);
    CHECK(run_cli({vol.string(), src.string(), outPlain.string(),
                   pReport.string()}) == 1);
    CHECK(listing(outPlain) == plainFiles);

    // An earlier report is replaced. Relative paths are recorded absolute.
    const std::string firstReport = slurp(reportDir / "report.json");
    REQUIRE(run_cli({"vol.zarr", "sheet.tifxyz", "measured", "report.json"},
                    tmp.path) == 0);
    CHECK(listing(reportDir) == REPORT_FILES);
    CHECK(slurp(reportDir / "report.json") == firstReport);

    // Report parameters out of range are refused before anything is written.
    const fs::path pBad = tmp.path / "bad.json";
    {
        std::ofstream p(pBad);
        p << "{\"refine\": false, \"offset_report\": true,"
             " \"offset_threshold\": -1, \"scale_group\": 0}\n";
        REQUIRE(p.good());
    }
    const fs::path outBad = tmp.path / "bad.tifxyz";
    CHECK(run_cli({vol.string(), src.string(), outBad.string(),
                   pBad.string()}) == 1);
    CHECK_FALSE(fs::exists(outBad));
}

#endif
