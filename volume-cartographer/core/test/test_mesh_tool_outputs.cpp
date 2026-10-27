// The mesh tools' outputs on small fixed inputs: the OBJ vc_tifxyz2obj writes
// (bytes) and the grids vc_obj2tifxyz and vc_obj2tifxyz_legacy rasterize from
// an OBJ. The expected values were recorded from the tools before their mesh
// code moved into vc::meshing; each tool's binary comes from VC_<TOOL>_BIN.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {
std::string readFile(const fs::path& path)
{
    std::stringstream text;
    text << std::ifstream(path, std::ios::binary).rdbuf();
    return text.str();
}

void run(const std::string& command, const fs::path& log)
{
    const int status = std::system((command + " > \"" + log.string() + "\" 2>&1").c_str());
    if (status != 0) MESSAGE(command << "\n" << readFile(log));
    REQUIRE(status == 0);
}

std::string tool(const char* variable)
{
    const char* path = std::getenv(variable);
    REQUIRE(path);
    return std::string("\"") + path + "\"";
}

std::string quoted(const fs::path& path) { return "\"" + path.string() + "\""; }

struct Dir {
    fs::path path = fs::temp_directory_path() / ("vc-mesh-tool-outputs-" + vc::objects::util::newUuid());
    Dir() { fs::create_directories(path); }
    ~Dir() { std::error_code ignored; if (!std::getenv("VC_KEEP_FIXTURE")) fs::remove_all(path, ignored); }
};

// A grid's shape, valid points and position-weighted coordinate sums: a change
// of coverage changes the count, a change of any point changes the sums.
struct GridStats {
    int rows = 0, cols = 0, valid = 0;
    double sum[3] = {0, 0, 0};
    double weighted[3] = {0, 0, 0};
};

template <typename Vec>
GridStats stats(const cv::Mat_<Vec>& grid, bool (*valid)(const Vec&))
{
    GridStats s;
    s.rows = grid.rows;
    s.cols = grid.cols;
    for (int r = 0; r < grid.rows; ++r)
        for (int c = 0; c < grid.cols; ++c) {
            const Vec& p = grid(r, c);
            if (!valid(p)) continue;
            ++s.valid;
            const double w = 1.0 + 0.01 * r + 0.001 * c;
            for (int k = 0; k < Vec::channels; ++k) {
                s.sum[k] += p[k];
                s.weighted[k] += w * p[k];
            }
        }
    return s;
}

bool validPoint(const cv::Vec3f& p) { return p[0] != -1.f; }

void checkStats(const GridStats& s, const GridStats& expected, int channels)
{
    MESSAGE("rows " << s.rows << " cols " << s.cols << " valid " << s.valid);
    for (int k = 0; k < channels; ++k) {
        char line[128];
        std::snprintf(line, sizeof line, "sum[%d] %.17g weighted[%d] %.17g", k, s.sum[k], k, s.weighted[k]);
        MESSAGE(line);
    }
    CHECK(s.rows == expected.rows);
    CHECK(s.cols == expected.cols);
    CHECK(s.valid == expected.valid);
    for (int k = 0; k < channels; ++k) {
        CHECK(s.sum[k] == doctest::Approx(expected.sum[k]).epsilon(1e-6));
        CHECK(s.weighted[k] == doctest::Approx(expected.weighted[k]).epsilon(1e-6));
    }
}

// A curved 12 x 16 surface with one invalid corner, the fixture the store-mode
// tool tests use.
QuadSurface curvedSurface()
{
    cv::Mat_<cv::Vec3f> points(12, 16);
    for (int r = 0; r < points.rows; ++r)
        for (int c = 0; c < points.cols; ++c)
            points(r, c) = {100.f + 4.f * c, 120.f + 4.f * r, 80.f + 0.05f * float((c - 8) * (c - 8))};
    points(0, 0) = {-1.f, -1.f, -1.f};
    return QuadSurface(points, {0.25f, 0.25f});
}

// A 6 x 7 vertex sheet with jittered UVs (no UV on a grid line) and two
// quads left out, as v/vt/vn faces.
void writeJitteredObj(const fs::path& path)
{
    std::ofstream obj(path);
    obj << "# jittered sheet\n";
    const int rows = 6, cols = 7;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const double j = 0.37 * std::sin(1.3 * r + 2.1 * c);
            obj << "v " << 50 + 5 * c + j << " " << 60 + 5 * r - j << " " << 40 + 0.2 * (c - 3) * (c - 3) + 0.1 * r << "\n";
            obj << "vt " << 5.13 * c + 0.41 * j << " " << 4.87 * r + 0.29 * std::cos(0.7 * r * c) << "\n";
            obj << "vn 0 0 1\n";
        }
    auto id = [&](int r, int c) { return std::to_string(r * cols + c + 1); };
    auto corner = [&](int r, int c) { return id(r, c) + "/" + id(r, c) + "/" + id(r, c); };
    for (int r = 0; r + 1 < rows; ++r)
        for (int c = 0; c + 1 < cols; ++c) {
            if ((r == 2 && c == 3) || (r == 4 && c == 0)) continue;
            obj << "f " << corner(r + 1, c) << " " << corner(r, c) << " " << corner(r, c + 1) << "\n";
            obj << "f " << corner(r + 1, c) << " " << corner(r, c + 1) << " " << corner(r + 1, c + 1) << "\n";
        }
}
}

TEST_CASE("vc_tifxyz2obj writes a grid's vertices, UVs, normals and faces")
{
    Dir dir;
    // A 3 x 4 plane on a 2 voxel lattice, its last corner invalid: every value is exact.
    cv::Mat_<cv::Vec3f> points(3, 4);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) points(r, c) = {10.f + 2.f * c, 20.f + 2.f * r, 30.f};
    points(2, 3) = {-1.f, -1.f, -1.f};
    QuadSurface(points, {0.5f, 0.5f}).save(dir.path / "plane", true);
    run(tool("VC_TIFXYZ2OBJ_BIN") + " " + quoted(dir.path / "plane") + " " + quoted(dir.path / "plane.obj"), dir.path / "obj.log");
    const std::string expected =
        "v 10.000000 20.000000 30.000000\nvt 0.000000 0.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 12.000000 20.000000 30.000000\nvt 2.000000 0.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 10.000000 22.000000 30.000000\nvt 0.000000 2.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 12.000000 22.000000 30.000000\nvt 2.000000 2.000000\nvn 0.000000 0.000000 1.000000\n"
        "f 3/3/3 1/1/1 2/2/2\nf 3/3/3 2/2/2 4/4/4\n"
        "v 14.000000 20.000000 30.000000\nvt 4.000000 0.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 14.000000 22.000000 30.000000\nvt 4.000000 2.000000\nvn 0.000000 0.000000 1.000000\n"
        "f 4/4/4 2/2/2 5/5/5\nf 4/4/4 5/5/5 6/6/6\n"
        "v 16.000000 20.000000 30.000000\nvt 6.000000 0.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 16.000000 22.000000 30.000000\nvt 6.000000 2.000000\nvn 0.000000 0.000000 1.000000\n"
        "f 6/6/6 5/5/5 7/7/7\nf 6/6/6 7/7/7 8/8/8\n"
        "v 10.000000 24.000000 30.000000\nvt 0.000000 4.000000\nvn 0.000000 0.000000 1.000000\n"
        "v 12.000000 24.000000 30.000000\nvt 2.000000 4.000000\nvn 0.000000 0.000000 1.000000\n"
        "f 9/9/9 3/3/3 4/4/4\nf 9/9/9 4/4/4 10/10/10\n"
        "v 14.000000 24.000000 30.000000\nvt 4.000000 4.000000\nvn 0.000000 0.000000 1.000000\n"
        "f 10/10/10 4/4/4 6/6/6\nf 10/10/10 6/6/6 11/11/11\n";
    CHECK(readFile(dir.path / "plane.obj") == expected);
}

TEST_CASE("vc_tifxyz2obj --keep writes the decimated grid")
{
    Dir dir;
    curvedSurface().save(dir.path / "curved", true);
    run(tool("VC_TIFXYZ2OBJ_BIN") + " " + quoted(dir.path / "curved") + " " + quoted(dir.path / "keep.obj") + " --keep=25",
        dir.path / "obj.log");
    const std::string obj = readFile(dir.path / "keep.obj");
    // 6 x 8 decimated grid, its first quad touching the invalid corner.
    int v = 0, f = 0;
    std::istringstream lines(obj);
    for (std::string line; std::getline(lines, line);) {
        v += line.rfind("v ", 0) == 0;
        f += line.rfind("f ", 0) == 0;
    }
    CHECK(v == 47);
    CHECK(f == 2 * (5 * 7 - 1));
    CHECK(obj.substr(0, obj.find("\nf ")) ==
          "v 108.000000 120.000000 81.800003\nvt 4.000000 0.000000\nvn 0.124035 0.000000 0.992278\n"
          "v 116.000000 120.000000 80.800003\nvt 8.000000 0.000000\nvn 0.107703 0.000000 0.994183\n"
          "v 108.000000 128.000000 81.800003\nvt 4.000000 4.000000\nvn 0.143499 0.000000 0.989651\n"
          "v 116.000000 128.000000 80.800003\nvt 8.000000 4.000000\nvn 0.099504 0.000000 0.995037");
}

TEST_CASE("vc_obj2tifxyz and vc_obj2tifxyz_legacy rasterize an OBJ")
{
    Dir dir;
    writeJitteredObj(dir.path / "sheet.obj");
    run(tool("VC_OBJ2TIFXYZ_BIN") + " " + quoted(dir.path / "sheet.obj") + " " + quoted(dir.path / "modern") + " 4",
        dir.path / "modern.log");
    const auto modern = load_quad_from_tifxyz(dir.path / "modern");
    checkStats(stats(*modern->rawPointsPtr(), validPoint), GridStats{99, 126, 11136, {727999.00440979004, 805493.74280548096, 455152.21654510498}, {1124922.4311063644, 1261103.0851978033, 700345.92459452897}}, 3);

    run(tool("VC_OBJ2TIFXYZ_LEGACY_BIN") + " " + quoted(dir.path / "sheet.obj") + " " + quoted(dir.path / "legacy") + " 1.5",
        dir.path / "legacy.log");
    const auto legacy = load_quad_from_tifxyz(dir.path / "legacy");
    checkStats(stats(*legacy->rawPointsPtr(), validPoint), GridStats{19, 22, 311, {20450.56852722168, 22397.15013885498, 12711.597316741943}, {22333.401655410762, 24536.34768153762, 13867.411323276525}}, 3);
}
