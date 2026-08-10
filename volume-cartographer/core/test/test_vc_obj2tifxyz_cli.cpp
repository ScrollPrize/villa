// End-to-end regression tests for the vc_obj2tifxyz CLI's meta.json `scale`
// output (scroll-prize/villa#1319): `scale = [sx, sy]` must be grid cells
// per volume voxel, measured from the grid actually emitted -- not raw UV/
// OBJ-unit spacing, and not derived from stretch_factor or mesh_units.
//
// ObjToTifxyzConverter lives entirely inside apps/src/vc_obj2tifxyz.cpp
// (its own main(), no header), so -- following the precedent in
// test_tifxyz_selfcross_cli.cpp -- this drives the built binary via
// std::system() against synthetic OBJ fixtures and inspects the resulting
// meta.json and x/y/z.tif directly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "utils/Json.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
TEST_CASE("cli tests are POSIX-only") {}
#else

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

// Single-quote every argument so paths/values cannot break the command or
// execute as shell syntax.
std::string sh(const std::string& s)
{
    std::string q = "'";
    for (char c : s)
        if (c == '\'') q += "'\\''"; else q += c;
    return q + "'";
}

int run_cli(const std::vector<std::string>& args)
{
    std::string cmd = sh(VC_OBJ2TIFXYZ_BIN);
    for (const auto& a : args) cmd += " " + sh(a);
    cmd += " >/dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    REQUIRE(rc != -1);
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -2;
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        path = fs::temp_directory_path()
             / ("vc_obj2tifxyz_cli_" + std::to_string(::getpid())
                + "_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

// Writes an (M x N)-vertex grid mesh: 3D position = (col*dx, row*dy, z0).
// UV is integer (col, row) by default, or normalized to [0,1] per axis when
// normalize_uv is set (matching the #1319 failure-mode input shape). With
// stretch_factor=1 and --uv-metric, the rasterized grid lands exactly on
// these vertices, so the true adjacent-grid spacing is exactly (dx, dy) --
// a known ground truth to check `scale` against.
void write_grid_obj(const fs::path& path, int M, int N, double dx, double dy, double z0,
                     bool normalize_uv = false)
{
    std::ofstream f(path);
    REQUIRE(f.good());
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < M; ++col)
            f << "v " << (col * dx) << " " << (row * dy) << " " << z0 << "\n";
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < M; ++col) {
            if (normalize_uv)
                f << "vt " << (double(col) / (M - 1)) << " " << (double(row) / (N - 1)) << "\n";
            else
                f << "vt " << col << " " << row << "\n";
        }
    auto idx = [&](int row, int col) { return row * M + col + 1; };  // 1-based
    for (int row = 0; row < N - 1; ++row) {
        for (int col = 0; col < M - 1; ++col) {
            const int a = idx(row, col), b = idx(row, col + 1);
            const int c = idx(row + 1, col + 1), d = idx(row + 1, col);
            f << "f " << a << "/" << a << " " << b << "/" << b << " " << c << "/" << c << "\n";
            f << "f " << a << "/" << a << " " << c << "/" << c << " " << d << "/" << d << "\n";
        }
    }
    REQUIRE(f.good());
}

// Minimal valid tifxyz directory used as a --tifxyz-source input.
void write_source_tifxyz(const fs::path& dir, const cv::Vec2f& src_scale)
{
    const int rows = 4, cols = 4;
    cv::Mat x(rows, cols, CV_32F), y(rows, cols, CV_32F), z(rows, cols, CV_32F);
    for (int v = 0; v < rows; ++v)
        for (int u = 0; u < cols; ++u) {
            x.at<float>(v, u) = 2.f * u;
            y.at<float>(v, u) = 3.f * v;
            z.at<float>(v, u) = 5.f;
        }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), z));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"scale\": [" << src_scale[0] << ", " << src_scale[1]
         << "], \"uuid\": \"src-fixture\"}\n";
    REQUIRE(meta.good());
}

cv::Vec2f read_meta_scale(const fs::path& dir)
{
    utils::Json j = utils::Json::parse_file(dir / "meta.json");
    return cv::Vec2f(j["scale"][size_t(0)].get_float(), j["scale"][size_t(1)].get_float());
}

// Reads x/y/z.tif directly and returns the (dx, dy) grid dimensions, plus
// the average 3D distance between adjacent valid points along each axis --
// an independent measurement of the "ground truth" `scale` should encode
// the reciprocal of.
struct GridReadback {
    int cols = 0, rows = 0;
    cv::Vec2f step;  // average 3D distance to the next column / next row
};

GridReadback read_grid(const fs::path& dir)
{
    cv::Mat x = cv::imread((dir / "x.tif").string(), cv::IMREAD_UNCHANGED);
    cv::Mat y = cv::imread((dir / "y.tif").string(), cv::IMREAD_UNCHANGED);
    cv::Mat z = cv::imread((dir / "z.tif").string(), cv::IMREAD_UNCHANGED);
    REQUIRE(!x.empty());
    REQUIRE(x.size() == y.size());
    REQUIRE(x.size() == z.size());

    GridReadback out;
    out.cols = x.cols;
    out.rows = x.rows;

    double sum_x = 0, sum_y = 0;
    int cnt_x = 0, cnt_y = 0;
    auto valid = [&](int r, int c) { return z.at<float>(r, c) > 0.f; };
    auto pos = [&](int r, int c) {
        return cv::Vec3f(x.at<float>(r, c), y.at<float>(r, c), z.at<float>(r, c));
    };
    for (int r = 0; r < out.rows; ++r) {
        for (int c = 0; c < out.cols; ++c) {
            if (!valid(r, c)) continue;
            if (c + 1 < out.cols && valid(r, c + 1)) {
                cv::Vec3f d = pos(r, c + 1) - pos(r, c);
                sum_x += std::sqrt(double(d.dot(d)));
                cnt_x++;
            }
            if (r + 1 < out.rows && valid(r + 1, c)) {
                cv::Vec3f d = pos(r + 1, c) - pos(r, c);
                sum_y += std::sqrt(double(d.dot(d)));
                cnt_y++;
            }
        }
    }
    REQUIRE(cnt_x > 0);
    REQUIRE(cnt_y > 0);
    out.step = cv::Vec2f(float(sum_x / cnt_x), float(sum_y / cnt_y));
    return out;
}

}  // namespace

TEST_CASE("uv-metric: scale is the reciprocal of the actual emitted grid "
          "spacing, per-axis (anisotropic mesh, known 3D spacing)")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0);
    const fs::path out = tmp.path / "out.tifxyz";

    REQUIRE(run_cli({obj.string(), out.string(), "1.0", "1.0", "--uv-metric"}) == 0);

    const cv::Vec2f scale = read_meta_scale(out);
    const GridReadback g = read_grid(out);

    // Ground truth: UV integer spacing == 1 and stretch_factor == 1, so the
    // rasterized grid lands exactly on the source vertices.
    CHECK(g.step[0] == doctest::Approx(10.0f).epsilon(1e-3));
    CHECK(g.step[1] == doctest::Approx(25.0f).epsilon(1e-3));

    // The core #1319 invariant: scale is a density, not a length.
    CHECK(scale[0] == doctest::Approx(0.1f).epsilon(1e-3));
    CHECK(scale[1] == doctest::Approx(0.04f).epsilon(1e-3));
    CHECK(scale[0] * g.step[0] == doctest::Approx(1.0f).epsilon(1e-3));
    CHECK(scale[1] * g.step[1] == doctest::Approx(1.0f).epsilon(1e-3));
    // Anisotropy preserved.
    CHECK(scale[0] != doctest::Approx(scale[1]));
}

TEST_CASE("uv-metric: scale does not depend on the mesh_units CLI parameter "
          "(mesh_units is a display-only micrometers-per-OBJ-unit value, not "
          "a scale unit conversion)")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0);

    const fs::path out_u1 = tmp.path / "out_units1.tifxyz";
    const fs::path out_u75 = tmp.path / "out_units75.tifxyz";
    REQUIRE(run_cli({obj.string(), out_u1.string(), "1.0", "1.0", "--uv-metric"}) == 0);
    REQUIRE(run_cli({obj.string(), out_u75.string(), "1.0", "7.5", "--uv-metric"}) == 0);

    const cv::Vec2f s1 = read_meta_scale(out_u1);
    const cv::Vec2f s75 = read_meta_scale(out_u75);

    // Expected values come from the known synthetic vertex spacing
    // (dx=10, dy=25), not from measureGridSpacing() -- an independent oracle.
    CHECK(s1[0] == doctest::Approx(0.1f).epsilon(1e-3));
    CHECK(s1[1] == doctest::Approx(0.04f).epsilon(1e-3));
    CHECK(s75[0] == doctest::Approx(0.1f).epsilon(1e-3));
    CHECK(s75[1] == doctest::Approx(0.04f).epsilon(1e-3));

    // mesh_units must not perturb the written scale at all.
    CHECK(s1[0] == doctest::Approx(s75[0]).epsilon(1e-6));
    CHECK(s1[1] == doctest::Approx(s75[1]).epsilon(1e-6));
}

TEST_CASE("uv-metric: normalized UV in [0,1] (the #1319 failure-mode input "
          "shape) still yields a scale that is the reciprocal of the "
          "emitted grid's physical spacing, not derived from stretch_factor")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    // UV normalized to [0,1] per axis; 3D spacing is still known exactly
    // from the vertex construction (dx=10, dy=25).
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0, /*normalize_uv=*/true);
    const fs::path out = tmp.path / "out.tifxyz";

    // uv_range is 1.0 per axis for normalized UVs, so gw_raw = ceil(uv_range
    // * stretch_factor) + 1 needs stretch_factor > 1 to avoid the degenerate
    // 2x2 grid stretch_factor=1 would give here; 5.0 -> 6x6 (grid sizing
    // itself is unchanged by #1319, so any non-degenerate value works).
    REQUIRE(run_cli({obj.string(), out.string(), "5.0", "1.0", "--uv-metric"}) == 0);

    const cv::Vec2f scale = read_meta_scale(out);
    const GridReadback g = read_grid(out);
    CHECK(g.cols > 2);
    CHECK(g.rows > 2);

    // `scale` must be self-consistent with the grid's own measured physical
    // spacing, not with stretch_factor or the UV range.
    CHECK(scale[0] * g.step[0] == doctest::Approx(1.0f).epsilon(1e-2));
    CHECK(scale[1] * g.step[1] == doctest::Approx(1.0f).epsilon(1e-2));
}

TEST_CASE("uv-metric: scale tracks the FINAL emitted grid, so the decoded "
          "physical extent is the same at different stretch_factors "
          "(round-trip invariant)")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0);

    const fs::path out1 = tmp.path / "out1.tifxyz";
    const fs::path out3 = tmp.path / "out3.tifxyz";
    REQUIRE(run_cli({obj.string(), out1.string(), "1.0", "1.0", "--uv-metric"}) == 0);
    REQUIRE(run_cli({obj.string(), out3.string(), "3.0", "1.0", "--uv-metric"}) == 0);

    const cv::Vec2f s1 = read_meta_scale(out1);
    const cv::Vec2f s3 = read_meta_scale(out3);
    const GridReadback g1 = read_grid(out1);
    const GridReadback g3 = read_grid(out3);

    // Higher stretch_factor -> finer grid -> larger cols/rows and larger
    // density (scale), not the same scale value.
    CHECK(g3.cols > g1.cols);
    CHECK(s3[0] > s1[0]);

    // But the *physical extent* decoded from (grid_size-1)/scale must agree
    // regardless of stretch_factor: 5 intervals * dx=10 -> 50, * dy=25 -> 125.
    const double extent1_x = (g1.cols - 1) / double(s1[0]);
    const double extent3_x = (g3.cols - 1) / double(s3[0]);
    const double extent1_y = (g1.rows - 1) / double(s1[1]);
    const double extent3_y = (g3.rows - 1) / double(s3[1]);
    CHECK(extent1_x == doctest::Approx(50.0).epsilon(1e-2));
    CHECK(extent3_x == doctest::Approx(50.0).epsilon(1e-2));
    CHECK(extent1_y == doctest::Approx(125.0).epsilon(1e-2));
    CHECK(extent3_y == doctest::Approx(125.0).epsilon(1e-2));
}

TEST_CASE("uv-metric: --uv-downsample changes resolution but scale still "
          "matches the coarser grid actually emitted")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 11, 11, /*dx=*/8.0, /*dy=*/8.0, /*z0=*/50.0);

    const fs::path baseline = tmp.path / "baseline.tifxyz";
    const fs::path coarse = tmp.path / "coarse.tifxyz";
    REQUIRE(run_cli({obj.string(), baseline.string(), "1.0", "1.0", "--uv-metric"}) == 0);
    REQUIRE(run_cli({obj.string(), coarse.string(), "1.0", "1.0", "--uv-metric",
                     "--uv-downsample=2"}) == 0);

    const cv::Vec2f sb = read_meta_scale(baseline);
    const cv::Vec2f sc = read_meta_scale(coarse);
    const GridReadback gb = read_grid(baseline);
    const GridReadback gc = read_grid(coarse);

    CHECK(gc.cols < gb.cols);
    // scale must be self-consistent with the grid actually emitted in BOTH
    // cases -- not the pre-downsample resolution.
    CHECK(sb[0] * gb.step[0] == doctest::Approx(1.0f).epsilon(1e-2));
    CHECK(sc[0] * gc.step[0] == doctest::Approx(1.0f).epsilon(1e-2));
    // Decoded physical extent still agrees between the two runs.
    const double extent_b = (gb.cols - 1) / double(sb[0]);
    const double extent_c = (gc.cols - 1) / double(sc[0]);
    CHECK(extent_b == doctest::Approx(extent_c).epsilon(0.05));
}

TEST_CASE("--tifxyz-source: output scale is copied verbatim from the "
          "source, independent of the mesh geometry (non-regression)")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0);

    const fs::path src = tmp.path / "src.tifxyz";
    const cv::Vec2f src_scale(0.037f, 0.091f);
    write_source_tifxyz(src, src_scale);

    const fs::path out = tmp.path / "out.tifxyz";
    REQUIRE(run_cli({obj.string(), out.string(), "--tifxyz-source=" + src.string()}) == 0);

    const cv::Vec2f scale = read_meta_scale(out);
    CHECK(scale[0] == doctest::Approx(src_scale[0]).epsilon(1e-4));
    CHECK(scale[1] == doctest::Approx(src_scale[1]).epsilon(1e-4));
}

TEST_CASE("uv-non-metric (legacy): scale is still the reciprocal of the "
          "actual emitted grid spacing")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_grid_obj(obj, 6, 6, /*dx=*/10.0, /*dy=*/25.0, /*z0=*/100.0);
    const fs::path out = tmp.path / "out.tifxyz";

    REQUIRE(run_cli({obj.string(), out.string(), "1.0", "1.0", "--uv-non-metric"}) == 0);

    const cv::Vec2f scale = read_meta_scale(out);
    const GridReadback g = read_grid(out);

    CHECK(scale[0] * g.step[0] == doctest::Approx(1.0f).epsilon(1e-2));
    CHECK(scale[1] * g.step[1] == doctest::Approx(1.0f).epsilon(1e-2));
}

#endif  // _WIN32
