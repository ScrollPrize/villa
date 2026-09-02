// End-to-end tests for the vc_obj2tifxyz command line: the tifxyz `scale`
// it writes must describe the grid it actually emitted (#1319). QuadSurface
// reads scale as grid cells per voxel (size() == cols / scale), so the 3D
// spacing of adjacent points in the written x/y/z.tif must equal 1 / scale
// and the nominal size must stay the mesh's extent whatever stretch,
// decimation or cap produced the grid, and whether the UVs were metric or
// normalized. Before the fix, scale was the UV spacing of the undecimated
// grid (a length, inverted, and blind to --uv-downsample / --grid-cap), so a
// 20x-decimated surface rendered 20x too small and a stretched one 4x too
// large.
//
// The binary path arrives from CMake as VC_OBJ2TIFXYZ_BIN. The fixture is a
// flat mesh written into a fresh temp directory in the layout vc_tifxyz2obj
// emits (v/vt/vn, f v/vt/vn), with UVs equal to the in-plane OBJ coordinates
// or, for the normalized case, divided by the extent.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
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

// Mesh vertex spacing and vertex counts: a 40 x 30 quad grid of 20-unit
// cells, i.e. UV extent 800 x 600 OBJ units (the geometry of a tracer
// segment written through vc_tifxyz2obj).
constexpr float kSpacing = 20.f;
constexpr int kCols = 41;
constexpr int kRows = 31;
constexpr float kExtentU = kSpacing * (kCols - 1);
constexpr float kExtentV = kSpacing * (kRows - 1);

void write_obj(const fs::path& path, bool normalized_uv = false)
{
    std::ofstream obj(path);
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            const float x = kSpacing * c, y = kSpacing * r;
            const float u = normalized_uv ? x / kExtentU : x;
            const float v = normalized_uv ? y / kExtentV : y;
            obj << "v " << x << " " << y << " 100\n"
                << "vt " << u << " " << v << "\n"
                << "vn 0 0 1\n";
        }
    for (int r = 0; r + 1 < kRows; ++r)
        for (int c = 0; c + 1 < kCols; ++c) {
            const int a = r * kCols + c + 1, b = a + 1;
            const int d = a + kCols, e = d + 1;
            obj << "f " << a << "/" << a << "/" << a << " " << b << "/" << b << "/" << b
                << " " << e << "/" << e << "/" << e << "\n";
            obj << "f " << a << "/" << a << "/" << a << " " << e << "/" << e << "/" << e
                << " " << d << "/" << d << "/" << d << "\n";
        }
    REQUIRE(obj.good());
}

// A tifxyz with the same geometry at the tracer's 1 cell / 20 units, for
// --tifxyz-source mode.
void write_source_tifxyz(const fs::path& dir)
{
    cv::Mat x(kRows, kCols, CV_32F), y(kRows, kCols, CV_32F), z(kRows, kCols, CV_32F, cv::Scalar(100.f));
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            x.at<float>(r, c) = kSpacing * c;
            y.at<float>(r, c) = kSpacing * r;
        }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), z));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"scale\": [0.05, 0.05], \"uuid\": \"cli-source\"}\n";
    REQUIRE(meta.good());
}

// Single-quote every argument so paths containing spaces or shell
// metacharacters cannot break the command or execute as syntax.
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
             / ("vc_obj2tifxyz_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

// Convert with `extra` flags and check the written surface: the grid has the
// expected column count, scale is cells per voxel of that grid (so the
// nominal size stays the mesh extent), and the 3D spacing of adjacent grid
// points read back from x/y/z.tif equals 1 / scale.
void check_conversion(const TempDir& tmp, const std::string& name,
                      const std::vector<std::string>& extra, int expect_cols)
{
    const fs::path obj = tmp.path / "mesh.obj";
    const fs::path out = tmp.path / name;  // must not pre-exist
    std::vector<std::string> args{obj.string(), out.string()};
    args.insert(args.end(), extra.begin(), extra.end());
    REQUIRE(run_cli(args) == 0);

    auto surf = load_quad_from_tifxyz(out);
    REQUIRE(surf);
    const cv::Mat_<cv::Vec3f> pts = surf->rawPoints();
    const cv::Vec2f scale = surf->scale();
    INFO(name << ": grid " << pts.cols << "x" << pts.rows
              << " scale " << scale[0] << "," << scale[1]);
    CHECK(pts.cols == expect_cols);
    CHECK(scale[0] == doctest::Approx((pts.cols - 1) / kExtentU).epsilon(1e-4));
    CHECK(scale[1] == doctest::Approx((pts.rows - 1) / kExtentV).epsilon(1e-4));
    // size() is cols / scale, so it counts cells rather than intervals and
    // may exceed the UV extent by one cell; before the fix it was off by
    // the whole decimation factor.
    const cv::Size nominal = surf->size();
    CHECK(std::abs(nominal.width - kExtentU) <= 1.0 / scale[0] + 0.5);
    CHECK(std::abs(nominal.height - kExtentV) <= 1.0 / scale[1] + 0.5);

    const int r = pts.rows / 2, c = pts.cols / 2;
    const double du = cv::norm(pts(r, c + 1) - pts(r, c));
    const double dv = cv::norm(pts(r + 1, c) - pts(r, c));
    CHECK(du == doctest::Approx(1.0 / scale[0]).epsilon(1e-3));
    CHECK(dv == doctest::Approx(1.0 / scale[1]).epsilon(1e-3));
}

}  // namespace

TEST_CASE("metric mode: scale follows the emitted grid under decimation, "
          "cap and stretch")
{
    TempDir tmp;
    write_obj(tmp.path / "mesh.obj");

    // 1 cell / OBJ unit: 801 columns, scale 1 (unchanged behaviour).
    check_conversion(tmp, "plain", {}, 801);
    // --uv-downsample=20 keeps the endpoints: 1 + 800/20 columns, scale 0.05.
    check_conversion(tmp, "ds20", {"--uv-downsample=20"}, 41);
    // --grid-cap forces ceil(sqrt(801*601/50000)) = 4x decimation.
    check_conversion(tmp, "cap", {"--grid-cap=50000"}, 201);
    // stretch_factor 2: 2 cells / OBJ unit, scale 2.
    check_conversion(tmp, "stretch2", {"2"}, 1601);
}

TEST_CASE("normalized UVs: scale comes from the emitted grid's 3D spacing")
{
    TempDir tmp;
    write_obj(tmp.path / "mesh.obj", /*normalized_uv=*/true);
    // stretch 40 on a unit UV square: a 41x41 grid over 800x600 voxels, so
    // an anisotropic scale of 0.05 x 0.0667 (#1319 wrote 1 / stretch).
    check_conversion(tmp, "norm40", {"40"}, 41);
}

TEST_CASE("source-scale mode: scale is the source's, reduced by any extra "
          "decimation")
{
    TempDir tmp;
    write_obj(tmp.path / "mesh.obj");
    const fs::path src = tmp.path / "source.tifxyz";
    write_source_tifxyz(src);

    // Sized to the source density: 41 columns at the source's 0.05.
    check_conversion(tmp, "src", {"--tifxyz-source=" + src.string()}, 41);
    // A further 4x decimation: 11 columns, scale 0.05 * 10/40.
    check_conversion(tmp, "src_ds4",
                     {"--tifxyz-source=" + src.string(), "--uv-downsample=4"}, 11);
}

#endif  // _WIN32
