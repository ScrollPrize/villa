// End-to-end regression for vc_obj2tifxyz meta.json `scale` (#1319).
// `scale` is grid cells per volume voxel, derived from the emitted grid.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "utils/Json.hpp"

#include <array>
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

// A 6x6 mesh with normalized UVs.  stretch_factor=5 emits a 6x6 grid, so
// its grid-cell spacing is known from the fixture: dx=10, dy=25.
void write_normalized_uv_grid(const fs::path& path)
{
    constexpr int size = 6;
    std::ofstream f(path);
    REQUIRE(f.good());
    for (int row = 0; row < size; ++row)
        for (int col = 0; col < size; ++col)
            f << "v " << (col * 10) << " " << (row * 25) << " 100\n";
    for (int row = 0; row < size; ++row)
        for (int col = 0; col < size; ++col)
            f << "vt " << (double(col) / (size - 1)) << " "
              << (double(row) / (size - 1)) << "\n";
    auto idx = [](int row, int col) { return row * 6 + col + 1; };
    for (int row = 0; row < size - 1; ++row)
        for (int col = 0; col < size - 1; ++col) {
            const int a = idx(row, col), b = idx(row, col + 1);
            const int c = idx(row + 1, col + 1), d = idx(row + 1, col);
            f << "f " << a << "/" << a << " " << b << "/" << b << " " << c << "/" << c << "\n";
            f << "f " << a << "/" << a << " " << c << "/" << c << " " << d << "/" << d << "\n";
        }
    REQUIRE(f.good());
}

std::array<float, 2> read_meta_scale(const fs::path& dir)
{
    utils::Json j = utils::Json::parse_file(dir / "meta.json");
    return {j["scale"][size_t(0)].get_float(), j["scale"][size_t(1)].get_float()};
}

}  // namespace

TEST_CASE("uv-metric normalized-UV scale uses emitted grid-cell spacing and ignores mesh_units")
{
    TempDir tmp;
    const fs::path obj = tmp.path / "mesh.obj";
    write_normalized_uv_grid(obj);
    const fs::path out_u1 = tmp.path / "out_units1.tifxyz";
    const fs::path out_u75 = tmp.path / "out_units75.tifxyz";

    REQUIRE(run_cli({obj.string(), out_u1.string(), "5.0", "1.0", "--uv-metric"}) == 0);
    REQUIRE(run_cli({obj.string(), out_u75.string(), "5.0", "7.5", "--uv-metric"}) == 0);

    const auto s1 = read_meta_scale(out_u1);
    const auto s75 = read_meta_scale(out_u75);
    // Fixture spacing is dx=10 and dy=25, so density is [1/10, 1/25].
    CHECK(s1[0] == doctest::Approx(0.1f).epsilon(1e-3));
    CHECK(s1[1] == doctest::Approx(0.04f).epsilon(1e-3));
    CHECK(s75[0] == doctest::Approx(s1[0]).epsilon(1e-6));
    CHECK(s75[1] == doctest::Approx(s1[1]).epsilon(1e-6));
}

#endif  // _WIN32
