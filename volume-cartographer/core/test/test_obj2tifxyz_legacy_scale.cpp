// End-to-end test for vc_obj2tifxyz_legacy's output grid on a synthetic mesh.
//
// vc_merge_tifxyz rasterizes its merged OBJ with vc_obj2tifxyz_legacy, whose
// determineGridDimensions sizes the grid from the UV range times an estimate of
// the UV-to-3D scale. A few stretched triangles (blend and seam triangles of a
// merge) must not change that estimate. The test writes two OBJs of a plane on
// a 4 voxel lattice with vt = cell * 20, as vc_merge_tifxyz writes them: a clean
// one, and one where 1 per cent of the vertices (rows and columns 5 mod 10) are
// pushed 9.8 * 4 voxels out of the plane, so 3 per cent of the triangles have a
// Jacobian norm about 6.9 times the median. It rasterizes both with step 4 and
// requires the median distance between 4-neighbour cells of the output to be
// within 1 per cent of 4 voxels.
//
// Opt-in: only runs when VC_RUN_E2E is set to "1", like test_merge_e2e_small.
// POSIX only, like the other CLI tests: the tool is started with posix_spawn and
// an argument vector, never through a shell.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
TEST_CASE("vc_obj2tifxyz_legacy scale test is POSIX-only") {}
#else

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>

extern char** environ;

namespace fs = std::filesystem;

namespace {

constexpr int kN = 200;
constexpr double kStep = 4.0;   // voxels per lattice cell
constexpr double kUv = 20.0;    // vt units per lattice cell, as vc_merge_tifxyz writes them

void writePlaneObj(const fs::path& path, bool stretched)
{
    std::ofstream f(path);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j) {
            double z = 500.0;
            if (stretched && i % 10 == 5 && j % 10 == 5) z += 9.8 * kStep;
            f << "v " << 1000.0 + j * kStep << " " << 1000.0 + i * kStep << " " << z << "\n";
        }
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            f << "vt " << j * kUv << " " << i * kUv << "\n";
    auto id = [](int i, int j) { return i * kN + j + 1; };
    for (int i = 0; i + 1 < kN; ++i)
        for (int j = 0; j + 1 < kN; ++j) {
            const int a = id(i, j), b = id(i, j + 1), c = id(i + 1, j), d = id(i + 1, j + 1);
            f << "f " << a << "/" << a << " " << b << "/" << b << " " << d << "/" << d << "\n";
            f << "f " << a << "/" << a << " " << d << "/" << d << " " << c << "/" << c << "\n";
        }
}

double medianNeighbourStep(const fs::path& dir)
{
    auto surf = load_quad_from_tifxyz(dir);
    REQUIRE(surf);
    const cv::Mat_<cv::Vec3f> p = surf->rawPoints();
    auto valid = [](const cv::Vec3f& v) { return v[0] != -1.f && std::isfinite(v[0]); };
    std::vector<double> d;
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.cols; ++c) {
            if (!valid(p(r, c))) continue;
            if (c + 1 < p.cols && valid(p(r, c + 1))) d.push_back(cv::norm(p(r, c + 1) - p(r, c)));
            if (r + 1 < p.rows && valid(p(r + 1, c))) d.push_back(cv::norm(p(r + 1, c) - p(r, c)));
        }
    REQUIRE(!d.empty());
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return d[d.size() / 2];
}

fs::path findBinary()
{
    if (const char* env = std::getenv("VC_OBJ2TIFXYZ_LEGACY_BIN"))
        if (fs::is_regular_file(env)) return env;
    for (const fs::path& base : {fs::path("build/bin"), fs::path("bin"), fs::path("../bin")})
        if (fs::is_regular_file(base / "vc_obj2tifxyz_legacy")) return base / "vc_obj2tifxyz_legacy";
    return {};
}

// Runs bin with the arguments obj, out, "4" and no shell: the path and the
// arguments go to posix_spawn as an argument vector, stdout and stderr go to log.
// Returns the exit status, or -1 if the process could not be started or did
// not exit normally.
int runTool(const fs::path& bin, const fs::path& obj, const fs::path& out, const fs::path& log)
{
    std::string a0 = bin.string(), a1 = obj.string(), a2 = out.string(), a3 = "4";
    std::vector<char*> argv = {a0.data(), a1.data(), a2.data(), a3.data(), nullptr};
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return -1;
    const std::string logPath = log.string();
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, a0.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

double rasterizedStep(const fs::path& bin, const fs::path& root, bool stretched)
{
    const std::string name = stretched ? "stretched" : "clean";
    const fs::path obj = root / (name + ".obj");
    const fs::path out = root / name;
    writePlaneObj(obj, stretched);
    REQUIRE_MESSAGE(runTool(bin, obj, out, root / (name + ".log")) == 0,
                    "vc_obj2tifxyz_legacy failed on the " << name << " mesh");
    return medianNeighbourStep(out);
}

}  // namespace

TEST_CASE("vc_obj2tifxyz_legacy keeps the cell size when a few triangles are stretched")
{
    const char* run = std::getenv("VC_RUN_E2E");
    if (!run || std::string(run) != "1") {
        MESSAGE("VC_RUN_E2E != 1; skipping (set VC_RUN_E2E=1 to enable)");
        return;
    }
    const fs::path bin = findBinary();
    REQUIRE_MESSAGE(!bin.empty(), "vc_obj2tifxyz_legacy not found; build it or set VC_OBJ2TIFXYZ_LEGACY_BIN");
    std::mt19937_64 rng(std::random_device{}());
    const fs::path root = fs::temp_directory_path() / ("vc_obj2tifxyz_legacy_scale_" + std::to_string(rng()));
    fs::create_directories(root);

    const double clean = rasterizedStep(bin, root, false);
    const double stretched = rasterizedStep(bin, root, true);
    MESSAGE("output step, clean mesh: " << clean << " voxels; stretched mesh: " << stretched << " voxels (expected 4)");
    CHECK(std::abs(clean - kStep) <= 0.01 * kStep);
    CHECK(std::abs(stretched - kStep) <= 0.01 * kStep);

    std::error_code ec;
    fs::remove_all(root, ec);
}

#endif  // _WIN32
