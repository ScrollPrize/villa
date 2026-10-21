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
// each scale method, and checks the median distance between 4-neighbour cells of
// the output: within 1 per cent of 4 voxels for the median (default) and for a
// manual --uv-scale=0.2 on both meshes, and for --scale-method=mean on the clean
// mesh; on the stretched mesh the mean keeps its old bias (under 3.6 voxels).
// Invalid scale options must be refused with a non-zero exit before the output
// directory is created.
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
#include <unistd.h>

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

bool isExecutableFile(const fs::path& p)
{
    std::error_code ec;
    return fs::is_regular_file(p, ec) && ::access(p.c_str(), X_OK) == 0;
}

// The tool to spawn: VC_OBJ2TIFXYZ_LEGACY_BIN if set, which must be an absolute
// path to an existing executable regular file (refused otherwise, never run), or
// the built binary next to the test. Empty if none qualifies.
fs::path findBinary()
{
    if (const char* env = std::getenv("VC_OBJ2TIFXYZ_LEGACY_BIN")) {
        const fs::path p(env);
        if (!p.is_absolute() || !isExecutableFile(p)) {
            MESSAGE("VC_OBJ2TIFXYZ_LEGACY_BIN refused (not an absolute path to an executable file): " << env);
            return {};
        }
        return p;
    }
    for (const fs::path& base : {fs::path("build/bin"), fs::path("bin"), fs::path("../bin")})
        if (isExecutableFile(base / "vc_obj2tifxyz_legacy")) return fs::absolute(base / "vc_obj2tifxyz_legacy");
    return {};
}

// Runs bin with the arguments obj, out, "4" and extra, and no shell: the path and
// the arguments go to posix_spawn as an argument vector, stdout and stderr go to
// log. Returns the exit status, or -1 if the process could not be started or did
// not exit normally.
int runTool(const fs::path& bin, const fs::path& obj, const fs::path& out, const fs::path& log,
            std::vector<std::string> extra)
{
    std::vector<std::string> args = {bin.string(), obj.string(), out.string(), "4"};
    for (auto& e : extra)
        if (!e.empty()) args.push_back(e);
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return -1;
    const std::string logPath = log.string();
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, args[0].c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Rasterizes the clean or stretched OBJ (written once into root) with the extra
// argument (empty: default method) and returns the output step in voxels.
double rasterizedStep(const fs::path& bin, const fs::path& root, bool stretched,
                      const std::string& tag, const std::string& extra)
{
    const std::string mesh = stretched ? "stretched" : "clean";
    const fs::path obj = root / (mesh + ".obj");
    if (!fs::exists(obj)) writePlaneObj(obj, stretched);
    const fs::path out = root / (mesh + "_" + tag);
    REQUIRE_MESSAGE(runTool(bin, obj, out, root / (mesh + "_" + tag + ".log"), {extra}) == 0,
                    "vc_obj2tifxyz_legacy " << extra << " failed on the " << mesh << " mesh");
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

    // tag, extra argument, whether the stretched mesh must keep the 4 voxel step
    const struct { const char* tag; const char* extra; bool robust; } methods[] = {
        {"median", "", true},                     // default
        {"mean", "--scale-method=mean", false},   // the behaviour before the median
        {"manual", "--uv-scale=0.2", true},       // vt = cell * 20 on a 4 voxel lattice
    };
    for (const auto& m : methods) {
        const double clean = rasterizedStep(bin, root, false, m.tag, m.extra);
        const double stretched = rasterizedStep(bin, root, true, m.tag, m.extra);
        MESSAGE(m.tag << ": output step, clean mesh: " << clean << " voxels; stretched mesh: " << stretched
                      << " voxels");
        CHECK(std::abs(clean - kStep) <= 0.01 * kStep);
        if (m.robust)
            CHECK(std::abs(stretched - kStep) <= 0.01 * kStep);
        else
            CHECK(stretched < 0.9 * kStep);
    }

    // invalid scale options: refused, and nothing written
    const std::vector<std::vector<std::string>> refused = {
        {"--uv-scale=abc"}, {"--uv-scale=0"}, {"--uv-scale=-1"},
        {"--uv-scale=0.2", "--scale-method=mean"}, {"--scale-method=foo"},
    };
    for (size_t i = 0; i < refused.size(); ++i) {
        const fs::path out = root / ("refused_" + std::to_string(i));
        const int rc = runTool(bin, root / "clean.obj", out, root / ("refused_" + std::to_string(i) + ".log"), refused[i]);
        CHECK_MESSAGE(rc > 0, refused[i][0] << " not refused (rc " << rc << ")");
        CHECK_MESSAGE(!fs::exists(out), refused[i][0] << " created " << out);
    }

    std::error_code ec;
    fs::remove_all(root, ec);
}

#endif  // _WIN32
