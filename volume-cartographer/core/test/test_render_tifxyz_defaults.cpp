// Regression test for the option defaults of vc_render_tifxyz.
//
// --voxel-unit used to default to "nanometer" while every physical size in
// this project is in micrometers (volpkg meta.json voxelsize, the docs, the
// volume names). Every OME-Zarr written on the default path therefore declared
// a scale 1000x too small, silently, which is the metadata viewers read to draw
// scale bars.
//
// The fix is a one-word default, so the failure mode is a silent revert: the
// code keeps compiling and the output keeps looking plausible. This test pins
// the value by reading it back out of --help, which boost::program_options
// renders as "(=micrometer)".
//
// Deliberately hermetic: no volume, no surface, no fixtures. It spawns the
// built binary and inspects its help text, so it runs in the default ctest
// pass instead of being gated behind VC_RUN_E2E like the heavier end-to-end
// tests. A guard that nobody runs guards nothing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace
{

fs::path locateBinary(const fs::path& candidate)
{
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) return candidate;
    return {};
}

// Mirrors the probe in test_merge_e2e_small.cpp: CMake passes the exact path
// via the environment, the rest are fallbacks for a manual run from a source
// tree or from PATH.
fs::path findRenderTifxyz()
{
    if (const char* env = std::getenv("VC_RENDER_TIFXYZ_BIN")) {
        if (auto p = locateBinary(env); !p.empty()) return p;
    }

#ifdef _WIN32
    const std::string exeName = "vc_render_tifxyz.exe";
    const char pathSep = ';';
#else
    const std::string exeName = "vc_render_tifxyz";
    const char pathSep = ':';
#endif

    for (const fs::path& base : {fs::path("build/bin"),
                                 fs::path("build-macos/bin"),
                                 fs::path("build-macos-rel/bin")}) {
        if (auto p = locateBinary(base / exeName); !p.empty()) return p;
    }

    if (const char* path = std::getenv("PATH")) {
        std::string s = path;
        std::string::size_type from = 0;
        while (from <= s.size()) {
            const auto next = s.find(pathSep, from);
            const std::string seg =
                s.substr(from, next == std::string::npos ? std::string::npos : next - from);
            if (!seg.empty()) {
                if (auto p = locateBinary(fs::path(seg) / exeName); !p.empty()) return p;
            }
            if (next == std::string::npos) break;
            from = next + 1;
        }
    }
    return {};
}

std::string runHelp(const fs::path& bin, int& exitCode)
{
    const fs::path out = fs::temp_directory_path() /
                         ("vc_render_tifxyz_help_" + std::to_string(std::rand()) + ".txt");

    // Quoted so a path with spaces survives the shell on both platforms.
    std::string cmd = "\"" + bin.string() + "\" --help > \"" + out.string() + "\" 2>&1";
#ifdef _WIN32
    // std::system goes through `cmd /c`, which strips the outer quote pair when
    // the string both starts with a quote and contains others — the command
    // then parses as garbage. Wrapping the whole line in one more pair is the
    // documented workaround.
    cmd = "\"" + cmd + "\"";
#endif
    exitCode = std::system(cmd.c_str());

    std::ifstream in(out);
    std::ostringstream ss;
    ss << in.rdbuf();
    in.close();
    std::error_code ec;
    fs::remove(out, ec);
    return ss.str();
}

}  // namespace

TEST_CASE("vc_render_tifxyz --help reports the option defaults")
{
    const fs::path bin = findRenderTifxyz();
    REQUIRE_MESSAGE(!bin.empty(),
                    "vc_render_tifxyz not found; set VC_RENDER_TIFXYZ_BIN to its path");

    int exitCode = -1;
    const std::string help = runHelp(bin, exitCode);

    CHECK(exitCode == 0);
    REQUIRE_MESSAGE(!help.empty(), "--help produced no output");

    SUBCASE("the option is still advertised")
    {
        CHECK(help.find("--voxel-unit") != std::string::npos);
    }

    SUBCASE("it defaults to micrometer, matching every other physical size in the project")
    {
        CHECK(help.find("(=micrometer)") != std::string::npos);
    }

    SUBCASE("it does not default to nanometer")
    {
        // The exact regression: a nanometer default makes every OME-Zarr
        // written on the default path declare a scale 1000x too small.
        CHECK(help.find("(=nanometer)") == std::string::npos);
    }
}
