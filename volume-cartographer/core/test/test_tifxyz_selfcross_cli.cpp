// End-to-end tests for the vc_tifxyz_selfcross command line: exit codes,
// report serialization, and the point-collection output. The kernel has its
// own fixtures in test_tifxyz_selfcross.cpp; this file covers what only the
// binary does -- argument validation, the report contract, the collection
// envelope, and the --fail-on-crossing gate scripts rely on.
//
// The binary path arrives from CMake as VC_SELFCROSS_BIN. Fixtures are tiny
// tifxyz directories written with OpenCV's TIFF encoder into a fresh temp
// directory, the same three-plane + meta.json layout the loader reads.

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

// A grid whose upper block is a flat plane and, when `crossing` is set,
// whose lower block is a tilted sheet driven through it; the two blocks are
// separated by four invalid rows so every contact is non-adjacent.
void write_tifxyz(const fs::path& dir, bool crossing)
{
    const int rows = crossing ? 24 : 10, cols = 12;
    cv::Mat x(rows, cols, CV_32F, cv::Scalar(-1.f));
    cv::Mat y(rows, cols, CV_32F, cv::Scalar(-1.f));
    cv::Mat z(rows, cols, CV_32F, cv::Scalar(-1.f));
    for (int v = 0; v < 10; ++v)
        for (int u = 0; u < cols; ++u) {
            x.at<float>(v, u) = 4.f * u;
            y.at<float>(v, u) = 4.f * v;
            z.at<float>(v, u) = 5.f;
        }
    if (crossing)
        for (int v = 0; v < 10; ++v)
            for (int u = 0; u < cols; ++u) {
                x.at<float>(14 + v, u) = 4.f * u + 0.3f * v + 1.7f;
                y.at<float>(14 + v, u) = 4.f * v;
                z.at<float>(14 + v, u) = (float)(4.0 * (u - 5) + 7.0);
            }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), z));
    std::ofstream meta(dir / "meta.json");
    // The loader requires both fields; uuid must be a string.
    meta << "{\"scale\": [1.0, 1.0], \"uuid\": \"cli-fixture\"}\n";
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
    std::string cmd = sh(VC_SELFCROSS_BIN);
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
             / ("vc_selfcross_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

}  // namespace

TEST_CASE("clean surface: exit 0 even under --fail-on-crossing, report "
          "says clean")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "clean.tifxyz";
    write_tifxyz(surf, false);
    const fs::path report = tmp.path / "report.json";
    CHECK(run_cli({surf.string(), "-o", report.string(),
                   "--fail-on-crossing"}) == 0);
    utils::Json j = utils::Json::parse_file(report);
    CHECK(j["clean_of_transverse_self_intersection"].get_bool());
    CHECK(j["report_only"].get_bool());
    for (size_t d = 0; d < 2; ++d) {
        CHECK(j["census"][d]["transverse"].get_int() == 0);
        CHECK(j["census"][d]["triangles"].get_int() > 0);
    }
}

TEST_CASE("crossing surface: exit 3 under --fail-on-crossing, 0 without, "
          "contacts serialized, collection loadable")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "cross.tifxyz";
    write_tifxyz(surf, true);
    const fs::path report = tmp.path / "report.json";
    const fs::path coll = tmp.path / "sites.json";

    CHECK(run_cli({surf.string(), "-o", report.string(),
                   "--collection", coll.string(),
                   "--fail-on-crossing"}) == 3);
    CHECK(run_cli({surf.string(), "-o", report.string()}) == 0);

    utils::Json j = utils::Json::parse_file(report);
    CHECK(!j["clean_of_transverse_self_intersection"].get_bool());
    bool any = false;
    for (size_t d = 0; d < 2; ++d) {
        const utils::Json& jd = j["census"][d];
        if (jd["transverse"].get_int() == 0) continue;
        any = true;
        const utils::Json& rows = jd["transverse_contacts"];
        CHECK((int64_t)rows.size() == jd["transverse"].get_int());
        const utils::Json& first = rows[size_t(0)];
        CHECK(first["site"].size() == 3);
        CHECK(first["penetration_vx"].get_float() > 0.0);
    }
    CHECK(any);

    // The collection must carry the envelope the point-collection loader
    // demands; without it the file is silently unloadable in the viewer.
    utils::Json c = utils::Json::parse_file(coll);
    CHECK(c["vc_pointcollections_json_version"].get_string() == "1");
    CHECK(c["collections"].size() > 0);
}

TEST_CASE("bad arguments and unwritable outputs exit 1, not 0")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "clean.tifxyz";
    write_tifxyz(surf, false);
    const std::string out = (tmp.path / "r.json").string();
    CHECK(run_cli({surf.string(), "-o", out, "--cell", "0"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--cell", "-3"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--cell", "1e-300"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--exclude", "-1"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--maxedge", "-1"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--maxedge", "nan"}) == 1);
    CHECK(run_cli({surf.string(), "-o", "/nonexistent-dir/r.json"}) == 1);
    CHECK(run_cli({(tmp.path / "missing.tifxyz").string(), "-o", out}) == 1);
    // Failure DURING the write, not at open: /dev/full accepts the open and
    // fails the flush. "Report written" plus exit 0 here would defeat the
    // one job a report-only tool has.
    if (fs::exists("/dev/full")) {
        CHECK(run_cli({surf.string(), "-o", "/dev/full"}) == 1);
        CHECK(run_cli({surf.string(), "-o", out,
                       "--collection", "/nonexistent-dir/c.json"}) == 1);
        // Mid-write failure on the COLLECTION path: needs the surface to
        // actually have crossing sites, and saveToJSON to check its stream
        // rather than only the open.
        const fs::path xsurf = tmp.path / "x.tifxyz";
        write_tifxyz(xsurf, true);
        CHECK(run_cli({xsurf.string(), "-o", out,
                       "--collection", "/dev/full"}) == 1);
    }
}

TEST_CASE("reports are byte-identical across runs and thread counts")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "cross.tifxyz";
    write_tifxyz(surf, true);
    const fs::path a = tmp.path / "a.json", b = tmp.path / "b.json";
    REQUIRE(run_cli({surf.string(), "-o", a.string()}) == 0);
    REQUIRE(run_cli({surf.string(), "-o", b.string(), "--threads", "1"})
            == 0);
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    std::string sa((std::istreambuf_iterator<char>(fa)), {});
    std::string sb((std::istreambuf_iterator<char>(fb)), {});
    CHECK(sa == sb);
    CHECK(!sa.empty());
}

#endif  // _WIN32
