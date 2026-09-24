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

enum Shape { PLAIN, ISLANDED, HOLED };

void write_tifxyz(const fs::path& dir, Shape shape)
{
    const int rows = 24, cols = 14;
    cv::Mat x(rows, cols, CV_32F, cv::Scalar(-1.f));
    cv::Mat y(rows, cols, CV_32F, cv::Scalar(-1.f));
    cv::Mat z(rows, cols, CV_32F, cv::Scalar(-1.f));
    auto put = [&](int row0, int nrows) {
        for (int v = 0; v < nrows; ++v)
            for (int u = 1; u < cols - 1; ++u) {
                x.at<float>(row0 + v, u) = 4.f * u;
                y.at<float>(row0 + v, u) = 4.f * (row0 + v);
                z.at<float>(row0 + v, u) = 5.f;
            }
    };
    put(1, 10);
    if (shape == ISLANDED) put(16, 6);
    if (shape == HOLED) {
        x.at<float>(5, 6) = -1.f;
        y.at<float>(5, 6) = -1.f;
        z.at<float>(5, 6) = -1.f;
    }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), z));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"scale\": [1.0, 1.0], \"uuid\": \"cli-fixture\"}\n";
    REQUIRE(meta.good());
}

std::string sh(const std::string& s)
{
    std::string q = "'";
    for (char c : s)
        if (c == '\'') q += "'\\''"; else q += c;
    return q + "'";
}

int run_cli(const std::vector<std::string>& args)
{
    std::string cmd = sh(VC_TOPOLOGY_BIN);
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
             / ("vc_topology_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

}  // namespace

TEST_CASE("a plain surface exits 0 even under --fail-on any")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "plain.tifxyz";
    write_tifxyz(surf, PLAIN);
    const fs::path report = tmp.path / "report.json";
    CHECK(run_cli({surf.string(), "-o", report.string(),
                   "--fail-on", "any"}) == 0);
    utils::Json j = utils::Json::parse_file(report);
    CHECK(j["report_only"].get_bool());
    CHECK(j["summary"]["censused"].get_int() == 1);
    CHECK(j["summary"]["with_islands"].get_int() == 0);
    CHECK(j["summary"]["flagged"].size() == 0);
    CHECK(j["surfaces"][size_t(0)]["islands"]["components"].get_int() == 1);
    CHECK(j["surfaces"][size_t(0)]["valid_quads"].get_int() > 0);
}

TEST_CASE("an islanded surface is flagged, and --fail-on gates on the named "
          "class only")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "islands.tifxyz";
    write_tifxyz(surf, ISLANDED);
    const std::string out = (tmp.path / "r.json").string();
    CHECK(run_cli({surf.string(), "-o", out}) == 0);
    CHECK(run_cli({surf.string(), "-o", out, "--fail-on", "islands"}) == 3);
    CHECK(run_cli({surf.string(), "-o", out, "--fail-on", "holes"}) == 0);
    CHECK(run_cli({surf.string(), "-o", out, "--fail-on", "any"}) == 3);

    utils::Json j = utils::Json::parse_file(out);
    CHECK(j["surfaces"][size_t(0)]["islands"]["components"].get_int() == 2);
    CHECK(j["surfaces"][size_t(0)]["islands"]["island_quads"].get_int() > 0);
    REQUIRE(j["summary"]["flagged"].size() == 1);
    CHECK(j["summary"]["flagged"][size_t(0)]["classes"][size_t(0)]
              .get_string() == "islands");
}

TEST_CASE("several surfaces are censused in one run and summarized")
{
    TempDir tmp;
    const fs::path a = tmp.path / "a.tifxyz";
    const fs::path b = tmp.path / "b.tifxyz";
    const fs::path c = tmp.path / "c.tifxyz";
    write_tifxyz(a, PLAIN);
    write_tifxyz(b, ISLANDED);
    write_tifxyz(c, HOLED);
    const fs::path report = tmp.path / "r.json";
    const fs::path coll = tmp.path / "sites.json";
    CHECK(run_cli({a.string(), b.string(), c.string(),
                   "-o", report.string(), "--collection", coll.string()})
          == 0);
    utils::Json j = utils::Json::parse_file(report);
    CHECK(j["summary"]["surfaces"].get_int() == 3);
    CHECK(j["summary"]["censused"].get_int() == 3);
    CHECK(j["summary"]["with_islands"].get_int() == 1);
    CHECK(j["summary"]["with_holes"].get_int() == 1);
    CHECK(j["summary"]["flagged"].size() == 2);
    CHECK(j["surfaces"][size_t(0)]["surface"].get_string() == a.string());
    CHECK(j["surfaces"][size_t(2)]["surface"].get_string() == c.string());

    utils::Json cj = utils::Json::parse_file(coll);
    CHECK(cj["vc_pointcollections_json_version"].get_string() == "1");
    CHECK(cj["collections"].size() > 0);
}

TEST_CASE("one unreadable surface exits 1 without discarding the others")
{
    TempDir tmp;
    const fs::path good = tmp.path / "good.tifxyz";
    write_tifxyz(good, PLAIN);
    const fs::path report = tmp.path / "r.json";
    CHECK(run_cli({(tmp.path / "missing.tifxyz").string(), good.string(),
                   "-o", report.string()}) == 1);
    utils::Json j = utils::Json::parse_file(report);
    CHECK(j["summary"]["failed_to_load"].get_int() == 1);
    CHECK(j["summary"]["censused"].get_int() == 1);
    CHECK(j["surfaces"][size_t(0)].contains("error"));
    CHECK(j["surfaces"][size_t(1)]["valid_quads"].get_int() > 0);
}

TEST_CASE("bad arguments and unwritable outputs exit 1, not 0")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "plain.tifxyz";
    write_tifxyz(surf, PLAIN);
    const std::string out = (tmp.path / "r.json").string();
    CHECK(run_cli({surf.string(), "-o", out, "--tear-factor", "1"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--tear-factor", "0.5"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--tear-factor", "nan"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--max-sites", "-1"}) == 1);
    CHECK(run_cli({surf.string(), "-o", out, "--fail-on", "island"}) == 1);
    CHECK(run_cli({surf.string(), "-o", "/nonexistent-dir/r.json"}) == 1);
    if (fs::exists("/dev/full")) {
        CHECK(run_cli({surf.string(), "-o", "/dev/full"}) == 1);
        const fs::path isl = tmp.path / "islands.tifxyz";
        write_tifxyz(isl, ISLANDED);
        CHECK(run_cli({isl.string(), "-o", out,
                       "--collection", "/dev/full"}) == 1);
    }
}

TEST_CASE("reports are byte-identical across runs")
{
    TempDir tmp;
    const fs::path surf = tmp.path / "islands.tifxyz";
    write_tifxyz(surf, ISLANDED);
    const fs::path a = tmp.path / "a.json", b = tmp.path / "b.json";
    REQUIRE(run_cli({surf.string(), "-o", a.string()}) == 0);
    REQUIRE(run_cli({surf.string(), "-o", b.string()}) == 0);
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    std::string sa((std::istreambuf_iterator<char>(fa)), {});
    std::string sb((std::istreambuf_iterator<char>(fb)), {});
    CHECK(sa == sb);
    CHECK(!sa.empty());
}

#endif  // _WIN32
