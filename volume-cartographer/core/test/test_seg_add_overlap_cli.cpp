#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "utils/Json.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
TEST_CASE("cli tests are POSIX-only") {}
#else

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

void write_tifxyz(const fs::path& dir, const std::string& id, float z)
{
    constexpr int rows = 8;
    constexpr int cols = 8;
    cv::Mat x(rows, cols, CV_32F);
    cv::Mat y(rows, cols, CV_32F);
    cv::Mat zs(rows, cols, CV_32F, cv::Scalar(z));
    for (int v = 0; v < rows; ++v) {
        for (int u = 0; u < cols; ++u) {
            x.at<float>(v, u) = 4.f * u;
            y.at<float>(v, u) = 4.f * v;
        }
    }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), zs));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"format\":\"tifxyz\",\"type\":\"seg\","
            "\"scale\":[1.0,1.0],\"uuid\":\""
         << id
         << "\",\"bbox\":[[0.0,0.0," << z
         << "],[28.0,28.0," << z << "]]}\n";
    REQUIRE(meta.good());
}

std::string sh(const std::string& value)
{
    std::string quoted = "'";
    for (char c : value) {
        quoted += c == '\'' ? "'\\''" : std::string(1, c);
    }
    return quoted + "'";
}

int run_cli(const std::vector<std::string>& args)
{
    std::string command = sh(VC_SEG_ADD_OVERLAP_BIN);
    for (const std::string& arg : args) {
        command += " " + sh(arg);
    }
    command += " >/dev/null 2>&1";
    const int status = std::system(command.c_str());
    REQUIRE(status != -1);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

std::string read_file(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

void write_collection_fixture(const fs::path& root)
{
    write_tifxyz(root / "sources" / "source-a", "source-a", 10.f);
    write_tifxyz(root / "sources" / "source-b", "source-b", 30.f);
    write_tifxyz(root / "targets" / "duplicate", "duplicate", 10.f);
    write_tifxyz(root / "targets" / "near", "near", 11.5f);
    write_tifxyz(root / "targets" / "far", "far", 13.f);
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        path = fs::temp_directory_path() /
               ("vc_seg_add_overlap_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

} // namespace

TEST_CASE("report measures duplicate and within-tolerance coverage")
{
    TempDir tmp;
    const fs::path source = tmp.path / "source";
    const fs::path targets = tmp.path / "targets";
    write_tifxyz(source, "source", 10.f);
    write_tifxyz(targets / "duplicate", "duplicate", 10.f);
    write_tifxyz(targets / "near", "near", 11.5f);
    write_tifxyz(targets / "far", "far", 13.f);

    const fs::path reportPath = tmp.path / "report.json";
    REQUIRE(run_cli({
        "--target", targets.string(),
        "--source", source.string(),
        "--workers", "1",
        "--point-stride", "2",
        "--report-json", reportPath.string(),
    }) == 0);

    const utils::Json report = utils::Json::parse_file(reportPath);
    CHECK(report["metric"]["name"].get_string() ==
          "directed_source_point_surface_coverage");
    CHECK(report["parameters"]["tolerance"].get_float() == 2.0);
    CHECK(report["parameters"]["point_stride"].get_int() == 2);
    CHECK(report["target_count"].get_int() == 3);
    CHECK(report["directed_overlap_pair_count"].get_int() == 2);

    const utils::Json& row = report["sources"][size_t(0)];
    CHECK(row["source_path"].get_string() == ".");
    CHECK(row["valid_source_points"].get_int() == 64);
    CHECK(row["queried_source_points"].get_int() == 32);
    REQUIRE(row["hits"].size() == 2);
    CHECK(row["hits"][size_t(0)]["target_path"].get_string() == "duplicate");
    CHECK(row["hits"][size_t(0)]["matched_source_points"].get_int() == 32);
    CHECK(row["hits"][size_t(0)]["source_coverage_fraction"].get_float() == 1.0);
    CHECK(row["hits"][size_t(1)]["target_path"].get_string() == "near");
    CHECK(row["hits"][size_t(1)]["matched_source_points"].get_int() == 32);
    CHECK(row["hits"][size_t(1)]["source_coverage_fraction"].get_float() == 1.0);

    const utils::Json sourceOverlap =
        utils::Json::parse_file(source / "overlapping.json");
    CHECK(sourceOverlap["overlapping"].size() == 2);
    CHECK(!fs::exists(targets / "far" / "overlapping.json"));
}

TEST_CASE("report refuses missing parents and surface-data aliases")
{
    TempDir tmp;
    const fs::path source = tmp.path / "source";
    const fs::path target = tmp.path / "target";
    write_tifxyz(source, "source", 10.f);
    write_tifxyz(target, "target", 10.f);

    CHECK(run_cli({
        "--target", target.string(),
        "--source", source.string(),
        "--report-json", (tmp.path / "missing" / "report.json").string(),
    }) == 1);
    CHECK(run_cli({
        "--target", target.string(),
        "--source", source.string(),
        "--report-json", (source / "meta.json").string(),
    }) == 1);

    const fs::path hardlink = tmp.path / "hardlink.json";
    fs::create_hard_link(source / "meta.json", hardlink);
    CHECK(run_cli({
        "--target", target.string(),
        "--source", source.string(),
        "--report-json", hardlink.string(),
    }) == 1);
}

TEST_CASE("numeric options reject signs")
{
    CHECK(run_cli({"--target", ".", "--source", ".",
                   "--point-stride", "-1"}) == 1);
    CHECK(run_cli({"--target", ".", "--source", ".",
                   "--workers", "+2"}) == 1);
}

TEST_CASE("canonical target aliases are counted once")
{
    TempDir tmp;
    const fs::path source = tmp.path / "source";
    const fs::path targets = tmp.path / "targets";
    write_tifxyz(source, "source", 10.f);
    write_tifxyz(targets / "real", "target", 10.f);
    fs::create_directory_symlink(targets / "real", targets / "alias");

    const fs::path reportPath = tmp.path / "report.json";
    REQUIRE(run_cli({
        "--target", targets.string(),
        "--source", source.string(),
        "--report-json", reportPath.string(),
    }) == 0);

    const utils::Json report = utils::Json::parse_file(reportPath);
    CHECK(report["target_count"].get_int() == 1);
    CHECK(report["directed_overlap_pair_count"].get_int() == 1);
    const utils::Json& row = report["sources"][size_t(0)];
    REQUIRE(row["hits"].size() == 1);
    CHECK(row["hits"][size_t(0)]["source_coverage_fraction"].get_float() == 1.0);
}

TEST_CASE("report mode preserves legacy output and is worker deterministic")
{
    TempDir tmp;
    const fs::path baseline = tmp.path / "baseline";
    const fs::path reportOne = tmp.path / "report-one";
    const fs::path reportTwo = tmp.path / "report-two";
    write_collection_fixture(baseline);
    write_collection_fixture(reportOne);
    write_collection_fixture(reportTwo);

    REQUIRE(run_cli({
        "--target", (baseline / "targets").string(),
        "--source", (baseline / "sources").string(),
        "--workers", "1",
    }) == 0);
    REQUIRE(run_cli({
        "--target", (reportOne / "targets").string(),
        "--source", (reportOne / "sources").string(),
        "--workers", "1",
        "--report-json", (reportOne / "report.json").string(),
    }) == 0);
    REQUIRE(run_cli({
        "--target", (reportTwo / "targets").string(),
        "--source", (reportTwo / "sources").string(),
        "--workers", "2",
        "--report-json", (reportTwo / "report.json").string(),
    }) == 0);

    for (const fs::path& relative : {
             fs::path("sources/source-a/overlapping.json"),
             fs::path("targets/near/overlapping.json"),
             fs::path("targets/duplicate/overlapping.json")}) {
        const std::string expected = read_file(baseline / relative);
        CHECK(read_file(reportOne / relative) == expected);
        CHECK(read_file(reportTwo / relative) == expected);
    }
    CHECK(!fs::exists(baseline / "sources/source-b/overlapping.json"));
    CHECK(!fs::exists(reportOne / "sources/source-b/overlapping.json"));
    CHECK(!fs::exists(reportTwo / "sources/source-b/overlapping.json"));
    CHECK(read_file(reportOne / "report.json") ==
          read_file(reportTwo / "report.json"));
}

#endif
