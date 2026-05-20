// Coverage for core/src/VolpkgConvert.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/VolpkgConvert.hpp"

#include "utils/Json.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_volpkg_convert_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

void writeJson(const fs::path& p, const std::string& contents)
{
    std::ofstream f(p);
    f << contents;
}

} // namespace

TEST_CASE("convertVolpkg: remote scheme returns the documented 'not supported' message")
{
    auto outFile = tmpDir("remote") / "project.json";
    auto r = vc::convertVolpkg("s3://bucket/key", outFile);
    CHECK_FALSE(r.ok);
    CHECK(r.message.find("remote") != std::string::npos);
    fs::remove_all(outFile.parent_path());
}

TEST_CASE("convertVolpkg: http(s) scheme is treated as remote")
{
    auto outFile = tmpDir("remote_http") / "project.json";
    auto r = vc::convertVolpkg("https://example.com/foo", outFile);
    CHECK_FALSE(r.ok);
    fs::remove_all(outFile.parent_path());
}

TEST_CASE("convertVolpkg: input not a directory is rejected")
{
    auto d = tmpDir("not_dir");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg("/__nonexistent__/__path__", outFile);
    CHECK_FALSE(r.ok);
    CHECK(r.message.find("not a directory") != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: empty volpkg dir produces a minimal project.json")
{
    auto d = tmpDir("empty_root");
    auto outFile = d / "out_project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    REQUIRE(fs::exists(r.output));
    auto j = utils::Json::parse_file(r.output);
    CHECK(j.contains("name"));
    CHECK(j.contains("volumes"));
    CHECK(j.contains("segments"));
    CHECK(j.contains("normal_grids"));
    CHECK(j["volumes"].is_array());
    CHECK(j["volumes"].size() == 0);
    CHECK(j["segments"].size() == 0);
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: picks up volumes/, paths/, normal_grids/ subdirs")
{
    auto d = tmpDir("full_layout");
    fs::create_directories(d / "volumes");
    fs::create_directories(d / "paths");
    fs::create_directories(d / "normal_grids");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    auto j = utils::Json::parse_file(r.output);
    CHECK(j["volumes"].size() == 1);
    CHECK(j["segments"].size() == 1);
    CHECK(j["normal_grids"].size() == 1);
    CHECK(j.contains("output_segments"));
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: traces/ and export/ also count as segments")
{
    auto d = tmpDir("traces_export");
    fs::create_directories(d / "traces");
    fs::create_directories(d / "export");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    auto j = utils::Json::parse_file(r.output);
    CHECK(j["segments"].size() == 2);
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: config.json name override is applied")
{
    auto d = tmpDir("named");
    writeJson(d / "config.json", R"({"name":"my-scroll"})");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    auto j = utils::Json::parse_file(r.output);
    CHECK(j["name"].get_string() == "my-scroll");
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: config.json name='NULL' is ignored")
{
    auto d = tmpDir("null_name");
    writeJson(d / "config.json", R"({"name":"NULL"})");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    auto j = utils::Json::parse_file(r.output);
    // Falls back to the directory name (which is the tmp folder).
    CHECK(j["name"].get_string() != "NULL");
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: malformed config.json surfaces a warning, still succeeds")
{
    auto d = tmpDir("bad_cfg");
    writeJson(d / "config.json", "{ this is not json");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    CHECK_FALSE(r.message.empty()); // warning lands in `message`
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: normal3d/<sub> dirs become tagged volume entries")
{
    auto d = tmpDir("normal3d");
    fs::create_directories(d / "normal3d" / "xx");
    fs::create_directories(d / "normal3d" / "yy");
    auto outFile = d / "project.json";
    auto r = vc::convertVolpkg(d.string(), outFile);
    REQUIRE(r.ok);
    auto j = utils::Json::parse_file(r.output);
    // Two tagged entries appended to volumes.
    CHECK(j["volumes"].size() == 2);
    fs::remove_all(d);
}

TEST_CASE("convertVolpkg: cannot create output directory propagates an error")
{
    // /proc/self is read-only on Linux; trying to create a subdir there should
    // fail. If the test is running on a system without /proc, just skip.
    if (!fs::exists("/proc/self")) return;
    auto d = tmpDir("ok_input");
    auto outFile = fs::path("/proc/self/cant_create_here/project.json");
    auto r = vc::convertVolpkg(d.string(), outFile);
    CHECK_FALSE(r.ok);
    fs::remove_all(d);
}
