#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/LoadJson.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace vc::json;

namespace {

fs::path makeTmpDir(const std::string& tag)
{
    auto p = fs::temp_directory_path() / ("vc_loadjson_" + tag);
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("load_json_file: missing file throws")
{
    CHECK_THROWS_AS(load_json_file("/__nonexistent__/x.json"), std::runtime_error);
}

TEST_CASE("load_json_file: round-trips a parsed object")
{
    auto dir = makeTmpDir("ok");
    auto p = dir / "t.json";
    {
        std::ofstream f(p);
        f << R"({"hello":"world","n":42})";
    }
    auto j = load_json_file(p);
    CHECK(j.is_object());
    CHECK(j["hello"].get<std::string>() == "world");
    fs::remove_all(dir);
}

TEST_CASE("require_fields throws when missing, passes when present")
{
    auto j = nlohmann::json::parse(R"({"a":1,"b":2})");
    require_fields(j, {"a", "b"}, "ctx");
    CHECK_THROWS_AS(require_fields(j, {"a", "missing"}, "ctx"), std::runtime_error);
}

TEST_CASE("require_type: matching type passes")
{
    auto j = nlohmann::json::parse(R"({"type":"seg"})");
    require_type(j, "type", "seg", "ctx");
    CHECK(true);
}

TEST_CASE("require_type: wrong type throws")
{
    auto j = nlohmann::json::parse(R"({"type":"vol"})");
    CHECK_THROWS_AS(require_type(j, "type", "seg", "ctx"), std::runtime_error);
}

TEST_CASE("require_type: missing field throws")
{
    auto j = nlohmann::json::parse(R"({"other":"x"})");
    CHECK_THROWS_AS(require_type(j, "type", "seg", "ctx"), std::runtime_error);
}

TEST_CASE("number_or: number, int, string, missing, non-object")
{
    auto j = nlohmann::json::parse(R"({"f":1.5,"i":7,"s":"3.25","bad":"nope"})");
    CHECK(number_or(j, "f", 0.0) == doctest::Approx(1.5));
    CHECK(number_or(j, "i", 0.0) == doctest::Approx(7.0));
    CHECK(number_or(j, "s", 0.0) == doctest::Approx(3.25));
    CHECK(number_or(j, "bad", 99.0) == doctest::Approx(99.0));
    CHECK(number_or(j, "absent", 5.0) == doctest::Approx(5.0));

    nlohmann::json arr = nlohmann::json::array();
    CHECK(number_or(arr, "x", 4.0) == doctest::Approx(4.0));
    nlohmann::json nul;
    CHECK(number_or(nul, "x", 8.0) == doctest::Approx(8.0));
}

TEST_CASE("string_or: present-string / missing / wrong-type / non-object")
{
    auto j = nlohmann::json::parse(R"({"s":"hi","n":3})");
    CHECK(string_or(j, "s", "def") == "hi");
    CHECK(string_or(j, "absent", "def") == "def");
    CHECK(string_or(j, "n", "def") == "def"); // wrong type
    nlohmann::json arr = nlohmann::json::array();
    CHECK(string_or(arr, "x", "def") == "def");
}
