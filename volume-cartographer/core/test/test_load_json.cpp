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
    CHECK(j["hello"].get_string() == "world");
    fs::remove_all(dir);
}

TEST_CASE("require_fields throws when missing, passes when present")
{
    auto j = utils::Json::parse(R"({"a":1,"b":2})");
    require_fields(j, {"a", "b"}, "ctx");
    CHECK_THROWS_AS(require_fields(j, {"a", "missing"}, "ctx"), std::runtime_error);
}

TEST_CASE("require_type: matching type passes")
{
    auto j = utils::Json::parse(R"({"type":"seg"})");
    require_type(j, "type", "seg", "ctx");
    CHECK(true);
}

TEST_CASE("require_type: wrong type throws")
{
    auto j = utils::Json::parse(R"({"type":"vol"})");
    CHECK_THROWS_AS(require_type(j, "type", "seg", "ctx"), std::runtime_error);
}

TEST_CASE("require_type: missing field throws")
{
    auto j = utils::Json::parse(R"({"other":"x"})");
    CHECK_THROWS_AS(require_type(j, "type", "seg", "ctx"), std::runtime_error);
}

TEST_CASE("number_or: number, int, string, missing, non-object")
{
    auto j = utils::Json::parse(R"({"f":1.5,"i":7,"s":"3.25","bad":"nope"})");
    CHECK(number_or(j, "f", 0.0) == doctest::Approx(1.5));
    CHECK(number_or(j, "i", 0.0) == doctest::Approx(7.0));
    CHECK(number_or(j, "s", 0.0) == doctest::Approx(3.25));
    CHECK(number_or(j, "bad", 99.0) == doctest::Approx(99.0));
    CHECK(number_or(j, "absent", 5.0) == doctest::Approx(5.0));

    utils::Json arr = utils::Json::array();
    CHECK(number_or(arr, "x", 4.0) == doctest::Approx(4.0));
    utils::Json nul;
    CHECK(number_or(nul, "x", 8.0) == doctest::Approx(8.0));
}

TEST_CASE("string_or: present-string / missing / wrong-type / non-object")
{
    auto j = utils::Json::parse(R"({"s":"hi","n":3})");
    CHECK(string_or(j, "s", "def") == "hi");
    CHECK(string_or(j, "absent", "def") == "def");
    CHECK(string_or(j, "n", "def") == "def"); // wrong type
    utils::Json arr = utils::Json::array();
    CHECK(string_or(arr, "x", "def") == "def");
}

TEST_CASE("tags_or_empty: present object / wrong type / missing / non-object root")
{
    auto j = utils::Json::parse(R"({"tags":{"a":"1"}})");
    auto t = tags_or_empty(j);
    CHECK(t.is_object());
    CHECK(t["a"].get_string() == "1");

    auto j2 = utils::Json::parse(R"({"tags":[1,2,3]})");
    CHECK(tags_or_empty(j2).is_object());
    CHECK_FALSE(tags_or_empty(j2).contains("anything"));

    auto j3 = utils::Json::parse(R"({"other":1})");
    CHECK(tags_or_empty(j3).is_object());

    utils::Json arr = utils::Json::array();
    CHECK(tags_or_empty(arr).is_object());
}

TEST_CASE("has_tag")
{
    auto j = utils::Json::parse(R"({"tags":{"foo":"x"}})");
    CHECK(has_tag(j, "foo"));
    CHECK_FALSE(has_tag(j, "bar"));
    auto j2 = utils::Json::parse(R"({"other":1})");
    CHECK_FALSE(has_tag(j2, "foo"));
}

TEST_CASE("ensure_object turns null/array into empty object")
{
    utils::Json nul;
    ensure_object(nul);
    CHECK(nul.is_object());
    utils::Json arr = utils::Json::array();
    ensure_object(arr);
    CHECK(arr.is_object());
    auto obj = utils::Json::parse(R"({"a":1})");
    ensure_object(obj);
    CHECK(obj.is_object());
    CHECK(obj.contains("a"));
}

TEST_CASE("ensure_tags creates tags object")
{
    utils::Json m;
    auto& t = ensure_tags(m);
    CHECK(m.is_object());
    CHECK(m.contains("tags"));
    CHECK(t.is_object());
    // Existing object with wrong-type tags should be replaced
    auto m2 = utils::Json::parse(R"({"tags":[1,2]})");
    auto& t2 = ensure_tags(m2);
    CHECK(t2.is_object());
}
