#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utils/Json.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TmpDir {
    fs::path path;
    TmpDir()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        path = fs::temp_directory_path() /
               ("vc_json_test_" + std::to_string(rng()));
        fs::create_directories(path);
    }
    ~TmpDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

}

TEST_CASE("Json: default constructor is null")
{
    utils::Json j;
    CHECK(j.is_null());
    CHECK_FALSE(j.is_object());
    CHECK_FALSE(j.is_array());
}

TEST_CASE("Json: type predicates")
{
    CHECK(utils::Json(nullptr).is_null());
    CHECK(utils::Json(true).is_boolean());
    CHECK(utils::Json(42).is_number());
    CHECK(utils::Json(42).is_number_integer());
    CHECK(utils::Json(3.14).is_number());
    CHECK(utils::Json(3.14).is_number_float());
    CHECK(utils::Json("hello").is_string());
    CHECK(utils::Json::object().is_object());
    CHECK(utils::Json::array().is_array());
}

TEST_CASE("Json: scalar accessors")
{
    CHECK(utils::Json(true).get_bool() == true);
    CHECK(utils::Json(42).get_int() == 42);
    CHECK(utils::Json(int64_t{-1234567890123LL}).get_int64() == -1234567890123LL);
    CHECK(utils::Json(uint64_t{12345}).get_uint64() == 12345ULL);
    CHECK(utils::Json(3.14).get_double() == doctest::Approx(3.14));
    CHECK(utils::Json(2.5f).get_float() == doctest::Approx(2.5f));
    CHECK(utils::Json(size_t{99}).get_size_t() == 99);
    CHECK(utils::Json("hello").get_string() == "hello");
    CHECK(utils::Json(std::string("world")).get_string() == "world");
}

TEST_CASE("Json: object construction and access")
{
    auto j = utils::Json::object();
    j["a"] = 1;
    j["b"] = "hello";
    j["c"] = 2.5;

    REQUIRE(j.is_object());
    CHECK(j.size() == 3);
    CHECK_FALSE(j.empty());
    CHECK(j["a"].get_int() == 1);
    CHECK(j["b"].get_string() == "hello");
    CHECK(j["c"].get_double() == doctest::Approx(2.5));

    CHECK(j.contains("a"));
    CHECK_FALSE(j.contains("z"));
    CHECK(j.count("a") == 1);
    CHECK(j.count("z") == 0);
}

TEST_CASE("Json: array construction, push_back, indexed access")
{
    auto a = utils::Json::array();
    CHECK(a.is_array());
    CHECK(a.empty());

    a.push_back(utils::Json(1));
    a.push_back(utils::Json(2));
    a.push_back(utils::Json("three"));

    CHECK(a.size() == 3);
    CHECK(a[0].get_int() == 1);
    CHECK(a[1].get_int() == 2);
    CHECK(a[2].get_string() == "three");
}

TEST_CASE("Json: typed array accessors")
{
    auto j = utils::Json::parse(R"({"strs":["a","b"],"ints":[1,2,3],"dbls":[1.5,2.5],"szs":[10,20]})");

    CHECK(j["strs"].get_string_array() == std::vector<std::string>{"a", "b"});
    CHECK(j["ints"].get_int_array() == std::vector<int>{1, 2, 3});
    auto dbls = j["dbls"].get_double_array();
    REQUIRE(dbls.size() == 2);
    CHECK(dbls[0] == doctest::Approx(1.5));
    CHECK(dbls[1] == doctest::Approx(2.5));
    CHECK(j["szs"].get_size_t_array() == std::vector<size_t>{10, 20});
}

TEST_CASE("Json: parse and dump round-trip")
{
    const std::string text = R"({"a":1,"b":"hi","c":[10,20,30],"d":{"nested":true}})";
    auto j = utils::Json::parse(text);

    CHECK(j["a"].get_int() == 1);
    CHECK(j["b"].get_string() == "hi");
    REQUIRE(j["c"].is_array());
    CHECK(j["c"].size() == 3);
    CHECK(j["c"][1].get_int() == 20);
    CHECK(j["d"]["nested"].get_bool() == true);

    auto dumped = j.dump();
    auto reparsed = utils::Json::parse(dumped);
    CHECK(reparsed["a"].get_int() == 1);
    CHECK(reparsed["d"]["nested"].get_bool() == true);
}

TEST_CASE("Json: dump with indent produces multi-line output")
{
    auto j = utils::Json::object();
    j["x"] = 1;
    j["y"] = 2;
    auto pretty = j.dump(2);
    CHECK(pretty.find('\n') != std::string::npos);
    auto compact = j.dump();
    CHECK(compact.find('\n') == std::string::npos);
}

TEST_CASE("Json: erase removes object key")
{
    auto j = utils::Json::object();
    j["a"] = 1;
    j["b"] = 2;
    REQUIRE(j.size() == 2);

    j.erase("a");
    CHECK_FALSE(j.contains("a"));
    CHECK(j.contains("b"));
    CHECK(j.size() == 1);
}

TEST_CASE("Json: update merges objects")
{
    auto base = utils::Json::object();
    base["a"] = 1;
    base["b"] = 2;

    auto patch = utils::Json::object();
    patch["b"] = 99;
    patch["c"] = 3;

    base.update(patch);

    CHECK(base["a"].get_int() == 1);
    CHECK(base["b"].get_int() == 99);
    CHECK(base["c"].get_int() == 3);
}

TEST_CASE("Json: equality")
{
    auto a = utils::Json::parse(R"({"k":[1,2,3]})");
    auto b = utils::Json::parse(R"({"k":[1,2,3]})");
    auto c = utils::Json::parse(R"({"k":[1,2]})");
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("Json: copy and move preserve content")
{
    auto j = utils::Json::object();
    j["x"] = 42;

    auto copy = j;
    CHECK(copy["x"].get_int() == 42);

    auto moved = std::move(j);
    CHECK(moved["x"].get_int() == 42);
}

TEST_CASE("Json: parse_file reads from disk")
{
    TmpDir tmp;
    auto p = tmp.path / "data.json";
    {
        std::ofstream out(p);
        out << R"({"answer":42})";
    }
    auto j = utils::Json::parse_file(p);
    CHECK(j["answer"].get_int() == 42);
}

TEST_CASE("Json: iterator visits all object members")
{
    auto j = utils::Json::object();
    j["a"] = 1;
    j["b"] = 2;
    j["c"] = 3;

    int count = 0;
    int sum = 0;
    for (auto it = j.begin(); it != j.end(); ++it) {
        sum += (*it).get_int();
        ++count;
    }
    CHECK(count == 3);
    CHECK(sum == 6);
}
