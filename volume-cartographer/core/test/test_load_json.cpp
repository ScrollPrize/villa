#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/LoadJson.hpp"

#include <stdexcept>
#include <string>

using namespace vc::json;

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
