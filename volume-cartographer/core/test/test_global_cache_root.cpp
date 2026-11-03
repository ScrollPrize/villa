#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/GlobalCacheRoot.hpp"
#include <chrono>

TEST_CASE("explicit global cache config overrides mount defaults without creating the cache")
{
    const auto temporary = std::filesystem::temp_directory_path() /
        ("vc-cache-config-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(std::filesystem::create_directory(temporary));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{temporary};
    const auto wanted = temporary / "not-created";
    const auto write = [&](const nlohmann::json& config) {
        std::ofstream file(temporary / "cache.json");
        file << config.dump();
    };
    write({{"version", 1}, {"root", wanted.string()}});
    CHECK(vc::core::util::globalCacheRoot("/some/old/suggestion", temporary) == wanted);
    CHECK_FALSE(std::filesystem::exists(wanted));
    write({{"version", 1}, {"root", "relative"}});
    CHECK_THROWS(vc::core::util::globalCacheRoot({}, temporary));
    write({{"version", 2}, {"root", wanted.string()}});
    CHECK_THROWS(vc::core::util::globalCacheRoot({}, temporary));
    write({{"version", 1}, {"unrecognized", wanted.string()}});
    CHECK_THROWS(vc::core::util::globalCacheRoot({}, temporary));
}
