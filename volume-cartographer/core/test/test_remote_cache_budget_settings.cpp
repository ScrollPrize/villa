#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

void setConfigDirectory(const fs::path& directory)
{
#ifdef _WIN32
    _putenv_s("VC3D_CONFIG_DIR", directory.string().c_str());
#else
    setenv("VC3D_CONFIG_DIR", directory.c_str(), 1);
#endif
}

} // namespace

// remoteCachePath() is resolved once per process, so this file holds a single
// test case and runs as its own executable.
TEST_CASE("configureRemoteCacheBudget applies the VC3D.ini disk budget to the remote cache root")
{
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    const fs::path root = fs::temp_directory_path() /
        ("vc_remote_cache_budget_settings_" + std::to_string(std::random_device{}()));
    const fs::path config = root / "config";
    const fs::path cache = root / "cache";
    fs::create_directories(config);
    {
        std::ofstream ini(config / "VC3D.ini");
        ini << "[viewer]\nremote_cache_dir=" << cache.string() << "\n"
            << "[perf]\nremote_cache_max_gib=3\nremote_cache_min_free_gib=7\n";
    }
    setConfigDirectory(config);

    CHECK_FALSE(vc::render::PersistentZarrCacheBudget::findForPath(cache / "volume"));
    vc::settings::configureRemoteCacheBudget();

    const auto budget = vc::render::PersistentZarrCacheBudget::findForPath(cache / "volume" / "0.0.0");
    REQUIRE(budget);
    const auto stats = budget->stats();
    REQUIRE(stats.maximumBytes.has_value());
    CHECK(*stats.maximumBytes == 3 * gib);
    CHECK(stats.minimumFreeBytes == 7 * gib);

    // A second call (another tool in the same process, or the VC3D app) reuses the budget.
    vc::settings::configureRemoteCacheBudget();
    CHECK(vc::render::PersistentZarrCacheBudget::findForPath(cache) == budget);

    vc::render::PersistentZarrCacheBudget::resetRegistryForTesting();
    std::error_code ec;
    fs::remove_all(root, ec);
}
