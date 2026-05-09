// Smoke test for opening the minimal s1_ds2 volpkg downloaded by
// scripts/download_minimal_volpkg.sh. Opt-in via VC_TEST_VOLPKG (path
// to the volpkg directory). Skips silently if the env var is unset, so
// the default ctest run remains hermetic.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VolumePkg.hpp"

#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace {
fs::path findVolpkgJson(const fs::path& root)
{
    if (fs::is_regular_file(root) && root.extension() == ".json") {
        return root;
    }
    if (!fs::is_directory(root)) return {};
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        const auto name = p.filename().string();
        if (name.size() >= 12 &&
            name.compare(name.size() - 12, 12, ".volpkg.json") == 0) {
            return p;
        }
    }
    return {};
}
}

TEST_CASE("VolumePkg::load on minimal s1_ds2 pull")
{
    const char* root = std::getenv("VC_TEST_VOLPKG");
    if (!root || !*root) {
        MESSAGE("VC_TEST_VOLPKG unset; skipping (set it to test-data/s1_ds2.volpkg)");
        return;
    }
    const fs::path volpkgRoot{root};
    REQUIRE_MESSAGE(fs::exists(volpkgRoot), "VC_TEST_VOLPKG path missing: " << root);

    const fs::path jsonFile = findVolpkgJson(volpkgRoot);
    REQUIRE_MESSAGE(!jsonFile.empty(),
                    "no *.volpkg.json found at " << volpkgRoot.string()
                    << " (download_minimal_volpkg.sh should have pulled it)");

    auto pkg = VolumePkg::load(jsonFile);
    REQUIRE(pkg);
    // Volumes are listed in the project JSON as remote entries; with no
    // chunks on disk they cannot be opened, but the entries should still
    // resolve.
    CHECK_FALSE(pkg->volumeEntries().empty());
    // Segments are local; the download pulls a couple of tifxyz dirs from
    // paths_2um_ds2/ and traces/.
    CHECK_FALSE(pkg->segmentEntries().empty());
}
