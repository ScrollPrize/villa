// Coverage for core/src/ScrollUmbilicus.cpp — the project-field-first
// umbilicus resolver and its ambiguity guard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;
using vc::core::util::resolveScrollUmbilicus;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_scroll_umb_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

void writeUmbilicus(const fs::path& path, double voxelsizeUm)
{
    std::ofstream f(path);
    f << R"({"metadata": {"total_points": 2, "voxelsize_um": )" << voxelsizeUm
      << R"(}, "control_points": [)"
      << R"({"x": 1, "y": 2, "z": 3, "score": 100},)"
      << R"({"x": 4, "y": 5, "z": 6, "score": 100}]})";
}

// Keeps the project autosaves the resolver's fixtures trigger out of the
// developer's ~/.VC3D, matching the fixture the other VolumePkg tests use.
struct TestAutosaveRoot {
    TestAutosaveRoot()
        : previous(VolumePkg::autosaveRoot())
        , root(tmpDir("autosave_root"))
    {
        VolumePkg::setAutosaveRoot(root);
    }

    ~TestAutosaveRoot()
    {
        VolumePkg::setAutosaveRoot(previous);
        fs::remove_all(root);
    }

    fs::path previous;
    fs::path root;
};

TestAutosaveRoot testAutosaveRoot;

// A saved project rooted at <dir>/project.json.
std::shared_ptr<VolumePkg> projectIn(const fs::path& dir)
{
    auto pkg = VolumePkg::newEmpty();
    pkg->save(dir / "project.json");
    return pkg;
}

} // namespace

TEST_CASE("resolver: the project's umbilicus field wins over a searchable file")
{
    auto d = tmpDir("field_wins");
    writeUmbilicus(d / "umbilicus.json", 2.4);
    writeUmbilicus(d / "declared.json", 9.6);

    auto pkg = projectIn(d);
    pkg->setUmbilicus((d / "declared.json").string());

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.path == d / "declared.json");
    REQUIRE(resolved.info.voxelsizeUm.has_value());
    CHECK(*resolved.info.voxelsizeUm == doctest::Approx(9.6));
    CHECK(resolved.info.controlPoints.size() == 2);
    fs::remove_all(d);
}

TEST_CASE("resolver: a single search hit in the package root loads")
{
    auto d = tmpDir("single_hit");
    writeUmbilicus(d / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.path == d / "umbilicus.json");
    CHECK(resolved.ambiguous.empty());
    CHECK(resolved.info.voxelsizeUm.value_or(0.0) == doctest::Approx(2.4));
    fs::remove_all(d);
}

TEST_CASE("resolver: estimated_umbilicus.json is the fallback name")
{
    auto d = tmpDir("estimated");
    writeUmbilicus(d / "estimated_umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    CHECK(resolveScrollUmbilicus(*pkg).path == d / "estimated_umbilicus.json");

    // With both present the root contributes only umbilicus.json.
    writeUmbilicus(d / "umbilicus.json", 2.4);
    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path == d / "umbilicus.json");
    CHECK(resolved.ambiguous.empty());
    fs::remove_all(d);
}

TEST_CASE("resolver: two distinct candidates are ambiguous, not resolved")
{
    auto d = tmpDir("ambiguous");
    auto packageRoot = d / "pkg";
    auto segmentsRoot = d / "other";
    fs::create_directories(packageRoot);
    fs::create_directories(segmentsRoot / "seg1");
    writeUmbilicus(packageRoot / "umbilicus.json", 2.4);
    writeUmbilicus(segmentsRoot / "umbilicus.json", 9.6);

    auto pkg = projectIn(packageRoot);
    REQUIRE(pkg->addSegmentsEntry((segmentsRoot / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.size() == 2);
    CHECK(resolved.error.find("umbilicus") != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: one file reachable through two roots is not ambiguous")
{
    auto d = tmpDir("symlinked");
    auto packageRoot = d / "pkg";
    fs::create_directories(packageRoot);
    writeUmbilicus(packageRoot / "umbilicus.json", 2.4);

    std::error_code ec;
    fs::create_directory_symlink(packageRoot, d / "mirror", ec);
    if (ec) {
        MESSAGE("directory symlinks unavailable; skipping canonicalization check");
        fs::remove_all(d);
        return;
    }

    auto pkg = projectIn(packageRoot);
    REQUIRE(pkg->addSegmentsEntry((d / "mirror" / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(fs::equivalent(resolved.path, packageRoot / "umbilicus.json"));
    fs::remove_all(d);
}

TEST_CASE("resolver: nothing found reports the searched roots")
{
    auto d = tmpDir("not_found");
    auto packageRoot = d / "pkg";
    auto segmentsRoot = d / "other";
    fs::create_directories(packageRoot);
    fs::create_directories(segmentsRoot / "seg1");

    auto pkg = projectIn(packageRoot);
    REQUIRE(pkg->addSegmentsEntry((segmentsRoot / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(resolved.error.find(packageRoot.string()) != std::string::npos);
    CHECK(resolved.error.find(segmentsRoot.string()) != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: a missing declared file errors without falling back")
{
    auto d = tmpDir("declared_missing");
    writeUmbilicus(d / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    pkg->setUmbilicus((d / "gone.json").string());

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(resolved.error.find("gone.json") != std::string::npos);
    CHECK(resolved.error.find("\"umbilicus\"") != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: an unparseable declared file errors without falling back")
{
    auto d = tmpDir("declared_broken");
    writeUmbilicus(d / "umbilicus.json", 2.4);
    { std::ofstream(d / "broken.json") << "{not json"; }

    auto pkg = projectIn(d);
    pkg->setUmbilicus((d / "broken.json").string());

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.error.find("broken.json") != std::string::npos);
    CHECK(resolved.error.find("\"umbilicus\"") != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: an unparseable search hit reports the parse failure")
{
    auto d = tmpDir("hit_broken");
    { std::ofstream(d / "umbilicus.json") << "{not json"; }

    auto pkg = projectIn(d);
    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK_FALSE(resolved.error.empty());
    fs::remove_all(d);
}

TEST_CASE("resolver: unstamped files resolve with metadata left unset")
{
    auto d = tmpDir("unstamped");
    { std::ofstream(d / "umbilicus.json") << "[[0,5,10],[10,6,11]]"; }

    auto pkg = projectIn(d);
    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.path == d / "umbilicus.json");
    CHECK_FALSE(resolved.info.voxelsizeUm.has_value());
    CHECK(resolved.info.controlPoints.size() == 2);
    fs::remove_all(d);
}
