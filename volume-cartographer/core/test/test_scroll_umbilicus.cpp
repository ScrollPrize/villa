// Coverage for core/src/ScrollUmbilicus.cpp — the project-field-first
// umbilicus resolver and its ambiguity guard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;
using vc::core::util::resolveScrollUmbilicus;
using vc::core::util::UmbilicusFileInfo;

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

// Same file shape, but with a caller-supplied metadata body so malformed
// frame fields can be exercised.
void writeUmbilicusWithMetadata(const fs::path& path, const std::string& metadata)
{
    std::ofstream f(path);
    f << R"({"metadata": {)" << metadata << R"(}, "control_points": [)"
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
// Discovery treats a directory as an individual segment only when it carries the
// tifxyz payload, so fixtures standing in for one have to as well.
void writeSegment(const fs::path& dir)
{
    fs::create_directories(dir);
    for (const char* plane : {"x.tif", "y.tif", "z.tif"}) {
        std::ofstream(dir / plane) << "tif";
    }
}

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

// ------- C1: malformed frame metadata is refused here, not downstream -------

TEST_CASE("resolver: malformed metadata on a search hit is refused, errors listed")
{
    auto d = tmpDir("hit_bad_metadata");
    writeUmbilicusWithMetadata(d / "umbilicus.json",
                               R"("voxelsize_um": 0, "volume_width": -5)");

    auto pkg = projectIn(d);
    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(resolved.error.find((d / "umbilicus.json").string()) !=
          std::string::npos);
    CHECK(resolved.error.find(
              "voxelsize_um: expected a positive number, got 0") !=
          std::string::npos);
    CHECK(resolved.error.find(
              "volume_width: expected a positive integer, got -5") !=
          std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: malformed metadata on the declared file is refused")
{
    auto d = tmpDir("declared_bad_metadata");
    writeUmbilicus(d / "umbilicus.json", 2.4);
    writeUmbilicusWithMetadata(d / "declared.json", R"("volume": "")");

    auto pkg = projectIn(d);
    pkg->setUmbilicus((d / "declared.json").string());

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.error.find("declared.json") != std::string::npos);
    CHECK(resolved.error.find(
              R"(volume: expected a non-empty string, got "")") !=
          std::string::npos);
    fs::remove_all(d);
}

// ------- C2: a configured location that cannot be used never searches -------

TEST_CASE("resolver: a configured remote location errors instead of searching")
{
    auto d = tmpDir("declared_remote");
    // Discoverable and perfectly good — must still not be returned.
    writeUmbilicus(d / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    pkg->setUmbilicus("s3://scrolls/PHercParis4/umbilicus.json");

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(resolved.error.find("s3://scrolls/PHercParis4/umbilicus.json") !=
          std::string::npos);
    CHECK(resolved.error.find("\"umbilicus\"") != std::string::npos);
    // Distinct from the missing-local-file wording, which would be misleading.
    CHECK(resolved.error.find("does not exist") == std::string::npos);
    // No search happened: the local hit was not picked up.
    CHECK(resolved.error.find((d / "umbilicus.json").string()) ==
          std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: a configured http location errors instead of searching")
{
    auto d = tmpDir("declared_http");
    writeUmbilicus(d / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    pkg->setUmbilicus("https://example.org/umbilicus.json");

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.error.find("https://example.org/umbilicus.json") !=
          std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: relative and file:// configured locations still resolve")
{
    auto d = tmpDir("declared_forms");
    writeUmbilicus(d / "declared.json", 9.6);
    auto pkg = projectIn(d);

    SUBCASE("absolute")
    {
        pkg->setUmbilicus((d / "declared.json").string());
        const auto resolved = resolveScrollUmbilicus(*pkg);
        CHECK(resolved.error.empty());
        CHECK(fs::equivalent(resolved.path, d / "declared.json"));
    }

    SUBCASE("relative to the project directory")
    {
        pkg->setUmbilicus("declared.json");
        const auto resolved = resolveScrollUmbilicus(*pkg);
        CHECK(resolved.error.empty());
        CHECK(fs::equivalent(resolved.path, d / "declared.json"));
    }

    SUBCASE("file:// url")
    {
        pkg->setUmbilicus("file://" + (d / "declared.json").string());
        const auto resolved = resolveScrollUmbilicus(*pkg);
        CHECK(resolved.error.empty());
        CHECK(fs::equivalent(resolved.path, d / "declared.json"));
    }

    fs::remove_all(d);
}

// ------- C3: both segment layouts are covered by the search -------

TEST_CASE("resolver: <volpkg>/paths/<segment> finds <volpkg>/umbilicus.json")
{
    auto d = tmpDir("paths_layout");
    // The project lives one level above the volpkg, so the package root alone
    // does not reach <volpkg>/umbilicus.json.
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(fs::equivalent(resolved.path, volpkg / "umbilicus.json"));
    CHECK(resolved.info.voxelsizeUm.value_or(0.0) == doctest::Approx(2.4));
    fs::remove_all(d);
}

TEST_CASE("resolver: one file reachable via parent and grandparent is not ambiguous")
{
    auto d = tmpDir("parent_grandparent");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "paths" / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    // "<volpkg>/paths/./seg1" makes the parent root "<volpkg>/paths/." and the
    // grandparent root "<volpkg>/paths": two distinct roots, one file.
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "." / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(resolved.ambiguous.empty());
    CHECK(fs::equivalent(resolved.path, volpkg / "paths" / "umbilicus.json"));
    fs::remove_all(d);
}

TEST_CASE("resolver: a segments-folder attachment does not search above the volpkg")
{
    auto d = tmpDir("folder_attachment_scope");
    // The project file and the volpkg live in different places, so the
    // directory holding the volpkg is not already a root by another route.
    const auto project = d / "project";
    const auto store = d / "scrolls";
    const auto volpkg = store / "scroll.volpkg";
    fs::create_directories(project);
    fs::create_directories(volpkg / "paths");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);
    // A neighbour of the volpkg. Taking the grandparent of the attached folder
    // would pull this in and either win outright or force a false ambiguity.
    writeUmbilicus(store / "umbilicus.json", 9.6);

    auto pkg = projectIn(project);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.ambiguous.empty());
    CHECK(fs::equivalent(resolved.path, volpkg / "umbilicus.json"));
    CHECK(resolved.info.voxelsizeUm.value_or(0.0) == doctest::Approx(2.4));
    fs::remove_all(d);
}

TEST_CASE("resolver: an individual-segment attachment still reaches the volpkg")
{
    auto d = tmpDir("segment_attachment_scope");
    const auto project = d / "project";
    const auto store = d / "scrolls";
    const auto volpkg = store / "scroll.volpkg";
    fs::create_directories(project);
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);

    auto pkg = projectIn(project);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.error.empty());
    CHECK(fs::equivalent(resolved.path, volpkg / "umbilicus.json"));
    fs::remove_all(d);
}

TEST_CASE("resolver: metadata that is not an object refuses the file")
{
    auto d = tmpDir("metadata_not_object");
    std::ofstream(d / "umbilicus.json")
        << R"({"metadata": "nope", "control_points": [)"
        << R"({"x": 1, "y": 2, "z": 3}]})";

    auto pkg = projectIn(d);

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.error.find("metadata: expected an object") != std::string::npos);
    fs::remove_all(d);
}

TEST_CASE("resolver: parent and grandparent holding different files still refuse")
{
    auto d = tmpDir("parent_grandparent_distinct");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);
    writeUmbilicus(volpkg / "paths" / "umbilicus.json", 9.6);

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto resolved = resolveScrollUmbilicus(*pkg);
    CHECK(resolved.path.empty());
    CHECK(resolved.ambiguous.size() == 2);
    CHECK(resolved.error.find("set the project's \"umbilicus\" field") !=
          std::string::npos);
    fs::remove_all(d);
}

namespace {

// The real PHercParis4 configuration: umbilicus.json stamps the 9.6 um ds2 grid
// while annotation runs on the project's 2.4 um store, whose level-0 shape is
// 32693 x 32693 x 75784. Only z is an exact multiple of the stamp -- 32693/4 is
// 8173.25, which the downsample rounded up to 8174 -- so the axes disagree by
// 0.0092% and a tolerance is genuinely required here.
UmbilicusFileInfo stampedPHercParis4()
{
    UmbilicusFileInfo info;
    info.controlPoints = {{1.0f, 2.0f, 3.0f}};
    info.volumeWidth = 8174;
    info.volumeHeight = 8174;
    info.volumeSlices = 18946;
    info.voxelsizeUm = 9.6;
    return info;
}

const std::array<double, 3> kPHercParis4AnnotationGrid{32693.0, 32693.0, 75784.0};

} // namespace

namespace {

using vc::core::util::scanUmbilicusCandidates;
using vc::core::util::umbilicusCandidatePaths;

bool offers(const std::vector<fs::path>& candidates, const std::string& name)
{
    return std::any_of(candidates.begin(), candidates.end(),
                       [&name](const fs::path& candidate) {
                           return candidate.filename() == name;
                       });
}

std::size_t countIn(const std::vector<fs::path>& candidates, const fs::path& root,
                    const std::string& name)
{
    return static_cast<std::size_t>(
        std::count_if(candidates.begin(), candidates.end(),
                      [&](const fs::path& candidate) {
                          return candidate == root / name;
                      }));
}

} // namespace

TEST_CASE("scanUmbilicusCandidates: what could change the answer, and nothing else")
{
    auto d = tmpDir("candidate_scope");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);
    writeUmbilicus(volpkg / "estimated_umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    SUBCASE("discovery: the deciding file is offered and agrees with the resolver")
    {
        const auto candidates = umbilicusCandidatePaths(*pkg);
        const auto resolved = resolveScrollUmbilicus(*pkg);
        REQUIRE(!resolved.path.empty());
        CHECK(std::any_of(candidates.begin(), candidates.end(),
                          [&](const fs::path& candidate) {
                              std::error_code ec;
                              return fs::exists(candidate, ec) && !ec &&
                                     fs::equivalent(candidate, resolved.path, ec) && !ec;
                          }));

        // A shadowed lower-priority file cannot change the answer, so watching it
        // would invalidate derived views over a file the resolver never opens.
        CHECK(countIn(candidates, volpkg, "umbilicus.json") == 1);
        CHECK(countIn(candidates, volpkg, "estimated_umbilicus.json") == 0);

        // And the record says which one the contents matter for.
        const auto scanned = scanUmbilicusCandidates(*pkg);
        for (const auto& candidate : scanned) {
            if (candidate.path == volpkg / "umbilicus.json") {
                CHECK(candidate.exists);
                CHECK(candidate.decidesResolution);
            }
            // An absent path is a dependency by its absence only.
            if (!candidate.exists) {
                CHECK_FALSE(candidate.decidesResolution);
            }
        }

        // Deduplicated by canonical path: no two entries name the same file.
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            for (std::size_t j = i + 1; j < candidates.size(); ++j) {
                CHECK(fs::weakly_canonical(candidates[i]) !=
                      fs::weakly_canonical(candidates[j]));
            }
        }
        // Small enough to stat on every interaction, which is the whole reason
        // this is separate from the resolver.
        CHECK(candidates.size() <= 8);
    }

    SUBCASE("an explicit project field is the only dependency")
    {
        pkg->setUmbilicus((volpkg / "umbilicus.json").string());
        const auto candidates = umbilicusCandidatePaths(*pkg);
        REQUIRE(candidates.size() == 1);
        CHECK(candidates.front() == volpkg / "umbilicus.json");
        // The discoverable file beside it is ignored by the resolver, so touching
        // it must not read as a change.
        CHECK_FALSE(offers(candidates, "estimated_umbilicus.json"));
    }

    SUBCASE("an unusable configured location depends on no file")
    {
        pkg->setUmbilicus("s3://bucket/umbilicus.json");
        CHECK(umbilicusCandidatePaths(*pkg).empty());
    }

    SUBCASE("a configured local path is offered before it exists")
    {
        pkg->setUmbilicus((volpkg / "not_there_yet.json").string());
        const auto scanned = scanUmbilicusCandidates(*pkg);
        REQUIRE(scanned.size() == 1);
        CHECK(scanned.front().path == volpkg / "not_there_yet.json");
        CHECK_FALSE(scanned.front().exists);
        // Its appearing is what has to be noticed.
        CHECK(scanned.front().decidesResolution);
    }

    fs::remove_all(d);
}

TEST_CASE("scanUmbilicusCandidates: an absent higher-priority name is still watched")
{
    auto d = tmpDir("candidate_absent");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    // Only the fallback name exists, so umbilicus.json appearing later would
    // change what resolves and must be in the list.
    writeUmbilicus(volpkg / "estimated_umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto candidates = umbilicusCandidatePaths(*pkg);
    CHECK(countIn(candidates, volpkg, "umbilicus.json") == 1);
    CHECK(countIn(candidates, volpkg, "estimated_umbilicus.json") == 1);
    CHECK(resolveScrollUmbilicus(*pkg).path == volpkg / "estimated_umbilicus.json");
    fs::remove_all(d);
}

TEST_CASE("umbilicusCandidatePaths: a path is offered before the file exists")
{
    auto d = tmpDir("candidate_none");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    // No umbilicus written at all.

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto candidates = umbilicusCandidatePaths(*pkg);
    CHECK_FALSE(candidates.empty());
    // Nothing exists, so every name in every root is watched: noticing a file
    // appear is exactly what a caller comparing these needs.
    CHECK(offers(candidates, "umbilicus.json"));
    CHECK(offers(candidates, "estimated_umbilicus.json"));
    for (const auto& candidate : candidates) {
        std::error_code ec;
        CHECK_FALSE(fs::exists(candidate, ec));
    }
    fs::remove_all(d);
}
