// Coverage for core/src/ScrollUmbilicus.cpp — the project-field-first
// umbilicus resolver and its ambiguity guard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;
using vc::core::util::deriveUmbilicusScale;
using vc::core::util::resolveScrollUmbilicus;
using vc::core::util::UmbilicusFileInfo;
using vc::core::util::UmbilicusScaleSource;
using vc::core::util::uniformRescaleFactor;

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

TEST_CASE("deriveUmbilicusScale: the real PHercParis4 stamp resolves to exactly 4")
{
    const auto scale =
        deriveUmbilicusScale(stampedPHercParis4(), kPHercParis4AnnotationGrid, 2.4);
    REQUIRE(scale.has_value());
    CHECK(scale->source == UmbilicusScaleSource::StampedDimensions);
    // Exactly 4, and specifically not the mean of (3.9996331, 3.9996331, 4.0)
    // = 3.9997553 that averaging the axis ratios produced.
    CHECK(scale->factor == doctest::Approx(4.0).epsilon(1e-12));
    CHECK(std::abs(scale->factor - 3.9997553) > 1e-6);
}

TEST_CASE("deriveUmbilicusScale: an identical grid gives exactly 1")
{
    const auto scale = deriveUmbilicusScale(
        stampedPHercParis4(), {8174.0, 8174.0, 18946.0}, 9.6);
    REQUIRE(scale.has_value());
    CHECK(scale->factor == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(scale->source == UmbilicusScaleSource::StampedDimensions);
}

TEST_CASE("uniformRescaleFactor: axes rounded in opposite directions still agree")
{
    // Both x/y and z three voxels off an exact x4, in opposite directions. One
    // factor of 4 satisfies |t - s*4| <= 3 on every axis, so this is a rescale.
    const auto factor = uniformRescaleFactor({8174.0, 8174.0, 18946.0},
                                             {32699.0, 32699.0, 75781.0});
    REQUIRE(factor.has_value());
    CHECK(*factor == doctest::Approx(4.0).epsilon(1e-12));
}

TEST_CASE("uniformRescaleFactor: a two percent spread is not a rescale")
{
    // Ratios (4, 4, 3.96): the spread the previous `hi <= lo * 1.02` accepted
    // and then averaged away.
    CHECK_FALSE(uniformRescaleFactor({8174.0, 8174.0, 18946.0},
                                     {32696.0, 32696.0, 75026.0})
                    .has_value());
    // And through the public entry point, so the weaker readings below the
    // stamped-dimensions arm are not reached as a consolation prize either.
    CHECK_FALSE(deriveUmbilicusScale(stampedPHercParis4(),
                                     {32696.0, 32696.0, 75026.0}, 2.4)
                    .has_value());
}

TEST_CASE("uniformRescaleFactor: small grids get no free tolerance")
{
    // Ratios (0.51, 0.51, 0.5). Halving 100 gives 50, and nothing rounds to 51,
    // so no integer factor explains these counts -- where a tolerance measured in
    // voxels, or a 2% spread, would have accepted them.
    CHECK_FALSE(uniformRescaleFactor({100.0, 100.0, 1000.0}, {51.0, 51.0, 500.0})
                    .has_value());
}

TEST_CASE("uniformRescaleFactor: several candidates is not an answer")
{
    // On a one-voxel grid every factor rounds to the same count, so the counts
    // identify nothing. Picking the smallest would be a guess dressed as a
    // derivation.
    CHECK_FALSE(uniformRescaleFactor({1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}).has_value());
}

TEST_CASE("uniformRescaleFactor: a cropped axis is refused")
{
    CHECK_FALSE(uniformRescaleFactor({8174.0, 8174.0, 18946.0},
                                     {32696.0, 32696.0, 40000.0})
                    .has_value());
}

TEST_CASE("uniformRescaleFactor: the rounding window is exactly floor/ceil")
{
    // 75787/4 is 18946.75, whose floor is the stamped 18946, so a factor of 4
    // explains it; 75800/4 is 18950 and nothing rounds to 18946.
    const std::array<double, 3> stamped{8174.0, 8174.0, 18946.0};
    CHECK(uniformRescaleFactor(stamped, {32696.0, 32696.0, 75787.0}).has_value());
    CHECK_FALSE(uniformRescaleFactor(stamped, {32696.0, 32696.0, 75800.0}).has_value());
}

TEST_CASE("uniformRescaleFactor: a coarser target works with a real residual")
{
    // The stamp is the finer grid, so the rounding sits on the target side, and
    // 32693/4 is 8173.25 whose ceiling is the target's 8174. Exactly 1/4, not a
    // point picked out of a range.
    const auto factor = uniformRescaleFactor({32693.0, 32693.0, 75784.0},
                                             {8174.0, 8174.0, 18946.0});
    REQUIRE(factor.has_value());
    CHECK(*factor == doctest::Approx(0.25).epsilon(1e-12));
    CHECK(*factor < 1.0);
}

TEST_CASE("uniformRescaleFactor: non-power-of-two factors, and unusable extents")
{
    // x3 is not a pyramid level but is still an integer rescale, which is exactly
    // what stamped counts are meant to express.
    const auto factor = uniformRescaleFactor({8174.0, 8174.0, 18946.0},
                                             {24522.0, 24522.0, 56838.0});
    REQUIRE(factor.has_value());
    CHECK(*factor == doctest::Approx(3.0).epsilon(1e-12));
    CHECK_FALSE(uniformRescaleFactor({8174.0, 8174.0, 18946.0},
                                     {0.0, 24522.0, 56838.0})
                    .has_value());
    CHECK_FALSE(uniformRescaleFactor({8174.0, 0.0, 18946.0},
                                     {24522.0, 24522.0, 56838.0})
                    .has_value());
}

TEST_CASE("deriveUmbilicusScale: dimensions that are not a uniform rescale say nothing")
{
    UmbilicusFileInfo info;
    info.controlPoints = {{1.0f, 2.0f, 3.0f}};
    info.volumeWidth = 8174;
    info.volumeHeight = 4000;   // a different aspect: not this grid at all
    info.volumeSlices = 18946;
    info.voxelsizeUm = 9.6;     // must not be used as a consolation prize

    CHECK_FALSE(deriveUmbilicusScale(info, kPHercParis4AnnotationGrid, 2.4).has_value());
}

TEST_CASE("deriveUmbilicusScale: voxel sizes are used when dimensions are absent")
{
    UmbilicusFileInfo info;
    info.controlPoints = {{1.0f, 2.0f, 3.0f}};
    info.voxelsizeUm = 9.6;

    const auto scale = deriveUmbilicusScale(info, {32696.0, 32696.0, 75784.0}, 2.4);
    REQUIRE(scale.has_value());
    CHECK(scale->factor == doctest::Approx(4.0));
    CHECK(scale->source == UmbilicusScaleSource::StampedVoxelSize);
}

TEST_CASE("deriveUmbilicusScale: an unstamped file is read off the grid it fills")
{
    UmbilicusFileInfo info;
    // Spans nearly all of a grid four times coarser than the target.
    info.controlPoints = {{100.0f, 100.0f, 400.0f}, {120.0f, 120.0f, 18000.0f}};

    const auto scale = deriveUmbilicusScale(info, {32696.0, 32696.0, 75784.0}, std::nullopt);
    REQUIRE(scale.has_value());
    CHECK(scale->factor == doctest::Approx(4.0));
    CHECK(scale->source == UmbilicusScaleSource::InferredFromGrid);

    // A short span fills no candidate grid, so nothing is claimed.
    UmbilicusFileInfo shortSpan;
    shortSpan.controlPoints = {{100.0f, 100.0f, 400.0f}, {120.0f, 120.0f, 900.0f}};
    CHECK_FALSE(
        deriveUmbilicusScale(shortSpan, {32696.0, 32696.0, 75784.0}, std::nullopt)
            .has_value());
}

TEST_CASE("deriveUmbilicusScale: no target grid and no voxel size means no answer")
{
    UmbilicusFileInfo info;
    info.controlPoints = {{1.0f, 2.0f, 3.0f}};
    info.voxelsizeUm = 9.6;
    CHECK_FALSE(deriveUmbilicusScale(info, {0.0, 0.0, 0.0}, std::nullopt).has_value());
}

TEST_CASE("umbilicusFrameClaim: a claim is what the file states, not what fits")
{
    using vc::core::util::umbilicusFrameClaim;

    UmbilicusFileInfo none;
    CHECK_FALSE(umbilicusFrameClaim(none).any());

    UmbilicusFileInfo dimensionsOnly;
    dimensionsOnly.volumeWidth = 8174;
    dimensionsOnly.volumeHeight = 8174;
    dimensionsOnly.volumeSlices = 18946;
    const auto dims = umbilicusFrameClaim(dimensionsOnly);
    CHECK(dims.dimensions);
    CHECK_FALSE(dims.voxelSize);
    CHECK(dims.any());

    UmbilicusFileInfo voxelOnly;
    voxelOnly.voxelsizeUm = 9.6;
    const auto voxel = umbilicusFrameClaim(voxelOnly);
    CHECK_FALSE(voxel.dimensions);
    CHECK(voxel.voxelSize);
    CHECK(voxel.any());

    UmbilicusFileInfo both;
    both.volumeWidth = 8174;
    both.volumeHeight = 8174;
    both.volumeSlices = 18946;
    both.voxelsizeUm = 9.6;
    CHECK(umbilicusFrameClaim(both).dimensions);
    CHECK(umbilicusFrameClaim(both).voxelSize);

    // Two of three counts describe no grid, so this is not a claim; treating it
    // as one would refuse a file over a typo.
    UmbilicusFileInfo partial;
    partial.volumeWidth = 8174;
    partial.volumeSlices = 18946;
    CHECK_FALSE(umbilicusFrameClaim(partial).any());

    // A stated frame that does not fit the target still counts as stated: that
    // pairing — no scale, but a claim — is what a consumer must refuse rather
    // than read as unstamped.
    UmbilicusFileInfo mismatched;
    mismatched.volumeWidth = 8174;
    mismatched.volumeHeight = 4000;
    mismatched.volumeSlices = 18946;
    CHECK(umbilicusFrameClaim(mismatched).any());
    CHECK_FALSE(
        deriveUmbilicusScale(mismatched, {32696.0, 32696.0, 75784.0}, 2.4).has_value());
}

TEST_CASE("decideUmbilicusLoadAction: refusal needs a claim and a target")
{
    using vc::core::util::decideUmbilicusLoadAction;
    using vc::core::util::UmbilicusFrameClaim;
    using vc::core::util::UmbilicusLoadAction;
    using vc::core::util::UmbilicusScale;

    const UmbilicusFrameClaim noClaim;
    UmbilicusFrameClaim claimed;
    claimed.dimensions = true;

    UmbilicusScale stamped;
    stamped.factor = 4.0;
    stamped.source = UmbilicusScaleSource::StampedDimensions;
    UmbilicusScale inferred;
    inferred.factor = 4.0;
    inferred.source = UmbilicusScaleSource::InferredFromGrid;

    // A derived scale is applied whatever its provenance: an umbilicus spans
    // nearly the whole scroll, which is what the inference tests.
    CHECK(decideUmbilicusLoadAction(stamped, claimed, true) ==
          UmbilicusLoadAction::Apply);
    CHECK(decideUmbilicusLoadAction(inferred, noClaim, true) ==
          UmbilicusLoadAction::Apply);

    // The case this exists for: stated, does not fit, refused rather than read
    // as though it had stated nothing.
    CHECK(decideUmbilicusLoadAction(std::nullopt, claimed, true) ==
          UmbilicusLoadAction::Refuse);

    // Nothing stated and nothing inferable is not a conflict.
    CHECK(decideUmbilicusLoadAction(std::nullopt, noClaim, true) ==
          UmbilicusLoadAction::UseLegacy);

    // No target frame means a stated frame could not be *checked*, which is not
    // the same as the file stating nothing. The legacy reading applies a
    // registration inverse or takes the points raw, so using it on a file whose
    // declaration we could not evaluate is proceeding exactly where the check
    // failed. Refused; only the diagnostic differs from a mismatch.
    CHECK(decideUmbilicusLoadAction(std::nullopt, claimed, false) ==
          UmbilicusLoadAction::Refuse);
    CHECK(decideUmbilicusLoadAction(stamped, claimed, false) ==
          UmbilicusLoadAction::Refuse);

    // A file that states nothing still keeps its previous reading, with or without
    // a target grid: that is the compatibility promise, not a guess.
    CHECK(decideUmbilicusLoadAction(std::nullopt, noClaim, false) ==
          UmbilicusLoadAction::UseLegacy);
}

TEST_CASE("umbilicusCandidatePaths: agrees with the search, and dedups like it")
{
    using vc::core::util::umbilicusCandidatePaths;

    auto d = tmpDir("candidate_paths");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    writeUmbilicus(volpkg / "umbilicus.json", 2.4);

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto candidates = umbilicusCandidatePaths(*pkg);

    // The file the resolver settles on must be among them, or a caller watching
    // these for change would miss the one that decides the answer.
    const auto resolved = resolveScrollUmbilicus(*pkg);
    REQUIRE(!resolved.path.empty());
    CHECK(std::any_of(candidates.begin(), candidates.end(),
                      [&](const fs::path& candidate) {
                          std::error_code ec;
                          return fs::exists(candidate, ec) && !ec &&
                                 fs::equivalent(candidate, resolved.path, ec) && !ec;
                      }));

    // Both recognised names are offered per root, since either could appear later
    // and change what resolves.
    CHECK(std::any_of(candidates.begin(), candidates.end(),
                      [](const fs::path& candidate) {
                          return candidate.filename() == "estimated_umbilicus.json";
                      }));

    // Deduplicated by canonical path: no two entries name the same file.
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            CHECK(fs::weakly_canonical(candidates[i]) !=
                  fs::weakly_canonical(candidates[j]));
        }
    }

    // Small enough to stat on every check, which is the whole reason this is
    // separate from the resolver.
    CHECK(candidates.size() <= 8);
    fs::remove_all(d);
}

TEST_CASE("umbilicusCandidatePaths: a path is offered before the file exists")
{
    using vc::core::util::umbilicusCandidatePaths;

    auto d = tmpDir("candidate_absent");
    const auto volpkg = d / "scroll.volpkg";
    writeSegment(volpkg / "paths" / "seg1");
    // No umbilicus written at all.

    auto pkg = projectIn(d);
    REQUIRE(pkg->addSegmentsEntry((volpkg / "paths" / "seg1").string()));

    const auto candidates = umbilicusCandidatePaths(*pkg);
    CHECK_FALSE(candidates.empty());
    // Absent candidates are still reported: noticing a file appear is exactly
    // what a caller comparing these needs.
    for (const auto& candidate : candidates) {
        std::error_code ec;
        CHECK_FALSE(fs::exists(candidate, ec));
    }
    fs::remove_all(d);
}
