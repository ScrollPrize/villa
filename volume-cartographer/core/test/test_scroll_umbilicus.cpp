// Coverage for core/src/ScrollUmbilicus.cpp — deciding the frame an umbilicus
// file's coordinates index.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/ScrollUmbilicus.hpp"
#include <vc_object_store/objects/UmbilicusFile.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::objects::deriveUmbilicusScale;
using vc::objects::UmbilicusFile;
using vc::objects::UmbilicusScaleSource;
using vc::objects::uniformRescaleFactor;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_scroll_umb_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

// Same file shape, but with caller-supplied metadata body so malformed
// frame fields can be exercised.
void writeUmbilicusWithMetadata(const fs::path& path, const std::string& metadata)
{
    std::ofstream f(path);
    f << R"({"metadata": {)" << metadata << R"(}, "control_points": [)"
      << R"({"x": 1, "y": 2, "z": 3, "score": 100},)"
      << R"({"x": 4, "y": 5, "z": 6, "score": 100}]})";
}

// Same file shape, but with caller-supplied control points so the
// point-inference fallback in deriveUmbilicusScale() can be exercised.
void writeUmbilicusWithPoints(const fs::path& path, const std::string& metadata,
                              const std::vector<std::array<double, 3>>& points)
{
    std::ofstream f(path);
    f << R"({"metadata": {)" << metadata << R"(}, "control_points": [)";
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i != 0) {
            f << ",";
        }
        f << R"({"x": )" << points[i][0] << R"(, "y": )" << points[i][1]
          << R"(, "z": )" << points[i][2] << R"(, "score": 100})";
    }
    f << "]}";
}

} // namespace

namespace {

// The real PHercParis4 configuration: umbilicus.json stamps the 9.6 um ds2 grid
// while annotation runs on the project's 2.4 um store, whose level-0 shape is
// 32693 x 32693 x 75784. Only z is an exact multiple of the stamp -- 32693/4 is
// 8173.25, which the downsample rounded up to 8174 -- so the axes disagree by
// 0.0092% and a tolerance is genuinely required here.
UmbilicusFile stampedPHercParis4()
{
    UmbilicusFile info;
    info.points = {{1.0f, 2.0f, 3.0f}};
    info.volumeWidth = 8174;
    info.volumeHeight = 8174;
    info.volumeSlices = 18946;
    info.voxelSizeUm = 9.6;
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
    UmbilicusFile info;
    info.points = {{1.0f, 2.0f, 3.0f}};
    info.volumeWidth = 8174;
    info.volumeHeight = 4000;   // a different aspect: not this grid at all
    info.volumeSlices = 18946;
    info.voxelSizeUm = 9.6;     // must not be used as a consolation prize

    CHECK_FALSE(deriveUmbilicusScale(info, kPHercParis4AnnotationGrid, 2.4).has_value());
}

TEST_CASE("deriveUmbilicusScale: voxel sizes are used when dimensions are absent")
{
    UmbilicusFile info;
    info.points = {{1.0f, 2.0f, 3.0f}};
    info.voxelSizeUm = 9.6;

    const auto scale = deriveUmbilicusScale(info, {32696.0, 32696.0, 75784.0}, 2.4);
    REQUIRE(scale.has_value());
    CHECK(scale->factor == doctest::Approx(4.0));
    CHECK(scale->source == UmbilicusScaleSource::StampedVoxelSize);
}

TEST_CASE("deriveUmbilicusScale: an unstamped file is read off the grid it fills")
{
    UmbilicusFile info;
    // Spans nearly all of a grid four times coarser than the target.
    info.points = {{100.0f, 100.0f, 400.0f}, {120.0f, 120.0f, 18000.0f}};

    const auto scale = deriveUmbilicusScale(info, {32696.0, 32696.0, 75784.0}, std::nullopt);
    REQUIRE(scale.has_value());
    CHECK(scale->factor == doctest::Approx(4.0));
    CHECK(scale->source == UmbilicusScaleSource::InferredFromGrid);

    // A short span fills no candidate grid, so nothing is claimed.
    UmbilicusFile shortSpan;
    shortSpan.points = {{100.0f, 100.0f, 400.0f}, {120.0f, 120.0f, 900.0f}};
    CHECK_FALSE(
        deriveUmbilicusScale(shortSpan, {32696.0, 32696.0, 75784.0}, std::nullopt)
            .has_value());
}

TEST_CASE("deriveUmbilicusScale: no target grid and no voxel size means no answer")
{
    UmbilicusFile info;
    info.points = {{1.0f, 2.0f, 3.0f}};
    info.voxelSizeUm = 9.6;
    CHECK_FALSE(deriveUmbilicusScale(info, {0.0, 0.0, 0.0}, std::nullopt).has_value());
}

TEST_CASE("umbilicusFrameClaim: a claim is what the file states, not what fits")
{
    using vc::objects::umbilicusFrameClaim;

    UmbilicusFile none;
    CHECK_FALSE(umbilicusFrameClaim(none).any());

    UmbilicusFile dimensionsOnly;
    dimensionsOnly.volumeWidth = 8174;
    dimensionsOnly.volumeHeight = 8174;
    dimensionsOnly.volumeSlices = 18946;
    const auto dims = umbilicusFrameClaim(dimensionsOnly);
    CHECK(dims.dimensions);
    CHECK_FALSE(dims.voxelSize);
    CHECK(dims.any());

    UmbilicusFile voxelOnly;
    voxelOnly.voxelSizeUm = 9.6;
    const auto voxel = umbilicusFrameClaim(voxelOnly);
    CHECK_FALSE(voxel.dimensions);
    CHECK(voxel.voxelSize);
    CHECK(voxel.any());

    UmbilicusFile both;
    both.volumeWidth = 8174;
    both.volumeHeight = 8174;
    both.volumeSlices = 18946;
    both.voxelSizeUm = 9.6;
    CHECK(umbilicusFrameClaim(both).dimensions);
    CHECK(umbilicusFrameClaim(both).voxelSize);

    // Two of three counts describe no grid, so this is not a claim; treating it
    // as one would refuse a file over a typo.
    UmbilicusFile partial;
    partial.volumeWidth = 8174;
    partial.volumeSlices = 18946;
    CHECK_FALSE(umbilicusFrameClaim(partial).any());

    // A stated frame that does not fit the target still counts as stated: that
    // pairing — no scale, but a claim — is what a consumer must refuse rather
    // than read as unstamped.
    UmbilicusFile mismatched;
    mismatched.volumeWidth = 8174;
    mismatched.volumeHeight = 4000;
    mismatched.volumeSlices = 18946;
    CHECK(umbilicusFrameClaim(mismatched).any());
    CHECK_FALSE(
        deriveUmbilicusScale(mismatched, {32696.0, 32696.0, 75784.0}, 2.4).has_value());
}

TEST_CASE("decideUmbilicusLoadAction: refusal needs a claim and a target")
{
    using vc::objects::decideUmbilicusLoadAction;
    using vc::objects::UmbilicusFrameClaim;
    using vc::objects::UmbilicusLoadAction;
    using vc::objects::UmbilicusScale;

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

TEST_CASE("umbilicusStampContradiction: a stamp is checked against the volume it names")
{
    using vc::objects::umbilicusStampContradiction;

    UmbilicusFile info;
    info.volume = "s1_2um_ds2.zarr";
    info.volumeWidth = 8174;
    info.volumeHeight = 8174;
    info.volumeSlices = 18946;
    info.voxelSizeUm = 9.6;

    // Consistent: the named volume is the stamped grid itself (factor 1), and
    // 9.6 um matches what it implies.
    CHECK_FALSE(umbilicusStampContradiction(
        info, {8174.0, 8174.0, 18946.0}, 9.6).has_value());

    // Consistent through a rescale: the named volume is the x4 finer scan, so
    // its 2.4 um carried through the factor is 9.6.
    CHECK_FALSE(umbilicusStampContradiction(
        info, {32693.0, 32693.0, 75784.0}, 2.4).has_value());

    // A grid the stamp cannot be a uniform rescale of contradicts it.
    CHECK(umbilicusStampContradiction(
        info, {32696.0, 32696.0, 40000.0}, 2.4).has_value());

    // A voxel size the named volume disproves contradicts it: implied is
    // 2.4 x 4 = 9.6, stamped says 19.2.
    UmbilicusFile wrongVoxel = info;
    wrongVoxel.voxelSizeUm = 19.2;
    CHECK(umbilicusStampContradiction(
        wrongVoxel, {32693.0, 32693.0, 75784.0}, 2.4).has_value());

    // Within metadata round-trip slack is not a contradiction: 9.596 recorded
    // against an implied 9.6 differs by four parts in ten thousand.
    UmbilicusFile rounded = info;
    rounded.voxelSizeUm = 9.6;
    CHECK_FALSE(umbilicusStampContradiction(
        rounded, {32693.0, 32693.0, 75784.0}, 2.399).has_value());

    // Nothing to check against is not a contradiction — an unverifiable stamp
    // is a different, weaker state than a disproven one.
    CHECK_FALSE(umbilicusStampContradiction(
        info, {0.0, 0.0, 0.0}, std::nullopt).has_value());

    // A voxel-size-only stamp is never contradicted by the voxel size
    // alone: the difference is the conversion deriveUmbilicusScale()
    // applies, not evidence against the stamp. Only a dimension triplet
    // implying a different voxel size contradicts it (covered above).
    UmbilicusFile voxelOnly;
    voxelOnly.volume = "s1_8um.zarr";
    voxelOnly.voxelSizeUm = 9.6;
    CHECK_FALSE(umbilicusStampContradiction(
        voxelOnly, {0.0, 0.0, 0.0}, 9.6).has_value());
    CHECK_FALSE(umbilicusStampContradiction(
        voxelOnly, {0.0, 0.0, 0.0}, 2.4).has_value());
    CHECK_FALSE(umbilicusStampContradiction(
        voxelOnly, {32693.0, 32693.0, 75784.0}, 2.4).has_value());
}

TEST_CASE("decideUmbilicusLoadAction: a contradicted stamp refuses outright")
{
    using vc::objects::decideUmbilicusLoadAction;
    using vc::objects::UmbilicusFrameClaim;
    using vc::objects::UmbilicusLoadAction;
    using vc::objects::UmbilicusScale;

    UmbilicusFrameClaim claimed;
    claimed.voxelSize = true;

    UmbilicusScale viaVoxel;
    viaVoxel.factor = 4.0;
    viaVoxel.source = UmbilicusScaleSource::StampedVoxelSize;

    // The finding this closes: a scale derived from a stamped voxel size the
    // named volume disproves used to be applied with only a warning beside it.
    CHECK(decideUmbilicusLoadAction(viaVoxel, claimed, true, true) ==
          UmbilicusLoadAction::Refuse);

    // The contradiction outranks everything, including the no-target arm that
    // would otherwise fall back to the legacy reading.
    CHECK(decideUmbilicusLoadAction(std::nullopt, claimed, false, true) ==
          UmbilicusLoadAction::Refuse);

    // Uncontradicted, the same inputs keep their previous outcomes.
    CHECK(decideUmbilicusLoadAction(viaVoxel, claimed, true, false) ==
          UmbilicusLoadAction::Apply);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an unstamped file keeps the legacy reading")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_legacy");
    const auto path = dir / "umbilicus.json";
    writeUmbilicusWithMetadata(path, "\"total_points\": 2");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Legacy reading: points used as-is, so z=4 interpolates a third of the
    // way from (1,2,3) to (4,5,6).
    const auto c = loaded.umbilicus->center_at(4);
    CHECK(c[0] == doctest::Approx(2.0));
    CHECK(c[1] == doctest::Approx(3.0));
    CHECK(c[2] == doctest::Approx(4.0));
}

TEST_CASE("loadUmbilicusWithFrameCheck: malformed frame metadata is refused")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_malformed");
    const auto path = dir / "umbilicus.json";
    writeUmbilicusWithMetadata(
        path,
        "\"volume_width\": \"wide\", \"volume_height\": 40, \"volume_slices\": 60");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK_FALSE(loaded.umbilicus.has_value());
    CHECK_FALSE(loaded.error.empty());
    CHECK(loaded.error.find("volume_width") != std::string::npos);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an inferred grid never applies a stamped rescale")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_inferred_no_apply");
    const auto path = dir / "umbilicus.json";
    writeUmbilicusWithMetadata(
        path, "\"volume_width\": 10, \"volume_height\": 20, \"volume_slices\": 30");

    // Stamped 10x20x30 against a 20x40x60 working grid: a uniform x2 — but
    // the grid is inferred from surface bounds, which cannot confirm the
    // rescale (a partial surface can exactly mimic a downsampled volume).
    // The loader warns and keeps the legacy reading instead.
    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK_FALSE(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Legacy reading kept: points as-is, so z=4 interpolates between
    // (1,2,3) and (4,5,6) to (2,3,4). The x2 is not applied.
    const auto c = loaded.umbilicus->center_at(4);
    CHECK(c[0] == doctest::Approx(2.0));
    CHECK(c[1] == doctest::Approx(3.0));
    CHECK(c[2] == doctest::Approx(4.0));
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an inferred grid does not rescale a stamp it merely resembles")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_inferred_mimic");
    const auto path = dir / "umbilicus.json";
    // Stamped 1000^3 against surface-derived 500^3 bounds: the bounds
    // exactly mimic a 2x downsampled volume, but both inputs may already be
    // in base coordinates. The x0.5 must not be applied: x=400 stays 400.
    writeUmbilicusWithPoints(
        path,
        "\"volume_width\": 1000, \"volume_height\": 1000, \"volume_slices\": 1000",
        {{400.0, 200.0, 100.0}, {400.0, 200.0, 500.0}});

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {500.0, 500.0, 500.0}, cv::Vec3i{512, 512, 512},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK_FALSE(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    const auto c = loaded.umbilicus->center_at(300);
    CHECK(c[0] == doctest::Approx(400.0));
    CHECK(c[1] == doctest::Approx(200.0));
    CHECK(c[2] == doctest::Approx(300.0));
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: a voxel-size-only stamp is converted, not refused")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_voxel_only");
    const auto path = dir / "umbilicus.json";
    // Stamped at 4 um/voxel against a 2 um/voxel target: the 2x difference
    // is the conversion, not a contradiction.
    writeUmbilicusWithMetadata(path, "\"voxelsize_um\": 4.0");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{60, 40, 20},
        UmbilicusTargetGridAuthority::Authoritative,
        /*targetVoxelSizeUm=*/2.0);
    CHECK(loaded.error.empty());
    CHECK(loaded.warning.empty());
    CHECK_FALSE(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Points carried x2 into the working frame: (1,2,3)-(4,5,6) became
    // (2,4,6)-(8,10,12), so z=9 interpolates to (5,7,9).
    const auto c = loaded.umbilicus->center_at(9);
    CHECK(c[0] == doctest::Approx(5.0));
    CHECK(c[1] == doctest::Approx(7.0));
    CHECK(c[2] == doctest::Approx(9.0));
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an inferred grid still applies an explicit voxel-size conversion")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_inferred_voxel");
    const auto path = dir / "umbilicus.json";
    // A stamped voxel size against the target's own is an explicit
    // coordinate conversion: it never depended on the inferred grid, so it
    // is applied even though the grid is inferred.
    writeUmbilicusWithMetadata(path, "\"voxelsize_um\": 4.0");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred,
        /*targetVoxelSizeUm=*/2.0);
    CHECK(loaded.error.empty());
    CHECK(loaded.warning.empty());
    CHECK_FALSE(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    const auto c = loaded.umbilicus->center_at(9);
    CHECK(c[0] == doctest::Approx(5.0));
    CHECK(c[1] == doctest::Approx(7.0));
    CHECK(c[2] == doctest::Approx(9.0));
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an inferred grid does not apply a point-inferred scale")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_inferred_points");
    const auto path = dir / "umbilicus.json";
    // A voxel-size-only stamp with no target voxel size to convert against:
    // the points fill the half grid well enough to infer x2 on their own,
    // but a scale read off the points against an inferred grid is doubly
    // unverifiable, so it warns and keeps the legacy reading.
    writeUmbilicusWithPoints(
        path, "\"voxelsize_um\": 9.6",
        {{1.0, 2.0, 4.0}, {9.0, 18.0, 26.0}});

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK_FALSE(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Legacy reading kept: points as-is, so z=15 interpolates halfway
    // from (1,2,4) to (9,18,26). The x2 the points infer is not applied.
    const auto c = loaded.umbilicus->center_at(15);
    CHECK(c[0] == doctest::Approx(5.0));
    CHECK(c[1] == doctest::Approx(10.0));
    CHECK(c[2] == doctest::Approx(15.0));
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an authoritative grid refuses a voxel-only stamp it cannot check")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_authoritative_unchecked");
    const auto path = dir / "umbilicus.json";
    // A voxel-size-only stamp with no target voxel size to convert against:
    // the points fill the half grid well enough to infer x2 on their own,
    // but the stated frame was never evaluated, so rescaling on point
    // inference would treat an unchecked frame as checked. Under an
    // authoritative grid the file is refused instead.
    writeUmbilicusWithPoints(
        path, "\"voxelsize_um\": 9.6",
        {{1.0, 2.0, 4.0}, {9.0, 18.0, 26.0}});

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Authoritative);
    CHECK_FALSE(loaded.umbilicus.has_value());
    CHECK(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    CHECK_FALSE(loaded.error.empty());
    CHECK(loaded.error.find("voxelsize_um") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: an unconfirmable stamp warns and keeps the legacy reading")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_warn");
    const auto path = dir / "umbilicus.json";
    // 11x20x30 is no uniform rescale of the 20x40x60 working grid -- but the
    // grid is inferred, so that proves nothing about the file.
    writeUmbilicusWithMetadata(
        path, "\"volume_width\": 11, \"volume_height\": 20, \"volume_slices\": 30");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK_FALSE(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Legacy reading kept: points as-is.
    const auto c = loaded.umbilicus->center_at(4);
    CHECK(c[0] == doctest::Approx(2.0));
    CHECK(c[1] == doctest::Approx(3.0));
    CHECK(c[2] == doctest::Approx(4.0));
}

TEST_CASE("loadUmbilicusWithFrameCheck: an unreadable file throws like FromFile")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    CHECK_THROWS(loadUmbilicusWithFrameCheck(
        tmpDir("frame_load_missing") / "no-such-file.json",
        {20.0, 40.0, 60.0},
        cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred));
}

TEST_CASE("loadUmbilicusWithFrameCheck: an authoritative grid refuses an unfitting stamp")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_refuse");
    const auto path = dir / "umbilicus.json";
    // 11x20x30 is no uniform rescale of the authoritative 20x40x60 volume
    // grid: the file is wrong about itself, so it is refused outright --
    // the GrowPatch case, where a silent misread would mis-orient every
    // patch normal.
    writeUmbilicusWithMetadata(
        path, "\"volume_width\": 11, \"volume_height\": 20, \"volume_slices\": 30");

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{60, 40, 20},
        UmbilicusTargetGridAuthority::Authoritative);
    CHECK_FALSE(loaded.umbilicus.has_value());
    CHECK_FALSE(loaded.error.empty());
    CHECK(loaded.error.find("refusing") != std::string::npos);
    CHECK(loaded.warning.empty());
}

TEST_CASE("loadUmbilicusWithFrameCheck: an authoritative grid still rescales a fitting stamp")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_auth_apply");
    const auto path = dir / "umbilicus.json";
    writeUmbilicusWithMetadata(
        path, "\"volume_width\": 10, \"volume_height\": 20, \"volume_slices\": 30");

    // Stamped 10x20x30 against the authoritative 20x40x60 grid: uniform x2.
    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{60, 40, 20},
        UmbilicusTargetGridAuthority::Authoritative);
    CHECK(loaded.error.empty());
    CHECK(loaded.warning.empty());
    CHECK_FALSE(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Points carried x2 into the working frame, as in the inferred case.
    const auto c = loaded.umbilicus->center_at(9);
    CHECK(c[0] == doctest::Approx(5.0));
    CHECK(c[1] == doctest::Approx(7.0));
    CHECK(c[2] == doctest::Approx(9.0));
}

TEST_CASE("loadUmbilicusWithFrameCheck: a contradicted stamp is refused even when the points infer a scale")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_contradicted");
    const auto path = dir / "umbilicus.json";
    // 11x20x30 is no uniform rescale of the authoritative 20x40x60 grid, so
    // the stamp is contradicted -- but the points fill the half grid well
    // enough (z 4..26 over a 30-deep half grid) that the point-inference
    // fallback would derive x2 on its own. The contradiction outranks the
    // derived scale: a file wrong about itself is refused, not rescaled on
    // a reading of points its own stamp disproves.
    writeUmbilicusWithPoints(
        path, "\"volume_width\": 11, \"volume_height\": 20, \"volume_slices\": 30",
        {{1.0, 2.0, 4.0}, {9.0, 18.0, 26.0}});

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{60, 40, 20},
        UmbilicusTargetGridAuthority::Authoritative);
    CHECK_FALSE(loaded.umbilicus.has_value());
    CHECK_FALSE(loaded.error.empty());
    CHECK(loaded.error.find("refusing") != std::string::npos);
    CHECK(loaded.warning.empty());
    fs::remove_all(dir);
}

TEST_CASE("loadUmbilicusWithFrameCheck: a contradicted stamp warns under an inferred grid even when the points infer a scale")
{
    using vc::core::util::loadUmbilicusWithFrameCheck;
    using vc::core::util::UmbilicusTargetGridAuthority;
    const auto dir = tmpDir("frame_load_contradicted_inferred");
    const auto path = dir / "umbilicus.json";
    // Same contradicted stamp and inferable points as the authoritative
    // case, but the grid is inferred: the mismatch proves nothing about
    // the file, so it warns and keeps the legacy reading instead of
    // applying the scale the points suggest.
    writeUmbilicusWithPoints(
        path, "\"volume_width\": 11, \"volume_height\": 20, \"volume_slices\": 30",
        {{1.0, 2.0, 4.0}, {9.0, 18.0, 26.0}});

    const auto loaded = loadUmbilicusWithFrameCheck(
        path, {20.0, 40.0, 60.0}, cv::Vec3i{64, 64, 64},
        UmbilicusTargetGridAuthority::Inferred);
    CHECK(loaded.error.empty());
    CHECK_FALSE(loaded.warning.empty());
    CHECK(loaded.scaleDescription.empty());
    REQUIRE(loaded.umbilicus.has_value());
    // Legacy reading kept: points as-is, so z=15 interpolates halfway
    // from (1,2,4) to (9,18,26). The x2 the points infer is not applied.
    const auto c = loaded.umbilicus->center_at(15);
    CHECK(c[0] == doctest::Approx(5.0));
    CHECK(c[1] == doctest::Approx(10.0));
    CHECK(c[2] == doctest::Approx(15.0));
    fs::remove_all(dir);
}
