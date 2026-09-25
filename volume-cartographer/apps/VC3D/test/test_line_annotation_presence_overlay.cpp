#include <QtTest/QtTest>

#include "LineAnnotationPresenceOverlay.hpp"
#include "volume_viewers/OverlayBlendLut.hpp"
#include "volume_viewers/OverlayLevelSelection.hpp"

#include <array>
#include <cstdint>

class TestLineAnnotationPresenceOverlay : public QObject {
    Q_OBJECT

private slots:
    void findsTheVolumeTaggedWithManifestAndGroup();
    void prefersTheStoredManifestLocationOverAliases();
    void rebaseLevelFollowsDyadicDownsampling();
    void constantAlphaKeepsColormapColours();
    void valueWeightedAlphaRampsAcrossTheWindow();
    void valueWeightedTintKeepsFullColour();
    void levelSelectionSkipsMissingFineLevels();
    void storedLevelFitFindsTheRebaseForEachScrollLevel();
    void frameConsistencyAcceptsConventionsAndRejectsLeaves();
};

using vc3d::line_annotation::TaggedVolumeId;

void TestLineAnnotationPresenceOverlay::findsTheVolumeTaggedWithManifestAndGroup()
{
    const std::vector<TaggedVolumeId> volumes{
        {"scroll", {"vc-open-data-coordinate-space:PHerc1451/2026@L0"}},
        {"nx", {"vc-lasagna-manifest:/cache/fibers.lasagna.json", "vc-lasagna-group:nx"}},
        {"presence", {"vc-lasagna-manifest:/cache/fibers.lasagna.json",
                      "vc-lasagna-group:presence", "vc-remote-auth:anonymous"}},
        {"other-presence", {"vc-lasagna-manifest:/cache/other.lasagna.json",
                            "vc-lasagna-group:presence"}},
    };

    const auto found = vc3d::line_annotation::findLasagnaGroupVolumeId(
        volumes, {"/cache/fibers.lasagna.json"}, "presence");
    QVERIFY(found.has_value());
    QCOMPARE(*found, std::string{"presence"});

    // Same manifest, different group.
    const auto nx = vc3d::line_annotation::findLasagnaGroupVolumeId(
        volumes, {"/cache/fibers.lasagna.json"}, "nx");
    QVERIFY(nx.has_value());
    QCOMPARE(*nx, std::string{"nx"});

    // A manifest that attached no presence group, and an empty candidate list.
    QVERIFY(!vc3d::line_annotation::findLasagnaGroupVolumeId(
                 volumes, {"/cache/missing.lasagna.json"}, "presence")
                 .has_value());
    QVERIFY(!vc3d::line_annotation::findLasagnaGroupVolumeId(volumes, {}, "presence")
                 .has_value());
    QVERIFY(!vc3d::line_annotation::findLasagnaGroupVolumeId(volumes, {""}, "presence")
                 .has_value());
}

void TestLineAnnotationPresenceOverlay::prefersTheStoredManifestLocationOverAliases()
{
    const std::vector<TaggedVolumeId> volumes{
        {"by-url", {"vc-lasagna-manifest:https://bucket/fibers.lasagna.json",
                    "vc-lasagna-group:presence"}},
        {"by-path", {"vc-lasagna-manifest:/cache/fibers.lasagna.json",
                     "vc-lasagna-group:presence"}},
    };
    const auto found = vc3d::line_annotation::findLasagnaGroupVolumeId(
        volumes,
        {"/cache/fibers.lasagna.json", "https://bucket/fibers.lasagna.json"},
        "presence");
    QVERIFY(found.has_value());
    QCOMPARE(*found, std::string{"by-path"});

    // Only the alias matches: it is still found.
    const auto alias = vc3d::line_annotation::findLasagnaGroupVolumeId(
        volumes, {"/elsewhere.json", "https://bucket/fibers.lasagna.json"}, "presence");
    QVERIFY(alias.has_value());
    QCOMPARE(*alias, std::string{"by-url"});
}

void TestLineAnnotationPresenceOverlay::rebaseLevelFollowsDyadicDownsampling()
{
    using vc3d::line_annotation::presenceRebaseLevel;
    const auto level = [](double scale) {
        const auto value = presenceRebaseLevel(scale);
        return value ? *value : -1;
    };
    QCOMPARE(level(1.0), 0);
    QCOMPARE(level(0.5), 1);
    QCOMPARE(level(0.25), 2);
    QCOMPARE(level(0.125), 3);
    // The active volume is finer than the fiber base, or not a power of two.
    QVERIFY(!presenceRebaseLevel(2.0).has_value());
    QVERIFY(!presenceRebaseLevel(0.3).has_value());
    QVERIFY(!presenceRebaseLevel(0.0).has_value());
    QVERIFY(!presenceRebaseLevel(-0.5).has_value());
}

namespace
{

std::array<uint32_t, 256> rampLut()
{
    std::array<uint32_t, 256> lut{};
    for (int v = 0; v < 256; ++v) {
        lut[v] = 0xFF000000u | (static_cast<uint32_t>(v) << 16);
    }
    return lut;
}

}  // namespace

void TestLineAnnotationPresenceOverlay::constantAlphaKeepsColormapColours()
{
    vc3d::overlay_blend::Luts luts;
    const auto lut = rampLut();
    vc3d::overlay_blend::build(luts, lut, 0.6f, 16.0f, 255.0f, false,
                               std::array<float, 3>{1.0f, 0.0f, 1.0f});
    // Constant mode ignores the tint: colours come from the colormap and every
    // value blends at the user opacity, exactly as the main Overlay panel does.
    QCOMPARE(luts.color[0], lut[0]);
    QCOMPARE(luts.color[200], lut[200]);
    QCOMPARE(luts.alpha[0], 0.6f);
    QCOMPARE(luts.alpha[16], 0.6f);
    QCOMPARE(luts.alpha[255], 0.6f);
}

void TestLineAnnotationPresenceOverlay::valueWeightedAlphaRampsAcrossTheWindow()
{
    vc3d::overlay_blend::Luts luts;
    const auto lut = rampLut();
    vc3d::overlay_blend::build(luts, lut, 0.8f, 16.0f, 255.0f, true, std::nullopt);
    // No tint: colours still come from the colormap.
    QCOMPARE(luts.color[200], lut[200]);
    // Alpha is 0 at and below the window low, the full opacity at the top.
    QCOMPARE(luts.alpha[0], 0.0f);
    QCOMPARE(luts.alpha[16], 0.0f);
    QCOMPARE(luts.alpha[255], 0.8f);
    const float mid = luts.alpha[(16 + 255) / 2];
    QVERIFY(mid > 0.35f && mid < 0.45f);
    // Monotone.
    for (int v = 1; v < 256; ++v) {
        QVERIFY(luts.alpha[v] >= luts.alpha[v - 1]);
    }
}

void TestLineAnnotationPresenceOverlay::valueWeightedTintKeepsFullColour()
{
    vc3d::overlay_blend::Luts luts;
    const auto lut = rampLut();
    vc3d::overlay_blend::build(luts, lut, 1.0f, 0.0f, 255.0f, true,
                               std::array<float, 3>{1.0f, 0.0f, 1.0f});
    // A single-colour tint stays saturated at every value; only alpha varies,
    // so weak presence reads as faint magenta rather than dark purple.
    QCOMPARE(luts.color[1], 0xFFFF00FFu);
    QCOMPARE(luts.color[128], 0xFFFF00FFu);
    QCOMPARE(luts.color[255], 0xFFFF00FFu);
    QCOMPARE(luts.alpha[0], 0.0f);
    QCOMPARE(luts.alpha[255], 1.0f);
    QVERIFY(luts.alpha[128] > 0.49f && luts.alpha[128] < 0.51f);
}

void TestLineAnnotationPresenceOverlay::levelSelectionSkipsMissingFineLevels()
{
    using vc3d::overlay_level::presentLevelAtOrCoarser;
    // Presence export: 7 slots, stored /6 only (beyond the UI's 0-5 range).
    const auto storedFromSix = [](int level) { return level >= 6; };
    QCOMPARE(presentLevelAtOrCoarser(0, 7, storedFromSix), 6);
    QCOMPARE(presentLevelAtOrCoarser(5, 7, storedFromSix), 6);
    QCOMPARE(presentLevelAtOrCoarser(6, 7, storedFromSix), 6);
    // Stored /3 and /4 of five: fine requests land on /3, coarse stay put.
    const auto threeAndFour = [](int level) { return level == 3 || level == 4; };
    QCOMPARE(presentLevelAtOrCoarser(0, 5, threeAndFour), 3);
    QCOMPARE(presentLevelAtOrCoarser(3, 5, threeAndFour), 3);
    QCOMPARE(presentLevelAtOrCoarser(4, 5, threeAndFour), 4);
    // An internal gap is stepped over.
    const auto gapAtTwo = [](int level) { return level != 2; };
    QCOMPARE(presentLevelAtOrCoarser(2, 4, gapAtTwo), 3);
    // Out-of-range requests clamp; a pyramid with nothing present returns the
    // coarsest slot rather than an invalid index.
    QCOMPARE(presentLevelAtOrCoarser(-3, 5, threeAndFour), 3);
    QCOMPARE(presentLevelAtOrCoarser(9, 5, threeAndFour), 4);
    QCOMPARE(presentLevelAtOrCoarser(1, 3, [](int) { return false; }), 2);
    QCOMPARE(presentLevelAtOrCoarser(1, 0, [](int) { return true; }), 0);
}

void TestLineAnnotationPresenceOverlay::storedLevelFitFindsTheRebaseForEachScrollLevel()
{
    using vc3d::line_annotation::rebaseLevelsFittingPyramid;
    using vc3d::line_annotation::StoredPyramidLevel;
    using Levels = std::vector<StoredPyramidLevel>;
    const std::array<std::size_t, 3> l0{59944, 20812, 20812};
    const std::array<std::size_t, 3> l1{29972, 10406, 10406};
    const std::array<std::size_t, 3> l2{14986, 5203, 5203};
    const std::array<std::size_t, 3> l4{3747, 1301, 1301};
    // PHerc1451 presence export: /3 and /4 of a [59944,20812,20812] scroll.
    // The synthesized frame (2602 x 8 = 20816) is not the scroll, but the
    // stored levels fit each scroll level, including @L4 where /4 becomes
    // level 0 of the view.
    const Levels sparse{{3, {7493, 2602, 2602}, {64, 64, 64}},
                        {4, {3747, 1301, 1301}, {64, 64, 64}}};
    QVERIFY(rebaseLevelsFittingPyramid(l0, sparse) == std::vector<int>{0});
    QVERIFY(rebaseLevelsFittingPyramid(l1, sparse) == std::vector<int>{1});
    QVERIFY(rebaseLevelsFittingPyramid(l2, sparse) == std::vector<int>{2});
    QVERIFY(rebaseLevelsFittingPyramid(l4, sparse) == std::vector<int>{4});
    // A full /0../4 pyramid of the same scroll fits @L1 at k = 1: the view
    // drops /0 and its level 0 is the stored /1.
    const Levels full{{0, {59944, 20812, 20812}, {64, 64, 64}},
                      {1, {29972, 10406, 10406}, {64, 64, 64}},
                      {2, {14986, 5203, 5203}, {64, 64, 64}},
                      {3, {7493, 2602, 2602}, {64, 64, 64}},
                      {4, {3747, 1301, 1301}, {64, 64, 64}}};
    QVERIFY(rebaseLevelsFittingPyramid(l0, full) == std::vector<int>{0});
    QVERIFY(rebaseLevelsFittingPyramid(l1, full) == std::vector<int>{1});
    QVERIFY(rebaseLevelsFittingPyramid(l4, full) == std::vector<int>{4});
    // A level padded out to whole storage chunks still fits (2624 = 41 x 64),
    // and for a sharded array the shard, not the inner chunk, is the
    // tolerance (7552 - 7493 = 59 < 64 but > 32).
    const Levels padded{{3, {7552, 2624, 2624}, {64, 64, 64}}};
    QVERIFY(rebaseLevelsFittingPyramid(l0, padded) == std::vector<int>{0});
    const Levels innerChunkOnly{{3, {7552, 2624, 2624}, {32, 32, 32}}};
    QVERIFY(rebaseLevelsFittingPyramid(l0, innerChunkOnly).empty());
    // The /3 array attached on its own as a single level 0 fits nothing: the
    // manifest frame must not be trusted for it.
    const Levels leaf{{0, {7493, 2602, 2602}, {64, 64, 64}}};
    QVERIFY(rebaseLevelsFittingPyramid(l0, leaf).empty());
    QVERIFY(rebaseLevelsFittingPyramid(l1, leaf).empty());
    // An unrelated volume fits nothing; an active grid finer than the finest
    // stored level implies is not a rebase.
    QVERIFY(rebaseLevelsFittingPyramid({1000, 1000, 1000}, sparse).empty());
    QVERIFY(rebaseLevelsFittingPyramid({119888, 41624, 41624}, sparse).empty());
    QVERIFY(rebaseLevelsFittingPyramid(l0, Levels{}).empty());
    // Degenerate tiny shapes can fit several candidates; callers must
    // disambiguate or refuse.
    const Levels tiny{{2, {1, 1, 1}, {1, 1, 1}}};
    QVERIFY(rebaseLevelsFittingPyramid({1, 1, 1}, tiny).size() > 1);
}

void TestLineAnnotationPresenceOverlay::frameConsistencyAcceptsConventionsAndRejectsLeaves()
{
    using vc3d::line_annotation::storedLevelsConsistentWithFrame;
    using vc3d::line_annotation::StoredPyramidLevel;
    using Levels = std::vector<StoredPyramidLevel>;
    const std::array<std::size_t, 3> frame{59944, 20812, 20812};
    // The PHerc1451 export is that frame's /3 and /4.
    const Levels sparse{{3, {7493, 2602, 2602}, {64, 64, 64}},
                        {4, {3747, 1301, 1301}, {64, 64, 64}}};
    QVERIFY(storedLevelsConsistentWithFrame(frame, sparse));
    // A frame recorded in the inclusive-maximum convention (one smaller) is
    // still that pyramid: the stored level is then one row above the implied
    // size, within the padding tolerance. (The dyadic matcher accepts the
    // frame against the active grid; this check must not undo that.)
    QVERIFY(storedLevelsConsistentWithFrame({59943, 20811, 20811}, sparse));
    const Levels full0{{0, {75784, 32693, 32693}, {64, 64, 64}}};
    QVERIFY(storedLevelsConsistentWithFrame({75784, 32693, 32693}, full0));
    // Padded to whole storage chunks: fine. Padded beyond one chunk: not.
    QVERIFY(storedLevelsConsistentWithFrame(frame, {{3, {7552, 2624, 2624}, {64, 64, 64}}}));
    QVERIFY(!storedLevelsConsistentWithFrame(frame, {{3, {7552, 2624, 2624}, {32, 32, 32}}}));
    // The /3 array attached as level 0 is not the frame's pyramid.
    QVERIFY(!storedLevelsConsistentWithFrame(frame, {{0, {7493, 2602, 2602}, {64, 64, 64}}}));
    // A different scroll's pyramid is not either; nothing stored is not. (A
    // frame within one storage chunk x 2^level of the real one is
    // indistinguishable at that level by design; the frame's dyadic match
    // against the active grid is what decides those.)
    QVERIFY(!storedLevelsConsistentWithFrame({40000, 20700, 20700}, sparse));
    QVERIFY(!storedLevelsConsistentWithFrame(frame, Levels{}));
}

QTEST_APPLESS_MAIN(TestLineAnnotationPresenceOverlay)
#include "test_line_annotation_presence_overlay.moc"
