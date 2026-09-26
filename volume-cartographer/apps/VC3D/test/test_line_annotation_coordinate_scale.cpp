#include <QtTest/QtTest>

#include "LineAnnotationCoordinateScale.hpp"

class TestLineAnnotationCoordinateScale : public QObject {
    Q_OBJECT

private slots:
    void acceptsSameLevelOffByOneDomains();
    void acceptsDyadicNormalAndFiberDomains();
    void mapsFiberBaseCoordinatesIntoDownsampledVolume();
    void preservesLegacyUnspecifiedDomains();
    void rejectsIncompatibleDomains();
    void ordersManifestCandidatesFiberFirstThenSelection();
    void omitsEmptyAndDuplicateManifestCandidates();
    void channelFrameSubstitutionMustResolveAtTheOpenedLevel();
};

void TestLineAnnotationCoordinateScale::ordersManifestCandidatesFiberFirstThenSelection()
{
    // A fiber without a stored base shape names the manifest its trace spans
    // used; that path may have moved, so the package's selected
    // fiber-inference dataset follows it as the fallback.
    const auto candidates =
        vc3d::line_annotation::fiberBaseShapeManifestCandidates(
            "/old/volpkg/fibers/a.lasagna.json",
            "fiber_zarrs/a.lasagna.json");

    QCOMPARE(candidates.size(), std::size_t{2});
    QCOMPARE(candidates[0], std::string{"/old/volpkg/fibers/a.lasagna.json"});
    QCOMPARE(candidates[1], std::string{"fiber_zarrs/a.lasagna.json"});
}

void TestLineAnnotationCoordinateScale::omitsEmptyAndDuplicateManifestCandidates()
{
    QVERIFY(vc3d::line_annotation::fiberBaseShapeManifestCandidates("", "").empty());

    const auto selectionOnly =
        vc3d::line_annotation::fiberBaseShapeManifestCandidates(
            "", "fiber_zarrs/a.lasagna.json");
    QCOMPARE(selectionOnly.size(), std::size_t{1});
    QCOMPARE(selectionOnly[0], std::string{"fiber_zarrs/a.lasagna.json"});

    const auto same =
        vc3d::line_annotation::fiberBaseShapeManifestCandidates(
            "fiber_zarrs/a.lasagna.json", "fiber_zarrs/a.lasagna.json");
    QCOMPARE(same.size(), std::size_t{1});
}

void TestLineAnnotationCoordinateScale::mapsFiberBaseCoordinatesIntoDownsampledVolume()
{
    const std::optional<std::array<std::size_t, 3>> fiberShape{
        std::array<std::size_t, 3>{75784, 32694, 32694}};

    const double scale =
        vc3d::line_annotation::resolveFiberBaseToVolumeScale(
            fiberShape, {18946, 8174, 8174});

    QCOMPARE(scale, 0.25);
}

void TestLineAnnotationCoordinateScale::acceptsSameLevelOffByOneDomains()
{
    const std::optional<std::array<std::size_t, 3>> normalShape{
        std::array<std::size_t, 3>{75784, 32693, 32693}};
    const std::optional<std::array<std::size_t, 3>> fiberShape{
        std::array<std::size_t, 3>{75784, 32694, 32694}};

    const auto scales =
        vc3d::line_annotation::resolveFiberNormalCoordinateScales(
            normalShape, fiberShape, 8.0);

    QCOMPARE(scales.fiberBaseToNormalBase, 1.0);
    QCOMPARE(scales.traceToNormalBase, 8.0);
}

void TestLineAnnotationCoordinateScale::acceptsDyadicNormalAndFiberDomains()
{
    const std::optional<std::array<std::size_t, 3>> normalShape{
        std::array<std::size_t, 3>{18946, 8174, 8174}};
    const std::optional<std::array<std::size_t, 3>> fiberShape{
        std::array<std::size_t, 3>{75784, 32694, 32694}};

    const auto scales =
        vc3d::line_annotation::resolveFiberNormalCoordinateScales(
            normalShape, fiberShape, 8.0);

    QCOMPARE(scales.fiberBaseToNormalBase, 0.25);
    QCOMPARE(scales.traceToNormalBase, 2.0);
}

void TestLineAnnotationCoordinateScale::preservesLegacyUnspecifiedDomains()
{
    const auto scales =
        vc3d::line_annotation::resolveFiberNormalCoordinateScales(
            std::nullopt, std::nullopt, 4.0);

    QCOMPARE(scales.fiberBaseToNormalBase, 1.0);
    QCOMPARE(scales.traceToNormalBase, 4.0);
}

void TestLineAnnotationCoordinateScale::rejectsIncompatibleDomains()
{
    const std::optional<std::array<std::size_t, 3>> normalShape{
        std::array<std::size_t, 3>{18946, 8175, 8174}};
    const std::optional<std::array<std::size_t, 3>> fiberShape{
        std::array<std::size_t, 3>{75784, 32694, 32694}};

    QVERIFY_EXCEPTION_THROWN(
        vc3d::line_annotation::resolveFiberNormalCoordinateScales(
            normalShape, fiberShape, 1.0),
        std::runtime_error);
}

void TestLineAnnotationCoordinateScale::channelFrameSubstitutionMustResolveAtTheOpenedLevel()
{
    using vc3d::line_annotation::resolveFiberBaseToVolumeScale;
    // frameVolume() may stand a channel opened at level 1 onto the scan's
    // level-1 twin only when the fiber's base frame (the manifest's) maps
    // onto that twin: exactly the ceiling or floor at a non-zero level.
    const std::optional<std::array<std::size_t, 3>> manifest{{59944, 20812, 20812}};
    QCOMPARE(resolveFiberBaseToVolumeScale(manifest, {29972, 10406, 10406}), 0.5);
    // A scan recorded one voxel larger (20813 -> 10407 at level 1) agrees
    // with the manifest within one voxel after ceiling division, which the
    // pyramid predicate accepts, but no fiber maps onto it: substituting it
    // would fail every consumer, so the channel keeps its own frame.
    bool threw = false;
    try {
        (void)resolveFiberBaseToVolumeScale(manifest, {29972, 10407, 10407});
    } catch (const std::exception&) {
        threw = true;
    }
    QVERIFY(threw);
    // At level 0 the one-voxel convention difference is accepted.
    QCOMPARE(resolveFiberBaseToVolumeScale(manifest, {59944, 20813, 20813}), 1.0);
}

QTEST_APPLESS_MAIN(TestLineAnnotationCoordinateScale)
#include "test_line_annotation_coordinate_scale.moc"
