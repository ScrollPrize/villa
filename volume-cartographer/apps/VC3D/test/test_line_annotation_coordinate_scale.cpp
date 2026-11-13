#include <QtTest/QtTest>

#include "LineAnnotationCoordinateScale.hpp"
#include "LineAnnotationNormalSource.hpp"

class TestLineAnnotationCoordinateScale : public QObject {
    Q_OBJECT

private slots:
    void acceptsSameLevelOffByOneDomains();
    void acceptsDyadicNormalAndFiberDomains();
    void preservesLegacyUnspecifiedDomains();
    void rejectsIncompatibleDomains();
    void normalSourceSelection();
};

void TestLineAnnotationCoordinateScale::normalSourceSelection()
{
    using vc3d::line_annotation::usePredictedFiberNormals;
    auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version":3,"source_to_base":1,"groups":{"sd2":{
        "zarr":"compound/sd2/2","scaledown":2,
        "channels":["presence","nx","ny","width","sheet_nx","sheet_ny","sheet_grad_mag"]}}})");
    QVERIFY_EXCEPTION_THROWN(usePredictedFiberNormals(manifest, "auto"), std::invalid_argument);
    QVERIFY(usePredictedFiberNormals(manifest, "fiber"));
    QVERIFY(!usePredictedFiberNormals(manifest, "lasagna"));
    manifest.workingToBaseScale = 4;
    const auto normals = vc::lasagna::fiberNormalManifest(manifest);
    QCOMPARE(normals.workingToBaseScale, 4.0);
    QCOMPARE(*normals.groupForChannel("nx")->channelIndex("nx"), size_t(4));
    QCOMPARE(*normals.groupForChannel("ny")->channelIndex("ny"), size_t(5));
    QCOMPARE(*normals.groupForChannel("grad_mag")->channelIndex("grad_mag"), size_t(6));
    manifest.groups.front().channels.pop_back();
    QVERIFY(!usePredictedFiberNormals(manifest, "lasagna"));
    QVERIFY_EXCEPTION_THROWN(usePredictedFiberNormals(manifest, "fiber"), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(usePredictedFiberNormals(manifest, "typo"), std::invalid_argument);
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

QTEST_APPLESS_MAIN(TestLineAnnotationCoordinateScale)
#include "test_line_annotation_coordinate_scale.moc"
