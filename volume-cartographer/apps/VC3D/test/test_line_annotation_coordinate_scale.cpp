#include <QtTest/QtTest>
#include <cstring>
#include <limits>

#include "LineAnnotationCoordinateScale.hpp"

class TestLineAnnotationCoordinateScale : public QObject {
    Q_OBJECT

private slots:
    void fieldlessSnapshotPreservesStoredBytes();
    void failedSnapshotScaleRequiresAnOpenField_data();
    void failedSnapshotScaleRequiresAnOpenField();
    void acceptsSameLevelOffByOneDomains();
    void acceptsDyadicNormalAndFiberDomains();
    void mapsFiberBaseCoordinatesIntoDownsampledVolume();
    void preservesLegacyUnspecifiedDomains();
    void rejectsIncompatibleDomains();
    void ordersManifestCandidatesFiberFirstThenSelection();
    void omitsEmptyAndDuplicateManifestCandidates();
};

void TestLineAnnotationCoordinateScale::fieldlessSnapshotPreservesStoredBytes()
{
    using namespace vc3d::line_annotation;
    const std::vector<cv::Vec3d> stored{cv::Vec3d(-0.0, 16000.0, 10000.0),
                                        cv::Vec3d(16100.0, 16000.0, 10000.0)};
    // Both valid-but-different and incompatible stamps are irrelevant on
    // the legacy path; no scale resolution or arithmetic should run.
    for (const auto& base : {std::array<std::size_t, 3>{32768, 32768, 32768},
                            std::array<std::size_t, 3>{123, 456, 789}}) {
        auto controls = stored;
        auto line = stored;
        scaleFiberMapGeometry(controls, line, fiberMapFrameScale(base, std::array<int, 3>{65536, 65536, 65536}, 1.0, false), false);
        QCOMPARE(std::memcmp(controls.data(), stored.data(), stored.size() * sizeof(cv::Vec3d)), 0);
        QCOMPARE(std::memcmp(line.data(), stored.data(), stored.size() * sizeof(cv::Vec3d)), 0);
    }
    // A selected field that fails to open is also the legacy geometry path.
    auto controls = stored;
    auto line = stored;
    const auto frameScale = fiberMapFrameScale(
        std::array<std::size_t, 3>{32768, 32768, 32768}, std::array<int, 3>{65536, 65536, 65536}, 1.0, true);
    QCOMPARE(frameScale.value, 2.0);
    QVERIFY(frameScale.error.empty());
    scaleFiberMapGeometry(controls, line, frameScale, false);
    QCOMPARE(std::memcmp(controls.data(), stored.data(), stored.size() * sizeof(cv::Vec3d)), 0);
    QCOMPARE(std::memcmp(line.data(), stored.data(), stored.size() * sizeof(cv::Vec3d)), 0);
}

void TestLineAnnotationCoordinateScale::failedSnapshotScaleRequiresAnOpenField_data()
{
    QTest::addColumn<int>("failure");
    QTest::newRow("incompatible-stamp") << 0;
    QTest::newRow("missing-volume") << 1;
    QTest::newRow("nonpositive-volume") << 2;
    QTest::newRow("zero-frame-factor") << 3;
    QTest::newRow("nan-frame-factor") << 4;
    QTest::newRow("infinite-frame-factor") << 5;
}

void TestLineAnnotationCoordinateScale::failedSnapshotScaleRequiresAnOpenField()
{
    using namespace vc3d::line_annotation;
    QFETCH(int, failure);
    const std::array<std::size_t, 3> base{32768, 32768, 32768};
    std::optional<std::array<int, 3>> volume{{65536, 65536, 65536}};
    double factor = 1.0;
    switch (failure) {
    case 0: (*volume)[0] = 60000; break;
    case 1: volume.reset(); break;
    case 2: (*volume)[0] = 0; break;
    case 3: factor = 0.0; break;
    case 4: factor = std::numeric_limits<double>::quiet_NaN(); break;
    case 5: factor = std::numeric_limits<double>::infinity(); break;
    }
    // Exercise the same snapshot helper and post-open conversion as the
    // controller and worker. Snapshot creation itself must not throw.
    const auto scale = fiberMapFrameScale(base, volume, factor, true);
    const std::vector<cv::Vec3d> stored{cv::Vec3d(-0.0, 16000.0, 10000.0)};
    auto controls = stored;
    auto line = stored;
    // A selected field that failed to open preserves main's bytes even
    // though this fiber cannot be converted to its coordinate frame.
    scaleFiberMapGeometry(controls, line, scale, false);
    QCOMPARE(std::memcmp(controls.data(), stored.data(), sizeof(cv::Vec3d)), 0);
    QCOMPARE(std::memcmp(line.data(), stored.data(), sizeof(cv::Vec3d)), 0);
    QVERIFY_EXCEPTION_THROWN(scaleFiberMapGeometry(controls, line, scale, true), std::runtime_error);
    QVERIFY(!scale.error.empty());
    QCOMPARE(std::memcmp(controls.data(), stored.data(), sizeof(cv::Vec3d)), 0);
    QCOMPARE(std::memcmp(line.data(), stored.data(), sizeof(cv::Vec3d)), 0);
    // No selection ignores the invalid conversion altogether.
    const auto fieldless = fiberMapFrameScale(base, volume, factor, false);
    QVERIFY(fieldless.error.empty());
    scaleFiberMapGeometry(controls, line, fieldless, false);
    QCOMPARE(std::memcmp(line.data(), stored.data(), sizeof(cv::Vec3d)), 0);
}

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

QTEST_APPLESS_MAIN(TestLineAnnotationCoordinateScale)
#include "test_line_annotation_coordinate_scale.moc"
