#include "fibers3d/Fiber3DData.hpp"
#include "LineAnnotationPolylineIndices.hpp"
#include "../../../core/test/FiberCollectionFixture.hpp"

#include <QtTest/QtTest>

using namespace vc3d::fibers3d;
using vc::fibers::test::FixtureFiber;
using vc::fibers::test::FixtureFile;

namespace
{
AutomatedSource automated(const FixtureFile& fixture)
{
    return {QString::fromStdString(fixture.path().string()), QStringLiteral("7f3c1a52-8f0e-4b8e-9d1a-2f6b0c4e5a91"), 0.5, 0.008, {{-1000, -1000, -1000}, {1000, 1000, 1000}}};
}

LineAnnotationController::FiberMapSnapshot annotated()
{
    LineAnnotationController::FiberMapSnapshot snapshot;
    snapshot.frame.factor = 2;
    snapshot.frame.extentXyz = {2000, 2000, 2000};
    snapshot.voxelSizeUm = 8.0;
    LineAnnotationController::FiberMapFiber fiber;
    fiber.id = 42;
    fiber.fileName = "source-fiber.json";
    fiber.label = QStringLiteral("source-42");
    fiber.hvTag = 'H';
    fiber.linePoints = {{-10, 0, 0}, {10, 0, 0}, {10, 10, 0}, {-10, 10, 0}, {-10, 0, 0}, {10, 0, 0}};
    fiber.controlPoints = {fiber.linePoints[0], fiber.linePoints[1], fiber.linePoints[4], fiber.linePoints[5]};
    fiber.gapSegments = {false, false, true};
    fiber.damagedSegments = {true, false, false};
    fiber.breaks = {false, false, true, true};
    fiber.kollesisTerminations = {true, false, false, false};
    snapshot.fibers.push_back(std::move(fiber));
    return snapshot;
}
}  // namespace

class Fiber3DDataTest : public QObject
{
    Q_OBJECT
private slots:
    void nativeRegionsKeepDisjointRunsAndFullSourceLength()
    {
        auto snapshot = annotated();
        QueryOptions options;
        options.bounds = Bounds{{-1, -1, -1}, {1, 1, 1}};
        LineAnnotationController::FiberSummary summary;
        summary.id = 42;
        summary.tags = {"reviewed", "custom-tag"};
        const auto scene = loadAnnotations(snapshot, {summary}, options);
        QVERIFY(scene.complete);
        QCOMPARE(scene.voxelSizeMm, 0.016);
        QCOMPARE(scene.fibers.size(), size_t(1));
        const auto& fiber = scene.fibers[0];
        QCOMPARE(fiber.key, QStringLiteral("annotation:42"));
        QCOMPARE(fiber.fileName, QStringLiteral("source-fiber.json"));
        QCOMPARE(fiber.lengthMm, 80.0 * 0.008);
        QCOMPARE(fiber.runs.size(), size_t(2));
        QCOMPARE(fiber.runs[0].firstPoint, int64_t(0));
        QCOMPARE(fiber.runs[1].firstPoint, int64_t(4));
        // Neither endpoint is inside the ROI: segment/box clipping must retain
        // the crossing, and preserve original endpoints for exact GPU clipping.
        QCOMPARE(fiber.runs[0].points.front()[0], -5.0);
        QCOMPARE(fiber.runs[0].points.back()[0], 5.0);
        QCOMPARE(fiber.runs[0].styles[0], SpanStyle::Damaged);
        QCOMPARE(fiber.runs[1].styles[0], SpanStyle::Gap);
        QCOMPARE(fiber.tags, QStringList({"reviewed", "custom-tag"}));
        QCOMPARE(fiber.markers.size(), size_t(7));
        // This folded/repeating geometry must map the final control to its
        // later occurrence, not to the same nearest point as the first control.
        const auto indices = vc3d::line_annotation::orderedControlPointLineIndices(snapshot.fibers[0].controlPoints, snapshot.fibers[0].linePoints);
        QVERIFY(indices.has_value());
        QCOMPARE(*indices, std::vector<size_t>({0, 1, 4, 5}));
    }

    void nativeFrameAndMissingPhysicalScaleAreDistinct()
    {
        auto snapshot = annotated();
        snapshot.voxelSizeUm.reset();
        auto scene = loadAnnotations(snapshot, {}, {});
        QVERIFY(std::isnan(scene.voxelSizeMm));
        QVERIFY(std::isnan(scene.fibers[0].lengthMm));
        QCOMPARE(scene.fibers[0].runs[0].points[0][0], -5.0);
        QueryOptions options;
        options.minLengthMm = 0.1;
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadAnnotations(snapshot, {}, options));
        snapshot.frame.extentXyz = {};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadAnnotations(snapshot, {}, {}));
    }

    void nativeBudgetAndCancellationAreExplicit()
    {
        auto snapshot = annotated();
        QueryOptions options;
        options.byteBudget = 2 * sizeof(Point);
        const auto scene = loadAnnotations(snapshot, {}, options);
        QVERIFY(!scene.complete);
        QVERIFY(!scene.message.isEmpty());
        QCOMPARE(scene.fibers[0].runs[0].points.size(), size_t(2));
        const std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadAnnotations(snapshot, {}, options, &cancelled));
    }

    void nativeIdentitySeparatesSameFilenameFromDifferentLoadedSources()
    {
        auto snapshot = annotated();
        auto second = snapshot.fibers[0];
        second.id = 43;
        snapshot.fibers.push_back(second);
        QueryOptions options;
        options.selectedKey = QStringLiteral("annotation:43");
        const auto scene = loadAnnotations(snapshot, {}, options);
        QCOMPARE(scene.fibers.size(), size_t(2));
        QCOMPARE(scene.fibers[0].key, QStringLiteral("annotation:43"));
        QCOMPARE(scene.fibers[1].key, QStringLiteral("annotation:42"));
        QCOMPARE(scene.fibers[0].fileName, scene.fibers[1].fileName);
    }

    void emptyNativeSourceDoesNotRequireAFrame()
    {
        const auto scene = loadAnnotations({}, {}, {});
        QVERIFY(scene.complete);
        QVERIFY(scene.fibers.empty());
    }

    void readOnlyInspectionCopyIsNotAProjectAnnotation()
    {
        auto snapshot = annotated();
        auto inspection = snapshot.fibers[0];
        inspection.id = 43;
        inspection.readOnly = true;
        snapshot.fibers.push_back(inspection);
        auto scene = loadAnnotations(snapshot, {}, {});
        QCOMPARE(scene.fibers.size(), size_t(1));
        QCOMPARE(scene.fibers[0].id, uint64_t(42));
        snapshot.fibers = {inspection};
        snapshot.frame.extentXyz = {};
        scene = loadAnnotations(snapshot, {}, {});
        QVERIFY(scene.complete);
        QVERIFY(scene.fibers.empty());
    }

    void afvRegionsPreserveOriginalIndicesAndInferredGaps()
    {
        const FixtureFile fixture(
            {{{{-10, 0, 0}, {10, 0, 0}, {10, 10, 0}, {-10, 10, 0}, {-10, 0, 0}, {10, 0, 0}},
              "V",
              R"({"type":"vc3d_fiber","version":1,"tags":["reviewed"],"inferred_gap_point_ranges":[[4,5]]})"}});
        QueryOptions options;
        options.bounds = Bounds{{-1, -1, -1}, {1, 1, 1}};
        const auto scene = loadAutomated(automated(fixture), options);
        QVERIFY(scene.complete);
        QCOMPARE(scene.voxelSizeMm, 0.016);
        QCOMPARE(scene.fibers.size(), size_t(1));
        const auto& fiber = scene.fibers[0];
        QCOMPARE(fiber.lengthMm, 0.64);
        QCOMPARE(fiber.family, 'V');
        QCOMPARE(fiber.key, QStringLiteral("7f3c1a52-8f0e-4b8e-9d1a-2f6b0c4e5a91:1"));
        QCOMPARE(fiber.runs.size(), size_t(2));
        QCOMPARE(fiber.runs[0].firstPoint, int64_t(0));
        QCOMPARE(fiber.runs[1].firstPoint, int64_t(4));
        QCOMPARE(fiber.runs[1].styles[0], SpanStyle::InferredGap);
        QCOMPARE(fiber.runs[1].points.front()[0], -5.0);
        QCOMPARE(fiber.tags, QStringList({"reviewed"}));
    }

    void afvSpanAnnotationsMapControlsOutsideVisibleRegionAcrossBlocks()
    {
        std::vector<vc::fibers::Point> points;
        for (int i = 0; i < 700; ++i)
            points.push_back({double(i), 0, 0});
        const FixtureFile fixture({{points, "H", R"({"type":"vc3d_fiber","version":4,
            "control_points":[
              {"position":[10,0,0],"tags":["break"],"segment_to_next":{"tags":["gap"]}},
              {"position":[690,0,0],"tags":["break","kollesis_termination"]}]})"}});
        QueryOptions options;
        options.bounds = Bounds{{150, -1, -1}, {151, 1, 1}};
        const auto scene = loadAutomated(automated(fixture), options);
        QVERIFY(scene.complete);
        QCOMPARE(scene.fibers.size(), size_t(1));
        const auto& fiber = scene.fibers[0];
        QVERIFY(!fiber.runs.empty());
        for (const auto& run : fiber.runs) {
            QVERIFY(!run.styles.empty());
            for (const auto style : run.styles)
                QCOMPARE(style, SpanStyle::Gap);
        }
        QCOMPARE(fiber.markers.size(), size_t(5));
        QCOMPARE(fiber.lengthMm, 699 * 0.008);
    }

    void afvLimitReportsPartialAndRetainsSelectedFiber()
    {
        std::vector<FixtureFiber> fibers;
        for (int i = 0; i < 20; ++i)
            fibers.push_back({{{-10, 0, 0}, {10, 0, 0}}, "H"});
        const FixtureFile fixture(fibers);
        auto source = automated(fixture);
        QueryOptions options;
        options.maximumFibers = 2;
        options.selectedKey = source.uuid + ":19";
        const auto a = loadAutomated(source, options);
        const auto b = loadAutomated(source, options);
        QVERIFY(!a.complete);
        QVERIFY(!a.message.isEmpty());
        QCOMPARE(a.fibers.size(), size_t(2));
        QCOMPARE(a.fibers[0].id, uint64_t(19));
        QCOMPARE(a.fibers[1].id, b.fibers[1].id);
        options.byteBudget = 48;
        QVERIFY(!loadAutomated(source, options).complete);
    }

    void afvFamilyFiltersDoNotSpendBudgetOnExcludedFibers()
    {
        const FixtureFile fixture({{{{-10, 0, 0}, {10, 0, 0}}, "H"}, {{{-10, 0, 0}, {10, 0, 0}}, "V"}, {{{-10, 0, 0}, {10, 0, 0}}, ""}});
        for (int mask = 0; mask < 8; ++mask) {
            QueryOptions options;
            options.horizontal = mask & 1;
            options.vertical = mask & 2;
            options.unclassified = mask & 4;
            const auto scene = loadAutomated(automated(fixture), options);
            QCOMPARE(scene.fibers.size(), size_t(bool(mask & 1) + bool(mask & 2) + bool(mask & 4)));
            QVERIFY(scene.complete);
        }
    }

    void afvRejectsChangedIdentityAndCancellation()
    {
        const FixtureFile fixture({{{{-10, 0, 0}, {10, 0, 0}}, "H"}});
        auto source = automated(fixture);
        source.uuid = "other-collection";
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadAutomated(source, {}));
        const std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadAutomated(automated(fixture), {}, &cancelled));
    }

    void afvLengthFilterUsesWholeFiberInNativeScale()
    {
        const FixtureFile fixture({{{{-100, 0, 0}, {100, 0, 0}}, "H"}});
        QueryOptions options;
        options.bounds = Bounds{{-1, -1, -1}, {1, 1, 1}};
        options.minLengthMm = 1.5;
        const auto scene = loadAutomated(automated(fixture), options);
        QCOMPARE(scene.fibers.size(), size_t(1));
        QCOMPARE(scene.fibers[0].lengthMm, 1.6);
        options.minLengthMm = 1.7;
        QVERIFY(loadAutomated(automated(fixture), options).fibers.empty());
    }
};

QTEST_APPLESS_MAIN(Fiber3DDataTest)
#include "test_fiber_3d_data.moc"
