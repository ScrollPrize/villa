#include "fibers3d/Fiber3DData.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include <vc_object_store/objects/FiberJson.hpp>
#include <vc_object_store/Uuid.hpp>
#include <QtTest/QtTest>
#include <QTemporaryDir>
using namespace vc3d::fibers3d;
using Json = nlohmann::json;
namespace
{
struct Fixture {
    QTemporaryDir dir;
    std::shared_ptr<vc::objects::ObjectStoreSession> session = std::make_shared<vc::objects::ObjectStoreSession>();
    vc::objects::AttachmentIdentity attachment;
    Fixture()
    {
        vc::objects::OpenOptions options;
        options.author = "fiber-3d-test";
        const auto path = dir.path().toStdString() + "/test.vcobj";
        {
            auto db = vc::objects::ObjectStore::create(path, options);
        }
        attachment = session->attach(path, options);
    }
    StoreSource source() { return {session, attachment, {"ct", 1}, 0.008, {}}; }
    std::string add(
        const std::string& family,
        const std::vector<Point>& points,
        double scale = 1,
        std::string volume = "ct",
        Json controls = Json::array(),
        Json extras = Json::object())
    {
        const auto id = vc::objects::util::newUuid();
        Json root =
            {{"type", "vc3d_fiber"},
             {"version", 4},
             {"uuid", id},
             {"name", family},
             {"coordinates", {{"volume_uuid", volume}, {"scale_factor", scale}}},
             {"line_points", points},
             {"control_points", controls},
             {"hv_classification", {{"automatic_tag", family}}}};
        root.update(extras);
        const auto entry = vc::objects::decodeFiberJson(root.dump());
        session->act(attachment, {{id, {}}}, [&](auto& db, auto& txn) { vc::objects::ingestGeometryBatch(db, txn, {entry}); });
        return id;
    }
};
}  // namespace
class Fiber3DDataTest : public QObject
{
    Q_OBJECT
private slots:
    void generatedSpansUseDisplayedLinePositionsAndGapPrecedence()
    {
        vc3d::line_annotation::GeneratedViews views;
        for (int i = 0; i < 9; ++i)
            views.linePoints.emplace_back(float(i), 0, 0);
        std::vector<vc3d::line_annotation::GeneratedOverlay::ControlPointMarker> controls(4);
        const int positions[] = {0, 2, 4, 8};
        for (int i = 0; i < 4; ++i) {
            controls[i].point = views.linePoints[positions[i]];
            controls[i].linePosition = (i + 1) * 100;
            controls[i].displayedLinePosition = positions[i];
        }
        controls[0].hasGapToNext = true;
        controls[0].hasDamagedToNext = true;
        controls[1].hasDamagedToNext = true;
        // Storage order and live line positions need not match the displayed line.
        views.controlPoints = {controls[2], controls[0], controls[3], controls[1]};
        const auto fiber = generatedAnnotationFiber(views);
        QCOMPARE(fiber.runs.size(), size_t(1));
        QCOMPARE(fiber.runs[0].styles, std::vector<SpanStyle>({SpanStyle::Gap, SpanStyle::Gap, SpanStyle::Damaged, SpanStyle::Damaged, SpanStyle::Solid, SpanStyle::Solid, SpanStyle::Solid, SpanStyle::Solid}));
        QCOMPARE(fiber.runs[0].points.size(), views.linePoints.size());
        for (size_t i = 0; i < views.linePoints.size(); ++i)
            QCOMPARE(fiber.runs[0].points[i], (Point{double(i), 0, 0}));
    }

    void generatedMarkersPreserveIndependentTags()
    {
        vc3d::line_annotation::GeneratedViews views;
        vc3d::line_annotation::GeneratedOverlay::ControlPointMarker control;
        control.point = {1, 2, 3};
        control.isBreak = true;
        control.isKollesisTermination = true;
        views.controlPoints.push_back(control);
        control.point = {4, 5, 6};
        control.isBreak = false;
        views.controlPoints.push_back(control);
        control.point = {7, 8, 9};
        control.isKollesisTermination = false;
        views.controlPoints.push_back(control);
        control.point[0] = std::numeric_limits<float>::quiet_NaN();
        views.controlPoints.push_back(control);
        const auto fiber = generatedAnnotationFiber(views);
        const std::vector<MarkerKind> kinds{MarkerKind::ControlPoint, MarkerKind::Break, MarkerKind::KollesisTermination, MarkerKind::ControlPoint, MarkerKind::KollesisTermination, MarkerKind::ControlPoint};
        QCOMPARE(fiber.markers.size(), kinds.size());
        for (size_t i = 0; i < kinds.size(); ++i)
            QCOMPARE(fiber.markers[i].kind, kinds[i]);
        QCOMPARE(fiber.markers[1].point, (Point{1, 2, 3}));
        QCOMPARE(fiber.markers[4].point, (Point{4, 5, 6}));
    }

    void generatedInvalidPointsSplitRunsWithoutLosingSpanIndices()
    {
        vc3d::line_annotation::GeneratedViews views;
        for (int i = 0; i < 9; ++i)
            views.linePoints.emplace_back(float(i), 0, 0);
        views.linePoints[0][0] = views.linePoints[3][0] = views.linePoints[4][0] = std::numeric_limits<float>::quiet_NaN();
        vc3d::line_annotation::GeneratedOverlay::ControlPointMarker a, b;
        a.point = views.linePoints[1];
        a.linePosition = 1;
        a.hasGapToNext = true;
        b.point = views.linePoints[7];
        b.linePosition = 7;
        views.controlPoints = {a, b};
        const auto fiber = generatedAnnotationFiber(views);
        QCOMPARE(fiber.runs.size(), size_t(2));
        QCOMPARE(fiber.runs[0].firstPoint, int64_t(1));
        QCOMPARE(fiber.runs[1].firstPoint, int64_t(5));
        QCOMPARE(fiber.runs[0].styles, std::vector<SpanStyle>({SpanStyle::Gap}));
        QCOMPARE(fiber.runs[1].styles, std::vector<SpanStyle>({SpanStyle::Gap, SpanStyle::Gap, SpanStyle::Solid}));
    }

    void generatedBreakMarkersDoNotInventGapSpans()
    {
        vc3d::line_annotation::GeneratedViews views;
        views.linePoints = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        for (int i : {0, 2}) {
            vc3d::line_annotation::GeneratedOverlay::ControlPointMarker control;
            control.point = views.linePoints[i];
            control.linePosition = i;
            control.isBreak = true;
            views.controlPoints.push_back(control);
        }
        const auto fiber = generatedAnnotationFiber(views);
        QCOMPARE(fiber.runs.size(), size_t(1));
        QVERIFY(fiber.runs[0].styles.empty());
        QCOMPARE(fiber.markers.size(), size_t(4));
    }

    void storeFramesAndIdentity()
    {
        Fixture f;
        const auto id = f.add("H", {{0, 0, 0}, {20, 0, 0}}, 2);
        f.add("V", {{0, 0, 0}, {100, 0, 0}}, 1, "unregistered");
        const auto scene = loadStore(f.source(), {});
        QCOMPARE(scene.fibers.size(), size_t(1));
        QCOMPARE(scene.fibers[0].objectUuid, id);
        QCOMPARE(scene.fibers[0].key, storeFiberKey(f.attachment, id));
        QCOMPARE(scene.fibers[0].runs[0].points.back(), (Point{10, 0, 0}));
        QCOMPARE(scene.fibers[0].lengthMm, 0.08);
    }
    void filtersSelectionAndLengthOrder()
    {
        Fixture f;
        const auto shortId = f.add("H", {{0, 0, 0}, {10, 0, 0}});
        f.add("V", {{0, 0, 0}, {80, 0, 0}});
        const auto longId = f.add("H", {{0, 0, 0}, {30, 0, 0}});
        QueryOptions o;
        o.maximumFibers = 1;
        o.vertical = false;
        auto scene = loadStore(f.source(), o);
        QCOMPARE(scene.fibers[0].objectUuid, longId);
        QVERIFY(!scene.complete);
        o.selectedKey = storeFiberKey(f.attachment, shortId);
        QCOMPARE(loadStore(f.source(), o).fibers[0].objectUuid, shortId);
        o.minLengthMm = 0.15;
        QCOMPARE(loadStore(f.source(), o).fibers[0].objectUuid, longId);
        o.horizontal = false;
        o.unclassified = false;
        QVERIFY(loadStore(f.source(), o).fibers.empty());
    }
    void exactRegionPreservesDisjointRunsAndLength()
    {
        Fixture f;
        const auto id = f.add("H", {{-10, 0, 0}, {10, 0, 0}, {10, 10, 0}, {-10, 10, 0}, {-10, 0, 0}, {10, 0, 0}});
        QueryOptions o;
        o.bounds = Bounds{{-1, -1, -1}, {1, 1, 1}};
        const auto scene = loadStore(f.source(), o);
        QCOMPARE(scene.fibers.size(), size_t(1));
        QCOMPARE(scene.fibers[0].runs.size(), size_t(2));
        QCOMPARE(scene.fibers[0].runs[1].firstPoint, int64_t(4));
        QCOMPARE(scene.fibers[0].lengthMm, 0.64);
        QCOMPARE(scene.fibers[0].runs[0].points.front(), (Point{-10, 0, 0}));
    }
    void annotationsAndKnownGapArePreserved()
    {
        Fixture f;
        Json controls = Json::array(
            {Json{{"position", {0, 0, 0}}, {"tags", {"break"}}, {"segment_to_next", {{"tags", {"gap"}}}}}, Json{{"position", {2, 0, 0}}}});
        f.add("H", {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}, 1, "ct", controls, Json{{"tags", {"reviewed"}}});
        const auto scene = loadStore(f.source(), {});
        const auto& fiber = scene.fibers[0];
        QCOMPARE(fiber.markers.size(), size_t(3));
        QCOMPARE(fiber.tags, QStringList{"reviewed"});
        QCOMPARE(fiber.runs[0].styles, std::vector<SpanStyle>({SpanStyle::Gap, SpanStyle::Gap}));
    }
    void budgetsCancellationAndDetach()
    {
        Fixture f;
        f.add("V", {{0, 0, 0}, {10, 0, 0}});
        QueryOptions o;
        o.byteBudget = 2 * sizeof(Point);
        QVERIFY(!loadStore(f.source(), o).complete);
        std::atomic_bool cancel{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, loadStore(f.source(), {}, &cancel));
        auto source = f.source();
        f.session->detach(f.attachment);
        QVERIFY_THROWS_EXCEPTION(std::exception, loadStore(source, {}));
    }
    void registeredCoordinatesAndNormalsFollowTheSameTransform()
    {
        Fixture f;
        Json cp =
            Json::array({Json{{"position", {0, 0, 0}}, {"display_normal", {1, 1, 0}}, {"direction", {1, 1, 0}}}, Json{{"position", {20, 0, 0}}}});
        const auto id = f.add("H", {{0, 0, 0}, {20, 0, 0}}, 2, "scan-b", cp);
        vc::objects::VolumeAffine matrix;
        matrix(0, 0) = 2;
        matrix(1, 1) = 3;
        matrix(0, 3) = 100;
        matrix(1, 3) = 200;
        vc::objects::registerVolumeTransform(*f.session, f.attachment, "scan-b", "ct", matrix, "test registration");
        QueryOptions o;
        o.bounds = Bounds{{99, 199, -1}, {121, 201, 1}};
        const auto scene = loadStore(f.source(), o);
        QCOMPARE(scene.fibers.size(), size_t(1));
        const auto& fiber = scene.fibers[0];
        QCOMPARE(fiber.runs[0].points.front(), (Point{100, 200, 0}));
        QCOMPARE(fiber.runs[0].points.back(), (Point{120, 200, 0}));
        QCOMPARE(fiber.lengthMm, .16);
        QVERIFY(fiber.displayNormals[0]);
        QVERIFY(fiber.controlDirections[0]);
        const auto n = *fiber.displayNormals[0], d = *fiber.controlDirections[0];
        QVERIFY(std::abs(n[0] / n[1] - 1.5) < 1e-12);
        QVERIFY(std::abs(d[0] / d[1] - 2.0 / 3.0) < 1e-12);
    }
    void pagingKeepsLongestFirstAcrossCatalogPages()
    {
        Fixture f;
        for (int i = 1; i <= 140; ++i)
            f.add(i % 2 ? "H" : "V", {{0, 0, 0}, {double(i), 0, 0}});
        QueryOptions o;
        o.maximumFibers = 0;
        const auto scene = loadStore(f.source(), o);
        QVERIFY(scene.complete);
        QCOMPARE(scene.fibers.size(), size_t(140));
        for (size_t i = 0; i < scene.fibers.size(); ++i)
            QVERIFY(std::abs(scene.fibers[i].lengthMm - double(140 - i) * .008) < 1e-12);
    }
    void singleObjectDoesNotFallBackToAnother()
    {
        Fixture f;
        f.add("V", {{0, 0, 0}, {10, 0, 0}});
        auto source = f.source();
        source.onlyUuid = vc::objects::util::newUuid();
        QVERIFY(loadStore(source, {}).fibers.empty());
    }
};
QTEST_GUILESS_MAIN(Fiber3DDataTest)
#include "test_fiber_3d_data.moc"
