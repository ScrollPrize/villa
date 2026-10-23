#include "fibers3d/Fiber3DGeometry.hpp"
#include "fibers3d/Fiber3DRenderGeometry.hpp"

#include <QTest>

using namespace vc3d::fibers3d;

class Fiber3DGeometryTest : public QObject
{
    Q_OBJECT
private slots:
    void simplificationBoundsAndHairpins()
    {
        std::vector<Point> points;
        for (int i = 0; i < 129; ++i) points.push_back({double(i), std::sin(i * .13) * 4, std::cos(i * .07)});
        for (double tolerance : render::tolerances) {
            std::vector<size_t> kept{0}; render::simplify(points, 0, points.size() - 1, tolerance, kept);
            QCOMPARE(kept.front(), size_t(0)); QCOMPARE(kept.back(), points.size() - 1);
            for (size_t k = 1; k < kept.size(); ++k)
                for (size_t i = kept[k - 1]; i <= kept[k]; ++i)
                    QVERIFY(render::distanceToSegment(points[i], points[kept[k - 1]], points[kept[k]]) <= tolerance + 1.e-12);
        }
        points = {{0, 0, 0}, {100, 0, 0}, {1, 0, 0}};
        std::vector<size_t> kept{0}; render::simplify(points, 0, 2, 1, kept);
        QCOMPARE(kept, (std::vector<size_t>{0, 1, 2}));
    }

    void lodKeepsRunsStylesAndSourceIndices()
    {
        auto scene = std::make_shared<Scene>();
        scene->clipBounds = Bounds{{0, -1, -1}, {10, 1, 1}};
        Fiber f; f.key = "one";
        f.runs = {Run{{{-2, 0, 0}, {2, 0, 0}, {5, 0, 0}, {7, 0, 0}, {12, 0, 0}}, 100,
                      {SpanStyle::Solid, SpanStyle::Solid, SpanStyle::Gap, SpanStyle::Gap}},
                  Run{{{1, .5, 0}, {9, .5, 0}}, 900}};
        scene->fibers.push_back(f);
        auto g = render::build(scene);
        QCOMPARE(g->levels[0].size(), size_t(5));
        for (const auto& c : g->chunks) {
            const auto& run = scene->fibers[c.fiber].runs[c.run];
            const auto style = c.first < run.styles.size() ? run.styles[c.first] : SpanStyle::Solid;
            for (size_t i = c.first; i < c.end; ++i) {
                QCOMPARE(i < run.styles.size() ? run.styles[i] : SpanStyle::Solid, style);
                render::Segment s; QVERIFY(render::segment(*g, c, i, s));
                QCOMPARE(s.originalIndex, run.firstPoint + int64_t(i));
                QVERIFY(s.a[0] >= 0 && s.b[0] <= 10);
            }
            for (size_t level = 0; level < render::tolerances.size(); ++level) {
                const auto r = c.levels[level]; QVERIFY(r.count > 0);
                for (size_t j = r.first; j < r.first + r.count; ++j)
                    QCOMPARE(g->levels[level][j].style, float(style));
            }
        }
        QCOMPARE(scene->fibers[0].runs[0].points[0][0], -2.); // never mutates source
    }

    void invalidPointsAndReentryNeverJoin()
    {
        auto scene = std::make_shared<Scene>();
        scene->clipBounds = Bounds{{0, 0, 0}, {10, 10, 10}};
        Fiber f;
        f.runs = {Run{{{1, 1, 1}, {20, 1, 1}, {20, 9, 1}, {1, 9, 1},
                      {NAN, 0, 0}, {2, 2, 2}, {3, 3, 3}}}};
        scene->fibers.push_back(f); auto g = render::build(scene);
        QCOMPARE(g->chunks.size(), size_t(3));
        for (auto& level : g->levels) QCOMPARE(level.size(), size_t(3));
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, render::build(scene, &cancelled));
    }

    void cullingAndPickingIncludeCrossingSegments()
    {
        auto scene = std::make_shared<Scene>();
        for (int i = 0; i < 100; ++i) {
            Fiber f; f.runs = {Run{{{-100., double(i), 0}, {100., double(i), 0}}}};
            scene->fibers.push_back(f);
        }
        auto g = render::build(scene);
        QMatrix4x4 m; m.ortho(-2.f, 2.f, -2.f, 2.f, -100.f, 100.f);
        // Matrix is relative to the geometry origin, looking at world Y=0.
        m.translate(float(g->origin[0]), float(g->origin[1]), float(g->origin[2]));
        render::View view(m, g->origin, 800, 600, 2);
        const auto candidates = render::pickChunks(*g, view, 0, 0);
        QCOMPARE(candidates.size(), size_t(1)); QCOMPARE(g->chunks[candidates[0]].fiber, size_t(0));
        auto draws = render::draws(*g, view, .5);
        QVERIFY(!draws.empty()); size_t count = 0;
        for (const auto& d : draws) count += d.range.count;
        QVERIFY(count >= 3 && count < 10); // y=0,1,2 intersect, others culled
        QVERIFY(view.classify(Bounds{{-100, 0, 0}, {100, 0, 0}}) >= 0);
        QVERIFY(view.classify(Bounds{{10, 10, 0}, {11, 11, 0}}) < 0);
    }

    void perspectiveErrorBoundAndEyePlane()
    {
        QMatrix4x4 m; m.perspective(45.f, 1.5f, .1f, 1000.f);
        render::View view(m, {}, 1200, 800, 4);
        const Bounds b{{-2, -1, -12}, {2, 1, -8}};
        const double rate = view.pixelsPerUnit(b);
        auto project = [&](const Point& p) {
            const auto q = m * QVector4D(float(p[0]), float(p[1]), float(p[2]), 1);
            return QPointF(q.x() / q.w() * 600, q.y() / q.w() * 400);
        };
        for (int i = 0; i < 50; ++i) {
            Point a{-1.8 + i * .07, -.8 + i * .03, -11.8 + i * .07};
            Point c{a[0] + .02, a[1] - .03, a[2] + .04};
            const auto delta = project(a) - project(c);
            QVERIFY(std::hypot(delta.x(), delta.y()) <= rate * render::distance(a, c));
        }
        QVERIFY(std::isinf(view.pixelsPerUnit(Bounds{{-1, -1, -1}, {1, 1, 1}})));
        QVERIFY(view.classify(Bounds{{-1, -1, 1}, {1, 1, 2}}) < 0);
    }

    void crossingWithBothEndpointsOutside()
    {
        const Bounds box{{0, 0, 0}, {10, 10, 10}};
        const Point a{-10, 5, 5}, b{20, 5, 5};
        double lo, hi;
        QVERIFY(clipSegment(a, b, box, lo, hi));
        QCOMPARE(lo, 1.0 / 3.0);
        QCOMPARE(hi, 2.0 / 3.0);
        QCOMPARE(interpolate(a, b, lo)[0], 0.0);
        QCOMPARE(interpolate(a, b, hi)[0], 10.0);
        QVERIFY(!clipSegment(Point{-10, 12, 5}, Point{20, 12, 5}, box, lo, hi));
    }

    void coordinatesRemainDoublePrecision()
    {
        const double origin = 1.0e12;
        const Bounds box{{origin + .25, 0, 0}, {origin + .75, 10, 10}};
        const Point a{origin, 5, 5}, b{origin + 1, 5, 5};
        double lo, hi;
        QVERIFY(clipSegment(a, b, box, lo, hi));
        QCOMPARE(lo, .25);
        QCOMPARE(hi, .75);
    }

    void pointMarkersAndInvalidInputs()
    {
        const Bounds box{{0, 0, 0}, {10, 10, 10}};
        double lo, hi;
        QVERIFY(clipSegment(Point{5, 5, 5}, Point{5, 5, 5}, box, lo, hi));
        QVERIFY(!clipSegment(Point{11, 5, 5}, Point{11, 5, 5}, box, lo, hi));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        QVERIFY(!clipSegment(Point{nan, 5, 5}, Point{1, 5, 5}, box, lo, hi));
        QVERIFY(!clipSegment(Point{1, 5, 5}, Point{2, 5, 5}, Bounds{{3, 0, 0}, {1, 10, 10}}, lo, hi));
    }

    void directionHasNoPreferredSign()
    {
        const Point a{2, 3, 4}, b{-1, 7, 4};
        const auto ab = axialDirection(a, b), ba = axialDirection(b, a);
        QCOMPARE(ab, ba);
        QCOMPARE(ab[0], .6);
        QCOMPARE(ab[1], .8);
        QCOMPARE(ab[2], 0.0);
        QCOMPARE(axialDirection(a, a), Point{});
    }

    void lengthRangeUsesWholeFibers()
    {
        Scene scene;
        Fiber shortFiber, longFiber, unknownFiber;
        shortFiber.lengthMm = 3;
        longFiber.lengthMm = 100;
        longFiber.runs = {Run{{Point{0, 0, 0}, Point{1, 0, 0}}}};
        scene.fibers = {shortFiber, longFiber, unknownFiber};
        DisplaySettings settings;
        QCOMPARE(lengthRange(scene, settings), std::make_pair(3.0, 100.0));
        settings.fixedLengthRange = true;
        settings.lengthMinMm = 1;
        settings.lengthMaxMm = 20;
        QCOMPARE(lengthRange(scene, settings), std::make_pair(1.0, 20.0));
        settings.lengthMaxMm = settings.lengthMinMm;
        QCOMPARE(lengthRange(scene, settings), std::make_pair(3.0, 100.0));
    }
};

QTEST_APPLESS_MAIN(Fiber3DGeometryTest)
#include "test_fiber_3d_geometry.moc"
