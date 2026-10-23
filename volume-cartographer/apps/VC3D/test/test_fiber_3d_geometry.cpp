#include "fibers3d/Fiber3DGeometry.hpp"

#include <QTest>

using namespace vc3d::fibers3d;

class Fiber3DGeometryTest : public QObject
{
    Q_OBJECT
private slots:
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
