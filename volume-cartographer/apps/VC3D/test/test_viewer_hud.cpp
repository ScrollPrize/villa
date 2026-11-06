// Coverage for ViewerHud.hpp: the histogram, line sampling, peak finding and
// volume grid behind the slice viewers' measurement overlays. All of it works
// on plain images and an affine map, so no viewer is involved.

#include <QtTest/QtTest>

#include "volume_viewers/ViewerHud.hpp"

#include <cmath>
#include <vector>

using namespace vc3d::hud;

namespace {

// Each column holds its x coordinate as its gray value.
QImage horizontalRamp(int width, int height)
{
    QImage img(width, height, QImage::Format_Grayscale8);
    for (int y = 0; y < height; ++y) {
        uchar* row = img.scanLine(y);
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<uchar>(x);
        }
    }
    return img;
}

AffineMap axisAlignedMap()
{
    // Viewport x is volume x, viewport y is volume y, the slice sits at z = 40.
    AffineMap map;
    map.origin = {0.0, 0.0, 40.0};
    map.dx = {1.0, 0.0, 0.0};
    map.dy = {0.0, 1.0, 0.0};
    return map;
}

}  // namespace

class TestViewerHud : public QObject
{
    Q_OBJECT

private slots:
    void optionsReportWhetherAnythingIsOn()
    {
        Options options;
        QVERIFY(!options.any());
        options.loupe = true;
        QVERIFY(options.any());
        QVERIFY(options != Options{});
    }

    void histogramCountsEveryValueOfARamp()
    {
        const Histogram h = computeHistogram(horizontalRamp(256, 4), 1);
        QCOMPARE(h.count, std::uint64_t(1024));
        for (int v = 0; v < 256; ++v) {
            QCOMPARE(h.bins[std::size_t(v)], std::uint32_t(4));
        }
        QCOMPARE(h.mean, 127.5);
        QCOMPARE(h.low, 2);
        QCOMPARE(h.high, 253);
        QCOMPARE(h.black, 1.0 / 256.0);
        QCOMPARE(h.white, 1.0 / 256.0);
    }

    void histogramStrideSkipsPixels()
    {
        const Histogram h = computeHistogram(horizontalRamp(256, 4), 2);
        QCOMPARE(h.count, std::uint64_t(256));
        QCOMPARE(h.bins[1], std::uint32_t(0));
        QCOMPARE(h.bins[2], std::uint32_t(2));
    }

    void histogramOfANullImageIsEmpty()
    {
        const Histogram h = computeHistogram(QImage(), 1);
        QCOMPARE(h.count, std::uint64_t(0));
    }

    void histogramReadsColorFramebuffers()
    {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(QColor(200, 200, 200));
        const Histogram h = computeHistogram(img, 1);
        QCOMPARE(h.bins[200], std::uint32_t(64));
    }

    void histogramCanLeaveOutBackground()
    {
        // Columns 0-127 stand for the black margin around a volume.
        const Histogram h = computeHistogram(horizontalRamp(256, 4), 1,
                                             [](int x, int) { return x >= 128; });
        QCOMPARE(h.count, std::uint64_t(512));
        QCOMPARE(h.bins[0], std::uint32_t(0));
        QCOMPARE(h.black, 0.0);
        QCOMPARE(h.low, 129);
    }

    void sampleLineInterpolatesAlongTheRamp()
    {
        const QImage img = horizontalRamp(64, 8);
        const std::vector<float> v = sampleLine(img, QPointF(10, 3), QPointF(20, 3));
        QCOMPARE(int(v.size()), 11);
        for (int i = 0; i < 11; ++i) {
            QCOMPARE(v[std::size_t(i)], float(10 + i));
        }
        const std::vector<float> half = sampleLine(img, QPointF(10.5, 3), QPointF(11.5, 3));
        QCOMPARE(half.front(), 10.5f);
        QCOMPARE(half.back(), 11.5f);
    }

    void sampleLineMarksOutsideSamplesAsMissing()
    {
        const QImage img = horizontalRamp(16, 16);
        const std::vector<float> v = sampleLine(img, QPointF(-4, 2), QPointF(4, 2));
        QVERIFY(std::isnan(v.front()));
        QCOMPARE(v.back(), 4.0f);
    }

    void findPeaksKeepsOnlyProminentOnes()
    {
        // Two clear peaks, a small bump on the slope between them, and a plateau.
        const std::vector<float> v{0, 10, 50, 10, 0, 5, 7, 5, 30, 60, 60, 20, 0};
        QCOMPARE(findPeaks(v, 20.0f), (std::vector<int>{2, 9}));
        QCOMPARE(findPeaks(v, 1.0f), (std::vector<int>{2, 6, 9}));
    }

    void findPeaksStopsAtMissingSamples()
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> v{nan, 40, nan, 0, 30, 0};
        // A sample next to a gap is never a peak, since nothing shows it rises
        // above that side; the 30 rises 30 above its measured neighbours.
        QCOMPARE(findPeaks(v, 10.0f), (std::vector<int>{4}));
    }

    void gridOnAnAxisAlignedPlaneHasNoSliceAxisLines()
    {
        const auto lines = gridLines(axisAlignedMap(), QRectF(0, 0, 100, 100), 25.0);
        int perAxis[3] = {0, 0, 0};
        for (const auto& gl : lines) {
            ++perAxis[gl.axis];
            // Every line follows its coordinate exactly at both ends.
            QCOMPARE(axisAlignedMap().at(gl.line.p1())[std::size_t(gl.axis)], gl.value);
            QCOMPARE(axisAlignedMap().at(gl.line.p2())[std::size_t(gl.axis)], gl.value);
        }
        QCOMPARE(perAxis[0], 4);  // x = 0, 25, 50, 75
        QCOMPARE(perAxis[1], 4);
        QCOMPARE(perAxis[2], 0);  // z is constant on this plane
    }

    void gridOnARotatedPlaneEndsOnTheViewportEdges()
    {
        AffineMap map;
        map.origin = {100.0, 0.0, 0.0};
        map.dx = {0.6, 0.0, 0.8};
        map.dy = {0.0, 1.0, 0.0};
        const QRectF rect(0, 0, 200, 120);
        const auto lines = gridLines(map, rect, 20.0);
        QVERIFY(!lines.empty());
        for (const auto& gl : lines) {
            for (const QPointF& end : {gl.line.p1(), gl.line.p2()}) {
                const bool onEdge = std::abs(end.x() - rect.left()) < 1e-6 || std::abs(end.x() - rect.right()) < 1e-6 ||
                                    std::abs(end.y() - rect.top()) < 1e-6 || std::abs(end.y() - rect.bottom()) < 1e-6;
                QVERIFY(onEdge);
                QVERIFY(std::abs(map.at(end)[std::size_t(gl.axis)] - gl.value) < 1e-6);
            }
        }
    }

    void gridSkipsAxesThatWouldNeedTooManyLines()
    {
        QVERIFY(gridLines(axisAlignedMap(), QRectF(0, 0, 1000, 10), 1.0, 60).size() <= 11);
        QVERIFY(gridLines(axisAlignedMap(), QRectF(0, 0, 100, 100), 0.0).empty());
    }
};

QTEST_APPLESS_MAIN(TestViewerHud)

#include "test_viewer_hud.moc"
