// Coverage for ViewerHud.hpp: the histogram, line sampling, peak finding, path
// length, volume grid and overlay placement behind the slice viewers' measurement
// overlays. All of it works on plain sample buffers and an affine map, so no viewer
// is involved.

#include <QtTest/QtTest>

#include "volume_viewers/ViewerHud.hpp"

#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using namespace vc3d::hud;

namespace {

// A frame of 8-bit samples with its coverage mask; every column holds its x
// coordinate as its value, and every pixel starts out sampled.
struct Frame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> coverage;

    Frame(int w, int h) : width(w), height(h), values(std::size_t(w) * h), coverage(std::size_t(w) * h, 1)
    {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                values[std::size_t(y) * w + x] = static_cast<std::uint8_t>(x);
            }
        }
    }
    void uncover(int x, int y) { coverage[std::size_t(y) * width + x] = 0; }
    void set(int x, int y, int v) { values[std::size_t(y) * width + x] = static_cast<std::uint8_t>(v); }
    Raster raster() const
    {
        Raster r;
        r.values = values.data();
        r.coverage = coverage.data();
        r.width = width;
        r.height = height;
        r.valuesStride = std::size_t(width);
        r.coverageStride = std::size_t(width);
        return r;
    }
    QRect all() const { return QRect(0, 0, width, height); }
};

AffineMap axisAlignedMap()
{
    // Frame x is volume x, frame y is volume y, the slice sits at z = 40.
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
        const Frame f(256, 4);
        const Histogram h = computeHistogram(f.raster(), f.all(), 1);
        QCOMPARE(h.count, std::uint64_t(1024));
        for (int v = 0; v < 256; ++v) {
            QCOMPARE(h.bins[std::size_t(v)], std::uint32_t(4));
        }
        QCOMPARE(h.mean, 127.5);
        QCOMPARE(h.low, 2);
        QCOMPARE(h.high, 253);
        QCOMPARE(h.belowWindow, 1.0 / 256.0);
        QCOMPARE(h.aboveWindow, 1.0 / 256.0);
    }

    void histogramReportsWhatTheWindowClips()
    {
        const Frame f(256, 1);
        const Histogram h = computeHistogram(f.raster(), f.all(), 1, 63, 192);
        QCOMPARE(h.belowWindow, 64.0 / 256.0);   // 0..63
        QCOMPARE(h.aboveWindow, 64.0 / 256.0);   // 192..255
    }

    void histogramStrideSkipsPixels()
    {
        const Frame f(256, 4);
        const Histogram h = computeHistogram(f.raster(), f.all(), 2);
        QCOMPARE(h.count, std::uint64_t(256));
        QCOMPARE(h.bins[1], std::uint32_t(0));
        QCOMPARE(h.bins[2], std::uint32_t(2));
    }

    void histogramOfAnEmptyFrameIsEmpty()
    {
        const Histogram h = computeHistogram(Raster{}, QRect(0, 0, 10, 10), 1);
        QCOMPARE(h.count, std::uint64_t(0));
    }

    void histogramLeavesOutUnsampledPixels()
    {
        // Columns 0-127 stand for a loading gap or the background around a volume.
        Frame f(256, 4);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 128; ++x) {
                f.uncover(x, y);
            }
        }
        const Histogram h = computeHistogram(f.raster(), f.all(), 1);
        QCOMPARE(h.count, std::uint64_t(512));
        QCOMPARE(h.bins[0], std::uint32_t(0));
        QCOMPARE(h.belowWindow, 0.0);
        QCOMPARE(h.low, 129);
    }

    void histogramCountsOnlyTheRegion()
    {
        // The region is the part of the frame on screen; nothing outside it counts.
        const Frame f(256, 4);
        const Histogram h = computeHistogram(f.raster(), QRect(10, 0, 5, 4), 1);
        QCOMPARE(h.count, std::uint64_t(20));
        QCOMPARE(h.bins[9], std::uint32_t(0));
        QCOMPARE(h.bins[10], std::uint32_t(4));
        QCOMPARE(h.bins[15], std::uint32_t(0));
        QCOMPARE(computeHistogram(f.raster(), QRect(300, 0, 5, 4), 1).count, std::uint64_t(0));
    }

    void sampleLineInterpolatesAlongTheRamp()
    {
        const Frame f(64, 8);
        const std::vector<float> v = sampleLine(f.raster(), QPointF(10, 3), QPointF(20, 3));
        QCOMPARE(int(v.size()), 11);
        for (int i = 0; i < 11; ++i) {
            QCOMPARE(v[std::size_t(i)], float(10 + i));
        }
        const std::vector<float> half = sampleLine(f.raster(), QPointF(10.5, 3), QPointF(11.5, 3));
        QCOMPARE(half.front(), 10.5f);
        QCOMPARE(half.back(), 11.5f);
    }

    void sampleLineMarksOutsideSamplesAsMissing()
    {
        const Frame f(16, 16);
        const std::vector<float> v = sampleLine(f.raster(), QPointF(-4, 2), QPointF(4, 2));
        QVERIFY(std::isnan(v.front()));
        QCOMPARE(v.back(), 4.0f);
    }

    void sampleLineNeverReadsAnUnsampledPixel()
    {
        // A gray placeholder (here 64) in a loading gap must not enter the profile,
        // not even blended into a neighbouring sample.
        Frame f(32, 4);
        for (int y = 0; y < 4; ++y) {
            f.set(10, y, 64);
            f.uncover(10, y);
        }
        const std::vector<float> v = sampleLine(f.raster(), QPointF(8, 1), QPointF(12, 1));
        QCOMPARE(int(v.size()), 5);
        QCOMPARE(v[0], 8.0f);
        QCOMPARE(v[1], 9.0f);
        QVERIFY(std::isnan(v[2]));
        QCOMPARE(v[3], 11.0f);
        QCOMPARE(v[4], 12.0f);
        // Halfway between a sampled 9 and the gap is missing, not (9 + 64) / 2.
        const std::vector<float> between = sampleLine(f.raster(), QPointF(9.5, 1), QPointF(9.5, 2));
        QVERIFY(std::isnan(between.front()));
        // A pixel with zero weight does not need to be sampled.
        const std::vector<float> exact = sampleLine(f.raster(), QPointF(9, 1), QPointF(9, 2));
        QCOMPARE(exact.front(), 9.0f);
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

    void cumulativeLengthFollowsThePath()
    {
        // A bent path: 3 along x, then 4 along z.
        const std::vector<std::array<double, 3>> path{{0, 0, 0}, {3, 0, 0}, {3, 0, 4}};
        const auto arc = cumulativeLength(path);
        QCOMPARE(arc.size(), std::size_t(3));
        QCOMPARE(arc[0], 0.0);
        QCOMPARE(arc[1], 3.0);
        QCOMPARE(arc[2], 7.0);  // along the path, not the 5 of the straight chord
    }

    void cumulativeLengthIsUnknownAfterAMissingPosition()
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::vector<std::array<double, 3>> path{{0, 0, 0}, {1, 0, 0}, {nan, nan, nan}, {3, 0, 0}};
        const auto arc = cumulativeLength(path);
        QCOMPARE(arc[1], 1.0);
        QVERIFY(std::isnan(arc[2]));
        QVERIFY(std::isnan(arc[3]));
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
    void fractionalGridStepsGetDistinctLabels()
    {
        QCOMPARE(gridLabelDecimals(25.0), 0);
        QCOMPARE(gridLabelDecimals(0.5), 1);
        QCOMPARE(gridLabelDecimals(2.5), 1);
        QCOMPARE(gridLabelDecimals(0.25), 2);
        QCOMPARE(gridLabelDecimals(0.1), 1);
        for (double step : {0.25, 0.5, 2.5, 0.1}) {
            const auto lines = gridLines(axisAlignedMap(), QRectF(0, 0, 4, 4), step);
            std::set<QString> xLabels;
            int xLines = 0;
            for (const auto& gl : lines) {
                if (gl.axis == 0) {
                    xLabels.insert(gridLabel(gl.axis, gl.value, step));
                    ++xLines;
                }
            }
            QVERIFY(xLines > 1);
            QCOMPARE(int(xLabels.size()), xLines);
        }
        QCOMPARE(gridLabel(0, 0.5, 0.5), QStringLiteral("x=0.5"));
        QCOMPARE(gridLabel(2, -1e-12, 0.5), QStringLiteral("z=0.0"));
    }

    void gridValuesAreExactMultiplesOfTheStep()
    {
        const auto lines = gridLines(axisAlignedMap(), QRectF(0, 0, 3, 3), 0.1, 60);
        for (const auto& gl : lines) {
            const double k = gl.value / 0.1;
            QVERIFY(std::abs(k - std::round(k)) < 1e-9);
        }
    }

    void gridHandlesTinyPanes()
    {
        // A 44 x 2 pane, as a squeezed split view can be; nothing may abort.
        const auto lines = gridLines(axisAlignedMap(), QRectF(0, 0, 44, 2), 10.0);
        for (const auto& gl : lines) {
            const QPointF at = gl.line.p1();
            const double x = clampInto(at.x() + 3, 2.0, 44.0 - 60);
            const double y = clampInto(at.y() + 12, 12.0, 2.0 - 4);
            QCOMPARE(x, 2.0);
            QCOMPARE(y, 12.0);
        }
    }

    void clampIntoToleratesInvertedBounds()
    {
        QCOMPARE(clampInto(5.0, 0.0, 10.0), 5.0);
        QCOMPARE(clampInto(-1.0, 0.0, 10.0), 0.0);
        QCOMPARE(clampInto(11.0, 0.0, 10.0), 10.0);
        QCOMPARE(clampInto(5.0, 10.0, 0.0), 10.0);
    }

    void loupeFitsInsideEveryPane()
    {
        const QSizeF panes[] = {{1000, 800}, {284, 124}, {80, 80}, {120, 600}, {600, 90}};
        for (const QSizeF& pane : panes) {
            for (const QPointF cursor : {QPointF(0, 0), QPointF(pane.width() - 1, 0),
                                         QPointF(pane.width() / 2, pane.height() / 2),
                                         QPointF(pane.width() - 1, pane.height() - 1)}) {
                const auto rect = loupeRect(pane, cursor, 168.0);
                if (!rect) {
                    // Only panes too small for a useful loupe go without one.
                    QVERIFY(std::min(pane.width() - 8, pane.height() - 30) < 48);
                    continue;
                }
                QVERIFY(rect->width() >= 48 && rect->width() <= 168);
                QVERIFY(rect->left() >= 4 - 1e-9);
                QVERIFY(rect->top() >= 4 - 1e-9);
                QVERIFY(rect->right() <= pane.width() - 4 + 1e-9);
                QVERIFY(rect->bottom() <= pane.height() - 4 - 22 + 1e-9);
            }
        }
        QVERIFY(loupeRect(QSizeF(40, 40), QPointF(5, 5), 168.0) == std::nullopt);
        QCOMPARE(loupeRect(QSizeF(1000, 800), QPointF(400, 400), 168.0)->width(), 168.0);
    }

    void panelsOnlyDrawWhereTheyFit()
    {
        QVERIFY(panelFits(QSizeF(400, 200), QSizeF(320, 150)));
        QVERIFY(!panelFits(QSizeF(330, 200), QSizeF(320, 150)));
        QVERIFY(!panelFits(QSizeF(400, 160), QSizeF(320, 150)));
    }
};

QTEST_APPLESS_MAIN(TestViewerHud)

#include "test_viewer_hud.moc"
