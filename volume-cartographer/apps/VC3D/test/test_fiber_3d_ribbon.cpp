#include "fibers3d/Fiber3DRibbon.hpp"
#include "vc/core/types/Volume.hpp"
#include <QTemporaryDir>
#include <QTest>

using namespace vc3d::fibers3d;

class Fiber3DRibbonTest : public QObject
{
    Q_OBJECT
private slots:
    void transportedFramesAndLasagnaSignContinuity()
    {
        Fiber fiber;
        Run run;
        for (int i = 0; i < 100; ++i) {
            const double t = i * .07;
            run.points.push_back({10 * std::cos(t), 10 * std::sin(t), t * 3});
        }
        fiber.runs.push_back(run);
        const auto frames = ribbon::makeFrames(fiber, {});
        QCOMPARE(frames.strips.size(), size_t(1));
        QCOMPARE(frames.sampledNormals, size_t(0));
        const auto& rows = frames.strips[0];
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            QVERIFY(std::abs(r.normal.dot(r.tangent)) < 1e-10);
            QVERIFY(std::abs(cv::norm(r.normal) - 1) < 1e-10);
            QVERIFY(std::abs(ribbon::across(r, 0).dot(ribbon::across(r, 90))) < 1e-10);
            QVERIFY(cv::norm(ribbon::across(r, 180) + ribbon::across(r, 0)) < 1e-10);
            if (i)
                QVERIFY(r.normal.dot(rows[i - 1].normal) > .98);
        }
        Fiber straight;
        straight.runs = {Run{{{4, 4, 0}, {4, 4, 10}}}};
        auto n = ribbon::makeFrames(straight, {}, [](const cv::Vec3d& p) -> std::optional<cv::Vec3d> {
            if (p[2] < 3 || p[2] == 7)
                return {};
            return cv::Vec3d(int(p[2]) % 2 ? 1 : -1, 0, 0);
        });
        QCOMPARE(n.sampledNormals, size_t(7));
        for (const auto& r : n.strips[0])
            QVERIFY(cv::norm(r.normal - cv::Vec3d(1, 0, 0)) < 1e-10);
    }

    void noBridgesAcrossGapsClippingOrInvalidPoints()
    {
        Fiber f;
        f.runs =
            {Run{{{0, 0, 0}, {4, 0, 0}, {6, 0, 0}, {10, 0, 0}}, 0, {SpanStyle::Solid, SpanStyle::Gap, SpanStyle::Solid}},
             Run{{{0, 1, 0}, {2, 1, 0}, {NAN, 0, 0}, {8, 1, 0}, {10, 1, 0}}}};
        const auto frames = ribbon::makeFrames(f, Bounds{{1, -1, -1}, {9, 2, 1}});
        QCOMPARE(frames.strips.size(), size_t(4));
        for (const auto& strip : frames.strips) {
            for (size_t i = 0; i < strip.size(); ++i) {
                QVERIFY(strip[i].center[0] >= 1 && strip[i].center[0] <= 9);
                if (i)
                    QVERIFY(cv::norm(strip[i].center - strip[i - 1].center) <= 1.00000001);
            }
        }
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, ribbon::makeFrames(f, {}, {}, &cancelled));
    }

    void nativeCtSamplingRotationAndTileSeams()
    {
        QTemporaryDir directory;
        Volume::ZarrCreateOptions opts;
        opts.shapeZYX = {1100, 16, 16};
        opts.chunkShapeZYX = {32, 16, 16};
        opts.compressor = "none";
        auto volume = Volume::New(directory.path().toStdString(), opts);
        Array3D<uint8_t> data({1100, 16, 16});
        for (int z = 0; z < 1100; ++z)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x)
                    data(z, y, x) = uint8_t(x + 2 * y + (z % 100));
        volume->writeZYX(data, {0, 0, 0});
        Fiber f;
        f.runs = {Run{{{8, 8, 1}, {8, 8, 1098}}}};
        auto frames = ribbon::makeFrames(f, {}, [](const cv::Vec3d&) { return cv::Vec3d(1, 0, 0); });
        auto image = ribbon::sample(frames, *volume, 6, 0);
        QCOMPARE(image->tiles.size(), size_t(2));
        const auto& first = image->tiles[0];
        const auto& second = image->tiles[1];
        QCOMPARE(first.width, 7);
        QCOMPARE(first.left.front(), (Point{8, 11, 1}));
        QCOMPARE(first.right.front(), (Point{8, 5, 1}));
        QCOMPARE(first.left.back(), second.left.front());
        for (int c = 0; c < 7; ++c) {
            QCOMPARE(first.pixels[c * 2], uint8_t(8 + 2 * (11 - c) + 1));
            QCOMPARE(first.pixels[c * 2 + 1], uint8_t(255));
            QCOMPARE(first.pixels[(size_t(first.height - 1) * 7 + c) * 2], second.pixels[c * 2]);
        }
        auto rotated = ribbon::sample(frames, *volume, 6, 90);
        for (int c = 0; c < 7; ++c)
            QCOMPARE(rotated->tiles[0].pixels[c * 2], uint8_t((5 + c) + 16 + 1));
        auto outside = ribbon::sample(frames, *volume, 32, 0);
        QCOMPARE(outside->tiles[0].pixels[1], uint8_t(0));
        QCOMPARE(outside->tiles[0].pixels[16 * 2 + 1], uint8_t(255));
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, ribbon::sample(frames, *volume, 6, 0, &cancelled));
        QVERIFY_THROWS_EXCEPTION(std::invalid_argument, ribbon::sample(frames, *volume, 513, 0));
    }
};
QTEST_GUILESS_MAIN(Fiber3DRibbonTest)
#include "test_fiber_3d_ribbon.moc"
