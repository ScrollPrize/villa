#include "fibers3d/Fiber3DRibbon.hpp"
#include "vc/core/types/Volume.hpp"
#include "../../../core/test/support/TestZarrVolume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/fiber_tracer/FiberDisplay.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include <QTemporaryDir>
#include <QTest>

using namespace vc3d::fibers3d;

// A deliberately unavailable fine level: any blocking read is a test failure.
class DelayedCT : public vc::render::IChunkedArray
{
public:
    bool fine = false, coarse = false, fail = false;
    int requested = 0;
    int numLevels() const override { return 2; }
    std::array<int, 3> shape(int l) const override { return l ? std::array<int, 3>{4, 4, 4} : std::array<int, 3>{8, 8, 8}; }
    std::array<int, 3> chunkShape(int l) const override { return shape(l); }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0; }
    LevelTransform levelTransform(int l) const override
    {
        return {{1.0 / double(1 << l), 1.0 / double(1 << l), 1.0 / double(1 << l)}, {0, 0, 0}};
    }
    vc::render::ChunkResult tryGetChunk(int l, int, int, int) override
    {
        ++requested;
        vc::render::ChunkResult r;
        r.shape = shape(l);
        r.dtype = dtype();
        if (!l && fail) {
            r.status = vc::render::ChunkStatus::Error;
            r.error = "test source failure";
        } else if (l ? coarse : fine) {
            r.status = vc::render::ChunkStatus::Data;
            r.bytes = std::make_shared<const std::vector<std::byte>>(l ? 64 : 512, std::byte(l ? 32 : 123));
        }
        return r;
    }
    vc::render::ChunkResult getChunkBlocking(int, int, int, int) override
    {
        throw std::runtime_error("Interactive ribbon blocked on CT I/O");
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override { throw std::runtime_error("Unexpected prefetch"); }
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 1; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

class Fiber3DRibbonTest : public QObject
{
    Q_OBJECT
private slots:
    void incompleteCTNeverBlocksOrDrawsMissingDataAsBlack()
    {
        DelayedCT cache;
        ct::Tile tile;
        tile.width = 4;
        tile.height = 2;
        tile.left = {{0, 0, 0}, {0, 1, 0}};
        tile.right = {{3, 0, 0}, {3, 1, 0}};
        auto state = ct::sampleTileProgressive(tile, cache, 0);
        QCOMPARE(state.valid, size_t(8));
        QCOMPARE(state.covered, size_t(0));
        QVERIFY(cache.requested > 0);
        for (size_t i = 1; i < tile.pixels.size(); i += 2)
            QCOMPARE(tile.pixels[i], uint8_t(0));
        cache.coarse = true;
        state = ct::sampleTileProgressive(tile, cache, 0);
        QCOMPARE(state.covered, size_t(8));
        QCOMPARE(state.requestedLevel, size_t(0));
        for (size_t i = 0; i < tile.pixels.size(); i += 2) {
            QCOMPARE(tile.pixels[i], uint8_t(32));
            QCOMPARE(tile.pixels[i + 1], uint8_t(255));
        }
        cache.fine = true;
        state = ct::sampleTileProgressive(tile, cache, 0);
        QCOMPARE(state.requestedLevel, state.valid);
        for (size_t i = 0; i < tile.pixels.size(); i += 2)
            QCOMPARE(tile.pixels[i], uint8_t(123));
        cache.fail = true;
        state = ct::sampleTileProgressive(tile, cache, 0);
        QVERIFY(state.errors > 0);
        QCOMPARE(state.covered, size_t(8));
        tile.left[0] = {-3, 0, 0};
        state = ct::sampleTileProgressive(tile, cache, 0);
        QCOMPARE(tile.pixels[1], uint8_t(0));
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, ct::sampleTileProgressive(tile, cache, 0, &cancelled));
    }
    void storageBlocksKeepTheSameFramesAsAnUnsplitFiber()
    {
        Run full;
        for (int i = 0; i < 560; ++i)
            full.points.push_back({double(i), 10 * std::sin(i * .01), 5 * std::cos(i * .02)});
        Fiber unsplit;
        unsplit.runs = {full};
        Fiber blocked;
        // Real AFV blocks share their boundary point every 256 segments.
        // A spatial query need not return blocks in point order.
        blocked.runs =
            {Run{{full.points.begin() + 512, full.points.end()}, 512},
             Run{{full.points.begin(), full.points.begin() + 257}, 0},
             Run{{full.points.begin() + 256, full.points.begin() + 513}, 256}};
        const auto expected = ribbon::makeFrames(unsplit, {});
        const auto actual = ribbon::makeFrames(blocked, {});
        QCOMPARE(actual.strips.size(), size_t(1));
        QCOMPARE(actual.rows, expected.rows);
        for (size_t i = 0; i < actual.rows; ++i) {
            const auto& a = actual.strips[0][i];
            const auto& b = expected.strips[0][i];
            QVERIFY(cv::norm(a.center - b.center) < 1e-12);
            QVERIFY(cv::norm(a.tangent - b.tangent) < 1e-12);
            QVERIFY(cv::norm(a.normal - b.normal) < 1e-12);
        }
        // Coincident positions do not justify bridging absent source indices.
        blocked.runs[2].firstPoint += 1;
        QCOMPARE(ribbon::makeFrames(blocked, {}).strips.size(), size_t(3));
        blocked.runs[2].firstPoint = 256;
        blocked.runs[2].points.front()[1] += .1;
        QCOMPARE(ribbon::makeFrames(blocked, {}).strips.size(), size_t(2));
        // An explicit gap at a block boundary remains a gap.
        blocked.runs[2].points.front() = full.points[256];
        blocked.runs[2].styles.assign(256, SpanStyle::Solid);
        blocked.runs[2].styles[0] = SpanStyle::InferredGap;
        QCOMPARE(ribbon::makeFrames(blocked, {}).strips.size(), size_t(2));
    }

    void matchesLineAnnotationSurfaceCoordinates()
    {
        Fiber fiber;
        Run run;
        for (int i = 0; i < 100; ++i) {
            const double t = i * .07;
            run.points.push_back({100 + 10 * std::cos(t), 100 + 10 * std::sin(t), 100 + t * 3});
        }
        fiber.runs.push_back(run);
        for (int i : {0, 29, 73, 99})
            fiber.markers.push_back({run.points[i], MarkerKind::ControlPoint});
        ribbon::ModelForPoints model = [](const std::vector<cv::Vec3d>& points) {
            auto m = vc::lasagna::lineModelForInspection(points);
            for (size_t i = 0; i < m.points.size(); ++i) {
                const auto p = m.points[i].position;
                // Sign flips and a missing segment must go through the very
                // same smoothing/resampling as Line Annotation.
                m.points[i].sampledNormal = {cv::Vec3d(p[0] - 100, p[1] - 100, .1) * (i % 2 ? -1 : 1), i < 35 || i > 43, {}};
            }
            return m;
        };
        std::vector<cv::Vec3d> points;
        for (const auto& p : run.points)
            points.emplace_back(p[0], p[1], p[2]);
        const auto line = model(points);
        vc::lasagna::LineViewConfig config;
        config.buildLineZSlices = false;
        config.controlPointLinePositions = {0, 29, 73, 99};
        config.orientedPointNormals = vc::lasagna::orientedLineNormals(line);
        for (bool manual : {false, true}) {
            if (manual) {
                fiber.displayNormals = {cv::Vec3d(1, 0, 1), std::nullopt, std::nullopt, cv::Vec3d(0, 1, 1)};
                fiber.controlDirections.resize(4);
                const std::vector<cv::Vec3f> fp(points.begin(), points.end());
                config.displayPointNormals =
                    vc::fiber_tracer::
                        fiberDisplayField(fp, config.orientedPointNormals, config.controlPointLinePositions, fiber.displayNormals, fiber.controlDirections)
                            .normals;
            }
            const auto canonical = vc::lasagna::buildLineViewSurfaces(line, config);
            const auto grid = canonical.lineSurface->rawPoints();
            const auto frames = ribbon::makeFrames(fiber, {}, {}, nullptr, model);
            QCOMPARE(frames.strips.size(), size_t(1));
            QCOMPARE(frames.rows, size_t((grid.cols - 1) * 8 + 1));
            const auto live =
                ribbon::framesFromSurface(grid, canonical.lineSideSlice->rawPoints(), canonical.stripPositionMap.stripGridSpacingBaseVoxels, true);
            QCOMPARE(live.rows, frames.rows);
            const auto& rows = frames.strips[0];
            for (size_t i = 0; i < rows.size(); ++i) {
                const int col = std::min(int(i / 8), grid.cols - 2);
                const double t = (double(i) - col * 8) / 8;
                for (int cross : {2, 3, 4}) {
                    const auto expected = cv::Vec3d(grid(cross, col)) * (1 - t) + cv::Vec3d(grid(cross, col + 1)) * t;
                    const auto actual = ribbon::position(rows[i], (cross - 3) * 32., 0);
                    QCOMPARE(expected, actual);
                    QCOMPARE(ribbon::position(live.strips[0][i], (cross - 3) * 32., 0), actual);
                }
            }
        }
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
        vc::test::ZarrVolumeSpec opts;
        opts.shapeZYX = {1100, 16, 16};
        opts.chunkShapeZYX = {32, 16, 16};
        auto volume = vc::test::createZarrVolume(directory.path().toStdString(), opts);
        Array3D<uint8_t> data({1100, 16, 16});
        for (int z = 0; z < 1100; ++z)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x)
                    data(z, y, x) = uint8_t(x + 2 * y + (z % 100));
        vc::test::writeZYX(*volume, data, {0, 0, 0});
        Fiber f;
        f.runs = {Run{{{8, 8, 1}, {8, 8, 1098}}}};
        auto frames = ribbon::makeFrames(f, {}, [](const cv::Vec3d&) { return cv::Vec3d(1, 0, 0); });
        std::shared_ptr<ribbon::Image> image;
        QTRY_VERIFY_WITH_TIMEOUT((image = ribbon::sampleProgressive(frames, *volume, 6, 0, 0))->complete, 5000);
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
        const auto progressive = ribbon::sampleProgressive(frames, *volume, 6, 0, 0);
        QVERIFY(progressive->complete);
        QCOMPARE(progressive->tiles.size(), image->tiles.size());
        for (size_t i = 0; i < image->tiles.size(); ++i) {
            QCOMPARE(progressive->tiles[i].left, image->tiles[i].left);
            QCOMPARE(progressive->tiles[i].right, image->tiles[i].right);
            QCOMPARE(progressive->tiles[i].pixels, image->tiles[i].pixels);
        }
        std::shared_ptr<ribbon::Image> rotated;
        QTRY_VERIFY_WITH_TIMEOUT((rotated = ribbon::sampleProgressive(frames, *volume, 6, 90, 0))->complete, 5000);
        for (int c = 0; c < 7; ++c)
            QCOMPARE(rotated->tiles[0].pixels[c * 2], uint8_t((5 + c) + 16 + 1));
        auto outside = ribbon::sampleProgressive(frames, *volume, 32, 0, 0);
        QCOMPARE(outside->tiles[0].pixels[1], uint8_t(0));
        QCOMPARE(outside->tiles[0].pixels[16 * 2 + 1], uint8_t(255));
        std::atomic_bool cancelled{true};
        QVERIFY_THROWS_EXCEPTION(std::runtime_error, ribbon::sampleProgressive(frames, *volume, 6, 0, 0, &cancelled));
        QVERIFY_THROWS_EXCEPTION(std::invalid_argument, ribbon::sampleProgressive(frames, *volume, 513, 0, 0));
    }
};
QTEST_GUILESS_MAIN(Fiber3DRibbonTest)
#include "test_fiber_3d_ribbon.moc"
