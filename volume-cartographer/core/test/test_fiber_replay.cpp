#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <opencv2/imgcodecs.hpp>

namespace
{

std::filesystem::path temporaryDirectory()
{
    std::mt19937_64 generator(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() / ("vc_fiber_replay_" + std::to_string(generator()));
    std::filesystem::create_directories(path);
    return path;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class RecordingNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d& point) const override
    {
        points.push_back(point);
        return {{0.0, 0.0, 1.0}, true, {}};
    }

    mutable std::vector<cv::Vec3d> points;
};

vc::fiber_tracer::FiberReplayThresholdMeasurement tangentMeasurement(
    double errorBaseVoxels,
    double normalThresholdBaseVoxels = 20.0)
{
    return {
        errorBaseVoxels,
        0.0,
        errorBaseVoxels,
        errorBaseVoxels /
            vc::fiber_tracer::kFiberReplayTangentialThresholdFactor,
        errorBaseVoxels /
            vc::fiber_tracer::kFiberReplayTangentialThresholdFactor /
            normalThresholdBaseVoxels,
        true,
    };
}

void attachTestCtValues(vc::fiber_tracer::FiberReplayStripMeshes& strips)
{
    strips.textureSource = {
        "/data/ct.zarr/2",
        {16, 16, 16},
        {0.25, 0.25, 0.25},
        {0.0, 0.0, 0.0},
    };
    uint8_t value = 0;
    for (auto* components : {
             &strips.reference, &strips.greedy, &strips.fiberlet}) {
        for (auto& component : *components) {
            REQUIRE(component.lineSurface);
            const auto* points = component.lineSurface->rawPointsPtr();
            REQUIRE(points);
            const auto maximumArc = [&](bool rows) {
                double maximum = 0.0;
                const int outer = rows ? points->cols : points->rows;
                const int inner = rows ? points->rows : points->cols;
                for (int outerIndex = 0; outerIndex < outer; ++outerIndex) {
                    double arc = 0.0;
                    for (int innerIndex = 1; innerIndex < inner; ++innerIndex) {
                        const auto& previous = rows
                            ? (*points)(innerIndex - 1, outerIndex)
                            : (*points)(outerIndex, innerIndex - 1);
                        const auto& current = rows
                            ? (*points)(innerIndex, outerIndex)
                            : (*points)(outerIndex, innerIndex);
                        arc += cv::norm(current - previous) * 0.25;
                    }
                    maximum = std::max(maximum, arc);
                }
                return std::max(2, static_cast<int>(std::ceil(maximum)) + 1);
            };
            component.texture.create(maximumArc(true), maximumArc(false));
            for (int row = 0; row < component.texture.rows; ++row) {
                for (int column = 0; column < component.texture.cols; ++column) {
                    component.texture(row, column) = value++;
                }
            }
        }
    }
}

void checkReplayStripCtRendering()
{
    const auto directory = temporaryDirectory() / "ct_uint8";
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {8, 8, 8};
    options.chunkShapeZYX = {4, 4, 4};
    options.dtype = vc::render::ChunkDtype::UInt8;
    options.numLevels = 2;
    options.compressor = "none";
    options.overwriteExisting = true;
    auto pyramid = Volume::New(directory, options);
    REQUIRE(pyramid);

    Array3D<uint8_t> values({8, 8, 8}, 0);
    for (size_t z = 0; z < 8; ++z) {
        for (size_t y = 0; y < 8; ++y) {
            for (size_t x = 0; x < 8; ++x) {
                values(z, y, x) = static_cast<uint8_t>(z * 20 + y * 4 + x);
            }
        }
    }
    pyramid->writeZYX(values, {0, 0, 0});
    pyramid.reset();
    auto volume = Volume::New(directory / "1");
    REQUIRE(volume);

    vc::fiber_tracer::FiberReplayStripMeshes strips;
    vc::fiber_tracer::FiberReplayStripComponent component;
    cv::Mat_<cv::Vec3f> points(2, 3);
    points(0, 0) = {0.5F, 0.5F, 1.0F};
    points(0, 1) = {3.0F, 0.5F, 1.0F};
    points(0, 2) = {5.5F, 0.5F, 1.0F};
    points(1, 0) = {0.5F, 5.5F, 1.0F};
    points(1, 1) = {3.0F, 5.5F, 1.0F};
    points(1, 2) = {5.5F, 5.5F, 1.0F};
    component.lineSurface =
        std::make_shared<QuadSurface>(points, cv::Vec2f{1.0F, 1.0F});
    component.centerlineBaseXYZ = {{1.0, 1.0, 1.0}};
    strips.reference.push_back(std::move(component));
    const auto groupLocator =
        (directory / "1").string() + std::filesystem::path::preferred_separator;
    vc::fiber_tracer::renderFiberReplayStripTextures(
        strips, *volume, groupLocator);

    REQUIRE(strips.textureSource.has_value());
    CHECK(strips.textureSource->locator == groupLocator);
    CHECK(strips.textureSource->shapeZYX == std::array<int, 3>{4, 4, 4});
    CHECK(strips.textureSource->scaleFromBaseXYZ ==
          std::array<double, 3>{0.5, 0.5, 0.5});
    REQUIRE(strips.reference[0].texture.rows == 4);
    REQUIRE(strips.reference[0].texture.cols == 4);
    CHECK(cv::countNonZero(strips.reference[0].texture) > 0);

    auto wholePyramid = Volume::New(directory);
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::validateFiberReplayStripCtVolume(
            *wholePyramid, directory.string()),
        doctest::Contains("one concrete Zarr array/group"),
        std::invalid_argument);

    wholePyramid.reset();
    volume.reset();
    std::filesystem::remove_all(directory.parent_path());
}

bool redDominant(const cv::Vec3b& pixel)
{
    return pixel[2] > pixel[1] + 40 && pixel[2] > pixel[0] + 40;
}

bool cyanDominant(const cv::Vec3b& pixel)
{
    return pixel[0] > pixel[2] + 40 && pixel[1] > pixel[2] + 40;
}

bool magentaDominant(const cv::Vec3b& pixel)
{
    return pixel[0] > pixel[1] + 40 && pixel[2] > pixel[1] + 40;
}

}  // namespace

TEST_CASE("fiber replay tube uses exact endpoint caps and sorted explicit cells")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 2.0;
    const auto tube = vc::fiber_tracer::makeFiberReplayTube({{2.0, 4.0, 4.0}, {10.0, 4.0, 4.0}}, 4.0, 2.0, 2.0, grid, 2);

    CHECK(tube.beginArcBase == doctest::Approx(2.0));
    CHECK(tube.endArcBase == doctest::Approx(6.0));
    REQUIRE(tube.referenceIntervalBase.size() == 2);
    CHECK(tube.referenceIntervalBase.front()[0] == doctest::Approx(4.0));
    CHECK(tube.referenceIntervalBase.back()[0] == doctest::Approx(8.0));
    CHECK(tube.containsBasePoint({4.0, 6.0, 4.0}));
    CHECK_FALSE(tube.containsBasePoint({1.9, 6.1, 4.0}));
    CHECK(std::is_sorted(tube.cellsZYX.begin(), tube.cellsZYX.end()));
    REQUIRE_FALSE(tube.cellsZYX.empty());
    CHECK(tube.cellsZYX == vc::fiber_tracer::fiberAnchorCellsNearPolyline(tube.referenceIntervalBase, tube.radiusBaseVoxels, grid, 2));
    CHECK(
        tube.cellsZYX == std::vector<std::array<size_t, 3>>{
                             {0, 0, 0},
                             {0, 0, 1},
                             {0, 0, 2},
                             {0, 1, 0},
                             {0, 1, 1},
                             {0, 1, 2},
                             {1, 0, 0},
                             {1, 0, 1},
                             {1, 0, 2},
                             {1, 1, 0},
                             {1, 1, 1},
                             {1, 1, 2},
                         });
    CHECK(tube.volumeCropBaseXYZWHD == std::array<size_t, 6>{2, 2, 2, 8, 4, 4});
}

TEST_CASE("forward replay matching uses caller supplied variable advance")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}});
    const auto shortStep = vc::fiber_tracer::matchForwardPolylinePoint(reference, {3.0, 1.0, 0.0}, 0.0, 1.0, 0.0);
    CHECK(shortStep.predictedArc == doctest::Approx(1.0));
    CHECK(shortStep.projection.arc == doctest::Approx(1.0));
    CHECK(shortStep.projection.distance == doctest::Approx(std::sqrt(5.0)));

    const auto longStep = vc::fiber_tracer::matchForwardPolylinePoint(reference, {3.0, 1.0, 0.0}, 0.0, 3.0, 0.0);
    CHECK(longStep.predictedArc == doctest::Approx(3.0));
    CHECK(longStep.projection.arc == doctest::Approx(3.0));
    CHECK(longStep.projection.distance == doctest::Approx(1.0));
}

TEST_CASE("forward polyline interval defaults to the complete remaining reference")
{
    const auto reference = vc::fiber_tracer::makePolylineArcGeometry({
        {0.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
        {7.0, 0.0, 0.0},
        {12.0, 0.0, 0.0},
    });

    const auto complete =
        vc::fiber_tracer::selectForwardPolylineArcInterval(reference, 1);
    CHECK(complete.beginArc == doctest::Approx(3.0));
    CHECK(complete.endArc == doctest::Approx(12.0));

    const auto limited = vc::fiber_tracer::selectForwardPolylineArcInterval(
        reference, 1, 4.0);
    CHECK(limited.beginArc == doctest::Approx(3.0));
    CHECK(limited.endArc == doctest::Approx(7.0));

    const auto clamped = vc::fiber_tracer::selectForwardPolylineArcInterval(
        reference, 1, 100.0);
    CHECK(clamped.endArc == doctest::Approx(12.0));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::selectForwardPolylineArcInterval(reference, 3),
        doctest::Contains("no forward extent"), std::invalid_argument);
}

TEST_CASE("replay strip meshes keep reset components and sample normals in working coordinates")
{
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 2.0;
    const auto tube = vc::fiber_tracer::makeFiberReplayTube(
        {{0.0, 0.0, 0.0}, {8.0, 0.0, 0.0}}, 4.0, 4.0, 2.0, grid, 2);

    vc::fiber_tracer::FiberReplayTraceResult greedy;
    vc::fiber_tracer::FiberReplayTraceSegment first;
    first.tracePointsBase = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    first.matches.push_back({1, 2.0, 2.0, {2.0, 0.0, 0.0}, 0.0, 2.0, 0.0, 0.0});
    vc::fiber_tracer::FiberReplayTraceSegment second;
    second.startReferenceArcBase = 2.0;
    second.tracePointsBase = {{2.0, 1.0, 0.0}, {8.0, 1.0, 0.0}};
    second.matches.push_back({1, 8.0, 8.0, {8.0, 0.0, 0.0}, 2.0, 8.0, 1.0, 0.5});
    greedy.segments = {first, second};

    vc::fiber_tracer::FiberletGraphReplayResult fiberlet;
    vc::fiber_tracer::FiberletGraphReplaySegment route;
    route.routePointsBaseXYZ = {{0.0, -1.0, 0.0}, {8.0, -1.0, 0.0}};
    route.matches = {
        {0, 0.0, 0.0, {0.0, 0.0, 0.0}, 0.0, 0.0, 1.0},
        {1, 8.0, 8.0, {8.0, 0.0, 0.0}, 0.0, 8.0, 1.0},
    };
    fiberlet.segments = {route};

    RecordingNormalSampler normals;
    const auto strips = vc::fiber_tracer::makeFiberReplayStripSurfaces(
        tube, greedy, fiberlet, normals, 2.0, 4);

    REQUIRE(strips.reference.size() == 1);
    REQUIRE(strips.greedy.size() == 2);
    REQUIRE(strips.fiberlet.size() == 1);
    CHECK(strips.greedy[0].sourceSegmentIndex == 0);
    CHECK(strips.greedy[1].sourceSegmentIndex == 1);
    REQUIRE(strips.reference[0].lineSurface);
    CHECK(strips.reference[0].lineSurface->rawPointsPtr()->rows == 21);
    CHECK(strips.reference[0].lineSurface->rawPointsPtr()->cols == 2);
    REQUIRE_FALSE(normals.points.empty());
    CHECK(normals.points.back()[0] <= 4.0);
}

TEST_CASE("replay strip CT rendering uses the existing surface renderer for uint8")
{
    checkReplayStripCtRendering();
}

TEST_CASE("replay strip CT rendering rejects unsupported uint16 volumes")
{
    const auto directory = temporaryDirectory() / "ct_uint16";
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {4, 4, 4};
    options.chunkShapeZYX = {4, 4, 4};
    options.dtype = vc::render::ChunkDtype::UInt16;
    options.numLevels = 1;
    options.compressor = "none";
    options.overwriteExisting = true;
    auto volume = Volume::New(directory, options);
    REQUIRE(volume);
    volume.reset();
    volume = Volume::New(directory / "0");
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::validateFiberReplayStripCtVolume(
            *volume, (directory / "0").string()),
        doctest::Contains("must use uint8"), std::invalid_argument);
    volume.reset();
    std::filesystem::remove_all(directory.parent_path());
}

TEST_CASE("full replay overview renders selected top and side strips with disconnected trace overlays")
{
    const auto directory = temporaryDirectory() / "overview_ct";
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {32, 32, 32};
    options.chunkShapeZYX = {16, 16, 16};
    options.dtype = vc::render::ChunkDtype::UInt8;
    options.numLevels = 2;
    options.compressor = "none";
    options.overwriteExisting = true;
    auto pyramid = Volume::New(directory, options);
    REQUIRE(pyramid);
    Array3D<uint8_t> values({32, 32, 32}, uint8_t(96));
    pyramid->writeZYX(values, {0, 0, 0});
    pyramid.reset();
    auto volume = Volume::New(directory / "1");
    REQUIRE(volume);

    const std::vector<cv::Vec3d> reference{
        {6.0, 16.0, 16.0},
        {10.0, 16.0, 16.0},
        {14.0, 16.0, 16.0},
        {18.0, 16.0, 16.0},
        {22.0, 16.0, 16.0},
    };
    vc::fiber_tracer::FiberReplayTraceResult greedy;
    greedy.referenceBeginArcBase = 10.0;
    greedy.referenceEndArcBase = 26.0;
    greedy.completedReferenceArcBase = 26.0;
    vc::fiber_tracer::FiberReplayTraceSegment greedyLeft;
    greedyLeft.startReferenceArcBase = 10.0;
    greedyLeft.endReferenceArcBase = 16.0;
    greedyLeft.tracePointsBase = {
        {6.0, 18.0, 16.0},
        {10.0, 18.0, 16.0},
        {12.0, 18.0, 16.0},
    };
    greedyLeft.matches = {
        {1, 14.0, 14.0, {10.0, 16.0, 16.0}},
        {2, 16.0, 16.0, {12.0, 16.0, 16.0}},
    };
    vc::fiber_tracer::FiberReplayTraceSegment greedyRight;
    greedyRight.startReferenceArcBase = 20.0;
    greedyRight.endReferenceArcBase = 26.0;
    greedyRight.tracePointsBase = {
        {16.0, 14.0, 16.0},
        {18.0, 14.0, 16.0},
        {22.0, 14.0, 16.0},
    };
    greedyRight.matches = {
        {1, 22.0, 22.0, {18.0, 16.0, 16.0}},
        {2, 26.0, 26.0, {22.0, 16.0, 16.0}},
    };
    greedy.segments = {greedyLeft, greedyRight};
    vc::fiber_tracer::FiberReplayFailure greedyFailure;
    greedyFailure.reason = "distance_above_threshold";
    greedyFailure.referenceArcBase = 14.0;
    greedyFailure.referenceArcFraction = 0.25;
    greedyFailure.referencePointBase = {10.0, 16.0, 16.0};
    greedy.failures.push_back(greedyFailure);
    greedyFailure.index = 1;
    greedyFailure.referenceArcBase = 22.0;
    greedyFailure.referenceArcFraction = 0.75;
    greedyFailure.referencePointBase = {18.0, 16.0, 16.0};
    greedy.failures.push_back(greedyFailure);

    vc::fiber_tracer::FiberletGraphReplayResult fiberlet;
    fiberlet.referenceBeginArcBase = 10.0;
    fiberlet.referenceEndArcBase = 26.0;
    fiberlet.completedReferenceArcBase = 26.0;
    vc::fiber_tracer::FiberletGraphReplaySegment route;
    route.routePointsBaseXYZ = {
        {6.0, 16.0, 18.0},
        {18.0, 16.0, 18.0},
    };
    route.matches = {
        {0, 10.0, 10.0, {6.0, 16.0, 16.0}},
        {1, 22.0, 22.0, {18.0, 16.0, 16.0}},
    };
    vc::fiber_tracer::FiberletGraphReplaySegment routeAfterReset;
    routeAfterReset.routePointsBaseXYZ = {
        {14.0, 16.0, 18.0},
        {22.0, 16.0, 18.0},
    };
    routeAfterReset.matches = {
        {0, 18.0, 18.0, {14.0, 16.0, 16.0}},
        {1, 26.0, 26.0, {22.0, 16.0, 16.0}},
    };
    fiberlet.segments = {route, routeAfterReset};
    vc::fiber_tracer::FiberReplayFailure fiberletFailure;
    fiberletFailure.reason = "distance_above_threshold";
    fiberletFailure.referenceArcBase = 18.0;
    fiberletFailure.referenceArcFraction = 0.5;
    fiberletFailure.referencePointBase = {14.0, 16.0, 16.0};
    fiberlet.failures.push_back(fiberletFailure);
    fiberletFailure.index = 1;
    fiberletFailure.segmentIndex = 1;
    fiberletFailure.referenceArcBase = 22.0;
    fiberletFailure.referenceArcFraction = 0.75;
    fiberletFailure.referencePointBase = {18.0, 16.0, 16.0};
    fiberlet.failures.push_back(fiberletFailure);

    RecordingNormalSampler normals;
    const auto overview =
        vc::fiber_tracer::renderFiberReplayOverview(reference, greedy, fiberlet, normals, 2.0, 4, *volume, (directory / "1").string());
    CHECK(overview.renderScale == 8);
    CHECK(overview.markerWidthPixels == 3);
    CHECK(overview.textureSource.scaleFromBaseXYZ ==
          std::array<double, 3>{0.5, 0.5, 0.5});
    CHECK(overview.referenceTopShapeYX == std::array<int, 2>{328, 72});
    CHECK(overview.referenceSideShapeYX == std::array<int, 2>{328, 72});
    CHECK(overview.fiberletTopShapeYX[0] >= 648);
    CHECK(overview.fiberletSideShapeYX[0] >= 648);
    REQUIRE(overview.fiberletComponents.size() == 2);
    CHECK(overview.fiberletComponents[0].sourceSegmentIndex == 0);
    CHECK(overview.fiberletComponents[0].referenceArcBeginBase == 10.0);
    CHECK(overview.fiberletComponents[0].referenceArcEndBase == 22.0);
    CHECK(overview.fiberletComponents[1].sourceSegmentIndex == 1);
    CHECK(overview.fiberletComponents[1].referenceArcBeginBase == 18.0);
    CHECK(overview.fiberletComponents[1].referenceArcEndBase == 26.0);
    CHECK(overview.fiberletComponents[1].topColumns[0] -
              overview.fiberletComponents[0].topColumns[1] ==
          8);
    CHECK(overview.fiberletComponents[1].sideColumns[0] -
              overview.fiberletComponents[0].sideColumns[1] ==
          8);
    REQUIRE(overview.pages.size() == 1);
    REQUIRE(overview.pages[0].panels.size() == 1);
    const auto& page = overview.pages[0];
    const auto& panel = page.panels[0];
    CHECK(normals.points.front()[0] == doctest::Approx(3.0));

    size_t redPixels = 0;
    size_t cyanPixels = 0;
    for (const auto& pixel : page.image) {
        redPixels += redDominant(pixel) ? 1 : 0;
        cyanPixels += cyanDominant(pixel) ? 1 : 0;
    }
    CHECK(redPixels > 0);
    CHECK(cyanPixels > 0);

    std::vector<uint8_t> encoded;
    REQUIRE(cv::imencode(".jpg", page.image, encoded, {cv::IMWRITE_JPEG_QUALITY, 95, cv::IMWRITE_JPEG_PROGRESSIVE, 0, cv::IMWRITE_JPEG_OPTIMIZE, 0}));
    const cv::Mat_<cv::Vec3b> decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
    size_t decodedRedPixels = 0;
    size_t decodedCyanPixels = 0;
    for (const auto& pixel : decoded) {
        decodedRedPixels += redDominant(pixel) ? 1 : 0;
        decodedCyanPixels += cyanDominant(pixel) ? 1 : 0;
    }
    CHECK(decodedRedPixels > 0);
    CHECK(decodedCyanPixels > 0);

    const auto markerColumn = [&](double arc, int columns) {
        return static_cast<int>(std::lround(
            (arc - 10.0) / 16.0 * (columns - 1)));
    };
    const auto countColumn = [&](const std::array<int, 2>& rows, int column,
                                 const auto& predicate) {
        size_t count = 0;
        for (int row = rows[0]; row < rows[1]; ++row)
            count += predicate(decoded(row, column)) ? 1 : 0;
        return count;
    };
    for (const auto& rows : {panel.referenceTopRows,
                             panel.referenceSideRows}) {
        CHECK(countColumn(
                  rows, markerColumn(14.0, overview.referenceTopShapeYX[1]),
                  redDominant) >
              static_cast<size_t>((rows[1] - rows[0]) * 3 / 4));
        CHECK(countColumn(
                  rows, markerColumn(18.0, overview.referenceTopShapeYX[1]),
                  cyanDominant) >
              static_cast<size_t>((rows[1] - rows[0]) * 3 / 4));
        CHECK(countColumn(
                  rows, markerColumn(22.0, overview.referenceTopShapeYX[1]),
                  magentaDominant) >
              static_cast<size_t>((rows[1] - rows[0]) * 3 / 4));
    }

    const int gapColumn = markerColumn(18.0, overview.referenceTopShapeYX[1]);
    size_t redGapPixels = 0;
    for (int row = panel.referenceTopRows[0];
         row < panel.referenceTopRows[1]; ++row) {
        redGapPixels += redDominant(page.image(row, gapColumn)) ? 1 : 0;
    }
    CHECK(redGapPixels == 0);

    size_t fiberletFrameYellow = 0;
    size_t fiberletFrameCyan = 0;
    for (int row = panel.fiberletTopRows[0];
         row < panel.fiberletTopRows[1]; ++row) {
        for (int column = 0; column < page.image.cols; ++column) {
            const auto& pixel = page.image(row, column);
            fiberletFrameYellow +=
                pixel[1] > 160 && pixel[2] > 160 && pixel[0] < 120 ? 1 : 0;
            fiberletFrameCyan += cyanDominant(pixel) ? 1 : 0;
        }
    }
    CHECK(fiberletFrameYellow > 0);
    CHECK(fiberletFrameCyan > 0);
    for (const auto& component : overview.fiberletComponents) {
        size_t componentRed = 0;
        for (int row = panel.fiberletTopRows[0] + component.topRows[0];
             row < panel.fiberletTopRows[0] + component.topRows[1]; ++row) {
            for (int column = component.topColumns[0];
                 column < component.topColumns[1]; ++column) {
                componentRed += redDominant(page.image(row, column)) ? 1 : 0;
            }
        }
        CHECK(componentRed > 0);
    }
    for (int row = panel.fiberletTopRows[0];
         row < panel.fiberletTopRows[1]; ++row) {
        for (int column = overview.fiberletComponents[0].topColumns[1];
             column < overview.fiberletComponents[1].topColumns[0];
             ++column) {
            CHECK(page.image(row, column) == cv::Vec3b{0, 0, 0});
        }
    }

    auto malformed = greedy;
    malformed.segments.front().matches.pop_back();
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::renderFiberReplayOverview(reference, malformed, fiberlet, normals, 2.0, 4, *volume, (directory / "1").string()),
        doctest::Contains("one match per non-seed point"),
        std::invalid_argument);
    volume.reset();
    std::filesystem::remove_all(directory.parent_path());
}

TEST_CASE("full replay overview compositor keeps four-strip blocks intact across pages")
{
    cv::Mat_<cv::Vec3b> referenceTop(3, 32002, cv::Vec3b{1, 2, 3});
    cv::Mat_<cv::Vec3b> referenceSide(2, 16001, cv::Vec3b{4, 5, 6});
    cv::Mat_<cv::Vec3b> fiberletTop(4, 24002, cv::Vec3b{7, 8, 9});
    cv::Mat_<cv::Vec3b> fiberletSide(5, 8001, cv::Vec3b{10, 11, 12});
    referenceTop.col(16000) = cv::Vec3b{10, 20, 30};
    referenceTop.col(16001) = cv::Vec3b{40, 50, 60};
    referenceSide.col(7999) = cv::Vec3b{70, 80, 90};
    referenceSide.col(8000) = cv::Vec3b{100, 110, 120};
    fiberletTop.col(12000) = cv::Vec3b{20, 40, 60};
    fiberletTop.col(12001) = cv::Vec3b{80, 100, 120};
    fiberletSide.col(4000) = cv::Vec3b{30, 60, 90};

    const auto overview =
        vc::fiber_tracer::testing::composeFiberReplayOverviewForTesting(
            referenceTop, referenceSide, fiberletTop, fiberletSide, 156);
    REQUIRE(overview.pages.size() == 2);
    REQUIRE(overview.pages[0].panels.size() == 1);
    REQUIRE(overview.pages[1].panels.size() == 1);
    CHECK(overview.referenceTopShapeYX == std::array<int, 2>{3, 32002});
    CHECK(overview.referenceSideShapeYX == std::array<int, 2>{2, 16001});
    CHECK(overview.fiberletTopShapeYX == std::array<int, 2>{4, 24002});
    CHECK(overview.fiberletSideShapeYX == std::array<int, 2>{5, 8001});
    const auto& first = overview.pages[0].panels[0];
    const auto& second = overview.pages[1].panels[0];
    CHECK(first.referenceTopColumns ==
          std::array<int, 2>{0, 16001});
    CHECK(second.referenceTopColumns ==
          std::array<int, 2>{16001, 32002});
    CHECK(first.referenceSideColumns ==
          std::array<int, 2>{0, 8000});
    CHECK(second.referenceSideColumns ==
          std::array<int, 2>{8000, 16001});
    CHECK(first.fiberletTopColumns == std::array<int, 2>{0, 12001});
    CHECK(second.fiberletTopColumns == std::array<int, 2>{12001, 24002});
    CHECK(first.fiberletSideColumns == std::array<int, 2>{0, 4000});
    CHECK(second.fiberletSideColumns == std::array<int, 2>{4000, 8001});
    CHECK(overview.pages[0].image.cols == 16001);
    CHECK(overview.pages[0].image.rows == 156);
    CHECK(overview.pages[1].image.rows == 156);
    CHECK(overview.pages[0].image(
              first.referenceTopRows[0], 16000) ==
          cv::Vec3b{10, 20, 30});
    CHECK(overview.pages[1].image(
              second.referenceTopRows[0], 0) ==
          cv::Vec3b{40, 50, 60});
    CHECK(overview.pages[0].image(
              first.referenceSideRows[0], 7999) ==
          cv::Vec3b{70, 80, 90});
    CHECK(overview.pages[1].image(
              second.referenceSideRows[0], 0) ==
          cv::Vec3b{100, 110, 120});
    CHECK(overview.pages[0].image(
              first.fiberletTopRows[0], 12000) ==
          cv::Vec3b{20, 40, 60});
    CHECK(overview.pages[1].image(
              second.fiberletTopRows[0], 0) ==
          cv::Vec3b{80, 100, 120});
    CHECK(overview.pages[1].image(
              second.fiberletSideRows[0], 0) ==
          cv::Vec3b{30, 60, 90});
    CHECK_NOTHROW(
        vc::fiber_tracer::testing::composeFiberReplayOverviewForTesting(
            referenceTop, referenceSide, fiberletTop, fiberletSide, 156));
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::testing::composeFiberReplayOverviewForTesting(
            referenceTop, referenceSide, fiberletTop, fiberletSide, 155),
        doctest::Contains("exceeds the JPEG height limit"),
        std::invalid_argument);
}

TEST_CASE("dual replay publication is deterministic and no-vis has only full traces")
{
    const auto directory = temporaryDirectory();
    vc::fiber_tracer::FiberReplayBundleInput input;
    input.request.fiber.linePointsXyzBase = {
        {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {8.0, 0.0, 0.0}};
    input.request.fiber.controlPointsXyzBase = {
        input.request.fiber.linePointsXyzBase.front(),
        input.request.fiber.linePointsXyzBase.back()};
    input.request.fiber.controlPointLineIndices = {0, 2};
    input.request.referenceEndArcBase = 4.0;
    input.requestedLengthBaseVoxels = 4.0;
    input.greedyReplay.referenceEndArcBase = 4.0;
    input.greedyReplay.completedReferenceArcBase = 4.0;
    vc::fiber_tracer::FiberReplayTraceSegment greedyFirst;
    greedyFirst.endReferenceArcBase = 2.0;
    greedyFirst.terminationReason = "distance_above_threshold";
    greedyFirst.tracePointsBase = {{0.0, 0.0, 0.0}, {2.0, 1.0, 0.0}};
    greedyFirst.cumulativeLosses = {0.0, 1.0};
    greedyFirst.matches.push_back({
        1, 2.0, 2.0, {2.0, 0.0, 0.0}, 0.0, 2.0,
        tangentMeasurement(1.0)});
    vc::fiber_tracer::FiberReplayTraceSegment greedySecond;
    greedySecond.startReferenceArcBase = 2.0;
    greedySecond.endReferenceArcBase = 4.0;
    greedySecond.terminationReason = "reference_end";
    greedySecond.tracePointsBase = {{2.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    greedySecond.cumulativeLosses = {0.0, 1.0};
    input.greedyReplay.segments = {greedyFirst, greedySecond};

    input.fiberletReplay.referenceEndArcBase = 4.0;
    input.fiberletReplay.completedReferenceArcBase = 4.0;
    vc::fiber_tracer::FiberletGraphReplaySegment fiberlet;
    fiberlet.endReferenceArcBase = 4.0;
    fiberlet.terminationReason = "reference_end";
    fiberlet.routePointsBaseXYZ = {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    vc::fiber_tracer::FiberletGraphReplayMatch fiberletBeginMatch;
    fiberletBeginMatch.routePointIndex = 0;
    fiberletBeginMatch.matchedReferencePointBaseXYZ = {0.0, 0.0, 0.0};
    vc::fiber_tracer::FiberletGraphReplayMatch fiberletEndMatch;
    fiberletEndMatch.routePointIndex = 1;
    fiberletEndMatch.predictedReferenceArcBase = 4.0;
    fiberletEndMatch.matchedReferenceArcBase = 4.0;
    fiberletEndMatch.matchedReferencePointBaseXYZ = {4.0, 0.0, 0.0};
    fiberlet.matches = {fiberletBeginMatch, fiberletEndMatch};
    input.fiberletReplay.segments = {fiberlet};
    input.fiberletReplayConfig.referenceEndArcBase = 4.0;
    input.referenceGeometryBase = {{0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    input.sources = nlohmann::json::object();
    input.traceBinding = nlohmann::json::object();
    input.predictionBinding = {
        {"mode", "canonical_stored_grid"},
        {"prediction_to_base_scale", 1.0},
        {"prediction_shape_zyx", {8, 8, 8}},
    };
    input.requestedTraceConfig = nlohmann::json::object();
    input.effectiveTraceConfig = nlohmann::json::object();

    const auto bundle = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(bundle.at("version") == 2);
    CHECK(bundle.at("requested_length_base_voxels") == 4.0);
    CHECK(bundle.at("reference_begin_arc_base") == 0.0);
    CHECK(bundle.at("reference_end_arc_base") == 4.0);
    CHECK(bundle.at("reference_length_base_voxels") == 4.0);
    CHECK(bundle.at("reference_points_base_xyz").size() == 2);
    CHECK(bundle.at("threshold").at("shape") ==
          "lasagna_normal_ellipsoid");
    CHECK(bundle.at("threshold").at("normal_radius_base_voxels") ==
          doctest::Approx(20.0));
    CHECK(bundle.at("threshold").at("tangential_factor") ==
          doctest::Approx(4.0));
    CHECK(bundle.at("threshold").at("tangential_radius_base_voxels") ==
          doctest::Approx(80.0));
    CHECK(bundle.at("greedy").at("threshold") ==
          bundle.at("threshold"));
    CHECK(bundle.at("fiberlet_config").at("threshold") ==
          bundle.at("threshold"));
    const auto& initialMatch =
        bundle.at("greedy").at("segments").at(0).at("matches").at(0);
    CHECK(initialMatch.at("euclidean_error_base_voxels") ==
          doctest::Approx(1.0));
    CHECK(initialMatch.at("normal_error_base_voxels") ==
          doctest::Approx(0.0));
    CHECK(initialMatch.at("tangential_error_base_voxels") ==
          doctest::Approx(1.0));
    CHECK(initialMatch.at("threshold_error_base_voxels") ==
          doctest::Approx(0.25));
    CHECK(initialMatch.at("threshold_error_ratio") ==
          doctest::Approx(0.0125));
    CHECK_FALSE(initialMatch.contains("error_base_voxels"));
    CHECK(bundle.at("visualizations").empty());
    REQUIRE(bundle.at("artifacts").size() == 5);
    CHECK(bundle.at("artifacts").contains("replay/reference.obj"));
    CHECK(bundle.at("artifacts").contains("replay/greedy.json"));
    CHECK(bundle.at("artifacts").contains("replay/greedy.obj"));
    CHECK(bundle.at("artifacts").contains("replay/fiberlet.json"));
    CHECK(bundle.at("artifacts").contains("replay/fiberlet.obj"));
    const auto greedyPath = directory / bundle.at("artifacts").at("replay/greedy.obj").at("path").get<std::string>();
    const std::string greedyObj = readText(greedyPath);
    CHECK(greedyObj.find("g segment_0") != std::string::npos);
    CHECK(greedyObj.find("g segment_1") != std::string::npos);
    CHECK(greedyObj.find("l 1 2\ng segment_1") != std::string::npos);
    CHECK(greedyObj.find("l 1 2 3") == std::string::npos);

    const std::string first = readText(directory / "fiber_replay.json");
    const auto repeated = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(repeated == bundle);
    CHECK(readText(directory / "fiber_replay.json") == first);

    auto invalid = input;
    invalid.fiberletReplay.completedReferenceArcBase = 3.0;
    CHECK_THROWS_AS(vc::fiber_tracer::writeFiberReplayBundle(directory, invalid), std::invalid_argument);
    CHECK(readText(directory / "fiber_replay.json") == first);

    input.sources = {
        {"fiber_manifest", "fiber.json"},
        {"fiber_manifest_content_hash", "fnv1a64:1"},
        {"normal_manifest", "normal.json"},
        {"normal_manifest_content_hash", "fnv1a64:2"},
        {"fiber_json", "reference.json"},
        {"fiber_json_content_hash", "fnv1a64:3"},
    };
    vc::fiber_tracer::FiberReplayFailure greedyFailure;
    greedyFailure.reason = "distance_above_threshold";
    greedyFailure.referenceArcBase = 2.0;
    greedyFailure.referenceArcFraction = 0.5;
    greedyFailure.referencePointBase = {2.0, 0.0, 0.0};
    greedyFailure.evaluatorPointBase = {2.0, 1.0, 0.0};
    greedyFailure.segmentPointIndex = 1;
    greedyFailure.thresholdMeasurement = tangentMeasurement(1.0);
    input.greedyReplay.failures.push_back(greedyFailure);
    vc::fiber_tracer::FiberPredictionGridInfo grid;
    grid.shapeZYX = {8, 8, 8};
    grid.predictionToBaseScale = 1.0;
    vc::fiber_tracer::FiberReplayVisualizationInput visualization;
    visualization.tracer = vc::fiber_tracer::FiberReplayTracer::Greedy;
    visualization.tube = vc::fiber_tracer::makeFiberReplayTube(
        input.request.fiber.linePointsXyzBase, 2.0, 2.0, 2.0, grid, 2);
    visualization.anchors.grid = grid;
    visualization.anchorArtifact.sourceLocator = "fiber.json";
    visualization.anchorArtifact.manifestContentHash = "fnv1a64:1";
    visualization.paths.grid = grid;
    visualization.paths.anchorCellSizePredictionVoxels = 2;
    visualization.pathArtifact.fiberManifestLocator = "fiber.json";
    visualization.pathArtifact.fiberManifestContentHash = "fnv1a64:1";
    visualization.pathArtifact.normalManifestLocator = "normal.json";
    visualization.pathArtifact.normalManifestContentHash = "fnv1a64:2";
    visualization.pathArtifact.anchorArtifactLocator = "anchors/anchors.json";
    visualization.pathArtifact.anchorArtifactContentHash = "fnv1a64:3";
    RecordingNormalSampler visualNormals;
    visualization.strips = vc::fiber_tracer::makeFiberReplayStripSurfaces(
        visualization.tube,
        input.greedyReplay,
        input.fiberletReplay,
        visualNormals,
        1.0,
        1);
    auto unsampledInput = input;
    unsampledInput.visualizations.push_back(visualization);
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::writeFiberReplayBundle(directory, unsampledInput),
        doctest::Contains("no rendered CT images"), std::invalid_argument);
    CHECK(readText(directory / "fiber_replay.json") == first);
    attachTestCtValues(*visualization.strips);
    const cv::Mat_<cv::Vec3b> overviewTop(
        2, 2, cv::Vec3b{32, 64, 96});
    const cv::Mat_<cv::Vec3b> overviewSide(
        2, 2, cv::Vec3b{32, 64, 96});
    auto overview =
        vc::fiber_tracer::testing::composeFiberReplayOverviewForTesting(
            overviewTop, overviewSide, overviewTop, overviewSide);
    overview.fiberletComponents.push_back({
        0, 0.0, 4.0, {0, 2}, {0, 2}, {0, 2}, {0, 2}});
    overview.textureSource = *visualization.strips->textureSource;
    input.overview = std::move(overview);
    input.visualizations.push_back(std::move(visualization));
    const auto visualBundle =
        vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    REQUIRE(visualBundle.at("visualizations").size() == 1);
    const auto alias = visualBundle.at("visualizations").at(0).at("manifest").at("path").get<std::string>();
    CHECK(alias == "fiber_replay_visualization.greedy.000000.json");
    REQUIRE(std::filesystem::exists(directory / alias));
    const auto local = nlohmann::json::parse(readText(directory / alias));
    CHECK(local.at("format") == "vc_fiber_replay_visualization");
    CHECK(local.at("artifacts").at("replay/reference.obj").at("path").get<std::string>().starts_with("runs/"));
    CHECK(local.at("trace_strips").at("geometry_builder") ==
          "buildLineViewSurfaces_default");
    CHECK(local.at("trace_strips").at("cross_samples") == 21);
    CHECK(local.at("failure").at("euclidean_error_base_voxels") ==
          doctest::Approx(1.0));
    CHECK(local.at("failure").at("threshold_error_base_voxels") ==
          doctest::Approx(0.25));
    CHECK(local.at("failure").at("local_normal_valid") == true);
    const auto& values = local.at("trace_strips").at("values");
    CHECK(values.at("semantic") == "ct_intensity");
    CHECK(values.at("encoding") == "obj_uv_grayscale_tiff_u8");
    CHECK(values.at("renderer") == "vc_line_probe_fine_to_coarse");
    CHECK(values.at("sampling_grid") == "source_group_voxel_pitch");
    CHECK(values.at("source_locator") == "/data/ct.zarr/2");
    CHECK(values.at("source_dtype") == "uint8");
    CHECK(values.at("source_shape_zyx") == std::array<int, 3>{16, 16, 16});
    CHECK(values.at("source_group_scale_from_base_xyz") ==
          std::array<double, 3>{0.25, 0.25, 0.25});
    REQUIRE(visualBundle.contains("overview"));
    CHECK(visualBundle.at("overview").at("reference_begin_arc_base") == 0.0);
    CHECK(visualBundle.at("overview").at("reference_end_arc_base") == 4.0);
    CHECK(visualBundle.at("overview").at("reference_point_count") == 2);
    CHECK(visualBundle.at("overview").at("reference_top_shape_yx") == std::array<int, 2>{2, 2});
    CHECK(visualBundle.at("overview").at("fiberlet_top_shape_yx") == std::array<int, 2>{2, 2});
    CHECK(visualBundle.at("overview").at("render_scale") == 8);
    CHECK(visualBundle.at("overview").at("layout").at("panel_count") == 1);
    CHECK(visualBundle.at("overview").at("layout").at("page_count") == 1);
    REQUIRE(visualBundle.at("overview").at("pages").size() == 1);
    CHECK(visualBundle.at("overview").at("pages").at(0).at("stable_path") ==
          "fiber_replay.000000.jpg");
    CHECK(visualBundle.at("overview").at("pages").at(0).at("image_shape_yx") ==
          std::array<int, 2>{150, 90});
    CHECK(visualBundle.at("overview").at("failure_markers").at("width_pixels") == 3);
    const auto& overviewComponent = visualBundle.at("overview")
                                        .at("fiberlet_components")
                                        .at("placements")
                                        .at(0);
    CHECK(overviewComponent.at("source_segment_index") == 0);
    CHECK(overviewComponent.at("reference_arc_begin_base") == 0.0);
    CHECK(overviewComponent.at("reference_arc_end_base") == 4.0);
    CHECK(overviewComponent.at("top_columns") == std::array<int, 2>{0, 2});
    CHECK(visualBundle.at("artifacts").contains("replay/full_strip.000000.jpg"));
    const auto immutableOverview = directory / visualBundle.at("artifacts").at("replay/full_strip.000000.jpg").at("path").get<std::string>();
    REQUIRE(std::filesystem::exists(immutableOverview));
    REQUIRE(std::filesystem::exists(directory / "fiber_replay.000000.jpg"));
    CHECK(readText(immutableOverview) == readText(directory / "fiber_replay.000000.jpg"));
    const cv::Mat decodedOverview = cv::imread((directory / "fiber_replay.000000.jpg").string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(decodedOverview.empty());
    CHECK(decodedOverview.type() == CV_8UC3);
    CHECK(decodedOverview.rows == 150);
    CHECK(decodedOverview.cols == 90);
    CHECK(local.at("artifacts").contains("replay/reference_strip.obj"));
    CHECK(local.at("artifacts").contains("replay/reference_strip.mtl"));
    CHECK(local.at("artifacts").contains("replay/reference_strip.tif"));
    CHECK(local.at("artifacts").contains("replay/greedy_strip.obj"));
    CHECK(local.at("artifacts").contains("replay/greedy_strip.mtl"));
    CHECK(local.at("artifacts").contains("replay/greedy_strip.tif"));
    CHECK(local.at("artifacts").contains("replay/fiberlet_strip.obj"));
    CHECK(local.at("artifacts").contains("replay/fiberlet_strip.mtl"));
    CHECK(local.at("artifacts").contains("replay/fiberlet_strip.tif"));
    const auto referenceStripPath = directory / alias;
    const auto referenceStrip =
        nlohmann::json::parse(readText(referenceStripPath))
            .at("artifacts")
            .at("replay/reference_strip.obj")
            .at("path")
            .get<std::string>();
    const auto stripObjText = readText(referenceStripPath.parent_path() /
                                       referenceStrip);
    CHECK(stripObjText.starts_with(
        "# vc_fiber_replay_reference_strip version 4\n"));
    CHECK(stripObjText.find("mtllib reference_strip.mtl\n") != std::string::npos);

    auto inconsistentMeasurement = input;
    inconsistentMeasurement.greedyReplay.failures.front()
        .thresholdMeasurement->thresholdErrorBaseVoxels = 2.0;
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::writeFiberReplayBundle(
            directory, inconsistentMeasurement),
        doctest::Contains("threshold error or ratio is inconsistent"),
        std::invalid_argument);
    CHECK(stripObjText.find("vt ") != std::string::npos);
    CHECK(stripObjText.find("f 1/1 ") != std::string::npos);
    const auto referenceTexture = referenceStripPath.parent_path() /
        nlohmann::json::parse(readText(referenceStripPath))
            .at("artifacts")
            .at("replay/reference_strip.tif")
            .at("path")
            .get<std::string>();
    const cv::Mat texture = cv::imread(referenceTexture.string(), cv::IMREAD_UNCHANGED);
    CHECK_FALSE(texture.empty());
    CHECK(texture.type() == CV_8UC1);

    const std::string firstOverview = readText(directory / "fiber_replay.000000.jpg");
    {
        std::ofstream(directory / "fiber_replay.000001.jpg") << "stale";
        std::ofstream(directory / "fiber_replay.jpg") << "legacy";
    }
    const auto repeatedVisual = vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(repeatedVisual == visualBundle);
    CHECK(readText(directory / "fiber_replay.000000.jpg") == firstOverview);
    CHECK_FALSE(std::filesystem::exists(directory / "fiber_replay.000001.jpg"));
    CHECK_FALSE(std::filesystem::exists(directory / "fiber_replay.jpg"));

    input.visualizations.clear();
    input.greedyReplay.failures.clear();
    input.overview.reset();
    const auto noVisualBundle =
        vc::fiber_tracer::writeFiberReplayBundle(directory, input);
    CHECK(noVisualBundle.at("visualizations").empty());
    CHECK_FALSE(noVisualBundle.contains("overview"));
    CHECK_FALSE(std::filesystem::exists(directory / alias));
    CHECK_FALSE(std::filesystem::exists(directory / "fiber_replay.000000.jpg"));
    std::filesystem::remove_all(directory);
}
