#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/fiber_tracer/FiberReplayVisualization.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/FiberletCropVisualization.hpp"
#include "vc/core/util/SurfaceTexture.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/lasagna/Dataset.hpp"
#include <cstdlib>
#include <iostream>

#include <filesystem>
#include <fstream>
#include <atomic>
#include <iterator>
#include <limits>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <thread>

TEST_CASE("real Paris4 replay renders aligned CT and prediction inspection")
{
    using namespace vc::fiber_tracer;
    const char* ctPath = std::getenv("LASAGNA_TEST_VOLUME");
    const char* normalPath = std::getenv("VC_TEST_GRAPH_NORMALS");
    const char* predictionPath = std::getenv("VC_TEST_GRAPH_PREDICTIONS");
    const char* fiberPath = std::getenv("VC_TEST_WIDTH_FIBER");
    const char* outputPath = std::getenv("VC_TEST_REPLAY_VIS_OUTPUT");
    if (!ctPath || !normalPath || !predictionPath || !fiberPath || !outputPath) {
        std::cout << "Real replay visualization not configured; skipping optional fixture\n";
        return;
    }
    nlohmann::json input;
    std::ifstream(fiberPath) >> input;
    const auto points = vc3dFiberPointArrayFromJson(input, "line_points", input.at("version"), "real fiber");
    const auto full = makePolylineArcGeometry(points);
    REQUIRE(full.length() > 128);
    const auto reference = slicePolylineArc(full, 0, 128);
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = 8;
    const auto normalData = vc::lasagna::LasagnaDataset::open(normalPath, options);
    const auto predictionData = vc::lasagna::LasagnaDataset::open(predictionPath, options);
    const vc::lasagna::LasagnaNormalSampler normals(normalData);
    const FiberPredictionField predictions(predictionData);
    FiberTraceConfig config;
    config.traceToBaseScale = 8;
    config.coneGridSize = 5;
    const auto replay = replayFiberBeamCase(predictions, config, 8, normals, normals, 8,
        FiberEvaluationCase{reference}, FiberChunkReplayConfig{});
    REQUIRE_FALSE(replay.segments.empty());
    auto volume = Volume::New(ctPath);
    const auto source = validateFiberReplayStripCtVolume(*volume, ctPath);
    CHECK(source.scaleFromBaseXYZ == std::array<double,3>{0.5,0.5,0.5});
    size_t ctNonzero = 0, presenceNonzero = 0;
    std::vector<cv::Mat_<cv::Vec3f>> ctCoordinates, presenceCoordinates;
    const auto ct = [&](const cv::Mat_<cv::Vec3f>& coordinates, int scale) -> cv::Mat_<uint8_t> {
        ctCoordinates.push_back(coordinates.clone());
        cv::Mat_<uint8_t> image = vc::core::util::renderCoordsTextureFineToCoarse(
            coordinates, *volume, 0, scale, "Real replay CT");
        ctNonzero += cv::countNonZero(image);
        return image;
    };
    const auto presence = [&](const cv::Mat_<cv::Vec3f>& coordinates, int scale) {
        presenceCoordinates.push_back(coordinates.clone());
        auto image = renderFiberReplayPresenceTexture(coordinates, scale, source, 8, predictions);
        presenceNonzero += cv::countNonZero(image);
        return image;
    };
    const auto inspection = renderFiberReplayInspection(reference, replay, normals, 8,
        source, ct, presence, 20, false, 2.4);
    REQUIRE_FALSE(inspection.pages.empty());
    CHECK(ctNonzero > 0);
    CHECK(presenceNonzero > 0);
    REQUIRE(ctCoordinates.size() == presenceCoordinates.size());
    for (size_t i = 0; i < ctCoordinates.size(); ++i) {
        REQUIRE(ctCoordinates[i].size() == presenceCoordinates[i].size());
        CHECK(cv::norm(ctCoordinates[i], presenceCoordinates[i], cv::NORM_INF) == 0);
    }
    const auto repeated = renderFiberReplayInspection(reference, replay, normals, 8,
        source, ct, presence, 20, false, 2.4);
    CHECK(repeated.metadata == inspection.metadata);
    REQUIRE(repeated.pages.size() == inspection.pages.size());
    const std::filesystem::path output(outputPath);
    std::filesystem::create_directories(output);
    for (size_t i = 0; i < inspection.pages.size(); ++i) {
        CHECK(cv::norm(repeated.pages[i], inspection.pages[i], cv::NORM_INF) == 0);
        vc::core::util::writeUncompressedTextureTiff(output / ("inspection_" + std::to_string(i) + ".tif"), inspection.pages[i]);
        REQUIRE(cv::imwrite((output / ("inspection_" + std::to_string(i) + ".png")).string(), inspection.pages[i]));
    }
    vc::core::util::atomicWriteString(output / "inspection.json", inspection.metadata.dump(2));
    const auto center = reference.front();
    writeFiberletCropBoxVisualization(*volume, ctPath, center - cv::Vec3d::all(32),
        center + cv::Vec3d::all(32), output / "crop", 128);
    for (const char* face : {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}) {
        const auto texture = output / (std::string("crop_") + face + ".tif");
        const auto image = cv::imread(texture.string(), cv::IMREAD_UNCHANGED);
        CHECK(image.size() == cv::Size(33, 33));
        CHECK(image.type() == CV_8UC1);
        CHECK(cv::countNonZero(image) > 0);
        CHECK(std::filesystem::exists(output / (std::string("crop_") + face + ".obj")));
    }
    std::cout << "Real Paris4 inspection: " << inspection.pages.size() << " pages, "
              << inspection.metadata.at("cross_sections").size() << " sections, "
              << replay.failures.size() << " restarts; exact CT/presence coordinate alignment"
              << " and repeated pixels/metadata; six CT crop faces\n";
}


namespace
{

std::filesystem::path temporaryDirectory()
{
    std::mt19937_64 generator(std::random_device{}());
    const auto path = std::filesystem::temp_directory_path() / ("vc_fiber_replay_" + std::to_string(generator()));
    std::filesystem::create_directories(path);
    return path;
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

TEST_CASE("crop CT source validates concrete uint8 arrays and preserves OME offsets")
{
    using namespace vc::fiber_tracer;
    const auto directory = temporaryDirectory();
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {8,8,8};
    options.chunkShapeZYX = {4,4,4};
    options.dtype = vc::render::ChunkDtype::UInt8;
    options.numLevels = 2;
    options.compressor = "none";
    auto pyramid = Volume::New(directory / "ct", options);
    REQUIRE(pyramid);
    CHECK_THROWS(validateFiberReplayStripCtVolume(*pyramid, (directory / "ct").string()));
    pyramid.reset();
    const auto attrsPath = directory / "ct/.zattrs";
    nlohmann::json attrs;
    std::ifstream(attrsPath) >> attrs;
    auto& datasets = attrs["multiscales"][0]["datasets"];
    datasets[0]["coordinateTransformations"] = {
        {{"type","scale"},{"scale",{1,2,3}}},
        {{"type","translation"},{"translation",{4,6,8}}}};
    datasets[1]["coordinateTransformations"] = {
        {{"type","scale"},{"scale",{2,4,6}}},
        {{"type","translation"},{"translation",{2,2,2}}}};
    vc::core::util::atomicWriteString(attrsPath, attrs.dump());
    auto volume = Volume::New(directory / "ct/1");
    REQUIRE(volume);
    const auto source = validateFiberReplayStripCtVolume(*volume, (directory / "ct/1").string());
    CHECK(source.scaleFromBaseXYZ == std::array<double,3>{0.5,0.5,0.5});
    CHECK(source.offsetFromBaseXYZ == std::array<double,3>{1,1,1});
    CHECK_THROWS(writeFiberletCropBoxVisualization(*volume, source.locator,
        {0,0,0}, {1,1,1}, directory / "box", 1));
    CHECK_THROWS(writeFiberletCropBoxVisualization(*volume, source.locator,
        {0,0,0}, {0,1,1}, directory / "box"));
    CHECK_THROWS(writeFiberletCropBoxVisualization(*volume, source.locator,
        {0,0,0}, {1,1,std::numeric_limits<double>::infinity()}, directory / "box"));
    attrs["multiscales"][0]["axes"][0]["name"] = "x";
    vc::core::util::atomicWriteString(attrsPath, attrs.dump());
    CHECK_THROWS(validateFiberReplayStripCtVolume(*volume, source.locator));
    volume.reset();
    options.dtype = vc::render::ChunkDtype::UInt16;
    options.numLevels = 1;
    auto wider = Volume::New(directory / "uint16", options);
    wider.reset();
    wider = Volume::New(directory / "uint16/0");
    CHECK_THROWS(validateFiberReplayStripCtVolume(*wider, (directory / "uint16/0").string()));
    wider.reset();
    std::filesystem::remove_all(directory);
}

TEST_CASE("presence strip reverses the declared source transform before sampling")
{
    class Predictions final : public vc::fiber_tracer::FiberPredictionSource {
    public:
        mutable std::vector<cv::Vec3d> points;
        vc::fiber_tracer::FiberPredictionSample sample(const cv::Vec3d& p, const cv::Vec3d&) const override {
            points.push_back(p);
            vc::fiber_tracer::FiberPredictionSample result;
            result.options.push_back({{1, 0, 0}, 0.25f, true});
            result.options.push_back({{0, 0, 0}, 0.75f, false});
            return result;
        }
    } predictions;
    vc::fiber_tracer::FiberReplayStripTextureSource source{
        "synthetic-presence", {32, 32, 32}, {0.25, 0.5, 0.125}, {1, -2, 3}};
    cv::Mat_<cv::Vec3f> coords(2, 2, cv::Vec3f{5, 6, 7});
    const auto image = vc::fiber_tracer::renderFiberReplayPresenceTexture(coords, 2, source, 2, predictions);
    CHECK(image.rows == 4);
    CHECK(image.cols == 4);
    REQUIRE(predictions.points.size() == 16);
    for (const auto& p : predictions.points) CHECK(p == cv::Vec3d{8, 8, 16});
    for (const auto value : image) CHECK(value == 191);
}

TEST_CASE("combined inspection aligns modalities and uses the true transverse acceptance ellipse")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
    public:
        cv::Vec3d normal{0, 0, 1}; bool valid = true;
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {normal, valid, {}}; }
    } normals;
    const std::vector<cv::Vec3d> reference{{4, 8, 8}, {12, 8, 8}};
    FiberReplayTraceResult replay;
    replay.referenceEndArcBase = replay.completedReferenceArcBase = 8;
    FiberReplayTraceSegment segment;
    segment.endReferenceArcBase = 8;
    segment.tracePointsBase = reference;
    segment.matches = {{1, 8, 8, reference.back()}};
    replay.segments = {segment};
    FiberReplayStripTextureSource source{"synthetic", {32, 32, 32}, {1, 1, 1}, {0, 0, 0}};
    const auto renderer = [](const cv::Mat_<cv::Vec3f>& coords, int scale) {
        return cv::Mat_<uint8_t>(coords.rows * scale, coords.cols * scale, uint8_t(70));
    };
    for (int variant = 0; variant < 4; ++variant) {
        normals.normal = variant == 1 ? cv::Vec3d{1, 0, 1} : variant == 2 ? cv::Vec3d{1, 0, 0} : cv::Vec3d{0, 0, 1};
        normals.valid = variant != 3;
        const auto frame = fiberReplaySectionFrame(reference.front(), {1, 0, 0}, normals, 1, 2);
        CHECK(frame.verticalRadius >= frame.horizontalRadius - 1e-9);
        CHECK(frame.verticalRadius == doctest::Approx(variant == 3 ? 2 : 8));
        if (variant == 2) CHECK(frame.horizontalRadius == doctest::Approx(8));
        for (int axis = 0; axis < 2; ++axis) {
            const auto offset = axis ? frame.vertical * frame.verticalRadius : frame.horizontal * frame.horizontalRadius;
            CHECK(measureFiberReplayThreshold(reference.front() + offset, reference.front(), normals, 1, 2).thresholdErrorRatio == doctest::Approx(1));
        }
        // Legacy longitudinal strip frames reject invalid/parallel anchor normals;
        // transverse geometry still has the metric's well-defined circular fallback.
        if (variant >= 2) continue;
        auto result = renderFiberReplayInspection(reference, replay, normals, 1, source,
                                                  renderer, renderer, 2, true, 2);
        REQUIRE_FALSE(result.pages.empty());
        CHECK(result.metadata.at("render_scale") == 1);
        CHECK(result.metadata.at("base_samples_per_pixel") == 2);
        const auto& section = result.metadata.at("cross_sections").front();
        const auto r = section.at("radii_base").get<std::array<double, 2>>();
        CHECK(r[1] >= r[0] - 1e-9);
        CHECK(r[1] == doctest::Approx(variant == 3 ? 2 : 8));
        if (variant == 0) CHECK(r[0] == doctest::Approx(2));
        if (variant == 2) CHECK(r[0] == doctest::Approx(8));
        for (int axis = 0; axis < 2; ++axis) {
            const auto a = section.at(axis ? "vertical_axis_xyz" : "horizontal_axis_xyz").get<std::array<double, 3>>();
            const auto boundary = reference.front() + cv::Vec3d{a[0], a[1], a[2]} * r[axis];
            CHECK(measureFiberReplayThreshold(boundary, reference.front(), normals, 1, 2).thresholdErrorRatio == doctest::Approx(1));
        }
        CHECK(section.at("original_fiber_arc_base").get<double>() == doctest::Approx(8));
        const auto& rows = result.metadata.at("rows");
        REQUIRE(rows.size() >= 2);
        CHECK(rows[0].at("modality") == "ct");
        CHECK(rows[1].at("modality") == "presence");
        CHECK(rows[0].at("label").get<std::string>().find("trilinear") != std::string::npos);
        for (const auto& s : result.metadata.at("cross_sections")) {
            const auto columns = s.at("columns").get<std::array<int, 2>>();
            CHECK((columns[0] + columns[1] - 1) / 2.0 == doctest::Approx(s.at("strip_center_column").get<double>()));
            CHECK(s.at("role") == "dense");
            CHECK(s.at("center_base_xyz")[1] == 8);
        }
    }
}

TEST_CASE("inspection uses replay ellipse factor for transverse frames and metadata")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
    public:
        bool valid = true;
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0,0,1}, valid, {}}; }
    } normals;
    for (const auto tangent : {cv::Vec3d{1,0,0}, cv::Vec3d{1,0,1}}) {
        const auto frame = fiberReplaySectionFrame({0,0,0}, tangent, normals, 1, 40, 2);
        CHECK(frame.verticalRadius == doctest::Approx(80));
        CHECK(frame.horizontalRadius <= frame.verticalRadius);
        for (const auto offset : {frame.horizontal * frame.horizontalRadius, frame.vertical * frame.verticalRadius})
            CHECK(measureFiberReplayThreshold(offset, {0,0,0}, normals, 1, 40, 2).thresholdErrorRatio == doctest::Approx(1));
    }
    normals.valid = false;
    CHECK(fiberReplaySectionFrame({0,0,0}, {1,0,0}, normals, 1, 40, 2).verticalRadius == doctest::Approx(40));
    normals.valid = true;
    CHECK_THROWS(fiberReplaySectionFrame({0,0,0}, {1,0,0}, normals, 1, 40, 0));
    const std::vector<cv::Vec3d> points{{4,8,8},{204,8,8}};
    FiberReplayTraceResult replay;
    replay.tangentialThresholdFactor = 2;
    replay.referenceEndArcBase = replay.completedReferenceArcBase = 200;
    FiberReplayTraceSegment segment;
    segment.endReferenceArcBase = 200;
    segment.tracePointsBase = points;
    segment.matches = {{1,200,200,points.back()}};
    replay.segments = {segment};
    FiberReplayStripTextureSource source{"synthetic", {512,512,512}, {1,1,1}, {0,0,0}};
    const auto renderer = [](const cv::Mat_<cv::Vec3f>& coords, int scale) {
        return cv::Mat_<uint8_t>(coords.rows * scale, coords.cols * scale, uint8_t(70));
    };
    const auto result = renderFiberReplayInspection(points, replay, normals, 1, source, renderer, renderer, 40, false, 2.4);
    REQUIRE_FALSE(result.pages.empty());
    CHECK(result.metadata.at("threshold") == fiberReplayThresholdDescriptorJson(40,2));
    REQUIRE_FALSE(result.metadata.at("cross_sections").empty());
    for (const auto& section : result.metadata.at("cross_sections")) {
        CHECK(section.at("radii_base")[0].get<double>() == doctest::Approx(40));
        CHECK(section.at("radii_base")[1].get<double>() == doctest::Approx(80));
    }
}

TEST_CASE("restart labels appear on both assembled traced strip views")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0, 0, 1}, true, {}}; }
    } normals;
    const std::vector<cv::Vec3d> reference{{4, 8, 8}, {204, 8, 8}};
    FiberReplayTraceResult replay;
    replay.referenceEndArcBase = replay.completedReferenceArcBase = 200;
    for (int i = 0; i < 2; ++i) {
        FiberReplayTraceSegment s;
        s.startReferenceArcBase = i * 100;
        s.endReferenceArcBase = (i + 1) * 100;
        s.tracePointsBase = {{4.0 + i * 100, 8, 8}, {104.0 + i * 100, 8, 8}};
        s.matches = {{1, s.endReferenceArcBase, s.endReferenceArcBase, s.tracePointsBase.back()}};
        replay.segments.push_back(s);
    }
    FiberReplayFailure failure;
    failure.index = 0;
    failure.segmentIndex = 0;
    failure.referenceArcBase = 100;
    failure.referenceArcFraction = 0.5;
    replay.failures.push_back(failure);
    FiberReplayStripTextureSource source{"synthetic", {256, 256, 256}, {1, 1, 1}, {0, 0, 0}};
    const auto renderer = [](const cv::Mat_<cv::Vec3f>& c, int scale) {
        return cv::Mat_<uint8_t>(c.rows * scale, c.cols * scale, uint8_t(70));
    };
    const auto plain = renderFiberReplayTraceOverview(reference, replay, normals, 1, 1, source, renderer, 1, false, 2);
    const auto labeled = renderFiberReplayTraceOverview(reference, replay, normals, 1, 1, source, renderer, 1, true, 2);
    REQUIRE(labeled.pages.size() == plain.pages.size());
    for (size_t p = 0; p < plain.pages.size(); ++p) {
        for (const auto& panel : plain.pages[p].panels) {
            for (const auto band : {panel.fiberletTopRows, panel.fiberletSideRows}) {
                CHECK(cv::norm(plain.pages[p].image.rowRange(band[0], band[1]),
                    labeled.pages[p].image.rowRange(band[0], band[1]), cv::NORM_L1) > 0);
            }
        }
    }
}

TEST_CASE("curved inspection aligns restart sections to both actual traced marker rasters")
{
    using namespace vc::fiber_tracer;
    class Normals final : public vc::lasagna::NormalSampler {
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0,0,1}, true, {}}; }
    } normals;
    const std::vector<cv::Vec3d> reference{{0,0,0}, {300,0,0}};
    FiberReplayTraceResult replay;
    replay.referenceEndArcBase = replay.completedReferenceArcBase = 300;
    for (int i = 0; i < 3; ++i) {
        FiberReplayTraceSegment s;
        s.startReferenceArcBase = 100 * i;
        s.endReferenceArcBase = 100 * (i + 1);
        for (int j = 0; j <= 10; ++j) {
            const double x = 100 * i + j * 10;
            s.tracePointsBase.push_back({x, 20 * std::sin(j * 0.3), 0});
            if (j) s.matches.push_back({size_t(j), x, x, {x,0,0}});
        }
        replay.segments.push_back(s);
        if (i < 2) {
            FiberReplayFailure f;
            f.index = i; f.segmentIndex = i;
            f.referenceArcBase = s.endReferenceArcBase;
            f.referenceArcFraction = f.referenceArcBase / 300;
            replay.failures.push_back(f);
        }
    }
    SUBCASE("progressing reference matches") {}
    SUBCASE("first segment moves without reference progress") {
        replay.segments[0].endReferenceArcBase = 0;
        replay.segments[1].startReferenceArcBase = 0;
        for (auto& match : replay.segments[0].matches) {
            match.matchedReferenceArcBase = 0;
            match.matchedReferencePointBase = {0,0,0};
        }
        replay.failures[0].referenceArcBase = 0;
        replay.failures[0].referenceArcFraction = 0;
    }
    FiberReplayStripTextureSource source{"synthetic", {512,512,512}, {1,1,1}, {0,0,0}};
    const auto render = [](const cv::Mat_<cv::Vec3f>& c, int scale) {
        return cv::Mat_<uint8_t>(c.rows * scale, c.cols * scale, uint8_t(70));
    };
    const auto overview = renderFiberReplayTraceOverview(reference, replay, normals, 1, 1, source, render, 1, true, 2);
    for (const auto& c : overview.fiberletComponents) CHECK(c.topColumns == c.sideColumns);
    const auto result = renderFiberReplayInspection(reference, replay, normals, 1, source, render, render, 2, false, 2);
    int breakoffs = 0;
    for (const auto& section : result.metadata.at("cross_sections")) {
        if (section.at("role") != "breakoff") continue;
        ++breakoffs;
        const auto columns = section.at("columns").get<std::array<int,2>>();
        const int center = (columns[0] + columns[1] - 1) / 2;
        for (const auto& row : result.metadata.at("rows")) {
            const auto view = row.at("view").get<std::string>();
            if (view != "traced top" && view != "traced side") continue;
            const auto& image = result.pages.at(row.at("page").get<size_t>());
            const int y = row.at("image_row_begin").get<int>();
            // Read real cyan failure-band pixels, above centerline and text.
            CHECK(image(y, center) == cv::Vec3b(255,255,0));
        }
    }
    CHECK(breakoffs == 2);
}

TEST_CASE("strip column inverse respects nonuniform line positions and refuses gaps")
{
    using namespace vc::fiber_tracer;
    FiberReplayOverviewFiberletComponent c;
    c.pointsBase = {{0, 0, 0}, {2, 0, 0}, {10, 0, 0}};
    c.referenceArcsBase = {0, 2, 10};
    c.positionMap.originalArclengths = {0, 2, 10};
    c.positionMap.stripGridArclengths = {0, 5, 10};
    c.positionMap.totalArclength = 10;
    c.positionMap.stripGridSpacingBaseVoxels = 5;
    c.positionMap.stripGridColumnCount = 3;
    c.topSourceColumns = 3;
    c.topColumns = {100, 106};
    const auto sample = sampleFiberReplayStripColumn(c, 102.5);
    CHECK(sample.pointBase[0] == doctest::Approx(5));
    CHECK(sample.originalPosition == doctest::Approx(1.375));
    CHECK(sample.referenceArcBase == doctest::Approx(5));
    CHECK(sampleFiberReplayStripColumn(c, 105).pointBase[0] == doctest::Approx(10));
    CHECK_THROWS(sampleFiberReplayStripColumn(c, 106));
    CHECK_THROWS(sampleFiberReplayStripColumn(c, 99));
    c.pointsBase.push_back(c.pointsBase.back());
    c.referenceArcsBase.push_back(10);
    c.positionMap.originalArclengths.push_back(10);
    CHECK(cv::norm(sampleFiberReplayStripColumn(c, 105).tangentBase) > 0);
}

bool magentaDominant(const cv::Vec3b& pixel)
{
    return pixel[0] > pixel[1] + 40 && pixel[2] > pixel[1] + 40;
}

bool redDominant(const cv::Vec3b& pixel)
{
    return pixel[2] > pixel[1] + 40 && pixel[2] > pixel[0] + 40;
}

bool cyanDominant(const cv::Vec3b& pixel)
{
    return pixel[0] > pixel[2] + 40 && pixel[1] > pixel[2] + 40;
}
} // namespace
TEST_CASE("long-span overview maps overlays and endpoint markers to resampled columns")
{
    const auto directory = temporaryDirectory();
    Volume::ZarrCreateOptions options;
    options.shapeZYX = {64, 64, 64};
    options.chunkShapeZYX = {16, 16, 16};
    options.dtype = vc::render::ChunkDtype::UInt8;
    options.numLevels = 1;
    options.compressor = "none";
    auto volume = Volume::New(directory / "ct", options);
    volume.reset();
    const auto ct = (directory / "ct/0").string();
    volume = Volume::New(ct);
    const std::vector<cv::Vec3d> reference{{8, 32, 32}, {40, 32, 32}};
    vc::fiber_tracer::FiberReplayTraceResult greedy;
    greedy.referenceEndArcBase = 32;
    vc::fiber_tracer::FiberReplayTraceSegment segment;
    segment.tracePointsBase = {{8, 36, 32}, {40, 36, 32}};
    segment.matches.push_back({1, 32, 32, reference.back(), 0, 32, 4, 0});
    greedy.segments.push_back(segment);
    vc::fiber_tracer::FiberletGraphReplayResult fiberlet;
    fiberlet.referenceEndArcBase = 32;
    vc::fiber_tracer::FiberletGraphReplaySegment route;
    route.routePointsBaseXYZ = {{8, 32, 36}, {40, 32, 36}};
    route.matches = {{0, 0, 0, reference.front()}, {1, 32, 32, reference.back()}};
    fiberlet.segments.push_back(route);
    vc::fiber_tracer::FiberReplayFailure failure;
    failure.reason = "distance_above_threshold";
    failure.referenceArcBase = 32;
    failure.referenceArcFraction = 1;
    failure.referencePointBase = reference.back();
    fiberlet.failures.push_back(failure);
    RecordingNormalSampler normals;
    const auto overview = vc::fiber_tracer::renderFiberReplayOverview(reference, greedy, fiberlet, normals, 1, 1, *volume, ct);
    REQUIRE(overview.pages.size() == 1);
    const auto& page = overview.pages.front();
    REQUIRE(page.panels.size() == 1);
    const auto& panel = page.panels.front();
    for (const auto& rows : {panel.referenceTopRows, panel.referenceSideRows,
                             panel.fiberletTopRows, panel.fiberletSideRows}) {
        // All strips have the same 32vx extent: 2 source vertices become
        // 5 ribbon columns. Endpoint markers belong at100%, not at25%.
        const int right = overview.referenceTopShapeYX[1] - 1;
        CHECK(page.image(rows[0] + 4, right) == cv::Vec3b(255, 255, 0));
        CHECK_FALSE(cyanDominant(page.image(rows[0] + 4, right / 4)));
    }
    const int column = (overview.referenceTopShapeYX[1] - 1) * 3 / 4;
    size_t referenceRed = 0, traceRed = 0, traceYellow = 0;
    for (int y = panel.referenceTopRows[0]; y < panel.referenceTopRows[1]; ++y)
        referenceRed += redDominant(page.image(y, column));
    for (int y = panel.fiberletTopRows[0]; y < panel.fiberletTopRows[1]; ++y)
        traceRed += redDominant(page.image(y, column));
    for (int y = panel.fiberletSideRows[0]; y < panel.fiberletSideRows[1]; ++y) {
        const auto pixel = page.image(y, column);
        traceYellow += pixel[1] > 180 && pixel[2] > 180 && pixel[0] < 80;
    }
    CHECK(referenceRed > 0); // native trace projected into reference strip
    CHECK(traceRed > 0); // other trace projected into resampled trace frame
    CHECK(traceYellow > 0); // reference projected into resampled trace frame
    volume.reset();
    std::filesystem::remove_all(directory);
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
    const int expectedHeight = static_cast<int>((vc::lasagna::kLineViewCrossSampleCount - 1) *
        vc::lasagna::kLineViewCrossRowSpacingBaseVoxels * 0.5 + 1) * overview.renderScale;
    CHECK(overview.referenceTopShapeYX == std::array<int, 2>{expectedHeight, 72});
    CHECK(overview.referenceSideShapeYX == std::array<int, 2>{expectedHeight, 72});
    CHECK(overview.fiberletTopShapeYX[0] >= expectedHeight);
    CHECK(overview.fiberletSideShapeYX[0] >= expectedHeight);
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
