#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/atlas/Atlas.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace {

class ConstantNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit ConstantNormalSampler(cv::Vec3d normal) : normal_(normal) {}

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

class InvalidNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {{0.0, 0.0, 0.0}, false, {}};
    }
};

class JumpNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& p) const override
    {
        if (p[0] > 2.5) {
            return {cv::Vec3d{7.0, 0.0, -1.0}, true, {}};
        }
        return {cv::Vec3d{0.0, 0.0, 1.0}, true, {}};
    }
};

std::shared_ptr<QuadSurface> makePlane(int rows,
                                       int cols,
                                       double z,
                                       double yBias = 0.0,
                                       double xBias = 0.0)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col + xBias),
                                         static_cast<float>(row + yBias),
                                         static_cast<float>(z));
        }
    }
    return std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
}

fs::path tempRoot(const std::string& name)
{
    const fs::path root = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    return root;
}

} // namespace

TEST_CASE("Atlas JSON round trips metadata links and fiber mapping")
{
    const fs::path root = tempRoot("vc_atlas_roundtrip");

    vc::atlas::Atlas atlas;
    atlas.metadata.name = "fiber_1";
    atlas.metadata.baseMeshPath = "base_mesh/shell.tifxyz";
    atlas.metadata.sourceBaseMeshPath = "segments/shell";
    atlas.metadata.idxRotationColumns = 3;
    atlas.metadata.seedLineIndex = 1;
    atlas.metadata.seedAtlasU = 4.5;
    atlas.metadata.seedAtlasV = 2.0;
    atlas.links.push_back("placeholder");

    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = "fibers/1.json";
    mapping.lineAnchors.push_back({0, {1.0, 2.0, 3.0}, 4.0, 5.0, 0.25});
    mapping.controlAnchors.push_back({0, {1.0, 2.0, 3.0}, 4.0, 5.0, 0.25});
    atlas.fibers.push_back(mapping);

    atlas.save(root);
    const auto loaded = vc::atlas::Atlas::load(root);

    REQUIRE(loaded.metadata.name == "fiber_1");
    CHECK(loaded.metadata.idxRotationColumns == 3);
    CHECK(loaded.metadata.seedLineIndex == 1);
    CHECK(loaded.metadata.seedAtlasU == doctest::Approx(4.5));
    REQUIRE(loaded.links.size() == 1);
    REQUIRE(loaded.fibers.size() == 1);
    CHECK(loaded.fibers[0].fiberPath == fs::path("fibers/1.json"));
    REQUIRE(loaded.fibers[0].lineAnchors.size() == 1);
    CHECK(loaded.fibers[0].lineAnchors[0].atlasV == doctest::Approx(5.0));
}

TEST_CASE("Atlas seed selection uses line points without requiring controls")
{
    auto surface = makePlane(4, 4, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});
    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"shell", "segments/shell", surface},
    };
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{1.0, 1.0, 1.0}, {2.0, 1.0, 1.0}};

    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    const auto selection = vc::atlas::selectBaseSurfaceBySeedRay(fiber, surfaces, index, sampler);
    CHECK(selection.seedLineIndex == 0);
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler);
    CHECK(mapping.lineAnchors.size() == 2);
    CHECK(mapping.controlAnchors.empty());
}

TEST_CASE("Atlas base selection chooses nearest seed-normal ray hit")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 4.8},
        {2.0, 1.0, 4.9},
    };
    fiber.controlPoints = {{1.0, 1.0, 4.8}};

    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"low", "segments/low", makePlane(4, 4, 0.0)},
        {"high", "segments/high", makePlane(4, 4, 5.0)},
    };
    SurfacePatchIndex index;
    index.rebuild({surfaces[0].surface, surfaces[1].surface});
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    const auto selection = vc::atlas::selectBaseSurfaceBySeedRay(fiber, surfaces, index, sampler);
    CHECK(selection.surfaceIndex == 1);
    CHECK(selection.seedLineIndex == 1);
}

TEST_CASE("Atlas base selection ignores shells not intersected by seed ray")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{1.0, 1.0, 0.0}};
    fiber.controlPoints = {{1.0, 1.0, 0.0}};

    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"closer_miss", "segments/closer_miss", makePlane(2, 2, 0.1, 1.2, 1.2)},
        {"far_hit", "segments/far_hit", makePlane(3, 3, 5.0)},
    };
    SurfacePatchIndex index;
    index.rebuild({surfaces[0].surface, surfaces[1].surface});
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    const auto selection = vc::atlas::selectBaseSurfaceBySeedRay(fiber, surfaces, index, sampler);
    CHECK(selection.surfaceIndex == 1);
    CHECK(selection.surfaceName == "far_hit");
    CHECK(selection.distance == doctest::Approx(5.0));
}

TEST_CASE("Atlas base selection expands seed ray beyond the initial probe length")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{1.0, 1.0, 0.0}};
    fiber.controlPoints = {{1.0, 1.0, 0.0}};

    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"distant_hit", "segments/distant_hit", makePlane(3, 3, 250.0)},
    };
    SurfacePatchIndex index;
    index.rebuild({surfaces[0].surface});
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    vc::atlas::LineMappingOptions options;
    options.rayHalfLength = 16.0;
    const auto selection = vc::atlas::selectBaseSurfaceBySeedRay(
        fiber, surfaces, index, sampler, options);
    CHECK(selection.surfaceIndex == 0);
    CHECK(selection.distance == doctest::Approx(250.0));
}

TEST_CASE("Atlas base selection reports invalid seed normals")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{1.0, 1.0, 0.0}};
    fiber.controlPoints = {{1.0, 1.0, 0.0}};
    auto surface = makePlane(3, 3, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});
    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"shell", "segments/shell", surface},
    };
    InvalidNormalSampler sampler;
    CHECK_THROWS_WITH_AS(
        vc::atlas::selectBaseSurfaceBySeedRay(fiber, surfaces, index, sampler),
        doctest::Contains("No valid normal at atlas seed point"),
        std::runtime_error);
}

TEST_CASE("Atlas base selection reports missing seed-ray intersections")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{1.0, 1.0, 0.0}};
    fiber.controlPoints = {{1.0, 1.0, 0.0}};
    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"miss", "segments/miss", makePlane(2, 2, 0.1, 2.0, 2.0)},
    };
    SurfacePatchIndex index;
    index.rebuild({surfaces[0].surface});
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    CHECK_THROWS_WITH_AS(
        vc::atlas::selectBaseSurfaceBySeedRay(fiber, surfaces, index, sampler),
        doctest::Contains("Atlas seed ray did not intersect any shell"),
        std::runtime_error);
}

TEST_CASE("Atlas ray projection returns fractional coordinates on bilinear quads")
{
    cv::Mat_<cv::Vec3f> points(2, 2);
    points(0, 0) = {0.0f, 0.0f, 0.0f};
    points(0, 1) = {1.0f, 0.0f, 0.0f};
    points(1, 0) = {0.0f, 1.0f, 0.0f};
    points(1, 1) = {1.0f, 1.0f, 1.0f};
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
    SurfacePatchIndex index;
    index.rebuild({surface});
    std::vector<vc::atlas::SurfaceCandidate> surfaces = {
        {"curved", "segments/curved", surface},
    };

    const auto hits = vc::atlas::projectPointAlongNormalToSurfaces(
        {0.25, 0.5, 2.0}, {0.0, 0.0, 1.0}, surfaces, index, 4.0);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].surfaceIndex == 0);
    CHECK(hits[0].atlasU == doctest::Approx(0.25));
    CHECK(hits[0].atlasV == doctest::Approx(0.5));
    CHECK(hits[0].world[2] == doctest::Approx(0.125));
}

TEST_CASE("Atlas idx rotation cyclically shifts columns without moving coordinates")
{
    cv::Mat_<cv::Vec3f> points(2, 4);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>((col == 2 ? -10 : col) + row),
                                         0.0f);
        }
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    CHECK(vc::atlas::computeIdxRotationColumns(surface) == 2);

    auto rotated = vc::atlas::idxRotatedSurface(surface, 2);
    const auto* out = rotated->rawPointsPtr();
    REQUIRE(out != nullptr);
    CHECK((*out)(0, 0)[0] == doctest::Approx(points(0, 2)[0]));
    CHECK((*out)(0, 1)[0] == doctest::Approx(points(0, 3)[0]));
    CHECK((*out)(0, 2)[0] == doctest::Approx(points(0, 0)[0]));
    CHECK((*out)(0, 3)[0] == doctest::Approx(points(0, 1)[0]));
}

TEST_CASE("Atlas mapped object covered size uses line anchors only")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping first;
    first.lineAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
    first.lineAnchors.push_back({1, {}, 4.0, 3.0, 0.0});
    first.controlAnchors.push_back({1, {}, 5.0, 4.0, 0.0});
    atlas.fibers.push_back(std::move(first));

    vc::atlas::FiberMapping second;
    second.lineAnchors.push_back({0, {}, -1.0, 2.0, 0.0});
    atlas.fibers.push_back(std::move(second));

    const auto size = vc::atlas::mappedObjectCoveredAtlasSize(atlas);
    REQUIRE(size.valid);
    CHECK(size.width == doctest::Approx(5.0));
    CHECK(size.height == doctest::Approx(2.0));

    const auto scaledSize = vc::atlas::mappedObjectCoveredAtlasSize(
        atlas, cv::Vec2f(2.0f, 4.0f));
    REQUIRE(scaledSize.valid);
    CHECK(scaledSize.width == doctest::Approx(2.5));
    CHECK(scaledSize.height == doctest::Approx(0.5));

    CHECK_THROWS_WITH_AS(
        vc::atlas::mappedObjectCoveredAtlasSize(atlas, cv::Vec2f(0.0f, -1.0f)),
        doctest::Contains("invalid scale"),
        std::runtime_error);
}

TEST_CASE("Atlas grid coordinates convert to QuadSurface surface coordinates with scale and center")
{
    cv::Mat_<cv::Vec3f> points(4, 6);
    points.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
    QuadSurface surface(points, cv::Vec2f(2.0f, 3.0f));

    const cv::Vec2f surfaceCoord =
        vc::atlas::atlasGridToSurfaceCoords(5.0, 7.0, surface, 2.0);
    CHECK(surfaceCoord[0] == doctest::Approx(0.0));
    CHECK(surfaceCoord[1] == doctest::Approx(5.0 / 3.0));

    QuadSurface invalidScaleSurface(points, cv::Vec2f(0.0f, 1.0f));
    const cv::Vec2f invalidCoord =
        vc::atlas::atlasGridToSurfaceCoords(5.0, 7.0, invalidScaleSurface, 2.0);
    CHECK(!std::isfinite(invalidCoord[0]));
    CHECK(!std::isfinite(invalidCoord[1]));
}

TEST_CASE("Atlas wrapped shell period uses unique columns")
{
    cv::Mat_<cv::Vec3f> points(2, 5);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col % 4),
                                         static_cast<float>(row),
                                         0.0f);
        }
    }
    QuadSurface wrapped(points, cv::Vec2f(1.0f, 1.0f));
    CHECK(vc::atlas::atlasHorizontalPeriodColumns(wrapped) == 4);

    points(0, points.cols - 1)[0] = 9.0f;
    QuadSurface open(points, cv::Vec2f(1.0f, 1.0f));
    CHECK(vc::atlas::atlasHorizontalPeriodColumns(open) == 5);
}

TEST_CASE("Atlas display range uses leftmost mapped unwrap as the minimum column offset")
{
    vc::atlas::Atlas atlas;
    atlas.metadata.seedAtlasU = 30.0;
    vc::atlas::FiberMapping mapping;
    mapping.lineAnchors.push_back({0, {}, 9.0, 1.0, 0.0});
    mapping.lineAnchors.push_back({1, {}, 13.0, 1.0, 0.0});
    mapping.controlAnchors.push_back({1, {}, 4.5, 1.0, 0.0});
    atlas.fibers.push_back(std::move(mapping));

    const auto range = vc::atlas::atlasDisplayRange(atlas, 4);
    CHECK(range.leftmostWinding == 2);
    CHECK(range.rightmostWinding == 3);
    CHECK(range.unwrapCount == 2);
    CHECK(range.atlasUOffset == doctest::Approx(8.0));
    CHECK(range.hasMappedObjects);
}

TEST_CASE("Atlas display range uses wrapped shell period for winding and offset")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping mapping;
    mapping.lineAnchors.push_back({0, {}, 3.25, 1.0, 0.0});
    mapping.lineAnchors.push_back({1, {}, 8.25, 1.0, 0.0});
    atlas.fibers.push_back(std::move(mapping));

    const auto range = vc::atlas::atlasDisplayRange(atlas, 4);
    CHECK(range.leftmostWinding == 0);
    CHECK(range.rightmostWinding == 2);
    CHECK(range.unwrapCount == 3);
    CHECK(range.atlasUOffset == doctest::Approx(0.0));
    CHECK(range.hasMappedObjects);
}

TEST_CASE("Atlas repeated display surface duplicates base mesh columns for multi unwrap views")
{
    cv::Mat_<cv::Vec3f> points(2, 3);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         static_cast<float>(col + row));
        }
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));

    auto repeated = vc::atlas::repeatedAtlasDisplaySurface(surface, 3);
    const auto* out = repeated->rawPointsPtr();
    REQUIRE(out != nullptr);
    CHECK(out->rows == 2);
    CHECK(out->cols == 9);
    CHECK((*out)(1, 0)[2] == doctest::Approx((*out)(1, 3)[2]));
    CHECK((*out)(1, 1)[2] == doctest::Approx((*out)(1, 7)[2]));
}

TEST_CASE("Atlas repeated wrapped display surface tiles unique period and closes seam")
{
    cv::Mat_<cv::Vec3f> points(2, 4);
    cv::Mat labels(2, 4, CV_8U);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            const int uniqueCol = col % 3;
            points(row, col) = cv::Vec3f(static_cast<float>(uniqueCol),
                                         static_cast<float>(row),
                                         static_cast<float>(10 + uniqueCol));
            labels.at<uint8_t>(row, col) = static_cast<uint8_t>(uniqueCol + 1);
        }
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    surface.setChannel("labels", labels);

    auto repeated = vc::atlas::repeatedAtlasDisplaySurface(surface, 3);
    const auto* out = repeated->rawPointsPtr();
    REQUIRE(out != nullptr);
    CHECK(out->rows == 2);
    CHECK(out->cols == 10);
    CHECK((*out)(0, 0)[2] == doctest::Approx((*out)(0, 3)[2]));
    CHECK((*out)(0, 0)[2] == doctest::Approx((*out)(0, 9)[2]));
    CHECK((*out)(0, 8)[2] == doctest::Approx(12.0));

    const cv::Mat repeatedLabels = repeated->channel("labels");
    REQUIRE(!repeatedLabels.empty());
    CHECK(repeatedLabels.cols == 10);
    CHECK(repeatedLabels.at<uint8_t>(0, 0) == repeatedLabels.at<uint8_t>(0, 3));
    CHECK(repeatedLabels.at<uint8_t>(0, 0) == repeatedLabels.at<uint8_t>(0, 9));
    CHECK(repeatedLabels.at<uint8_t>(0, 8) == 3);
}

TEST_CASE("Atlas maps a synthetic fiber over a simple grid")
{
    auto surface = makePlane(5, 8, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});

    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/1.json";
    fiber.linePoints = {
        {1.0, 2.0, 1.0},
        {2.0, 2.0, 1.0},
        {3.0, 2.0, 1.0},
    };
    fiber.controlPoints = {{2.0, 2.0, 1.0}};

    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler);
    REQUIRE(mapping.lineAnchors.size() == 3);
    CHECK(mapping.lineAnchors[0].atlasU == doctest::Approx(1.0));
    CHECK(mapping.lineAnchors[1].atlasU == doctest::Approx(2.0));
    CHECK(mapping.lineAnchors[2].atlasV == doctest::Approx(2.0));
    REQUIRE(mapping.controlAnchors.size() == 1);
    CHECK(mapping.controlAnchors[0].atlasU == doctest::Approx(2.0));
}

TEST_CASE("Atlas mapping keeps wrapped seam hits continuous")
{
    cv::Mat_<cv::Vec3f> points(2, 5);
    for (int row = 0; row < points.rows; ++row) {
        const float radius = static_cast<float>(row + 1);
        points(row, 0) = cv::Vec3f(radius, 0.0f, 0.0f);
        points(row, 1) = cv::Vec3f(0.0f, radius, 0.0f);
        points(row, 2) = cv::Vec3f(-radius, 0.0f, 0.0f);
        points(row, 3) = cv::Vec3f(0.0f, -radius, 0.0f);
        points(row, 4) = points(row, 0);
    }
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
    SurfacePatchIndex index;
    index.rebuild({surface});

    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/wrapped.json";
    fiber.linePoints = {
        {0.3, -1.2, 1.0},
        {1.35, 0.15, 1.0},
    };

    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler);
    REQUIRE(mapping.lineAnchors.size() == 2);
    CHECK(mapping.lineAnchors[0].atlasU == doctest::Approx(3.2).epsilon(1.0e-4));
    CHECK(mapping.lineAnchors[1].atlasU > 4.0);
    CHECK(mapping.lineAnchors[1].atlasU < 4.2);
}

TEST_CASE("Atlas mapping stops when grid and line step mismatch")
{
    auto surface = makePlane(5, 16, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});

    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/1.json";
    fiber.linePoints = {
        {1.0, 2.0, 1.0},
        {2.0, 2.0, 1.0},
        {3.0, 2.0, 1.0},
    };
    fiber.controlPoints = {{1.0, 2.0, 1.0}};

    JumpNormalSampler sampler;
    vc::atlas::LineMappingOptions options;
    options.rayHalfLength = 16.0;
    options.mismatchRatio = 1.5;
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler, options);
    REQUIRE(mapping.lineAnchors.size() == 2);
    CHECK(mapping.lineAnchors.back().sourceIndex == 1);
}

TEST_CASE("Atlas manifest init_shell_dir resolves relative to lasagna manifest")
{
    const fs::path root = tempRoot("vc_atlas_manifest_init_shell_dir");
    const fs::path manifestPath = root / "dataset.lasagna.json";
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(
        R"({"version":1,"init_shell_dir":"init_shells"})",
        manifestPath);
    REQUIRE(manifest.initShellDir.has_value());
    CHECK(*manifest.initShellDir == fs::absolute(root / "init_shells").lexically_normal());
}

TEST_CASE("Atlas init shell loading accepts only shell tifxyz directories")
{
    const fs::path root = tempRoot("vc_atlas_init_shell_candidates");
    const fs::path initDir = root / "init_shells";
    fs::create_directories(initDir);
    makePlane(3, 4, 0.0)->save(initDir / "shell_a.tifxyz", true);
    makePlane(3, 4, 1.0)->save(initDir / "other.tifxyz", true);
    fs::create_directories(initDir / "shell_b");

    const auto candidates = vc::atlas::loadInitShellCandidates(initDir);
    REQUIRE(candidates.size() == 1);
    CHECK(candidates[0].name == "shell_a");
    CHECK(candidates[0].path.filename() == fs::path("shell_a.tifxyz"));
}

TEST_CASE("Atlas init shell loading reports missing and empty dirs")
{
    const fs::path root = tempRoot("vc_atlas_init_shell_missing");
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(
        R"({"version":1})",
        root / "dataset.lasagna.json");
    CHECK_THROWS_WITH_AS(
        vc::atlas::initShellDirectoryFromManifest(manifest),
        doctest::Contains("missing init_shell_dir"),
        std::runtime_error);

    const fs::path initDir = root / "empty";
    fs::create_directories(initDir);
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadInitShellCandidates(initDir),
        doctest::Contains("contains no shell_*.tifxyz"),
        std::runtime_error);
}

TEST_CASE("Atlas mapping reports incomplete fibers with fewer than two line anchors")
{
    auto surface = makePlane(5, 8, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});

    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/1.json";
    fiber.linePoints = {{1.0, 2.0, 1.0}};

    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    CHECK_THROWS_WITH_AS(
        vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler),
        doctest::Contains("incomplete atlas mapping"),
        std::runtime_error);
}
