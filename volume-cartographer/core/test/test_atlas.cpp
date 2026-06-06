#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/atlas/Atlas.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

std::shared_ptr<QuadSurface> makeWrappedPlane(int rows,
                                              int uniqueCols,
                                              double z,
                                              double yBias = 0.0,
                                              double xBias = 0.0)
{
    cv::Mat_<cv::Vec3f> points(rows, uniqueCols + 1);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < uniqueCols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col + xBias),
                                         static_cast<float>(row + yBias),
                                         static_cast<float>(z));
        }
        points(row, uniqueCols) = points(row, 0);
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

std::string readText(const fs::path& path)
{
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("Atlas JSON round trips metadata links and fiber mapping")
{
    const fs::path root = tempRoot("vc_atlas_roundtrip");

    vc::atlas::Atlas atlas;
    atlas.metadata.name = "fiber_1";
    atlas.metadata.baseMeshPath = "base_mesh/shell.tifxyz";
    atlas.metadata.sourceBaseMeshPath = "segments/shell";
    atlas.metadata.zeroWindingColumn = 3;
    atlas.metadata.seedLineIndex = 1;
    atlas.metadata.seedAtlasU = 4.5;
    atlas.metadata.seedAtlasV = 2.0;
    vc::atlas::AtlasLink link;
    link.first.fiberPath = "fibers/1.json";
    link.first.sourceIndex = 0;
    link.first.arclength = 1.25;
    link.first.atlasU = 4.0;
    link.first.atlasV = 5.0;
    link.second.fiberPath = "fibers/2.json";
    link.second.sourceIndex = 2;
    link.second.arclength = 6.5;
    link.second.atlasU = 12.0;
    link.second.atlasV = 7.0;
    link.desiredWindingDelta = -1;
    atlas.links.push_back(link);

    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = "fibers/1.json";
    mapping.windingOffset = 2;
    mapping.lineAnchors.push_back({0, {1.0, 2.0, 3.0}, 4.0, 5.0, 0.25});
    mapping.controlAnchors.push_back({0, {1.0, 2.0, 3.0}, 4.0, 5.0, 0.25});
    atlas.fibers.push_back(mapping);

    atlas.save(root);
    const std::string metadata = readText(root / "metadata.json");
    CHECK(metadata.find("\"version\": 3") != std::string::npos);
    CHECK(metadata.find("\"zero_winding_column\": 3") != std::string::npos);
    CHECK(metadata.find("idx_rotation_columns") == std::string::npos);

    const auto loaded = vc::atlas::Atlas::load(root);

    REQUIRE(loaded.metadata.name == "fiber_1");
    CHECK(loaded.metadata.version == 3);
    CHECK(loaded.metadata.zeroWindingColumn == 3);
    CHECK(loaded.metadata.seedLineIndex == 1);
    CHECK(loaded.metadata.seedAtlasU == doctest::Approx(4.5));
    REQUIRE(loaded.links.size() == 1);
    CHECK(loaded.links[0].first.fiberPath == fs::path("fibers/1.json"));
    CHECK(loaded.links[0].first.arclength == doctest::Approx(1.25));
    CHECK(loaded.links[0].second.fiberPath == fs::path("fibers/2.json"));
    CHECK(loaded.links[0].desiredWindingDelta == -1);
    REQUIRE(loaded.fibers.size() == 1);
    CHECK(loaded.fibers[0].fiberPath == fs::path("fibers/1.json"));
    CHECK(loaded.fibers[0].windingOffset == 2);
    REQUIRE(loaded.fibers[0].lineAnchors.size() == 1);
    CHECK(loaded.fibers[0].lineAnchors[0].atlasV == doctest::Approx(5.0));
}

TEST_CASE("Atlas loader accepts version 2 mappings without offsets and ignores string links")
{
    const fs::path root = tempRoot("vc_atlas_v2_compat");
    fs::create_directories(root / "mappings" / "fibers");
    {
        std::ofstream out(root / "metadata.json");
        out << R"({
  "type": "vc3d_atlas",
  "version": 2,
  "name": "compat",
  "base_mesh_path": "base_mesh/shell.tifxyz",
  "source_base_mesh_path": "segments/shell",
  "zero_winding_column": 0,
  "seed_line_index": 0,
  "seed_atlas": [0, 0]
})";
    }
    {
        std::ofstream out(root / "links.json");
        out << R"({"links":["legacy placeholder"]})";
    }
    {
        std::ofstream out(root / "mappings" / "fibers" / "one.json");
        out << R"({
  "type": "vc3d_atlas_fiber_mapping",
  "version": 1,
  "fiber_path": "fibers/1.json",
  "line_anchors": [
    {"source_index": 0, "world": [0, 0, 0], "atlas": [1, 2], "distance": 0}
  ],
  "control_anchors": []
})";
    }

    const auto loaded = vc::atlas::Atlas::load(root);
    CHECK(loaded.metadata.version == 2);
    CHECK(loaded.links.empty());
    REQUIRE(loaded.fibers.size() == 1);
    CHECK(loaded.fibers[0].windingOffset == 0);
}

TEST_CASE("Atlas fiber runtime identity map uses canonical paths, not numeric stems")
{
    const auto ids = vc::atlas::makeFiberRuntimeIdentityMap({
        fs::path("fibers/3.json"),
        fs::path("fibers/alice_20260605T184821587_000001.json"),
        fs::path("fibers/bob_20260605T184821587_000001.json"),
        fs::path("fibers/kb_20260605T184821587_000002.json"),
        fs::path("fibers/3.json"),
    });

    CHECK(ids.idForPath("fibers/3.json") == 1);
    CHECK(ids.idForPath("fibers/alice_20260605T184821587_000001.json") == 2);
    CHECK(ids.idForPath("fibers/bob_20260605T184821587_000001.json") == 3);
    CHECK(ids.idForPath("fibers/kb_20260605T184821587_000002.json") == 4);
    CHECK(ids.pathForId(1) == fs::path("fibers/3.json"));
    CHECK(ids.pathForId(4) == fs::path("fibers/kb_20260605T184821587_000002.json"));
    CHECK(ids.canonicalPaths.size() == 4);
}

TEST_CASE("Atlas fiber search split uses atlas path membership")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping mapped;
    mapped.fiberPath = "fibers/kb_20260605T184821587_000002.json";
    atlas.fibers.push_back(std::move(mapped));

    const auto ids = vc::atlas::makeFiberRuntimeIdentityMap({
        fs::path("fibers/3.json"),
        fs::path("fibers/kb_20260605T184821587_000002.json"),
    });
    const auto sets = vc::atlas::atlasFiberSearchSets(atlas, ids);

    CHECK(sets.sourceFiberPaths == std::vector<fs::path>{
        fs::path("fibers/kb_20260605T184821587_000002.json"),
    });
    CHECK(sets.targetFiberPaths == std::vector<fs::path>{
        fs::path("fibers/3.json"),
    });
    CHECK(sets.sourceFiberIds == std::vector<uint64_t>{2});
    CHECK(sets.targetFiberIds == std::vector<uint64_t>{1});
}

TEST_CASE("Atlas loader rejects legacy idx rotation metadata")
{
    const fs::path root = tempRoot("vc_atlas_legacy_metadata");
    {
        std::ofstream out(root / "metadata.json");
        out << R"({
  "type": "vc3d_atlas",
  "version": 1,
  "name": "legacy",
  "base_mesh_path": "base_mesh/shell.tifxyz",
  "source_base_mesh_path": "segments/shell",
  "idx_rotation_columns": 3,
  "seed_line_index": 0,
  "seed_atlas": [0, 0]
})";
    }

    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("unsupported atlas metadata"),
        std::runtime_error);
}

TEST_CASE("Atlas seed selection uses line points without requiring controls")
{
    auto surface = makeWrappedPlane(4, 4, 0.0);
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

TEST_CASE("Atlas zero winding column finds the lowest average Y column")
{
    cv::Mat_<cv::Vec3f> points(2, 5);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols - 1; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>((col == 2 ? -10 : col) + row),
                                         0.0f);
        }
        points(row, points.cols - 1) = points(row, 0);
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    CHECK(vc::atlas::computeZeroWindingColumn(surface) == 2);
}

TEST_CASE("Atlas base mesh copy preserves source columns and explicit wrapped seam")
{
    const fs::path root = tempRoot("vc_atlas_base_mesh_copy");
    cv::Mat_<cv::Vec3f> points(2, 5);
    cv::Mat labels(2, 5, CV_8U);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols - 1; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         static_cast<float>(10 + col));
            labels.at<uint8_t>(row, col) = static_cast<uint8_t>(col + 1);
        }
        points(row, points.cols - 1) = points(row, 0);
        labels.at<uint8_t>(row, points.cols - 1) = 99;
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    surface.setChannel("labels", labels);

    vc::atlas::saveAtlasBaseMeshCopy(surface, root / "base_mesh" / "shell.tifxyz");
    QuadSurface saved(root / "base_mesh" / "shell.tifxyz");
    const auto* out = saved.rawPointsPtr();
    REQUIRE(out != nullptr);
    CHECK(out->cols == 5);
    for (int col = 0; col < out->cols; ++col) {
        CHECK((*out)(0, col)[0] == doctest::Approx(points(0, col)[0]));
        CHECK((*out)(0, col)[2] == doctest::Approx(points(0, col)[2]));
    }
    CHECK((*out)(0, 4)[0] == doctest::Approx((*out)(0, 0)[0]));
    CHECK((*out)(1, 4)[2] == doctest::Approx((*out)(1, 0)[2]));
    CHECK(vc::atlas::atlasHorizontalPeriodColumns(saved) == 4);

    const cv::Mat savedLabels = saved.channel("labels");
    REQUIRE(!savedLabels.empty());
    CHECK(savedLabels.at<uint8_t>(0, 0) == 1);
    CHECK(savedLabels.at<uint8_t>(0, 1) == 2);
    CHECK(savedLabels.at<uint8_t>(0, 2) == 3);
    CHECK(savedLabels.at<uint8_t>(0, 3) == 4);
    CHECK(savedLabels.at<uint8_t>(0, 4) == 99);
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

TEST_CASE("Atlas covered size applies object winding offsets when period is known")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping first;
    first.fiberPath = "fibers/1.json";
    first.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(first));

    vc::atlas::FiberMapping second;
    second.fiberPath = "fibers/2.json";
    second.windingOffset = 2;
    second.lineAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(second));

    const auto size = vc::atlas::mappedObjectCoveredAtlasSize(atlas, cv::Vec2f(1.0f, 1.0f), 4);
    REQUIRE(size.valid);
    CHECK(size.width == doctest::Approx(9.0));
    CHECK(atlas.fibers[1].lineAnchors[0].atlasU == doctest::Approx(2.0));
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
    const cv::Mat_<cv::Vec3f> wrappedPoints = points.clone();

    points(0, points.cols - 1)[0] = 9.0f;
    QuadSurface open(points, cv::Vec2f(1.0f, 1.0f));
    CHECK_THROWS_WITH_AS(
        vc::atlas::atlasHorizontalPeriodColumns(open),
        doctest::Contains("atlas init shell is not explicitly wrapped"),
        std::runtime_error);

    cv::Mat_<cv::Vec3f> oneColumn(2, 1);
    oneColumn.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
    QuadSurface tooNarrow(oneColumn, cv::Vec2f(1.0f, 1.0f));
    CHECK_THROWS_WITH_AS(
        vc::atlas::atlasHorizontalPeriodColumns(tooNarrow),
        doctest::Contains("at least two columns"),
        std::runtime_error);

    cv::Mat_<cv::Vec3f> nonFinite = wrappedPoints.clone();
    nonFinite(1, 0)[0] = std::numeric_limits<float>::quiet_NaN();
    QuadSurface invalidEndpoint(nonFinite, cv::Vec2f(1.0f, 1.0f));
    CHECK_THROWS_WITH_AS(
        vc::atlas::atlasHorizontalPeriodColumns(invalidEndpoint),
        doctest::Contains("atlas init shell is not explicitly wrapped"),
        std::runtime_error);
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

TEST_CASE("Atlas display range includes object winding offsets without mutating anchors")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping first;
    first.fiberPath = "fibers/1.json";
    first.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(first));

    vc::atlas::FiberMapping second;
    second.fiberPath = "fibers/2.json";
    second.windingOffset = 2;
    second.lineAnchors.push_back({0, {}, 2.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(second));

    const auto range = vc::atlas::atlasDisplayRange(atlas, 4);
    CHECK(range.leftmostWinding == 0);
    CHECK(range.rightmostWinding == 2);
    CHECK(range.unwrapCount == 3);
    CHECK(atlas.fibers[1].lineAnchors[0].atlasU == doctest::Approx(2.0));
}

TEST_CASE("Atlas layout flood fills same-winding link offsets from root")
{
    vc::atlas::Atlas atlas;
    atlas.metadata.zeroWindingColumn = 0;
    vc::atlas::FiberMapping root;
    root.fiberPath = "fibers/root.json";
    root.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(root));

    vc::atlas::FiberMapping linked;
    linked.fiberPath = "fibers/linked.json";
    linked.lineAnchors.push_back({0, {}, 9.0, 1.0, 0.0});
    atlas.fibers.push_back(std::move(linked));

    vc::atlas::AtlasLink link;
    link.first.fiberPath = "fibers/root.json";
    link.first.atlasU = 1.0;
    link.first.atlasV = 1.0;
    link.second.fiberPath = "fibers/linked.json";
    link.second.atlasU = 9.0;
    link.second.atlasV = 1.0;
    link.desiredWindingDelta = 0;
    atlas.links.push_back(link);

    vc::atlas::layoutAtlasObjects(atlas, 4);
    CHECK(atlas.fibers[0].windingOffset == 0);
    CHECK(atlas.fibers[1].windingOffset == -2);
}

TEST_CASE("Atlas layout propagates offsets through a link chain")
{
    vc::atlas::Atlas atlas;
    atlas.metadata.zeroWindingColumn = 0;
    for (int i = 0; i < 3; ++i) {
        vc::atlas::FiberMapping mapping;
        mapping.fiberPath = fs::path("fibers") / (std::to_string(i) + ".json");
        mapping.lineAnchors.push_back({0, {}, static_cast<double>(1 + i * 4), 1.0, 0.0});
        atlas.fibers.push_back(std::move(mapping));
    }

    vc::atlas::AtlasLink first;
    first.first.fiberPath = "fibers/0.json";
    first.first.atlasU = 1.0;
    first.first.atlasV = 1.0;
    first.second.fiberPath = "fibers/1.json";
    first.second.atlasU = 5.0;
    first.second.atlasV = 1.0;
    first.desiredWindingDelta = 0;
    atlas.links.push_back(first);

    vc::atlas::AtlasLink second;
    second.first.fiberPath = "fibers/1.json";
    second.first.atlasU = 5.0;
    second.first.atlasV = 1.0;
    second.second.fiberPath = "fibers/2.json";
    second.second.atlasU = 9.0;
    second.second.atlasV = 1.0;
    second.desiredWindingDelta = 0;
    atlas.links.push_back(second);

    vc::atlas::layoutAtlasObjects(atlas, 4);
    CHECK(atlas.fibers[0].windingOffset == 0);
    CHECK(atlas.fibers[1].windingOffset == -1);
    CHECK(atlas.fibers[2].windingOffset == -2);
}

TEST_CASE("Atlas zero winding column changes winding interpretation only")
{
    vc::atlas::Atlas atlas;
    atlas.metadata.zeroWindingColumn = 2;
    vc::atlas::FiberMapping mapping;
    mapping.lineAnchors.push_back({0, {}, 1.0, 1.0, 0.0});
    mapping.lineAnchors.push_back({1, {}, 5.0, 1.0, 0.0});
    atlas.fibers.push_back(mapping);

    const auto shiftedRange = vc::atlas::atlasDisplayRange(atlas, 4);
    CHECK(shiftedRange.leftmostWinding == -1);
    CHECK(shiftedRange.rightmostWinding == 0);
    CHECK(shiftedRange.unwrapCount == 2);
    CHECK(shiftedRange.atlasUOffset == doctest::Approx(-2.0));
    CHECK(vc::atlas::atlasWindingForColumn(1.0, 4, 2) == -1);
    CHECK(vc::atlas::atlasWindingForColumn(1.0, 4, 0) == 0);
    CHECK(atlas.fibers[0].lineAnchors[0].atlasU == doctest::Approx(1.0));
    CHECK(atlas.fibers[0].lineAnchors[1].atlasU == doctest::Approx(5.0));
}

TEST_CASE("Atlas creation keeps base-relative anchor coordinates")
{
    const fs::path root = tempRoot("vc_atlas_base_relative_anchors");
    auto surface = makeWrappedPlane(3, 4, 0.0);
    vc::atlas::SurfaceCandidate base{"shell_a", root / "segments" / "shell_a.tifxyz", surface};
    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/1.json";
    fiber.linePoints = {{1.0, 1.0, 1.0}, {2.0, 1.0, 1.0}};

    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = fiber.fiberPath;
    mapping.lineAnchors.push_back({0, {}, 1.25, 1.0, 0.0});
    mapping.lineAnchors.push_back({1, {}, 2.25, 1.0, 0.0});

    auto atlas = vc::atlas::createSingleFiberAtlas(
        root, "fiber_1", fiber, base, 3, std::move(mapping));

    CHECK(atlas.metadata.zeroWindingColumn == 3);
    REQUIRE(atlas.fibers.size() == 1);
    REQUIRE(atlas.fibers[0].lineAnchors.size() == 2);
    CHECK(atlas.fibers[0].lineAnchors[0].atlasU == doctest::Approx(1.25));
    CHECK(atlas.fibers[0].lineAnchors[1].atlasU == doctest::Approx(2.25));
}

TEST_CASE("Atlas repeated display surface rejects non-wrapped base meshes")
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

    CHECK_THROWS_WITH_AS(
        vc::atlas::repeatedAtlasDisplaySurface(surface, 3),
        doctest::Contains("atlas init shell is not explicitly wrapped"),
        std::runtime_error);
}

TEST_CASE("Atlas repeated wrapped display surface tiles unique period without duplicate seam")
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
        labels.at<uint8_t>(row, points.cols - 1) = 99;
    }
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    surface.setChannel("labels", labels);

    auto single = vc::atlas::repeatedAtlasDisplaySurface(surface, 1);
    const auto* singleOut = single->rawPointsPtr();
    REQUIRE(singleOut != nullptr);
    CHECK(singleOut->rows == 2);
    CHECK(singleOut->cols == 3);
    CHECK((*singleOut)(0, 2)[2] == doctest::Approx(12.0));
    const cv::Mat singleLabels = single->channel("labels");
    REQUIRE(!singleLabels.empty());
    CHECK(singleLabels.cols == 3);
    CHECK(singleLabels.at<uint8_t>(0, 0) == 1);
    CHECK(singleLabels.at<uint8_t>(0, 1) == 2);
    CHECK(singleLabels.at<uint8_t>(0, 2) == 3);

    auto shifted = vc::atlas::repeatedAtlasDisplaySurface(surface, 1, 2);
    const auto* shiftedOut = shifted->rawPointsPtr();
    REQUIRE(shiftedOut != nullptr);
    CHECK(shiftedOut->cols == 3);
    CHECK((*shiftedOut)(0, 0)[2] == doctest::Approx(12.0));
    CHECK((*shiftedOut)(0, 1)[2] == doctest::Approx(10.0));
    CHECK((*shiftedOut)(0, 2)[2] == doctest::Approx(11.0));
    const cv::Mat shiftedLabels = shifted->channel("labels");
    REQUIRE(!shiftedLabels.empty());
    CHECK(shiftedLabels.at<uint8_t>(0, 0) == 3);
    CHECK(shiftedLabels.at<uint8_t>(0, 1) == 1);
    CHECK(shiftedLabels.at<uint8_t>(0, 2) == 2);

    auto repeated = vc::atlas::repeatedAtlasDisplaySurface(surface, 3);
    const auto* out = repeated->rawPointsPtr();
    REQUIRE(out != nullptr);
    CHECK(out->rows == 2);
    CHECK(out->cols == 9);
    CHECK((*out)(0, 0)[2] == doctest::Approx((*out)(0, 3)[2]));
    CHECK((*out)(0, 8)[2] == doctest::Approx(12.0));

    const cv::Mat repeatedLabels = repeated->channel("labels");
    REQUIRE(!repeatedLabels.empty());
    CHECK(repeatedLabels.cols == 9);
    CHECK(repeatedLabels.at<uint8_t>(0, 0) == repeatedLabels.at<uint8_t>(0, 3));
    CHECK(repeatedLabels.at<uint8_t>(0, 6) == 1);
    CHECK(repeatedLabels.at<uint8_t>(0, 7) == 2);
    CHECK(repeatedLabels.at<uint8_t>(0, 8) == 3);
}

TEST_CASE("Atlas maps a synthetic fiber over a simple grid")
{
    auto surface = makeWrappedPlane(5, 8, 0.0);
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
    auto surface = makeWrappedPlane(5, 16, 0.0);
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
    auto surface = makeWrappedPlane(5, 8, 0.0);
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
