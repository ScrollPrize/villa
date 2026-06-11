#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/atlas/Atlas.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
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

class InvalidAtXNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit InvalidAtXNormalSampler(double x) : x_(x) {}

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& p) const override
    {
        if (std::abs(p[0] - x_) < 1.0e-9) {
            return {{0.0, 0.0, 0.0}, false, {}};
        }
        return {cv::Vec3d{0.0, 0.0, 1.0}, true, {}};
    }

private:
    double x_ = 0.0;
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

std::shared_ptr<QuadSurface> makeWrappedCylinder(int rows,
                                                 int uniqueCols,
                                                 double radius,
                                                 bool reverseColumns)
{
    cv::Mat_<cv::Vec3f> points(rows, uniqueCols + 1);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < uniqueCols; ++col) {
            const int sourceCol = reverseColumns ? (uniqueCols - col) % uniqueCols : col;
            const double theta =
                2.0 * 3.14159265358979323846 * static_cast<double>(sourceCol) /
                static_cast<double>(uniqueCols);
            points(row, col) = cv::Vec3f(static_cast<float>(radius * std::cos(theta)),
                                         static_cast<float>(radius * std::sin(theta)),
                                         static_cast<float>(row));
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

void writeText(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
}

void saveSurface(const fs::path& path, const std::shared_ptr<QuadSurface>& surface)
{
    fs::create_directories(path.parent_path());
    surface->save(path, true);
}

void corruptWrappedSeam(const fs::path& tifxyzPath)
{
    cv::Mat x = cv::imread((tifxyzPath / "x.tif").string(), cv::IMREAD_UNCHANGED);
    if (x.empty()) {
        throw std::runtime_error("failed to read saved x.tif");
    }
    x.col(x.cols - 1).setTo(42.0f);
    if (!cv::imwrite((tifxyzPath / "x.tif").string(), x)) {
        throw std::runtime_error("failed to rewrite saved x.tif");
    }
}

void writeValidLasagnaAtlasFixture(const fs::path& volpkgRoot,
                                   const std::string& atlasName = "fiber_atlas")
{
    const fs::path atlasDir = volpkgRoot / "atlases" / atlasName;
    saveSurface(atlasDir / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    writeText(volpkgRoot / "fibers" / "fiber.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[10,20,30]],"control_points":[]})");
    writeText(atlasDir / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/fiber.json","winding_offset":2,"line_anchors":[{"source_index":0,"world":[1,1,0],"atlas":[1,1],"distance":0}],"control_anchors":[]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":")" + atlasName +
              R"(","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":1})");
}

fs::path atlas21FixtureRoot()
{
    if (const char* env = std::getenv("VC_ATLAS21_FIXTURE_ROOT");
        env && *env) {
        return fs::path(env);
    }
    fs::path repoRoot = fs::current_path();
    if (!fs::is_directory(repoRoot / "volume-cartographer")) {
        fs::path cursor = fs::absolute(fs::path(__FILE__)).parent_path();
        for (int i = 0; i < 8; ++i) {
            if (fs::is_directory(cursor / "volume-cartographer")) {
                repoRoot = cursor;
                break;
            }
            cursor = cursor.parent_path();
        }
    }
    return repoRoot.parent_path() / "data" / "test_data" / "atlas_export" / "fiber_21";
}

bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE";
}

bool finiteVec3(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

vc::atlas::AtlasPredSnapSampling predSnapSamplingForXInside(
    const std::function<std::optional<double>(double)>& predDtForX)
{
    vc::atlas::AtlasPredSnapSampling sampling;
    sampling.sampleNormal = [](const cv::Vec3d&) {
        return vc::lasagna::NormalSample{{1.0, 0.0, 0.0}, true, {}};
    };
    sampling.samplePredDt = [predDtForX](const cv::Vec3d& point) {
        return predDtForX(point[0]);
    };
    sampling.windingDistance = [](const cv::Vec3d& a, const cv::Vec3d& b, double) {
        return cv::norm(b - a);
    };
    return sampling;
}

std::optional<nlohmann::json> atlas21SampleRecord(const std::string& objectId,
                                                  const vc::atlas::FiberMapping& mapping,
                                                  const vc::atlas::AtlasAnchor& anchor,
                                                  bool isControlPoint,
                                                  const QuadSurface& baseSurface)
{
    const int periodColumns = vc::atlas::atlasHorizontalPeriodColumns(baseSurface);
    const double actualU = vc::atlas::actualAtlasU(anchor, mapping, periodColumns);
    const auto basePoint =
        vc::atlas::atlasAnchorBasePoint(anchor, mapping, baseSurface);
    if (!basePoint.has_value() || !finiteVec3(*basePoint)) {
        return std::nullopt;
    }
    return nlohmann::json{
        {"object_id", objectId},
        {"source_index", anchor.sourceIndex},
        {"is_control_point", isControlPoint},
        {"atlas_u", anchor.atlasU},
        {"atlas_v", anchor.atlasV},
        {"winding_offset", mapping.windingOffset},
        {"actual_u", actualU},
        {"base_xyz", {(*basePoint)[0], (*basePoint)[1], (*basePoint)[2]}},
    };
}

nlohmann::json atlas21ExpectedSamples(const vc::atlas::LasagnaAtlasExport& exportData,
                                      const QuadSurface& baseSurface)
{
    nlohmann::json records = nlohmann::json::array();
    constexpr size_t kMaxLineSamplesPerMapping = 8;
    for (const auto& mapping : exportData.atlas.fibers) {
        const std::string objectId = mapping.fiberPath.generic_string();
        std::unordered_set<int> controlIndices;
        int controlStart = std::numeric_limits<int>::max();
        int controlEnd = std::numeric_limits<int>::min();
        for (const auto& anchor : mapping.controlAnchors) {
            controlIndices.insert(anchor.sourceIndex);
            controlStart = std::min(controlStart, anchor.sourceIndex);
            controlEnd = std::max(controlEnd, anchor.sourceIndex);
            if (auto record =
                    atlas21SampleRecord(objectId, mapping, anchor, true, baseSurface)) {
                records.push_back(*record);
            }
        }

        if (mapping.controlAnchors.empty()) {
            continue;
        }
        std::vector<const vc::atlas::AtlasAnchor*> lineCandidates;
        for (const auto& anchor : mapping.lineAnchors) {
            if (anchor.sourceIndex < controlStart || anchor.sourceIndex > controlEnd) {
                continue;
            }
            if (controlIndices.count(anchor.sourceIndex) != 0) {
                continue;
            }
            lineCandidates.push_back(&anchor);
        }
        if (lineCandidates.empty()) {
            continue;
        }
        const size_t wanted = std::min(kMaxLineSamplesPerMapping, lineCandidates.size());
        std::unordered_set<size_t> picked;
        for (size_t i = 0; i < wanted; ++i) {
            const size_t pick = wanted == 1
                ? 0
                : (i * (lineCandidates.size() - 1)) / (wanted - 1);
            if (picked.insert(pick).second) {
                if (auto record = atlas21SampleRecord(
                        objectId, mapping, *lineCandidates[pick], false, baseSurface)) {
                    records.push_back(*record);
                }
            }
        }
    }
    return records;
}

} // namespace

TEST_CASE("Atlas JSON round trips metadata links and fiber mapping")
{
    const fs::path volpkgRoot = tempRoot("vc_atlas_roundtrip");
    const fs::path atlasDir = volpkgRoot / "atlases" / "fiber_1";
    writeText(volpkgRoot / "fibers" / "1.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[1,2,3]],"control_points":[[1,2,3]]})");

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

    atlas.save(atlasDir);
    const std::string metadata = readText(atlasDir / "metadata.json");
    CHECK(metadata.find("\"version\": 5") != std::string::npos);
    CHECK(metadata.find("\"zero_winding_column\": 3") != std::string::npos);
    CHECK(metadata.find("idx_rotation_columns") == std::string::npos);
    const std::string mappingJson = readText(atlasDir / "mappings" / "fibers" / "1.json");
    CHECK(mappingJson.find("\"version\": 4") != std::string::npos);
    CHECK(mappingJson.find("winding_offset") == std::string::npos);

    const auto loaded = vc::atlas::Atlas::load(atlasDir);

    REQUIRE(loaded.metadata.name == "fiber_1");
    CHECK(loaded.metadata.version == 5);
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
    CHECK(loaded.fibers[0].windingOffset == 0);
    REQUIRE(loaded.fibers[0].lineAnchors.size() == 1);
    CHECK(loaded.fibers[0].lineAnchors[0].atlasV == doctest::Approx(5.0));
}

TEST_CASE("Atlas pred-snap search applies winding limits and inward weighting")
{
    const cv::Vec3d control{0.0, 0.0, 0.0};
    const cv::Vec3d normal{1.0, 0.0, 0.0};

    auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.30) return 166.0;
        if (x <= -0.10) return 170.0;
        return 80.0;
    });
    auto snap = vc::atlas::findAtlasPredSnapPoint(control, normal, sampling);
    REQUIRE(snap.has_value());
    REQUIRE(snap->predSnapPoint.has_value());
    CHECK(snap->direction == vc::atlas::AtlasPredSnapDirection::Outside);
    CHECK((*snap->predSnapPoint)[0] == doctest::Approx(0.30).epsilon(0.05));
    CHECK(snap->weightedFirstHitWindingDistance.value() == doctest::Approx(0.30).epsilon(0.05));

    sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.20) return 114.0;
        return 80.0;
    });
    snap = vc::atlas::findAtlasPredSnapPoint(control, normal, sampling);
    REQUIRE(snap.has_value());
    REQUIRE(snap->predSnapPoint.has_value());
    CHECK((*snap->predSnapPoint)[0] == doctest::Approx(0.20).epsilon(0.05));

    sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.30) return 166.0;
        if (x <= -0.05) return 170.0;
        return 80.0;
    });
    snap = vc::atlas::findAtlasPredSnapPoint(control, normal, sampling);
    REQUIRE(snap.has_value());
    REQUIRE(snap->predSnapPoint.has_value());
    CHECK(snap->direction == vc::atlas::AtlasPredSnapDirection::Inside);
    CHECK((*snap->predSnapPoint)[0] == doctest::Approx(-0.05).epsilon(0.05));
    CHECK(snap->weightedFirstHitWindingDistance.value() == doctest::Approx(0.20).epsilon(0.05));

    sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.55) return 166.0;
        if (x <= -0.30) return 170.0;
        return 80.0;
    });
    snap = vc::atlas::findAtlasPredSnapPoint(control, normal, sampling);
    REQUIRE(snap.has_value());
    CHECK_FALSE(snap->predSnapPoint.has_value());
}

TEST_CASE("Atlas pred-snap search starts inside by direct maximum climb")
{
    const auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x < 0.0) return 175.0;
        if (x >= 0.20) return 170.0;
        return 166.0;
    });
    const auto snap = vc::atlas::findAtlasPredSnapPoint(
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        sampling);

    REQUIRE(snap.has_value());
    REQUIRE(snap->predSnapPoint.has_value());
    CHECK(snap->direction == vc::atlas::AtlasPredSnapDirection::Inside);
    CHECK((*snap->predSnapPoint)[0] == doctest::Approx(-0.05).epsilon(0.05));
    CHECK(snap->predDtValue.value() == doctest::Approx(175.0));
}

TEST_CASE("Atlas pred-snap search policy can favor deeper inward hits")
{
    auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.30) return 166.0;
        if (x <= -0.40) return 170.0;
        return 80.0;
    });
    sampling.outwardWindingLimit = 0.25;
    sampling.inwardWindingLimit = 0.5;
    sampling.inwardFirstHitWeight = 0.25;

    const auto snap = vc::atlas::findAtlasPredSnapPoint(
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        sampling);

    REQUIRE(snap.has_value());
    REQUIRE(snap->predSnapPoint.has_value());
    CHECK(snap->direction == vc::atlas::AtlasPredSnapDirection::Inside);
    CHECK((*snap->predSnapPoint)[0] == doctest::Approx(-0.40).epsilon(0.05));
    CHECK(snap->weightedFirstHitWindingDistance.value() == doctest::Approx(0.10).epsilon(0.05));
}

TEST_CASE("Atlas pred-snap generation uses control source line indices for anchors")
{
    vc::atlas::FiberInput fiber;
    fiber.fiberPath = fs::path("fibers") / "fiber.json";
    fiber.linePoints = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.5, 0.0, 0.0},
    };
    fiber.controlPoints = {
        {1.0, 0.0, 0.0},
    };
    vc::atlas::validateFiberInputControlPoints(fiber);
    REQUIRE(fiber.controlLineIndices == std::vector<int>{2});

    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = fiber.fiberPath;
    mapping.controlAnchors.push_back({2, fiber.controlPoints[0], 2.0, 2.0, 0.0});

    auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        return x >= 1.10 ? 170.0 : 80.0;
    });
    auto baseSurface = makeWrappedPlane(6, 6, 0.0);

    const auto set = vc::atlas::generateAtlasPredSnapSet(
        fiber,
        mapping,
        *baseSurface,
        sampling);

    REQUIRE(set.points.size() == 1);
    REQUIRE(set.points[0].predSnapPoint.has_value());
    CHECK((*set.points[0].predSnapPoint)[0] == doctest::Approx(1.10).epsilon(0.05));
}

TEST_CASE("Atlas pred-snap candidate generation returns first hits and deduplicates")
{
    const cv::Vec3d control{0.0, 0.0, 0.0};
    const cv::Vec3d normal{1.0, 0.0, 0.0};
    auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        if (x >= 0.20) return 170.0;
        if (x <= -0.35) return 168.0;
        return 80.0;
    });

    const auto candidates =
        vc::atlas::findAtlasPredSnapCandidates(control, normal, sampling);
    REQUIRE(candidates.size() == 2);
    CHECK(candidates[0].direction == vc::atlas::AtlasPredSnapDirection::Outside);
    CHECK(candidates[0].point[0] == doctest::Approx(0.20).epsilon(0.05));
    CHECK(candidates[1].direction == vc::atlas::AtlasPredSnapDirection::Inside);
    CHECK(candidates[1].point[0] == doctest::Approx(-0.35).epsilon(0.05));

    sampling = predSnapSamplingForXInside([](double) -> std::optional<double> {
        return 170.0;
    });
    const auto deduped =
        vc::atlas::findAtlasPredSnapCandidates(control, normal, sampling);
    REQUIRE(deduped.size() == 1);
    CHECK(deduped[0].point[0] == doctest::Approx(0.0));
}

TEST_CASE("Atlas pred-snap candidate threshold is configurable")
{
    const cv::Vec3d control{0.0, 0.0, 0.0};
    const cv::Vec3d normal{1.0, 0.0, 0.0};
    auto sampling = predSnapSamplingForXInside([](double x) -> std::optional<double> {
        return x >= 0.20 ? 114.0 : 80.0;
    });

    sampling.predDtThreshold = 110.0;
    auto candidates = vc::atlas::findAtlasPredSnapCandidates(control, normal, sampling);
    REQUIRE(candidates.size() == 1);
    CHECK(candidates[0].point[0] == doctest::Approx(0.20).epsilon(0.05));

    sampling.predDtThreshold = 120.0;
    candidates = vc::atlas::findAtlasPredSnapCandidates(control, normal, sampling);
    CHECK(candidates.empty());
}

TEST_CASE("Atlas snap rank cache keys track candidates and options")
{
    const std::vector<cv::Vec3d> a{{1.0, 2.0, 3.0}};
    const std::vector<cv::Vec3d> b{{4.0, 5.0, 6.0}};
    const nlohmann::json options = {
        {"threshold", 110},
        {"margin_base_voxels", 1000},
        {"source_depth", 0},
        {"amgx_config", nullptr},
    };
    const std::string key1 =
        vc::atlas::atlasSnapRankTermCacheKey("dataset.lasagna.json", options, a, b);
    const std::string key2 =
        vc::atlas::atlasSnapRankTermCacheKey("dataset.lasagna.json", options, a, b);
    CHECK(key1 == key2);

    const std::vector<cv::Vec3d> changedB{{4.0, 5.0, 7.0}};
    const std::string key3 =
        vc::atlas::atlasSnapRankTermCacheKey("dataset.lasagna.json", options, a, changedB);
    CHECK(key1 != key3);

    const std::string key4 =
        vc::atlas::atlasSnapRankTermCacheKey(
            "dataset.lasagna.json",
            nlohmann::json{
                {"threshold", 111},
                {"margin_base_voxels", 1000},
                {"source_depth", 0},
                {"amgx_config", nullptr},
            },
            a,
            b);
    CHECK(key1 != key4);
}

TEST_CASE("Atlas optimization adds adjacent control terms along fibers")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = fs::path("fibers") / "a.json";
    mapping.controlAnchors.push_back({10, {10.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    mapping.controlAnchors.push_back({30, {30.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    mapping.controlAnchors.push_back({50, {50.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    mapping.controlAnchors.push_back({70, {70.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    atlas.fibers = {mapping};

    vc::atlas::AtlasPredSnapSet set;
    set.fiberPath = mapping.fiberPath;
    for (const auto& anchor : mapping.controlAnchors) {
        vc::atlas::AtlasPredSnapPoint point;
        point.fiberPath = mapping.fiberPath;
        point.controlPoint = anchor.world;
        point.predSnapPoint = anchor.world;
        point.candidates.push_back({anchor.world, std::nullopt, std::nullopt, std::nullopt});
        set.points.push_back(point);
    }
    std::unordered_map<std::string, vc::atlas::AtlasPredSnapSet> sets;
    sets.emplace(vc::atlas::atlasFiberPathKey(mapping.fiberPath), std::move(set));

    const auto problem = vc::atlas::buildAtlasSnapOptimizationProblem(atlas, sets);
    CHECK(problem.controls.size() == 4);
    REQUIRE(problem.terms.size() == 3);
    CHECK(problem.terms[0].firstControl == 0);
    CHECK(problem.terms[0].secondControl == 1);
    CHECK(problem.terms[1].firstControl == 1);
    CHECK(problem.terms[1].secondControl == 2);
    CHECK(problem.terms[2].firstControl == 2);
    CHECK(problem.terms[2].secondControl == 3);
}

TEST_CASE("Atlas link expansion creates six pair terms for four bracketing controls")
{
    vc::atlas::Atlas atlas;
    vc::atlas::FiberMapping first;
    first.fiberPath = fs::path("fibers") / "a.json";
    first.controlAnchors.push_back({10, {10.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    first.controlAnchors.push_back({30, {30.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    vc::atlas::FiberMapping second;
    second.fiberPath = fs::path("fibers") / "b.json";
    second.controlAnchors.push_back({15, {15.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    second.controlAnchors.push_back({35, {35.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
    atlas.fibers = {first, second};
    atlas.links.push_back({{first.fiberPath, 20, 0.0, 0.0, 0.0},
                           {second.fiberPath, 25, 0.0, 0.0, 0.0},
                           0});

    std::unordered_map<std::string, vc::atlas::AtlasPredSnapSet> sets;
    for (const auto& mapping : atlas.fibers) {
        vc::atlas::AtlasPredSnapSet set;
        set.fiberPath = mapping.fiberPath;
        for (const auto& anchor : mapping.controlAnchors) {
            vc::atlas::AtlasPredSnapPoint point;
            point.fiberPath = mapping.fiberPath;
            point.controlPoint = anchor.world;
            point.predSnapPoint = anchor.world;
            point.candidates.push_back({anchor.world, std::nullopt, std::nullopt, std::nullopt});
            set.points.push_back(point);
        }
        sets.emplace(vc::atlas::atlasFiberPathKey(mapping.fiberPath), std::move(set));
    }

    const auto problem = vc::atlas::buildAtlasSnapOptimizationProblem(atlas, sets);
    CHECK(problem.controls.size() == 4);
    CHECK(problem.terms.size() == 6);
}

TEST_CASE("Atlas snap optimizer maximizes normalized pair scores and respects fixed controls")
{
    vc::atlas::AtlasSnapOptimizationProblem problem;
    problem.controls = {
        {"a", {}, 0, {}, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}, false, false},
        {"b", {}, 0, {}, {{0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}}, false, false},
        {"c", {}, 0, {}, {{9.0, 9.0, 9.0}}, true, true},
    };
    problem.terms = {
        {"ab", 0, 1},
        {"ac", 0, 2},
    };

    vc::atlas::AtlasSnapPairMatrix ab;
    ab.id = "ab";
    ab.normalizedValues = {{0.0, 0.1}, {0.2, 1.0}};
    vc::atlas::AtlasSnapPairMatrix ac;
    ac.id = "ac";
    ac.normalizedValues = {{0.0}, {0.5}};

    const auto result = vc::atlas::optimizeAtlasSnapCandidates(problem, {ab, ac});
    REQUIRE(result.selectedCandidateIndices.size() == 3);
    CHECK(result.selectedCandidateIndices[0] == 1);
    CHECK(result.selectedCandidateIndices[1] == 1);
    CHECK(result.selectedCandidateIndices[2] == 0);
    CHECK(result.objective == doctest::Approx(1.5));
}

TEST_CASE("Atlas snap pair rank no-accepted-lambda becomes zero contribution")
{
    const vc::atlas::AtlasSnapPairTerm term{"ab", 0, 1};
    const nlohmann::json result = {
        {"id", "ab"},
        {"status", "error"},
        {"error", {
            {"code", "no_accepted_lambda"},
            {"message", "adaptive lambda search found no accepted value"},
        }},
    };

    const auto matrix =
        vc::atlas::atlasSnapPairMatrixFromRankResult(term, 2, 1, result);
    REQUIRE(matrix.rawValues.size() == 2);
    REQUIRE(matrix.normalizedValues.size() == 2);
    CHECK(matrix.rawValues[0][0] == doctest::Approx(0.0));
    CHECK(matrix.rawValues[1][0] == doctest::Approx(0.0));
    CHECK(matrix.normalizedValues[0][0] == doctest::Approx(0.0));
    CHECK(matrix.normalizedValues[1][0] == doctest::Approx(0.0));
    CHECK(matrix.metadata.value("zero_contribution", false));
}

TEST_CASE("Atlas base normal orientation uses central ring perimeter")
{
    auto inwardWoundCylinder = makeWrappedCylinder(9, 32, 10.0, true);
    vc::atlas::FiberMapping mapping;
    mapping.fiberPath = fs::path("fibers") / "fiber.json";
    vc::atlas::AtlasAnchor anchor;
    anchor.sourceIndex = 0;
    anchor.world = {10.0, 0.0, 4.0};
    anchor.atlasU = 0.0;
    anchor.atlasV = 4.0;
    mapping.controlAnchors.push_back(anchor);

    const auto normal = vc::atlas::atlasAnchorBaseNormal(
        anchor,
        mapping,
        *inwardWoundCylinder);
    const auto basePoint = vc::atlas::atlasAnchorBasePoint(
        anchor,
        mapping,
        *inwardWoundCylinder);

    REQUIRE(normal.has_value());
    REQUIRE(basePoint.has_value());
    const cv::Vec3d radial{(*basePoint)[0], (*basePoint)[1], 0.0};
    CHECK(normal->dot(radial) > 0.0);
}

TEST_CASE("Atlas pred-snap attachments are coordinate keyed and ignore stale records")
{
    const fs::path root = tempRoot("atlas_pred_snap_attachment");
    const fs::path atlasDir = root / "atlases" / "a";
    const fs::path fiberPath = fs::path("fibers") / "fiber.json";

    vc::atlas::AtlasPredSnapSet existing;
    existing.fiberPath = fiberPath;
    vc::atlas::AtlasPredSnapPoint manual;
    manual.fiberPath = fiberPath;
    manual.controlPoint = {1.0, 2.0, 3.0};
    manual.predSnapPoint = cv::Vec3d{1.0, 2.0, 4.0};
    manual.source = vc::atlas::AtlasPredSnapSource::Manual;
    existing.points.push_back(manual);
    vc::atlas::AtlasPredSnapPoint stale;
    stale.fiberPath = fiberPath;
    stale.controlPoint = {9.0, 9.0, 9.0};
    stale.predSnapPoint = cv::Vec3d{9.0, 9.0, 10.0};
    existing.points.push_back(stale);
    vc::atlas::AtlasPredSnapPoint oldAutoNull;
    oldAutoNull.fiberPath = fiberPath;
    oldAutoNull.controlPoint = {4.0, 5.0, 6.0};
    oldAutoNull.source = vc::atlas::AtlasPredSnapSource::Auto;
    existing.points.push_back(oldAutoNull);
    vc::atlas::AtlasPredSnapPoint oldAutoSelected;
    oldAutoSelected.fiberPath = fiberPath;
    oldAutoSelected.controlPoint = {7.0, 8.0, 9.0};
    oldAutoSelected.predSnapPoint = cv::Vec3d{7.0, 8.0, 11.0};
    oldAutoSelected.source = vc::atlas::AtlasPredSnapSource::Auto;
    existing.points.push_back(oldAutoSelected);

    vc::atlas::AtlasPredSnapSet generated;
    generated.fiberPath = fiberPath;
    vc::atlas::AtlasPredSnapPoint unchangedAuto;
    unchangedAuto.fiberPath = fiberPath;
    unchangedAuto.controlPoint = manual.controlPoint;
    unchangedAuto.predSnapPoint = cv::Vec3d{1.0, 2.0, 5.0};
    generated.points.push_back(unchangedAuto);
    vc::atlas::AtlasPredSnapPoint added;
    added.fiberPath = fiberPath;
    added.controlPoint = {4.0, 5.0, 6.0};
    added.predSnapPoint = cv::Vec3d{4.0, 5.0, 7.0};
    generated.points.push_back(added);
    vc::atlas::AtlasPredSnapPoint preservedAuto;
    preservedAuto.fiberPath = fiberPath;
    preservedAuto.controlPoint = oldAutoSelected.controlPoint;
    preservedAuto.predSnapPoint = cv::Vec3d{7.0, 8.0, 10.0};
    preservedAuto.candidates.push_back(
        {cv::Vec3d{7.0, 8.0, 10.0}, std::nullopt, std::nullopt, std::nullopt});
    preservedAuto.candidates.push_back(
        {cv::Vec3d{7.0, 8.0, 11.0}, std::nullopt, std::nullopt, std::nullopt});
    preservedAuto.selectedCandidateIndex = 0;
    generated.points.push_back(preservedAuto);

    auto merged = vc::atlas::mergeAtlasPredSnapSetByControlPoint(std::move(existing), generated);
    REQUIRE(merged.points.size() == 3);
    CHECK(merged.points[0].source == vc::atlas::AtlasPredSnapSource::Manual);
    REQUIRE(merged.points[0].predSnapPoint.has_value());
    CHECK((*merged.points[0].predSnapPoint)[2] == doctest::Approx(4.0));
    CHECK(merged.points[1].controlPoint[0] == doctest::Approx(4.0));
    REQUIRE(merged.points[1].predSnapPoint.has_value());
    CHECK((*merged.points[1].predSnapPoint)[2] == doctest::Approx(7.0));
    CHECK(merged.points[2].selectedCandidateIndex == 1);
    REQUIRE(merged.points[2].predSnapPoint.has_value());
    CHECK((*merged.points[2].predSnapPoint)[2] == doctest::Approx(11.0));

    const fs::path attachmentPath =
        vc::atlas::atlasPredSnapAttachmentPath(atlasDir, fiberPath);
    vc::atlas::saveAtlasPredSnapSet(attachmentPath, merged);
    const auto loaded = vc::atlas::loadAtlasPredSnapSet(attachmentPath);
    REQUIRE(loaded.points.size() == 3);
    CHECK(loaded.fiberPath == fiberPath);

    const auto rootJson = nlohmann::json::parse(readText(attachmentPath));
    CHECK(rootJson["type"] == "vc3d_atlas_pred_snap_points");
    CHECK(rootJson["version"] == 1);
    CHECK(rootJson["entries"].is_object());
    fs::remove_all(root);
}

TEST_CASE("Atlas loader rejects obsolete atlas versions with rebuild required")
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
  "winding_offset": 7,
  "line_anchors": [
    {"source_index": 0, "world": [0, 0, 0], "atlas": [1, 2], "distance": 0}
  ],
  "control_anchors": []
})";
    }

    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("rebuild required"),
        std::runtime_error);
}

TEST_CASE("Atlas loader rejects v4 atlases so pred-snap attachments can be rebuilt")
{
    const fs::path root = tempRoot("vc_atlas_v4_pred_snap_rebuild");
    fs::create_directories(root / "mappings" / "fibers");
    writeText(root / "metadata.json",
              R"({"type":"vc3d_atlas","version":4,"name":"old","base_mesh_path":"base_mesh/shell.tifxyz","source_base_mesh_path":"segments/shell","zero_winding_column":0,"seed_line_index":0,"seed_atlas":[0,0]})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("rebuild required"),
        std::runtime_error);
}

TEST_CASE("Atlas loader requires current fiber mapping versions")
{
    const fs::path root = tempRoot("vc_atlas_mapping_version_required");
    fs::create_directories(root / "mappings" / "fibers");
    writeText(root / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"current","base_mesh_path":"base_mesh/shell.tifxyz","source_base_mesh_path":"segments/shell","zero_winding_column":0,"seed_line_index":0,"seed_atlas":[0,0]})");
    writeText(root / "mappings" / "fibers" / "missing_version.json",
              R"({"type":"vc3d_atlas_fiber_mapping","fiber_path":"fibers/1.json","line_anchors":[],"control_anchors":[]})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("missing version"),
        std::runtime_error);

    writeText(root / "mappings" / "fibers" / "missing_version.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/1.json","line_anchors":[],"control_anchors":[]})");
    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("rebuild required"),
        std::runtime_error);

    writeText(root / "mappings" / "fibers" / "missing_version.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":3,"fiber_path":"fibers/1.json","line_anchors":[],"control_anchors":[]})");
    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(root),
        doctest::Contains("rebuild required"),
        std::runtime_error);
}

TEST_CASE("Atlas loader rejects v4 control anchors with stale world coordinates")
{
    const fs::path root = tempRoot("vc_atlas_load_stale_control_world");
    const fs::path atlasDir = root / "atlases" / "fiber_atlas";
    writeText(root / "fibers" / "fiber.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[0,0,0],[1,0,0],[2,0,0]],"control_points":[[1,0,0]]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"fiber_atlas","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})");
    writeText(atlasDir / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/fiber.json","line_anchors":[{"source_index":1,"world":[1,0,0],"atlas":[1,1],"distance":0}],"control_anchors":[{"source_index":1,"world":[99,0,0],"atlas":[1,1],"distance":0}]})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::Atlas::load(atlasDir),
        doctest::Contains("world does not match source fiber control point"),
        std::runtime_error);
}

TEST_CASE("Atlas rebuild remaps source fibers and refreshes link endpoints")
{
    const fs::path root = tempRoot("vc_atlas_rebuild_old_versions");
    const fs::path atlasDir = root / "atlases" / "fiber_atlas";
    cv::Mat_<cv::Vec3f> points(5, 9);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < 8; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         static_cast<float>(0.1 * col));
        }
        points(row, 8) = points(row, 0);
    }
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
    saveSurface(atlasDir / "base_mesh" / "base.tifxyz", surface);
    writeText(root / "fibers" / "a.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[1,2,1.1],[2,2,1.2],[3,2,1.3]],"control_points":[[2,2,1.2]]})");
    writeText(root / "fibers" / "b.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[4,2,1.4],[5,2,1.5],[6,2,1.6]],"control_points":[[5,2,1.5]]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":4,"name":"fiber_atlas","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0,"seed_line_index":1,"seed_atlas":[99,99]})");
    writeText(atlasDir / "links.json",
              R"({"version":1,"links":[{"first":{"object_type":"fiber","fiber_path":"fibers/a.json","source_index":1,"arclength":12.5,"base_atlas":[99,99]},"second":{"object_type":"fiber","fiber_path":"fibers/b.json","source_index":1,"arclength":42.5,"base_atlas":[88,88]},"desired_winding_delta":1}]})");
    writeText(atlasDir / "mappings" / "fibers" / "a.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/a.json","line_anchors":[{"source_index":1,"world":[0,0,0],"atlas":[99,99],"distance":0}],"control_anchors":[{"source_index":0,"world":[0,0,0],"atlas":[99,99],"distance":0}]})");
    writeText(atlasDir / "mappings" / "fibers" / "b.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":2,"fiber_path":"fibers/b.json","line_anchors":[{"source_index":1,"world":[0,0,0],"atlas":[88,88],"distance":0}],"control_anchors":[{"source_index":0,"world":[0,0,0],"atlas":[88,88],"distance":0}]})");

    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    const auto rebuilt = vc::atlas::rebuildAtlasFromSourceFibers(
        atlasDir, root, sampler);

    CHECK(rebuilt.metadata.version == 5);
    REQUIRE(rebuilt.fibers.size() == 2);
    REQUIRE(rebuilt.fibers[0].controlAnchors.size() == 1);
    CHECK(rebuilt.fibers[0].controlAnchors[0].sourceIndex == 1);
    REQUIRE(rebuilt.links.size() == 1);
    CHECK(rebuilt.links[0].first.fiberPath == fs::path("fibers/a.json"));
    CHECK(rebuilt.links[0].first.sourceIndex == 1);
    CHECK(rebuilt.links[0].first.arclength == doctest::Approx(12.5));
    CHECK(rebuilt.links[0].first.atlasU == doctest::Approx(2.0));
    CHECK(rebuilt.links[0].second.fiberPath == fs::path("fibers/b.json"));
    CHECK(rebuilt.links[0].second.sourceIndex == 1);
    CHECK(rebuilt.links[0].second.arclength == doctest::Approx(42.5));
    CHECK(rebuilt.links[0].second.atlasU == doctest::Approx(5.0));
    CHECK(rebuilt.links[0].desiredWindingDelta == 1);

    CHECK(readText(atlasDir / "metadata.json").find("\"version\": 5") != std::string::npos);
    CHECK(readText(atlasDir / "mappings" / "fibers" / "a.json")
              .find("\"version\": 4") != std::string::npos);
    const auto loaded = vc::atlas::Atlas::load(atlasDir);
    CHECK(loaded.metadata.version == 5);
    REQUIRE(loaded.links.size() == 1);
    CHECK(loaded.links[0].first.atlasU == doctest::Approx(2.0));
}

TEST_CASE("Atlas discovery lists metadata-backed atlas directories with display names")
{
    const fs::path root = tempRoot("vc_atlas_discovery");
    writeValidLasagnaAtlasFixture(root, "fiber_atlas");
    fs::create_directories(root / "atlases" / "not_an_atlas");
    writeText(root / "atlases" / "not_an_atlas" / "metadata.json",
              R"({"type":"something_else","name":"skip"})");

    const auto atlases = vc::atlas::discoverAtlasDirectories(root);
    REQUIRE(atlases.size() == 1);
    CHECK(atlases[0].path == root / "atlases" / "fiber_atlas");
    CHECK(atlases[0].name == "fiber_atlas");
}

TEST_CASE("Lasagna atlas export is derived from native atlas metadata and mappings")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export");
    writeValidLasagnaAtlasFixture(root, "fiber_atlas");
    const fs::path fiberPath = root / "fibers" / "fiber.json";
    const std::string fiberBefore = readText(fiberPath);

    const auto exported = vc::atlas::loadLasagnaAtlasExport(root / "atlases" / "fiber_atlas", root);
    CHECK(readText(fiberPath) == fiberBefore);
    CHECK(exported.atlas.metadata.zeroWindingColumn == 1);
    CHECK(exported.basePath == root / "atlases" / "fiber_atlas" / "base_mesh" / "base.tifxyz");
    REQUIRE(exported.objects.size() == 1);
    CHECK(exported.objects[0].id == "fibers/fiber.json");
    CHECK(exported.objects[0].fiberPath == root / "fibers" / "fiber.json");
    CHECK(exported.objects[0].mappingRelativePath == fs::path("mappings/fibers/fiber.json"));
    CHECK(exported.objects[0].windingOffset == 0);

    const auto& compact = exported.compactJson;
    CHECK(compact.at("metadata").at("zero_winding_column").get<int>() == 1);
    CHECK_FALSE(compact.at("metadata").contains("period_columns"));
    CHECK_FALSE(compact.at("metadata").contains("u_offset_columns"));
    REQUIRE(compact.at("maps").size() == 1);
    CHECK(compact.at("maps")[0].at("winding_offset").get<int>() == 0);
    CHECK(compact.at("maps")[0].at("mapping_path").get<std::string>() ==
          "mappings/fibers/fiber.json");
}

TEST_CASE("Lasagna atlas export rejects obsolete fiber mappings before compact export")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export_obsolete_mapping");
    writeValidLasagnaAtlasFixture(root, "fiber_atlas");
    writeText(root / "atlases" / "fiber_atlas" / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":3,"fiber_path":"fibers/fiber.json","winding_offset":2,"line_anchors":[{"source_index":0,"world":[1,1,0],"atlas":[1,1],"distance":0}],"control_anchors":[]})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(root / "atlases" / "fiber_atlas", root),
        doctest::Contains("rebuild required"),
        std::runtime_error);
}

TEST_CASE("Lasagna atlas export rejects v4 control anchors with stale world coordinates")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export_stale_control_world");
    const fs::path atlasDir = root / "atlases" / "fiber_atlas";
    saveSurface(atlasDir / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    writeText(root / "fibers" / "fiber.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[0,0,0],[1,0,0],[2,0,0]],"control_points":[[1,0,0]]})");
    writeText(atlasDir / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/fiber.json","winding_offset":0,"line_anchors":[{"source_index":1,"world":[1,0,0],"atlas":[1,1],"distance":0}],"control_anchors":[{"source_index":1,"world":[99,0,0],"atlas":[1,1],"distance":0}]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"fiber_atlas","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":1})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(atlasDir, root),
        doctest::Contains("world does not match source fiber control point"),
        std::runtime_error);
}

TEST_CASE("Atlas 21 Lasagna export fixture resolves mappings and native base samples")
{
    const fs::path fixtureRoot = atlas21FixtureRoot();
    if (!fs::is_directory(fixtureRoot)) {
        MESSAGE("Atlas 21 fixture root is absent: " << fixtureRoot);
        return;
    }

    const fs::path atlasDir = fixtureRoot / "atlases" / "fiber_21";
    vc::atlas::LasagnaAtlasExport exported;
    try {
        exported = vc::atlas::loadLasagnaAtlasExport(atlasDir, fixtureRoot);
    } catch (const std::runtime_error& ex) {
        if (vc::atlas::atlasLoadErrorRequiresRebuild(ex)) {
            MESSAGE("Atlas 21 fixture needs rebuild for current atlas metadata: " << ex.what());
            return;
        }
        throw;
    }
    CHECK(exported.atlas.metadata.version == 5);
    CHECK(exported.atlas.metadata.name == "fiber_21");
    CHECK(exported.atlas.metadata.baseMeshPath ==
          fs::path("base_mesh/shell_0034.tifxyz"));
    CHECK(exported.basePath ==
          atlasDir / "base_mesh" / "shell_0034.tifxyz");
    REQUIRE(exported.objects.size() == 8);
    REQUIRE(exported.atlas.fibers.size() == 8);

    std::unordered_set<std::string> objectIds;
    for (const auto& object : exported.objects) {
        CHECK(objectIds.insert(object.id).second);
        CHECK(object.fiberPath.parent_path() == fixtureRoot / "fibers");
        CHECK(fs::is_regular_file(object.fiberPath));
        CHECK(fs::is_regular_file(object.mappingPath));
    }

    const QuadSurface baseSurface(exported.basePath);
    for (const auto& mapping : exported.atlas.fibers) {
        CHECK(mapping.controlAnchors.size() > 0);
        CHECK(mapping.lineAnchors.size() > mapping.controlAnchors.size());
        size_t finiteControlAnchors = 0;
        for (const auto& anchor : mapping.controlAnchors) {
            const auto basePoint =
                vc::atlas::atlasAnchorBasePoint(anchor, mapping, baseSurface);
            if (basePoint.has_value() && finiteVec3(*basePoint)) {
                ++finiteControlAnchors;
            }
        }
        CHECK_MESSAGE(
            finiteControlAnchors > 0,
            "mapping has no finite control base samples fiber="
                << mapping.fiberPath.generic_string());
        size_t finiteLineAnchors = 0;
        for (size_t i = 0; i < mapping.lineAnchors.size();
             i += std::max<size_t>(1, mapping.lineAnchors.size() / 5)) {
            const auto basePoint = vc::atlas::atlasAnchorBasePoint(
                mapping.lineAnchors[i], mapping, baseSurface);
            if (basePoint.has_value() && finiteVec3(*basePoint)) {
                ++finiteLineAnchors;
            }
        }
        CHECK_MESSAGE(
            finiteLineAnchors > 0,
            "mapping has no finite representative line base samples fiber="
                << mapping.fiberPath.generic_string());
    }

    const nlohmann::json expected =
        atlas21ExpectedSamples(exported, baseSurface);
    REQUIRE(expected.is_array());
    CHECK(expected.size() > exported.atlas.fibers.size());

    const fs::path expectedPath = fixtureRoot / "expected_cpp_base_samples.json";
    if (envFlagEnabled("VC_ATLAS21_WRITE_EXPECTED")) {
        std::ofstream out(expectedPath);
        out << expected.dump(2) << '\n';
    }
    if (!fs::is_regular_file(expectedPath)) {
        MESSAGE("Atlas 21 expected sample file is absent: " << expectedPath);
        return;
    }

    const nlohmann::json stored = nlohmann::json::parse(readText(expectedPath));
    REQUIRE(stored.is_array());
    REQUIRE(stored.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(stored[i].at("object_id").get<std::string>() ==
              expected[i].at("object_id").get<std::string>());
        CHECK(stored[i].at("source_index").get<int>() ==
              expected[i].at("source_index").get<int>());
        CHECK(stored[i].at("is_control_point").get<bool>() ==
              expected[i].at("is_control_point").get<bool>());
        CHECK(stored[i].at("atlas_u").get<double>() ==
              doctest::Approx(expected[i].at("atlas_u").get<double>()));
        CHECK(stored[i].at("atlas_v").get<double>() ==
              doctest::Approx(expected[i].at("atlas_v").get<double>()));
        CHECK(stored[i].at("winding_offset").get<int>() ==
              expected[i].at("winding_offset").get<int>());
        CHECK(stored[i].at("actual_u").get<double>() ==
              doctest::Approx(expected[i].at("actual_u").get<double>()));
        for (int c = 0; c < 3; ++c) {
            CHECK(stored[i].at("base_xyz").at(c).get<double>() ==
                  doctest::Approx(expected[i].at("base_xyz").at(c).get<double>())
                      .epsilon(1.0e-5));
        }
    }
}

TEST_CASE("Lasagna atlas export derives winding offsets from links without rewriting mappings")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export_link_layout");
    const fs::path atlasDir = root / "atlases" / "fiber_atlas";
    saveSurface(atlasDir / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    fs::create_directories(root / "fibers");
    fs::create_directories(atlasDir / "mappings" / "fibers");
    writeText(root / "fibers" / "root.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[0,0,0]],"control_points":[]})");
    writeText(root / "fibers" / "linked.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[1,0,0]],"control_points":[]})");
    writeText(atlasDir / "mappings" / "fibers" / "0_root.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/root.json","winding_offset":5,"line_anchors":[{"source_index":0,"world":[0,0,0],"atlas":[1,1],"distance":0}],"control_anchors":[]})");
    writeText(atlasDir / "mappings" / "fibers" / "1_linked.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/linked.json","winding_offset":9,"line_anchors":[{"source_index":0,"world":[1,0,0],"atlas":[7,1],"distance":0}],"control_anchors":[]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"fiber_atlas","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})");
    writeText(atlasDir / "links.json",
              R"({"version":1,"links":[{"first":{"object_type":"fiber","fiber_path":"fibers/root.json","source_index":0,"arclength":0,"base_atlas":[1,1]},"second":{"object_type":"fiber","fiber_path":"fibers/linked.json","source_index":0,"arclength":0,"base_atlas":[7,1]},"desired_winding_delta":0}]})");
    const fs::path linkedMapping = atlasDir / "mappings" / "fibers" / "1_linked.json";
    const std::string linkedBefore = readText(linkedMapping);

    const auto exported = vc::atlas::loadLasagnaAtlasExport(atlasDir, root);

    CHECK(readText(linkedMapping) == linkedBefore);
    REQUIRE(exported.objects.size() == 2);
    CHECK(exported.objects[0].windingOffset == 0);
    CHECK(exported.objects[1].windingOffset == -2);
    const auto& maps = exported.compactJson.at("maps");
    REQUIRE(maps.size() == 2);
    CHECK(maps[0].at("winding_offset").get<int>() == 0);
    CHECK(maps[1].at("winding_offset").get<int>() == -2);
}

TEST_CASE("Lasagna atlas export validates missing metadata base fibers and mappings")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export_validation");

    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(root / "atlases" / "missing", root),
        doctest::Contains("Atlas directory not found"),
        std::runtime_error);

    const fs::path missingMetadata = root / "atlases" / "missing_metadata";
    fs::create_directories(missingMetadata);
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(missingMetadata, root),
        doctest::Contains("Atlas metadata.json not found"),
        std::runtime_error);

    const fs::path missingBase = root / "atlases" / "missing_base";
    writeText(root / "fibers" / "fiber.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[10,20,30]],"control_points":[]})");
    writeText(missingBase / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"missing_base","base_mesh_path":"base_mesh/missing.tifxyz","zero_winding_column":0})");
    writeText(missingBase / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/fiber.json","winding_offset":0,"line_anchors":[]})");
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(missingBase, root),
        doctest::Contains("base mesh does not exist"),
        std::runtime_error);

    const fs::path missingFiber = root / "atlases" / "missing_fiber";
    saveSurface(missingFiber / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    writeText(missingFiber / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"missing_fiber","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})");
    writeText(missingFiber / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/does_not_exist.json","winding_offset":0,"line_anchors":[]})");
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(missingFiber, root),
        doctest::Contains("references missing fiber path"),
        std::runtime_error);

    const fs::path missingMap = root / "atlases" / "missing_map";
    saveSurface(missingMap / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    writeText(missingMap / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"missing_map","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})");
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(missingMap, root),
        doctest::Contains("no fiber mappings directory"),
        std::runtime_error);
}

TEST_CASE("Lasagna atlas export rejects non-wrapped base shells")
{
    const fs::path root = tempRoot("vc_atlas_lasagna_export_nonwrapped");
    const fs::path atlasDir = root / "atlases" / "nonwrapped";
    saveSurface(atlasDir / "base_mesh" / "base.tifxyz", makeWrappedPlane(3, 3, 1.0));
    corruptWrappedSeam(atlasDir / "base_mesh" / "base.tifxyz");
    writeText(root / "fibers" / "fiber.json",
              R"({"type":"vc3d_fiber","version":1,"line_points":[[10,20,30]],"control_points":[]})");
    writeText(atlasDir / "mappings" / "fibers" / "fiber.json",
              R"({"type":"vc3d_atlas_fiber_mapping","version":4,"fiber_path":"fibers/fiber.json","winding_offset":0,"line_anchors":[]})");
    writeText(atlasDir / "metadata.json",
              R"({"type":"vc3d_atlas","version":5,"name":"nonwrapped","base_mesh_path":"base_mesh/base.tifxyz","zero_winding_column":0})");

    CHECK_THROWS_WITH_AS(
        vc::atlas::atlasHorizontalPeriodColumns(QuadSurface(atlasDir / "base_mesh" / "base.tifxyz")),
        doctest::Contains("not explicitly wrapped"),
        std::runtime_error);
    CHECK_THROWS_WITH_AS(
        vc::atlas::loadLasagnaAtlasExport(atlasDir, root),
        doctest::Contains("not explicitly wrapped"),
        std::runtime_error);
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
  "version": 5,
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

TEST_CASE("Atlas fiber validation derives ordered control line indices")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
    };
    fiber.controlPoints = {
        {0.0, 0.0, 0.0},
        {1.0 + 5.0e-9, 0.0, 0.0},
        {2.0, 0.0, 0.0},
    };

    vc::atlas::validateFiberInputControlPoints(fiber);
    CHECK(fiber.controlLineIndices == std::vector<int>{0, 1, 2});
}

TEST_CASE("Atlas fiber validation rejects controls not present in line points")
{
    vc::atlas::FiberInput fiber;
    fiber.linePoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    fiber.controlPoints = {{0.5, 0.0, 0.0}};

    CHECK_THROWS_WITH_AS(
        vc::atlas::validateFiberInputControlPoints(fiber),
        doctest::Contains("not an ordered subset"),
        std::runtime_error);
}

TEST_CASE("Atlas fiber validation rejects out-of-order and duplicate controls")
{
    {
        vc::atlas::FiberInput fiber;
        fiber.linePoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        fiber.controlPoints = {{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        CHECK_THROWS_WITH_AS(
            vc::atlas::validateFiberInputControlPoints(fiber),
            doctest::Contains("not an ordered subset"),
            std::runtime_error);
    }
    {
        vc::atlas::FiberInput fiber;
        fiber.linePoints = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        fiber.controlPoints = {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        CHECK_THROWS_WITH_AS(
            vc::atlas::validateFiberInputControlPoints(fiber),
            doctest::Contains("not an ordered subset"),
            std::runtime_error);
    }
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

TEST_CASE("Atlas base point helper samples wrapped base mesh at anchor coordinates")
{
    auto surface = makeWrappedPlane(4, 4, 0.0);
    const auto p = vc::atlas::atlasBasePointAt(5.5, 1.25, *surface);
    REQUIRE(p.has_value());
    CHECK((*p)[0] == doctest::Approx(1.5));
    CHECK((*p)[1] == doctest::Approx(1.25));
    CHECK((*p)[2] == doctest::Approx(0.0));

    vc::atlas::FiberMapping mapping;
    mapping.windingOffset = 1;
    vc::atlas::AtlasAnchor anchor;
    anchor.atlasU = 1.5;
    anchor.atlasV = 1.25;
    const auto viaAnchor = vc::atlas::atlasAnchorBasePoint(anchor, mapping, *surface);
    REQUIRE(viaAnchor.has_value());
    CHECK((*viaAnchor)[0] == doctest::Approx(1.5));
    CHECK((*viaAnchor)[1] == doctest::Approx(1.25));
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
    CHECK(mapping.lineAnchors[1].world[0] == doctest::Approx(2.0));
    CHECK(mapping.lineAnchors[1].world[1] == doctest::Approx(2.0));
    CHECK(mapping.lineAnchors[1].world[2] == doctest::Approx(1.0));
    REQUIRE(mapping.controlAnchors.size() == 1);
    CHECK(mapping.controlAnchors[0].sourceIndex == 1);
    CHECK(mapping.controlAnchors[0].world[2] == doctest::Approx(1.0));
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
    fiber.controlPoints = {
        {1.0, 2.0, 1.0},
        {3.0, 2.0, 1.0},
    };

    JumpNormalSampler sampler;
    vc::atlas::LineMappingOptions options;
    options.rayHalfLength = 16.0;
    options.mismatchRatio = 1.5;
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler, options);
    REQUIRE(mapping.lineAnchors.size() == 2);
    CHECK(mapping.lineAnchors[0].sourceIndex == 0);
    CHECK(mapping.lineAnchors.back().sourceIndex == 1);
    REQUIRE(mapping.controlAnchors.size() == 1);
    CHECK(mapping.controlAnchors[0].sourceIndex == 0);
}

TEST_CASE("Atlas mapping truncates at interior failures without sparse line anchors")
{
    auto surface = makeWrappedPlane(5, 16, 0.0);
    SurfacePatchIndex index;
    index.rebuild({surface});

    vc::atlas::FiberInput fiber;
    fiber.fiberPath = "fibers/1.json";
    fiber.linePoints = {
        {0.0, 2.0, 1.0},
        {1.0, 2.0, 1.0},
        {2.0, 2.0, 1.0},
        {3.0, 2.0, 1.0},
        {4.0, 2.0, 1.0},
    };
    fiber.controlPoints = {
        {0.0, 2.0, 1.0},
        {2.0, 2.0, 1.0},
        {4.0, 2.0, 1.0},
    };

    InvalidAtXNormalSampler sampler(1.0);
    const auto mapping = vc::atlas::mapFiberToBaseSurface(fiber, *surface, index, sampler);
    REQUIRE(mapping.lineAnchors.size() == 3);
    CHECK(mapping.lineAnchors[0].sourceIndex == 2);
    CHECK(mapping.lineAnchors[1].sourceIndex == 3);
    CHECK(mapping.lineAnchors[2].sourceIndex == 4);
    REQUIRE(mapping.controlAnchors.size() == 2);
    CHECK(mapping.controlAnchors[0].sourceIndex == 2);
    CHECK(mapping.controlAnchors[1].sourceIndex == 4);
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
