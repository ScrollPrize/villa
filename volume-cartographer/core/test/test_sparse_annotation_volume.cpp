#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/FiberAnnotationVolume.hpp"
#include "vc/core/util/SparseAnnotationVolume.hpp"

namespace fs = std::filesystem;

namespace {

fs::path tempPath(const std::string& tag)
{
    static std::atomic<unsigned> sequence{0};
    auto path = fs::temp_directory_path() /
                ("vc_sparse_annotation_" + tag + "_" +
                 std::to_string(sequence.fetch_add(1)));
    fs::remove_all(path);
    return path;
}

vc::SparseAnnotationVolumeSpec basicSpec()
{
    vc::SparseAnnotationVolumeSpec spec;
    spec.shapeZYX = {16, 16, 16};
    spec.chunkShapeZYX = {8, 8, 8};
    spec.compressor = "none";
    return spec;
}

std::vector<std::uint16_t> readChunk(const fs::path& root,
                                     std::size_t cz = 0,
                                     std::size_t cy = 0,
                                     std::size_t cx = 0,
                                     std::size_t level = 0)
{
    vc::VcDataset dataset(root / std::to_string(level));
    std::vector<std::uint16_t> values(dataset.defaultChunkSize());
    dataset.readChunkOrFill(cz, cy, cx, values.data());
    return values;
}

std::uint16_t at(const std::vector<std::uint16_t>& chunk,
                 std::size_t z,
                 std::size_t y,
                 std::size_t x,
                 std::size_t side = 8)
{
    return chunk[(z * side + y) * side + x];
}

vc::AnnotationPointBatch batch(std::uint16_t label,
                               std::initializer_list<std::array<double, 3>> points,
                               vc::AnnotationGeometryMode mode =
                                   vc::AnnotationGeometryMode::Points,
                               double radius = 0.0)
{
    vc::AnnotationPointBatch result;
    result.label = label;
    result.coordinates = points;
    result.coordinateOrder = vc::AnnotationCoordinateOrder::ZYX;
    result.geometryMode = mode;
    result.radius = radius;
    return result;
}

} // namespace

TEST_CASE("Sparse annotation points do not connect, while ordered polylines do")
{
    auto pointsPath = tempPath("points");
    vc::writeSparseAnnotationVolume(
        basicSpec(), {batch(3, {{1, 1, 1}, {1, 1, 5}})}, pointsPath);
    auto pointsChunk = readChunk(pointsPath);
    CHECK(at(pointsChunk, 1, 1, 1) == 3);
    CHECK(at(pointsChunk, 1, 1, 3) == 0);
    CHECK(at(pointsChunk, 1, 1, 5) == 3);

    auto linePath = tempPath("line");
    vc::writeSparseAnnotationVolume(
        basicSpec(),
        {batch(9, {{1, 1, 1}, {1, 1, 5}}, vc::AnnotationGeometryMode::OrderedPolyline)},
        linePath);
    auto lineChunk = readChunk(linePath);
    for (std::size_t x = 1; x <= 5; ++x)
        CHECK(at(lineChunk, 1, 1, x) == 9);

    fs::remove_all(pointsPath);
    fs::remove_all(linePath);
}

TEST_CASE("Sparse annotation batches never connect and later batches win")
{
    auto path = tempPath("batch_priority");
    vc::SparseAnnotationVolumeWriter writer(basicSpec());
    writer.addBatch(batch(5, {{2, 2, 1}, {2, 2, 2}},
                          vc::AnnotationGeometryMode::OrderedPolyline));
    writer.addBatch(batch(5, {{2, 2, 5}, {2, 2, 6}},
                          vc::AnnotationGeometryMode::OrderedPolyline));
    writer.addBatch(batch(65535, {{2, 2, 2}}));
    CHECK(writer.batchCount() == 3);
    CHECK(writer.occupiedChunkCount() == 1);
    writer.finish(path);

    auto chunk = readChunk(path);
    CHECK(at(chunk, 2, 2, 1) == 5);
    CHECK(at(chunk, 2, 2, 2) == 65535);
    CHECK(at(chunk, 2, 2, 3) == 0);
    CHECK(at(chunk, 2, 2, 5) == 5);
    fs::remove_all(path);
}

TEST_CASE("Sparse annotation coordinates, rounding, spheres, and clipping")
{
    auto path = tempPath("geometry");
    auto spec = basicSpec();
    vc::AnnotationPointBatch xyz;
    xyz.label = 7;
    xyz.coordinates = {{3.1, 2.2, 1.4}};
    xyz.coordinateOrder = vc::AnnotationCoordinateOrder::XYZ;
    xyz.radius = 1.0;
    vc::AnnotationPointBatch clipped = batch(11, {{-0.4, 0, 0}},
                                              vc::AnnotationGeometryMode::Points,
                                              1.0);
    vc::writeSparseAnnotationVolume(spec, {xyz, clipped}, path);

    auto chunk = readChunk(path);
    CHECK(at(chunk, 1, 2, 3) == 7);
    CHECK(at(chunk, 1, 2, 4) == 7);
    CHECK(at(chunk, 1, 3, 3) == 7);
    CHECK(at(chunk, 0, 0, 0) == 11);
    CHECK(at(chunk, 0, 0, 1) == 11);
    fs::remove_all(path);
}

TEST_CASE("Sparse annotation validates labels, coordinates, radius, and destination")
{
    auto spec = basicSpec();
    vc::SparseAnnotationVolumeWriter writer(spec);
    CHECK_THROWS(writer.addBatch(batch(0, {{1, 1, 1}})));

    auto invalidPoint = batch(1, {{1, 1, std::numeric_limits<double>::infinity()}});
    CHECK_THROWS(writer.addBatch(invalidPoint));
    auto invalidRadius = batch(1, {{1, 1, 1}});
    invalidRadius.radius = -1.0;
    CHECK_THROWS(writer.addBatch(invalidRadius));

    auto path = tempPath("finish_once");
    writer.finish(path);
    CHECK_THROWS(writer.finish(path));
    CHECK_THROWS(writer.addBatch(batch(1, {{1, 1, 1}})));
    fs::remove_all(path);
}

TEST_CASE("Sparse annotation output is sparse OME-Zarr with caller metadata and pyramids")
{
    auto path = tempPath("output");
    auto spec = basicSpec();
    spec.compressor = "zstd";
    spec.compressionLevel = 3;
    spec.rootAttributes["caller"] = utils::Json{{"kind", "fiber"}};
    spec.pyramidLevels.push_back({{8, 8, 8}, {2.0, 2.0, 2.0}, {0.0, 0.0, 0.0}});
    vc::writeSparseAnnotationVolume(spec, {batch(256, {{4, 4, 4}})}, path);

    const auto attrs = vc::readZarrAttributes(path);
    REQUIRE(attrs.contains("vc_annotation_volume"));
    CHECK(attrs["vc_annotation_volume"]["dtype"].get_string() == "uint16");
    CHECK(attrs["caller"]["kind"].get_string() == "fiber");
    REQUIRE(attrs["multiscales"][0]["datasets"].size() == 2);

    const auto zarray = utils::Json::parse_file(path / "0" / ".zarray");
    CHECK(zarray["dtype"].get_string() == "<u2");
    CHECK(zarray["compressor"]["id"].get_string() == "zstd");
    CHECK(zarray["compressor"]["level"].get_int() == 3);

    vc::VcDataset base(path / "0");
    CHECK(base.chunkExists(0, 0, 0));
    CHECK_FALSE(base.chunkExists(1, 1, 1));
    auto pyramidChunk = readChunk(path, 0, 0, 0, 1);
    CHECK(at(pyramidChunk, 2, 2, 2) == 256);
    fs::remove_all(path);
}

TEST_CASE("Fiber adapter assigns stable labels, filters coordinates, and builds two radii")
{
    vc::FiberAnnotationInput beta;
    beta.identity = "beta";
    beta.sourcePath = "b.json";
    beta.controlPointsXYZ = {{{1, 2, 3}}, {{4, 5, 6}}};

    vc::FiberAnnotationInput alpha;
    alpha.identity = "alpha";
    alpha.sourcePath = "a.json";
    alpha.linePointsXYZ = {{{2, 2, 2}}, {{3, 3, 3}}};
    alpha.controlPointsXYZ = {{{2, 2, 2}}};
    alpha.coordinateSpace = "target";

    vc::FiberAnnotationInput foreign;
    foreign.identity = "foreign";
    foreign.linePointsXYZ = {{{1, 1, 1}}};
    foreign.coordinateSpace = "other";

    const auto result = vc::makeFiberAnnotationBatches(
        {beta, foreign, alpha}, "target", 2.0);
    REQUIRE(result.labels.size() == 2);
    CHECK(result.labels[0].identity == "alpha");
    CHECK(result.labels[0].label == 1);
    CHECK(result.labels[1].identity == "beta");
    CHECK(result.labels[1].label == 2);
    REQUIRE(result.batches.size() == 4);
    CHECK(result.batches[0].geometryMode == vc::AnnotationGeometryMode::OrderedPolyline);
    CHECK(result.batches[0].radius == 1.0);
    CHECK(result.batches[0].coordinates[0][0] == 4.0);
    CHECK(result.batches[1].geometryMode == vc::AnnotationGeometryMode::Points);
    CHECK(result.batches[1].radius == 3.0);
    // beta has no line_points, so its controls are also the ordered centerline.
    CHECK(result.batches[2].label == 2);
    CHECK(result.batches[2].geometryMode == vc::AnnotationGeometryMode::OrderedPolyline);
    CHECK(result.batches[3].geometryMode == vc::AnnotationGeometryMode::Points);

    const auto attrs = vc::fiberAnnotationAttributes(result, "target");
    CHECK(attrs["fiber_labels"]["1"]["path"].get_string() == "a.json");
    CHECK(attrs["vc_open_data_coordinate_space"].get_string() == "target");
}
