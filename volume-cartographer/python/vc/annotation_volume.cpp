#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vc/core/util/SparseAnnotationVolume.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

utils::Json pythonToJson(const nb::object& value)
{
    nb::object dumps = nb::module_::import_("json").attr("dumps");
    return utils::Json::parse(nb::cast<std::string>(dumps(value)));
}

nb::object jsonToPython(const utils::Json& value)
{
    nb::object loads = nb::module_::import_("json").attr("loads");
    return loads(value.dump());
}

std::uint16_t checkedLabel(std::uint64_t label, bool allowZero)
{
    if ((!allowZero && label == 0) || label > std::numeric_limits<std::uint16_t>::max())
        throw std::invalid_argument(allowZero
            ? "fill_value must be in the uint16 range"
            : "label must be between 1 and 65535");
    return static_cast<std::uint16_t>(label);
}

std::vector<std::array<double, 3>> parseCoordinates(const nb::object& input)
{
    nb::object numpy = nb::module_::import_("numpy");
    nb::object contiguous = numpy.attr("ascontiguousarray")(input, "dtype"_a = "float64");
    using Coordinates = nb::ndarray<nb::numpy, const double,
                                    nb::shape<-1, 3>, nb::c_contig>;
    Coordinates values = nb::cast<Coordinates>(contiguous);
    std::vector<std::array<double, 3>> result(values.shape(0));
    for (std::size_t row = 0; row < values.shape(0); ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis)
            result[row][axis] = values(row, axis);
    }
    return result;
}

vc::SparseAnnotationPyramidLevel makeLevel(
    const std::array<std::size_t, 3>& shape,
    const std::array<double, 3>& scale,
    const std::array<double, 3>& translation)
{
    return {shape, scale, translation};
}

vc::SparseAnnotationVolumeSpec makeSpec(
    const std::array<std::size_t, 3>& shape,
    const std::vector<vc::SparseAnnotationPyramidLevel>& pyramidLevels,
    const std::array<std::size_t, 3>& chunkShape,
    const std::string& compressor,
    int compressionLevel,
    std::uint64_t fillValue,
    const nb::object& rootAttributes)
{
    vc::SparseAnnotationVolumeSpec spec;
    spec.shapeZYX = shape;
    spec.pyramidLevels = pyramidLevels;
    spec.chunkShapeZYX = chunkShape;
    spec.compressor = compressor;
    spec.compressionLevel = compressionLevel;
    spec.fillValue = checkedLabel(fillValue, true);
    spec.rootAttributes = pythonToJson(rootAttributes);
    return spec;
}

vc::AnnotationPointBatch makeBatch(
    std::uint64_t label,
    const nb::object& coordinates,
    vc::AnnotationCoordinateOrder coordinateOrder,
    vc::AnnotationGeometryMode geometryMode,
    double radius)
{
    vc::AnnotationPointBatch batch;
    batch.label = checkedLabel(label, false);
    batch.coordinates = parseCoordinates(coordinates);
    batch.coordinateOrder = coordinateOrder;
    batch.geometryMode = geometryMode;
    batch.radius = radius;
    return batch;
}

} // namespace

NB_MODULE(annotation_volume, m)
{
    m.doc() = "Generic sparse uint16 annotation-volume creation";

    nb::enum_<vc::AnnotationCoordinateOrder>(m, "CoordinateOrder")
        .value("XYZ", vc::AnnotationCoordinateOrder::XYZ)
        .value("ZYX", vc::AnnotationCoordinateOrder::ZYX);

    nb::enum_<vc::AnnotationGeometryMode>(m, "GeometryMode")
        .value("POINTS", vc::AnnotationGeometryMode::Points)
        .value("ORDERED_POLYLINE", vc::AnnotationGeometryMode::OrderedPolyline);

    nb::class_<vc::SparseAnnotationPyramidLevel>(m, "PyramidLevel")
        .def(nb::new_(&makeLevel),
             "shape_zyx"_a,
             "scale_zyx"_a = std::array<double, 3>{2.0, 2.0, 2.0},
             "translation_zyx"_a = std::array<double, 3>{0.0, 0.0, 0.0})
        .def_rw("shape_zyx", &vc::SparseAnnotationPyramidLevel::shapeZYX)
        .def_rw("scale_zyx", &vc::SparseAnnotationPyramidLevel::scaleZYX)
        .def_rw("translation_zyx", &vc::SparseAnnotationPyramidLevel::translationZYX);

    nb::class_<vc::SparseAnnotationVolumeSpec>(m, "SparseAnnotationVolumeSpec")
        .def(nb::new_(&makeSpec),
             "shape_zyx"_a,
             "pyramid_levels"_a = std::vector<vc::SparseAnnotationPyramidLevel>{},
             "chunk_shape_zyx"_a = std::array<std::size_t, 3>{64, 64, 64},
             "compressor"_a = "zstd",
             "compression_level"_a = 3,
             "fill_value"_a = 0,
             "root_attributes"_a = nb::dict())
        .def_rw("shape_zyx", &vc::SparseAnnotationVolumeSpec::shapeZYX)
        .def_rw("pyramid_levels", &vc::SparseAnnotationVolumeSpec::pyramidLevels)
        .def_rw("chunk_shape_zyx", &vc::SparseAnnotationVolumeSpec::chunkShapeZYX)
        .def_rw("compressor", &vc::SparseAnnotationVolumeSpec::compressor)
        .def_rw("compression_level", &vc::SparseAnnotationVolumeSpec::compressionLevel)
        .def_prop_rw("fill_value",
            [](const vc::SparseAnnotationVolumeSpec& spec) { return spec.fillValue; },
            [](vc::SparseAnnotationVolumeSpec& spec, std::uint64_t value) {
                spec.fillValue = checkedLabel(value, true);
            })
        .def_prop_rw("root_attributes",
            [](const vc::SparseAnnotationVolumeSpec& spec) {
                return jsonToPython(spec.rootAttributes);
            },
            [](vc::SparseAnnotationVolumeSpec& spec, const nb::object& value) {
                spec.rootAttributes = pythonToJson(value);
            });

    nb::class_<vc::AnnotationPointBatch>(m, "AnnotationPointBatch")
        .def(nb::new_(&makeBatch),
             "label"_a,
             "coordinates"_a,
             "coordinate_order"_a = vc::AnnotationCoordinateOrder::XYZ,
             "geometry_mode"_a = vc::AnnotationGeometryMode::Points,
             "radius"_a = 0.0)
        .def_prop_rw("label",
            [](const vc::AnnotationPointBatch& batch) { return batch.label; },
            [](vc::AnnotationPointBatch& batch, std::uint64_t value) {
                batch.label = checkedLabel(value, false);
            })
        .def_prop_rw("coordinates",
            [](const vc::AnnotationPointBatch& batch) { return batch.coordinates; },
            [](vc::AnnotationPointBatch& batch, const nb::object& value) {
                batch.coordinates = parseCoordinates(value);
            })
        .def_rw("coordinate_order", &vc::AnnotationPointBatch::coordinateOrder)
        .def_rw("geometry_mode", &vc::AnnotationPointBatch::geometryMode)
        .def_rw("radius", &vc::AnnotationPointBatch::radius);

    nb::class_<vc::SparseAnnotationVolumeWriter>(m, "SparseAnnotationVolumeWriter")
        .def(nb::init<vc::SparseAnnotationVolumeSpec>(), "spec"_a)
        .def("add_batch", &vc::SparseAnnotationVolumeWriter::addBatch,
             "batch"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("finish", &vc::SparseAnnotationVolumeWriter::finish,
             "destination"_a, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("batch_count", &vc::SparseAnnotationVolumeWriter::batchCount)
        .def("occupied_chunk_count", &vc::SparseAnnotationVolumeWriter::occupiedChunkCount,
             "level"_a = 0);

    m.def("write_sparse_annotation_volume",
          &vc::writeSparseAnnotationVolume,
          "spec"_a,
          "batches"_a,
          "destination"_a,
          nb::call_guard<nb::gil_scoped_release>());
}
