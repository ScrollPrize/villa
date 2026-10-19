// Python bindings for the native fiber beam tracer (vc::fiber_tracer).
//
// Coordinates are VC trace voxels (xyz), exactly as the C++ API.
// Dataset handles own their samplers; native tracing releases the GIL.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <Python.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace ft = vc::fiber_tracer;

namespace {

using Vec3 = std::array<double, 3>;
using PointsIn = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

cv::Vec3d toVec(const Vec3& value)
{
    return {value[0], value[1], value[2]};
}

Vec3 fromVec(const cv::Vec3d& value)
{
    return {value[0], value[1], value[2]};
}

std::vector<cv::Vec3d> toPoints(const PointsIn& array, const char* name)
{
    if (array.shape(1) != 3) {
        throw std::invalid_argument(std::string(name) + " must have shape (N, 3)");
    }
    std::vector<cv::Vec3d> out;
    out.reserve(array.shape(0));
    const double* data = array.data();
    for (size_t index = 0; index < array.shape(0); ++index) {
        out.emplace_back(data[3 * index], data[3 * index + 1], data[3 * index + 2]);
    }
    return out;
}

// Returns a numpy array that owns a heap copy of `data`; returned as a generic
// object so property getters do not fall under the reference_internal policy.
template <typename T>
nb::object ownedArray(std::vector<T>&& data, std::initializer_list<size_t> shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::cast(nb::ndarray<T, nb::numpy, nb::c_contig>(heap->data(), shape, owner),
                    nb::rv_policy::move);
}

nb::object pointsArray(const std::vector<cv::Vec3d>& points)
{
    std::vector<double> flat;
    flat.reserve(points.size() * 3);
    for (const auto& point : points) {
        flat.push_back(point[0]);
        flat.push_back(point[1]);
        flat.push_back(point[2]);
    }
    return ownedArray(std::move(flat), {points.size(), size_t{3}});
}

// Decode the exact compact axial encoding used by the native sampler. Zero
// bytes are missing observations and must not become a usable direction.
nb::object decodeDirections(
    nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> nx,
    nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> ny)
{
    if (nx.size() != ny.size())
        throw std::invalid_argument("nx and ny must have the same length");
    std::vector<double> values(nx.size() * 3, 0.0);
    {
        nb::gil_scoped_release release;
        for (size_t i = 0; i < nx.size(); ++i) {
            if (nx.data()[i] == 0 || ny.data()[i] == 0)
                continue;
            const auto direction = vc::lasagna::decodeCompactNormalFromRaw(nx.data()[i], ny.data()[i]);
            for (size_t axis = 0; axis < 3; ++axis)
                values[i * 3 + axis] = direction[axis];
        }
    }
    return ownedArray(std::move(values), {nx.size(), size_t{3}});
}

// ---------------------------------------------------------------------------
// Dataset handles. The LasagnaDataset must outlive the field/sampler built on it.

struct PredictionFieldHandle {
    std::unique_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::unique_ptr<ft::FiberPredictionField> field;
    ft::FiberPredictionTraceScales scales;
    std::string location;
};

struct NormalSamplerHandle {
    std::unique_ptr<vc::lasagna::LasagnaDataset> dataset;
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    double workingToBaseScale = 1.0;
    std::string location;
};

vc::lasagna::LasagnaDatasetOpenOptions openOptions(
    const std::optional<std::string>& remoteCacheRoot,
    double workingToBaseScale)
{
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = workingToBaseScale;
    if (remoteCacheRoot.has_value())
        options.remoteCacheRoot = *remoteCacheRoot;
    return options;
}

std::unique_ptr<PredictionFieldHandle> openPredictionField(
    const std::string& location,
    size_t cacheBytes,
    int scaledownPower,
    const std::optional<std::string>& remoteCacheRoot)
{
    auto handle = std::make_unique<PredictionFieldHandle>();
    {
        nb::gil_scoped_release release;
        const auto opened = vc::lasagna::LasagnaDataset::openLocation(
            location, openOptions(remoteCacheRoot, 1.0));
        handle->scales = ft::resolveFiberPredictionTraceScales(
            opened.manifest(), scaledownPower);
        auto manifest = opened.manifest();
        manifest.workingToBaseScale = handle->scales.traceToBaseScale;
        handle->dataset = std::make_unique<vc::lasagna::LasagnaDataset>(std::move(manifest));
        handle->field = std::make_unique<ft::FiberPredictionField>(*handle->dataset, cacheBytes);
        handle->location = location;
    }
    return handle;
}

std::unique_ptr<NormalSamplerHandle> openNormalSampler(
    const std::string& location,
    double workingToBaseScale,
    size_t cacheBytes,
    const std::optional<std::string>& remoteCacheRoot)
{
    if (!(workingToBaseScale > 0.0) || !std::isfinite(workingToBaseScale))
        throw std::invalid_argument("working_to_base_scale must be positive");
    auto handle = std::make_unique<NormalSamplerHandle>();
    {
        nb::gil_scoped_release release;
        handle->dataset = std::make_unique<vc::lasagna::LasagnaDataset>(
            vc::lasagna::LasagnaDataset::openLocation(
                location, openOptions(remoteCacheRoot, workingToBaseScale)));
        handle->sampler = std::make_unique<vc::lasagna::LasagnaNormalSampler>(
            *handle->dataset, vc::lasagna::LasagnaNormalSamplerOptions{cacheBytes});
        handle->workingToBaseScale = workingToBaseScale;
        handle->location = location;
    }
    return handle;
}

const vc::lasagna::NormalSampler* samplerPtr(const NormalSamplerHandle* handle)
{
    return handle == nullptr ? nullptr : handle->sampler.get();
}

// ---------------------------------------------------------------------------
// TraceConfig <-> dict. Keys follow the persisted VC3D config naming.

struct ConfigField {
    const char* key;
    void (*set)(ft::FiberTraceConfig&, nb::handle);
    nb::object (*get)(const ft::FiberTraceConfig&);
};

template <typename T, T ft::FiberTraceConfig::*Member>
ConfigField field(const char* key)
{
    return {
        key,
        [](ft::FiberTraceConfig& config, nb::handle value) {
            config.*Member = nb::cast<T>(value);
        },
        [](const ft::FiberTraceConfig& config) { return nb::cast(config.*Member); },
    };
}

const std::vector<ConfigField>& configFields()
{
    static const std::vector<ConfigField> fields = {
        field<double, &ft::FiberTraceConfig::stepVoxels>("step_voxels"),
        field<double, &ft::FiberTraceConfig::coneAngleDegrees>("cone_angle_degrees"),
        field<double, &ft::FiberTraceConfig::coneAngleStepDegrees>("cone_angle_step_degrees"),
        field<int, &ft::FiberTraceConfig::coneGridSize>("cone_grid_size"),
        field<int, &ft::FiberTraceConfig::beamWidth>("beam_width"),
        field<double, &ft::FiberTraceConfig::beamPruneDistanceVoxels>("beam_prune_distance_voxels"),
        field<int, &ft::FiberTraceConfig::beamLookaheadSteps>("beam_lookahead_steps"),
        field<bool, &ft::FiberTraceConfig::lazyLookahead>("lazy_lookahead"),
        field<size_t, &ft::FiberTraceConfig::lookaheadParentCap>("lookahead_parent_cap"),
        field<size_t, &ft::FiberTraceConfig::lookaheadRetryParentCap>("lookahead_retry_parent_cap"),
        field<int, &ft::FiberTraceConfig::parallelThreads>("parallel_threads"),
        field<double, &ft::FiberTraceConfig::smoothnessWeight>("smoothness_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessNormalWeight>("smoothness_normal_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessTangentWeight>("smoothness_tangent_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessFreeAngleDegrees>("smoothness_free_angle_degrees"),
        field<int, &ft::FiberTraceConfig::cumulativeSmoothnessSteps>("cumulative_smoothness_steps"),
        field<double, &ft::FiberTraceConfig::cumulativeSmoothnessTangentWeight>("cumulative_smoothness_tangent_weight"),
        field<double, &ft::FiberTraceConfig::initialFreeAngleDegrees>("initial_free_angle_degrees"),
        field<double, &ft::FiberTraceConfig::maxStepFactor>("max_step_factor"),
        field<double, &ft::FiberTraceConfig::meetingAcceptMaxErrorRatio>("meeting_accept_max_error_ratio"),
        field<double, &ft::FiberTraceConfig::endpointAcceptThresholdBaseVoxels>("endpoint_accept_threshold_base_voxels"),
        field<double, &ft::FiberTraceConfig::traceToBaseScale>("trace_to_base_scale"),
        field<std::optional<double>, &ft::FiberTraceConfig::baseVoxelSizeUm>("base_voxel_size_um"),
    };
    return fields;
}

void setConfigField(ft::FiberTraceConfig& config, const std::string& key, nb::handle value)
{
    for (const auto& entry : configFields()) {
        if (key == entry.key) {
            entry.set(config, value);
            return;
        }
    }
    throw nb::attribute_error(("TraceConfig has no field '" + key + "'").c_str());
}

nb::dict configToDict(const ft::FiberTraceConfig& config)
{
    nb::dict out;
    for (const auto& entry : configFields())
        out[entry.key] = entry.get(config);
    return out;
}

ft::FiberTraceConfig configFromDict(nb::dict values)
{
    ft::FiberTraceConfig config;
    for (auto item : values)
        setConfigField(config, nb::cast<std::string>(item.first), item.second);
    return config;
}


ft::FiberTraceConfig checkedConfig(const PredictionFieldHandle& field,
                                   const ft::FiberTraceConfig& config,
                                   const NormalSamplerHandle* normals)
{
    if (std::abs(config.traceToBaseScale - field.scales.traceToBaseScale) > 1e-9)
        throw std::invalid_argument("config.trace_to_base_scale must match field.trace_to_base_scale");
    if (normals && std::abs(normals->workingToBaseScale - field.scales.traceToBaseScale) > 1e-9)
        throw std::invalid_argument("normal sampler and prediction field coordinate scales differ");
    auto out = config;
    out.profile = nullptr;
    return out;
}

ft::FiberTraceSegmentResult traceSegment(const PredictionFieldHandle& field,
    const PointsIn& line, size_t start, size_t target, const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals)
{
    ft::FiberTraceSegmentRequest request;
    request.referenceLine = toPoints(line, "reference_line");
    for (const auto& p : request.referenceLine)
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(p[axis]))
                throw std::invalid_argument("reference_line must contain finite points");
    request.startIndex = start;
    request.targetIndex = target;
    request.config = checkedConfig(field, config, normals);
    nb::gil_scoped_release release;
    return ft::traceFiberSegment(*field.field, request, samplerPtr(normals));
}

ft::FiberTraceOneWayResult traceExtrapolation(const PredictionFieldHandle& field,
    Vec3 start, Vec3 direction, double distance, const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals)
{
    const auto checked = checkedConfig(field, config, normals);
    nb::gil_scoped_release release;
    return ft::traceFiberExtrapolation(*field.field, toVec(start), toVec(direction),
                                      distance, checked, samplerPtr(normals));
}

} // namespace

NB_MODULE(fiber_trace, m)
{
    m.doc() = "Python bindings for the Volume Cartographer native fiber beam tracer";

    m.def("decode_directions", &decodeDirections, "nx"_a.noconvert(), "ny"_a.noconvert(),
          "Decode flat uint8 nx/ny arrays to unit xyz axes; missing bytes give zero axes.");

    nb::class_<PredictionFieldHandle>(m, "PredictionField")
        .def_prop_ro("trace_to_base_scale",
                     [](const PredictionFieldHandle& self) { return self.scales.traceToBaseScale; })
        .def_prop_ro("prediction_to_base_scale",
                     [](const PredictionFieldHandle& self) { return self.scales.predictionToBaseScale; })
        .def_prop_ro("prediction_spacing_trace_voxels",
                     [](const PredictionFieldHandle& self) {
                         return self.scales.predictionSpacingInTraceVoxels;
                     })
        .def_prop_ro("option_count",
                     [](const PredictionFieldHandle& self) { return self.field->optionCount(); })
        .def_prop_ro("location", [](const PredictionFieldHandle& self) { return self.location; });

    nb::class_<NormalSamplerHandle>(m, "NormalSampler")
        .def_prop_ro("working_to_base_scale",
                     [](const NormalSamplerHandle& self) { return self.workingToBaseScale; })
        .def_prop_ro("location", [](const NormalSamplerHandle& self) { return self.location; });

    m.def("open_prediction_field", &openPredictionField,
          "location"_a, "cache_bytes"_a = size_t{512} << 20, "scaledown_power"_a = 2,
          "remote_cache_root"_a = nb::none(),
          "Open a Lasagna fiber-prediction dataset (presence/nx/ny) as a beam tracer field. "
          "Trace voxels are prediction voxels divided by 2**scaledown_power.");
    m.def("open_normal_sampler", &openNormalSampler,
          "location"_a, "working_to_base_scale"_a, "cache_bytes"_a = size_t{512} << 20,
          "remote_cache_root"_a = nb::none(),
          "Open a Lasagna normal dataset (nx/ny/grad_mag) at the tracer's working scale.");

    nb::class_<ft::FiberTraceConfig>(m, "TraceConfig")
        .def("__init__", [](ft::FiberTraceConfig* self, nb::kwargs kwargs) {
            new (self) ft::FiberTraceConfig(configFromDict(kwargs));
        })
        .def("to_dict", &configToDict)
        .def_static("from_dict", &configFromDict, "values"_a)
        .def("__repr__", [](const ft::FiberTraceConfig& self) {
            return "TraceConfig(" + nb::cast<std::string>(nb::str(configToDict(self))) + ")";
        })
        .def_rw("step_voxels", &ft::FiberTraceConfig::stepVoxels)
        .def_rw("cone_angle_degrees", &ft::FiberTraceConfig::coneAngleDegrees)
        .def_rw("cone_angle_step_degrees", &ft::FiberTraceConfig::coneAngleStepDegrees)
        .def_rw("cone_grid_size", &ft::FiberTraceConfig::coneGridSize)
        .def_rw("beam_width", &ft::FiberTraceConfig::beamWidth)
        .def_rw("beam_prune_distance_voxels", &ft::FiberTraceConfig::beamPruneDistanceVoxels)
        .def_rw("beam_lookahead_steps", &ft::FiberTraceConfig::beamLookaheadSteps)
        .def_rw("lazy_lookahead", &ft::FiberTraceConfig::lazyLookahead)
        .def_rw("lookahead_parent_cap", &ft::FiberTraceConfig::lookaheadParentCap)
        .def_rw("lookahead_retry_parent_cap", &ft::FiberTraceConfig::lookaheadRetryParentCap)
        .def_rw("parallel_threads", &ft::FiberTraceConfig::parallelThreads)
        .def_rw("smoothness_weight", &ft::FiberTraceConfig::smoothnessWeight)
        .def_rw("smoothness_normal_weight", &ft::FiberTraceConfig::smoothnessNormalWeight)
        .def_rw("smoothness_tangent_weight", &ft::FiberTraceConfig::smoothnessTangentWeight)
        .def_rw("smoothness_free_angle_degrees", &ft::FiberTraceConfig::smoothnessFreeAngleDegrees)
        .def_rw("cumulative_smoothness_steps", &ft::FiberTraceConfig::cumulativeSmoothnessSteps)
        .def_rw("cumulative_smoothness_tangent_weight",
                &ft::FiberTraceConfig::cumulativeSmoothnessTangentWeight)
        .def_rw("initial_free_angle_degrees", &ft::FiberTraceConfig::initialFreeAngleDegrees)
        .def_rw("max_step_factor", &ft::FiberTraceConfig::maxStepFactor)
        .def_rw("meeting_accept_max_error_ratio", &ft::FiberTraceConfig::meetingAcceptMaxErrorRatio)
        .def_rw("endpoint_accept_threshold_base_voxels",
                &ft::FiberTraceConfig::endpointAcceptThresholdBaseVoxels)
        .def_rw("trace_to_base_scale", &ft::FiberTraceConfig::traceToBaseScale)
        .def_rw("base_voxel_size_um", &ft::FiberTraceConfig::baseVoxelSizeUm);

    nb::class_<ft::FiberTraceTargetPlane>(m, "TargetPlane")
        .def("__init__", [](ft::FiberTraceTargetPlane* self, std::string name, Vec3 point, Vec3 normal) {
            new (self) ft::FiberTraceTargetPlane{std::move(name), toVec(point), toVec(normal)};
        }, "name"_a, "point"_a, "normal"_a)
        .def_rw("name", &ft::FiberTraceTargetPlane::name)
        .def_prop_rw("point",
                     [](const ft::FiberTraceTargetPlane& self) { return fromVec(self.point); },
                     [](ft::FiberTraceTargetPlane& self, Vec3 value) { self.point = toVec(value); })
        .def_prop_rw("normal",
                     [](const ft::FiberTraceTargetPlane& self) { return fromVec(self.normal); },
                     [](ft::FiberTraceTargetPlane& self, Vec3 value) { self.normal = toVec(value); });

    nb::class_<ft::FiberTraceTargetPlaneCrossing>(m, "TargetPlaneCrossing")
        .def_ro("name", &ft::FiberTraceTargetPlaneCrossing::name)
        .def_prop_ro("point",
                     [](const ft::FiberTraceTargetPlaneCrossing& self) { return fromVec(self.point); })
        .def_ro("in_plane_error_voxels", &ft::FiberTraceTargetPlaneCrossing::inPlaneErrorVoxels);

    nb::class_<ft::FiberTraceOneWayResult>(m, "OneWayResult")
        .def_prop_ro("points",
                     [](const ft::FiberTraceOneWayResult& self) { return pointsArray(self.points); })
        .def_ro("reached_target_plane", &ft::FiberTraceOneWayResult::reachedTargetPlane)
        .def_ro("reached_trace_length", &ft::FiberTraceOneWayResult::reachedTraceLength)
        .def_ro("reason", &ft::FiberTraceOneWayResult::reason)
        .def_ro("steps", &ft::FiberTraceOneWayResult::steps)
        .def_ro("target_plane_crossings", &ft::FiberTraceOneWayResult::targetPlaneCrossings)
        .def_ro("selected_target_plane_name", &ft::FiberTraceOneWayResult::selectedTargetPlaneName)
        .def_prop_ro("selected_target_plane_crossing",
                     [](const ft::FiberTraceOneWayResult& self) -> std::optional<Vec3> {
                         if (!self.selectedTargetPlaneCrossing.has_value())
                             return std::nullopt;
                         return fromVec(*self.selectedTargetPlaneCrossing);
                     })
        .def_ro("selected_target_plane_error_voxels",
                &ft::FiberTraceOneWayResult::selectedTargetPlaneErrorVoxels);

    nb::class_<ft::FiberTraceSegmentResult>(m, "SegmentResult")
        .def_ro("forward", &ft::FiberTraceSegmentResult::forward)
        .def_ro("reverse", &ft::FiberTraceSegmentResult::reverse)
        .def_prop_ro("fused_line",
                     [](const ft::FiberTraceSegmentResult& self) { return pointsArray(self.fusedLine); })
        .def_ro("forward_endpoint_error_trace_voxels",
                &ft::FiberTraceSegmentResult::forwardEndpointErrorTraceVoxels)
        .def_ro("reverse_endpoint_error_trace_voxels",
                &ft::FiberTraceSegmentResult::reverseEndpointErrorTraceVoxels)
        .def_ro("max_endpoint_error_trace_voxels", &ft::FiberTraceSegmentResult::maxEndpointErrorTraceVoxels)
        .def_ro("max_endpoint_error_base_voxels", &ft::FiberTraceSegmentResult::maxEndpointErrorBaseVoxels)
        .def_ro("max_endpoint_error_um", &ft::FiberTraceSegmentResult::maxEndpointErrorUm)
        .def_ro("meeting_error_trace_voxels", &ft::FiberTraceSegmentResult::meetingErrorTraceVoxels)
        .def_ro("meeting_error_base_voxels", &ft::FiberTraceSegmentResult::meetingErrorBaseVoxels)
        .def_ro("meeting_error_ratio", &ft::FiberTraceSegmentResult::meetingErrorRatio)
        .def_ro("meeting_trace_length_trace_voxels",
                &ft::FiberTraceSegmentResult::meetingTraceLengthTraceVoxels)
        .def_ro("meeting_error_um", &ft::FiberTraceSegmentResult::meetingErrorUm)
        .def_ro("meeting_source", &ft::FiberTraceSegmentResult::meetingSource)
        .def_ro("accepted", &ft::FiberTraceSegmentResult::accepted)
        .def_ro("reason", &ft::FiberTraceSegmentResult::reason)
        .def_ro("detail", &ft::FiberTraceSegmentResult::detail);


    m.def("trace_segment", &traceSegment,
          "field"_a, "reference_line"_a, "start_index"_a, "target_index"_a,
          "config"_a, "normal_sampler"_a = nb::none(),
          "Trace between two points of a reference polyline, with VC3D's bidirectional meeting check.");
    m.def("trace_extrapolation", &traceExtrapolation,
          "field"_a, "start"_a, "direction"_a, "distance_voxels"_a,
          "config"_a, "normal_sampler"_a = nb::none(),
          "Extrapolate from a point along an outward tangent for the requested trace-voxel distance.");
}
