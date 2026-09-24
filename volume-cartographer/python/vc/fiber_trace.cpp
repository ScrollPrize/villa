// Python bindings for the native fiber beam tracer (vc::fiber_tracer).
//
// All coordinates are VC trace voxels (xyz), exactly as the C++ API. Tracing
// runs with the GIL released; the optional beam hook and progress callbacks
// re-acquire it. A Python exception raised inside a callback aborts the trace
// and is re-raised from the binding call.
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
#include "vc/lasagna/LasagnaNormalSampler.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace ft = vc::fiber_tracer;

namespace {

using Vec3 = std::array<double, 3>;
using PointsIn = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using IndicesIn = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using LossesIn = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

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

nb::object vecArray(const cv::Vec3d& value)
{
    return ownedArray(std::vector<double>{value[0], value[1], value[2]}, {size_t{3}});
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

// ---------------------------------------------------------------------------
// Callback bridge: Python exceptions inside C++ callbacks are stored and the
// trace is aborted with HookAbort; the binding rethrows once it holds the GIL.

struct HookAbort final : std::runtime_error {
    HookAbort() : std::runtime_error("beam hook aborted the trace") {}
};

struct HookPool {
    int round = 0;
    int step = 0;
    int maxSteps = 0;
    std::string phase;
    cv::Vec3d start;
    cv::Vec3d target;
    std::vector<cv::Vec3d> pathPoints;
    std::vector<int64_t> pathOffsets;
    std::vector<float> losses;
    std::vector<int32_t> depth;
    std::vector<double> tracedLength;
    std::vector<bool> reached;
    std::vector<cv::Vec3d> previousStepDirection;
    std::vector<cv::Vec3d> currentSampleDirection;
    std::vector<cv::Vec3d> historyDirection;

    explicit HookPool(const ft::FiberTraceBeamHookEvent& event)
        : round(event.round)
        , step(event.step)
        , maxSteps(event.maxSteps)
        , phase(event.phase)
        , start(event.startPoint)
        , target(event.targetPoint)
    {
        pathOffsets.push_back(0);
        for (const auto& candidate : event.pool) {
            pathPoints.insert(pathPoints.end(), candidate.path.begin(), candidate.path.end());
            pathOffsets.push_back(static_cast<int64_t>(pathPoints.size()));
            losses.push_back(candidate.loss);
            depth.push_back(candidate.depth);
            tracedLength.push_back(candidate.tracedLength);
            reached.push_back(candidate.reached);
            previousStepDirection.push_back(candidate.previousStepDirection);
            currentSampleDirection.push_back(candidate.currentSampleDirection);
            historyDirection.push_back(candidate.historyDirection);
        }
    }

    [[nodiscard]] size_t size() const { return losses.size(); }
};

struct CallbackBridge {
    nb::object hook;
    nb::object progress;
    std::exception_ptr error;

    [[nodiscard]] bool hasHook() const { return hook.is_valid() && !hook.is_none(); }
    [[nodiscard]] bool hasProgress() const { return progress.is_valid() && !progress.is_none(); }

    void rethrow() const
    {
        if (error)
            std::rethrow_exception(error);
    }

    ft::FiberTraceBeamHookOptions hookOptions(int everyRounds, int poolSize)
    {
        ft::FiberTraceBeamHookOptions options;
        options.everyRounds = everyRounds;
        options.poolSize = poolSize;
        if (!hasHook())
            return options;
        options.hook = [this](const ft::FiberTraceBeamHookEvent& event) {
            nb::gil_scoped_acquire acquire;
            try {
                if (PyErr_CheckSignals() != 0)
                    throw nb::python_error();
                nb::object result = hook(nb::cast(HookPool(event)));
                return parseResponse(result, event.pool.size());
            } catch (...) {
                error = std::current_exception();
                throw HookAbort{};
            }
        };
        return options;
    }

    static ft::FiberTraceBeamHookResponse parseResponse(nb::handle result, size_t poolSize)
    {
        ft::FiberTraceBeamHookResponse response;
        if (result.is_none())
            return response;
        if (!nb::isinstance<nb::sequence>(result) || nb::len(result) != 2) {
            throw std::invalid_argument(
                "beam hook must return None or a (losses, stop) pair");
        }
        nb::object losses = result[0];
        nb::object stop = result[1];
        response.stop = nb::cast<bool>(stop);
        if (losses.is_none())
            return response;
        nb::object numpy = nb::module_::import_("numpy");
        nb::object asArray = numpy.attr("ascontiguousarray")(losses, "dtype"_a = numpy.attr("float32"));
        auto array = nb::cast<LossesIn>(asArray);
        if (array.shape(0) != poolSize) {
            throw std::invalid_argument(
                "beam hook returned " + std::to_string(array.shape(0)) +
                " losses for a pool of " + std::to_string(poolSize));
        }
        response.losses.emplace(array.data(), array.data() + array.shape(0));
        return response;
    }

    ft::FiberTraceProgressCallback progressCallback()
    {
        if (!hasProgress())
            return {};
        return [this](const ft::FiberTraceProgress& event) {
            nb::gil_scoped_acquire acquire;
            try {
                nb::dict payload;
                payload["step"] = event.step;
                payload["max_steps"] = event.maxSteps;
                payload["target_plane_progress"] = event.targetPlaneProgress;
                payload["phase"] = event.phase;
                payload["reason"] = event.reason;
                progress(payload);
            } catch (...) {
                error = std::current_exception();
                throw HookAbort{};
            }
        };
    }

    ft::FiberTraceWholeFiberProgressCallback wholeFiberProgressCallback()
    {
        if (!hasProgress())
            return {};
        return [this](const ft::FiberTraceWholeFiberProgress& event) {
            nb::gil_scoped_acquire acquire;
            try {
                nb::dict payload;
                payload["completed_segments"] = event.completedSegments;
                payload["segment_count"] = event.segmentCount;
                payload["current_segment"] = event.currentSegment;
                payload["restart_count"] = event.restartCount;
                payload["restarts_per_kvx"] = event.restartsPerKvx;
                payload["restarts_per_meter"] = event.restartsPerMeter;
                payload["reference_length_meters"] = event.referenceLengthMeters;
                payload["status"] = event.status;
                if (event.hasTraceProgress) {
                    nb::dict trace;
                    trace["step"] = event.traceProgress.step;
                    trace["max_steps"] = event.traceProgress.maxSteps;
                    trace["target_plane_progress"] = event.traceProgress.targetPlaneProgress;
                    trace["phase"] = event.traceProgress.phase;
                    trace["reason"] = event.traceProgress.reason;
                    payload["trace"] = trace;
                }
                progress(payload);
            } catch (...) {
                error = std::current_exception();
                throw HookAbort{};
            }
        };
    }
};

// Runs `fn` without the GIL; a callback abort re-raises the stored Python
// exception once the GIL is held again.
template <typename Fn>
auto runTrace(CallbackBridge& bridge, Fn&& fn) -> decltype(fn())
{
    std::optional<decltype(fn())> result;
    {
        nb::gil_scoped_release release;
        try {
            result.emplace(fn());
        } catch (const HookAbort&) {
        }
    }
    bridge.rethrow();
    if (!result.has_value())
        throw std::runtime_error("beam hook aborted the trace without an error");
    return std::move(*result);
}

ft::FiberTraceConfig checkedConfig(const ft::FiberTraceConfig& config)
{
    ft::FiberTraceConfig out = config;
    out.profile = nullptr;
    return out;
}

// ---------------------------------------------------------------------------
// Trace entry points.

ft::FiberTraceOneWayResult traceOneWay(
    const PredictionFieldHandle& field,
    Vec3 start,
    Vec3 target,
    Vec3 initialDirection,
    std::vector<ft::FiberTraceTargetPlane> targetPlanes,
    std::optional<double> acceptThresholdVoxels,
    double budgetSpanVoxels,
    const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals,
    bool snapTraceToSelectedCrossing,
    nb::object hook,
    int hookEveryRounds,
    int hookPoolSize,
    nb::object progress)
{
    CallbackBridge bridge{std::move(hook), std::move(progress), {}};
    ft::FiberTraceOneWayRequest request;
    request.startPoint = toVec(start);
    request.targetPoint = toVec(target);
    request.initialDirection = toVec(initialDirection);
    request.targetPlanes = std::move(targetPlanes);
    request.targetPlaneAcceptThresholdVoxels = acceptThresholdVoxels;
    request.snapTraceToSelectedCrossing = snapTraceToSelectedCrossing;
    request.budgetSpanVoxels = budgetSpanVoxels;
    request.config = checkedConfig(config);
    request.beamHook = bridge.hookOptions(hookEveryRounds, hookPoolSize);
    const auto callback = bridge.progressCallback();
    return runTrace(bridge, [&] {
        return ft::traceFiberOneWay(*field.field, request, samplerPtr(normals), callback);
    });
}

ft::FiberTraceSegmentResult traceSegment(
    const PredictionFieldHandle& field,
    const PointsIn& referenceLine,
    size_t startIndex,
    size_t targetIndex,
    const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals,
    nb::object hook,
    int hookEveryRounds,
    int hookPoolSize,
    nb::object progress)
{
    CallbackBridge bridge{std::move(hook), std::move(progress), {}};
    ft::FiberTraceSegmentRequest request;
    request.referenceLine = toPoints(referenceLine, "reference_line");
    request.startIndex = startIndex;
    request.targetIndex = targetIndex;
    request.config = checkedConfig(config);
    request.beamHook = bridge.hookOptions(hookEveryRounds, hookPoolSize);
    const auto callback = bridge.progressCallback();
    return runTrace(bridge, [&] {
        return ft::traceFiberSegment(*field.field, request, samplerPtr(normals), callback);
    });
}

ft::FiberTraceOneWayResult traceExtrapolation(
    const PredictionFieldHandle& field,
    Vec3 start,
    Vec3 direction,
    double distanceVoxels,
    const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals,
    nb::object hook,
    int hookEveryRounds,
    int hookPoolSize,
    nb::object progress)
{
    CallbackBridge bridge{std::move(hook), std::move(progress), {}};
    const auto options = bridge.hookOptions(hookEveryRounds, hookPoolSize);
    const auto callback = bridge.progressCallback();
    const auto checked = checkedConfig(config);
    return runTrace(bridge, [&] {
        return ft::traceFiberExtrapolation(
            *field.field, toVec(start), toVec(direction), distanceVoxels, checked,
            samplerPtr(normals), callback, options);
    });
}

ft::FiberTraceWholeFiberResult traceWholeFiberMetric(
    const PredictionFieldHandle& field,
    const ft::FiberInput& fiber,
    double workingToBaseScale,
    double errorThresholdBaseVoxels,
    const ft::FiberTraceConfig& config,
    const NormalSamplerHandle* normals,
    std::optional<double> voxelSizeUm,
    nb::object hook,
    int hookEveryRounds,
    int hookPoolSize,
    nb::object progress)
{
    CallbackBridge bridge{std::move(hook), std::move(progress), {}};
    ft::FiberTraceWholeFiberMetricRequest request;
    request.fiber = fiber;
    request.workingToBaseScale = workingToBaseScale;
    request.errorThresholdBaseVoxels = errorThresholdBaseVoxels;
    request.voxelSizeUm = voxelSizeUm;
    request.config = checkedConfig(config);
    request.beamHook = bridge.hookOptions(hookEveryRounds, hookPoolSize);
    const auto callback = bridge.wholeFiberProgressCallback();
    return runTrace(bridge, [&] {
        return ft::traceWholeFiberMetric(*field.field, request, samplerPtr(normals), callback);
    });
}

std::vector<ft::FiberTraceTargetPlane> targetLocalPlanes(
    const PredictionFieldHandle& field,
    const PointsIn& referenceLine,
    size_t targetIndex,
    size_t sourceIndex,
    Vec3 targetPoint)
{
    const auto line = toPoints(referenceLine, "reference_line");
    nb::gil_scoped_release release;
    return ft::targetLocalPlanes(*field.field, line, targetIndex, sourceIndex, toVec(targetPoint));
}

ft::FiberInput makeFiberInput(
    const PointsIn& linePoints,
    const PointsIn& controlPoints,
    const IndicesIn& controlPointLineIndices,
    const std::string& path)
{
    ft::FiberInput fiber;
    fiber.path = path;
    fiber.linePointsXyzBase = toPoints(linePoints, "line_points");
    fiber.controlPointsXyzBase = toPoints(controlPoints, "control_points");
    if (controlPointLineIndices.shape(0) != fiber.controlPointsXyzBase.size()) {
        throw std::invalid_argument(
            "control_point_line_indices must have one entry per control point");
    }
    for (size_t index = 0; index < controlPointLineIndices.shape(0); ++index) {
        const int64_t value = controlPointLineIndices.data()[index];
        if (value < 0 || static_cast<size_t>(value) >= fiber.linePointsXyzBase.size())
            throw std::invalid_argument("control_point_line_indices out of range");
        fiber.controlPointLineIndices.push_back(static_cast<size_t>(value));
    }
    return fiber;
}

nb::object indexArray(const std::vector<size_t>& values)
{
    std::vector<int64_t> out(values.begin(), values.end());
    return ownedArray(std::move(out), {values.size()});
}

} // namespace

NB_MODULE(fiber_trace, m)
{
    m.doc() = "Python bindings for the Volume Cartographer native fiber beam tracer";

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
            new (self) ft::FiberTraceConfig();
            for (auto item : kwargs)
                setConfigField(*self, nb::cast<std::string>(item.first), item.second);
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

    nb::class_<ft::FiberTraceWholeFiberSegmentResult>(m, "WholeFiberSegmentResult")
        .def_ro("start_control_point_index",
                &ft::FiberTraceWholeFiberSegmentResult::startControlPointIndex)
        .def_ro("target_control_point_index",
                &ft::FiberTraceWholeFiberSegmentResult::targetControlPointIndex)
        .def_ro("trace", &ft::FiberTraceWholeFiberSegmentResult::trace)
        .def_ro("success", &ft::FiberTraceWholeFiberSegmentResult::success)
        .def_ro("restart", &ft::FiberTraceWholeFiberSegmentResult::restart)
        .def_ro("reason", &ft::FiberTraceWholeFiberSegmentResult::reason)
        .def_ro("in_plane_error_trace_voxels",
                &ft::FiberTraceWholeFiberSegmentResult::inPlaneErrorTraceVoxels)
        .def_ro("in_plane_error_base_voxels",
                &ft::FiberTraceWholeFiberSegmentResult::inPlaneErrorBaseVoxels)
        .def_ro("reference_arc_distance_voxels",
                &ft::FiberTraceWholeFiberSegmentResult::referenceArcDistanceVoxels);

    nb::class_<ft::FiberTraceWholeFiberResult>(m, "WholeFiberResult")
        .def_ro("segments", &ft::FiberTraceWholeFiberResult::segments)
        .def_prop_ro("stitched_trace",
                     [](const ft::FiberTraceWholeFiberResult& self) {
                         return pointsArray(self.stitchedTrace);
                     })
        .def_ro("restart_count", &ft::FiberTraceWholeFiberResult::restartCount)
        .def_ro("lookahead_retry_count", &ft::FiberTraceWholeFiberResult::lookaheadRetryCount)
        .def_ro("lookahead_retry_recovered_count",
                &ft::FiberTraceWholeFiberResult::lookaheadRetryRecoveredCount)
        .def_ro("segment_count", &ft::FiberTraceWholeFiberResult::segmentCount)
        .def_ro("restarts_per_kvx", &ft::FiberTraceWholeFiberResult::restartsPerKvx)
        .def_ro("reference_length_voxels", &ft::FiberTraceWholeFiberResult::referenceLengthVoxels)
        .def_ro("reference_length_meters", &ft::FiberTraceWholeFiberResult::referenceLengthMeters)
        .def_ro("restarts_per_meter", &ft::FiberTraceWholeFiberResult::restartsPerMeter);

    nb::class_<ft::FiberInput>(m, "FiberInput")
        .def("__init__", [](ft::FiberInput* self, const PointsIn& linePoints, const PointsIn& controlPoints,
                            const IndicesIn& indices, const std::string& path) {
            new (self) ft::FiberInput(makeFiberInput(linePoints, controlPoints, indices, path));
        }, "line_points"_a, "control_points"_a, "control_point_line_indices"_a, "path"_a = "")
        .def_prop_ro("path", [](const ft::FiberInput& self) { return self.path.string(); })
        .def_prop_ro("line_points",
                     [](const ft::FiberInput& self) { return pointsArray(self.linePointsXyzBase); })
        .def_prop_ro("control_points",
                     [](const ft::FiberInput& self) { return pointsArray(self.controlPointsXyzBase); })
        .def_prop_ro("control_point_line_indices",
                     [](const ft::FiberInput& self) { return indexArray(self.controlPointLineIndices); });

    m.def("load_fiber_json",
          [](const std::string& path) {
              nb::gil_scoped_release release;
              return ft::loadFiberJson(path);
          },
          "path"_a,
          "Load a vc3d_fiber JSON (base voxels) as a FiberInput for trace_whole_fiber_metric.");

    nb::class_<HookPool>(m, "HookPool")
        .def_ro("round", &HookPool::round)
        .def_ro("step", &HookPool::step)
        .def_ro("max_steps", &HookPool::maxSteps)
        .def_ro("phase", &HookPool::phase)
        .def_prop_ro("start", [](const HookPool& self) { return vecArray(self.start); })
        .def_prop_ro("target", [](const HookPool& self) { return vecArray(self.target); })
        .def_prop_ro("path_points", [](const HookPool& self) { return pointsArray(self.pathPoints); })
        .def_prop_ro("path_offsets",
                     [](const HookPool& self) {
                         return ownedArray(std::vector<int64_t>(self.pathOffsets), {self.pathOffsets.size()});
                     })
        .def("paths",
             [](const HookPool& self) {
                 nb::list out;
                 for (size_t index = 0; index < self.size(); ++index) {
                     const auto begin = static_cast<size_t>(self.pathOffsets[index]);
                     const auto end = static_cast<size_t>(self.pathOffsets[index + 1]);
                     out.append(pointsArray(std::vector<cv::Vec3d>(
                         self.pathPoints.begin() + static_cast<std::ptrdiff_t>(begin),
                         self.pathPoints.begin() + static_cast<std::ptrdiff_t>(end))));
                 }
                 return out;
             },
             "One (n_i, 3) float64 array per pool candidate, trace start first.")
        .def_prop_ro("losses",
                     [](const HookPool& self) {
                         return ownedArray(std::vector<float>(self.losses), {self.losses.size()});
                     })
        .def_prop_ro("depth",
                     [](const HookPool& self) {
                         return ownedArray(std::vector<int32_t>(self.depth), {self.depth.size()});
                     })
        .def_prop_ro("traced_length",
                     [](const HookPool& self) {
                         return ownedArray(std::vector<double>(self.tracedLength), {self.tracedLength.size()});
                     })
        .def_prop_ro("reached",
                     [](const HookPool& self) {
                         std::vector<uint8_t> bytes(self.reached.begin(), self.reached.end());
                         return ownedArray(std::move(bytes), {self.reached.size()}).attr("astype")("bool");
                     })
        .def_prop_ro("previous_step_direction",
                     [](const HookPool& self) { return pointsArray(self.previousStepDirection); })
        .def_prop_ro("current_sample_direction",
                     [](const HookPool& self) { return pointsArray(self.currentSampleDirection); })
        .def_prop_ro("history_direction",
                     [](const HookPool& self) { return pointsArray(self.historyDirection); })
        .def("__len__", &HookPool::size);

    m.def("trace_one_way", &traceOneWay,
          "field"_a, "start"_a, "target"_a, "initial_direction"_a, "target_planes"_a,
          "accept_threshold_voxels"_a = nb::none(), "budget_span_voxels"_a = 0.0,
          "config"_a = ft::FiberTraceConfig{}, "normal_sampler"_a = nb::none(),
          "snap_trace_to_selected_crossing"_a = true, "hook"_a = nb::none(),
          "hook_every_rounds"_a = 1, "hook_pool_size"_a = 32, "progress"_a = nb::none(),
          "One-way beam trace from start toward target until every target plane is crossed.");
    m.def("trace_segment", &traceSegment,
          "field"_a, "reference_line"_a, "start_index"_a, "target_index"_a,
          "config"_a = ft::FiberTraceConfig{}, "normal_sampler"_a = nb::none(),
          "hook"_a = nb::none(), "hook_every_rounds"_a = 1, "hook_pool_size"_a = 32,
          "progress"_a = nb::none(),
          "Bidirectional control-point-to-control-point trace with meeting fusion, as VC3D does.");
    m.def("trace_extrapolation", &traceExtrapolation,
          "field"_a, "start"_a, "direction"_a, "distance_voxels"_a,
          "config"_a = ft::FiberTraceConfig{}, "normal_sampler"_a = nb::none(),
          "hook"_a = nb::none(), "hook_every_rounds"_a = 1, "hook_pool_size"_a = 32,
          "progress"_a = nb::none(),
          "Open-ended trace for a fixed distance with no target planes.");
    m.def("trace_whole_fiber_metric", &traceWholeFiberMetric,
          "field"_a, "fiber"_a, "working_to_base_scale"_a, "error_threshold_base_voxels"_a = 20.0,
          "config"_a = ft::FiberTraceConfig{}, "normal_sampler"_a = nb::none(),
          "voxel_size_um"_a = nb::none(), "hook"_a = nb::none(), "hook_every_rounds"_a = 1,
          "hook_pool_size"_a = 32, "progress"_a = nb::none(),
          "Chain one-way traces control point to control point, restarting at each miss.");
    m.def("target_local_planes", &targetLocalPlanes,
          "field"_a, "reference_line"_a, "target_index"_a, "source_index"_a, "target_point"_a,
          "Target planes through target_point for a trace toward reference_line[target_index].");
    m.def("reference_tangent_toward",
          [](const PointsIn& line, size_t startIndex, size_t targetIndex) {
              const auto points = toPoints(line, "line");
              return fromVec(ft::referenceTangentToward(points, startIndex, targetIndex));
          },
          "line"_a, "start_index"_a, "target_index"_a);
    m.def("effective_endpoint_accept_threshold_base_voxels",
          &ft::effectiveEndpointAcceptThresholdBaseVoxels,
          "config"_a, "span_length_base_voxels"_a);
    m.def("_version_tag", [] { return std::string("fiber_trace_bindings_v1"); });
}
