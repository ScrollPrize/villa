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
#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberLiveBridge.hpp"
#include "vc/fiber_tracer/FiberCaseQueue.hpp"
#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include <atomic>
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
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

struct LiveBridgeHandle {
    std::shared_ptr<ft::FiberLiveBridge> bridge;
    LiveBridgeHandle(int port, const std::string& token, int timeout, bool heartbeat)
        : bridge(std::make_shared<ft::FiberLiveBridge>(nlohmann::json{
              {"port",port},{"token",token},{"timeout_seconds",timeout},{"heartbeat_wait",heartbeat}})) {}
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
    const std::optional<std::string>& remoteCacheRoot,
    bool enableWidth)
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
        ft::FiberPredictionFieldOptions options;
        options.maxCachedBytes=cacheBytes;
        options.enableWidth=enableWidth;
        handle->field = std::make_unique<ft::FiberPredictionField>(*handle->dataset, options);
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
        field<double, &ft::FiberTraceConfig::replayPresenceCutoff>("replay_presence_cutoff"),
        field<bool, &ft::FiberTraceConfig::replayPresenceModulated>("replay_presence_modulated"),
        field<double, &ft::FiberTraceConfig::widthRelativeTolerance>("width_relative_tolerance"),
        field<bool, &ft::FiberTraceConfig::widthCosineWeight>("width_cosine_weight"),
        field<double, &ft::FiberTraceConfig::widthReferenceMean>("width_reference_mean"),
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

struct ChunkPathsHandle {
    std::shared_ptr<const ft::FiberChunkPaths> paths;
};

// Python keep_alive on the constructor retains these dataset handles. Copy
// settings before releasing the GIL: later Python mutations cannot race reads
// or change the identity of an in-flight cache build.
struct ChunkCacheHandle {
    const PredictionFieldHandle& predictions;
    const NormalSamplerHandle& normals;
    const ft::FiberTraceConfig trace;
    ft::FiberChunkTraceCache cache;
    ChunkCacheHandle(const PredictionFieldHandle& p,const NormalSamplerHandle& n,
        const ft::FiberTraceConfig& t,const ft::FiberChunkTraceConfig& c,
        const std::string& output,const std::string& identity)
        : predictions(p),normals(n),trace(checkedConfig(p,t,&n)),cache(c,output,identity) {}
    ChunkPathsHandle get(const std::array<int,3>& key) {
        nb::gil_scoped_release release;
        return {cache.get(key,*predictions.field,trace,*normals.sampler,predictions.scales.traceToBaseScale)};
    }
    std::vector<ChunkPathsHandle> getMany(const std::vector<std::array<int,3>>& keys,
                                         size_t workers,size_t contexts) {
        if(trace.parallelThreads!=1)
            throw std::invalid_argument("get_many requires parallel_threads=1; the case queue owns parallelism");
        std::vector<ChunkPathsHandle> result(keys.size());
        nb::gil_scoped_release release;
        std::atomic<size_t> next{0};
        ft::FiberCaseQueue queue(workers,keys.size(),[&] {
            const size_t index=next.fetch_add(1);
            result[index].paths=cache.get(keys[index],*predictions.field,trace,*normals.sampler,
                predictions.scales.traceToBaseScale,ft::FiberCaseQueue::parallelFor,
                [](const auto& ready) { ft::FiberCaseQueue::await(ready); });
        },contexts);
        queue.join();
        return result;
    }
};

nb::dict chunkRoute(ChunkCacheHandle& self,Vec3 center,Vec3 direction,
    const PointsIn& metric,double maximumDistance,int maximumSteps)
{
    const auto rows=toPoints(metric,"metric");
    if(rows.size()!=3) throw std::invalid_argument("metric must have shape (3, 3)");
    ft::FiberChunkInitialRegion region;region.center=toVec(center);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) region.metric(i,j)=rows[i][j];
    std::vector<cv::Vec3d> points;
    std::vector<ft::FiberChunkRouteJoin> joins;
    std::string reason;
    {
        nb::gil_scoped_release release;
        const ft::FiberChunkProvider get=[&](const auto& key){
            return self.cache.get(key,*self.predictions.field,self.trace,*self.normals.sampler,
                self.predictions.scales.traceToBaseScale);
        };
        reason=ft::traceFiberChunkRoute(get,self.cache.config(),*self.predictions.field,self.trace,
            self.predictions.scales.traceToBaseScale,region,toVec(direction),maximumDistance,maximumSteps,
            [&](const auto& p,double,int){points.push_back(p);return true;},
            [&](const auto& j){joins.push_back(j);});
    }
    nb::dict out;out["points_base"]=pointsArray(points);out["reason"]=reason;out["joins"]=nb::cast(joins);
    return out;
}

ft::FiberReplayTraceResult chunkReplay(ChunkCacheHandle& self,const PointsIn& points,
    const NormalSamplerHandle& evaluationNormals,const ft::FiberChunkReplayConfig& config,
    double begin,const std::optional<double>& end,double reverseInterval)
{
    const auto reference=ft::makePolylineArcGeometry(toPoints(points,"reference_base"));
    const auto c=config;
    const auto policy=ft::fiberChunkReplayConsistency(*self.predictions.field,*self.normals.sampler,
                                                    self.trace,c,reverseInterval);
    nb::gil_scoped_release release;
    return ft::traceFiberChunkReferenceReplay(self.cache,*self.predictions.field,self.trace,
        self.predictions.scales.traceToBaseScale,*self.normals.sampler,reference,begin,end.value_or(reference.length()),
        *evaluationNormals.sampler,evaluationNormals.workingToBaseScale,c,{},{},{},{},policy);
}

} // namespace

NB_MODULE(fiber_trace, m)
{
    m.def("reference_plan_json", [](const std::string& text) {
        return ft::vc3dEvaluationReferencePlan(
            ft::parseVc3dFiberJson(nlohmann::json::parse(text), "reference"), "reference").dump();
    }, "annotation_json"_a);
    m.def("beam_replay", [](const PredictionFieldHandle& field, const NormalSamplerHandle& normals,
        const NormalSamplerHandle& evaluationNormals, const ft::FiberTraceConfig& config,
        const ft::FiberEvaluationCase& input, const ft::FiberChunkReplayConfig& replay, double interval) {
        const auto trace = checkedConfig(field, config, &normals);
        const auto c = input;
        const auto r = replay;
        nb::gil_scoped_release release;
        return ft::replayFiberBeamCase(*field.field, trace, field.scales.traceToBaseScale,
            *normals.sampler, *evaluationNormals.sampler, evaluationNormals.workingToBaseScale, c, r, interval);
    }, "field"_a, "normals"_a, "evaluation_normals"_a, "trace"_a, "case"_a, "replay"_a,
       "reverse_interval_base"_a=0.);
    m.def("beam_replay_many", [](const PredictionFieldHandle& field, const NormalSamplerHandle& normals,
        const NormalSamplerHandle& evaluationNormals, const ft::FiberTraceConfig& config,
        const std::vector<ft::FiberEvaluationCase>& input, const ft::FiberChunkReplayConfig& replay,
        ft::FiberEvaluationControl& control, size_t workers, size_t contexts, double interval) {
        const auto trace = checkedConfig(field, config, &normals);
        const auto cases = input;
        const auto r = replay;
        nb::gil_scoped_release release;
        return ft::evaluateFiberBeamCases(*field.field, trace, field.scales.traceToBaseScale,
            *normals.sampler, *evaluationNormals.sampler, evaluationNormals.workingToBaseScale,
            cases, r, control, workers, contexts, interval);
    }, "field"_a, "normals"_a, "evaluation_normals"_a, "trace"_a, "cases"_a, "replay"_a,
       "control"_a, "workers"_a=4, "contexts_per_worker"_a=8, "reverse_interval_base"_a=0.);
    m.doc() = "Python bindings for the Volume Cartographer native fiber beam tracer";

    nb::class_<ft::FiberChunkTraceConfig>(m,"ChunkConfig")
        .def(nb::init<>())
        .def_rw("core_base",&ft::FiberChunkTraceConfig::coreBase)
        .def_rw("halo_base",&ft::FiberChunkTraceConfig::haloBase)
        .def_rw("grid_base",&ft::FiberChunkTraceConfig::gridBase)
        .def_rw("seed_presence",&ft::FiberChunkTraceConfig::seedPresence)
        .def_rw("nms_base",&ft::FiberChunkTraceConfig::nmsBase)
        .def_rw("tube_half_base",&ft::FiberChunkTraceConfig::tubeHalfBase)
        .def_rw("tube_step_base",&ft::FiberChunkTraceConfig::tubeStepBase)
        .def_rw("tube_presence",&ft::FiberChunkTraceConfig::tubePresence)
        .def_rw("tube_turn_degrees",&ft::FiberChunkTraceConfig::tubeTurnDegrees)
        .def_rw("tube_total_degrees",&ft::FiberChunkTraceConfig::tubeTotalDegrees)
        .def_rw("rim_drop",&ft::FiberChunkTraceConfig::rimDrop)
        .def_rw("rim_normal_base",&ft::FiberChunkTraceConfig::rimNormalBase)
        .def_rw("rim_tangent_base",&ft::FiberChunkTraceConfig::rimTangentBase)
        .def_rw("join_distance_base",&ft::FiberChunkTraceConfig::joinDistanceBase)
        .def_rw("join_angle_degrees",&ft::FiberChunkTraceConfig::joinAngleDegrees)
        .def_rw("join_minimum_base",&ft::FiberChunkTraceConfig::joinMinimumBase)
        .def_rw("join_width_weight",&ft::FiberChunkTraceConfig::joinWidthWeight)
        .def_rw("pin_sampling_region",&ft::FiberChunkTraceConfig::pinSamplingRegion)
        .def_rw("max_grid_samples",&ft::FiberChunkTraceConfig::maxGridSamples);
    nb::class_<ft::FiberChunkReplayConfig>(m,"ChunkReplayConfig")
        .def(nb::init<>())
        .def_rw("error_threshold_base",&ft::FiberChunkReplayConfig::errorThresholdBase)
        .def_rw("excursion_grace_base",&ft::FiberChunkReplayConfig::excursionGraceBase)
        .def_rw("tangential_threshold_factor",&ft::FiberChunkReplayConfig::tangentialThresholdFactor)
        .def_rw("match_refine_steps",&ft::FiberChunkReplayConfig::matchRefineSteps);
    nb::class_<ChunkPathsHandle>(m,"ChunkPaths")
        .def_prop_ro("key",[](const ChunkPathsHandle& s){return s.paths->key;})
        .def_prop_ro("lo_base",[](const ChunkPathsHandle& s){return fromVec(s.paths->lo);})
        .def_prop_ro("hi_base",[](const ChunkPathsHandle& s){return fromVec(s.paths->hi);})
        .def_prop_ro("maxima",[](const ChunkPathsHandle& s){return s.paths->maxima;})
        .def_prop_ro("seeds",[](const ChunkPathsHandle& s){return s.paths->seeds;})
        .def_prop_ro("end_reasons",[](const ChunkPathsHandle& s){return s.paths->endReasons;})
        .def_prop_ro("lines_base",[](const ChunkPathsHandle& s){
            nb::list lines;for(const auto& line:s.paths->lines) lines.append(pointsArray(line));return lines;
        });
    nb::class_<ft::FiberChunkMatch>(m,"ChunkMatch")
        .def_ro("line",&ft::FiberChunkMatch::line).def_ro("segment",&ft::FiberChunkMatch::segment)
        .def_ro("reverse",&ft::FiberChunkMatch::reverse).def_ro("valid",&ft::FiberChunkMatch::valid)
        .def_prop_ro("point_base",[](const ft::FiberChunkMatch& s){return fromVec(s.point);})
        .def_ro("arc_base",&ft::FiberChunkMatch::arc).def_ro("overlap_base",&ft::FiberChunkMatch::overlap)
        .def_ro("error_base",&ft::FiberChunkMatch::error).def_ro("angle_error",&ft::FiberChunkMatch::angleError)
        .def_ro("width_compatibility",&ft::FiberChunkMatch::widthCompatibility)
        .def_ro("weighted_overlap_base",&ft::FiberChunkMatch::weightedOverlap);
    nb::class_<ft::FiberChunkRouteJoin>(m,"ChunkRouteJoin")
        .def_ro("key",&ft::FiberChunkRouteJoin::key).def_ro("match",&ft::FiberChunkRouteJoin::match)
        .def_ro("initial",&ft::FiberChunkRouteJoin::initial)
        .def_ro("splice_distance_base",&ft::FiberChunkRouteJoin::spliceDistanceBase);
    nb::class_<ft::FiberReplayThresholdMeasurement>(m,"ReplayThresholdMeasurement")
        .def_ro("euclidean_error_base",&ft::FiberReplayThresholdMeasurement::euclideanErrorBaseVoxels)
        .def_ro("normal_error_base",&ft::FiberReplayThresholdMeasurement::normalErrorBaseVoxels)
        .def_ro("tangential_error_base",&ft::FiberReplayThresholdMeasurement::tangentialErrorBaseVoxels)
        .def_ro("threshold_error_base",&ft::FiberReplayThresholdMeasurement::thresholdErrorBaseVoxels)
        .def_ro("threshold_error_ratio",&ft::FiberReplayThresholdMeasurement::thresholdErrorRatio)
        .def_ro("local_normal_valid",&ft::FiberReplayThresholdMeasurement::localNormalValid)
        .def_ro("tangential_factor",&ft::FiberReplayThresholdMeasurement::tangentialFactor);
    nb::class_<ft::FiberReplayMatch>(m,"ReplayMatch")
        .def_ro("trace_point_index",&ft::FiberReplayMatch::tracePointIndex)
        .def_ro("predicted_reference_arc_base",&ft::FiberReplayMatch::predictedReferenceArcBase)
        .def_ro("matched_reference_arc_base",&ft::FiberReplayMatch::matchedReferenceArcBase)
        .def_prop_ro("matched_reference_point_base",[](const ft::FiberReplayMatch& s){return fromVec(s.matchedReferencePointBase);})
        .def_ro("search_begin_arc_base",&ft::FiberReplayMatch::searchBeginArcBase)
        .def_ro("search_end_arc_base",&ft::FiberReplayMatch::searchEndArcBase)
        .def_ro("threshold_measurement",&ft::FiberReplayMatch::thresholdMeasurement);
    nb::class_<ft::FiberReplayFailure>(m,"ReplayFailure")
        .def_ro("index",&ft::FiberReplayFailure::index).def_ro("segment_index",&ft::FiberReplayFailure::segmentIndex)
        .def_ro("reason",&ft::FiberReplayFailure::reason)
        .def_ro("reference_arc_base",&ft::FiberReplayFailure::referenceArcBase)
        .def_ro("reference_arc_fraction",&ft::FiberReplayFailure::referenceArcFraction)
        .def_prop_ro("reference_point_base",[](const ft::FiberReplayFailure& s){return fromVec(s.referencePointBase);})
        .def_prop_ro("evaluator_point_base",[](const ft::FiberReplayFailure& s)->std::optional<Vec3>{
            if(s.evaluatorPointBase) return fromVec(*s.evaluatorPointBase);return std::nullopt;
        })
        .def_ro("segment_point_index",&ft::FiberReplayFailure::segmentPointIndex)
        .def_ro("threshold_measurement",&ft::FiberReplayFailure::thresholdMeasurement);
    nb::class_<ft::FiberReplayTraceSegment>(m,"ReplaySegment")
        .def_ro("start_reference_arc_base",&ft::FiberReplayTraceSegment::startReferenceArcBase)
        .def_ro("end_reference_arc_base",&ft::FiberReplayTraceSegment::endReferenceArcBase)
        .def_ro("termination_reason",&ft::FiberReplayTraceSegment::terminationReason)
        .def_prop_ro("points_base",[](const ft::FiberReplayTraceSegment& s){return pointsArray(s.tracePointsBase);})
        .def_ro("cumulative_losses",&ft::FiberReplayTraceSegment::cumulativeLosses)
        .def_ro("matches",&ft::FiberReplayTraceSegment::matches)
        .def_ro("filtered_distance_failure",&ft::FiberReplayTraceSegment::filteredDistanceFailure);
    nb::class_<ft::FiberReplayTraceResult>(m,"ReplayResult")
        .def_ro("filtered_stops",&ft::FiberReplayTraceResult::filteredStops)
        .def_ro("reverse_checks",&ft::FiberReplayTraceResult::reverseChecks)
        .def_ro("tangential_threshold_factor",&ft::FiberReplayTraceResult::tangentialThresholdFactor)
        .def_ro("reference_begin_arc_base",&ft::FiberReplayTraceResult::referenceBeginArcBase)
        .def_ro("reference_end_arc_base",&ft::FiberReplayTraceResult::referenceEndArcBase)
        .def_ro("completed_reference_arc_base",&ft::FiberReplayTraceResult::completedReferenceArcBase)
        .def_ro("seed_search_skipped_base",&ft::FiberReplayTraceResult::seedSearchSkippedBase)
        .def_ro("segments",&ft::FiberReplayTraceResult::segments)
        .def_ro("failures",&ft::FiberReplayTraceResult::failures);

    m.def("decode_directions", &decodeDirections, "nx"_a.noconvert(), "ny"_a.noconvert(),
          "Decode flat uint8 nx/ny arrays to unit xyz axes; missing bytes give zero axes.");

    nb::class_<LiveBridgeHandle>(m, "LiveBridge")
        .def(nb::init<int,const std::string&,int,bool>(),
             "port"_a,"token"_a,"timeout_seconds"_a=600,"heartbeat_wait"_a=false,
             nb::call_guard<nb::gil_scoped_release>())
        .def("cancel", [](LiveBridgeHandle& h,const std::string& reason) { h.bridge->cancel(reason); },
             "reason"_a="cancelled", nb::call_guard<nb::gil_scoped_release>())
        .def("check_active", [](LiveBridgeHandle& h) { h.bridge->checkActive(); },
             nb::call_guard<nb::gil_scoped_release>())
        .def("wait", [](LiveBridgeHandle& h,const std::string& source,size_t caseIndex,
                       double remaining,const std::vector<std::string>& paths,
                       const std::vector<std::array<uint32_t,3>>& indices) {
            if(paths.size()!=indices.size()) throw std::invalid_argument("paths and indices sizes differ");
            std::vector<ft::FiberPredictionChunkDependency> chunks;
            for(size_t i=0;i<paths.size();++i) chunks.push_back({paths[i],indices[i]});
            nb::gil_scoped_release release;
            h.bridge->await(source,caseIndex,remaining,chunks);
        }, "source"_a,"case_index"_a,"remaining_base_voxels"_a,"array_paths"_a,"indices_zyx"_a);

    nb::class_<PredictionFieldHandle>(m, "PredictionField")
        .def("set_live_bridge", [](PredictionFieldHandle& field,const LiveBridgeHandle& h,
                                   const std::string& source,size_t caseIndex,double remaining,size_t capacity) {
            if(source.empty() || !std::isfinite(remaining) || remaining<0)
                throw std::invalid_argument("invalid live source or remaining distance");
            auto readiness=std::make_shared<ft::FiberCaseReadiness>(capacity);
            field.field->setReadinessGate([bridge=h.bridge,readiness,source,caseIndex,remaining](const auto& chunks) {
                readiness->await(chunks,[&](const auto& missing) {
                    auto wait=[&] { bridge->await(source,caseIndex,remaining,missing); };
                    if(auto* permit=ft::FiberConstructionCpuPermit::current()) permit->suspend(wait);
                    else wait();
                },[&] { bridge->checkActive(); });
            });
        }, "bridge"_a,"source"_a,"case_index"_a=0,"remaining_base_voxels"_a=0,
           "proof_cache_capacity"_a=4096,
           "Attach before sampling. The field retains this session; no Python callbacks run during native work.")
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
          "remote_cache_root"_a = nb::none(), "enable_width"_a = false,
          "Open a Lasagna fiber-prediction dataset (presence/nx/ny) as a beam tracer field. "
          "Trace voxels are prediction voxels divided by 2**scaledown_power.");
    m.def("open_normal_sampler", &openNormalSampler,
          "location"_a, "working_to_base_scale"_a, "cache_bytes"_a = size_t{512} << 20,
          "remote_cache_root"_a = nb::none(),
          "Open a Lasagna normal dataset (nx/ny/grad_mag) at the tracer's working scale.");

    nb::class_<ChunkCacheHandle>(m,"ChunkCache")
        .def(nb::init<const PredictionFieldHandle&,const NormalSamplerHandle&,const ft::FiberTraceConfig&,
             const ft::FiberChunkTraceConfig&,const std::string&,const std::string&>(),
             "field"_a,"normal_sampler"_a,"trace_config"_a,"chunk_config"_a,"output_directory"_a,"source_identity"_a,
             nb::keep_alive<1,2>(),nb::keep_alive<1,3>())
        .def("get",&ChunkCacheHandle::get,"key_xyz"_a)
        .def("get_many",&ChunkCacheHandle::getMany,"keys_xyz"_a,"workers"_a=4,"contexts_per_worker"_a=8,
             "Bounded native cases; ordered results, cooperative publication and cache waits.")
        .def("replay_many", [](ChunkCacheHandle& self, const std::vector<ft::FiberEvaluationCase>& cases,
                              const NormalSamplerHandle& normals, const ft::FiberChunkReplayConfig& config,
                              ft::FiberEvaluationControl& control, size_t workers, size_t contexts,
                              double reverseInterval) {
            // Copy while holding the GIL, so Python mutations cannot race work.
            const auto copiedCases = cases;
            const auto copiedConfig = config;
            nb::gil_scoped_release release;
            return ft::evaluateFiberChunkCases(self.cache, *self.predictions.field, self.trace,
                self.predictions.scales.traceToBaseScale, *self.normals.sampler, *normals.sampler,
                normals.workingToBaseScale, copiedCases, copiedConfig, control, workers, contexts, reverseInterval);
        }, "cases"_a, "evaluation_normals"_a, "config"_a, "control"_a,
           "workers"_a=4, "contexts_per_worker"_a=8, "reverse_interval_base"_a=0.,
           "Ordered batch replay; poll/cancel control from another thread. Raises after joining on failure.")
        .def("route",&chunkRoute,"center_base"_a,"direction"_a,"metric"_a.noconvert(),
             "maximum_distance_base"_a,"maximum_steps"_a,
             "Return committed suffix points (excluding initial snap), reason and join diagnostics.")
        .def("replay",&chunkReplay,"reference_base"_a.noconvert(),"evaluation_normals"_a,"config"_a,
             "begin_arc_base"_a=0.,"end_arc_base"_a=nb::none(),"reverse_interval_base"_a=0.,
             "Shared reference replay; optional reverse checks use the configured acceptance radii.");

    m.def("case_queue_supported",&ft::FiberCaseQueue::supported);

    nb::class_<ft::FiberEvaluationCase>(m,"EvaluationCase")
        .def("__init__", [](ft::FiberEvaluationCase* self, const PointsIn& points,
                            double begin, const std::optional<double>& end) {
            new (self) ft::FiberEvaluationCase{toPoints(points,"reference_base"), begin, end};
        }, "reference_base"_a.noconvert(), "begin_arc_base"_a=0., "end_arc_base"_a=nb::none());
    nb::class_<ft::FiberEvaluationControl>(m,"EvaluationControl")
        .def(nb::init<>())
        .def("cancel", &ft::FiberEvaluationControl::cancel, "reason"_a="fiber evaluation cancelled",
             nb::call_guard<nb::gil_scoped_release>())
        .def("status", [](const ft::FiberEvaluationControl& self) {
            const auto s = self.status();
            nb::dict out;
            out["total"] = s.total; out["completed"] = s.completed;
            out["started"] = s.started; out["finished"] = s.finished;
            out["cancelled"] = s.cancelled; out["error"] = s.error;
            return out;
        });

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
        .def_rw("replay_presence_cutoff", &ft::FiberTraceConfig::replayPresenceCutoff)
        .def_rw("replay_presence_modulated", &ft::FiberTraceConfig::replayPresenceModulated)
        .def_rw("width_relative_tolerance", &ft::FiberTraceConfig::widthRelativeTolerance)
        .def_rw("width_cosine_weight", &ft::FiberTraceConfig::widthCosineWeight)
        .def_rw("width_reference_mean", &ft::FiberTraceConfig::widthReferenceMean)
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
