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
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <chrono>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/FiberVolumeCoverage.hpp"
#include "vc/fiber_tracer/FiberStreamVolume.hpp"
#include <future>
#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
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

// Synchronous per-call collector. active is accessed only while holding the GIL;
// readers/reusers cannot touch values while native tracing mutates them.
struct TraceProfileHandle {
    ft::FiberTraceProfile value;
    bool active = false;
};
struct TraceProfileUse {
    TraceProfileHandle* handle;
    explicit TraceProfileUse(TraceProfileHandle* p) : handle(p) {
        if (!p) return;
        if (p->active) throw std::runtime_error("trace profile is already in use");
        p->value = {};
        p->active = true;
    }
    ~TraceProfileUse() { if (handle) handle->active = false; }
};

nb::dict profileDict(const TraceProfileHandle& handle) {
    if (handle.active) throw std::runtime_error("trace profile is already in use");
    const auto& p=handle.value; nb::dict out;
#define VC_PROFILE_FIELD(name) out[#name]=nb::cast(p.name)
    VC_PROFILE_FIELD(oneWayCalls); VC_PROFILE_FIELD(generations); VC_PROFILE_FIELD(candidateTasks);
    VC_PROFILE_FIELD(candidateDepth1BatchSizes); VC_PROFILE_FIELD(candidateDepth2BatchSizes);
    VC_PROFILE_FIELD(startSampleSeconds); VC_PROFILE_FIELD(widthSampleSeconds); VC_PROFILE_FIELD(widthSamplePoints);
    VC_PROFILE_FIELD(combinedChannelSampleSeconds);
    VC_PROFILE_FIELD(samplingWindowLoads); VC_PROFILE_FIELD(samplingWindowFallbacks);
    VC_PROFILE_FIELD(samplingWindowPeakBytes); VC_PROFILE_FIELD(samplingWindowLoadSeconds);
    VC_PROFILE_FIELD(samplingWindowPeakTableBytes); VC_PROFILE_FIELD(samplingWindowCopiedBytes);
    VC_PROFILE_FIELD(taskBuildSeconds); VC_PROFILE_FIELD(predictionBatchSeconds);
    VC_PROFILE_FIELD(readinessPlanSeconds); VC_PROFILE_FIELD(readinessPlanPoints); VC_PROFILE_FIELD(readinessPlanDependencies);
    VC_PROFILE_FIELD(predictionPrepareSeconds); VC_PROFILE_FIELD(predictionPrefetchSeconds);
    VC_PROFILE_FIELD(predictionAssignSeconds); VC_PROFILE_FIELD(predictionMaterializeSeconds);
    VC_PROFILE_FIELD(predictionCornerSeconds); VC_PROFILE_FIELD(predictionCornerPrepareSeconds);
    VC_PROFILE_FIELD(predictionCornerLayoutSeconds); VC_PROFILE_FIELD(predictionCornerPinSeconds);
    VC_PROFILE_FIELD(predictionCornerGatherSeconds); VC_PROFILE_FIELD(predictionDecodeSeconds);
    VC_PROFILE_FIELD(normalDecodeSeconds); VC_PROFILE_FIELD(normalBatchSeconds);
    VC_PROFILE_FIELD(normalPrefetchSeconds); VC_PROFILE_FIELD(normalMaterializeSeconds);
    VC_PROFILE_FIELD(candidateScoreSeconds); VC_PROFILE_FIELD(frontierSeconds); VC_PROFILE_FIELD(pruneSeconds);
    VC_PROFILE_FIELD(lookaheadDecisionSeconds); VC_PROFILE_FIELD(lookaheadParentOrderSeconds);
    VC_PROFILE_FIELD(lookaheadFrontierStorageSeconds);
#undef VC_PROFILE_FIELD
    return out;
}

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
        field<int, &ft::FiberTraceConfig::samplingWindowVoxels>("sampling_window_voxels"),
        field<double, &ft::FiberTraceConfig::smoothnessWeight>("smoothness_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessNormalWeight>("smoothness_normal_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessTangentWeight>("smoothness_tangent_weight"),
        field<double, &ft::FiberTraceConfig::smoothnessFreeAngleDegrees>("smoothness_free_angle_degrees"),
        field<double, &ft::FiberTraceConfig::sheetPlaneWeight>("sheet_plane_weight"),
        field<int, &ft::FiberTraceConfig::cumulativeSmoothnessSteps>("cumulative_smoothness_steps"),
        field<double, &ft::FiberTraceConfig::cumulativeSmoothnessTangentWeight>("cumulative_smoothness_tangent_weight"),
        field<double, &ft::FiberTraceConfig::initialFreeAngleDegrees>("initial_free_angle_degrees"),
        field<double, &ft::FiberTraceConfig::maxStepFactor>("max_step_factor"),
        field<double, &ft::FiberTraceConfig::meetingAcceptMaxErrorRatio>("meeting_accept_max_error_ratio"),
        field<double, &ft::FiberTraceConfig::endpointAcceptThresholdBaseVoxels>("endpoint_accept_threshold_base_voxels"),
        field<double, &ft::FiberTraceConfig::traceToBaseScale>("trace_to_base_scale"),
        field<double, &ft::FiberTraceConfig::replayPresenceCutoff>("replay_presence_cutoff"),
        field<bool, &ft::FiberTraceConfig::replayPresenceModulated>("replay_presence_modulated"),
        field<bool, &ft::FiberTraceConfig::replayPresenceIgnoreWidth>("replay_presence_ignore_width"),
        field<double, &ft::FiberTraceConfig::widthHardRelativeTolerance>("width_hard_relative_tolerance"),
        field<double, &ft::FiberTraceConfig::replayReverseIntervalBase>("replay_reverse_interval_base"),
        field<double, &ft::FiberTraceConfig::replayReverseNormalRadiusBase>("replay_reverse_normal_radius_base"),
        field<double, &ft::FiberTraceConfig::replayReverseTangentialRadiusBase>("replay_reverse_tangential_radius_base"),
        field<int, &ft::FiberTraceConfig::replayReverseHistorySteps>("replay_reverse_history_steps"),
        field<int, &ft::FiberTraceConfig::replayReverseDelaySteps>("replay_reverse_delay_steps"),
        field<double, &ft::FiberTraceConfig::replayReverseConvergenceMinBase>("replay_reverse_convergence_min_base"),
        field<double, &ft::FiberTraceConfig::replayReverseConvergenceDistanceBase>("replay_reverse_convergence_distance_base"),
        field<double, &ft::FiberTraceConfig::replayReverseConvergenceAngleDegrees>("replay_reverse_convergence_angle_degrees"),
        field<int, &ft::FiberTraceConfig::replayReverseConvergenceSteps>("replay_reverse_convergence_steps"),
        field<bool, &ft::FiberTraceConfig::replayReverseToStart>("replay_reverse_to_start"),
        field<bool, &ft::FiberTraceConfig::replayReverseAllEndpoints>("replay_reverse_all_endpoints"),
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
    for (const auto& entry : configFields()) {
        // Preserve existing default serialization/cache identity for this opt-in.
        if(std::string_view(entry.key)=="sampling_window_voxels" && config.samplingWindowVoxels==0)
            continue;
        out[entry.key] = entry.get(config);
    }
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

// Native handoff: trace workers sleep on their own condition variable, without
// acquiring the GIL. Only the batched policy coordinator materializes Python.
struct StreamControl {
    struct Pending {
        size_t row;Vec3 point,direction;double width;int step;
        bool answered=false,keep=false;std::condition_variable ready;
    };
    std::mutex mutex;std::condition_variable available;
    bool stopped=false;
    std::deque<std::shared_ptr<Pending>> events;
    std::unordered_map<size_t,std::shared_ptr<Pending>> pending;
    bool request(size_t row,Vec3 p,Vec3 d,double width,int step) {
        auto e=std::make_shared<Pending>();e->row=row;e->point=p;e->direction=d;e->width=width;e->step=step;
        std::unique_lock lock(mutex);
        if(stopped)return false;
        if(pending.count(row))throw std::logic_error("overlapping streaming requests for one row");
        pending[row]=e;events.push_back(e);available.notify_one();
        e->ready.wait(lock,[&]{return stopped||e->answered;});
        pending.erase(row);
        return !stopped&&e->keep;
    }
    std::vector<std::shared_ptr<Pending>> poll(int milliseconds) {
        std::unique_lock lock(mutex);
        available.wait_for(lock,std::chrono::milliseconds(milliseconds),[&]{return stopped||!events.empty();});
        // Brief bounded coalescing avoids a separate CT/KD query for each head.
        // Never wait for all traces: slow/I/O-blocked rows cannot stall others.
        if(!events.empty()&&events.size()<16&&!stopped)
            available.wait_for(lock,std::chrono::microseconds(500),[&]{return stopped||events.size()>=16;});
        std::vector<std::shared_ptr<Pending>> out(events.begin(),events.end());events.clear();return out;
    }
    void respond(size_t row,bool keep) {
        std::lock_guard lock(mutex);
        const auto i=pending.find(row);
        if(i==pending.end())throw std::logic_error("unknown streaming response");
        i->second->keep=keep;i->second->answered=true;i->second->ready.notify_one();
    }
    void cancel() {
        std::lock_guard lock(mutex);stopped=true;
        for(auto& [id,e]:pending)e->ready.notify_one();
        available.notify_all();
    }
};

// Continuous native trace; one handoff per neural-equivalent decision,
// never per candidate/sample. Callback may wait on a batched coordinator.
nb::dict traceStream(const PredictionFieldHandle& field, const NormalSamplerHandle& normals,
    Vec3 start, Vec3 direction, double distance, const ft::FiberTraceConfig& config,
    int decisionSteps, nb::object observe, double initialWidth, size_t row)
{
    if(decisionSteps<1)throw std::invalid_argument("decision_steps must be positive");
    auto checked=checkedConfig(field,config,&normals);
    if(!std::isfinite(initialWidth)||initialWidth<=0)
        throw std::invalid_argument("initial width must be positive");
    checked.initialWidthMean=initialWidth;
    checked.preserveInitialDirection=true;
    auto* control=nb::isinstance<StreamControl>(observe)?nb::cast<StreamControl*>(observe):nullptr;
    double waitSeconds=0;
    double width=checked.initialWidthMean;
    std::vector<double> widths,losses;
    std::vector<cv::Vec3d> observed;
    observed.push_back(toVec(start));widths.push_back(width);losses.push_back(0);
    ft::FiberTraceOneWayResult result;
    {
        nb::gil_scoped_release release;
        const auto submit=[&](Vec3 p,Vec3 d,double w,int step) {
            const auto began=std::chrono::steady_clock::now();
            bool keep;
            if(control)keep=control->request(row,p,d,w,step);
            else {
                nb::gil_scoped_acquire acquire;
                keep=nb::cast<bool>(observe(p,d,w,step));
            }
            waitSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            return keep;
        };
        if(control&&!submit(start,direction,width,0)) {
            result.points={toVec(start)};result.reason="observer_stop";
        } else result=ft::traceFiberExtrapolation(*field.field,toVec(start),toVec(direction),
            distance,checked,samplerPtr(&normals),{},false,
            [&](const cv::Vec3d& p,double loss,int step) {
                observed.push_back(p);widths.push_back(width);losses.push_back(loss);
                if(step%decisionSteps)return true;
                auto d=p-observed[observed.size()-2];
                const auto n=cv::norm(d);if(n>0)d/=n;
                return submit(fromVec(p),fromVec(d),width,step);
            },{},&width);
    }
    // Clipping can change the last returned point; observations cover all
    // irrevocable nodes. Store the returned geometry with matching width history.
    if(result.points.size()>widths.size())
        throw std::runtime_error("stream missed committed points");
    widths.resize(result.points.size());losses.resize(result.points.size());
    nb::dict out;out["points"]=pointsArray(result.points);
    const auto count=widths.size();
    out["widths"]=ownedArray(std::move(widths),{count});
    out["losses"]=ownedArray(std::move(losses),{count});
    out["reason"]=result.reason;out["wait_seconds"]=waitSeconds;
    out["width_count"]=std::max<size_t>(1,result.widthSamples);
    out["width_mean"]=result.widthSamples?result.widthMean:initialWidth;
    return out;
}

nb::dict chunkSeeds(const PredictionFieldHandle& field,const NormalSamplerHandle& normals,
    const std::array<int,3>& key,const ft::FiberChunkTraceConfig& input)
{
    auto c=input;
    if(!c.gpuLibrary.empty())throw std::invalid_argument("seed extraction is CPU only");
    const double scale=field.scales.traceToBaseScale;
    ft::FiberChunkPaths bounds;bounds.key=key;
    for(int a=0;a<3;++a) {
        bounds.lo[a]=key[a]*c.coreBase-c.haloBase;
        bounds.hi[a]=(key[a]+1)*c.coreBase+c.haloBase;
    }
    std::vector<ft::FiberChunkSeedFit> seeds;
    {
        nb::gil_scoped_release release;
        auto* normalField=dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(normals.sampler.get());
        if(!normalField)throw std::invalid_argument("native seed extraction requires channel normals");
        ft::detail::FiberChunkTensor tensor(*field.field,*normalField,
            bounds.lo/scale,bounds.hi/scale,true,c.channelHints,c.workerCacheBytes);
        seeds=ft::findFiberChunkSeeds(tensor,tensor,*field.field,bounds,c,scale,true);
    }
    std::vector<cv::Vec3d> points,dirs;std::vector<double> widths,scores;
    for(const auto& s:seeds) {
        // Only the core owns seeds; halo is read support, not extra owners.
        bool core=true;for(int a=0;a<3;++a)
            core=core&&s.point[a]>=key[a]*c.coreBase&&s.point[a]<(key[a]+1)*c.coreBase;
        if(!core)continue;
        points.push_back(s.point);dirs.push_back(s.direction);
        widths.push_back(s.width*scale);scores.push_back(s.score);
    }
    nb::dict out;out["points_base"]=pointsArray(points);out["directions"]=pointsArray(dirs);
    const auto count=widths.size();out["widths_base"]=ownedArray(std::move(widths),{count});
    out["presence"]=ownedArray(std::move(scores),{count});out["maxima"]=bounds.maxima;
    return out;
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
    const std::string directory,identity;
    ft::FiberChunkTraceCache cache;
    ChunkCacheHandle(const PredictionFieldHandle& p,const NormalSamplerHandle& n,
        const ft::FiberTraceConfig& t,const ft::FiberChunkTraceConfig& c,
        const std::string& output,const std::string& identity)
        : predictions(p),normals(n),trace(checkedConfig(p,t,&n)),directory(output),identity(identity),cache(c,output,identity) {}
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
            [&](const auto& j){joins.push_back(j);},nullptr,self.normals.sampler.get());
    }
    nb::dict out;out["points_base"]=pointsArray(points);out["reason"]=reason;out["joins"]=nb::cast(joins);
    return out;
}

ft::FiberReplayTraceResult chunkReplay(ChunkCacheHandle& self,const PointsIn& points,
    const NormalSamplerHandle& evaluationNormals,const ft::FiberChunkReplayConfig& config,
    double begin,const std::optional<double>& end,double reverseInterval,const ft::FiberChunkRoutingRules* rules)
{
    const auto reference=ft::makePolylineArcGeometry(toPoints(points,"reference_base"));
    const auto c=config;
    const auto policy=ft::fiberChunkReplayConsistency(*self.predictions.field,*self.normals.sampler,
                                                    self.trace,c,reverseInterval);
    nb::gil_scoped_release release;
    return ft::traceFiberChunkReferenceReplay(self.cache,*self.predictions.field,self.trace,
        self.predictions.scales.traceToBaseScale,*self.normals.sampler,reference,begin,end.value_or(reference.length()),
        *evaluationNormals.sampler,evaluationNormals.workingToBaseScale,c,{},{},{},{},policy,{},rules);
}

} // namespace

NB_MODULE(fiber_trace, m)
{
    nb::class_<TraceProfileHandle>(m,"TraceProfile")
        .def(nb::init<>())
        .def("to_dict",&profileDict,
             "Snapshot completed-call counters. Read/reuse while active is rejected.");
    m.def("configure_chunk_loader",[](size_t cores,size_t bytes,const std::string& backend,size_t window){
        nb::gil_scoped_release release;
        auto options=vc::render::ChunkCacheService::batchOptions(cores,bytes);
        if(backend=="queued")options.loaderBackend=vc::render::ChunkCacheService::Options::LoaderBackend::Queued;
        else if(backend!="direct")throw std::invalid_argument("loader backend must be direct or queued");
        options.queuedChunkWindow=window;
        const auto service=vc::render::configureProcessChunkCacheService(options);
        return nlohmann::json{{"local_read_workers",service->localReadWorkers()},
            {"decode_workers",service->decodeWorkerCapacity()},{"cache_bytes",bytes},
            {"backend",backend},{"queued_chunk_window",service->queuedChunkWindow()}}.dump();
    },"cores"_a,"cache_bytes"_a,"backend"_a="direct","queued_chunk_window"_a=0);
    m.def("chunk_loader_stats",[]{
        const auto s=vc::render::processChunkCacheService()->loaderStats();
        return nlohmann::json{{"requests",s.requests},{"keys",s.keys},{"coalesced",s.coalesced},
            {"completed",s.completed},{"polls",s.polls},{"wakes",s.wakes},{"in_flight",s.inFlight},
            {"peak_in_flight",s.peakInFlight},{"pending",s.pending},{"peak_pending",s.peakPending},
            {"reply_bytes",s.replyBytes},{"peak_reply_bytes",s.peakReplyBytes}}.dump();
    });
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
    m.def("training_replay", [](const PredictionFieldHandle& field, const NormalSamplerHandle& normals,
        const NormalSamplerHandle& evaluationNormals, const ft::FiberTraceConfig& config,
        const ft::FiberEvaluationCase& input, const ft::FiberChunkReplayConfig& replay,
        bool predictionBasedSeeding, TraceProfileHandle* profile) {
        auto trace = checkedConfig(field, config, &normals);
        const auto c = input;
        const auto r = replay;
        TraceProfileUse use(profile);
        if (profile) trace.profile=&profile->value;
        nb::gil_scoped_release release;
        return ft::replayFiberTrainingCase(*field.field, trace, field.scales.traceToBaseScale,
            *normals.sampler, *evaluationNormals.sampler, evaluationNormals.workingToBaseScale,
            c, r, {}, predictionBasedSeeding);
    }, "field"_a, "normals"_a, "evaluation_normals"_a, "trace"_a, "case"_a, "replay"_a,
       "prediction_based_seeding"_a=true,"profile"_a=nb::none());
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
        .def_rw("channel_hints",&ft::FiberChunkTraceConfig::channelHints)
        .def_rw("worker_cache_bytes",&ft::FiberChunkTraceConfig::workerCacheBytes)
        .def_rw("max_grid_samples",&ft::FiberChunkTraceConfig::maxGridSamples)
        .def_rw("initial_alignment_power",&ft::FiberChunkTraceConfig::initialAlignmentPower)
        .def_rw("initial_normal_radius_base",&ft::FiberChunkTraceConfig::initialNormalRadiusBase)
        .def_rw("initial_tangential_radius_base",&ft::FiberChunkTraceConfig::initialTangentialRadiusBase)
        .def_rw("joint_join_score",&ft::FiberChunkTraceConfig::jointJoinScore)
        .def_rw("join_weights",&ft::FiberChunkTraceConfig::joinWeights)
        .def_rw("conflict_normal_base",&ft::FiberChunkTraceConfig::conflictNormalBase)
        .def_rw("conflict_tangent_base",&ft::FiberChunkTraceConfig::conflictTangentBase)
        .def_rw("conflict_agreement_base",&ft::FiberChunkTraceConfig::conflictAgreementBase)
        .def_rw("conflict_separation_base",&ft::FiberChunkTraceConfig::conflictSeparationBase)
        .def_rw("cache_only",&ft::FiberChunkTraceConfig::cacheOnly)
        .def_rw("presence_peak_restarts",&ft::FiberChunkTraceConfig::presencePeakRestarts)
        .def_rw("refine_seed",&ft::FiberChunkTraceConfig::refineSeed)
        .def_rw("seed_movement_base",&ft::FiberChunkTraceConfig::seedMovementBase)
        .def_rw("gpu_library",&ft::FiberChunkTraceConfig::gpuLibrary)
        .def_rw("gpu_devices",&ft::FiberChunkTraceConfig::gpuDevices)
        .def_rw("online_refine_iterations",&ft::FiberChunkTraceConfig::onlineRefineIterations)
        .def_rw("online_refine_window",&ft::FiberChunkTraceConfig::onlineRefineWindow)
        .def_rw("online_refine_mode",&ft::FiberChunkTraceConfig::onlineRefineMode)
        .def_rw("beam_commit_mode",&ft::FiberChunkTraceConfig::beamCommitMode)
        .def_rw("gpu_batch_halves",&ft::FiberChunkTraceConfig::gpuBatchHalves)
        .def_rw("gpu_batch_memory_mib",&ft::FiberChunkTraceConfig::gpuBatchMemoryMiB)
        .def_rw("online_refine_epsilon_base",&ft::FiberChunkTraceConfig::onlineRefineEpsilonBase)
        .def_rw("online_refine_step_base",&ft::FiberChunkTraceConfig::onlineRefineStepBase)
        .def_rw("online_refine_cap_base",&ft::FiberChunkTraceConfig::onlineRefineCapBase)
        .def("to_json",[](const ft::FiberChunkTraceConfig& c){return ft::fiberChunkConfigJson(c).dump();})
        .def_static("from_json",[](const std::string& j){return ft::fiberChunkConfigFromJson(nlohmann::json::parse(j));});
    m.def("chunk_cache_specification",[](const ft::FiberChunkTraceConfig& c,const ft::FiberTraceConfig& t,double scale,const std::string& identity) {
        return ft::fiberChunkCacheSpecification(c,t,scale,identity).dump();
    },"chunk"_a,"trace"_a,"trace_to_base"_a,"source_identity"_a);
    m.def("load_chunk_paths",[](const std::string& path) {
        nb::gil_scoped_release release;
        return ChunkPathsHandle{ft::loadFiberChunkPaths(path)};
    },"path"_a);
    m.def("prepare_chunk_volume",[](const PredictionFieldHandle& field,const NormalSamplerHandle& normals,
        const ft::FiberTraceConfig& trace,const ft::FiberChunkTraceConfig& chunks,
        const std::string& output,const std::string& identity,const std::vector<std::array<int,3>>& keys,
        ft::FiberEvaluationControl& control,size_t workers,size_t contexts) {
        const auto t=checkedConfig(field,trace,&normals);
        const auto c=chunks;const auto copiedKeys=keys;
        nb::gil_scoped_release release;
        ft::prepareFiberChunkVolume(*field.field,t,field.scales.traceToBaseScale,*normals.sampler,c,
            output,identity,copiedKeys,control,workers,contexts);
    },"field"_a,"normals"_a,"trace"_a,"chunks"_a,"output"_a,"source_identity"_a,"keys_xyz"_a,
       "control"_a,"workers"_a=4,"contexts_per_worker"_a=8);
    m.def("export_chunk_routes",[](const PredictionFieldHandle& field,const NormalSamplerHandle& normals,
        const ft::FiberTraceConfig& trace,const ft::FiberChunkTraceConfig& chunks,
        const std::string& directory,const std::string& output,const std::string& identity,
        const std::vector<std::array<int,3>>& keys,const std::vector<std::array<int,3>>& seeds,
        ft::FiberEvaluationControl& control,size_t workers,size_t contexts,size_t cacheBytes,
        double maximumDistance,int maximumSteps,size_t pagePoints,bool reduceDuplicates,bool rejectAmbiguous) {
        const auto t=checkedConfig(field,trace,&normals);const auto c=chunks;
        const auto copiedKeys=keys,copiedSeeds=seeds;
        const ft::FiberChunkRouteVolumeConfig config{cacheBytes,maximumDistance,maximumSteps,pagePoints,reduceDuplicates,rejectAmbiguous};
        nb::gil_scoped_release release;
        ft::exportFiberChunkRoutes(*field.field,t,field.scales.traceToBaseScale,*normals.sampler,c,
            directory,output,identity,copiedKeys,copiedSeeds,config,control,workers,contexts);
    },"field"_a,"normals"_a,"trace"_a,"chunks"_a,"chunk_directory"_a,"output"_a,"source_identity"_a,
        "keys_xyz"_a,"seed_keys_xyz"_a,"control"_a,"workers"_a=4,"contexts_per_worker"_a=1,
        "cache_bytes"_a=size_t(1024ULL*1024*1024),"maximum_distance_base"_a=1000000.,
        "maximum_steps"_a=100000,"page_points"_a=50000,"reduce_duplicate_paths"_a=true,"reject_ambiguous_paths"_a=true);
    m.def("build_chunk_paths",[](const PredictionFieldHandle& field,const NormalSamplerHandle& normals,
                                 const ft::FiberTraceConfig& t,const ft::FiberChunkTraceConfig& c,
                                 const std::array<int,3>& key,size_t workers) {
        if(workers<1)throw std::invalid_argument("workers must be positive");
        const auto trace=checkedConfig(field,t,&normals);
        const auto config=c;
        if(trace.parallelThreads!=1 || config.cacheOnly)
            throw std::invalid_argument("uncached construction requires parallel_threads=1 and cache_only=false");
        std::shared_ptr<ft::FiberChunkPaths> paths;
        {nb::gil_scoped_release release;
         ft::FiberCaseQueue queue(workers,1,[&]{
             paths=ft::buildFiberChunkPaths(key,*field.field,trace,*normals.sampler,config,
                 field.scales.traceToBaseScale,ft::FiberCaseQueue::parallelFor,
                 [](const auto& ready){ft::FiberCaseQueue::await(ready);});
         });queue.join();}
        return ChunkPathsHandle{paths};
    },"field"_a,"normals"_a,"trace"_a,"config"_a,"key_xyz"_a,"workers"_a=1);
    m.def("filter_chunk_paths",[](const ChunkPathsHandle& paths,const ft::FiberChunkTraceConfig& c,
                                 const NormalSamplerHandle& normals,double scale,size_t workers) {
        if(workers<1)throw std::invalid_argument("workers must be positive");
        auto copy=std::make_shared<ft::FiberChunkPaths>(*paths.paths);
        const auto config=c;
        std::string report;
        {nb::gil_scoped_release release;
         if(workers==1)report=ft::filterFiberChunkConflicts(*copy,config,*normals.sampler,scale).dump();
         else {
             ft::FiberCaseQueue queue(workers,1,[&]{
                 report=ft::filterFiberChunkConflicts(*copy,config,*normals.sampler,scale,ft::FiberCaseQueue::parallelFor).dump();
             });
             queue.join();
         }}
        return nb::make_tuple(ChunkPathsHandle{copy},report);
    },"paths"_a,"config"_a,"normals"_a,"trace_to_base"_a,"workers"_a=1);
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
        .def_prop_ro("timings_json",[](const ChunkPathsHandle& s){return s.paths->timings.dump();})
        .def_prop_ro("end_reasons",[](const ChunkPathsHandle& s){return s.paths->endReasons;})
        .def_prop_ro("conflict_flags",[](const ChunkPathsHandle& s){return s.paths->conflictFlags;})
        .def_prop_ro("conflict_annotation_json",[](const ChunkPathsHandle& s){return s.paths->conflictAnnotation.dump();})
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
        .def_ro("reverse_diagnostics",&ft::FiberReplayTraceResult::reverseDiagnostics)
        .def_ro("reverse_stop_tail_checks",&ft::FiberReplayTraceResult::reverseStopTailChecks)
        .def_ro("reverse_stop_tail_passed",&ft::FiberReplayTraceResult::reverseStopTailPassed)
        .def_ro("reverse_stop_tail_history_only",&ft::FiberReplayTraceResult::reverseStopTailHistoryOnly)
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

    m.def("open_fiber_normal_sampler",[](const PredictionFieldHandle& field,size_t cacheBytes) {
        auto handle=std::make_unique<NormalSamplerHandle>();
        nb::gil_scoped_release release;
        handle->dataset=std::make_unique<vc::lasagna::LasagnaDataset>(
            vc::lasagna::fiberNormalManifest(field.dataset->manifest()));
        handle->sampler=std::make_unique<vc::lasagna::LasagnaNormalSampler>(*handle->dataset,
            vc::lasagna::LasagnaNormalSamplerOptions{cacheBytes});
        handle->workingToBaseScale=field.scales.traceToBaseScale;
        handle->location=field.location;
        return handle;
    },"field"_a,"cache_bytes"_a=size_t{512}<<20);

    m.def("prediction_presence_storage",[](const PredictionFieldHandle& field) {
        const auto& manifest=field.dataset->manifest();
        const auto* group=manifest.groupForChannel("presence");
        if(!group||group->isRemote()||!manifest.baseShapeZYX)
            throw std::invalid_argument("sparse enumeration requires local presence and explicit base shape");
        return nb::make_tuple(group->zarrPath.string(),*group->channelIndex("presence"),
            manifest.sourceToBase*group->scaleFactor(),*manifest.baseShapeZYX);
    },"field"_a);

    nb::class_<ft::FiberStoredRouteSet>(m,"StoredRouteSet")
        .def("__init__",[](ft::FiberStoredRouteSet* self,const std::vector<PointsIn>& arrays,
            const std::vector<std::array<std::string,2>>& reasons){
            std::vector<std::vector<cv::Vec3d>> lines;for(const auto& a:arrays)lines.push_back(toPoints(a,"route"));
            nb::gil_scoped_release release;new(self)ft::FiberStoredRouteSet(std::move(lines),reasons);
        },"lines_base"_a,"end_reasons"_a)
        .def("replay",[](const ft::FiberStoredRouteSet& routes,const PredictionFieldHandle& field,
            const ft::FiberTraceConfig& trace,const PointsIn& points,const NormalSamplerHandle& normals,
            const ft::FiberChunkTraceConfig& chunks,const ft::FiberChunkReplayConfig& replay){
            const auto reference=ft::makePolylineArcGeometry(toPoints(points,"reference_base"));
            const auto t=checkedConfig(field,trace,&normals);const auto c=chunks;const auto r=replay;
            nb::gil_scoped_release release;
            return routes.replay(*field.field,t,field.scales.traceToBaseScale,reference,*normals.sampler,
                normals.workingToBaseScale,c,r);
        },"field"_a,"trace"_a,"reference_base"_a.noconvert(),"evaluation_normals"_a,"chunks"_a,"config"_a);

    nb::class_<ft::FiberChunkRoutingRules>(m,"ChunkRoutingRules");
    nb::class_<ChunkCacheHandle>(m,"ChunkCache")
        .def(nb::init<const PredictionFieldHandle&,const NormalSamplerHandle&,const ft::FiberTraceConfig&,
             const ft::FiberChunkTraceConfig&,const std::string&,const std::string&>(),
             "field"_a,"normal_sampler"_a,"trace_config"_a,"chunk_config"_a,"output_directory"_a,"source_identity"_a,
             nb::keep_alive<1,2>(),nb::keep_alive<1,3>())
        .def("get",&ChunkCacheHandle::get,"key_xyz"_a)
        .def("routing_rules",[](const ChunkCacheHandle& self,const std::string& directory){
            nb::gil_scoped_release release;
            return ft::FiberChunkRoutingRules(directory,self.directory,self.cache.config(),self.trace,
                self.predictions.scales.traceToBaseScale,self.identity);
        },"export_directory"_a,"Load complete, source/config-validated export rules; no inference or tracing.")
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
             "begin_arc_base"_a=0.,"end_arc_base"_a=nb::none(),"reverse_interval_base"_a=0.,"rules"_a=nb::none(),
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
            if(s.chunkVolume){out["built"]=s.built;out["reused"]=s.reused;out["annotated"]=s.annotated;}
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
        .def_rw("sampling_window_voxels", &ft::FiberTraceConfig::samplingWindowVoxels)
        .def_rw("smoothness_weight", &ft::FiberTraceConfig::smoothnessWeight)
        .def_rw("smoothness_normal_weight", &ft::FiberTraceConfig::smoothnessNormalWeight)
        .def_rw("sheet_plane_weight", &ft::FiberTraceConfig::sheetPlaneWeight)
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
        .def_rw("replay_presence_ignore_width", &ft::FiberTraceConfig::replayPresenceIgnoreWidth)
        .def_rw("width_hard_relative_tolerance", &ft::FiberTraceConfig::widthHardRelativeTolerance)
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
    nb::class_<StreamControl>(m,"StreamControl")
        .def(nb::init<>())
        .def("poll",[](StreamControl& c,int milliseconds) {
            std::vector<std::shared_ptr<StreamControl::Pending>> events;
            {nb::gil_scoped_release release;events=c.poll(milliseconds);}
            nb::list out;
            for(const auto& e:events) {
                nb::dict d;d["row"]=e->row;d["point"]=e->point;d["direction"]=e->direction;
                d["width"]=e->width;d["step"]=e->step;out.append(d);
            }
            return out;
        },"milliseconds"_a=5)
        .def("respond",&StreamControl::respond,"row"_a,"keep"_a)
        .def("cancel",&StreamControl::cancel);
    m.def("trace_stream", &traceStream, "predictions"_a,"normals"_a,
        "start"_a,"direction"_a,"distance"_a,"config"_a,"decision_steps"_a,"observe"_a,"initial_width"_a,
        "row"_a=0);
    m.def("chunk_seeds", &chunkSeeds, "predictions"_a,"normals"_a,"key_xyz"_a,"config"_a);
    nb::class_<ft::FiberVolumeCoverage>(m,"VolumeCoverage")
        .def(nb::init<double,size_t>(),"bin_size"_a,"merge_points"_a=50000)
        .def("add_owned",[](ft::FiberVolumeCoverage& c,const PointsIn& p,int64_t id,double width){
            auto points=toPoints(p,"points");nb::gil_scoped_release release;c.addOwned(points,id,width);
        },"points"_a,"id"_a,"width"_a)
        .def("flush_owned",&ft::FiberVolumeCoverage::flushOwned,nb::call_guard<nb::gil_scoped_release>())
        .def("owner",[](const ft::FiberVolumeCoverage& c,Vec3 p,Vec3 d,double r,double w,double t,int64_t exclude){
            nb::gil_scoped_release release;return c.owner(toVec(p),toVec(d),r,w,t,exclude);
        },"point"_a,"direction"_a,"radius"_a,"width"_a,"tolerance"_a,"exclude"_a=-1)
        .def("add_live",[](ft::FiberVolumeCoverage& c,Vec3 a,Vec3 b,int64_t pair,uint64_t epoch,double width){
            nb::gil_scoped_release release;c.addLive(toVec(a),toVec(b),pair,epoch,width);
        })
        .def("live_hit",[](const ft::FiberVolumeCoverage& c,Vec3 p,Vec3 d,double r,int64_t pair,uint64_t epoch,double w,double t){
            nb::gil_scoped_release release;return c.liveHit(toVec(p),toVec(d),r,pair,epoch,w,t);
        })
        .def("drop_live",&ft::FiberVolumeCoverage::dropLive,nb::call_guard<nb::gil_scoped_release>());
    m.def("run_stream_volume",[](const PredictionFieldHandle& field,const NormalSamplerHandle& normals,
        const ft::FiberTraceConfig& trace,const ft::FiberChunkTraceConfig& chunks,const std::string& options,
        const std::vector<std::array<int,3>>& keys,const std::string& output) {
        const auto checked=checkedConfig(field,trace,&normals);
        const auto cfg=nlohmann::json::parse(options);
        std::atomic_bool cancel{false};std::string result;
        auto future=std::async(std::launch::async,[&]{return ft::runFiberStreamVolume(
            *field.field,*normals.sampler,checked,chunks,cfg,keys,output,cancel).dump();});
        try {
            while(true) {
                bool ready;
                {nb::gil_scoped_release release;ready=future.wait_for(std::chrono::milliseconds(100))==std::future_status::ready;}
                if(PyErr_CheckSignals()!=0)throw nb::python_error();
                if(ready){nb::gil_scoped_release release;result=future.get();break;}
            }
        }catch(...) {
            cancel=true;
            // get() consumes the future even when the native worker throws.
            // Only an interrupted, still-associated future needs joining here.
            if(future.valid()){nb::gil_scoped_release release;future.wait();}
            throw;
        }
        return result;
    },"predictions"_a,"normals"_a,"trace"_a,"chunks"_a,"options_json"_a,"keys_xyz"_a,"output"_a);
    m.def("trace_extrapolation", &traceExtrapolation,
          "field"_a, "start"_a, "direction"_a, "distance_voxels"_a,
          "config"_a, "normal_sampler"_a = nb::none(),
          "Extrapolate from a point along an outward tangent for the requested trace-voxel distance.");
}
