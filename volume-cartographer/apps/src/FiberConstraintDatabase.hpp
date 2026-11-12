#pragma once
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <stop_token>
#include <span>
#include <unordered_map>
#include <vc_reader/affine.hpp>
#include <vc_object_store/objects/PackedStore.hpp>
namespace vc::objects { class InitialShardImport; }

struct FiberConstraintNormalSamples {
    // Two final attachment normals per constraint, in source endpoint order.
    // Regular along-fiber samples are transient and never retained here.
    std::vector<cv::Vec3d> endpoints;
    std::vector<unsigned char> valid;
    std::unordered_map<std::size_t,std::string> invalidReasons;
    std::vector<double> outwardAgreement,validArcWeight;
    nlohmann::json statistics;
};

struct FiberNormalBatchOptions {
    double regionBaseVoxels=4096.;
    std::size_t maximumPoints=262144;
};

FiberConstraintNormalSamples sampleFiberConstraintNormals(
    const std::vector<std::string>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals,int threads,double sampleSpacing,
    const vc::core::util::Umbilicus* umbilicus = nullptr,
    const FiberNormalBatchOptions& batching = {});

void writeFiberConstraintOutwardPriors(const std::filesystem::path& output,
    const std::vector<std::string>& sourceIds,const FiberConstraintNormalSamples& samples,
    const nlohmann::json& sourceSnapshot,const nlohmann::json& provenance);

struct FiberConstraintDatabaseOptions {
    int workers=32;
    int preparationWorkers=32;
    std::size_t preparationBatchRows=4096;
    std::size_t sqliteCacheBytes=128*1024*1024;
    vc::objects::PackedStoreOptions packedOptions=[] {
        vc::objects::PackedStoreOptions o;o.disposableBulk=true;o.memoryFirst=true;
        o.backgroundFlush=false;o.residentBytes=std::size_t{64}*1024*1024*1024;
        o.exportWorkers=8;return o;
    }();
    bool packed=true;
    // Borrowed for the duration of a materialized import. When supplied,
    // encoding consumes these values and does not resample endpoints.
    const FiberConstraintNormalSamples* normalSamples=nullptr;
    // Synchronous push producer: borrowed normals for this worker's current
    // batch, indexed from zero. Never reread completed chunk samples.
    std::function<const FiberConstraintNormalSamples*(std::size_t)> batchNormalSamples;
    // Transfer transient measurements after local-frame reconciliation. Native
    // edges are inserted immediately; only corrected standard-v2 measurements
    // are persisted. The callback runs on each existing shard writer.
    std::function<int64_t(std::size_t,vc::objects::InitialShardImport&)> finalizeMeasurements;
    // One inverse volume mapping per source fiber. Empty explicitly declares
    // identity. Core anchors use stored geometry coordinates; measured points
    // remain in base XYZ for the unchanged scientific calculations.
    std::vector<vcr::catalog::Affine> baseToStored;
};
FiberConstraintDatabaseOptions fiberConstraintDatabaseOptions(const nlohmann::json& config);

// Ordered source cursor: calls are serialized only to claim coarse batches
// and assign stable event IDs. Sampling/encoding runs on parallel import workers.
// Empty means successful end of stream;
// failures must throw. Blocking sources must honor cancellation on writer failure.
using FiberConstraintBatchSource = std::function<
    std::vector<vc::fiber_tracer::FiberTraceConstraint>(std::stop_token)>;

// Source workers call the consumer directly after scoring. Worker indices are
// exclusive per concurrent caller; event IDs are deterministic source ordinals,
// not completion-order counters. The producer joins its workers before returning.
using FiberConstraintBatchConsumer = std::function<void(std::size_t worker,
    std::size_t firstEvent,std::span<const vc::fiber_tracer::FiberTraceConstraint>)>;
using FiberConstraintBatchProducer = std::function<void(const FiberConstraintBatchConsumer&)>;

nlohmann::json streamFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<std::string>& sourceIds,
    std::span<const vc::fiber_tracer::FiberTraceConstraintPiece> pieces,
    const FiberConstraintBatchSource& source,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {}, std::size_t expectedEvents = 0,
    const FiberConstraintBatchProducer& producer = {}, std::size_t producerWorkers = 0);

nlohmann::json writeFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<std::string>& sourceIds,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {});

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,const std::vector<std::string>& sourceIds);

nlohmann::json orientFiberConstraintDatabase(
    const std::filesystem::path& input, const std::filesystem::path& output,
    const std::vector<std::string>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    double sampleSpacing, const nlohmann::json& provenance,
    const vc::core::util::Umbilicus* priorUmbilicus = nullptr,
    bool exportNormalCurves = false, bool selectedNormalCurves = false);
