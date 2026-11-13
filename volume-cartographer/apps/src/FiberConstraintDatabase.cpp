#include "FiberConstraintDatabase.hpp"
#include "vc/fiber_tracer/FiberNormalOrientation.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/core/util/SpatialBatches.hpp"
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <vc_object_store/objects/imports/ParallelImportStream.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/sources.hpp>
#include <algorithm>
#include <bit>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <iterator>
namespace {
namespace vo=vc::objects;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
Json point(const cv::Vec3d& p){return {p[0],p[1],p[2]};}
Json measurement(const vc::fiber_tracer::FiberTraceConstraint& c) {
    Json j;
#define FIELD(x) j[#x]=c.x
    FIELD(pieceA); FIELD(pieceB); FIELD(arcABaseVoxels); FIELD(arcBBaseVoxels);
    j["pointABaseXYZ"]=point(c.pointABaseXYZ); j["pointBBaseXYZ"]=point(c.pointBBaseXYZ);
    FIELD(closestDistanceBaseVoxels); FIELD(parallelScore); FIELD(perpendicularScore);
    FIELD(windingDistance); FIELD(hardContinuity); FIELD(artificialTileSeam);
    FIELD(allowMeasurementDisable); FIELD(evidenceWeight); FIELD(localEvent); FIELD(localOrientation);
    FIELD(supportABegin); FIELD(supportAEnd); FIELD(supportBBegin); FIELD(supportBEnd);
    FIELD(parallelWindingDistance); FIELD(parallelCorrespondenceSamples);
    FIELD(parallelMeanAdvanceResidualFraction); FIELD(parallelMeanConnectorTangentAbsDot);
    FIELD(parallelMeanConnectorLengthChangeFraction); FIELD(parallelMeanConnectorDirectionChange);
    FIELD(parallelLimitHitFraction);
#undef FIELD
#define OPTIONAL(x) j[#x]=c.x ? Json(*c.x) : Json(nullptr)
    OPTIONAL(signedWindingDelta); OPTIONAL(perpendicularNormalAlignment); OPTIONAL(windingNormalComponent);
    OPTIONAL(signedParallelWindingDelta); OPTIONAL(parallelNormalAlignment);
#undef OPTIONAL
    return j;
}
vo::ExtensionSource snapshot(const vo::ObjectCollection& store,const vo::CollectionTxn& txn){
    const auto p=store.provisioning(txn);return {store.uuid(txn),p.generation,p.commitSequence};
}
vo::InitialImportOptions importOptions(const std::filesystem::path& dependency,const FiberConstraintDatabaseOptions& settings){
    const auto workers=settings.workers;
    if(workers<1||workers>1024)throw std::invalid_argument("import_workers must be in 1..1024");
    vo::InitialImportOptions options;options.store.author="fiber-constraints";
    options.store.edgeColumns=std::vector<std::string>{};options.store.edgeAdjacencyIndex=false;
    options.dependencies={dependency};options.shards.resize(workers);
    const auto stride=INT64_MAX/workers;
    for(int i=0;i<workers;++i)options.shards[i].objectIds={1+i*stride,(i+1)*stride};
    options.store.sqliteCacheBytes=settings.sqliteCacheBytes;
    if(settings.packed)options.packed=settings.packedOptions;
    return options;
}
void appendRun(vo::InitialShardImport& writer,const std::string& run,const std::string& stage,
    const vo::ExtensionSource& input,const std::map<std::string,vo::FiberGraphSetting>& settings={}){
    const auto r=vo::fiberGraphRunRow(run,stage,"native",settings);
    writer.appendExtensionRows(r.table,std::span(&r.values,1));
    const auto i=vo::fiberGraphInputRow(run,"input",input);
    writer.appendExtensionRows(i.table,std::span(&i.values,1));
}
void resultDefinitions(vo::InitialImportOptions& options,const vo::ExtensionSource& input){
    for(auto kind:{vo::FiberGraphTable::Runs,vo::FiberGraphTable::Inputs,vo::FiberGraphTable::FiberResults}){
        auto table=vo::fiberGraphSolveTable(kind);table.basedOn={input};options.extensions.push_back(std::move(table));
    }
}
Json normalJson(const std::array<double,3>& normal,bool valid){
    return {{"valid",valid},{"xyz",valid?Json(normal):Json(nullptr)}};
}
// Preserve the source equality check without storing manifests, paths or
// selected-fiber lists in extension provisioning.
std::string normalSourceIdentity(const Json& metadata){
    return vcr::sources::snapshotSha256({
        {"fiber_source",metadata.at("fiber_source")},
        {"coordinate_space",metadata.at("config").at("coordinate_space")},
        {"manifest",metadata.at("normal_manifest")},
        {"provenance",metadata.at("normal_provenance")}});
}
}

FiberConstraintDatabaseOptions fiberConstraintDatabaseOptions(const nlohmann::json& cfg){
    FiberConstraintDatabaseOptions options;
    options.workers=cfg.value("import_workers",options.workers);
    options.preparationWorkers=cfg.value("import_readers",options.workers);
    options.preparationBatchRows=cfg.value("import_batch_rows",options.preparationBatchRows);
    options.sqliteCacheBytes=cfg.value("sqlite_cache_bytes",options.sqliteCacheBytes);
    options.packed=cfg.value("packed_store",options.packed);
    options.packedOptions.residentBytes=cfg.value("import_resident_bytes",options.packedOptions.residentBytes);
    options.packedOptions.exportWorkers=cfg.value("import_export_workers",options.packedOptions.exportWorkers);
    return options;
}

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,const std::vector<std::string>& sourceIds){
    vo::OpenOptions options;options.readOnly=true;auto store=vo::ObjectCollection::open(path,options);auto txn=store.read();
    const auto rows=vo::readFiberConstraintGraph(store,txn);
    const auto key=[](const std::string& aUuid,const std::string& bUuid,bool parallel,std::vector<uint8_t> a,std::vector<uint8_t> b,std::vector<uint8_t> connector,uint8_t distance){
        const auto pack=[&](const auto& x,const auto& y,const auto& c){std::vector<uint8_t> out{uint8_t(parallel),distance};
            const auto au=std::get<std::vector<uint8_t>>(vo::extensionUuid(aUuid)),bu=std::get<std::vector<uint8_t>>(vo::extensionUuid(bUuid));
            out.insert(out.end(),au.begin(),au.end());out.insert(out.end(),bu.begin(),bu.end());
            out.insert(out.end(),x.begin(),x.end());out.insert(out.end(),y.begin(),y.end());out.insert(out.end(),c.begin(),c.end());return out;};
        return pack(a,b,connector);
    };
    std::vector<std::vector<uint8_t>> expected,actual;
    for(const auto& c:report.constraints){const bool parallel=c.parallelScore>c.perpendicularScore;
        const auto code=vo::encodeFiberWindingDistance(parallel?c.parallelWindingDistance:c.windingDistance);if(!code)continue;
        const auto& a=sourceIds.at(report.pieces.at(c.pieceA).traceIndex);
        const auto& b=sourceIds.at(report.pieces.at(c.pieceB).traceIndex);
        const bool reversed=b<a;
        const auto delta=reversed?c.pointABaseXYZ-c.pointBBaseXYZ:c.pointBBaseXYZ-c.pointABaseXYZ;
        expected.push_back(key(reversed?b:a,reversed?a:b,parallel,vo::encodeFiberArc(reversed?c.arcBBaseVoxels:c.arcABaseVoxels),vo::encodeFiberArc(reversed?c.arcABaseVoxels:c.arcBBaseVoxels),
            vo::encodeFiberDirection({delta[0],delta[1],delta[2]}),*code));
    }
    for(const auto& row:rows){const auto& v=row.measurement.values;
        actual.push_back(key(row.a,row.b,row.parallel,std::get<std::vector<uint8_t>>(v[1]),std::get<std::vector<uint8_t>>(v[2]),
            std::get<std::vector<uint8_t>>(v[5]),std::get<std::vector<uint8_t>>(v[6]).at(0)));}
    std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
    if(actual!=expected)throw std::runtime_error("Compact constraint measurements differ after canonical ordering/quantization");
}

nlohmann::json writeFiberConstraintDatabase(const std::filesystem::path& output,
    const std::vector<std::string>& ids,const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals,int threads,const Json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress,const FiberConstraintDatabaseOptions& options){
    size_t position=0;
    return streamFiberConstraintDatabase(output,ids,report.pieces,[&](std::stop_token){
        const auto end=std::min(position+8192,report.constraints.size());
        std::vector<vc::fiber_tracer::FiberTraceConstraint> batch(report.constraints.begin()+position,report.constraints.begin()+end);
        position=end;return batch;
    },normals,threads,metadata,progress,options,report.constraints.size());
}

nlohmann::json streamFiberConstraintDatabase(const std::filesystem::path& output,
    const std::vector<std::string>& ids,std::span<const vc::fiber_tracer::FiberTraceConstraintPiece> pieces,
    const FiberConstraintBatchSource& source,const vc::lasagna::LasagnaNormalSampler& normals,
    int threads,const Json& metadata,const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress,
    const FiberConstraintDatabaseOptions& settings,size_t expectedEvents,
    const FiberConstraintBatchProducer& producer,size_t producerWorkers){
    (void)threads; // Sampling parallelism is owned by preparationWorkers, not nested teams.
    const auto start=Clock::now();
    if(pieces.size()!=ids.size())throw std::invalid_argument("Extraction import requires whole source fibers");
    if(!settings.baseToStored.empty()&&settings.baseToStored.size()!=ids.size())
        throw std::invalid_argument("Expected one anchor-frame mapping per source fiber");
    const std::filesystem::path dependency=metadata.at("config").at("fibers").get<std::string>();
    vo::OpenOptions readOptions;readOptions.readOnly=true;
    auto input=vo::ObjectCollection::open(dependency,readOptions);auto read=input.read();const auto pin=snapshot(input,read);
    auto options=importOptions(dependency,settings);
    if(settings.preparationWorkers<1||settings.preparationWorkers>1024||!settings.preparationBatchRows)
        throw std::invalid_argument("Positive import preparation workers and batch rows required");
    if(settings.normalSamples && (producer ||
       settings.normalSamples->endpoints.size()!=2*expectedEvents || settings.normalSamples->valid.size()!=2*expectedEvents))
        throw std::invalid_argument("Prepared attachment normals require a matching materialized source");
    auto constraintDefinition=vo::fiberConstraintExtension(
        {pin.storeUuid,{},{},metadata.at("config").at("coordinate_space").get<std::string>()},std::span(&pin,1));
    auto provisioning=Json::parse(constraintDefinition.provisioning);
    provisioning["fiber_source"]=metadata.at("fiber_source");
    provisioning["normal_source_sha256"]=normalSourceIdentity(metadata);
    if(metadata.contains("winding_manifest"))
        provisioning["winding_source_sha256"]=vcr::sources::snapshotSha256({
            {"manifest",metadata.at("winding_manifest")},{"provenance",metadata.at("winding_provenance")}});
    constraintDefinition.provisioning=provisioning.dump();
    options.extensions.push_back(std::move(constraintDefinition));
    resultDefinitions(options,pin);
    if(settings.finalizeMeasurements)options.edges.stagedExtensions={vo::kFiberConstraintTable};
    std::vector<double> transferSeconds(settings.workers);
    std::vector<int64_t> transferRows(settings.workers);
    const auto run=vo::util::newUuid();
    const auto normalRun=settings.normalSamples||settings.batchNormalSamples?vo::util::newUuid():std::string{};
    std::vector<std::vector<vo::ExtensionRow>> participants(settings.workers);
    for(const auto& id:ids){
        auto row=vo::fiberGraphSolveRow(vo::FiberGraphTable::FiberResults,
            {{"fiber_uuid",vo::extensionUuid(id)},{"run_uuid",vo::extensionUuid(run)}});
        participants[vo::uuidShardIndex(id,settings.workers)].push_back(std::move(row.values));
    }
    vo::imports::ParallelImportOptions queueOptions;
    queueOptions.shards=settings.workers;queueOptions.readers=settings.preparationWorkers;
    vo::imports::ParallelImportStream stream(queueOptions);
    struct Stats {size_t events=0,invalid=0,parallel=0,skippedDistance=0;double sampling=0,encoding=0,claiming=0;};
    std::vector<Stats> stats(producer?producerWorkers:queueOptions.readers);
    size_t claimed=0;bool eof=false;std::mutex sourceMutex;
    std::atomic<size_t> completed{0};
    std::atomic<size_t> skippedDistance{0};
    std::stop_source stop;
    const auto encode=[&](size_t worker,size_t first,
        std::span<const vc::fiber_tracer::FiberTraceConstraint> batch,auto& emitter) {
            auto& counters=stats.at(worker);
            std::vector<vc::lasagna::NormalSampleWithDerivative> sampled;
            const auto* batchNormals=settings.batchNormalSamples?settings.batchNormalSamples(worker):nullptr;
            if(settings.batchNormalSamples&&!batchNormals)throw std::invalid_argument("Chunk producer supplied no attachment normals");
            if(batchNormals && (batchNormals->endpoints.size()!=2*batch.size() || batchNormals->valid.size()!=2*batch.size()))
                throw std::invalid_argument("Chunk attachment normal count differs");
            if(!settings.normalSamples && !batchNormals) {
                std::vector<cv::Vec3d> points;points.reserve(2*batch.size());
                for(const auto& c:batch){points.push_back(c.pointABaseXYZ);points.push_back(c.pointBBaseXYZ);}
                const auto sampling=Clock::now();
                (void)normals.sampleNormalBatch(points,false,1,sampled);
                counters.sampling+=std::chrono::duration<double>(Clock::now()-sampling).count();
                if(sampled.size()!=points.size())throw std::runtime_error("Constraint normal sample count mismatch");
            }
            const auto encoding=Clock::now();const auto waitBefore=emitter.waitSeconds;
            for(size_t i=0;i<batch.size();++i){
                const auto& c=batch[i];vo::InitialCollectionEdge edge;
                const bool parallel=c.parallelScore>c.perpendicularScore;
                const auto distance=parallel?c.parallelWindingDistance:c.windingDistance;
                if(!vo::encodeFiberWindingDistance(distance)){++counters.skippedDistance;++skippedDistance;continue;}
                const auto event=first+i;
                edge.record.uuid=vo::util::nameUuid(run,"constraint:"+std::to_string(event));
                if(c.pieceA>=pieces.size()||c.pieceB>=pieces.size())throw std::runtime_error("Invalid piece index");
                const auto a=pieces[c.pieceA].traceIndex,b=pieces[c.pieceB].traceIndex;
                edge.record.aUuid=ids.at(a);edge.record.bUuid=ids.at(b);
                edge.record.kind=parallel?"parallel":"perpendicular";counters.parallel+=parallel;
                const std::array<std::array<double,3>,2> measuredPoints{{
                    {c.pointABaseXYZ[0],c.pointABaseXYZ[1],c.pointABaseXYZ[2]},
                    {c.pointBBaseXYZ[0],c.pointBBaseXYZ[1],c.pointBBaseXYZ[2]}}};
                const bool reversed=edge.record.bUuid<edge.record.aUuid;
                if(reversed){std::swap(edge.record.aUuid,edge.record.bUuid);std::swap(edge.record.aAnchor,edge.record.bAnchor);}
                edge.a={"",edge.record.aUuid};edge.b={"",edge.record.bUuid};
                vo::FiberConstraintData data;
                for(int side=0;side<2;++side){
                    auto& n=data.normals[side];
                    if(const auto* prepared=batchNormals?batchNormals:settings.normalSamples) {
                        const auto slot=2*(batchNormals?i:event)+side;const bool ok=prepared->valid.at(slot);n.valid=ok;
                        if(ok)for(int k=0;k<3;++k)n.xyz[k]=prepared->endpoints.at(slot)[k];
                    } else {
                        const auto& sample=sampled[2*i+side].sample;n.valid=sample.valid;
                        if(sample.valid)for(int k=0;k<3;++k)n.xyz[k]=sample.normal[k];
                    }
                    counters.invalid+=!n.valid.value();}
                data.values={c.arcABaseVoxels,c.arcBBaseVoxels,c.parallelScore,c.perpendicularScore,
                    c.windingDistance,c.parallelWindingDistance,c.evidenceWeight};
                data.measuredPoints=measuredPoints;
                edge.extensions.push_back(vo::fiberConstraintRow(edge.record.uuid,event,a,b,reversed,std::move(data)));
                if(settings.finalizeMeasurements){
                    // Exact source arcs and source indices are TEMP-only; final
                    // measurements contain neither piece IDs nor source IDs.
                    const std::array<uint64_t,4> transient{uint64_t(reversed?b:a),uint64_t(reversed?a:b),
                        std::bit_cast<uint64_t>(reversed?c.arcBBaseVoxels:c.arcABaseVoxels),
                        std::bit_cast<uint64_t>(reversed?c.arcABaseVoxels:c.arcBBaseVoxels)};
                    auto& payload=edge.extensions.back().stagingPayload;payload.resize(sizeof(transient));
                    std::memcpy(payload.data(),transient.data(),sizeof(transient));
                }
                const auto id=edge.record.uuid;const auto bytes=sizeof(edge)+vo::extensionPayloadBytes(edge.extensions);
                emitter.emit(std::move(edge),bytes,id);
                ++counters.events;
            }
            counters.encoding+=std::chrono::duration<double>(Clock::now()-encoding).count()-(emitter.waitSeconds-waitBefore);
            const auto done=completed.fetch_add(batch.size())+batch.size();
            if(progress)progress({"export",done,expectedEvents,expectedEvents});
    };
    if(producer) {
        stream.startEdgeProducer(producerWorkers,[&](auto& emitters) {
            producer([&](size_t worker,size_t first,auto batch) {
                encode(worker,first,batch,*emitters.at(worker));
            });
        });
    } else {
        // Materialized input has a cheap sequential cursor; sampling and
        // encoding remain parallel. Live extraction uses the push API above.
        stream.start(0,queueOptions.readers,[](size_t,size_t,auto&){},[&](size_t worker,size_t,auto& emitter){
          try {auto& counters=stats[worker];while(true){
            std::vector<vc::fiber_tracer::FiberTraceConstraint> batch;
            size_t first=0;const auto claiming=Clock::now();
            {
                std::lock_guard lock(sourceMutex);
                if(stop.stop_requested()||eof)break;
                first=claimed;batch.reserve(settings.preparationBatchRows);
                while(batch.size()<settings.preparationBatchRows){
                    auto next=source(stop.get_token());
                    if(next.empty()){eof=true;break;}
                    batch.insert(batch.end(),std::make_move_iterator(next.begin()),std::make_move_iterator(next.end()));
                }
                claimed+=batch.size();
            }
            counters.claiming+=std::chrono::duration<double>(Clock::now()-claiming).count();
            if(batch.empty())break;
            encode(worker,first,batch,emitter);
          }}catch(...){stop.request_stop();throw;}
        });
    }
    options.beforePublish=[&]{
        stream.finish();if(expectedEvents&&completed.load()!=expectedEvents)throw std::runtime_error("Extraction event count changed");
        if(settings.finalizeMeasurements) {
            int64_t transferred=0;for(const auto count:transferRows)transferred+=count;
            if(transferred!=int64_t(completed.load()-skippedDistance.load()))throw std::logic_error("Transferred measurement count differs from emitted constraints");
        }
        if(metadata.at("config").value("profile_stages",false)) {
            double queueWait=0.,encoding=0.;
            for(const auto& reader:stream.readers)queueWait+=reader.queueWaitSeconds;
            for(const auto& worker:stats)encoding+=worker.encoding;
            std::cout<<"IMPORT_PROFILE "<<Json{{"phase","ingestion_finished"},
                {"seconds",std::chrono::duration<double>(Clock::now()-start).count()},
                {"events",completed.load()},{"queue_wait_worker_seconds",queueWait},{"encoding_worker_seconds",encoding},
                {"monotonic_seconds",std::chrono::duration<double>(Clock::now().time_since_epoch()).count()}}.dump()<<std::endl;
        }
    };
    std::filesystem::create_directories(output.parent_path());vo::InitialImportResult result;
    try{result=vo::importInitialCollection(output,options,[&](size_t shard,auto& writer){
        if(shard==0) {
            appendRun(writer,run,"constraint_extraction",pin);
            if(!normalRun.empty())appendRun(writer,normalRun,"normal_orientation",pin,
                {{"sample_spacing_base",settings.normalSamples?
                    settings.normalSamples->statistics.at("sample_spacing_base").get<double>():
                    metadata.at("config").value("normal_sample_spacing",4.)}});
        }
        writer.appendExtensionRows(vo::fiberGraphSolveTable(vo::FiberGraphTable::FiberResults).name,participants[shard]);
    },[&](size_t shard,auto& writer){
        writer.importEdges([&](auto& batch){return stream.popEdges(shard,batch,[&]{writer.checkCancelled();});});
        if(settings.finalizeMeasurements) {
            const auto transferring=Clock::now();
            transferRows[shard]=settings.finalizeMeasurements(shard,writer);
            transferSeconds[shard]=std::chrono::duration<double>(Clock::now()-transferring).count();
        }
    });}catch(...){stop.request_stop();stream.fail(std::current_exception());throw;}
    size_t events=0,invalid=0,parallel=0;double sampleSeconds=0,encodingSeconds=0,claimSeconds=0,queueSeconds=0;
    for(const auto& s:stats){events+=s.events;invalid+=s.invalid;parallel+=s.parallel;sampleSeconds+=s.sampling;
        encodingSeconds+=s.encoding;claimSeconds+=s.claiming;}
    for(const auto& s:stream.readers)queueSeconds+=s.queueWaitSeconds;
    int64_t transferred=0;double transferWorkerSeconds=0.,transferMaximumSeconds=0.;
    for(size_t i=0;i<transferRows.size();++i){transferred+=transferRows[i];transferWorkerSeconds+=transferSeconds[i];transferMaximumSeconds=std::max(transferMaximumSeconds,transferSeconds[i]);}
    return {{"format","vcobj"},{"schema_version",22},{"path",result.path.string()},{"run_uuid",run},
        {"measurement_format",vo::kFiberConstraintFormat},{"skipped_winding_distance",skippedDistance.load()},
        {"measurement_staging",bool(settings.finalizeMeasurements)},
        {"measurement_transfer_rows",transferred},{"measurement_transfer_worker_seconds",transferWorkerSeconds},
        {"measurement_transfer_max_shard_seconds",transferMaximumSeconds},
        {"source_store",{{"store_uuid",result.storeUuid},{"generation",result.generation},{"commit_sequence",result.commitSequence}}},
        {"normal_samples_reused",settings.normalSamples!=nullptr || bool(settings.batchNormalSamples)},
        {"fibers",ids.size()},{"constraints",events},{"parallel",parallel},{"perpendicular",events-parallel},
        {"invalid_endpoint_normals",invalid},{"sampling_seconds",sampleSeconds},
        {"encoding_worker_seconds",encodingSeconds},{"source_claim_worker_seconds",claimSeconds},
        {"queue_wait_worker_seconds",queueSeconds},{"preparation_workers",producer?producerWorkers:size_t(settings.preparationWorkers)},
        {"shared_extraction_preparation_workers",bool(producer)},
        {"writer_workers",settings.workers},{"preparation_batch_rows",settings.preparationBatchRows},
        {"ingestion_seconds",result.ingestionSeconds},{"persistence_seconds",result.persistenceSeconds},
        {"seconds",std::chrono::duration<double>(Clock::now()-start).count()}};
}

namespace {
struct NormalEndpoint {size_t slot;double arc;};
struct NormalSweepResult {FiberConstraintNormalSamples data;Json curveIndex;};
NormalSweepResult sampleFiberNormalSweep(
    const std::vector<std::string>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const std::vector<std::vector<NormalEndpoint>>& endpoints,size_t events,
    const vc::lasagna::LasagnaNormalSampler& normals,int threads,double sampleSpacing,
    const vc::core::util::Umbilicus* priorUmbilicus,std::ofstream* curveOutput=nullptr,
    const FiberNormalBatchOptions& batching={}) {
    using namespace vc::fiber_tracer;
    if(threads<1 || !(sampleSpacing>0.) || !std::isfinite(sampleSpacing) ||
       sourceIds.size()!=lines.size() || endpoints.size()!=lines.size() ||
       !(batching.regionBaseVoxels>0.) || !std::isfinite(batching.regionBaseVoxels) || !batching.maximumPoints)
        throw std::invalid_argument("Invalid fiber normal sweep inputs");
    const bool exportNormalCurves=curveOutput!=nullptr;
    std::ofstream unused;auto& curveStream=curveOutput?*curveOutput:unused;
    Json curveIndex=Json::array();size_t curveSamples=0;
    std::vector<cv::Vec3d> baked(2*events);
    std::vector<double> priorSum(lines.size(),0.),priorWeight(lines.size(),0.);
    std::vector<unsigned char> valid(2*events,0);
    struct SweepStats {
        size_t samples=0,flips=0,endpointFlips=0,invalid=0,invalidEndpoints=0,invalidFibers=0;
        double sampleSeconds=0.;
        std::vector<std::pair<size_t,std::string>> invalidReasons;
    };
    const auto sweepStarted=Clock::now();
    const size_t regular=std::numeric_limits<size_t>::max();
    struct Sample {double arc;cv::Vec3d point;size_t slot;double weight=0.;};
    // Construct the unchanged arc-ordered scientific samples, independently of
    // their read order. UUID/storage order never selects an I/O batch.
    std::vector<std::vector<Sample>> perFiber(lines.size());
    std::exception_ptr preparationFailure;std::mutex preparationMutex;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(std::ptrdiff_t f=0;f<static_cast<std::ptrdiff_t>(lines.size());++f)try {
        auto& samples=perFiber[f];const auto geometry=makePolylineArcGeometry(lines[f].pointsBaseXYZ);
        for(double arc=0;arc<geometry.length();arc+=sampleSpacing)
            samples.push_back({arc,samplePolylineArc(geometry,arc).point,regular});
        samples.push_back({geometry.length(),samplePolylineArc(geometry,geometry.length()).point,regular});
        for(size_t j=0;j<samples.size();++j) {
            const double left=j==0?samples[j].arc:samples[j-1].arc;
            const double right=j+1==samples.size()?samples[j].arc:samples[j+1].arc;
            samples[j].weight=(right-left)*.5;
        }
        // Explicit source-geometry evaluation for this sampling operation;
        // stored measurement XYZ is intentionally unavailable in format 4.
        for(const auto& e:endpoints[f])samples.push_back({e.arc,samplePolylineArc(geometry,std::min(e.arc,geometry.length())).point,e.slot});
        std::stable_sort(samples.begin(),samples.end(),[](const auto& a,const auto& b){return a.arc<b.arc;});
    }catch(...){std::lock_guard lock(preparationMutex);if(!preparationFailure)preparationFailure=std::current_exception();}
    if(preparationFailure)std::rethrow_exception(preparationFailure);
    std::vector<size_t> offsets(lines.size()+1,0);
    for(size_t f=0;f<lines.size();++f)offsets[f+1]=offsets[f]+perFiber[f].size();
    std::vector<Sample> ordered(offsets.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(std::ptrdiff_t f=0;f<static_cast<std::ptrdiff_t>(lines.size());++f) {
        std::move(perFiber[f].begin(),perFiber[f].end(),ordered.begin()+offsets[f]);
        std::vector<Sample>().swap(perFiber[f]);
    }
    std::vector<std::vector<Sample>>().swap(perFiber);
    const auto preparedAt=Clock::now();
    auto plan=vc::core::util::spatialBatches(ordered.size(),batching.regionBaseVoxels,
        batching.maximumPoints,threads,[&](size_t i){return ordered[i].point;});
    const auto plannedAt=Clock::now();
    std::vector<cv::Vec3d> sampledNormals(ordered.size());
    std::vector<unsigned char> sampledValid(ordered.size(),0);
    const size_t readWorkers=std::min<size_t>(threads,plan.batches.size());
    std::vector<double> readSeconds(readWorkers,0.);
    std::vector<std::unordered_map<size_t,std::string>> readReasons(readWorkers);
    struct ReadScratch {
        std::vector<cv::Vec3d> points;
        std::vector<vc::lasagna::NormalSampleWithDerivative> sampled;
    };
    std::vector<ReadScratch> readScratch(readWorkers);
    std::atomic<size_t> donePoints{0};
    std::atomic<double> lastReadLog{0.};std::mutex readLogMutex;
    vc::core::util::parallelSpatialBatches(plan.batches.size(),threads,[&](size_t job,size_t worker,int innerWorkers){
        auto& points=readScratch[worker].points;auto& sampled=readScratch[worker].sampled;
            const auto [begin,end,weight]=plan.batches[job];points.resize(end-begin);
            for(size_t j=begin;j<end;++j)points[j-begin]=ordered[plan.order[j]].point;
            const auto started=Clock::now();
            (void)normals.sampleNormalBatch(points,false,innerWorkers,sampled);
            readSeconds[worker]+=std::chrono::duration<double>(Clock::now()-started).count();
            if(sampled.size()!=points.size())throw std::runtime_error("Normal batch size differs");
            for(size_t j=begin;j<end;++j) {
                const auto index=plan.order[j];const auto& value=sampled[j-begin].sample;
                sampledNormals[index]=value.normal;sampledValid[index]=value.valid;
                if(!value.valid&&ordered[index].slot!=regular)readReasons[worker].emplace(index,value.reason);
            }
            const auto done=donePoints.fetch_add(weight)+weight;
            const auto seconds=std::chrono::duration<double>(Clock::now()-plannedAt).count();
            if(done==ordered.size()||seconds-lastReadLog.load()>5.) {
                std::lock_guard lock(readLogMutex);
                if(done==ordered.size()||seconds-lastReadLog.load()>5.) {
                    lastReadLog.store(seconds);
                    std::cout<<"sample normals points="<<done<<"/"<<ordered.size()<<" seconds="<<seconds<<std::endl;
                }
            }
    });
    std::unordered_map<size_t,std::string> sampledReasons;
    for(auto& reasons:readReasons)for(auto& [index,reason]:reasons)sampledReasons.emplace(index,std::move(reason));
    const auto sampledAt=Clock::now();
    const auto readBatches=plan.batches.size(),readRegions=plan.regions,maximumBatchPoints=plan.maximumWeight;
    plan={};
    // Only the cheap orientation scan is fiber-based: it consumes the already
    // sampled values in exactly the original arc order. It performs no reads.
    const size_t fiberBatch=std::max<size_t>(1,std::min<size_t>(256,(lines.size()+threads-1)/threads));
    const auto workerCount=exportNormalCurves?size_t{1}:std::min<size_t>(threads,(lines.size()+fiberBatch-1)/fiberBatch);
    std::vector<SweepStats> sweepStats(workerCount);
    std::atomic<size_t> nextFiber{0},doneFibers{0};std::atomic<bool> failed{false};
    std::mutex progressMutex;std::vector<std::future<void>> sweepJobs;
    for(size_t worker=0;worker<workerCount;++worker)sweepJobs.push_back(std::async(std::launch::async,[&,worker]{
      try {
       auto& stats=sweepStats[worker];
       for(size_t begin;!failed.load()&&(begin=nextFiber.fetch_add(fiberBatch))<lines.size();) {
        const size_t end=std::min(begin+fiberBatch,lines.size());
        for(size_t f=begin;f<end;++f) {
            FiberNormalOrientationSweep sweep;size_t good=0;
            const auto firstSample=curveSamples;
            for(size_t j=offsets[f];j<offsets[f+1];++j) {
                auto n=sampledNormals[j];const bool ok=sampledValid[j];
                if(ok && (!std::isfinite(n.dot(n)) || std::abs(n.dot(n)-1.)>1e-6))
                    throw std::runtime_error("Non-unit valid normal");
                const bool flipped=sweep.orient(n,ok);
                stats.flips+=flipped;stats.invalid+=!ok;good+=ok;
                if(priorUmbilicus && ok && ordered[j].weight>0.) {
                    const cv::Vec3f p(ordered[j].point);
                    const cv::Vec3d radial(-priorUmbilicus->vector_to_umbilicus(p));
                    const double r=cv::norm(radial);
                    if(r>0.) {
                        priorSum[f]+=ordered[j].weight*n.dot(radial/r);
                        priorWeight[f]+=ordered[j].weight;
                    }
                }
                if(ordered[j].slot!=regular) {
                    const auto slot=ordered[j].slot;baked[slot]=n;valid[slot]=ok;
                    if(!ok)stats.invalidReasons.emplace_back(slot,sampledReasons.at(j));
                    stats.endpointFlips+=flipped;stats.invalidEndpoints+=!ok;
                }
                else if(exportNormalCurves) {
                    std::array<double,5> record{ordered[j].arc,n[0],n[1],n[2],ok?1.:0.};
                    for(double value:record) {
                        auto bytes=std::bit_cast<std::array<char,8>>(value);
                        if constexpr(std::endian::native!=std::endian::little)std::reverse(bytes.begin(),bytes.end());
                        curveStream.write(bytes.data(),bytes.size());
                    }
                    ++curveSamples;
                }
            }
            if(exportNormalCurves)curveIndex.push_back({sourceIds[f],firstSample,curveSamples-firstSample});
            stats.invalidFibers+=good==0;
        }
        stats.samples+=offsets[end]-offsets[begin];const auto done=doneFibers.fetch_add(end-begin)+(end-begin);
        if(done%8192==0 || done==lines.size()){
            std::lock_guard lock(progressMutex);
            std::cout<<"orient normals fibers="<<done<<"/"<<lines.size()
                <<" seconds="<<std::chrono::duration<double>(Clock::now()-sweepStarted).count()<<std::endl;
        }
       }
      }catch(...){failed.store(true);throw;}
    }));
    std::exception_ptr sweepFailure;
    for(auto& job:sweepJobs)try{job.get();}catch(...){if(!sweepFailure)sweepFailure=std::current_exception();}
    if(sweepFailure)std::rethrow_exception(sweepFailure);
    size_t samplesTotal=0,flips=0,endpointFlips=0,invalid=0,invalidEndpoints=0,invalidFibers=0;
    double sampleSeconds=std::accumulate(readSeconds.begin(),readSeconds.end(),0.);
    for(const auto& s:sweepStats){samplesTotal+=s.samples;flips+=s.flips;endpointFlips+=s.endpointFlips;
        invalid+=s.invalid;invalidEndpoints+=s.invalidEndpoints;invalidFibers+=s.invalidFibers;sampleSeconds+=s.sampleSeconds;}

    const double sweepSeconds=std::chrono::duration<double>(Clock::now()-sweepStarted).count();
    size_t attachmentSamples=0;
    for(const auto& e:endpoints)attachmentSamples+=e.size();
    FiberConstraintNormalSamples result;
    result.endpoints=std::move(baked);result.valid=std::move(valid);
    for(auto& worker:sweepStats)for(auto& [slot,reason]:worker.invalidReasons)
        result.invalidReasons.emplace(slot,std::move(reason));
    result.outwardAgreement.resize(lines.size());
    for(size_t i=0;i<lines.size();++i)
        result.outwardAgreement[i]=priorWeight[i]>0?priorSum[i]/priorWeight[i]:0.;
    result.validArcWeight=std::move(priorWeight);
    result.statistics={{"fibers",lines.size()},{"constraints",events},{"samples",samplesTotal},
        {"regular_samples",samplesTotal-attachmentSamples},{"attachment_samples",attachmentSamples},
        {"flipped_samples",flips},{"flipped_endpoints",endpointFlips},{"invalid_samples",invalid},
        {"invalid_endpoints",invalidEndpoints},{"fibers_without_valid_normals",invalidFibers},
        {"sample_spacing_base",sampleSpacing},{"sampling_seconds",sampleSeconds},
        {"sweep_seconds",sweepSeconds},{"sweep_workers",workerCount},
        {"normal_sample_passes",1},{"cross_fiber_alignment",false},
        {"read_region_base_voxels",batching.regionBaseVoxels},{"read_point_limit",batching.maximumPoints},
        {"read_batches",readBatches},{"read_regions",readRegions},{"maximum_batch_points",maximumBatchPoints},
        {"read_workers",readWorkers},{"coordinate_prepare_seconds",std::chrono::duration<double>(preparedAt-sweepStarted).count()},
        {"spatial_plan_seconds",std::chrono::duration<double>(plannedAt-preparedAt).count()},
        {"sampling_wall_seconds",std::chrono::duration<double>(sampledAt-plannedAt).count()},
        {"orientation_seconds",std::chrono::duration<double>(Clock::now()-sampledAt).count()}};
    return {std::move(result),std::move(curveIndex)};
}
} // namespace

FiberConstraintNormalSamples sampleFiberConstraintNormals(
    const std::vector<std::string>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals,int threads,double sampleSpacing,
    const vc::core::util::Umbilicus* umbilicus,const FiberNormalBatchOptions& batching) {
    std::vector<std::vector<NormalEndpoint>> endpoints(lines.size());
    for(size_t i=0;i<report.constraints.size();++i) {
        const auto& c=report.constraints[i];
        endpoints.at(report.pieces.at(c.pieceA).traceIndex).push_back({2*i,c.arcABaseVoxels});
        endpoints.at(report.pieces.at(c.pieceB).traceIndex).push_back({2*i+1,c.arcBBaseVoxels});
    }
    return sampleFiberNormalSweep(sourceIds,lines,endpoints,report.constraints.size(),
        normals,threads,sampleSpacing,umbilicus,nullptr,batching).data;
}

void writeFiberConstraintOutwardPriors(const std::filesystem::path& output,
    const std::vector<std::string>& sourceIds,const FiberConstraintNormalSamples& samples,
    const Json& sourceSnapshot,const Json& provenance) {
    if(std::filesystem::exists(output))throw std::runtime_error("Prior output already exists");
    Json document={{"format","fiber-outward-priors"},{"version",1},{"source_store",sourceSnapshot},
        {"umbilicus_sha256",provenance.at("config").at("umbilicus_sha256")},
        {"source_ids",sourceIds},{"agreement",samples.outwardAgreement},
        {"valid_arc_weight",samples.validArcWeight},{"provenance",provenance},{"statistics",samples.statistics}};
    std::filesystem::create_directories(output.parent_path());
    vcr::zarr::atomicWrite(output,document.dump()+"\n");
}

nlohmann::json orientFiberConstraintDatabase(const std::filesystem::path& input,const std::filesystem::path& output,
    const std::vector<std::string>& sourceIds,const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaNormalSampler& normals,int threads,double sampleSpacing,const Json& provenance,
    const vc::core::util::Umbilicus* priorUmbilicus,bool exportNormalCurves,bool selectedNormalCurves){
    using namespace vc::fiber_tracer;
    const auto started=Clock::now();
    if(sampleSpacing<=0||!std::isfinite(sampleSpacing)||threads<1||lines.size()!=sourceIds.size())
        throw std::invalid_argument("Invalid normal sampling configuration");
    if(selectedNormalCurves&&(!exportNormalCurves||sourceIds.empty()))throw std::invalid_argument("Selected export requires fibers and curve output");
    if(std::filesystem::exists(output))throw std::runtime_error("Output already exists");
    vo::OpenOptions options;options.readOnly=true;
    auto store=vo::ObjectCollection::open(input,options);auto txn=store.read();const auto pin=snapshot(store,txn);
    const auto graphStarted=Clock::now();
    std::cout<<"reading constraint graph"<<std::endl;
    auto records=vo::readCompactFiberConstraintGraph(store,txn,threads,[&](const char* stage,size_t count,size_t shard){
        std::cout<<"graph "<<stage<<" rows="<<count<<" shard="<<shard
                 <<" seconds="<<std::chrono::duration<double>(Clock::now()-graphStarted).count()<<std::endl;
    });
    const double graphSeconds=std::chrono::duration<double>(Clock::now()-graphStarted).count();
    std::cout<<"read constraint graph events="<<records.size()<<" seconds="<<graphSeconds<<std::endl;
    const auto schemas=store.extensionTables(txn);
    const auto spec=std::find_if(schemas.begin(),schemas.end(),[](const auto& t){return t.name==vo::kFiberConstraintTable;});
    if(spec==schemas.end())throw std::runtime_error("Missing measurement definition");
    if(Json::parse(spec->provisioning).at("normal_source_sha256")!=normalSourceIdentity(provenance))
        throw std::runtime_error("Normal source identity differs from extracted measurements");
    const auto runTable=vo::fiberGraphSolveTable(vo::FiberGraphTable::Runs);
    bool oriented=false;
    store.scanExtensionRows(txn,runTable.name,[&](const auto& row){
        if(std::get<std::string>(row[1])!="normal_orientation")return true;
        oriented=true;
        if((priorUmbilicus||exportNormalCurves) &&
            Json::parse(std::get<std::string>(row[3])).at("sample_spacing_base")!=sampleSpacing)
            throw std::runtime_error("Normal export spacing must match baked orientation");
        return true;
    });
    if(oriented!=bool(priorUmbilicus||exportNormalCurves))
        throw std::runtime_error("Orientation requires unaligned input; normal exports require oriented input");
    std::unordered_map<std::string,size_t> columns;
    for(size_t i=0;i<spec->columns.size();++i)columns[spec->columns[i].name]=i;
    std::vector<std::vector<NormalEndpoint>> endpoints(sourceIds.size());
    struct UuidHash{size_t operator()(const vo::FiberGraphUuid& key)const{
        uint64_t h=14695981039346656037ULL;for(auto b:key){h^=b;h*=1099511628211ULL;}return h;}};
    std::unordered_map<vo::FiberGraphUuid,size_t,UuidHash> index;
    std::vector<double> lengths(sourceIds.size());
    for(size_t i=0;i<sourceIds.size();++i){
        if(!index.emplace(vo::fiberGraphUuid(sourceIds[i]),i).second)throw std::runtime_error("Duplicate source fiber");
        lengths[i]=makePolylineArcGeometry(lines[i].pointsBaseXYZ).length();
    }
    if(selectedNormalCurves)std::erase_if(records,[&](const auto& r){return !index.contains(r.a)&&!index.contains(r.b);});
    const auto events=records.size();const auto fiberCount=sourceIds.size();
    std::vector<unsigned char> selectedSlots(2*events);
    for(size_t e=0;e<events;++e)for(int side=0;side<2;++side){
        const auto& r=records[e];const auto found=index.find(side?r.b:r.a);
        selectedSlots[2*e+side]=found!=index.end();
        if(found==index.end()){if(!selectedNormalCurves)throw std::runtime_error("Missing constraint fiber");continue;}
        const auto arc=side?r.arcB:r.arcA;const auto length=lengths[found->second];
        if(arc<0||arc>std::max(length,double(float(length))))throw std::runtime_error("Invalid constraint arc");
        endpoints[found->second].push_back({2*e+side,arc});
    }
    auto partial=output;partial+=".partial";
    if(std::filesystem::exists(partial))throw std::runtime_error("Partial export already exists");
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{partial};
    std::ofstream curveStream;
    if(exportNormalCurves){
        if(priorUmbilicus)throw std::invalid_argument("Choose curve export or outward priors");
        std::filesystem::create_directories(partial);curveStream.open(partial/"normals.bin",std::ios::binary);
        if(!curveStream)throw std::runtime_error("Cannot create curve payload");
    }
    auto sweep=sampleFiberNormalSweep(sourceIds,lines,endpoints,events,normals,threads,sampleSpacing,
        priorUmbilicus,exportNormalCurves?&curveStream:nullptr);
    const auto& baked=sweep.data.endpoints;const auto& valid=sweep.data.valid;
    const auto& curveIndex=sweep.curveIndex;
    // Keep the original geometric sweep: validity, axes and gauge are checked.
    size_t checkedEndpoints=0,changed=0,comparedNormals=0;
    double axisAngleSum=0.,axisAngleMax=0.,orientedAngleSum=0.,orientedAngleMax=0.;
    for(size_t e=0;e<events;++e)for(int side=0;side<2;++side){
        const auto slot=2*e+side;if(!selectedSlots[slot])continue;++checkedEndpoints;
        const auto& r=records[e];const bool ok=side?r.validB:r.validA;const auto& p=side?r.normalB:r.normalA;
        if(ok!=bool(valid[slot]))throw std::runtime_error("Endpoint validity changed");
        if(ok){const cv::Vec3d old(p[0],p[1],p[2]);
            const auto quantized=vo::decodeFiberDirection(vo::encodeFiberDirection({baked[slot][0],baked[slot][1],baked[slot][2]}));
            const cv::Vec3d decoded(quantized[0],quantized[1],quantized[2]);
            // Arc float32 storage and compact axes are deliberately quantized;
            // source identity is checked above, orientation is checked here.
            const auto dot=std::clamp(old.dot(decoded),-1.,1.);
            const double angle=std::acos(dot)*180./std::acos(-1.);
            const double axisAngle=std::acos(std::abs(dot))*180./std::acos(-1.);
            ++comparedNormals;axisAngleSum+=axisAngle;axisAngleMax=std::max(axisAngleMax,axisAngle);
            orientedAngleSum+=angle;orientedAngleMax=std::max(orientedAngleMax,angle);
            if((priorUmbilicus||exportNormalCurves)&&old.dot(decoded)<0.)throw std::runtime_error("Requested export needs previously oriented normals");
        }
    }
    auto stats=sweep.data.statistics;
    stats["compact_resampled_normal_comparisons"]=comparedNormals;
    stats["compact_resampled_axis_angle_mean_degrees"]=comparedNormals?axisAngleSum/comparedNormals:0.;
    stats["compact_resampled_axis_angle_max_degrees"]=axisAngleMax;
    stats["compact_resampled_oriented_angle_mean_degrees"]=comparedNormals?orientedAngleSum/comparedNormals:0.;
    stats["compact_resampled_oriented_angle_max_degrees"]=orientedAngleMax;
    stats["checked_endpoints"]=checkedEndpoints;stats["graph_read_seconds"]=graphSeconds;
    if(exportNormalCurves){
        curveStream.close();if(!curveStream)throw std::runtime_error("Normal curve write failed");
        Json doc={{"format","fiber-normal-curves"},{"version",1},{"dtype","<f8"},
            {"columns",{"arc","nx","ny","nz","valid"}},{"index",curveIndex},{"statistics",stats},
            {"provenance",provenance},{"normal_convention","fiber-local baked gauge; apply stage1 normal_sign"}};
        vcr::zarr::atomicWrite(partial/"index.json",doc.dump()+"\n");
        std::filesystem::rename(partial,output);return stats;
    }
    if(priorUmbilicus){
        writeFiberConstraintOutwardPriors(output,sourceIds,sweep.data,
            {{"store_uuid",pin.storeUuid},{"generation",pin.generation},{"commit_sequence",pin.commitSequence}},provenance);
        return stats;
    }
    auto importing=importOptions(input,fiberConstraintDatabaseOptions(provenance.at("config")));
    vo::ExtensionTable measured;measured.name=spec->name;measured.extension=spec->extension;measured.formatVersion=spec->formatVersion;
    measured.provisioning=spec->provisioning;measured.basedOn={pin};measured.columns=spec->columns;measured.primaryKey=spec->primaryKey;
    measured.indexes=spec->indexes;measured.keySpace=spec->keySpace;
    importing.extensions.push_back(measured);resultDefinitions(importing,pin);
    vo::imports::ParallelImportOptions queueOptions;
    queueOptions.shards=importing.shards.size();queueOptions.batchRows=4096;
    vo::imports::StreamFailure failure;
    vo::imports::ShardQueues<vo::ExtensionRow> queues(queueOptions.shards,queueOptions.queueBytes/queueOptions.shards,failure);
    std::atomic<size_t> changedRows{0};
    auto scanMeasurements=vo::prepareFiberConstraintMeasurementScan(store,txn,records,threads,[&](vo::FiberMeasurementBatch&& batch){
            vo::imports::ShardEmitter<vo::ExtensionRow> emitter(queues,failure,queueOptions);
            for(auto& [e,row]:batch){bool different=false;
                for(int side=0;side<2;++side)if(valid[2*e+side]){
                    auto& value=std::get<std::vector<uint8_t>>(row[columns.at(side?"normal_b":"normal_a")]);
                    const auto& n=baked[2*e+side];const auto now=vo::encodeFiberDirection({n[0],n[1],n[2]});
                    different|=value!=now;value=now;
                }
                if(different){const auto bytes=row.size()*sizeof(vo::ExtensionValue);
                    emitter.emit(std::move(row),bytes,vo::fiberGraphUuid(records[e].uuid));++changedRows;}
            }
            emitter.finish();
        });
    const auto run=vo::util::newUuid();
    std::filesystem::create_directories(output.parent_path());
    auto producer=std::async(std::launch::async,[&]{try{scanMeasurements();queues.close();
    }catch(...){failure.fail(std::current_exception());throw;}});
    vo::InitialImportResult result;
    try{result=vo::importInitialCollection(output,importing,[&](size_t shard,auto& writer){
        if(shard==0)appendRun(writer,run,"normal_orientation",pin,{{"sample_spacing_base",sampleSpacing}});
        std::vector<vo::ExtensionRow> batch;
        while(queues.pop(shard,batch,[&]{writer.checkCancelled();}))writer.appendExtensionRows(vo::kFiberConstraintTable,batch);
    });producer.get();}catch(...){failure.fail(std::current_exception());if(producer.valid())producer.wait();throw;}
    changed=changedRows.load();
    stats["updated_constraints"]=changed;stats["run_uuid"]=run;stats["path"]=result.path.string();
    stats["ingestion_seconds"]=result.ingestionSeconds;stats["persistence_seconds"]=result.persistenceSeconds;
    stats["total_seconds"]=std::chrono::duration<double>(Clock::now()-started).count();return stats;
}
