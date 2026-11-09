#include "FiberConstraintDatabase.hpp"
#include "vc/fiber_tracer/FiberNormalOrientation.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <vc_object_store/objects/imports/ParallelImportStream.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/sources.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
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
vo::InitialImportOptions importOptions(const std::filesystem::path& dependency,int workers,bool packed){
    if(workers<1||workers>1024)throw std::invalid_argument("import_workers must be in 1..1024");
    vo::InitialImportOptions options;options.store.author="fiber-constraints";
    options.dependencies={dependency};options.shards.resize(workers);
    const auto stride=INT64_MAX/workers;
    for(int i=0;i<workers;++i){options.shards[i].objectIds={1+i*stride,(i+1)*stride};options.shards[i].edgeIds=options.shards[i].objectIds;}
    if(packed)options.packed=vo::PackedStoreOptions{};
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
        {"fiber_store_uuid",metadata.at("fibers_store_uuid")},
        {"coordinate_space",metadata.at("config").at("coordinate_space")},
        {"manifest",metadata.at("normal_manifest")},
        {"provenance",metadata.at("normal_provenance")}});
}
}

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report){
    vo::OpenOptions options;options.readOnly=true;auto store=vo::ObjectCollection::open(path,options);auto txn=store.read();
    const auto rows=vo::readFiberConstraintGraph(store,txn);
    if(rows.size()!=report.constraints.size())throw std::runtime_error("Constraint comparison count differs");
    for(size_t i=0;i<rows.size();++i){
        const auto& row=rows[i];const auto& c=report.constraints[i];
        auto m=measurement(c);m.erase("pointABaseXYZ");m.erase("pointBBaseXYZ");
        const auto expected=vo::fiberConstraintRow(row.edge.uuid,i,report.pieces.at(c.pieceA).traceIndex,
            report.pieces.at(c.pieceB).traceIndex,row.a!=row.edge.aUuid,
            {c.arcABaseVoxels,c.arcBBaseVoxels,c.parallelScore,c.perpendicularScore,c.windingDistance,c.parallelWindingDistance,c.evidenceWeight},
            normalJson(row.normalA,row.validA),normalJson(row.normalB,row.validB),m,
            std::array<std::array<double,3>,2>{{{c.pointABaseXYZ[0],c.pointABaseXYZ[1],c.pointABaseXYZ[2]},
                {c.pointBBaseXYZ[0],c.pointBBaseXYZ[1],c.pointBBaseXYZ[2]}}});
        for(size_t j=0;j<expected.values.size();++j)
            if(j<12||j>=22)if(expected.values[j]!=row.measurement.values[j])
                throw std::runtime_error("Constraint measurement differs at event "+std::to_string(i));
        for(int k=0;k<3;++k)if(row.pointA[k]!=c.pointABaseXYZ[k]||row.pointB[k]!=c.pointBBaseXYZ[k])
            throw std::runtime_error("Constraint base point differs");
    }
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
    const FiberConstraintDatabaseOptions& settings,size_t expectedEvents){
    const auto start=Clock::now();
    if(pieces.size()!=ids.size())throw std::invalid_argument("Extraction import requires whole source fibers");
    if(!settings.baseToStored.empty()&&settings.baseToStored.size()!=ids.size())
        throw std::invalid_argument("Expected one anchor-frame mapping per source fiber");
    const std::filesystem::path dependency=metadata.at("config").at("fibers").get<std::string>();
    vo::OpenOptions readOptions;readOptions.readOnly=true;
    auto input=vo::ObjectCollection::open(dependency,readOptions);auto read=input.read();const auto pin=snapshot(input,read);
    auto options=importOptions(dependency,settings.workers,settings.packed);
    auto constraintDefinition=vo::fiberConstraintExtension(
        {pin.storeUuid,{},{},metadata.at("config").at("coordinate_space").get<std::string>()},std::span(&pin,1));
    auto provisioning=Json::parse(constraintDefinition.provisioning);
    provisioning["normal_source_sha256"]=normalSourceIdentity(metadata);
    constraintDefinition.provisioning=provisioning.dump();
    options.extensions.push_back(std::move(constraintDefinition));
    resultDefinitions(options,pin);
    const auto run=vo::util::newUuid();
    std::vector<std::vector<vo::ExtensionRow>> participants(settings.workers);
    for(const auto& id:ids){
        auto row=vo::fiberGraphSolveRow(vo::FiberGraphTable::FiberResults,
            {{"fiber_uuid",vo::extensionUuid(id)},{"run_uuid",vo::extensionUuid(run)}});
        participants[vo::uuidShardIndex(id,settings.workers)].push_back(std::move(row.values));
    }
    vo::imports::ParallelImportOptions queueOptions;queueOptions.shards=settings.workers;queueOptions.readers=1;
    vo::imports::ParallelImportStream stream(queueOptions);
    size_t events=0,invalid=0,parallel=0;double sampleSeconds=0;
    std::stop_source stop;
    stream.start(0,1,[](size_t,size_t,auto&){},[&](size_t,size_t,auto& emitter){
        while(true){
            auto batch=source(stop.get_token());if(batch.empty())break;
            std::vector<cv::Vec3d> points;points.reserve(2*batch.size());
            for(const auto& c:batch){points.push_back(c.pointABaseXYZ);points.push_back(c.pointBBaseXYZ);}
            std::vector<vc::lasagna::NormalSampleWithDerivative> sampled;
            const auto sampling=Clock::now();(void)normals.sampleNormalBatch(points,false,threads,sampled);
            sampleSeconds+=std::chrono::duration<double>(Clock::now()-sampling).count();
            if(sampled.size()!=points.size())throw std::runtime_error("Constraint normal sample count mismatch");
            for(size_t i=0;i<batch.size();++i){
                const auto& c=batch[i];vo::InitialCollectionEdge edge;
                edge.record.uuid=vo::util::nameUuid(run,"constraint:"+std::to_string(events));
                if(c.pieceA>=pieces.size()||c.pieceB>=pieces.size())throw std::runtime_error("Invalid piece index");
                const auto a=pieces[c.pieceA].traceIndex,b=pieces[c.pieceB].traceIndex;
                edge.record.aUuid=ids.at(a);edge.record.bUuid=ids.at(b);
                edge.record.kind=c.parallelScore>c.perpendicularScore?"parallel":"perpendicular";parallel+=edge.record.kind=="parallel";
                edge.record.aAnchor.position={c.pointABaseXYZ[0],c.pointABaseXYZ[1],c.pointABaseXYZ[2]};
                edge.record.bAnchor.position={c.pointBBaseXYZ[0],c.pointBBaseXYZ[1],c.pointBBaseXYZ[2]};
                const std::array<std::array<double,3>,2> measuredPoints{*edge.record.aAnchor.position,*edge.record.bAnchor.position};
                if(!settings.baseToStored.empty()){
                    edge.record.aAnchor.position=settings.baseToStored.at(a).apply(measuredPoints[0]);
                    edge.record.bAnchor.position=settings.baseToStored.at(b).apply(measuredPoints[1]);
                }
                edge.record.aAnchor.wholeGeometry=edge.record.bAnchor.wholeGeometry=true;
                const bool reversed=edge.record.bUuid<edge.record.aUuid;
                if(reversed){std::swap(edge.record.aUuid,edge.record.bUuid);std::swap(edge.record.aAnchor,edge.record.bAnchor);}
                edge.a={"",edge.record.aUuid};edge.b={"",edge.record.bUuid};
                std::array<Json,2> n;
                for(int side=0;side<2;++side){const auto& sample=sampled[2*i+side].sample;
                    n[side]={{"valid",sample.valid},{"xyz",sample.valid?point(sample.normal):Json(nullptr)},
                        {"reason",sample.reason}};invalid+=!sample.valid;}
                auto m=measurement(c);m.erase("pointABaseXYZ");m.erase("pointBBaseXYZ");
                edge.extensions.push_back(vo::fiberConstraintRow(edge.record.uuid,events,a,b,reversed,
                    {c.arcABaseVoxels,c.arcBBaseVoxels,c.parallelScore,c.perpendicularScore,c.windingDistance,c.parallelWindingDistance,c.evidenceWeight},
                    n[0],n[1],m,measuredPoints));
                const auto id=edge.record.uuid;const auto bytes=sizeof(edge)+vo::extensionPayloadBytes(edge.extensions);
                emitter.emit(std::move(edge),bytes,id);++events;
            }
            if(progress)progress({"export",events,expectedEvents,expectedEvents});
        }
    });
    options.beforePublish=[&]{stream.finish();if(expectedEvents&&events!=expectedEvents)throw std::runtime_error("Extraction event count changed");};
    std::filesystem::create_directories(output.parent_path());vo::InitialImportResult result;
    try{result=vo::importInitialCollection(output,options,[&](size_t shard,auto& writer){
        if(shard==0)appendRun(writer,run,"constraint_extraction",pin);
        writer.appendExtensionRows(vo::fiberGraphSolveTable(vo::FiberGraphTable::FiberResults).name,participants[shard]);
    },[&](size_t shard,auto& writer){
        writer.importEdges([&](auto& batch){return stream.popEdges(shard,batch,[&]{writer.checkCancelled();});});
    });}catch(...){stop.request_stop();stream.fail(std::current_exception());throw;}
    return {{"format","vcobj"},{"schema_version",21},{"path",result.path.string()},{"run_uuid",run},
        {"fibers",ids.size()},{"constraints",events},{"parallel",parallel},{"perpendicular",events-parallel},
        {"invalid_endpoint_normals",invalid},{"sampling_seconds",sampleSeconds},
        {"ingestion_seconds",result.ingestionSeconds},{"persistence_seconds",result.persistenceSeconds},
        {"seconds",std::chrono::duration<double>(Clock::now()-start).count()}};
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
    auto records=vo::readFiberConstraintGraph(store,txn);
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
    struct Endpoint{size_t slot;double arc;cv::Vec3d point;};
    std::vector<std::vector<Endpoint>> endpoints(sourceIds.size());
    std::unordered_map<std::string,size_t> index;
    std::vector<double> lengths(sourceIds.size());
    for(size_t i=0;i<sourceIds.size();++i){
        if(!index.emplace(sourceIds[i],i).second)throw std::runtime_error("Duplicate source fiber");
        lengths[i]=makePolylineArcGeometry(lines[i].pointsBaseXYZ).length();
    }
    if(selectedNormalCurves)std::erase_if(records,[&](const auto& r){return !index.contains(r.a)&&!index.contains(r.b);});
    const auto events=records.size();const auto fiberCount=sourceIds.size();
    std::vector<unsigned char> selectedSlots(2*events);
    for(size_t e=0;e<events;++e)for(int side=0;side<2;++side){
        const auto& r=records[e];const auto found=index.find(side?r.b:r.a);
        selectedSlots[2*e+side]=found!=index.end();
        if(found==index.end()){if(!selectedNormalCurves)throw std::runtime_error("Missing constraint fiber");continue;}
        const auto arc=side?r.arcB:r.arcA;const auto& p=side?r.pointB:r.pointA;
        if(arc<0||arc>lengths[found->second]+1e-6)throw std::runtime_error("Invalid constraint arc");
        endpoints[found->second].push_back({2*e+side,arc,{p[0],p[1],p[2]}});
    }
    std::vector<cv::Vec3d> baked(2*events);
    std::vector<double> priorSum(lines.size(),0.),priorWeight(lines.size(),0.);
    std::vector<unsigned char> valid(2*events,0);
    size_t samplesTotal=0,flips=0,endpointFlips=0,invalid=0,invalidEndpoints=0,invalidFibers=0;
    double sampleSeconds=0.;const auto sweepStarted=Clock::now();
    auto partial=output;partial+=".partial";
    if(std::filesystem::exists(partial))throw std::runtime_error("Partial export already exists");
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{partial};
    std::ofstream curveStream;Json curveIndex=Json::array();size_t curveSamples=0;
    if(exportNormalCurves){
        if(priorUmbilicus)throw std::invalid_argument("Choose curve export or outward priors");
        std::filesystem::create_directories(partial);curveStream.open(partial/"normals.bin",std::ios::binary);
        if(!curveStream)throw std::runtime_error("Cannot create curve payload");
    }
    for(size_t begin=0;begin<lines.size();begin+=256) {
        struct Sample { double arc; cv::Vec3d point; size_t slot; double weight=0.; };
        const size_t end=std::min(begin+256,lines.size()), regular=std::numeric_limits<size_t>::max();
        std::vector<Sample> ordered;std::vector<size_t> offsets{0};
        for(size_t i=begin;i<end;++i) {
            const auto geometry=makePolylineArcGeometry(lines[i].pointsBaseXYZ);
            const size_t first=ordered.size();
            for(double arc=0;arc<geometry.length();arc+=sampleSpacing)
                ordered.push_back({arc,samplePolylineArc(geometry,arc).point,regular});
            ordered.push_back({geometry.length(),samplePolylineArc(geometry,geometry.length()).point,regular});
            for(size_t j=first;j<ordered.size();++j) {
                const double left=j==first?ordered[j].arc:ordered[j-1].arc;
                const double right=j+1==ordered.size()?ordered[j].arc:ordered[j+1].arc;
                ordered[j].weight=(right-left)*.5;
            }
            for(const auto& e:endpoints[i])ordered.push_back({e.arc,e.point,e.slot});
            std::stable_sort(ordered.begin()+first,ordered.end(),[](const auto& a,const auto& b){return a.arc<b.arc;});
            offsets.push_back(ordered.size());
        }
        std::vector<cv::Vec3d> points;points.reserve(ordered.size());
        for(const auto& s:ordered)points.push_back(s.point);
        std::vector<vc::lasagna::NormalSampleWithDerivative> sampled;
        const auto sampleStarted=Clock::now();
        (void)normals.sampleNormalBatch(points,false,threads,sampled);
        sampleSeconds+=std::chrono::duration<double>(Clock::now()-sampleStarted).count();
        if(sampled.size()!=ordered.size())throw std::runtime_error("Normal batch size differs");
        for(size_t f=0;f+1<offsets.size();++f) {
            FiberNormalOrientationSweep sweep;size_t good=0;
            const auto firstSample=curveSamples;
            for(size_t j=offsets[f];j<offsets[f+1];++j) {
                auto n=sampled[j].sample.normal;const bool ok=sampled[j].sample.valid;
                if(ok && (!std::isfinite(n.dot(n)) || std::abs(n.dot(n)-1.)>1e-6))
                    throw std::runtime_error("Non-unit valid normal");
                const bool flipped=sweep.orient(n,ok);
                flips+=flipped;invalid+=!ok;good+=ok;
                if(priorUmbilicus && ok && ordered[j].weight>0.) {
                    const cv::Vec3f p(ordered[j].point);
                    const cv::Vec3d radial(-priorUmbilicus->vector_to_umbilicus(p));
                    const double r=cv::norm(radial);
                    if(r>0.) {
                        priorSum[begin+f]+=ordered[j].weight*n.dot(radial/r);
                        priorWeight[begin+f]+=ordered[j].weight;
                    }
                }
                if(ordered[j].slot!=regular) {
                    const auto slot=ordered[j].slot;baked[slot]=n;valid[slot]=ok;
                    endpointFlips+=flipped;invalidEndpoints+=!ok;
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
            if(exportNormalCurves)curveIndex.push_back({sourceIds[begin+f],firstSample,curveSamples-firstSample});
            invalidFibers+=good==0;
        }
        samplesTotal+=ordered.size();
        if(begin%8192==0 || end==lines.size())std::cout<<"orient normals fibers="<<end<<"/"<<lines.size()
            <<" samples="<<samplesTotal<<" seconds="<<std::chrono::duration<double>(Clock::now()-sweepStarted).count()<<std::endl;
    }

    const double sweepSeconds=std::chrono::duration<double>(Clock::now()-sweepStarted).count();
    // Keep the original geometric sweep: validity, axes and gauge are checked.
    size_t checkedEndpoints=0,changed=0;
    for(size_t e=0;e<events;++e)for(int side=0;side<2;++side){
        const auto slot=2*e+side;if(!selectedSlots[slot])continue;++checkedEndpoints;
        const auto& r=records[e];const bool ok=side?r.validB:r.validA;const auto& p=side?r.normalB:r.normalA;
        if(ok!=bool(valid[slot]))throw std::runtime_error("Endpoint validity changed");
        if(ok){const cv::Vec3d old(p[0],p[1],p[2]);
            const double difference=cv::norm(old-baked[slot]);
            if(std::min(difference,cv::norm(old+baked[slot]))>1e-12)throw std::runtime_error("Normal axis changed, not merely sign");
            if((priorUmbilicus||exportNormalCurves)&&difference>1e-12)throw std::runtime_error("Requested export needs previously oriented normals");
        }
    }
    Json stats={{"fibers",fiberCount},{"constraints",events},{"samples",samplesTotal},{"flipped_samples",flips},
        {"flipped_endpoints",endpointFlips},{"invalid_samples",invalid},{"invalid_endpoints",invalidEndpoints},
        {"fibers_without_valid_normals",invalidFibers},{"sample_spacing_base",sampleSpacing},
        {"sampling_seconds",sampleSeconds},{"sweep_seconds",sweepSeconds},{"checked_endpoints",checkedEndpoints},
        {"cross_fiber_alignment",false}};
    if(exportNormalCurves){
        curveStream.close();if(!curveStream)throw std::runtime_error("Normal curve write failed");
        Json doc={{"format","fiber-normal-curves"},{"version",1},{"dtype","<f8"},
            {"columns",{"arc","nx","ny","nz","valid"}},{"index",curveIndex},{"statistics",stats},
            {"provenance",provenance},{"normal_convention","fiber-local baked gauge; apply stage1 normal_sign"}};
        vcr::zarr::atomicWrite(partial/"index.json",doc.dump()+"\n");
        std::filesystem::rename(partial,output);return stats;
    }
    if(priorUmbilicus){
        std::vector<double> agreement(lines.size());
        for(size_t i=0;i<lines.size();++i)agreement[i]=priorWeight[i]>0?priorSum[i]/priorWeight[i]:0.;
        Json document={{"format","fiber-outward-priors"},{"version",1},
            {"source_store",{{"store_uuid",pin.storeUuid},{"generation",pin.generation},{"commit_sequence",pin.commitSequence}}},
            {"umbilicus_sha256",provenance.at("config").at("umbilicus_sha256")},
            {"source_ids",sourceIds},{"agreement",agreement},{"valid_arc_weight",priorWeight},
            {"provenance",provenance},{"statistics",stats}};
        std::filesystem::create_directories(output.parent_path());vcr::zarr::atomicWrite(output,document.dump()+"\n");return stats;
    }
    auto importing=importOptions(input,provenance.at("config").value("import_workers",32),provenance.at("config").value("packed_store",true));
    vo::ExtensionTable measured;measured.name=spec->name;measured.extension=spec->extension;measured.formatVersion=spec->formatVersion;
    measured.provisioning=spec->provisioning;measured.basedOn={pin};measured.columns=spec->columns;measured.primaryKey=spec->primaryKey;
    measured.indexes=spec->indexes;measured.keySpace=spec->keySpace;
    importing.extensions.push_back(measured);resultDefinitions(importing,pin);
    std::vector<std::vector<vo::ExtensionRow>> updated(importing.shards.size());
    for(size_t e=0;e<events;++e){
        auto& record=records[e];auto& row=record.measurement.values;bool different=false;
        for(int side=0;side<2;++side)if(valid[2*e+side]){
            const std::string prefix=side?"normal_b_":"normal_a_";
            for(int k=0;k<3;++k){auto& value=row[columns.at(prefix+"xyz"[k])];const double now=baked[2*e+side][k];
                different|=std::get<double>(value)!=now;value=now;}
        }
        if(different){updated[vo::uuidShardIndex(record.edge.uuid,updated.size())].push_back(std::move(row));++changed;}
    }
    const auto run=vo::util::newUuid();
    std::filesystem::create_directories(output.parent_path());
    const auto result=vo::importInitialCollection(output,importing,[&](size_t shard,auto& writer){
        if(shard==0)appendRun(writer,run,"normal_orientation",pin,{{"sample_spacing_base",sampleSpacing}});
        writer.appendExtensionRows(vo::kFiberConstraintTable,updated[shard]);
    });
    stats["updated_constraints"]=changed;stats["run_uuid"]=run;stats["path"]=result.path.string();
    stats["ingestion_seconds"]=result.ingestionSeconds;stats["persistence_seconds"]=result.persistenceSeconds;
    stats["total_seconds"]=std::chrono::duration<double>(Clock::now()-started).count();return stats;
}
