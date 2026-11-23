#include "FiberConstraintChunks.hpp"
#include "ChunkSaturationWindow.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/FiberNormalOrientation.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstring>
#include <ctime>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>

namespace {
using namespace vc::fiber_tracer;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using Cell=std::array<int64_t,3>;
double elapsed(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
std::optional<double> threadCpu() {
#ifdef CLOCK_THREAD_CPUTIME_ID
    timespec t{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0)return double(t.tv_sec)+double(t.tv_nsec)*1e-9;
#endif
    return {};
}

struct Frame {
    size_t source=0,ordinal=0;
    FiberNormalPieceFrame ends;
    bool arrived=false;
    signed char sign=2; // 2=pending; 0=explicitly unresolved; +/-1=resolved
    double outwardSum=0.,outwardWeight=0.;
};
struct Job {
    Cell cell;
    std::vector<size_t> ownedFrames;
    std::vector<size_t> encounters;
    std::vector<FiberTraceConstraint> events;
    FiberConstraintNormalSamples normals;
    Json metrics;
};
struct ChunkBenchmarkComplete:std::exception {
    const char* what()const noexcept override{return "intentional saturated chunk benchmark cutoff";}
};

// One regular arc grid per ORIGINAL fiber. Endpoints are evaluated against the
// preceding regular normal without changing that grid's local gauge. Thus two
// jobs referencing the same piece obtain the same frame independently of which
// attachments they own. Spatial cuts add identical boundary references, not
// scientific fiber endpoints or independent resampled curve geometries.
void sampleJob(Job& job,const std::vector<size_t>& owned,
    const std::vector<Frame>& frames,const std::vector<size_t>& offsets,
    const std::vector<PolylineArcGeometry>& geometries,
    const std::vector<PolylineGridPartition>& partitions,
    const vc::lasagna::LasagnaNormalSampler& reader,double spacing,
    const vc::core::util::Umbilicus& axis,
    std::vector<std::pair<size_t,Frame>>& completed)
{
    const auto started=Clock::now();
    std::map<size_t,std::vector<size_t>> attachments;
    for(size_t i=0;i<job.events.size();++i)for(int side=0;side<2;++side) {
        const auto& c=job.events[i];const auto source=side?c.pieceB:c.pieceA;
        const auto arc=side?c.arcBBaseVoxels:c.arcABaseVoxels;
        const auto& p=partitions[source];
        auto end=std::upper_bound(p.boundaries.begin(),p.boundaries.end(),arc,
            [](double a,const auto& b){return a<b.arc;});
        const auto ordinal=std::min(p.pieces.size()-1,size_t(end-p.boundaries.begin()-1));
        const auto frame=offsets[source]+ordinal;
        attachments[frame].push_back(2*i+side);
    }
    for(auto f:owned)attachments.try_emplace(f);
    struct Sample {double arc;cv::Vec3d point;size_t slot;double weight;};
    constexpr auto regular=std::numeric_limits<size_t>::max();
    struct Piece {size_t id,first,last;};
    std::vector<Sample> samples;std::vector<Piece> pieces;
    for(const auto& [id,slots]:attachments) {
        const auto& frame=frames[id];const auto source=frame.source;
        const auto& g=geometries[source];const auto& p=partitions[source];
        const auto& piece=p.pieces[frame.ordinal];
        const auto& a=p.boundaries[piece.beginBoundary];const auto& b=p.boundaries[piece.endBoundary];
        const auto first=samples.size();
        samples.push_back({a.arc,a.point,regular,0.});
        // Multiplication defines a stable, common original-fiber grid. Never
        // restart arc sampling at a spatial cut or repeatedly accumulate steps.
        const auto begin=size_t(std::ceil(a.arc/spacing));
        for(size_t k=begin;;++k) {
            const auto arc=double(k)*spacing;if(!(arc<b.arc))break;
            if(arc<a.arc)continue;
            const auto pnt=samplePolylineArc(g,arc).point;
            const double left=k?double(k-1)*spacing:0.;
            const double right=std::min(g.length(),double(k+1)*spacing);
            const auto weight=(right-left)*.5;
            if(arc==a.arc)samples.back().weight=weight;
            else samples.push_back({arc,pnt,regular,weight});
        }
        const double endWeight=b.arc==g.length()?
            (g.length()-std::max(0.,(std::ceil(g.length()/spacing)-1.)*spacing))*.5:0.;
        samples.push_back({b.arc,b.point,regular,endWeight});
        for(auto slot:slots) {
            const auto& c=job.events[slot/2];
            samples.push_back({slot%2?c.arcBBaseVoxels:c.arcABaseVoxels,
                slot%2?c.pointBBaseXYZ:c.pointABaseXYZ,slot,0.});
        }
        std::stable_sort(samples.begin()+first,samples.end(),[](const auto& a,const auto& b){return a.arc<b.arc;});
        pieces.push_back({id,first,samples.size()});
    }
    std::vector<cv::Vec3d> points;points.reserve(samples.size());
    for(const auto& sample:samples)points.push_back(sample.point);
    std::vector<vc::lasagna::NormalSampleWithDerivative> values;
    (void)reader.sampleNormalBatch(points,false,1,values);
    if(values.size()!=samples.size())throw std::runtime_error("Chunk normal sample count differs");
    job.normals.endpoints.resize(2*job.events.size());job.normals.valid.resize(2*job.events.size());
    size_t invalid=0;
    for(const auto& piece:pieces) {
        Frame frame;frame.source=frames[piece.id].source;frame.ordinal=frames[piece.id].ordinal;
        const auto& partition=partitions[frame.source];const auto& range=partition.pieces[frame.ordinal];
        frame.ends.beginBoundary=range.beginBoundary;frame.ends.endBoundary=range.endBoundary;
        FiberNormalOrientationSweep sweep;
        for(size_t i=piece.first;i<piece.last;++i) {
            auto n=values[i].sample.normal;const bool valid=values[i].sample.valid;
            if(valid && (!std::isfinite(n.dot(n)) || std::abs(n.dot(n)-1.)>1e-6))
                throw std::runtime_error("Non-unit valid chunk normal");
            if(samples[i].slot==regular) {
                sweep.orient(n,valid);
                if(samples[i].arc==partition.boundaries[range.beginBoundary].arc) {
                    frame.ends.beginNormal=n;frame.ends.beginValid=valid;
                }
                if(samples[i].arc==partition.boundaries[range.endBoundary].arc) {
                    frame.ends.endNormal=n;frame.ends.endValid=valid;
                }
                if(valid && samples[i].weight>0.) {
                    const cv::Vec3d radial(-axis.vector_to_umbilicus(cv::Vec3f(samples[i].point)));
                    const auto length=cv::norm(radial);
                    if(length>0.) {frame.outwardSum+=samples[i].weight*n.dot(radial/length);
                        frame.outwardWeight+=samples[i].weight;}
                }
            } else {
                auto atEndpoint=sweep;atEndpoint.orient(n,valid);
                const auto slot=samples[i].slot;job.normals.endpoints[slot]=n;job.normals.valid[slot]=valid;
                if(!valid)job.normals.invalidReasons.emplace(slot,values[i].sample.reason);
            }
            invalid+=!valid;
        }
        if(std::binary_search(owned.begin(),owned.end(),piece.id))completed.emplace_back(piece.id,std::move(frame));
    }
    job.metrics["normal_samples"]=samples.size();job.metrics["invalid_samples"]=invalid;
    job.metrics["normal_seconds"]=elapsed(started);
}
}

Json extractFiberConstraintChunks(const Json& cfg,Json& metadata,
    const std::vector<std::string>& ids,const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& extraction,
    const vc::lasagna::LasagnaNormalSampler& normals,const vc::lasagna::LasagnaNormalSampler& windings,
    const vc::core::util::Umbilicus& axis,FiberConstraintDatabaseOptions databaseOptions,
    std::function<Json()> memorySnapshot)
{
    const auto started=Clock::now();const double size=cfg.value("chunk_size_base_voxels",defaultFiberConstraintChunkSizeBaseVoxels);
    const double spacing=cfg.value("normal_sample_spacing",4.);
    const bool profile=cfg.value("profile_stages",false);
    const bool benchmark=cfg.value("saturated_chunk_benchmark",false);
    if(benchmark)databaseOptions.packedOptions.memoryStats=std::make_shared<vc::objects::PackedStoreMemoryStats>();
    const size_t workers=std::max<size_t>(1,extraction.parallelThreads);
    if(!(size>0.)||!std::isfinite(size)||!(spacing>0.)||!std::isfinite(spacing))
        throw std::invalid_argument("Invalid spatial chunk dimensions or normal spacing");
    if(ids.size()!=lines.size() || extraction.crossQueryPrefix || extraction.extraPieceCutsBaseVoxels.size())
        throw std::invalid_argument("Chunk pipeline requires ordinary whole source fibers");
    // Original fiber pieces supply normal frames, not geometry search halos.
    std::vector<PolylineArcGeometry> geometries(lines.size());
    std::vector<PolylineGridPartition> partitions(lines.size());
    std::atomic<size_t> next{0};std::vector<std::future<void>> preparation;
    for(size_t w=0;w<std::min(workers,lines.size());++w)preparation.push_back(std::async(std::launch::async,[&] {
        for(size_t i;(i=next.fetch_add(1))<lines.size();) {
            geometries[i]=makePolylineArcGeometry(lines[i].pointsBaseXYZ);
            partitions[i]=partitionPolylineArcGrid(geometries[i],size);
        }
    }));
    std::exception_ptr preparationFailure;
    for(auto& task:preparation)try{task.get();}catch(...){if(!preparationFailure)preparationFailure=std::current_exception();}
    if(preparationFailure)std::rethrow_exception(preparationFailure);
    std::map<Cell,size_t> cells;
    for(const auto& p:partitions)for(const auto& piece:p.pieces)cells.try_emplace(piece.cell,0);
    std::vector<Job> jobs(cells.size());size_t ordinal=0;
    for(auto& [cell,id]:cells){id=ordinal++;jobs[id].cell=cell;}
    std::vector<size_t> offsets(lines.size()+1);
    for(size_t f=0;f<lines.size();++f)offsets[f+1]=offsets[f]+partitions[f].pieces.size();
    std::vector<Frame> frames(offsets.back());
    for(size_t f=0;f<lines.size();++f)for(size_t p=0;p<partitions[f].pieces.size();++p) {
        const auto id=offsets[f]+p,owner=cells.at(partitions[f].pieces[p].cell);
        frames[id].source=f;frames[id].ordinal=p;jobs[owner].ownedFrames.push_back(id);
    }
    std::vector<FiberTraceConstraintPiece> pieces(lines.size());
    for(size_t i=0;i<lines.size();++i){pieces[i].traceIndex=i;pieces[i].endArcBaseVoxels=geometries[i].length();}
    const double planning=elapsed(started);
    std::cout<<"chunk catalog jobs="<<jobs.size()<<" pieces="<<frames.size()<<" workers="<<std::min(workers,jobs.size())
        <<" core="<<size<<" seconds="<<planning<<std::endl;
    std::mutex logMutex;std::atomic<size_t> nextChunk{0};std::atomic<bool> failed{false};
    std::vector<const FiberConstraintNormalSamples*> workerNormals(workers,nullptr);
    std::atomic<size_t> activeChunks{0},peakChunks{0};
    std::vector<Json> statistics(jobs.size());
    double poolSeconds=0.,poolCpu=0.,fusionSeconds=0.;
    double geometrySeconds=0.;FiberTraceConstraintReport geometryReport;
    std::optional<Json> benchmarkResult;
    databaseOptions.batchNormalSamples=[&](size_t worker){return workerNormals.at(worker);};
    std::promise<void> framesDone;auto framesReady=framesDone.get_future().share();
    std::vector<size_t> fusionInvalidated(databaseOptions.workers);
    // Per-piece frame bookkeeping is TEMP-only and removed by the transfer API.
    databaseOptions.finalizeMeasurements=[&](size_t shard,vc::objects::InitialShardImport& writer) {
        framesReady.get();
        return writer.transferStagedExtensionRows(vc::objects::kFiberConstraintTable,[&](vc::objects::ExtensionRow& row) {
            if(row.size()!=9)throw std::runtime_error("Missing transient normal-frame attachment payload");
            const auto& payload=std::get<std::vector<uint8_t>>(row.back());
            std::array<uint64_t,4> transient;
            if(payload.size()!=sizeof(transient))throw std::runtime_error("Invalid normal-frame attachment payload");
            std::memcpy(transient.data(),payload.data(),sizeof(transient));
            for(size_t side=0;side<2;++side) {
                const auto source=transient[side];const auto arc=std::bit_cast<double>(transient[2+side]);
                if(source>=partitions.size())throw std::runtime_error("Invalid staged fiber index");
                const auto& partition=partitions[source];
                const auto end=std::upper_bound(partition.boundaries.begin(),partition.boundaries.end(),arc,[](double a,const auto& b){return a<b.arc;});
                if(end==partition.boundaries.begin()||!std::isfinite(arc)||arc>partition.boundaries.back().arc)
                    throw std::runtime_error("Staged normal arc outside source fiber");
                const auto piece=std::min(partition.pieces.size()-1,size_t(end-partition.boundaries.begin()-1));
                const auto sign=frames[offsets[source]+piece].sign;
                auto& normal=std::get<std::vector<uint8_t>>(row.at(3+side));
                if(normal.size()!=3)throw std::runtime_error("Invalid staged compact normal");
                if(!sign) {
                    fusionInvalidated[shard]+=normal[2]!=0;normal={0,0,0};
                }else if(sign<0&&normal[2])normal[2]=normal[2]==1?255:1;
            }
        });
    };
    const auto producer=[&](const FiberConstraintBatchConsumer& consume) {
      bool framesCompleted=false;
      try {
        FiberTraceConstraintStream stages;
        stages.geometryReady=[&](std::span<const FiberTraceConstraintPiece> sourcePieces,
            std::span<const cv::Vec3d> positions,const FiberTraceConstraintStream::Measure& measure) {
        // Only finalized encounter locations determine ownership. No geometry
        // is copied, re-searched or re-refined in these spatial jobs.
        for(size_t i=0;i<positions.size();++i) {
            Cell cell;for(int k=0;k<3;++k)cell[k]=int64_t(std::floor(positions[i][k]/size));
            auto [it,inserted]=cells.try_emplace(cell,jobs.size());
            if(inserted){jobs.emplace_back();jobs.back().cell=cell;}
            jobs[it->second].encounters.push_back(i);
        }
        statistics.resize(jobs.size());
        std::cout<<"sampling partition encounters="<<positions.size()<<" jobs="<<jobs.size()<<std::endl;
        if(benchmark&&jobs.size()<workers) {
            benchmarkResult=Json{{"valid",false},{"reason","fewer_jobs_than_workers"},
                {"worker_count",workers},{"total_jobs",jobs.size()},
                {"chunk_size_base_voxels",size},{"total_geometric_encounters",positions.size()}};
            throw ChunkBenchmarkComplete{};
        }
        std::vector<size_t> execution(jobs.size());std::iota(execution.begin(),execution.end(),0);
        std::sort(execution.begin(),execution.end(),[&](size_t a,size_t b) {
            const auto wa=jobs[a].encounters.size()+jobs[a].ownedFrames.size();
            const auto wb=jobs[b].encounters.size()+jobs[b].ownedFrames.size();
            return wa!=wb?wa>wb:a<b;
        });
        const auto poolStart=Clock::now();const auto cpuStart=std::clock();
        const auto poolMonotonic=std::chrono::duration<double>(poolStart.time_since_epoch()).count();
        if(benchmark&&cfg.contains("benchmark_pool_start_file"))
            vcr::zarr::atomicWrite(cfg.at("benchmark_pool_start_file").get<std::string>(),Json{{"monotonic_seconds",poolMonotonic}}.dump()+"\n");
        std::atomic<size_t> liveJobs{0},finishedJobs{0};
        std::atomic<size_t> completedWorkers{0};
        std::array<std::atomic<uint64_t>,3> throughput{};
        ChunkSaturationWindow window(workers,cfg.value("benchmark_warmup_seconds",1.),
            cfg.value("benchmark_window_seconds",28.),cfg.value("benchmark_require_worker_completion",false));
        const auto snapshot=[&] {
            ChunkSaturationWindow::Sample sample;sample.seconds=elapsed(poolStart);
            sample.cpuSeconds=double(std::clock()-cpuStart)/CLOCKS_PER_SEC;
            sample.active=liveJobs.load();sample.remaining=jobs.size()-finishedJobs.load();
            sample.completedWorkers=completedWorkers.load();
            for(size_t i=0;i<throughput.size();++i)sample.counts[i]=throughput[i].load();
            return sample;
        };
        std::optional<std::jthread> monitor;
        if(benchmark)monitor.emplace([&](std::stop_token stop) {
            // Also bound a workload that never manages to occupy the pool.
            while(!stop.stop_requested()&&!window.done()) {
                window.observe(snapshot());
                if(elapsed(poolStart)>cfg.value("benchmark_warmup_seconds",1.)+cfg.value("benchmark_window_seconds",28.)+2.) {
                    window.abortUnstarted(snapshot());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Record the window BEFORE waiting for in-flight synchronous
            // samplers to drain. The owning benchmark harness terminates the
            // process after this atomic checkpoint and cleans its temporary
            // staging directory; no incomplete graph is ever published.
            if(window.done()&&cfg.contains("benchmark_cutoff_file")) {
                auto measured=window.summary();measured["total_jobs"]=jobs.size();
                measured["pool_start_monotonic_seconds"]=poolMonotonic;
                measured["chunk_size_base_voxels"]=size;
                measured["total_geometric_encounters"]=positions.size();
                measured["final_persistence_seconds"]=0.;
                Json checkpoint={{"benchmark_cutoff_ready",true},{"config",cfg},
                    {"saturated_chunk_benchmark",measured},
                    {"selected_fibers",ids.size()},{"points",metadata.at("points")},
                    {"query_seconds",metadata.at("query_seconds")},{"load_seconds",metadata.at("load_seconds")},
                    {"chunk_pipeline",{{"geometry_stage_seconds",geometrySeconds},{"catalog_seconds",planning}}}};
                if(memorySnapshot)checkpoint["memory_snapshot"]=memorySnapshot();
                if(const auto& stats=databaseOptions.packedOptions.memoryStats)
                    checkpoint["memory_snapshot"]["packed_vfs"]={{"resident_bytes",stats->residentBytes.load()},
                        {"peak_resident_bytes",stats->peakResidentBytes.load()}};
                checkpoint["memory_budgets"]={{"packed_vfs_bytes",databaseOptions.packedOptions.residentBytes},
                    {"sqlite_page_cache_bytes",databaseOptions.sqliteCacheBytes*databaseOptions.workers}};
                vcr::zarr::atomicWrite(cfg.at("benchmark_cutoff_file").get<std::string>(),checkpoint.dump(2)+"\n");
                std::cout<<"SATURATED_WINDOW_CUTOFF "<<measured.dump()<<std::endl;
            }
        });
        std::vector<std::future<void>> tasks;
        for(size_t worker=0;worker<std::min(workers,jobs.size());++worker)tasks.push_back(std::async(std::launch::async,[&,worker] {
          try {
            bool completedOnce=false;
            for(size_t ticket;!failed.load()&&(!benchmark||!window.done())&&(ticket=nextChunk.fetch_add(1))<execution.size();) {
                const auto id=execution[ticket];auto& job=jobs[id];
                if(benchmark)liveJobs.fetch_add(1);
                const auto active=activeChunks.fetch_add(1)+1;
                auto peak=peakChunks.load();while(peak<active&&!peakChunks.compare_exchange_weak(peak,active)){}
                const auto jobStarted=Clock::now();
                const auto cpuStarted=threadCpu();
                const auto samplingStart=Clock::now();
                for(size_t begin=0;begin<job.encounters.size();begin+=extraction.candidateBatchSize) {
                    if(benchmark&&window.done())break;
                    const auto count=std::min(extraction.candidateBatchSize,job.encounters.size()-begin);
                    auto measured=measure(std::span<const size_t>(job.encounters).subspan(begin,count),worker);
                    if(benchmark)throughput[0].fetch_add(measured.size());
                    job.events.insert(job.events.end(),std::make_move_iterator(measured.begin()),std::make_move_iterator(measured.end()));
                }
                if(benchmark&&window.done()){liveJobs.fetch_sub(1);activeChunks.fetch_sub(1);break;}
                job.metrics={{"chunk",id},{"worker",worker},{"cell",job.cell},
                    {"candidates",job.encounters.size()},{"refined",0},
                    {"events",job.events.size()},{"winding_seconds",elapsed(samplingStart)},
                    {"start_seconds",std::chrono::duration<double>(jobStarted-started).count()}};
                const auto windingCpu=profile?threadCpu():std::optional<double>{};
                if(windingCpu&&cpuStarted)job.metrics["winding_thread_cpu_seconds"]=*windingCpu-*cpuStarted;
                for(auto& c:job.events) {
                    c.pieceA=sourcePieces[c.pieceA].traceIndex;
                    c.pieceB=sourcePieces[c.pieceB].traceIndex;
                }
                std::vector<std::pair<size_t,Frame>> completed;
                sampleJob(job,job.ownedFrames,frames,offsets,geometries,partitions,normals,spacing,axis,completed);
                if(benchmark)throughput[1].fetch_add(job.metrics.at("normal_samples").get<uint64_t>());
                if(benchmark&&window.done()){liveJobs.fetch_sub(1);activeChunks.fetch_sub(1);break;}
                job.metrics["compute_seconds"]=elapsed(jobStarted);
                if(const auto cpu=threadCpu();cpu&&cpuStarted)job.metrics["chunk_thread_cpu_seconds"]=*cpu-*cpuStarted;
                if(profile&&windingCpu)if(const auto cpu=threadCpu())job.metrics["normal_thread_cpu_seconds"]=*cpu-*windingCpu;
                activeChunks.fetch_sub(1);
                {std::lock_guard lock(logMutex);std::cout<<"CHUNK_COMPUTED "<<job.metrics.dump()<<std::endl;}
                for(auto& [f,value]:completed) {
                    frames[f].ends=value.ends;frames[f].outwardSum=value.outwardSum;
                    frames[f].outwardWeight=value.outwardWeight;frames[f].arrived=true;
                }
                // Import immediately in the deterministic local-piece gauge.
                // No job waits for another job's geometry, reads or signs.
                const auto writing=Clock::now();workerNormals[worker]=&job.normals;
                const auto outputCpu=profile?threadCpu():std::optional<double>{};
                if(id>0x7fffffffULL || job.events.size()>0xffffffffULL)throw std::overflow_error("Chunk event identifier exceeds int64");
                consume(worker,id*0x100000000ULL,job.events);workerNormals[worker]=nullptr;
                if(benchmark) {
                    throughput[2].fetch_add(job.events.size());liveJobs.fetch_sub(1);
                    const auto finished=finishedJobs.fetch_add(1)+1;
                    if(!completedOnce){completedOnce=true;completedWorkers.fetch_add(1);}
                    if(jobs.size()-finished<workers)window.observe(snapshot());
                }
                job.metrics["output_seconds"]=elapsed(writing);
                if(outputCpu)if(const auto cpu=threadCpu())job.metrics["output_thread_cpu_seconds"]=*cpu-*outputCpu;
                {std::lock_guard lock(logMutex);std::cout<<"CHUNK "<<job.metrics.dump()<<std::endl;}
                statistics[id]=std::move(job.metrics);
                std::vector<FiberTraceConstraint>().swap(job.events);job.normals={};
                std::vector<size_t>().swap(job.encounters);
            }
          } catch(...) {failed.store(true);throw;}
        }));
        std::exception_ptr error;
        for(auto& task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
        if(benchmark) {
            monitor->request_stop();monitor->join();
            if(!window.done())window.observe(snapshot());
            benchmarkResult=window.summary();
            (*benchmarkResult)["total_jobs"]=jobs.size();(*benchmarkResult)["chunk_size_base_voxels"]=size;
            (*benchmarkResult)["total_geometric_encounters"]=positions.size();
            (*benchmarkResult)["completed_jobs_including_drain"]=finishedJobs.load();
            (*benchmarkResult)["pool_seconds_including_drain"]=elapsed(poolStart);
            (*benchmarkResult)["chunks"]=statistics;
            std::cout<<"SATURATED_CHUNK_RESULT "<<benchmarkResult->dump()<<std::endl;
            // Never publish an incomplete graph or perform final persistence.
            throw ChunkBenchmarkComplete{};
        }
        const auto fusion=Clock::now();
        for(size_t source=0;source<lines.size();++source)for(size_t f=offsets[source];f<offsets[source+1];++f) {
            auto& frame=frames[f];if(!frame.arrived)throw std::logic_error("Missing completed normal piece");
            if(f==offsets[source])frame.sign=1;
            else {
                const auto& previous=frames[f-1];
                const std::array<FiberNormalPieceFrame,2> pair{previous.ends,frame.ends};
                frame.sign=static_cast<signed char>(previous.sign*reconcileFiberNormalPieceFrames(pair)[1]);
            }
        }
        fusionSeconds=elapsed(fusion);
        std::cout<<"normal sign fusion pieces="<<frames.size()<<" seconds="<<fusionSeconds<<std::endl;
        framesDone.set_value();
        framesCompleted=true;
        poolSeconds=elapsed(poolStart);const auto cpuEnd=std::clock();
        if(cpuStart!=std::clock_t(-1)&&cpuEnd!=std::clock_t(-1))poolCpu=double(cpuEnd-cpuStart)/CLOCKS_PER_SEC;
        };
        const auto geometryStart=Clock::now();
        geometryReport=extractFiberTraceConstraints(lines,extraction,
            [&](const auto& a,const auto& b,double step){return windings.normalAlignedWindingDistance(a,b,step);},
            [&](const auto& pairs,double step,int inner){return windings.normalAlignedWindingDistancesBatch(pairs,step,inner);},
            {},nullptr,[&](const auto& progress) {
                if(profile && (progress.completed==0 || progress.completed==progress.total)) {
                    std::lock_guard lock(logMutex);
                    std::cout<<"GEOMETRY_PROFILE "<<Json{{"phase",progress.phase},
                        {"completed",progress.completed},{"total",progress.total},
                        {"seconds",elapsed(geometryStart)},{"process_cpu_seconds",double(std::clock())/CLOCKS_PER_SEC},
                        {"monotonic_seconds",std::chrono::duration<double>(Clock::now().time_since_epoch()).count()}}.dump()<<std::endl;
                }
                if(progress.phase=="geometry_complete") {
                    geometrySeconds=elapsed(geometryStart);
                    std::cout<<"global geometry seconds="<<geometrySeconds<<" candidates="<<progress.candidates<<std::endl;
                }
            },stages);
      } catch(...) {if(!framesCompleted)framesDone.set_exception(std::current_exception());throw;}
    };
    const auto processing=Clock::now();
    Json exported;
    try {exported=streamFiberConstraintDatabase(cfg.at("database").get<std::string>(),ids,pieces,{},
        normals,1,metadata,{},databaseOptions,0,producer,workers);}
    catch(...) {
        if(!benchmarkResult)throw;
        (*benchmarkResult)["partial_graph_discarded"]=true;
        (*benchmarkResult)["final_persistence_seconds"]=0.;
        return {{"benchmark",*benchmarkResult},{"geometry_stage_seconds",geometrySeconds},
            {"catalog_seconds",planning},{"seconds_including_cleanup",elapsed(started)}};
    }
    size_t newlyInvalidated=0;for(const auto count:fusionInvalidated)newlyInvalidated+=count;
    exported["invalid_endpoint_normals_before_piece_fusion"]=exported.at("invalid_endpoint_normals");
    exported["piece_fusion_invalidated_endpoint_normals"]=newlyInvalidated;
    exported["invalid_endpoint_normals"]=exported.at("invalid_endpoint_normals").get<size_t>()+newlyInvalidated;
    FiberConstraintNormalSamples priors;priors.outwardAgreement.resize(lines.size());priors.validArcWeight.resize(lines.size());
    size_t unresolved=0;
    for(size_t f=0;f<lines.size();++f) {
        double sum=0.,weight=0.;
        for(size_t p=offsets[f];p<offsets[f+1];++p) {
            const auto& frame=frames[p];unresolved+=frame.sign==0;
            if(frame.sign){sum+=frame.sign*frame.outwardSum;weight+=frame.outwardWeight;}
        }
        priors.outwardAgreement[f]=weight>0?sum/weight:0.;priors.validArcWeight[f]=weight;
    }
    priors.statistics={{"sample_spacing_base",spacing}};
    writeFiberConstraintOutwardPriors(cfg.at("normal_prior_output").get<std::string>(),ids,priors,exported.at("source_store"),metadata);
    // Global geometry counts must not be confused with sums of halo searches.
    return {{"export",exported},{"chunks",statistics},{"chunk_count",jobs.size()},{"chunk_size_base_voxels",size},
        {"chunk_workers",std::min(workers,jobs.size())},{"peak_active_chunks",peakChunks.load()},{"normal_pieces",frames.size()},
        {"pool_seconds",poolSeconds},{"pool_process_cpu_seconds",poolCpu},
        {"sign_fusion_seconds",fusionSeconds},
        {"geometry_stage_seconds",geometrySeconds},{"initial_candidates",geometryReport.measuredCandidates},
        {"refined_candidates",geometryReport.localRefinedCandidates},
        {"geometry_accepted_encounters",geometryReport.localCertifiedEvents},
        {"measured_encounters",geometryReport.localRetainedCertifiedEvents},
        {"pool_process_effective_cores",poolSeconds>0?poolCpu/poolSeconds:0.},
        {"unresolved_pieces",unresolved},{"catalog_seconds",planning},
        {"processing_including_persistence_seconds",elapsed(processing)},{"seconds",elapsed(started)},
        {"scheduling","global parallel geometry once; halo-free spatial sampling/import jobs; final piece sign fusion"}};
}
