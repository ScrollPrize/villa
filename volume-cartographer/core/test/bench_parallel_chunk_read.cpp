// Generic reader-only benchmark. No inference, tracing, sampling or GPU work.
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/Volume.hpp"
#include "../src/render/SamplingWaitProfile.hpp"
#include "../src/render/ChunkReadBatch.hpp"
#include "utils/zarr.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace vc::render;
using Json=nlohmann::json;
using Bytes=std::vector<std::byte>;
using Clock=std::chrono::steady_clock;
struct Usage {
    double user=0,system=0;
    long switches=0,preemptions=0,minorFaults=0,majorFaults=0,inputBlocks=0,outputBlocks=0;
    long long readBytes=-1,writeBytes=-1,readChars=-1;
};
Usage usage(){
    Usage out;
#if defined(__unix__) || defined(__APPLE__)
    rusage r{};getrusage(RUSAGE_SELF,&r);
    out.user=r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6;
    out.system=r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;
    out.switches=r.ru_nvcsw;
    out.preemptions=r.ru_nivcsw;out.minorFaults=r.ru_minflt;out.majorFaults=r.ru_majflt;
    out.inputBlocks=r.ru_inblock;out.outputBlocks=r.ru_oublock;
#endif
#if defined(__linux__)
    std::ifstream io("/proc/self/io");std::string key;long long value;
    while(io>>key>>value){
        if(key=="read_bytes:")out.readBytes=value;
        else if(key=="write_bytes:")out.writeBytes=value;
        else if(key=="rchar:")out.readChars=value;
    }
#endif
    return out;
}
int currentCpu(){
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}
template<class F> Json parallel(size_t workers,size_t jobs,F fn,
                                std::function<void(size_t)> prepare={},bool fixedJobs=false){
    std::atomic<size_t> next{0};std::barrier prepared(int(workers+1)),gate(int(workers+1));
    std::vector<std::thread> threads;std::vector<double> latencies(jobs),waits(workers);
    std::vector<int> prepCpus(workers),startCpus(workers),endCpus(workers);
    std::mutex errorMutex;std::exception_ptr error;
    for(size_t w=0;w<workers;++w)threads.emplace_back([&,w]{
        auto& profile=detail::samplingWaitStats();profile.seconds={};profile.calls={};
        bool failed=false;
        try {if(prepare)for(size_t i=w;i<jobs;i+=workers)prepare(i);}
        catch(...){std::lock_guard lock(errorMutex);if(!error)error=std::current_exception();failed=true;}
        prepCpus[w]=currentCpu();prepared.arrive_and_wait();
        gate.arrive_and_wait();
        startCpus[w]=currentCpu();
        try{for(size_t ordinal=0;!failed;++ordinal){auto i=fixedJobs?w+ordinal*workers:next.fetch_add(1);if(i>=jobs)break;
            auto start=Clock::now();fn(i);
            latencies[i]=std::chrono::duration<double>(Clock::now()-start).count();}}
        catch(...){std::lock_guard lock(errorMutex);if(!error)error=std::current_exception();}
        waits[w]=profile.seconds[1]+profile.seconds[3]+profile.seconds[5];
        endCpus[w]=currentCpu();
        gate.arrive_and_wait();
    });
    prepared.arrive_and_wait(); // buffer first-touch is excluded from measured time
    auto before=usage();auto start=Clock::now();gate.arrive_and_wait();gate.arrive_and_wait();
    const double seconds=std::chrono::duration<double>(Clock::now()-start).count();auto after=usage();
    for(auto& t:threads)t.join();if(error)std::rethrow_exception(error);
    std::sort(latencies.begin(),latencies.end());double wait=0,sum=0;
    for(auto w:waits)wait+=w;for(auto t:latencies)sum+=t;
    Json row={{"seconds",seconds},{"user_seconds",after.user-before.user},{"system_seconds",after.system-before.system},
        {"voluntary_switches",after.switches-before.switches},{"caller_lock_seconds",wait},
        {"involuntary_switches",after.preemptions-before.preemptions},
        {"minor_faults",after.minorFaults-before.minorFaults},{"major_faults",after.majorFaults-before.majorFaults},
        {"input_blocks",after.inputBlocks-before.inputBlocks},{"output_blocks",after.outputBlocks-before.outputBlocks},
        {"worker_cpus_prepared",prepCpus},{"worker_cpus_start",startCpus},{"worker_cpus_end",endCpus},
        {"latency_mean",sum/jobs},{"latency_p50",latencies[jobs/2]},{"latency_p95",latencies[std::min(jobs-1,size_t(jobs*.95))]}};
    row["physical_read_bytes"]=before.readBytes<0||after.readBytes<0?Json(nullptr):Json(after.readBytes-before.readBytes);
    row["physical_write_bytes"]=before.writeBytes<0||after.writeBytes<0?Json(nullptr):Json(after.writeBytes-before.writeBytes);
    row["read_chars"]=before.readChars<0||after.readChars<0?Json(nullptr):Json(after.readChars-before.readChars);
    return row;
}
int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"Usage: bench_parallel_chunk_read CONFIG.json OUTPUT.json\n";return 2;}
    try{
        std::ifstream input(argv[1]);Json cfg;input>>cfg;
        const size_t workers=cfg.at("workers"),pools=cfg.value("pool_workers",workers),batch=cfg.value("batch",size_t(32));
        if(!workers||!pools||!batch)throw std::invalid_argument("workers, pools and batch must be positive");
        const std::string backend=cfg.at("backend"),path=cfg.at("array");
        if(backend!="direct"&&backend!="queued")throw std::invalid_argument("backend");
        auto array=std::make_shared<utils::ZarrArray>(utils::ZarrArray::open(path,vc::buildZarrCodecRegistry(1)));
        if(array->metadata().dtype!=utils::ZarrDtype::uint8)throw std::invalid_argument("benchmark currently requires uint8 input");
        auto options=ChunkCacheService::batchOptions(pools,cfg.value("cache_bytes",size_t(10)<<30));
        if(backend=="queued")options.loaderBackend=ChunkCacheService::Options::LoaderBackend::Queued;
        auto service=configureProcessChunkCacheService(options);
        auto cache=acquireProcessChunkCache(path,array);
        auto channel=acquireProcessZarrChannel(path,array,0);
        const size_t roiChannels=cfg.value("roi_channels",size_t(1));
        if(!roiChannels || (roiChannels>1 && (array->metadata().ndim()!=4 ||
            roiChannels>array->metadata().shape[0] ||
            array->metadata().chunks[0]!=array->metadata().shape[0] || array->is_sharded())))
            throw std::invalid_argument("multi-channel benchmark requires all channels packed in each unsharded physical chunk");
        std::vector<std::shared_ptr<IChunkedArray>> channels{channel};
        for(size_t c=1;c<roiChannels;++c)channels.push_back(acquireProcessZarrChannel(path,array,c));
        std::vector<ChunkKey> keys;std::vector<std::vector<size_t>> physical;
        for(const auto& k:cfg.at("keys_zyx")){
            int z=k.at(0),y=k.at(1),x=k.at(2);keys.push_back({0,z,y,x});
            physical.push_back({size_t(z),size_t(y),size_t(x)});
            if(array->metadata().shape.size()==4)physical.back().insert(physical.back().begin(),0);
        }
        if(keys.empty())throw std::invalid_argument("empty workload");
        const size_t n=keys.size(),jobs=(n+batch-1)/batch;
        std::vector<Bytes> reference(n),encoded(n);
        // Warm source pages, produce exact byte oracle; excluded from every timing.
        parallel(16,n,[&](size_t i){auto e=array->read_chunk_encoded(physical[i]);
            if(!e)throw std::runtime_error("benchmark key missing");encoded[i]=std::move(*e);
            reference[i]=array->decode_chunk_payload(encoded[i]);});
        size_t rawBytes=0,encodedBytes=0;for(size_t i=0;i<n;++i){rawBytes+=reference[i].size();encodedBytes+=encoded[i].size();}
        Json report={{"config",cfg},{"raw_bytes",rawBytes},{"encoded_bytes",encodedBytes},{"rows",Json::array()}};
        std::vector<std::string> modes=cfg.value("modes",std::vector<std::string>{"raw","decode","copy","cold","hot"});
        const auto shape=channel->chunkShape(0);
        const int roiChunks=cfg.value("roi_chunks",1);
        if(roiChunks<1)throw std::invalid_argument("roi_chunks must be positive");
        const std::array<int,3> roiShape{shape[0]*roiChunks,shape[1]*roiChunks,shape[2]*roiChunks};
        const bool hasRoi=std::any_of(modes.begin(),modes.end(),[](const auto& m){return m=="roi"||m=="roi-hot"||m=="roi-hot-reuse";});
        const size_t channelChunkBytes=size_t(shape[0])*shape[1]*shape[2];
        report["roi_output_bytes"]=n*roiChannels*size_t(roiShape[0])*roiShape[1]*roiShape[2];
        report["roi_nominal_decoded_bytes"]=n*size_t(roiChunks)*roiChunks*roiChunks*array->metadata().chunk_byte_size();
        std::vector<Array3D<uint8_t>> roiOracle(hasRoi?n*roiChannels:0);
        if(hasRoi)parallel(16,n,[&](size_t i){
            for(size_t c=0;c<roiChannels;++c)roiOracle[i*roiChannels+c].resize({size_t(roiShape[0]),size_t(roiShape[1]),size_t(roiShape[2])});
            for(int dz=0;dz<roiChunks;++dz)for(int dy=0;dy<roiChunks;++dy)for(int dx=0;dx<roiChunks;++dx){
                auto index=physical[i];const auto start=index.size()-3;
                index[start]+=dz;index[start+1]+=dy;index[start+2]+=dx;
                for(size_t axis=0;axis<3;++axis)
                    if((index[start+axis]+1)*size_t(shape[axis])>size_t(channel->shape(0)[axis]))
                        throw std::invalid_argument("ROI fixture must be fully inside volume");
                auto decoded=array->read_chunk(index);
                if(!decoded)throw std::runtime_error("ROI fixture chunk missing");
                if(decoded->size()<roiChannels*channelChunkBytes)throw std::runtime_error("ROI channel payload truncated");
                for(size_t c=0;c<roiChannels;++c)
                for(int z=0;z<shape[0];++z)for(int y=0;y<shape[1];++y)for(int x=0;x<shape[2];++x)
                    roiOracle[i*roiChannels+c](dz*shape[0]+z,dy*shape[1]+y,dx*shape[2]+x)=std::to_integer<uint8_t>((*decoded)[c*channelChunkBytes+(size_t(z)*shape[1]+y)*shape[2]+x]);
            }
        });
        for(int repeat=0;repeat<cfg.value("repeats",3);++repeat)for(const auto& mode:modes){
            std::vector<ChunkResult> results(n);std::vector<Bytes> owned(n);
            const bool roi=mode=="roi"||mode=="roi-hot"||mode=="roi-hot-reuse";
            std::vector<Array3D<uint8_t>> rois(roi?n*roiChannels:0);
            if(mode=="cold")cache->invalidate();
            if(mode=="roi")cache->invalidate();
            if(mode=="hot"||mode=="roi-hot"||mode=="roi-hot-reuse")for(size_t i=0;i<n;i+=batch){
                std::vector<ChunkKey> group;
                const int extent=roi?roiChunks:1;
                for(size_t j=i;j<std::min(n,i+batch);++j)
                    for(int z=0;z<extent;++z)for(int y=0;y<extent;++y)for(int x=0;x<extent;++x)
                        group.push_back({0,keys[j].iz+z,keys[j].iy+y,keys[j].ix+x});
                detail::ChunkReadBatch::read(*cache,group);
            }
            std::function<void(size_t)> prepare;
            if(mode=="copy-reuse"||mode=="decode-reuse"||mode=="roi-hot-reuse")prepare=[&](size_t job){
                for(size_t i=job*batch;i<std::min(n,(job+1)*batch);++i){
                    if(roi){
                        for(size_t c=0;c<roiChannels;++c){
                            rois[i*roiChannels+c].resize({size_t(roiShape[0]),size_t(roiShape[1]),size_t(roiShape[2])});
                            rois[i*roiChannels+c].fill(0);
                        }
                    }else owned[i].assign(reference[i].size(),std::byte{});
                }
            };
            const auto before=cache->stats();auto qbefore=service->loaderStats();
            const auto reuseBefore=service->decodedByteBudget()->stats().reusedBuffers;
            const auto poolBefore=service->decodedByteBudget()->stats();
            auto row=parallel(workers,jobs,[&](size_t job){
                const size_t begin=job*batch,end=std::min(n,begin+batch);
                if(mode=="cold"||mode=="hot"){
                    std::vector<ChunkKey> group(keys.begin()+begin,keys.begin()+end);
                    auto loaded=detail::ChunkReadBatch::read(*cache,group);
                    std::move(loaded.begin(),loaded.end(),results.begin()+begin);
                }else for(size_t i=begin;i<end;++i){
                    if(roi){
                        for(size_t c=0;c<roiChannels;++c){
                            auto& out=rois[i*roiChannels+c];
                            if(mode!="roi-hot-reuse")out.resize({size_t(roiShape[0]),size_t(roiShape[1]),size_t(roiShape[2])});
                            Volume::readZYX(out,{keys[i].iz*shape[0],keys[i].iy*shape[1],keys[i].ix*shape[2]},*channels[c]);
                        }
                    }
                    else if(mode=="raw"){auto data=array->read_chunk(physical[i]);if(!data)throw std::runtime_error("raw key missing");owned[i]=std::move(*data);}
                    else if(mode=="decode")owned[i]=array->decode_chunk_payload(encoded[i]);
                    else if(mode=="decode-reuse")array->decode_chunk_payload_into(encoded[i],owned[i]);
                    else if(mode=="copy")owned[i]=reference[i];
                    else if(mode=="copy-reuse")std::memcpy(owned[i].data(),reference[i].data(),reference[i].size());
                    else throw std::invalid_argument("unknown mode");
                }
            },prepare,cfg.value("fixed_jobs",false)||bool(prepare));
            // Byte-for-byte validation is outside the measured interval.
            for(size_t i=0;i<n;++i){
                if(mode=="cold"||mode=="hot"){
                    const auto& r=results[i];
                    if(r.status==ChunkStatus::AllFill){if(!std::all_of(reference[i].begin(),reference[i].end(),[](auto v){return v==std::byte{};}))throw std::runtime_error("wrong fill");}
                    else if(r.status!=ChunkStatus::Data||!r.bytes||*r.bytes!=reference[i])throw std::runtime_error("cache value mismatch at "+std::to_string(i)+" status="+std::to_string(int(r.status))+" error="+r.error+" actual="+std::to_string(r.bytes?r.bytes->size():0)+" expected="+std::to_string(reference[i].size()));
                }else if(roi){
                    for(size_t c=0;c<roiChannels;++c){
                        const auto& actual=rois[i*roiChannels+c];
                        const auto& expected=roiOracle[i*roiChannels+c];
                        if(actual.size()!=expected.size() || !std::equal(actual.data(),actual.data()+actual.size(),expected.data()))
                            throw std::runtime_error("ROI mismatch, channel "+std::to_string(c));
                    }
                }else if(owned[i]!=reference[i])throw std::runtime_error("control mismatch");
            }
            const auto after=cache->stats();auto q=service->loaderStats();
            row.update({{"mode",mode},{"repeat",repeat},{"logical_bytes",roi?n*roiChannels*size_t(roiShape[0])*roiShape[1]*roiShape[2]:rawBytes},{"decode_attempts",after.decodeAttempts-before.decodeAttempts},
                {"reused_decode_buffers",service->decodedByteBudget()->stats().reusedBuffers-reuseBefore},
                {"buffer_requests",service->decodedByteBudget()->stats().bufferRequests-poolBefore.bufferRequests},
                {"recycle_rejected",service->decodedByteBudget()->stats().recycleRejected-poolBefore.recycleRejected},
                {"trimmed_buffers",service->decodedByteBudget()->stats().trimmedBuffers-poolBefore.trimmedBuffers},
                {"trimmed_bytes",service->decodedByteBudget()->stats().trimmedBytes-poolBefore.trimmedBytes},
                {"reuse_enforcement_busy",service->decodedByteBudget()->stats().reuseEnforcementBusy-poolBefore.reuseEnforcementBusy},
                {"queue_polls",q.polls-qbefore.polls},{"queue_queried_keys",q.queriedKeys-qbefore.queriedKeys},
                {"queue_requests",q.requests-qbefore.requests},{"queue_completed",q.completed-qbefore.completed}});
            report["rows"].push_back(row);std::cout<<row.dump()<<std::endl;
            std::ofstream output(argv[2]);output<<report.dump(2)<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
