#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include <atomic>
#include "vc/core/util/AtomicFile.hpp"
#include <condition_variable>
#include <deque>
#include <thread>
#include <fstream>
#include <iostream>
#include <mutex>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace vc::fiber_tracer {
namespace {
using Json=nlohmann::json;
Json xyz(const cv::Vec3d& p){return Json::array({p[0],p[1],p[2]});}
cv::Vec3d vec(const Json& p){return {p.at(0).get<double>(),p.at(1).get<double>(),p.at(2).get<double>()};}
class GpuChunks {
    using Run=const char*(*)(int,const char*,const unsigned char*,size_t,const float*,size_t);
    void* library_=nullptr;Run run_=nullptr;
    std::mutex mutex_;std::condition_variable changed_;
    std::deque<std::packaged_task<Json(int)>> jobs_;
    std::vector<std::thread> workers_;bool stopping_=false;
    // Host pinning/readiness is much slower than device execution. Its work-ahead
    // budget must not be tied to GPU count. This bounds queued/in-progress host
    // chunks; each GPU still has only one bounded device batch in flight.
    std::atomic<int> prepared_{0};static constexpr int capacity_=128;
public:
    GpuChunks(const std::string& library,int devices) {
        if(devices<1||devices>256)throw std::runtime_error("invalid GPU worker count");
#if defined(__unix__) || defined(__APPLE__)
        library_=dlopen(library.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!library_)throw std::runtime_error(dlerror());
        run_=reinterpret_cast<Run>(dlsym(library_,"vc_chunk_gpu_run"));
        if(!run_){dlclose(library_);throw std::runtime_error("missing GPU chunk ABI");}
#else
        throw std::runtime_error("GPU plugin loading unavailable on this platform");
#endif
        for(int device=0;device<devices;++device)workers_.emplace_back([this,device]{
            for(;;){std::packaged_task<Json(int)> job;
                {std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stopping_||!jobs_.empty();});
                 if(jobs_.empty())return;job=std::move(jobs_.front());jobs_.pop_front();}
                job(device);
            }
        });
    }
    ~GpuChunks(){
        {std::lock_guard lock(mutex_);stopping_=true;}changed_.notify_all();
        for(auto& w:workers_)w.join();
#if defined(__unix__) || defined(__APPLE__)
        dlclose(library_);
#endif
    }
    void acquire(const FiberChunkWait& wait){
        bool acquired=false;
        auto ready=[&]{if(acquired)return true;int n=prepared_.load();
            while(n<capacity_)if(prepared_.compare_exchange_weak(n,n+1)){acquired=true;return true;}return false;};
        try {
            if(!ready()) {if(wait)wait(ready);else while(!ready())std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            if(!acquired)throw std::logic_error("GPU admission wait returned before ready");
        } catch(...) {if(acquired)release();throw;}
    }
    void release(){--prepared_;}
    Json execute(Json meta,std::vector<unsigned char> bytes,std::vector<float> widths,const FiberChunkWait& wait){
        std::packaged_task<Json(int)> job([this,meta=meta.dump(),bytes=std::move(bytes),widths=std::move(widths)](int device){
            auto result=Json::parse(run_(device,meta.c_str(),bytes.data(),bytes.size(),widths.data(),widths.size()));
            if(result.contains("error"))throw std::runtime_error(result["error"].get<std::string>());return result;
        });
        auto future=job.get_future();
        {std::lock_guard lock(mutex_);jobs_.push_back(std::move(job));}changed_.notify_one();
        auto ready=[&]{return future.wait_for(std::chrono::seconds(0))==std::future_status::ready;};
        if(!ready()&&wait)wait(ready);return future.get();
    }
};
GpuChunks& gpuChunks(const FiberChunkTraceConfig& c){
    static std::mutex mutex;
    static std::map<std::pair<std::string,int>,std::unique_ptr<GpuChunks>> pools;
    std::lock_guard lock(mutex);
    auto& pool=pools[{c.gpuLibrary,c.gpuDevices}];
    if(!pool)pool=std::make_unique<GpuChunks>(c.gpuLibrary,c.gpuDevices);
    return *pool;
}

}
std::shared_ptr<void> acquireFiberChunkGpu(const FiberChunkTraceConfig& c,const FiberChunkWait& wait) {
    auto& gpu=gpuChunks(c);
    gpu.acquire(wait);
    return std::shared_ptr<void>(&gpu,[](void* p){static_cast<GpuChunks*>(p)->release();});
}
void runFiberChunkGpu(FiberChunkPaths& output,const FiberPredictionSource& source,
    const vc::lasagna::NormalSampler& normals,const FiberTraceConfig& cfg,
    const FiberChunkTraceConfig& c,double scale,const std::vector<Json>& gpuSeeds,
    const std::shared_ptr<vc::render::PinnedChunkRegion>& pinned,const FiberChunkWait& wait)
{
    if(c.gpuLibrary.empty())throw std::invalid_argument("GPU library required");
    auto& gpu=gpuChunks(c);
    auto* out=&output;const auto& key=out->key;
    const auto start=std::chrono::steady_clock::now();
    auto elapsed=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
        const auto* field=dynamic_cast<const FiberPredictionField*>(&source);
        const auto* normalField=dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(&normals);
        if(!field || !normalField || field->optionCount()!=1 || c.refineSeed || cfg.embeddingWeight>0)
            throw std::runtime_error("GPU fixture requires single native option, no embeddings/seed fit");
        const double spacing=field->storedGridInfo().predictionToBaseScale/scale;
        const double margin=cfg.stepVoxels*std::max(1,cfg.beamLookaheadSteps)+
            (c.onlineRefineIterations?(c.onlineRefineCapBase+c.onlineRefineEpsilonBase)/scale:0);
        std::array<int,3> origin{},shape{};
        for(int a=0;a<3;++a) {
            origin[a]=int(std::floor((out->lo[a]/scale-margin)/spacing));
            shape[a]=int(std::ceil((out->hi[a]/scale+margin)/spacing))-origin[a]+1;
        }
        std::vector<unsigned char> byteValues;std::vector<float> widthValues;
        byteValues.reserve(size_t(shape[0])*shape[1]*shape[2]*6);
        widthValues.reserve(size_t(shape[0])*shape[1]*shape[2]);
        {

        for(int z=0;z<shape[2];++z) {
            std::vector<cv::Vec3f> points;
            for(int y=0;y<shape[1];++y) for(int x=0;x<shape[0];++x)
                points.emplace_back((origin[0]+x)*spacing,(origin[1]+y)*spacing,(origin[2]+z)*spacing);
            vc::lasagna::LasagnaCornerBatch corners;
            corners.values.resize(6);for(auto& channel:corners.values)channel.resize(points.size());
            corners.fractionsXYZ.resize(points.size());corners.valid.resize(points.size());
            auto visitor=[](void* context,size_t i,const cv::Vec3f& fraction,bool valid,
                            std::span<const std::array<uint8_t,8>> values){
                auto& c=*static_cast<vc::lasagna::LasagnaCornerBatch*>(context);
                if(values.size()!=6)throw std::runtime_error("GPU export requires six aligned channels");
                c.valid[i]=valid;c.fractionsXYZ[i]=fraction;
                for(size_t k=0;k<6;++k)c.values[k][i]=values[k];
            };
            if(!field->visitCornerBatchWithNormals(*normalField,points,1,&corners,visitor,0,nullptr,pinned.get()))
                throw std::runtime_error("GPU fixture requires aligned native corner grids");
            std::vector<float> widths;if(pinned)field->sampleWidths(points,widths,*pinned);else field->sampleWidths(points,widths);
            for(size_t i=0;i<points.size();++i) {
                if(!corners.valid[i] || cv::norm(corners.fractionsXYZ[i])>1e-6)
                    throw std::runtime_error("GPU fixture lattice mismatch/outside volume");
                for(int channel=0;channel<6;++channel) byteValues.push_back(corners.values[channel][i][0]);
            }
            widthValues.insert(widthValues.end(),widths.begin(),widths.end());
        }
        }
        Json seeds=Json::array();for(const auto& seed:gpuSeeds) if(!seed.is_null()) seeds.push_back(seed);
        Json meta={{"seeds",seeds},{"origin",origin},{"shape_xyz",shape},{"spacing",spacing},
            {"bounds",{xyz(out->lo/scale),xyz(out->hi/scale)}},{"trace_to_base",scale},
            {"distance",4*(c.coreBase+2*c.haloBase)/scale},{"step",cfg.stepVoxels},
            {"beam",cfg.beamWidth},{"lookahead",cfg.beamLookaheadSteps},{"parent_cap",cfg.lookaheadParentCap},
            {"angle",cfg.coneAngleDegrees},{"angle_step",cfg.coneAngleStepDegrees},
            {"prune_distance",cfg.beamPruneDistanceVoxels},{"width_tolerance",cfg.widthRelativeTolerance},
            {"width_cosine",cfg.widthCosineWeight},{"cutoff",cfg.replayPresenceCutoff},
            {"modulated",cfg.replayPresenceModulated},{"smoothness",cfg.smoothnessWeight},
            {"normal_smoothness",cfg.smoothnessNormalWeight},{"tangent_smoothness",cfg.smoothnessTangentWeight},
            {"free_angle",cfg.smoothnessFreeAngleDegrees},{"cumulative_weight",cfg.cumulativeSmoothnessTangentWeight},
            {"cumulative_steps",cfg.cumulativeSmoothnessSteps},{"mask_normals",normalField->masksInvalidCorners()}};
        meta["online_refine_iterations"]=c.onlineRefineIterations;
        meta["online_refine_window"]=c.onlineRefineWindow;
        meta["online_refine_epsilon"]=c.onlineRefineEpsilonBase/scale;
        meta["online_refine_step"]=c.onlineRefineStepBase/scale;
        meta["online_refine_cap"]=c.onlineRefineCapBase/scale;
        if(!c.benchmarkExport.empty()) {
            std::filesystem::create_directories(c.benchmarkExport);
            std::ofstream bytes(c.benchmarkExport/"fields.u8",std::ios::binary),floats(c.benchmarkExport/"width.f32",std::ios::binary);
            bytes.write(reinterpret_cast<const char*>(byteValues.data()),byteValues.size());
            floats.write(reinterpret_cast<const char*>(widthValues.data()),widthValues.size()*sizeof(float));
            if(!bytes||!floats)throw std::runtime_error("GPU fixture write failed");
            vc::core::util::atomicWriteString(c.benchmarkExport/"gpu_input.json",meta.dump(2)+"\n");
        }
        if(!c.gpuLibrary.empty()) {
            auto result=gpu.execute(meta,std::move(byteValues),std::move(widthValues),wait);
            out->timings["gpu_device"]=result["device"];
            out->timings["gpu_kernel_seconds"]=result["kernel_seconds"];
            out->timings["gpu_seconds"]=result["seconds"];
            const auto& traces=result.at("traces");
            if(traces.size()!=seeds.size()*2)throw std::runtime_error("GPU seed result count mismatch");
            const std::array<std::string,5> reasons{"trace_distance","presence_cutoff","chunk_boundary","no_valid_candidates","invalid_width"};
            for(size_t i=0;i<seeds.size();++i){
                const auto& fw=traces[2*i];const auto& bw=traces[2*i+1];
                std::vector<cv::Vec3d> line;
                for(auto it=bw["points"].rbegin();it!=bw["points"].rend();++it)line.push_back(vec(*it)*scale);
                for(size_t k=1;k<fw["points"].size();++k)line.push_back(vec(fw["points"][k])*scale);
                if(line.size()>=2){out->lines.push_back(std::move(line));out->endReasons.push_back({reasons.at(bw["reason"].get<int>()),reasons.at(fw["reason"].get<int>())});}
            }
            out->timings["total_seconds"]=elapsed();
            // libc++ on macOS does not provide std::osyncstream yet.
            // Keep progress lines intact across concurrent chunk workers.
            static std::mutex outputMutex;
            {
                std::lock_guard lock(outputMutex);
                std::cout<<"chunk-trace GPU built "<<key[0]<<','<<key[1]<<','<<key[2]<<" device="<<result["device"]<<" seeds="<<seeds.size()<<" paths="<<out->lines.size()<<" seconds="<<elapsed()<<std::endl;
            }
        }

}
} // namespace vc::fiber_tracer
