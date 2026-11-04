// Native, serial preparation profiler. No GPU use; emitted grid checksum and
// stage CPU/wall times make before/after runs independently auditable.
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/lasagna/Manifest.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>

using Json = nlohmann::json;
using namespace vc::fiber_tracer;
namespace {
double cpuSeconds()
{
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0)
        return double(t.tv_sec) + double(t.tv_nsec)*1e-9;
#endif
    return -1;
}
struct Timer {
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    double cpu=cpuSeconds();
    Json finish() const {
        return {{"wall",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
                {"thread_cpu",cpu>=0?cpuSeconds()-cpu:-1}};
    }
};
Json read(const char* path){std::ifstream file(path);Json j;file>>j;return j;}
}
int main(int argc,char** argv)
{
    if(argc!=4 && argc!=5){std::cerr<<"Usage: bench_fiber_seed_grid CONFIG INPUTS OUTPUT [point]\n";return 2;}
    try {
        const bool point=argc==5 && std::string(argv[4])=="point";
        if(argc==5 && !point)throw std::invalid_argument("unknown profiling mode");
        const auto cfg=read(argv[1]),inputs=read(argv[2]);
        const auto opened=vc::lasagna::LasagnaDataset::openLocation(cfg.at("prediction_manifest"));
        auto manifest=opened.manifest();
        const auto scales=resolveFiberPredictionTraceScales(manifest,cfg.value("scaledown_power",0));
        const double scale=scales.traceToBaseScale;manifest.workingToBaseScale=scale;
        vc::lasagna::LasagnaDataset dataset(manifest);
        FiberPredictionFieldOptions options;options.enableWidth=true;options.maxCachedBytes=cfg.value("cache_bytes",size_t(1)<<30);
        FiberPredictionField field(dataset,options);
        vc::lasagna::LasagnaNormalSampler normals{vc::lasagna::LasagnaDataset(vc::lasagna::fiberNormalManifest(manifest))};
        const auto& c=cfg.at("chunks");
        const double core=c.value("core_base",512.),halo=c.value("halo_base",64.),grid=c.value("grid_base",4.);
        const double threshold=c.value("seed_presence",.8);
        const double margin=cfg.at("trace").value("step_voxels",8.)*
            std::max(1,cfg.at("trace").value("beam_lookahead_steps",2))+
            (c.value("online_refine_iterations",0)?
                (c.value("online_refine_cap_base",4.)+c.value("online_refine_epsilon_base",.25))/scale:0.);
        const int side=std::lround((core+2*halo)/grid);
        Json results=Json::array();
        for(int repeat=0;repeat<3;++repeat)for(const auto& item:inputs){
            const auto key=item.at("key").get<std::array<int,3>>();
            const cv::Vec3d lo(key[0]*core-halo,key[1]*core-halo,key[2]*core-halo),hi=lo+cv::Vec3d::all(core+2*halo);
            Json row={{"key",key},{"repeat",repeat}};
            Timer loading;
            detail::FiberChunkTensor tensor(field,normals,lo/scale-cv::Vec3d::all(margin),hi/scale+cv::Vec3d::all(margin),!point);
            row["load"]=loading.finish();
            Timer scanning;
            std::vector<float> presence;
            if(point){
                presence.resize(size_t(side)*side*side);
                for(int z=0;z<side;++z)for(int y=0;y<side;++y)for(int x=0;x<side;++x)
                    presence[(size_t(z)*side+y)*side+x]=tensor.seedPresence((lo+cv::Vec3d(x,y,z)*grid)/scale,threshold);
            }else row["grid"]=tensor.seedGrid(lo,scale,grid,side,threshold,presence);
            row["scan"]=scanning.finish();
            uint64_t hash=1469598103934665603ULL;size_t eligible=0;
            for(float v:presence){uint32_t bits;std::memcpy(&bits,&v,4);hash=(hash^bits)*1099511628211ULL;eligible+=v>=threshold;}
            row["grid_hash"]=hash;row["eligible"]=eligible;
            Timer packing;
            std::vector<unsigned char> bytes;std::vector<float> widths;
            if(eligible)tensor.packGpu(bytes,widths);
            row["pack"]=packing.finish();
            results.push_back(row);std::cout<<row.dump()<<std::endl;
        }
        std::ofstream output(argv[3]);output<<results.dump(2)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
