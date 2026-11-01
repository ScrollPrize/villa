// Diagnostic only: same sample positions and shared interpolation math, with
// request/cache handling versus owned ROI arrays or retained decoded chunks.
#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/lasagna/Manifest.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>

using Json = nlohmann::json;
using namespace vc::fiber_tracer;
namespace las = vc::lasagna;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
Json read(const char* path) {std::ifstream f(path);Json j;f>>j;return j;}
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

// Benchmark-only direct byte view: own ChunkResults once, no cache API calls or
// shared_ptr copying per sample. This is not a new production sampler/API.
struct ByteView {
    struct Entry {vc::render::ChunkResult owner;const std::byte* data=nullptr;uint8_t fill=0;};
    std::array<int,3> first{},count{},chunk{}; // XYZ
    std::vector<Entry> entries;
    double spacing=1;
    ByteView(const las::LasagnaChannelBinding& b,const cv::Vec3d& lo,const cv::Vec3d& hi) {
        spacing=b.spacing;
        require(b.chunkSource->dtype()==vc::render::ChunkDtype::UInt8,"raw view requires uint8");
        for(int a=0;a<3;++a){
            chunk[a]=int(b.chunksZYX[2-a]);
            first[a]=int(std::floor(lo[a]/spacing))/chunk[a];
            const int last=int(std::floor(hi[a]/spacing))+1;
            require(lo[a]>=0 && last<int(b.shapeZYX[2-a]),"benchmark ROI outside volume");
            count[a]=last/chunk[a]-first[a]+1;
        }
        entries.resize(size_t(count[0])*count[1]*count[2]);
        for(int z=0;z<count[2];++z)for(int y=0;y<count[1];++y)for(int x=0;x<count[0];++x){
            auto& e=entries[(size_t(z)*count[1]+y)*count[0]+x];
            e.owner=b.chunkSource->getChunkBlocking(0,z+first[2],y+first[1],x+first[0]);
            using S=vc::render::ChunkStatus;
            if(e.owner.status==S::Data){
                require(e.owner.bytes && e.owner.dtype==vc::render::ChunkDtype::UInt8,"bad decoded bytes");
                require(e.owner.byteOffset+size_t(e.owner.shape[0])*e.owner.shape[1]*e.owner.shape[2]<=e.owner.bytes->size(),"short payload");
                e.data=e.owner.bytes->data()+e.owner.byteOffset;
            }else{
                require(e.owner.status==S::Missing || e.owner.status==S::AllFill,"chunk load failed");
                const double f=b.chunkSource->fillValue();
                require(std::isfinite(f)&&f>=0&&f<=255&&std::trunc(f)==f,"bad fill");e.fill=uint8_t(f);
            }
        }
    }
    double voxel(int x,int y,int z) const {
        const int cx=x/chunk[0],cy=y/chunk[1],cz=z/chunk[2];
        const auto& e=entries[(size_t(cz-first[2])*count[1]+cy-first[1])*count[0]+cx-first[0]];
        if(!e.data)return e.fill;
        return std::to_integer<uint8_t>(e.data[(size_t(z%chunk[2])*e.owner.shape[1]+y%chunk[1])*e.owner.shape[2]+x%chunk[0]]);
    }
    void gather(const cv::Vec3d& p,std::array<double,8>& values,cv::Vec3d& f) const {
        // Match the scalar request's coordinate arithmetic, including division.
        const double x=p[0]/spacing,y=p[1]/spacing,z=p[2]/spacing;
        const int ix=int(std::floor(x)),iy=int(std::floor(y)),iz=int(std::floor(z));
        f={x-ix,y-iy,z-iz};
        for(int k=0;k<8;++k)values[k]=voxel(ix+(k&1),iy+((k>>1)&1),iz+((k>>2)&1));
    }
};
struct Corners {std::array<double,8> values;cv::Vec3d fraction;};

void samePred(const std::vector<FiberPredictionSample>& a,const std::vector<FiberPredictionSample>& b) {
    require(a.size()==b.size(),"prediction count mismatch");
    for(size_t i=0;i<a.size();++i){
        require(a[i].options.size()==b[i].options.size(),"prediction options mismatch");
        for(size_t j=0;j<a[i].options.size();++j){
            const auto& x=a[i].options[j];const auto& y=b[i].options[j];
            require(x.valid==y.valid,"prediction validity mismatch");
            if(x.valid)require(x.presence==y.presence && x.direction==y.direction,"prediction numeric mismatch");
        }
    }
}
void sameNormal(const std::vector<las::NormalSampleWithDerivative>& a,const std::vector<las::NormalSampleWithDerivative>& b) {
    require(a.size()==b.size(),"normal count mismatch");
    for(size_t i=0;i<a.size();++i){
        require(a[i].sample.valid==b[i].sample.valid,"normal validity mismatch");
        if(a[i].sample.valid)require(a[i].sample.normal==b[i].sample.normal,"normal numeric mismatch");
    }
}
}

int main(int argc,char** argv) {
    if(argc!=4){std::cerr<<"Usage: bench_fiber_sampling_overhead CONFIG SITES OUTPUT\n";return 2;}
    try {
        const auto cfg=read(argv[1]),sites=read(argv[2]);
        auto manifest=las::LasagnaDataset::openLocation(cfg.at("prediction_manifest")).manifest();
        const double scale=resolveFiberPredictionTraceScales(manifest,cfg.value("scaledown_power",0)).traceToBaseScale;
        manifest.workingToBaseScale=scale;
        FiberPredictionFieldOptions options;options.enableWidth=true;options.maxCachedBytes=size_t(4)<<30;
        FiberPredictionField field{las::LasagnaDataset(manifest),options};
        las::LasagnaNormalSampler normals{las::LasagnaDataset(las::fiberNormalManifest(manifest))};
        Json result={{"trace_to_base",scale},{"sites",sites},{"rows",Json::array()},
            {"cube_request_bytes",sizeof(las::LasagnaCubeRequest)},
            {"chunk_result_bytes",sizeof(vc::render::ChunkResult)}};
        for(size_t site=0;site<sites.size();++site){
            const auto xyz=sites[site].at("base_xyz").get<std::array<double,3>>();
            cv::Vec3d center=cv::Vec3d(xyz[0],xyz[1],xyz[2])/scale;
            auto bindings=field.chunkRegionChannels(center,center).first;
            const auto& presence=bindings[0];
            const double spacing=presence.spacing;
            if(sites[site].value("boundary",false))
                center[0]=std::round(center[0]/(spacing*presence.chunksZYX[2]))*(spacing*presence.chunksZYX[2]);
            // Configurable stored-voxel cube; deliberately spans seam sites.
            const double radius=cfg.value("sampling_radius_stored",16.)*spacing;
            const auto lo=center-cv::Vec3d::all(radius),hi=center+cv::Vec3d::all(radius);
            const auto t=Clock::now();ByteView bytes(presence,lo,hi);const double viewLoad=elapsed(t);
            const auto r=Clock::now();detail::FiberChunkTensor dense(field,normals,lo,hi);const double roiLoad=elapsed(r);
            // Reload after both paths warmed; separate warm assembly from first use.
            const auto w=Clock::now();detail::FiberChunkTensor warmDense(field,normals,lo,hi);const double roiWarm=elapsed(w);
            const auto rt=Clock::now();detail::FiberChunkTensor retained(field,normals,lo,hi,false,true,0,true);
            const double retainedLoad=elapsed(rt);
            const auto rw=Clock::now();detail::FiberChunkTensor warmRetained(field,normals,lo,hi,false,true,0,true);
            const double retainedWarm=elapsed(rw);
            require(retained.ownedVoxelBytes()==0,"retained backend copied voxels");
            auto region=std::make_shared<vc::render::PinnedChunkRegion>();
            las::pinLasagnaRegion(*region,presence,lo,hi);region->seal();
            const std::array<int,3> gridLo{int(std::floor(lo[0]/spacing)),int(std::floor(lo[1]/spacing)),int(std::floor(lo[2]/spacing))};
            const std::array<int,3> gridHi{int(std::ceil(hi[0]/spacing))+1,int(std::ceil(hi[1]/spacing))+1,int(std::ceil(hi[2]/spacing))+1};
            vc::render::PinnedChunkGrid grid(region,presence.chunkSource,0,gridLo,gridHi);
            // Actual channel grids, retained once for stage-only diagnostics.
            const auto normalBindings=normals.regionChannels();
            const std::array<las::LasagnaChannelBinding,7> allBindings{
                bindings[0],bindings[1],bindings[2],normalBindings[0],normalBindings[1],normalBindings[2],bindings[3]};
            auto allRegion=std::make_shared<vc::render::PinnedChunkRegion>();
            for(const auto& b:allBindings)las::pinLasagnaRegion(*allRegion,b,lo,hi);
            allRegion->seal();
            std::array<std::unique_ptr<vc::render::PinnedChunkGrid>,7> allGrids;
            for(size_t c=0;c<7;++c)allGrids[c]=std::make_unique<vc::render::PinnedChunkGrid>(
                allRegion,allBindings[c].chunkSource,0,gridLo,gridHi,c?allGrids[0].get():nullptr);
            const auto& shape=dense.shape();
            const size_t roiBytes=size_t(shape[0]+1)*(shape[1]+1)*(shape[2]+1)*
                (6+(bindings[3].chunkSource->dtype()==vc::render::ChunkDtype::Float32?4:1));
            for(size_t n:{size_t(648),size_t(52488)}){
                std::vector<cv::Vec3d> points(n),directions(n,cv::Vec3d(1,0,0));
                std::vector<cv::Vec3f> floatPoints(n);
                uint64_t state=0x4d595df4d0f33173ULL+site;
                for(size_t i=0;i<n;++i)for(int a=0;a<3;++a){
                    state=state*6364136223846793005ULL+1442695040888963407ULL;
                    const double u=double(state>>11)*(1./9007199254740992.);
                    points[i][a]=float(center[a]+(2*u-1)*(radius-spacing));
                    floatPoints[i][a]=float(points[i][a]);
                }
                if(sites[site].value("boundary",false)){
                    const auto [min,max]=std::minmax_element(points.begin(),points.end(),[](const auto& a,const auto& b){return a[0]<b[0];});
                    require(int((*min)[0]/spacing)/int(presence.chunksZYX[2])!=
                            int((*max)[0]/spacing)/int(presence.chunksZYX[2]),"seam cloud did not cross storage boundary");
                }
                std::vector<double> scalar(n),scalarRef(n);
                std::vector<Corners> corners(n);
                for(size_t i=0;i<n;++i)bytes.gather(points[i],corners[i].values,corners[i].fraction);
                struct Coordinates {std::array<int,3> lo,hi;cv::Vec3d f;};
                auto coordinates=[&](const cv::Vec3d& point){
                    Coordinates c;
                    for(int a=0;a<3;++a){const double v=point[a]/spacing;c.lo[a]=int(std::floor(v));c.hi[a]=c.lo[a]+1;c.f[a]=v-c.lo[a];}
                    return c;
                };
                std::vector<Coordinates> prepared;prepared.reserve(n);
                for(const auto& point:points)prepared.push_back(coordinates(point));
                std::vector<Coordinates> coordinateOutput(n);
                using Address=vc::render::PinnedChunkGrid::CornerAddress;
                std::vector<Address> addresses(n),addressOutput(n);
                std::vector<std::array<double,8>> gatherOutput(n);
                struct ChannelCorners {std::array<std::array<uint8_t,8>,6> bytes;std::array<double,8> width;};
                std::vector<ChannelCorners> channelCorners(n),channelOutput(n);
                auto gatherChannels=[&](size_t i,ChannelCorners& out){
                    auto address=allGrids[0]->prepareCorners(prepared[i].lo,prepared[i].hi);
                    for(size_t c=0;c<7;++c){
                        if(!allGrids[c]->accepts(address))address=allGrids[c]->prepareCorners(prepared[i].lo,prepared[i].hi);
                        if(c<6)out.bytes[c]=allGrids[c]->corners<uint8_t>(address);
                        else out.width=allGrids[c]->corners<double>(address);
                    }
                };
                std::vector<las::LasagnaCompactTensor> tensors(n),tensorOutput(n);
                std::vector<std::optional<cv::Vec3d>> axes(n),axisOutput(n);
                std::array<uint8_t,8> valid;valid.fill(255);
                for(size_t i=0;i<n;++i){
                    addresses[i]=grid.prepareCorners(prepared[i].lo,prepared[i].hi);
                    gatherChannels(i,channelCorners[i]);
                    tensors[i]=las::accumulateLasagnaCompactTensorCorners(channelCorners[i].bytes[1],channelCorners[i].bytes[2],prepared[i].f,valid);
                    axes[i]=las::finalizeLasagnaCompactTensor(tensors[i]);
                }
                std::vector<FiberPredictionSampleOption> materialized(n);
                // Diagnostic recreation of the existing index representation:
                // XYZ -> hypothetical dense z-contiguous index -> XYZ again.
                // Same actual PinnedChunkGrid/buffers/interpolation in both cases.
                auto roundtrip=[&](std::array<int,3>& p){
                    const auto& origin=dense.origin();const auto& shape=dense.shape();
                    size_t index=size_t(p[2]-origin[2])+size_t(p[1]-origin[1])*(shape[2]+1)+
                        size_t(p[0]-origin[0])*(shape[2]+1)*(shape[1]+1);
                    p[2]=origin[2]+int(index%size_t(shape[2]+1));index/=size_t(shape[2]+1);
                    p[1]=origin[1]+int(index%size_t(shape[1]+1));index/=size_t(shape[1]+1);
                    p[0]=origin[0]+int(index);
                };
                std::vector<FiberPredictionSample> prediction,predRef;
                std::vector<las::NormalSampleWithDerivative> normal,normalRef;
                std::vector<float> width,widthRef;
                struct Method {std::string name;std::function<void()> run;std::function<double()> check;};
                auto scalarCheck=[&]{require(scalar==scalarRef,"scalar mismatch");return std::accumulate(scalar.begin(),scalar.end(),0.);};
                auto normalCheck=[&]{sameNormal(normal,normalRef);double sum=0;for(const auto& v:normal)if(v.sample.valid)sum+=v.sample.normal[0]+v.sample.normal[1]+v.sample.normal[2];return sum;};
                auto predCheck=[&]{samePred(prediction,predRef);double sum=0;for(const auto& v:prediction)for(const auto& o:v.options)if(o.valid)sum+=o.presence+o.direction[0];return sum;};
                auto widthCheck=[&]{require(width==widthRef,"width mismatch");return std::accumulate(width.begin(),width.end(),0.);};
                std::vector<Method> methods={
                    {"scalar_cached_requests",[&]{las::LasagnaLocalChunkResolver resolver(presence);for(size_t i=0;i<n;++i){auto q=las::prepareLasagnaCubeRequest(presence,points[i]);resolver.resolve(q);auto v=las::sampleLasagnaChannel(presence,q);require(v.has_value(),"scalar invalid");scalar[i]=*v;}},scalarCheck},
                    {"scalar_retained_chunks",[&]{for(size_t i=0;i<n;++i){std::array<double,8> v;cv::Vec3d f;bytes.gather(points[i],v,f);scalar[i]=las::interpolateLasagnaScalarCorners(v,f);}},scalarCheck},
                    {"scalar_pregathered_math_only",[&]{for(size_t i=0;i<n;++i)scalar[i]=las::interpolateLasagnaScalarCorners(corners[i].values,corners[i].fraction);},scalarCheck},
                    {"normal_cached_batch",[&]{(void)normals.sampleNormalBatchScalarEquivalent(points,1,normal);},normalCheck},
                    {"normal_dense_roi",[&]{(void)dense.las::NormalSampler::sampleNormalBatch(points,false,normal);},normalCheck},
                    {"prediction_cached_batch",[&]{field.sampleBatchScalarEquivalent(points,directions,1,prediction,nullptr);},predCheck},
                    // Qualified base path retains scalar double arithmetic. The
                    // tensor batch override intentionally uses different float math.
                    {"prediction_dense_roi",[&]{dense.FiberPredictionSource::sampleBatch(points,directions,1,prediction);},predCheck},
                    {"width_cached_batch",[&]{field.sampleWidths(floatPoints,width);},widthCheck},
                    {"width_dense_roi",[&]{dense.sampleWidths(floatPoints,width);},widthCheck}
                };
                methods[0].run();scalarRef=scalar;
                methods[3].run();normalRef=normal;
                methods[5].run();predRef=prediction;
                methods[7].run();widthRef=width;
                methods.push_back({"normal_retained_grid",[&]{(void)retained.las::NormalSampler::sampleNormalBatch(points,false,normal);},normalCheck});
                methods.push_back({"prediction_retained_grid",[&]{retained.FiberPredictionSource::sampleBatch(points,directions,1,prediction);},predCheck});
                methods.push_back({"width_retained_grid",[&]{retained.sampleWidths(floatPoints,width);},widthCheck});
                methods.push_back({"scalar_grid_direct_xyz",[&]{for(size_t i=0;i<n;++i){auto c=coordinates(points[i]);scalar[i]=las::interpolateLasagnaScalarCorners(grid.corners<double>(c.lo,c.hi),c.f);}},scalarCheck});
                methods.push_back({"scalar_grid_dense_roundtrip",[&]{for(size_t i=0;i<n;++i){auto c=coordinates(points[i]);roundtrip(c.lo);roundtrip(c.hi);scalar[i]=las::interpolateLasagnaScalarCorners(grid.corners<double>(c.lo,c.hi),c.f);}},scalarCheck});
                methods.push_back({"scalar_grid_prepared_xyz",[&]{for(size_t i=0;i<n;++i){const auto& c=prepared[i];scalar[i]=las::interpolateLasagnaScalarCorners(grid.corners<double>(c.lo,c.hi),c.f);}},scalarCheck});
                const std::vector<cv::Vec3f> floatDirs(n,cv::Vec3f(1,0,0));
                std::vector<las::NormalSample> combinedNormals;
                detail::FiberSamplingProfile inpass;
                auto combinedCheck=[&]{
                    samePred(prediction,predRef);require(width==widthRef,"combined width mismatch");
                    require(combinedNormals.size()==normalRef.size(),"combined normal count");
                    double sum=0;
                    for(size_t i=0;i<n;++i){
                        require(combinedNormals[i].valid==normalRef[i].sample.valid,"combined normal validity");
                        if(combinedNormals[i].valid){require(combinedNormals[i].normal==normalRef[i].sample.normal,"combined normal values");sum+=combinedNormals[i].normal[0];}
                    }
                    return sum+predCheck()+widthCheck();
                };
                auto axisCheck=[&]{double sum=0;for(size_t i=0;i<n;++i){require(axisOutput[i]==axes[i],"axis mismatch");if(axisOutput[i])sum+=(*axisOutput[i])[0];}return sum;};
                if(cfg.value("diagnose_stages",false)){
                    methods.push_back({"channels_separate",[&]{
                        retained.FiberPredictionSource::sampleBatch(points,directions,1,prediction);
                        (void)retained.las::NormalSampler::sampleNormalBatch(points,false,normal);
                        retained.sampleWidths(floatPoints,width);
                    },[&]{return predCheck()+normalCheck()+widthCheck();}});
                    methods.push_back({"channels_combined",[&]{retained.sampleCandidateChannels(floatPoints,floatDirs,true,true,prediction,combinedNormals,width);},combinedCheck});
                    methods.push_back({"channels_combined_inpass",[&]{retained.sampleCandidateChannels(floatPoints,floatDirs,true,true,prediction,combinedNormals,width,&inpass);},combinedCheck});
                    methods.push_back({"coordinates_only",[&]{for(size_t i=0;i<n;++i)coordinateOutput[i]=coordinates(points[i]);},[&]{double sum=0;for(size_t i=0;i<n;++i){const auto& a=coordinateOutput[i];const auto& b=prepared[i];require(a.lo==b.lo&&a.hi==b.hi&&a.f==b.f,"coords mismatch");sum+=a.f[0];}return sum;}});
                    methods.push_back({"chunk_address_only",[&]{for(size_t i=0;i<n;++i)addressOutput[i]=grid.prepareCorners(prepared[i].lo,prepared[i].hi);},[&]{double sum=0;for(size_t i=0;i<n;++i){auto g=grid.corners<double>(addressOutput[i]);require(g==corners[i].values,"address mismatch");sum+=g[0];}return sum;}});
                    methods.push_back({"prepared_gather_only",[&]{for(size_t i=0;i<n;++i)gatherOutput[i]=grid.corners<double>(addresses[i]);},[&]{double sum=0;for(size_t i=0;i<n;++i){require(gatherOutput[i]==corners[i].values,"gather mismatch");sum+=gatherOutput[i][0];}return sum;}});
                    methods.push_back({"address_and_gather7",[&]{for(size_t i=0;i<n;++i)gatherChannels(i,channelOutput[i]);},[&]{double sum=0;for(size_t i=0;i<n;++i){require(channelOutput[i].bytes==channelCorners[i].bytes&&channelOutput[i].width==channelCorners[i].width,"channels mismatch");sum+=channelOutput[i].bytes[0][0];}return sum;}});
                    methods.push_back({"vector_lookup_tensor_accumulate",[&]{for(size_t i=0;i<n;++i)tensorOutput[i]=las::accumulateLasagnaCompactTensorCorners(channelCorners[i].bytes[1],channelCorners[i].bytes[2],prepared[i].f,valid);},[&]{double sum=0;for(size_t i=0;i<n;++i){const auto& a=tensorOutput[i];const auto& b=tensors[i];require(a.t00==b.t00&&a.t01==b.t01&&a.t02==b.t02&&a.t11==b.t11&&a.t12==b.t12&&a.t22==b.t22&&a.hint==b.hint&&a.totalWeight==b.totalWeight,"tensor mismatch");sum+=a.t00;}return sum;}});
                    methods.push_back({"vector_axis_solve_normalize",[&]{for(size_t i=0;i<n;++i)axisOutput[i]=las::finalizeLasagnaCompactTensor(tensors[i]);},axisCheck});
                    methods.push_back({"vector_complete_pregathered",[&]{for(size_t i=0;i<n;++i)axisOutput[i]=las::interpolateLasagnaCompactTensorCorners(channelCorners[i].bytes[1],channelCorners[i].bytes[2],prepared[i].f,valid);},axisCheck});
                    methods.push_back({"prediction_materialize_only",[&]{for(size_t i=0;i<n;++i)materialized[i]=axes[i]?materializeFiberScalarPrediction(scalarRef[i],*axes[i],directions[i]):FiberPredictionSampleOption{};},[&]{double sum=0;for(size_t i=0;i<n;++i){const auto& a=materialized[i];const auto& b=predRef[i].options[0];require(a.valid==b.valid&&a.direction==b.direction&&a.presence==b.presence,"materialize mismatch");sum+=a.presence;}return sum;}});
                }
                if(cfg.value("diagnose_retained",false))std::erase_if(methods,[](const auto& m){
                    return m.name.find("cached")!=std::string::npos || m.name=="scalar_retained_chunks";
                });
                if(cfg.value("profile_inpass",false))std::erase_if(methods,[](const auto& m){return !m.name.starts_with("channels_");});
                for(auto& method:methods){method.run();(void)method.check();}
                for(size_t repeat=0;repeat<5;++repeat)for(size_t m=0;m<methods.size();++m){
                    auto& method=methods[(m+repeat+site)%methods.size()];
                    // More iterations for short batches; each includes normal
                    // sampler's real allocation/destruction, not just its math.
                    const size_t loops=n<1000?32:2;
                    inpass={};
                    const auto start=Clock::now();for(size_t k=0;k<loops;++k)method.run();
                    const double seconds=elapsed(start);const double checksum=method.check();
                    Json row={{"site",site},{"n",n},{"repeat",repeat},{"method",method.name},
                        {"loops",loops},{"seconds",seconds},{"ns_per_point",seconds*1e9/(n*loops)},
                        {"checksum",checksum},{"chunk_view_load_seconds",viewLoad},
                        {"roi_load_seconds",roiLoad},{"roi_warm_load_seconds",roiWarm},
                        {"retained_load_seconds",retainedLoad},{"retained_warm_load_seconds",retainedWarm},
                        {"retained_bytes",retained.retainedVoxelBytes()},{"pointer_table_bytes",retained.pointerTableBytes()},
                        {"roi_bytes",roiBytes},{"spacing_trace",spacing},
                        {"center_base_xyz",{center[0]*scale,center[1]*scale,center[2]*scale}},
                        {"presence_chunks",bytes.entries.size()},{"normal_valid",std::count_if(normalRef.begin(),normalRef.end(),[](const auto& v){return v.sample.valid;})}};
                    if(method.name=="channels_combined_inpass"){
                        require(inpass.calls==loops,"inpass call count mismatch");
                        require(std::accumulate(inpass.nanoseconds.begin(),inpass.nanoseconds.end(),int64_t(0))==inpass.elapsedNanoseconds,"inpass accounting did not close");
                        row["inpass_total_ns"]=inpass.elapsedNanoseconds;
                        row["inpass_calls"]=inpass.calls;
                        for(size_t s=0;s<inpass.names.size();++s){
                            row["inpass_stage_ns"][inpass.names[s]]=inpass.nanoseconds[s];
                            row["inpass_stage_entries"][inpass.names[s]]=inpass.entries[s];
                        }
                    }
                    result["rows"].push_back(row);
                }
                std::cout<<"site "<<site<<" points "<<n<<" parity passed"<<std::endl;
                std::ofstream out(argv[3]);out<<result.dump(2)<<'\n';
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
