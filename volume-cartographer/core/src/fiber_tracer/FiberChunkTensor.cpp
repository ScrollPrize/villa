#include "vc/fiber_tracer/detail/FiberChunkTensor.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "../render/SamplingWaitProfile.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <chrono>
#include <ctime>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace vc::fiber_tracer::detail {
namespace {
std::array<uint8_t,8> allValid(){std::array<uint8_t,8> v;v.fill(255);return v;}
// ROI loader uses OpenMP internally. Chunk concurrency is owned by FiberCaseQueue,
// not independent nested teams. Restore this worker's previous OpenMP setting.
struct SerialRoi {
#ifdef _OPENMP
    int threads=omp_get_max_threads();
    SerialRoi(){omp_set_num_threads(1);}
    ~SerialRoi(){omp_set_num_threads(threads);}
#endif
};

double threadCpuSeconds()
{
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec value{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&value)==0)
        return double(value.tv_sec)+double(value.tv_nsec)*1e-9;
#endif
    return -1;
}

struct StageTimer {
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    double cpu=threadCpuSeconds();
    std::array<double,12> waits=vc::render::detail::samplingWaitProfileEnabled()?
        vc::render::detail::samplingWaitStats().seconds:std::array<double,12>{};
    nlohmann::json finish() const {
        nlohmann::json result={{"wall",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
                {"thread_cpu",cpu>=0?threadCpuSeconds()-cpu:-1}};
        if(vc::render::detail::samplingWaitProfileEnabled()) {
            constexpr const char* names[]={"blocking_total","blocking_lock","prefetch_total","prefetch_lock",
                "cached_total","cached_lock","corner_prepare_layout","corner_pin","corner_gather",
                "payload_wait","entry_budget","shared_budget"};
            const auto& current=vc::render::detail::samplingWaitStats().seconds;
            for(size_t i=0;i<waits.size();++i)result["cache_waits"][names[i]]=current[i]-waits[i];
        }
        return result;
    }
};

FiberPredictionSampleOption batchDirection(float presence,
    const std::array<uint8_t,8>& nxValues,const std::array<uint8_t,8>& nyValues,
    const cv::Vec3f& fraction,const cv::Vec3f& reference)
{
    const vc::lasagna::LasagnaCornerSample nx{nxValues,fraction,true},ny{nyValues,fraction,true};
    auto direction=vc::lasagna::interpolateLasagnaCompactAxisCorners(nx,ny,reference);
    if(direction.dot(reference)<0)direction*=-1.f;
    return {direction,presence,direction.dot(direction)>1.e-12f};
}
}

FiberChunkTensor::FiberChunkTensor(const FiberPredictionField& field,
    const vc::lasagna::LasagnaNormalSampler& normals,const cv::Vec3d& lo,const cv::Vec3d& hi,bool presenceOnly,
    bool channelHints,std::size_t workerCacheBytes)
{
    const auto [prediction,multiplier]=field.chunkRegionChannels(lo,hi);
    const auto normal=normals.regionChannels();
    spacing_=prediction[0].spacing;widthMultiplier_=multiplier;maskNormals_=normals.masksInvalidCorners();
    for(size_t channel=0;channel<3;++channel)
        if(!vc::lasagna::sameLasagnaSamplingGrid(prediction[0],prediction[channel]))
            throw std::invalid_argument("dense chunk prediction channels must share one lattice");
    // FiberPredictionField already requires width's shape/spacing to match,
    // but its storage chunk dimensions may differ. ROI loading handles those.
    for(const auto& binding:normal)
        if(!vc::lasagna::sameLasagnaSamplingGrid(prediction[0],binding))
            throw std::invalid_argument("dense chunk normals must share prediction lattice");
    for(int a=0;a<3;++a){
        const double start=std::floor(lo[a]/spacing_),end=std::ceil(hi[a]/spacing_);
        if(!std::isfinite(start)||!std::isfinite(end)||start<INT_MIN||end>=INT_MAX||end<start)
            throw std::invalid_argument("invalid dense chunk bounds");
        origin_[a]=int(start);shape_[a]=int(end-start)+1;
        volumeShape_[a]=int(prediction[0].shapeZYX[2-a]);
    }
    // GPU packing samples widths at the final lattice point too. The historical
    // interpolator reads its upper corner even at zero fractional weight, so
    // retain that one-voxel interpolation apron without enlarging the GPU grid.
    const std::array<size_t,3> dims{size_t(shape_[2])+1,size_t(shape_[1])+1,size_t(shape_[0])+1};
    size_t count=1;
    for(auto n:dims){if(n>size_t(1)<<28 || count>(size_t(1)<<28)/n)throw std::invalid_argument("dense chunk tensor too large");count*=n;}
    const std::array<int,3> offset{origin_[2],origin_[1],origin_[0]};
    for(size_t c=0;c<6;++c)pendingBindings_[c]=c<3?prediction[c]:normal[c-3];
    pendingBindings_[6]=prediction[3];
    if(workerCacheBytes){
        auto partition=vc::render::threadLocalDecodedPartition(vc::render::processChunkCacheService(),workerCacheBytes);
        for(auto& binding:pendingBindings_)binding.chunkSource=vc::render::rebindChunkCache(binding.chunkSource,partition);
    }
    widthIsFloat_=prediction[3].chunkSource->dtype()==vc::render::ChunkDtype::Float32;
    SerialRoi serial;
    channels_[0].resize(dims);
    vc::render::ChunkRequestContext request;
    if(channelHints){
        std::vector<std::shared_ptr<vc::render::IChunkedArray>> channels;
        for(const auto& binding:pendingBindings_)channels.push_back(binding.chunkSource);
        request.channelHints=std::make_shared<vc::render::ChunkChannelHints>(std::move(channels));
    }
    Volume::readZYX(channels_[0],offset,*pendingBindings_[0].chunkSource,0,request);
    if(!presenceOnly)loadCompanions();
}

void FiberChunkTensor::loadCompanions()
{
    if(companionsLoaded_)return;
    SerialRoi serial;
    const auto dims=channels_[0].shape();
    const std::array<int,3> offset{origin_[2],origin_[1],origin_[0]};
    for(size_t c=1;c<6;++c){
        channels_[c].resize(dims);
        Volume::readZYX(channels_[c],offset,*pendingBindings_[c].chunkSource);
    }
    if(widthIsFloat_){floatWidth_.resize(dims);Volume::readZYX(floatWidth_,offset,*pendingBindings_[6].chunkSource);}
    else {byteWidth_.resize(dims);Volume::readZYX(byteWidth_,offset,*pendingBindings_[6].chunkSource);}
    companionsLoaded_=true;
    pendingBindings_={};
}

void FiberChunkTensor::requireCompanions() const
{
    if(!companionsLoaded_)
        throw std::logic_error("presence-only tensor must finish an eligible seedGrid before sampling companions");
}

bool FiberChunkTensor::cube(const cv::Vec3d& p,bool scalar,std::array<size_t,8>& indices,cv::Vec3d& f) const
{
    std::array<int,3> low{},high{};
    for(int a=0;a<3;++a){
        if(!std::isfinite(p[a])||p[a]<0||(scalar?p[a]>volumeShape_[a]-1:p[a]>=volumeShape_[a]))return false;
        low[a]=int(std::floor(p[a]));high[a]=std::min(low[a]+1,volumeShape_[a]-1);
        f[a]=p[a]-low[a];low[a]-=origin_[a];high[a]-=origin_[a];
        if(low[a]<0||high[a]>shape_[a])throw std::out_of_range("dense chunk interpolation exceeded loaded region: axis="+std::to_string(a)+" lattice="+std::to_string(p[a])+" origin="+std::to_string(origin_[a])+" shape="+std::to_string(shape_[a])+" spacing="+std::to_string(spacing_));
    }
    for(size_t k=0;k<8;++k){
        const auto x=(k&1)?high[0]:low[0],y=(k&2)?high[1]:low[1],z=(k&4)?high[2]:low[2];
        indices[k]=size_t(z)+size_t(y)*(shape_[2]+1)+size_t(x)*(shape_[2]+1)*(shape_[1]+1);
    }
    return true;
}

std::array<uint8_t,8> FiberChunkTensor::bytes(size_t c,const std::array<size_t,8>& indices) const
{
    std::array<uint8_t,8> out;for(size_t k=0;k<8;++k)out[k]=channels_[c].data()[indices[k]];return out;
}

FiberPredictionSample FiberChunkTensor::sample(const cv::Vec3d& p,const cv::Vec3d& ref) const
{
    requireCompanions();
    std::array<size_t,8> indices;cv::Vec3d f;
    FiberPredictionSample result;
    if(!cube(p/spacing_,true,indices,f)){result.options.push_back({});return result;}
    const auto direction=vc::lasagna::interpolateLasagnaCompactTensorCorners(bytes(1,indices),bytes(2,indices),f,allValid());
    if(!direction){result.options.push_back({});return result;}
    std::array<double,8> presence;for(size_t k=0;k<8;++k)presence[k]=channels_[0].data()[indices[k]];
    result.options.push_back(materializeFiberScalarPrediction(vc::lasagna::interpolateLasagnaScalarCorners(presence,f),*direction,ref));
    return result;
}

FiberPredictionSampleOption FiberChunkTensor::batchOption(const cv::Vec3d& p,const cv::Vec3d& ref,double threshold) const
{
    // Match the existing double -> float point conversion, then float division.
    const cv::Vec3f fp=cv::Vec3f(p)/float(spacing_);std::array<size_t,8> indices;cv::Vec3d fd;
    if(!cube(cv::Vec3d(fp),false,indices,fd))return {};
    const cv::Vec3f f(fd),reference(ref);
    const auto weights=vc::lasagna::lasagnaCornerWeights(f);
    const float presence=std::clamp(vc::lasagna::interpolateLasagnaCorners(bytes(0,indices),weights)/255.0f,0.0f,1.0f);
    if(presence<threshold)return {};
    return batchDirection(presence,bytes(1,indices),bytes(2,indices),f,reference);
}

float FiberChunkTensor::seedPresence(const cv::Vec3d& p,double threshold) const
{
    requireCompanions();
    const auto option=batchOption(p,{1,0,0},threshold);return option.valid?option.presence:0;
}

nlohmann::json FiberChunkTensor::seedGrid(const cv::Vec3d& lo,double scale,double grid,
    int side,double threshold,std::vector<float>& output)
{
    if(side<1 || side>1024 || !std::isfinite(scale) || scale<=0 ||
       !std::isfinite(grid) || grid<=0 || !std::isfinite(threshold) || threshold<0 || threshold>1)
        throw std::invalid_argument("invalid regular seed grid");
    struct AxisPoint {size_t low=0,high=0,phase=0;float fraction=0;bool valid=false;};
    struct Run {int begin,end;};
    std::array<std::vector<AxisPoint>,3> axes;
    std::array<std::vector<Run>,3> runs;
    std::array<std::vector<float>,3> phases;
    const std::array<size_t,3> stride{size_t(shape_[2]+1)*(shape_[1]+1),size_t(shape_[2]+1),1};
    nlohmann::json profile;
    StageTimer setup;
    for(int a=0;a<3;++a){
        if(!std::isfinite(lo[a]))throw std::invalid_argument("nonfinite grid origin");
        axes[a].resize(side);
        for(int i=0;i<side;++i){
            auto& sample=axes[a][i];
            // Match point(base)->point(trace)->float->float lattice exactly;
            // do not replace this expression by an incremental recurrence.
            // OpenCV Vec division is multiplication by a reciprocal, with a
            // double reciprocal for trace scale and float for lattice spacing.
            const float p=static_cast<float>((lo[a]+double(i)*grid)*(1./scale))*(1.f/static_cast<float>(spacing_));
            if(!std::isfinite(p)||p<0||double(p)>=volumeShape_[a])continue;
            const int low=int(std::floor(double(p))),high=std::min(low+1,volumeShape_[a]-1);
            if(low<origin_[a]||high-origin_[a]>shape_[a])
                throw std::out_of_range("seed grid exceeded loaded tensor");
            sample.low=size_t(low-origin_[a])*stride[a];
            sample.high=size_t(high-origin_[a])*stride[a];
            sample.fraction=static_cast<float>(double(p)-low);sample.valid=true;
            auto found=std::find(phases[a].begin(),phases[a].end(),sample.fraction);
            sample.phase=size_t(found-phases[a].begin());
            if(found==phases[a].end())phases[a].push_back(sample.fraction);
        }
        for(int begin=0;begin<side;){
            int end=begin+1;const auto& first=axes[a][begin];
            while(end<side && axes[a][end].valid==first.valid &&
                  axes[a][end].low==first.low && axes[a][end].high==first.high)++end;
            if(first.valid)runs[a].push_back({begin,end});
            begin=end;
        }
    }
    const size_t phaseCount=phases[0].size()*phases[1].size()*phases[2].size();
    std::vector<std::array<float,8>> weights;
    if(phaseCount<=4096){
        weights.reserve(phaseCount);
        for(float z:phases[2])for(float y:phases[1])for(float x:phases[0])
            weights.push_back(vc::lasagna::lasagnaCornerWeights({x,y,z}));
    }
    auto corners=[&](const AxisPoint& x,const AxisPoint& y,const AxisPoint& z){
        std::array<size_t,8> indices;
        for(size_t k=0;k<8;++k)indices[k]=((k&1)?x.high:x.low)+((k&2)?y.high:y.low)+((k&4)?z.high:z.low);
        return indices;
    };
    output.assign(size_t(side)*side*side,0);
    profile["setup"]=setup.finish();
    StageTimer presenceTimer;
    std::vector<size_t> eligible;
    size_t cells=0,skipped=0,interpolated=0;
    for(const auto& zr:runs[2])for(const auto& yr:runs[1])for(const auto& xr:runs[0]){
        ++cells;
        const auto indices=corners(axes[0][xr.begin],axes[1][yr.begin],axes[2][zr.begin]);
        const auto values=bytes(0,indices);
        const float maximum=*std::max_element(values.begin(),values.end());
        // Positive trilinear weights sum to one in exact arithmetic. Products,
        // eight ordered float FMAs and normalization have a much smaller error
        // than 64 eps absolute normalized presence. Slack is ONLY a conservative
        // culling bound; final acceptance uses unchanged arithmetic/threshold.
        const double upper=double(maximum)/255.+64.*std::numeric_limits<float>::epsilon();
        if(upper<threshold){++skipped;continue;}
        for(int z=zr.begin;z<zr.end;++z)for(int y=yr.begin;y<yr.end;++y)for(int x=xr.begin;x<xr.end;++x){
            const auto& ax=axes[0][x];const auto& ay=axes[1][y];const auto& az=axes[2][z];
            std::array<float,8> w;
            if(weights.empty())w=vc::lasagna::lasagnaCornerWeights({ax.fraction,ay.fraction,az.fraction});
            else w=weights[(az.phase*phases[1].size()+ay.phase)*phases[0].size()+ax.phase];
            const float value=std::clamp(vc::lasagna::interpolateLasagnaCorners(values,w)/255.f,0.f,1.f);
            ++interpolated;
            if(value<threshold)continue;
            const size_t index=(size_t(z)*side+y)*side+x;
            output[index]=value;eligible.push_back(index);
        }
    }
    profile["presence"]=presenceTimer.finish();
    profile["grid_points"]=output.size();profile["native_cells"]=cells;
    profile["skipped_cells"]=skipped;profile["interpolated_points"]=interpolated;
    profile["presence_eligible_points"]=eligible.size();profile["phase_count"]=phaseCount;
    StageTimer loading;
    // At threshold zero, even padded/invalid zero-valued grid sites can enter
    // the existing NMS. Its later scalar validity check still needs safe arrays.
    if(!eligible.empty() || threshold==0)loadCompanions();
    profile["companions"]=loading.finish();
    StageTimer directions;
    size_t valid=0;
    for(size_t index:eligible){
        const auto& ax=axes[0][index%side];
        const auto& ay=axes[1][(index/side)%side];
        const auto& az=axes[2][index/(size_t(side)*side)];
        const auto indices=corners(ax,ay,az);
        const auto result=batchDirection(output[index],bytes(1,indices),bytes(2,indices),
            {ax.fraction,ay.fraction,az.fraction},{1,0,0});
        if(!result.valid)output[index]=0;
        else ++valid;
    }
    profile["directions"]=directions.finish();profile["valid_direction_points"]=valid;
    return profile;
}

void FiberChunkTensor::sampleBatch(const std::vector<cv::Vec3d>& points,const std::vector<cv::Vec3d>& dirs,
    int,std::vector<FiberPredictionSample>& out) const
{
    requireCompanions();
    if(points.size()!=dirs.size())throw std::invalid_argument("dense point/direction count mismatch");
    out.resize(points.size());for(size_t i=0;i<points.size();++i){out[i].options.clear();out[i].options.push_back(batchOption(points[i],dirs[i],-1));}
}

float FiberChunkTensor::width(const cv::Vec3f& p) const
{
    std::array<size_t,8> indices;cv::Vec3d f;
    if(!cube(cv::Vec3d(p)/spacing_,true,indices,f))return 0;
    std::array<double,8> values;
    for(size_t k=0;k<8;++k)values[k]=widthIsFloat_?double(floatWidth_.data()[indices[k]]):double(byteWidth_.data()[indices[k]]);
    const auto value=vc::lasagna::interpolateLasagnaMaskedScalarCorners(values,f,bytes(0,indices));
    if(!value||!std::isfinite(*value)||!std::isfinite(*value * *value))return 0;
    // Existing vector materialization rounds before unit conversion.
    return static_cast<float>(static_cast<float>(*value)*widthMultiplier_);
}

void FiberChunkTensor::sampleWidths(const std::vector<cv::Vec3f>& points,std::vector<float>& out) const
{
    requireCompanions();
    out.resize(points.size());for(size_t i=0;i<points.size();++i)out[i]=width(points[i]);
}

vc::lasagna::NormalSample FiberChunkTensor::sampleNormal(const cv::Vec3d& p) const
{
    requireCompanions();
    std::array<size_t,8> indices;cv::Vec3d f;
    if(!cube(p/spacing_,true,indices,f))return {{},false,"outside normal volume"};
    const auto mask=bytes(3,indices);std::array<double,8> grad;
    for(size_t k=0;k<8;++k)grad[k]=mask[k];
    if(vc::lasagna::interpolateLasagnaScalarCorners(grad,f)<=0)return {{},false,"zero normal magnitude"};
    const auto normal=vc::lasagna::interpolateLasagnaCompactTensorCorners(bytes(4,indices),bytes(5,indices),f,maskNormals_?mask:allValid());
    return normal?vc::lasagna::NormalSample{*normal,true,{}}:vc::lasagna::NormalSample{{},false,"invalid normal"};
}

void FiberChunkTensor::packGpu(std::vector<unsigned char>& output,std::vector<float>& widths) const
{
    requireCompanions();
    const size_t count=size_t(shape_[0])*shape_[1]*shape_[2];output.resize(count*6);widths.resize(count);
    size_t i=0;
    for(int z=0;z<shape_[2];++z)for(int y=0;y<shape_[1];++y)for(int x=0;x<shape_[0];++x,++i){
        // Preserve original GPU lattice validity checks; no silent padding change.
        for(int a=0;a<3;++a){const int v=origin_[a]+(a==0?x:a==1?y:z);
            if(v<0||v>=volumeShape_[a])throw std::out_of_range("GPU fixture lattice outside volume");}
        for(size_t c=0;c<6;++c)output[6*i+c]=channels_[c](z,y,x);
        const cv::Vec3f p((origin_[0]+x)*spacing_,(origin_[1]+y)*spacing_,(origin_[2]+z)*spacing_);
        widths[i]=width(p);
    }
}
}
