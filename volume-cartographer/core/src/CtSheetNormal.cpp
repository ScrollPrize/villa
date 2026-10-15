#include "vc/core/util/CtSheetNormal.hpp"
#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cmath>
#include <deque>
#include <map>
#include <list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>

namespace vc {
namespace {
// Independent voxels/slices parallelize without changing reduction order.
const int workers=int(std::max(1u,std::min(8u,std::thread::hardware_concurrency())));

// Opt-in stage profiler; disabled calls do not read the clock or emit output.
struct Profile {
    bool enabled=std::getenv("VC_CT_NORMAL_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point last=enabled ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
    void mark(const char* name) {
        if (!enabled) return;
        const auto now=std::chrono::steady_clock::now();
        std::fprintf(stderr,"CT_NORMAL_PROFILE %s %.9f\n",name,std::chrono::duration<double>(now-last).count());
        last=now;
    }
};
// Filtering uses float32 storage/accumulation; tensor integration and the
// eigensystem remain float64. Validate changes against patch-normal angular
// error, including low-contrast/curved sheets, rather than bitwise parity.
using FilterValue = float;
constexpr int sampleStep=3;
// Radius 48 (two tangent sigmas) keeps 33^3 quadrature samples. The shorter
// support is validated by patch-normal error, not parity with radius 78.
constexpr int radius=48;
constexpr int blockSize=48;
constexpr int halo=6;
constexpr int sampleN=blockSize/sampleStep;
constexpr int inputN=(sampleN-1)*sampleStep+2*halo+1;
constexpr int queryN=2*radius/sampleStep+1;
// The smaller tiles retain the previous 96 MiB gradient-cache byte budget.
constexpr size_t gradientCacheSize=1024;
constexpr int maxQueryTilesPerAxis=(2*radius+blockSize-1)/blockSize+1;
constexpr size_t maxQueryTiles=maxQueryTilesPerAxis*maxQueryTilesPerAxis*maxQueryTilesPerAxis;
static_assert(gradientCacheSize>=maxQueryTiles); // gathered pointers remain resident
static_assert(radius%sampleStep==0 && blockSize%sampleStep==0);

// Scratch arrays are completely overwritten before they are read. Avoid
// value-initializing hundreds of MB which the next pass immediately replaces.
// resize preserves existing values, like vector; newly added values are unset.
class Buffer {
public:
    explicit Buffer(size_t size=0) { resize(size); }
    void resize(size_t size) {
        if (size > capacity_) {
            auto next=std::make_unique_for_overwrite<FilterValue[]>(size);
            if (size_) std::copy_n(data_.get(),size_,next.get());
            data_=std::move(next);
            capacity_=size;
        }
        size_=size;
    }
    size_t size() const { return size_; }
    FilterValue* data() { return data_.get(); }
    const FilterValue* data() const { return data_.get(); }
    FilterValue* begin() { return data(); }
    FilterValue* end() { return data()+size_; }
    FilterValue& operator[](size_t i) { return data_[i]; }
    const FilterValue& operator[](size_t i) const { return data_[i]; }
private:
    std::unique_ptr<FilterValue[]> data_;
    size_t size_=0, capacity_=0;
};
void cancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("CT normal sampling cancelled");
}

// C-contiguous ZYX scratch buffers (unlike Array3D's column-major storage).
size_t at(int z, int y, int x, int n) { return (size_t(z) * n + y) * n + x; }

std::vector<double> kernel(double sigma, bool derivative)
{
    const int r = int(3 * sigma + 0.5);
    std::vector<double> k(2 * r + 1);
    double sum = 0;
    for (int j = -r; j <= r; ++j) sum += k[j+r] = std::exp(-0.5 * j*j / (sigma*sigma));
    for (int j = -r; j <= r; ++j) {
        k[j+r] /= sum;
        if (derivative) k[j+r] *= j / (sigma*sigma); // correlation, not convolution
    }
    return k;
}

// Evaluate full-resolution separable Gaussian derivatives only at lattice
// sites used by the tensor quadrature. Decimate each axis after filtering it;
// this is not a derivative of downsampled CT.
struct GradientWorkspace {
    std::array<Buffer,3> values;
    std::array<std::array<Buffer,3>,8> planes;
};

void filterZ(const FilterValue* in, FilterValue* out, int z, const std::vector<double>& k)
{
    const int r=int(k.size()/2);
    std::fill_n(out,size_t(inputN)*inputN,0.f);
    for (int j=-r;j<=r;++j) {
        const auto* src=in+at(std::clamp(z+j,0,inputN-1),0,0,inputN);
        const float w=float(k[j+r]);
        #pragma omp simd
        for (int i=0;i<inputN*inputN;++i) out[i]+=w*src[i];
    }
}
void filterY(const FilterValue* in, FilterValue* out, const std::vector<double>& k)
{
    const int r=int(k.size()/2);
    for (int y=0;y<sampleN;++y) {
        auto* dst=out+size_t(y)*inputN;
        std::fill_n(dst,inputN,0.f);
        for (int j=-r;j<=r;++j) {
            const auto* src=in+size_t(std::clamp(halo+y*sampleStep+j,0,inputN-1))*inputN;
            const float w=float(k[j+r]);
            #pragma omp simd
            for (int x=0;x<inputN;++x) dst[x]+=w*src[x];
        }
    }
}
void filterX(const FilterValue* in, FilterValue* out, const std::vector<double>& k)
{
    const int r=int(k.size()/2);
    for (int y=0;y<sampleN;++y) for (int x=0;x<sampleN;++x) {
        float sum=0;
        for (int j=-r;j<=r;++j)
            sum+=float(k[j+r])*in[size_t(y)*inputN+std::clamp(halo+x*sampleStep+j,0,inputN-1)];
        out[size_t(y)*sampleN+x]=sum;
    }
}
void gradients(const Buffer& image, double sigma, GradientWorkspace& workspace,
               const std::atomic_bool* cancel)
{
    const auto g=kernel(sigma,false), d=kernel(sigma,true);
    for (auto& value:workspace.values) value.resize(size_t(sampleN)*sampleN*sampleN);
    for (int worker=0;worker<workers;++worker)
        for (auto& plane:workspace.planes[worker]) plane.resize(size_t(inputN)*inputN);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int worker=0;worker<workers;++worker) {
        auto& planes=workspace.planes[worker];
        auto* smoothZ=planes[0].data();
        auto* derivativeZ=planes[1].data();
        auto* scratch=planes[2].data();
        for (int z=worker;z<sampleN;z+=workers) {
            if (cancel && cancel->load()) continue;
            const size_t offset=at(z,0,0,sampleN);
            filterZ(image.data(),smoothZ,halo+z*sampleStep,g);
            filterZ(image.data(),derivativeZ,halo+z*sampleStep,d);
            filterY(derivativeZ,scratch,g);
            filterX(scratch,workspace.values[2].data()+offset,g);
            filterY(smoothZ,scratch,d);
            filterX(scratch,workspace.values[1].data()+offset,g);
            filterY(smoothZ,scratch,g);
            filterX(scratch,workspace.values[0].data()+offset,d);
        }
    }
    cancelled(cancel);
}

CtSheetNormal principal(const cv::Matx33d& tensor)
{
    cv::Vec3d values;
    cv::Matx33d vectors;
    if (!cv::eigen(tensor,values,vectors) || values[0] <= 1e-12)
        return {{0,0,0},false,0,"CT has no normal energy"};
    const double gap = (values[0]-values[1])/values[0];
    if (gap <= 1e-6) return {{0,0,0},false,gap,"CT normal axis is ambiguous"};
    return {{vectors(0,0),vectors(0,1),vectors(0,2)},true,gap,{}};
}

CtSheetNormal integrate(const std::array<Buffer,3>& g, const cv::Vec3d& centerXYZ,
                        const cv::Vec3d& axis, double tangentSigma, double normalSigma,
                        const Buffer* magnitude, double floor, const std::atomic_bool* cancel)
{
    // Fixed per-slice partials preserve deterministic summation under OpenMP.
    constexpr int n=queryN;
    std::vector<std::array<double,7>> slices(n);
    const double tangentInv = 1/(tangentSigma*tangentSigma);
    const double excessInv = 1/(normalSigma*normalSigma)-tangentInv;
    // Along each X row the Gaussian exponent is quadratic. Advance its value
    // and first ratio multiplicatively, replacing one exp per voxel with two
    // per row. Only 53 steps are taken, keeping recurrence error negligible.
    const double curvature=sampleStep*sampleStep*(tangentInv+axis[0]*axis[0]*excessInv);
    const double ratioStep=std::exp(-curvature);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int z=0; z<n; ++z) {
        if (cancel && cancel->load()) continue;
        auto& sum=slices[z];
        const double dz=z*sampleStep-centerXYZ[2];
        for (int y=0; y<n; ++y) {
            const double dy=y*sampleStep-centerXYZ[1];
            const double dx=-centerXYZ[0];
            const double along=axis[0]*dx+axis[1]*dy+axis[2]*dz;
            double weight=std::exp(-0.5*((dx*dx+dy*dy+dz*dz)*tangentInv+along*along*excessInv));
            double ratio=std::exp(-sampleStep*(dx*tangentInv+axis[0]*along*excessInv)-0.5*curvature);
            for (int x=0; x<n; ++x) {
                double w=weight;
                const size_t i=at(z,y,x,n);
                if (magnitude) w/=std::max(double((*magnitude)[i]),floor);
                const double gx=g[0][i],gy=g[1][i],gz=g[2][i];
                sum[0]+=w*gx*gx; sum[1]+=w*gx*gy; sum[2]+=w*gx*gz;
                sum[3]+=w*gy*gy; sum[4]+=w*gy*gz; sum[5]+=w*gz*gz;
                sum[6]+=w;
                weight*=ratio;
                ratio*=ratioStep;
            }
        }
    }
    cancelled(cancel);
    std::array<double,7> total{};
    for (const auto& s:slices) for (size_t i=0;i<total.size();++i) total[i]+=s[i];
    if (!(total[6]>0)) return {{0,0,0},false,0,"CT has no integration weight"};
    const auto& t=total;
    return principal(cv::Matx33d(t[0],t[1],t[2],t[1],t[3],t[4],t[2],t[4],t[5])*(1/t[6]));
}

double medianMagnitude(const Buffer& magnitude, const std::atomic_bool* cancel)
{
    // Monotone bins only locate the median's bucket; nth_element still selects
    // an original magnitude, with no histogram quantization of the floor.
    // Magnitudes are finite/nonnegative (bounded uint8 input and fixed kernels).
    constexpr size_t bins=4096;
    const auto bucket=[](double value) {
        return size_t(std::min(value*4096.,4095.));
    };
    std::vector<std::array<size_t,bins>> counts(workers);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int worker=0;worker<workers;++worker) {
        auto& histogram=counts[worker];
        for (size_t i=magnitude.size()*worker/workers;
             i<magnitude.size()*(worker+1)/workers;++i)
            ++histogram[bucket(magnitude[i])];
    }
    cancelled(cancel);
    size_t rank=magnitude.size()/2, selected=0, count=0;
    for (;selected<bins;++selected) {
        count=0;
        for (const auto& histogram:counts) count+=histogram[selected];
        if (rank<count) break;
        rank-=count;
    }
    Buffer candidates(count);
    std::vector<size_t> offsets(workers);
    for (int worker=1;worker<workers;++worker)
        offsets[worker]=offsets[worker-1]+counts[worker-1][selected];
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int worker=0;worker<workers;++worker) {
        size_t dest=offsets[worker];
        for (size_t i=magnitude.size()*worker/workers;
             i<magnitude.size()*(worker+1)/workers;++i)
            if (bucket(magnitude[i])==selected) candidates[dest++]=magnitude[i];
    }
    cancelled(cancel);
    auto mid=candidates.begin()+rank;
    std::nth_element(candidates.begin(),mid,candidates.end());
    return *mid;
}

// Fields use a global lattice so a cache hit or eviction cannot change the
// estimated normal. Every derivative has its full CT halo, including at the
// quadrature window's edge. No normal interpolation/rounding is involved.
struct GradientField {
    std::array<Buffer,3> seed, fine;
};
struct GradientBlock : GradientField {
    std::array<int,3> key{}; // ZYX block coordinates
};
struct EstimateWorkspace {
    Buffer image, magnitude;
    GradientWorkspace gradients;
    GradientField query;
};

void fillBlock(const uint8_t* raw, const std::array<size_t,3>& shape,
               const std::array<int,3>& offset, GradientBlock& block,
               EstimateWorkspace& scratch, const std::atomic_bool* cancel)
{
    Profile profile;
    auto& image=scratch.image;
    image.resize(size_t(inputN)*inputN*inputN);
    #pragma omp parallel for collapse(2) schedule(static) num_threads(workers)
    for (int z=0;z<inputN;++z) for (int y=0;y<inputN;++y) {
        const auto* src=raw+((z+offset[0])*shape[1]+y+offset[1])*shape[2]+offset[2];
        auto* dst=image.data()+at(z,y,0,inputN);
        for (int x=0;x<inputN;++x) dst[x]=src[x]/255.;
    }
    cancelled(cancel);
    profile.mark("convert");
    // Swap in the evicted block's allocations, reusing them on cache misses.
    auto& workspace=scratch.gradients;
    workspace.values.swap(block.seed);
    gradients(image,2,workspace,cancel);
    workspace.values.swap(block.seed);
    profile.mark("seed_gradients");
    workspace.values.swap(block.fine);
    gradients(image,1,workspace,cancel);
    workspace.values.swap(block.fine);
    profile.mark("fine_gradients");
}

CtSheetNormal estimate(const GradientField& block,
                       const cv::Vec3d& centerXYZ, EstimateWorkspace& scratch,
                       const std::atomic_bool* cancel)
{
    Profile profile;
    auto result=integrate(block.seed,centerXYZ,{0,0,0},24,24,nullptr,0,cancel);
    profile.mark("seed_integration");
    if (!result.valid) return result;
    const auto& g=block.fine;
    auto& magnitude=scratch.magnitude;
    magnitude.resize(size_t(queryN)*queryN*queryN);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int z=0;z<queryN;++z) for (int y=0;y<queryN;++y) for (int x=0;x<queryN;++x) {
        const size_t i=at(z,y,x,queryN);
        const double gx=g[0][i], gy=g[1][i], gz=g[2][i];
        magnitude[at(z,y,x,queryN)]=std::sqrt(gx*gx+gy*gy+gz*gz);
    }
    const double floor=std::max(medianMagnitude(magnitude,cancel),1e-8);
    profile.mark("magnitude_median");
    for (int iteration=0;iteration<3 && result.valid;++iteration)
        result=integrate(g,centerXYZ,result.normal,24,4,&magnitude,floor,cancel);
    profile.mark("refinements");
    return result;
}
} // namespace

struct CtSheetNormalSampler::Impl {
    std::shared_ptr<Volume> volume;
    int level;
    double scale;
    mutable std::mutex mutex;
    mutable EstimateWorkspace scratch;
    mutable std::unique_ptr<uint8_t[]> raw;
    mutable size_t rawCapacity=0;
    using Blocks=std::list<std::unique_ptr<GradientBlock>>;
    mutable Blocks blocks; // MRU first
    mutable std::map<std::array<int,3>,Blocks::iterator> blockIndex;
    mutable std::map<std::array<double,3>,CtSheetNormal> cache;
    mutable std::deque<std::array<double,3>> order;
};

CtSheetNormalSampler::CtSheetNormalSampler(const std::filesystem::path& path, int level, double scale)
    : impl_(std::make_unique<Impl>())
{
    if (level<0 || !std::isfinite(scale) || scale<=0)
        throw std::invalid_argument("CT normals require a nonnegative level and positive finite coordinate scale");
    impl_->volume=Volume::New(path);
    if (!impl_->volume->hasScaleLevel(level)) throw std::invalid_argument("CT normals: selected CT level is missing");
    if (impl_->volume->dtype()!=vc::render::ChunkDtype::UInt8)
        throw std::invalid_argument("CT normals currently require uint8 CT (the validated estimator input)");
    impl_->level=level;
    impl_->scale=scale;
}
CtSheetNormalSampler::~CtSheetNormalSampler()=default;

CtSheetNormal CtSheetNormalSampler::sampleNormal(const cv::Vec3d& point, const std::atomic_bool* cancel) const
{
    for (double v:point.val) if (!std::isfinite(v)) throw std::invalid_argument("CT normal point must be finite");
    cancelled(cancel);
    std::lock_guard lock(impl_->mutex);
    cancelled(cancel);
    const std::array<double,3> key{point[0],point[1],point[2]};
    if (const auto it=impl_->cache.find(key);it!=impl_->cache.end()) return it->second;
    const auto shape=impl_->volume->shape(impl_->level);
    std::array<int,3> queryStart;
    cv::Vec3d center;
    for (int axis=0;axis<3;++axis) {
        const double coordinate=point[2-axis]*impl_->scale;
        const double lattice=std::floor(coordinate/sampleStep)*sampleStep;
        if (!std::isfinite(coordinate) || lattice<radius+halo || lattice+radius+halo+1>shape[axis])
            return {{0,0,0},false,0,"CT normal context crosses volume boundary"};
        queryStart[axis]=int(lattice)-radius;
        center[2-axis]=coordinate-lattice+radius;
    }
    Profile profile;
    auto& query=impl_->scratch.query;
    for (auto* field:{&query.seed,&query.fine})
        for (auto& component:*field) component.resize(size_t(queryN)*queryN*queryN);
    auto& blocks=impl_->blocks;
    struct CopyRegion {
        const GradientBlock* block;
        std::array<int,3> lo, hi, tileOffset;
    };
    std::vector<CopyRegion> regions;
    regions.reserve(maxQueryTiles);
    struct Request { std::array<int,3> key; const GradientBlock* block; };
    std::vector<Request> requests;
    requests.reserve(maxQueryTiles);
    std::array<int,3> readStart{}, readEnd{};
    bool missing=false;
    // Promote every requested hit before choosing victims. This pins the
    // current query's tiles within the bounded LRU for the following fills.
    for (int bz=queryStart[0]/blockSize;bz<=(queryStart[0]+2*radius)/blockSize;++bz)
    for (int by=queryStart[1]/blockSize;by<=(queryStart[1]+2*radius)/blockSize;++by)
    for (int bx=queryStart[2]/blockSize;bx<=(queryStart[2]+2*radius)/blockSize;++bx) {
        cancelled(cancel);
        const std::array<int,3> blockKey{bz,by,bx};
        const auto found=impl_->blockIndex.find(blockKey);
        const GradientBlock* hit=nullptr;
        if (found!=impl_->blockIndex.end()) {
            blocks.splice(blocks.begin(),blocks,found->second);
            hit=blocks.front().get();
        } else {
            for (int axis=0;axis<3;++axis) {
                const int start=blockKey[axis]*blockSize-halo;
                readStart[axis]=missing ? std::min(readStart[axis],start) : start;
                readEnd[axis]=missing ? std::max(readEnd[axis],start+inputN) : start+inputN;
            }
            missing=true;
        }
        requests.push_back({blockKey,hit});
    }
    std::array<size_t,3> readShape{};
    if (missing) {
        for (int axis=0;axis<3;++axis) readShape[axis]=readEnd[axis]-readStart[axis];
        const size_t readSize=readShape[0]*readShape[1]*readShape[2];
        // Chunk reads overwrite every byte, including clipped/sparse areas.
        if (impl_->rawCapacity<readSize) {
            impl_->raw=std::make_unique_for_overwrite<uint8_t[]>(readSize);
            impl_->rawCapacity=readSize;
        }
        // One bounding read batches chunk fetches and shares overlapping tile
        // halos. Cached tiles inside this box need not be filtered again.
        Volume::readContiguousZYX(std::span(impl_->raw.get(),readSize),readShape,readStart,
                                  *impl_->volume->chunkedCache(),impl_->level);
        profile.mark("tiles_read");
    }
    for (const auto& request:requests) {
        cancelled(cancel);
        const auto& blockKey=request.key;
        const GradientBlock* current=request.block;
        if (!current) {
            // Failed or cancelled fills are not published to the cache.
            std::unique_ptr<GradientBlock> block;
            if (blocks.size()==gradientCacheSize) {
                impl_->blockIndex.erase(blocks.back()->key);
                block=std::move(blocks.back());
                blocks.pop_back();
            }
            else block=std::make_unique<GradientBlock>();
            block->key=blockKey;
            std::array<int,3> offset;
            for (int axis=0;axis<3;++axis) offset[axis]=blockKey[axis]*blockSize-halo-readStart[axis];
            fillBlock(impl_->raw.get(),readShape,offset,*block,impl_->scratch,cancel);
            profile.mark("tile_gradients");
            blocks.push_front(std::move(block));
            try { impl_->blockIndex.emplace(blockKey,blocks.begin()); }
            catch (...) { blocks.pop_front(); throw; }
            current=blocks.front().get();
        }
        // Assemble the requested quadrature window with contiguous row copies.
        // Tile prefetch may cross volume bounds; the validated query plus its
        // derivative halo never does, so padded derivatives are never used.
        const auto& block=*current;
        std::array<int,3> lo, hi, tileOffset;
        for (int axis=0;axis<3;++axis) {
            tileOffset[axis]=(blockKey[axis]*blockSize-queryStart[axis])/sampleStep;
            lo[axis]=std::max(0,tileOffset[axis]);
            hi[axis]=std::min(queryN,tileOffset[axis]+sampleN);
        }
        regions.push_back({&block,lo,hi,tileOffset});
    }
    // Cache updates are complete. All referenced tiles remain resident and
    // each worker writes disjoint output Z slices, irrespective of tile layout.
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int z=0;z<queryN;++z) for (const auto& region:regions) {
        const auto& [block,lo,hi,tileOffset]=region;
        if (z<lo[0] || z>=hi[0]) continue;
        for (int y=lo[1];y<hi[1];++y) {
            const auto src=at(z-tileOffset[0],y-tileOffset[1],lo[2]-tileOffset[2],sampleN);
            const auto dst=at(z,y,lo[2],queryN);
            for (int axis=0;axis<3;++axis) {
                std::copy_n(block->seed[axis].data()+src,hi[2]-lo[2],query.seed[axis].data()+dst);
                std::copy_n(block->fine[axis].data()+src,hi[2]-lo[2],query.fine[axis].data()+dst);
            }
        }
    }
    profile.mark("query_fields");
    auto result=estimate(query,center,impl_->scratch,cancel);
    cancelled(cancel);
    if (impl_->order.size()==256) { impl_->cache.erase(impl_->order.front()); impl_->order.pop_front(); }
    impl_->cache.emplace(key,result); impl_->order.push_back(key);
    profile.mark("estimate_total");
    return result;
}
} // namespace vc
