#pragma once
// Numerical kernels imported with user authorization from origin/graph-unification,
// de1701f1a, volume-cartographer/core/src/CtSheetNormal.cpp.
// Original method: ct-anisotropic-1-24-4-grid3-r48-v4.
// Changes: one worker; in-memory crop adapter instead of Volume/IO/LRU services.
// OpenCV is only an optional dependency of this standalone experiment.
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace vcr::elastic_experiment::ct {
struct CtSheetNormal {
    cv::Vec3d normal{0,0,0};
    bool valid = false;
    double gap = 0;
    std::string reason;
};
// Independent voxels/slices parallelize without changing reduction order.
constexpr int workers = 1; // Single-threaded experiment, including tensor work.

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

// Coordinates/origin are global ZYX. This preserves the branch's global
// 3-voxel quadrature lattice even when the reader crop starts off that lattice.
class CropNormals {
    std::span<const uint8_t> raw_;
    std::array<size_t,3> shape_;
    std::array<int,3> origin_;
    std::map<std::array<int,3>, std::unique_ptr<GradientBlock>> blocks_;
    EstimateWorkspace scratch_;
public:
    CropNormals(std::span<const uint8_t> raw, std::array<size_t,3> shape, std::array<int,3> origin)
        : raw_(raw), shape_(shape), origin_(origin) {
        if (raw.size() != shape[0]*shape[1]*shape[2]) throw std::invalid_argument("CT crop size mismatch");
    }
    CtSheetNormal sample(const std::array<double,3>& point) {
        std::array<int,3> start;
        cv::Vec3d center;
        for (int a=0; a<3; ++a) {
            const double lattice=std::floor(point[a]/sampleStep)*sampleStep;
            if (!std::isfinite(point[a]) || lattice-radius-halo < origin_[a] ||
                lattice+radius+halo >= origin_[a]+double(shape_[a]))
                return {{0,0,0},false,0,"normal query lacks full CT halo"};
            start[a]=int(lattice)-radius;
            center[2-a]=point[a]-lattice+radius;
        }
        auto& query=scratch_.query;
        for (auto* field:{&query.seed,&query.fine})
            for (auto& component:*field) component.resize(size_t(queryN)*queryN*queryN);
        for (int bz=start[0]/blockSize;bz<=(start[0]+2*radius)/blockSize;++bz)
        for (int by=start[1]/blockSize;by<=(start[1]+2*radius)/blockSize;++by)
        for (int bx=start[2]/blockSize;bx<=(start[2]+2*radius)/blockSize;++bx) {
            const std::array<int,3> key{bz,by,bx};
            auto it=blocks_.find(key);
            if (it==blocks_.end()) {
                // Only unused tile-prefetch halo can be clipped: the query
                // support and every derivative it uses were checked above.
                std::vector<uint8_t> tile(size_t(inputN)*inputN*inputN);
                for (int z=0;z<inputN;++z) for (int y=0;y<inputN;++y) for (int x=0;x<inputN;++x) {
                    auto iz=std::clamp(bz*blockSize-halo+z-origin_[0],0,int(shape_[0])-1);
                    auto iy=std::clamp(by*blockSize-halo+y-origin_[1],0,int(shape_[1])-1);
                    auto ix=std::clamp(bx*blockSize-halo+x-origin_[2],0,int(shape_[2])-1);
                    tile[at(z,y,x,inputN)]=raw_[(size_t(iz)*shape_[1]+iy)*shape_[2]+ix];
                }
                auto block=std::make_unique<GradientBlock>(); block->key=key;
                fillBlock(tile.data(),{inputN,inputN,inputN},{0,0,0},*block,scratch_,nullptr);
                it=blocks_.emplace(key,std::move(block)).first;
            }
            const auto& block=*it->second;
            std::array<int,3> offset,lo,hi;
            for (int a=0;a<3;++a) {
                offset[a]=(key[a]*blockSize-start[a])/sampleStep;
                lo[a]=std::max(0,offset[a]); hi[a]=std::min(queryN,offset[a]+sampleN);
            }
            for (int z=lo[0];z<hi[0];++z) for (int y=lo[1];y<hi[1];++y) {
                const auto src=at(z-offset[0],y-offset[1],lo[2]-offset[2],sampleN);
                const auto dst=at(z,y,lo[2],queryN);
                for (int a=0;a<3;++a) {
                    std::copy_n(block.seed[a].data()+src,hi[2]-lo[2],query.seed[a].data()+dst);
                    std::copy_n(block.fine[a].data()+src,hi[2]-lo[2],query.fine[a].data()+dst);
                }
            }
        }
        return estimate(query,center,scratch_,nullptr);
    }
};
} // namespace vcr::elastic_experiment::ct
