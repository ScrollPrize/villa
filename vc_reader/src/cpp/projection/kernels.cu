#include "plan.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::projection {
template <class T>
__global__ void project(Raster g, const float* uv, const float* distance, const std::int64_t* ids, const std::int64_t* offsets, const T* label, T* out, bool* coverage, bool* conflict,
                        float* best, std::int64_t* winner, float* uvOut, bool* geometry)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < g.count; i += std::size_t(blockDim.x) * gridDim.x)
        raster(g, uv, distance, ids, offsets, label, out, coverage, conflict, best, winner, uvOut, geometry, i);
}
void launch(const Plan& p, Raster g, const void* label, ElementType type, void* out, bool* coverage, bool* conflict, float* best, std::int64_t* winner, float* uvOut, bool* geometry, int id)
{
    auto upload = [&](const auto& values) {
        auto a = device::allocate(values.size() * sizeof(values[0]), id);
        if (a->bytes) device::check(cudaMemcpyAsync(a->ptr, values.data(), a->bytes, cudaMemcpyHostToDevice, device::stream()));
        return a;
    };
    auto uv = upload(p.uv), distance = upload(p.distance), ids = upload(p.ids), offsets = upload(p.offsets);
    type.visit_load([&](auto load) {
        using T = decltype(load(static_cast<const std::uint8_t*>(nullptr)));
        project<<<unsigned((g.count + 127) / 128), 128, 0, device::stream()>>>(g, static_cast<const float*>(uv->ptr), static_cast<const float*>(distance->ptr), static_cast<const std::int64_t*>(ids->ptr),
                                                                               static_cast<const std::int64_t*>(offsets->ptr), static_cast<const T*>(label), static_cast<T*>(out), coverage, conflict, best, winner,
                                                                               uvOut, geometry);
    });
    device::check(cudaGetLastError());
    device::complete();
}
}  // namespace vcr::projection

#include "transfer.hpp"
namespace vcr::projection {
__global__ void transferKernel(RasterRequest request,unsigned long long* totals,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    RasterResult result{};transferPixel(request,&result,i);
    atomicAdd(totals+0,static_cast<unsigned long long>(result.target_surface_valid));
    atomicAdd(totals+1,static_cast<unsigned long long>(result.measured_pixels));
    atomicAdd(totals+2,static_cast<unsigned long long>(result.seam_filled_pixels));
    atomicAdd(totals+3,static_cast<unsigned long long>(result.seam_blocked_pixels));
    atomicAdd(totals+4,static_cast<unsigned long long>(result.inherited_filled_pixels));
}
void transferCuda(const RasterRequest& r,RasterResult& out,int id){
    device::Guard guard(id);auto stats=device::allocate(5*sizeof(unsigned long long),id);
    device::check(cudaMemsetAsync(stats->ptr,0,stats->bytes,device::stream()));
    auto n=(r.row_end-r.row_start)*(r.col_end-r.col_start);
    transferKernel<<<(n+255)/256,256,0,device::stream()>>>(r,static_cast<unsigned long long*>(stats->ptr),n);
    unsigned long long values[5];device::check(cudaMemcpyAsync(values,stats->ptr,stats->bytes,cudaMemcpyDeviceToHost,device::stream()));device::complete();
    out={std::int64_t(values[0]),std::int64_t(values[1]),std::int64_t(values[2]),std::int64_t(values[3]),std::int64_t(values[4])};
}
}

#include "mask.hpp"
namespace vcr::projection {
struct MaskMark {MaskBand grid;std::uint32_t* output;
 __device__ void operator()(V p)const {auto i=maskIndex(grid,p);if(i>=0)atomicOr(output+i,1u);}
};
__global__ void maskKernel(MaskBand g,std::uint32_t* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<g.rows*g.cols)maskCell(g,i,MaskMark{g,output});}
void projectMaskCuda(const MaskBand& g,std::uint32_t* output,int id){device::Guard guard(id);
    std::size_t n=std::size_t(g.shape[0])*g.shape[1]*g.shape[2];device::check(cudaMemsetAsync(output,0,n*4,device::stream()));
    maskKernel<<<(g.rows*g.cols+127)/128,128,0,device::stream()>>>(g,output);device::check(cudaGetLastError());device::complete();}
}
