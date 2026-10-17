#include "types.hpp"
#include "../sampling_math.hpp"
#include "../cuda_buffer.hpp"
#include <algorithm>
#include <cmath>
namespace vcr::reader_cuda {
namespace {
struct Cell {
    using value_type = float;
    std::int64_t lo[3]; float frac[3]; unsigned axes[3], k = 0;
};
struct Lookup {
    Source s;
    __device__ const unsigned char* address(const std::int64_t* at) const {
        std::int64_t key = 0; std::size_t local = 0;
        for (int a=0;a<3;++a) {
            if (at[a]<0 || at[a]>=s.shape[a]) return nullptr;
            key=key*s.grid[a]+at[a]/s.chunks[a];
            local=local*s.chunks[a]+at[a]%s.chunks[a];
        }
        std::size_t slot=(std::uint64_t(key)*UINT64_C(0x9e3779b97f4a7c15))&s.mask;
        while (s.keys[slot]!=key) {
            if (s.keys[slot]<0) return nullptr;
            slot=(slot+1)&s.mask;
        }
        return s.pointers[slot]+local;
    }
};
struct Corner {
    Lookup lookup; Cell cell;
    __device__ float operator()(std::size_t corner) const {
        std::int64_t at[3]{cell.lo[0],cell.lo[1],cell.lo[2]};
        for (unsigned a=0;a<cell.k;++a) at[cell.axes[a]]+=(corner>>a)&1;
        const auto* p=lookup.address(at); return p ? float(*p) : float(lookup.s.fill);
    }
};
struct Load { __device__ float operator()(const unsigned char* p) const { return float(*p); } };
__device__ float interpolate(Source s, const float* c, const std::int64_t* base, int order) {
    Cell cell;
    bool finite=true, outside=false;
    for (int a=0;a<3;++a) {
        finite &= isfinite(c[a]);
        outside |= !(c[a]>float(-1-base[a]) && c[a]<float(s.shape[a]-base[a])+(order ? 0.f : 1.f));
    }
    if (!finite || outside) return !finite && order ? nanf("") : float(s.fill);
    for (int a=0;a<3;++a) {
        const float low=floorf(c[a]), f=c[a]-low;
        const auto integer=base[a]+static_cast<std::int64_t>(low);
        cell.lo[a]=integer+(order ? 0 : (f>.5f || (f==.5f && (integer&1))));
        cell.frac[a]=order ? f : 0;
        if (cell.frac[a]!=0) cell.axes[cell.k++]=a;
    }
    Corner corner{Lookup{s},cell};
    if (!order) return corner(0);
    bool interior=cell.k==3;
    for (int a=0;a<3;++a) interior &= cell.lo[a]>=0 && cell.lo[a]+1<s.shape[a] && cell.lo[a]%s.chunks[a]+1<s.chunks[a];
    const auto* p=interior ? corner.lookup.address(cell.lo) : nullptr;
    if (p) {
        const std::int64_t strides[3]{s.chunks[1]*s.chunks[2],s.chunks[2],1};
        Load load;
        return sampling::trilinear(cell.frac,p,strides,load);
    }
    return sampling::interpolate(cell,corner);
}
struct Origin { std::int64_t v[3]; };
template<class Out> __global__ void sample(Source s, const Grid* grids, const void* points,
        int bits, std::size_t count, Origin base, int kind, int order, Out* out) {
    Grid g{};
    if (kind<2) { g=grids[blockIdx.y]; count=g.count; }
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x) {
        float value;
        if (kind==0 || kind==3) {
            std::int64_t at[3];
            if (kind==0) {
                at[0]=g.base[0]+std::int64_t(i/(g.shape[1]*g.shape[2]));
                at[1]=g.base[1]+std::int64_t(i/g.shape[2]%g.shape[1]); at[2]=g.base[2]+std::int64_t(i%g.shape[2]);
            } else for (int a=0;a<3;++a) at[a]=bits==32 ? static_cast<const std::int32_t*>(points)[3*i+a] : static_cast<const std::int64_t*>(points)[3*i+a];
            const auto* p=Lookup{s}.address(at); value=p ? *p : s.fill;
        } else {
            float c[3];
            if (kind==1) {
                const float z=float(i/(g.shape[1]*g.shape[2])), y=float(i/g.shape[2]%g.shape[1]), x=float(i%g.shape[2]);
                for (int a=0;a<3;++a) c[a]=sampling::affine(g.origin[a],g.axes+3*a,z,y,x);
            } else for (int a=0;a<3;++a) c[a]=static_cast<const float*>(points)[3*i+a];
            value=interpolate(s,c,kind==1 ? g.base : base.v,order);
        }
        out[g.offset+i]=Out(value);
    }
}
template<class Out, class In> __global__ void convert(const In* input, Out* output, std::size_t n) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(blockDim.x)*gridDim.x) output[i]=Out(input[i]);
}
}
void launch(Source s, const Grid* grids, std::size_t jobs, std::size_t largest,
        const void* points, int bits, std::size_t count, const std::int64_t* base,
        int kind, int order, bool floating, void* output) {
    if (!count) return;
    Origin origin{}; if (base) std::copy_n(base,3,origin.v);
    const auto n=kind<2 ? largest : count;
    dim3 blocks(unsigned(std::min<std::size_t>((n+255)/256,65535)),unsigned(kind<2 ? jobs : 1));
    if (floating) sample<<<blocks,256>>>(s,grids,points,bits,count,origin,kind,order,static_cast<float*>(output));
    else sample<<<blocks,256>>>(s,grids,points,bits,count,origin,kind,order,static_cast<unsigned char*>(output));
    device::check(cudaGetLastError());
}
void cast(const void* input, void* output, std::size_t n, int in, int out) {
    if (!n) return;
    const unsigned blocks=unsigned(std::min<std::size_t>((n+255)/256,65535));
    if (in==out) device::check(cudaMemcpyAsync(output,input,n*in,cudaMemcpyDeviceToDevice));
    else if (in==1 && out==4) convert<<<blocks,256>>>(static_cast<const unsigned char*>(input),static_cast<float*>(output),n);
    else throw std::invalid_argument("CUDA reader conversion supports uint8 to float32");
    device::check(cudaGetLastError());
}
}
