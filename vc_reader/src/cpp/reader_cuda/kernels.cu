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
            key=key*s.grid[a]+at[a]/s.brick[a];
            local=local*s.brick[a]+at[a]%s.brick[a];
        }
        std::size_t slot=home_slot(key,s.mask);
        for (;;) {
            const Slot entry=s.table[slot];
            if (entry.key==key) return entry.pointer+local;
            if (entry.key<0) return nullptr;
            slot=(slot+1)&s.mask;
        }
    }
};
// Validity: `ok` clears when a neighbour read has no brick (outside the
// volume, or not resident).
struct Corner {
    Lookup lookup; Cell cell; bool* ok;
    __device__ float operator()(std::size_t corner) const {
        std::int64_t at[3]{cell.lo[0],cell.lo[1],cell.lo[2]};
        for (unsigned a=0;a<cell.k;++a) at[cell.axes[a]]+=(corner>>a)&1;
        const auto* p=lookup.address(at);
        if (!p) *ok=false;
        return p ? float(*p) : float(lookup.s.fill);
    }
};
struct Load { __device__ float operator()(const unsigned char* p) const { return float(*p); } };
__device__ float interpolate(Source s, const float* c, const std::int64_t* base, int order, bool& ok) {
    Cell cell;
    bool finite=true, outside=false;
    for (int a=0;a<3;++a) {
        finite &= isfinite(c[a]);
        outside |= !(c[a]>float(-1-base[a]) && c[a]<float(s.shape[a]-base[a])+(order ? 0.f : 1.f));
    }
    if (!finite || outside) { ok=false; return !finite && order ? nanf("") : float(s.fill); }
    for (int a=0;a<3;++a) {
        const float low=floorf(c[a]), f=c[a]-low;
        const auto integer=base[a]+static_cast<std::int64_t>(low);
        cell.lo[a]=integer+(order ? 0 : (f>.5f || (f==.5f && (integer&1))));
        cell.frac[a]=order ? f : 0;
        if (cell.frac[a]!=0) cell.axes[cell.k++]=a;
    }
    Corner corner{Lookup{s},cell,&ok};
    if (!order) return corner(0);
    bool interior=cell.k==3;
    for (int a=0;a<3;++a) interior &= cell.lo[a]>=0 && cell.lo[a]+1<s.shape[a] && cell.lo[a]%s.brick[a]+1<s.brick[a];
    const auto* p=interior ? corner.lookup.address(cell.lo) : nullptr;
    if (p) {
        const std::int64_t strides[3]{s.brick[1]*s.brick[2],s.brick[2],1};
        Load load;
        return sampling::trilinear(cell.frac,p,strides,load);
    }
    return sampling::interpolate(cell,corner);
}
// Affine crops: 32-bit indices (count < 2**32, coordinates <= 2**30) and, for
// power-of-two bricks, shifts/masks instead of 64-bit division. Coordinates,
// path selection and interpolation arithmetic are exactly those of interpolate().
template<bool Pow2> __device__ const unsigned char* address32(const Source& s, const int* at) {
    std::int64_t key=0; std::uint64_t local=0;
    for (int a=0;a<3;++a) {
        if (unsigned(at[a])>=unsigned(s.shape[a])) return nullptr;
        const unsigned b=Pow2 ? unsigned(at[a])>>s.shift[a] : unsigned(at[a])/unsigned(s.brick[a]);
        const unsigned l=Pow2 ? unsigned(at[a])&((1u<<s.shift[a])-1) : unsigned(at[a])-b*unsigned(s.brick[a]);
        key=key*s.grid[a]+b; local=local*std::uint64_t(s.brick[a])+l;
    }
    std::size_t slot=home_slot(key,s.mask);
    for (;;) {
        const Slot entry=s.table[slot];
        if (entry.key==key) return entry.pointer+local;
        if (entry.key<0) return nullptr;
        slot=(slot+1)&s.mask;
    }
}
// A cell off the grid on every axis. interpolate() over Cell3 is the generic
// k=3 weighted sum, unrolled: the arithmetic of interpolate(Cell,Corner) for k=3.
struct Cell3 {
    using value_type = float;
    float frac[3];
    unsigned axes[3]{0,1,2}, k=3;
};
// Gathered corner values. trilinear() addresses corner c at byte c of the array.
struct Gathered {
    const float* v;
    __device__ float operator()(std::size_t corner) const { return v[corner]; }
    __device__ float operator()(const unsigned char* corner) const { return v[corner-reinterpret_cast<const unsigned char*>(v)]; }
};
template<bool Pow2> __device__ float interpolate32(const Source& s, const float* c, const std::int64_t* base, int order, bool& ok) {
    bool finite=true, outside=false;
    for (int a=0;a<3;++a) {
        finite &= isfinite(c[a]);
        outside |= !(c[a]>float(-1-base[a]) && c[a]<float(s.shape[a]-base[a])+(order ? 0.f : 1.f));
    }
    if (!finite || outside) { ok=false; return !finite && order ? nanf("") : float(s.fill); }
    int lo[3]; float frac[3]; bool whole=true, brick=true;
    for (int a=0;a<3;++a) {
        const float low=floorf(c[a]), f=c[a]-low;
        const int integer=int(base[a])+int(low);
        lo[a]=integer+(order ? 0 : (f>.5f || (f==.5f && (integer&1))));
        frac[a]=order ? f : 0;
        const int within=Pow2 ? (lo[a]&((1<<s.shift[a])-1)) : lo[a]%int(s.brick[a]);
        whole &= frac[a]!=0 && lo[a]>=0 && lo[a]+1<s.shape[a];
        brick &= within+1<s.brick[a];
    }
    if (!order) { const auto* p=address32<Pow2>(s,lo); ok=p; return p ? float(*p) : float(s.fill); }
    if (whole) {
        // Same corner values and per-case arithmetic as interpolate(): trilinear
        // when one resident brick holds the cell, otherwise the generic k=3 sum.
        float v[8];
        const unsigned char* p=brick ? address32<Pow2>(s,lo) : nullptr;
        if (p) {
            const int s1=int(s.brick[2]), s0=int(s.brick[1])*s1;
            v[0]=p[0]; v[1]=p[s0]; v[2]=p[s1]; v[3]=p[s0+s1];
            v[4]=p[1]; v[5]=p[s0+1]; v[6]=p[s1+1]; v[7]=p[s0+s1+1];
            const std::int64_t corners[3]{1,2,4};
            Gathered load{v};
            return sampling::trilinear(frac,reinterpret_cast<const unsigned char*>(v),corners,load);
        }
        for (int corner=0;corner<8;++corner) {
            const int at[3]{lo[0]+(corner&1),lo[1]+(corner>>1&1),lo[2]+(corner>>2&1)};
            const auto* q=address32<Pow2>(s,at); v[corner]=q ? float(*q) : float(s.fill); ok &= q!=nullptr;
        }
        return sampling::interpolate(Cell3{{frac[0],frac[1],frac[2]}},Gathered{v});
    }
    Cell cell;
    for (int a=0;a<3;++a) { cell.lo[a]=lo[a]; cell.frac[a]=frac[a]; if (frac[a]!=0) cell.axes[cell.k++]=a; }
    return sampling::interpolate(cell,Corner{Lookup{s},cell,&ok});
}
template<class Out, bool Pow2> __global__ void sample_affine(Source s, const Grid* grids, int order, Out* out, unsigned char* valid) {
    const Grid g=grids[blockIdx.y];
    const unsigned count=unsigned(g.count), n1=unsigned(g.shape[1]), n2=unsigned(g.shape[2]);
    for (unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const unsigned row=i/n2, x=i-row*n2, z=row/n1, y=row-z*n1;
        float c[3];
        for (int a=0;a<3;++a) c[a]=sampling::affine(g.origin[a],g.axes+3*a,float(z),float(y),float(x));
        bool ok=true;
        out[g.offset+i]=Out(interpolate32<Pow2>(s,c,g.base,order,ok));
        if (valid) valid[g.offset+i]=ok;
    }
}
struct Origin { std::int64_t v[3]; };
template<class Out> __global__ void sample(Source s, const Grid* grids, const void* points,
        int bits, std::size_t count, Origin base, int kind, int order, Out* out, unsigned char* valid) {
    Grid g{};
    if (kind<2) { g=grids[blockIdx.y]; count=g.count; }
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x) {
        float value; bool ok=true;
        if (kind==0 || kind==3) {
            std::int64_t at[3];
            if (kind==0) {
                at[0]=g.base[0]+std::int64_t(i/(g.shape[1]*g.shape[2]));
                at[1]=g.base[1]+std::int64_t(i/g.shape[2]%g.shape[1]); at[2]=g.base[2]+std::int64_t(i%g.shape[2]);
            } else for (int a=0;a<3;++a) at[a]=bits==32 ? static_cast<const std::int32_t*>(points)[3*i+a] : static_cast<const std::int64_t*>(points)[3*i+a];
            const auto* p=Lookup{s}.address(at); value=p ? *p : s.fill; ok=p;
        } else {
            float c[3];
            if (kind==1) {
                const float z=float(i/(g.shape[1]*g.shape[2])), y=float(i/g.shape[2]%g.shape[1]), x=float(i%g.shape[2]);
                for (int a=0;a<3;++a) c[a]=sampling::affine(g.origin[a],g.axes+3*a,z,y,x);
            } else for (int a=0;a<3;++a) c[a]=static_cast<const float*>(points)[3*i+a];
            value=interpolate(s,c,kind==1 ? g.base : base.v,order,ok);
        }
        out[g.offset+i]=Out(value);
        if (valid) valid[g.offset+i]=ok;
    }
}
// Validity: same coordinate evaluation as sampling, neighbour checks only.
__device__ bool present(const Source& s, const std::int64_t* at, bool resident) {
    for (int a=0;a<3;++a) if (at[a]<0 || at[a]>=s.shape[a]) return false;
    return !resident || Lookup{s}.address(at)!=nullptr;
}
__device__ unsigned char supported(const Source& s, const float* c, const std::int64_t* base, int order, bool resident) {
    std::int64_t lo[3]; float frac[3];
    for (int a=0;a<3;++a) {
        if (!isfinite(c[a])) return 0;
        const float low=floorf(fminf(fmaxf(c[a],-4.f-float(base[a])),float(s.shape[a]-base[a])+4.f)), f=c[a]-low;
        const auto integer=base[a]+static_cast<std::int64_t>(low);
        lo[a]=integer+(order ? 0 : (f>.5f || (f==.5f && (integer&1))));
        frac[a]=order ? f : 0;
    }
    for (unsigned corner=0;corner<8;++corner) {
        std::int64_t at[3]; bool needed=true;
        for (int a=0;a<3;++a) { const bool up=corner>>a&1; needed &= !up || frac[a]!=0; at[a]=lo[a]+up; }
        if (needed && !present(s,at,resident)) return 0;
    }
    return 1;
}
__global__ void valid_samples(Source s, const Grid* grids, const void* points, int bits, std::size_t count,
                              Origin base, int kind, int order, bool resident, unsigned char* out) {
    Grid g{};
    if (kind<2) { g=grids[blockIdx.y]; count=g.count; }
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x) {
        unsigned char ok;
        if (kind==0 || kind==3) {
            std::int64_t at[3];
            if (kind==0) {
                at[0]=g.base[0]+std::int64_t(i/(g.shape[1]*g.shape[2]));
                at[1]=g.base[1]+std::int64_t(i/g.shape[2]%g.shape[1]); at[2]=g.base[2]+std::int64_t(i%g.shape[2]);
            } else for (int a=0;a<3;++a) at[a]=bits==32 ? static_cast<const std::int32_t*>(points)[3*i+a] : static_cast<const std::int64_t*>(points)[3*i+a];
            ok=present(s,at,resident);
        } else {
            float c[3];
            if (kind==1) {
                const float z=float(i/(g.shape[1]*g.shape[2])), y=float(i/g.shape[2]%g.shape[1]), x=float(i%g.shape[2]);
                for (int a=0;a<3;++a) c[a]=sampling::affine(g.origin[a],g.axes+3*a,z,y,x);
            } else for (int a=0;a<3;++a) c[a]=static_cast<const float*>(points)[3*i+a];
            ok=supported(s,c,kind==1 ? g.base : base.v,order,resident);
        }
        out[g.offset+i]=ok;
    }
}
template<class O> __global__ void build_grids(const O* origins, const float* axes, std::size_t jobs, Origin shape,
                                              Origin fixed, bool explicit_origin, Grid* grids) {
    for (std::size_t j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;j<jobs;j+=std::size_t(blockDim.x)*gridDim.x) {
        Grid g{}; g.count=1;
        for (int a=0;a<3;++a) {
            g.shape[a]=shape.v[a]; g.count*=std::size_t(shape.v[a]);
            const O o=origins[3*j+a];
            if (explicit_origin) { g.base[a]=fixed.v[a]; g.origin[a]=float(o); }
            else {
                const double low=floor(double(o));
                g.base[a]=std::int64_t(low<0 ? 0 : low>double(1<<30) ? double(1<<30) : low);
                g.origin[a]=float(o-O(g.base[a]));
            }
        }
        for (int k=0;k<9;++k) g.axes[k]=axes[9*j+k];
        g.offset=j*g.count;
        grids[j]=g;
    }
}
template<class T> __global__ void scatter_blocks(const T* staging, void* const* targets, std::size_t n) {
    const T* from=staging+blockIdx.x*n; T* to=static_cast<T*>(targets[blockIdx.x]);
    for (std::size_t i=threadIdx.x;i<n;i+=blockDim.x) to[i]=from[i];
}
template<class Out, class In> __global__ void convert(const In* input, Out* output, std::size_t n) {
    for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(blockDim.x)*gridDim.x) output[i]=Out(input[i]);
}
}
void launch(Source s, const Grid* grids, std::size_t jobs, std::size_t largest,
        const void* points, int bits, std::size_t count, const std::int64_t* base,
        int kind, int order, bool floating, void* output, void* valid_output) {
    if (!count) return;
    auto* valid=static_cast<unsigned char*>(valid_output);
    Origin origin{}; if (base) std::copy_n(base,3,origin.v);
    const auto n=kind<2 ? largest : count;
    dim3 blocks(unsigned(std::min<std::size_t>((n+255)/256,65535)),unsigned(kind<2 ? jobs : 1));
    if (kind==1 && largest<(std::size_t(1)<<32)) {
        const bool pow2=s.shift[0]>=0 && s.shift[1]>=0 && s.shift[2]>=0;
        if (floating) {
            if (pow2) sample_affine<float,true><<<blocks,256,0,vcr::device::stream()>>>(s,grids,order,static_cast<float*>(output),valid);
            else sample_affine<float,false><<<blocks,256,0,vcr::device::stream()>>>(s,grids,order,static_cast<float*>(output),valid);
        } else {
            if (pow2) sample_affine<unsigned char,true><<<blocks,256,0,vcr::device::stream()>>>(s,grids,order,static_cast<unsigned char*>(output),valid);
            else sample_affine<unsigned char,false><<<blocks,256,0,vcr::device::stream()>>>(s,grids,order,static_cast<unsigned char*>(output),valid);
        }
        device::check(cudaGetLastError());
        return;
    }
    if (floating) sample<<<blocks,256,0,vcr::device::stream()>>>(s,grids,points,bits,count,origin,kind,order,static_cast<float*>(output),valid);
    else sample<<<blocks,256,0,vcr::device::stream()>>>(s,grids,points,bits,count,origin,kind,order,static_cast<unsigned char*>(output),valid);
    device::check(cudaGetLastError());
}
void scatter(const void* staging, void* const* targets, std::size_t count, std::size_t bytes) {
    if (!count) return;
    for (std::size_t first=0;first<count;first+=65535) {
        const unsigned n=unsigned(std::min<std::size_t>(count-first,65535));
        // Pool slots and staging offsets are multiples of `bytes` from 256-byte aligned bases.
        if (bytes%16==0) scatter_blocks<<<n,256,0,vcr::device::stream()>>>(static_cast<const uint4*>(staging)+first*(bytes/16),targets+first,bytes/16);
        else scatter_blocks<<<n,256,0,vcr::device::stream()>>>(static_cast<const unsigned char*>(staging)+first*bytes,targets+first,bytes);
    }
    device::check(cudaGetLastError());
}
void validity(Source s, const Grid* grids, std::size_t jobs, std::size_t largest,
        const void* points, int bits, std::size_t count, const std::int64_t* base,
        int kind, int order, bool resident, void* output) {
    if (!count) return;
    Origin origin{}; if (base) std::copy_n(base,3,origin.v);
    const auto n=kind<2 ? largest : count;
    dim3 blocks(unsigned(std::min<std::size_t>((n+255)/256,65535)),unsigned(kind<2 ? jobs : 1));
    valid_samples<<<blocks,256,0,vcr::device::stream()>>>(s,grids,points,bits,count,origin,kind,order,resident,static_cast<unsigned char*>(output));
    device::check(cudaGetLastError());
}
void device_grids(const void* origins, int bits, const float* axes, std::size_t jobs,
        const std::int64_t* shape, const std::int64_t* coordinate_origin, Grid* grids) {
    if (!jobs) return;
    Origin extent{}, fixed{}; std::copy_n(shape,3,extent.v);
    if (coordinate_origin) std::copy_n(coordinate_origin,3,fixed.v);
    const unsigned blocks=unsigned(std::min<std::size_t>((jobs+255)/256,65535));
    if (bits==64) build_grids<<<blocks,256,0,vcr::device::stream()>>>(static_cast<const double*>(origins),axes,jobs,extent,fixed,coordinate_origin!=nullptr,grids);
    else build_grids<<<blocks,256,0,vcr::device::stream()>>>(static_cast<const float*>(origins),axes,jobs,extent,fixed,coordinate_origin!=nullptr,grids);
    device::check(cudaGetLastError());
}
void cast(const void* input, void* output, std::size_t n, int in, int out) {
    if (!n) return;
    const unsigned blocks=unsigned(std::min<std::size_t>((n+255)/256,65535));
    if (in==out) device::check(cudaMemcpyAsync(output,input,n*in,cudaMemcpyDeviceToDevice,vcr::device::stream()));
    else if (in==1 && out==4) convert<<<blocks,256,0,vcr::device::stream()>>>(static_cast<const unsigned char*>(input),static_cast<float*>(output),n);
    else throw std::invalid_argument("CUDA reader conversion supports uint8 to float32");
    device::check(cudaGetLastError());
}
}
