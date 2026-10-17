// Optional feasibility experiment, not part of the production reader build.
// nvcc -O3 -lineinfo -arch=sm_120 -std=c++20 -shared -Xcompiler=-fPIC
//   bench_cuda_sample.cu -o /tmp/libvc_sample.so
#include "../cpp/sampling_math.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
thread_local std::string error;
struct Geometry {
    int shape[3], start[3], volume[3], output[3];
    int chunks[3]{}, map_shape[3]{};
    const int* map = nullptr;
    float origin[3], axes[9];
    unsigned char fill;
};
struct Job {
    Geometry geometry;
    int box_start[3];
    std::size_t offset, count;
};
struct Context {
    unsigned char* source = nullptr;
    float* points = nullptr;
    int* map = nullptr;
    void* integers = nullptr;
    Job* jobs = nullptr;
    std::vector<Job> host_jobs;
    std::size_t max_job = 0;
    void* result = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    std::size_t bytes = 0, count = 0;
    Geometry g{};
    ~Context() {
        cudaFree(source); cudaFree(points); cudaFree(result); cudaFree(map); cudaFree(integers); cudaFree(jobs);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
struct Cell {
    using value_type = float;
    int lo[3];
    float frac[3];
    unsigned axes[3], k = 0;
};
struct Load {
    __host__ __device__ float operator()(const unsigned char* p) const { return *p; }
};
struct Corner {
    const unsigned char* source;
    Geometry g;
    Cell cell;
    __device__ const unsigned char* address(const int* absolute) const {
        int at[3];
        for (int a = 0; a < 3; ++a) {
            at[a] = absolute[a]-g.start[a];
            if (at[a] < 0 || at[a] >= g.shape[a]) return nullptr;
        }
        if (g.map) {
            const auto slot = (std::size_t(at[0]/g.chunks[0])*g.map_shape[1]+at[1]/g.chunks[1])*g.map_shape[2]+at[2]/g.chunks[2];
            const int chunk = g.map[slot];
            if (chunk < 0) return nullptr;
            const auto local = (std::size_t(at[0]%g.chunks[0])*g.chunks[1]+at[1]%g.chunks[1])*g.chunks[2]+at[2]%g.chunks[2];
            return source+std::size_t(chunk)*g.chunks[0]*g.chunks[1]*g.chunks[2]+local;
        }
        return source+(std::size_t(at[0])*g.shape[1]+at[1])*g.shape[2]+at[2];
    }
    __device__ float operator()(std::size_t corner) const {
        int at[3]{cell.lo[0], cell.lo[1], cell.lo[2]};
        for (unsigned a = 0; a < cell.k; ++a) at[cell.axes[a]] += (corner >> a) & 1;
        const auto* ptr = address(at);
        return ptr ? *ptr : g.fill;
    }
};
template<class Output, bool Affine, bool Linear>
__device__ Output sample_value(const unsigned char* source, const float* points,
                               std::size_t p, Geometry g) {
    float c[3];
    if constexpr (Affine) {
        const auto x = p % g.output[2], y = p / g.output[2] % g.output[1];
        const auto z = p / (std::size_t(g.output[1]) * g.output[2]);
        for (int a = 0; a < 3; ++a) {
            // Same float32 affine evaluation on CPU and GPU. Interpolation
            // may contract, but coordinate contraction can flip nearest ties.
            c[a] = vcr::sampling::affine(g.origin[a],g.axes+3*a,float(z),float(y),float(x));
        }
    } else {
        for (int a = 0; a < 3; ++a) c[a] = points[3*p+a];
    }
    Cell cell;
    bool finite = true, outside = false;
    for (int a = 0; a < 3; ++a) {
        finite &= isfinite(c[a]);
        outside |= !(c[a] > float(-1-g.start[a]) && c[a] < float(g.volume[a]-g.start[a]) + (Linear ? 0.f : 1.f));
    }
    if (!finite || outside) {
        return Linear && !finite ? Output(nan("")) : Output(g.fill);

    }
    for (int a = 0; a < 3; ++a) {
        const float low = floorf(c[a]), fraction = c[a]-low;
        const int integer = g.start[a]+int(low);
        cell.lo[a] = Linear ? integer : integer+(fraction>.5f || (fraction==.5f && (integer&1)));
        cell.frac[a] = Linear ? fraction : 0;
        if (cell.frac[a] != 0) cell.axes[cell.k++] = a;
    }
    Corner value{source, g, cell};
    if constexpr (!Linear) {
        return Output(value(0));
    } else {
        bool interior = cell.k == 3;
        for (int a = 0; a < 3; ++a)
            interior &= cell.lo[a] >= g.start[a] && cell.lo[a] + 1 < g.start[a] + g.shape[a];
        if (g.map) for (int a = 0; a < 3; ++a)
            interior &= (cell.lo[a]-g.start[a]) % g.chunks[a] + 1 < g.chunks[a];
        const auto* at = interior ? value.address(cell.lo) : nullptr;
        interior &= at != nullptr;
        if (interior) {
            const auto* dims = g.map ? g.chunks : g.shape;
            const std::int64_t strides[3]{std::int64_t(dims[1])*dims[2], dims[2], 1};
            Load load;
            return Output(vcr::sampling::trilinear(cell.frac, at, strides, load));
        } else return Output(vcr::sampling::interpolate(cell, value));
    }
}
template<class Output, bool Affine, bool Linear>
__global__ void sample(const unsigned char* source, const float* points, Output* out,
                       std::size_t n, Geometry g) {
    for (std::size_t p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < n; p += std::size_t(blockDim.x) * gridDim.x)
        out[p] = sample_value<Output,Affine,Linear>(source,points,p,g);
}
template<class T, bool Linear>
void launch(Context& c, bool affine) {
    const unsigned blocks = unsigned(std::min<std::size_t>((c.count+255)/256, 65535));
    if (affine) sample<T, true, Linear><<<blocks,256>>>(c.source,c.points,static_cast<T*>(c.result),c.count,c.g);
    else sample<T, false, Linear><<<blocks,256>>>(c.source,c.points,static_cast<T*>(c.result),c.count,c.g);
    check(cudaGetLastError());
}
__global__ void cast_crop(const unsigned char* input, float* output, std::size_t count) {
    for (std::size_t p=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x; p<count;
         p+=std::size_t(blockDim.x)*gridDim.x) output[p]=input[p];
}

// Integer gathers never convert indices to floating point.
template<class Index>
__global__ void gather_integer(const unsigned char* source, const Index* points,
                               unsigned char* out, std::size_t n, Geometry g) {
    for (std::size_t p=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x; p<n;
         p+=std::size_t(blockDim.x)*gridDim.x) {
        int at[3]; bool inside=true;
        for (int a=0;a<3;++a) {
            const auto v=points[3*p+a];
            inside &= v>=0 && v<g.volume[a];
            at[a]=(v>=0 && v<g.volume[a]) ? int(v) : 0;
        }
        Corner lookup{source,g,{}};
        const auto* ptr=inside ? lookup.address(at) : nullptr;
        out[p]=ptr ? *ptr : g.fill;
    }
}

template<class Output, bool Affine>
__global__ void read_batch(const unsigned char* source, const Job* jobs, Output* output) {
    const auto job=jobs[blockIdx.y];
    const auto g=job.geometry;
    for (std::size_t p=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x; p<job.count;
         p+=std::size_t(blockDim.x)*gridDim.x) {
        if constexpr (Affine) {
            output[job.offset+p]=sample_value<Output,true,true>(source,nullptr,p,g);
        } else {
            const int at[3]{job.box_start[0]+int(p/(std::size_t(g.output[1])*g.output[2])),
                            job.box_start[1]+int(p/g.output[2]%g.output[1]),
                            job.box_start[2]+int(p%g.output[2])};
            Corner lookup{source,g,{}};
            const auto* ptr=lookup.address(at);
            output[job.offset+p]=ptr ? Output(*ptr) : Output(g.fill);
        }
    }
}
template<class Output, bool Affine>
void launch_batch(Context& c, bool serial) {
    if (serial) {
        for (std::size_t i=0;i<c.host_jobs.size();++i) {
            const auto n=c.host_jobs[i].count;
            if (n) read_batch<Output,Affine><<<unsigned(std::min<std::size_t>((n+255)/256,65535)),256>>>(
                c.source,c.jobs+i,static_cast<Output*>(c.result));
        }
    } else if (c.max_job) {
        dim3 blocks(unsigned(std::min<std::size_t>((c.max_job+255)/256,65535)),unsigned(c.host_jobs.size()));
        read_batch<Output,Affine><<<blocks,256>>>(c.source,c.jobs,static_cast<Output*>(c.result));
    }
    check(cudaGetLastError());
}

}

extern "C" const char* sample_error() { return error.c_str(); }
extern "C" void* sample_create(const int* shape, const int* start, const int* volume,
                                std::size_t count, unsigned char fill) {
    Context* c = nullptr;
    try {
        c = new Context;
        c->count = count; c->bytes = 1; c->g.fill = fill;
        if (!count) throw std::runtime_error("empty workload");
        for (int a = 0; a < 3; ++a) {
            if (shape[a] <= 0 || start[a] < 0 || shape[a] > volume[a]-start[a])
                throw std::runtime_error("source box must be inside volume");
            c->g.shape[a]=shape[a]; c->g.start[a]=start[a]; c->g.volume[a]=volume[a];
            c->bytes *= shape[a];
        }
        check(cudaMalloc(&c->source,c->bytes));
        check(cudaMalloc(&c->points,count*3*sizeof(float)));
        check(cudaMalloc(&c->result,count*sizeof(float)));
        check(cudaEventCreate(&c->begin)); check(cudaEventCreate(&c->end));
        return c;
    } catch (const std::exception& e) { error=e.what(); delete c; return nullptr; }
}
extern "C" void sample_destroy(void* handle) { delete static_cast<Context*>(handle); }
extern "C" int sample_chunks(void* handle, const int* chunks, const int* map_shape,
                              const int* map, std::size_t count) {
    try {
        auto& c = *static_cast<Context*>(handle);
        std::size_t slots = 1;
        c.bytes = count;
        for (int a=0;a<3;++a) {
            c.g.chunks[a]=chunks[a]; c.g.map_shape[a]=map_shape[a];
            c.bytes *= chunks[a]; slots *= map_shape[a];
        }
        check(cudaFree(c.source)); c.source=nullptr;
        check(cudaMalloc(&c.source,c.bytes));
        check(cudaMalloc(&c.map,slots*sizeof(int)));
        check(cudaMemcpy(c.map,map,slots*sizeof(int),cudaMemcpyHostToDevice));
        c.g.map=c.map;
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_download(void* handle, void* output, std::size_t bytes) {
    try {
        auto& c = *static_cast<Context*>(handle);
        if (bytes > c.count*sizeof(float)) throw std::runtime_error("output too large");
        check(cudaMemcpy(output,c.result,bytes,cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_run(void* handle, const void* source, const void* points, void* output,
                           const int* shape, const float* origin, const float* axes,
                           int order, int out64, int uploads, float* kernel_ms) {
    try {
        auto& c = *static_cast<Context*>(handle);
        const bool affine = shape != nullptr;
        if (affine) {
            for (int a=0;a<3;++a) { c.g.output[a]=shape[a]; c.g.origin[a]=origin[a]; }
            for (int a=0;a<9;++a) c.g.axes[a]=axes[a];
        }
        if (uploads & 1) check(cudaMemcpy(c.source,source,c.bytes,cudaMemcpyHostToDevice));
        if (!affine && (uploads & 2)) check(cudaMemcpy(c.points,points,c.count*3*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaEventRecord(c.begin));
        if (!order) launch<unsigned char,false>(c,affine);
        else if (out64) throw std::runtime_error("float32 experiment has no float64 path");
        else launch<float,true>(c,affine);
        check(cudaEventRecord(c.end));
        if (output) check(cudaMemcpy(output,c.result,c.count*(order ? 4 : 1),cudaMemcpyDeviceToHost));
        else check(cudaEventSynchronize(c.end));
        check(cudaEventElapsedTime(kernel_ms,c.begin,c.end));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}


extern "C" int sample_upload_output(void* handle, const void* output, std::size_t bytes,
                                     std::size_t offset) {
    try {
        auto& c=*static_cast<Context*>(handle);
        if (offset>c.count*4 || bytes>c.count*4-offset) throw std::runtime_error("output too large");
        check(cudaMemcpy(static_cast<unsigned char*>(c.result)+offset,output,bytes,cudaMemcpyHostToDevice));
        check(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_upload_cast(void* handle, const void* raw) {
    try {
        auto& c=*static_cast<Context*>(handle);
        check(cudaMemcpy(c.points,raw,c.count,cudaMemcpyHostToDevice));
        cast_crop<<<unsigned(std::min<std::size_t>((c.count+255)/256,65535)),256>>>(
            reinterpret_cast<unsigned char*>(c.points),static_cast<float*>(c.result),c.count);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_integer_init(void* handle) {
    try {
        auto& c=*static_cast<Context*>(handle);
        check(cudaMalloc(&c.integers,c.count*3*sizeof(std::int64_t)));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_integer_run(void* handle, const void* source, const void* points,
                                  int bits, int uploads, float* kernel_ms) {
    try {
        auto& c=*static_cast<Context*>(handle);
        if (!c.integers || (bits!=32 && bits!=64)) throw std::runtime_error("invalid integer request");
        if (uploads&1) check(cudaMemcpy(c.source,source,c.bytes,cudaMemcpyHostToDevice));
        if (uploads&2) check(cudaMemcpy(c.integers,points,c.count*3*(bits/8),cudaMemcpyHostToDevice));
        check(cudaEventRecord(c.begin));
        const unsigned blocks=unsigned(std::min<std::size_t>((c.count+255)/256,65535));
        if (bits==64) gather_integer<<<blocks,256>>>(c.source,static_cast<const std::int64_t*>(c.integers),
            static_cast<unsigned char*>(c.result),c.count,c.g);
        else gather_integer<<<blocks,256>>>(c.source,static_cast<const std::int32_t*>(c.integers),
            static_cast<unsigned char*>(c.result),c.count,c.g);
        check(cudaGetLastError()); check(cudaEventRecord(c.end)); check(cudaEventSynchronize(c.end));
        check(cudaEventElapsedTime(kernel_ms,c.begin,c.end));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_batch_init(void* handle, int count, const int* shapes,
                                 const int* starts, const float* origins, const float* axes) {
    try {
        auto& c=*static_cast<Context*>(handle);
        if (count<1 || count>65535) throw std::runtime_error("invalid batch size");
        c.host_jobs.resize(count); c.max_job=0;
        std::size_t offset=0;
        for (int i=0;i<count;++i) {
            auto& j=c.host_jobs[i]; j.geometry=c.g; j.offset=offset; j.count=1;
            for (int a=0;a<3;++a) {
                if (shapes[3*i+a]<0 || (shapes[3*i+a] && j.count>c.count/std::size_t(shapes[3*i+a])))
                    throw std::runtime_error("invalid job shape");
                j.geometry.output[a]=shapes[3*i+a]; j.count*=shapes[3*i+a];
                j.box_start[a]=starts ? starts[3*i+a] : 0;
                j.geometry.origin[a]=origins ? origins[3*i+a] : 0;
            }
            for (int a=0;a<9;++a) j.geometry.axes[a]=axes ? axes[9*i+a] : 0;
            if (j.count>c.count-offset) throw std::runtime_error("batch exceeds output capacity");
            offset+=j.count; c.max_job=std::max(c.max_job,j.count);
        }
        if (offset!=c.count) throw std::runtime_error("batch does not match output capacity");
        check(cudaMalloc(&c.jobs,count*sizeof(Job)));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
extern "C" int sample_batch_run(void* handle, const void* source, int affine,
                                int floating, int upload, int serial, float* kernel_ms) {
    try {
        auto& c=*static_cast<Context*>(handle);
        if (!c.jobs || (affine && !floating)) throw std::runtime_error("invalid batch request");
        if (upload) check(cudaMemcpy(c.source,source,c.bytes,cudaMemcpyHostToDevice));
        // Small per-request descriptors are included even for resident source data.
        check(cudaMemcpy(c.jobs,c.host_jobs.data(),c.host_jobs.size()*sizeof(Job),cudaMemcpyHostToDevice));
        check(cudaEventRecord(c.begin));
        if (affine) launch_batch<float,true>(c,serial);
        else if (floating) launch_batch<float,false>(c,serial);
        else launch_batch<unsigned char,false>(c,serial);
        check(cudaEventRecord(c.end)); check(cudaEventSynchronize(c.end));
        check(cudaEventElapsedTime(kernel_ms,c.begin,c.end));
        return 0;
    } catch (const std::exception& e) { error=e.what(); return -1; }
}
