#include "plan.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::aligned {
struct Job { Grid grid; const std::int64_t* keys; const void* const* tiles; std::size_t count; };
template<class T> __global__ void samples(const Job* jobs,T fill,void* out,bool* valid) {
    const auto& job=jobs[blockIdx.y];
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<job.grid.count;i+=std::size_t(blockDim.x)*gridDim.x)
        sample(job.grid,job.keys,reinterpret_cast<const T* const*>(job.tiles),job.count,fill,out,valid,i);
}
__global__ void validity(Grid g,bool* valid) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<g.count;i+=std::size_t(blockDim.x)*gridDim.x) {
        valid[output_index(g,i)]=supported(g,i);
    }
}
__global__ void fields(float* data,bool* valid,std::size_t n,Field f) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=std::size_t(blockDim.x)*gridDim.x) if(valid[i]) field(data,valid,n,i,f);
}
void launch_many(const std::vector<Grid>& grids,const std::vector<std::vector<std::int64_t>>& keys,
                 const std::vector<std::vector<const void*>>& tiles,const void* fill,ElementType type,void* out,bool* valid,int id) {
    if(grids.empty()) return;
    std::vector<std::int64_t> all_keys;
    std::vector<const void*> all_tiles;
    std::vector<std::size_t> offsets;
    std::size_t largest=0;
    for(std::size_t i=0;i<grids.size();++i) {
        offsets.push_back(all_keys.size()); largest=std::max(largest,grids[i].count);
        all_keys.insert(all_keys.end(),keys[i].begin(),keys[i].end());
        all_tiles.insert(all_tiles.end(),tiles[i].begin(),tiles[i].end());
    }
    if(!largest) return;
    auto k=device::allocate(all_keys.size()*8,id), t=device::allocate(all_tiles.size()*sizeof(void*),id);
    auto requests=device::allocate(grids.size()*sizeof(Job),id);
    std::vector<Job> jobs;
    for(std::size_t i=0;i<grids.size();++i) jobs.push_back({grids[i],
        all_keys.empty() ? nullptr : static_cast<std::int64_t*>(k->ptr)+offsets[i],
        all_tiles.empty() ? nullptr : static_cast<const void**>(t->ptr)+offsets[i],keys[i].size()});
    if(!all_keys.empty()) {
        device::check(cudaMemcpyAsync(k->ptr,all_keys.data(),k->bytes,cudaMemcpyHostToDevice,device::stream()));
        device::check(cudaMemcpyAsync(t->ptr,all_tiles.data(),t->bytes,cudaMemcpyHostToDevice,device::stream()));
    }
    device::check(cudaMemcpyAsync(requests->ptr,jobs.data(),requests->bytes,cudaMemcpyHostToDevice,device::stream()));
    type.visit_load([&](auto load) {
        samples<<<dim3(unsigned((largest+255)/256),unsigned(grids.size())),256,0,device::stream()>>>(
            static_cast<Job*>(requests->ptr),load(static_cast<const std::uint8_t*>(fill)),out,valid);
    });
    device::check(cudaGetLastError());
    // Host pointer tables must outlive their asynchronous copies.
    device::complete();
}
void launch(const Grid& g,const std::vector<std::int64_t>& keys,const std::vector<const void*>& tiles,const void* fill,
            ElementType type,void* out,bool* valid,int id) {
    launch_many({g},{keys},{tiles},fill,type,out,valid,id);
}
void launch_valid(const Grid& g,bool* valid) {
    if(g.count) validity<<<unsigned((g.count+255)/256),256,0,device::stream()>>>(g,valid);
    device::check(cudaGetLastError());
}
__global__ void box_valid(Grid g,bool* valid) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<g.count;i+=std::size_t(blockDim.x)*gridDim.x) {
        const std::int64_t p[3]{std::int64_t(i/g.shape[2]/g.shape[1])+g.base[0],
            std::int64_t(i/g.shape[2]%g.shape[1])+g.base[1],std::int64_t(i%g.shape[2])+g.base[2]};
        valid[i]=p[0]>=0 && p[0]<g.source[0] && p[1]>=0 && p[1]<g.source[1] && p[2]>=0 && p[2]<g.source[2];
    }
}
void launch_box(Triple source,Triple start,Triple shape,bool* valid) {
    Grid g; g.count=1;
    for(int a=0;a<3;++a) { g.source[a]=source[a]; g.base[a]=start[a]; g.shape[a]=shape[a]; g.count*=shape[a]; }
    if(g.count) box_valid<<<unsigned(std::min<std::size_t>((g.count+255)/256,65535)),256,0,device::stream()>>>(g,valid);
    device::check(cudaGetLastError());
}
void launch_field(float* data,bool* valid,std::size_t n,Field f) {
    if(n) fields<<<unsigned(std::min<std::size_t>((n+255)/256,65535)),256,0,device::stream()>>>(data,valid,n,f);
    device::check(cudaGetLastError());
}
}
