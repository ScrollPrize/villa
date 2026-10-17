#pragma once
#include "../array.hpp"
#include "../cuda_buffer.hpp"
#include "types.hpp"
#include <unordered_set>
#include <set>
namespace vcr::reader_cuda {
struct Result { device::Storage storage; std::size_t count; bool floating; };
// One cache per array/device. Requests serialize so eviction cannot invalidate a
// running kernel. Outputs own independent storage and survive cache/reader close.
class Reader;
struct PointPlan {
    Reader* owner;
    std::vector<std::uint8_t> host;
    device::Storage coordinates;
    std::set<std::int64_t> needed;
    Index base;
    std::size_t count;
    int bits, kind, order;
};
class Reader {
    Array& array_;
    Source source_{};
    int device_;
    pid_t pid_=getpid();
    std::size_t budget_, chunk_bytes_=1, clock_=0;
    struct Entry { device::Storage data; std::size_t used; };
    std::unordered_map<std::int64_t,Entry> cache_;
    struct Staging {
        device::PinnedBuffer buffer;
        cudaEvent_t done = nullptr;
        int owner = 0;
        pid_t pid = getpid();
        void reserve(std::size_t bytes) {
            if (!done) {
                device::check(cudaGetDevice(&owner));
                device::check(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
            }
            buffer.reserve(bytes);
        }
        ~Staging() {
            if (!done || pid!=getpid()) return;
            int previous;
            if (cudaGetDevice(&previous)!=cudaSuccess || cudaSetDevice(owner)!=cudaSuccess) return;
            cudaEventDestroy(done);
            cudaSetDevice(previous);
        }
    } staging_[2];
    device::Storage keys_, pointers_, requests_;
    std::mutex mutex_;
    std::uint64_t hits_=0, uploads_=0, uploaded_=0, cpu_routes_=0, gpu_routes_=0;
    void ready() const {
        if (pid_!=getpid()) throw std::runtime_error("CUDA reader cannot be used after fork; create it in the worker");
    }
    void reserve(device::Storage& b, std::size_t bytes) {
        if (!b || b->bytes<bytes) b=device::allocate(bytes,device_);
    }
    Index index(std::int64_t key) const {
        Index idx(3); for (int a=2;a>=0;--a) { idx[a]=key%source_.grid[a]; key/=source_.grid[a]; } return idx;
    }
    template<class Set> void add(Set& keys, const std::int64_t* at) const {
        std::int64_t key=0;
        for (int a=0;a<3;++a) {
            if (at[a]<0 || at[a]>=source_.shape[a]) return;
            key=key*source_.grid[a]+at[a]/source_.chunks[a];
        }
        keys.insert(key);
    }
    void box(std::set<std::int64_t>& keys, const std::int64_t* start, const std::int64_t* size) const {
        std::int64_t lo[3],hi[3];
        for (int a=0;a<3;++a) {
            if (!size[a]) return;
            lo[a]=std::max<std::int64_t>(start[a],0); hi[a]=std::min(start[a]+size[a],source_.shape[a]);
            if (lo[a]>=hi[a]) return;
            lo[a]/=source_.chunks[a]; hi[a]=(hi[a]-1)/source_.chunks[a];
        }
        for (auto z=lo[0];z<=hi[0];++z) for (auto y=lo[1];y<=hi[1];++y) for (auto x=lo[2];x<=hi[2];++x)
            keys.insert((z*source_.grid[1]+y)*source_.grid[2]+x);
    }
    void upload_missing(const std::vector<std::int64_t>& missing) {
        if (missing.empty()) return;
        // Two bounded pinned buffers: CPU reading batch N+1 overlaps the
        // default-stream uploads of N. Events prevent reusing a buffer early.
        // Allocation/copies/sampling share stream 0, including cudaMallocAsync.
        constexpr std::size_t staging_limit=64ull<<20;
        const auto batch=std::max<std::size_t>(1,(staging_limit/2)/chunk_bytes_);
        // Allocate before enqueueing so pinning cannot serialize this pipeline.
        for (std::size_t slot=0;slot<2 && slot*batch<missing.size();++slot)
            staging_[slot].reserve(std::min(batch,missing.size()-slot*batch)*chunk_bytes_);
        // Growing the CUDA pool between copies can stall the submitting thread.
        // Reserve this request's already-budgeted destinations before uploads.
        std::vector<device::Storage> destinations;
        destinations.reserve(missing.size());
        for (std::size_t i=0;i<missing.size();++i)
            destinations.push_back(device::allocate(chunk_bytes_,device_));
        try {
            for (std::size_t first=0;first<missing.size();first+=batch) {
                auto& staging=staging_[(first/batch)%2];
                device::check(cudaEventSynchronize(staging.done));
                auto n=std::min(batch,missing.size()-first);
                std::vector<Index> starts, sizes(n,Index(source_.chunks,source_.chunks+3));
                std::vector<std::uint8_t*> outputs;
                for (std::size_t i=0;i<n;++i) {
                    auto idx=index(missing[first+i]); for (int a=0;a<3;++a) idx[a]*=source_.chunks[a];
                    starts.push_back(std::move(idx)); outputs.push_back(staging.buffer.data()+i*chunk_bytes_);
                }
                array_.read_many(starts,sizes,outputs);
                for (std::size_t i=0;i<n;++i) {
                    auto data=std::move(destinations[first+i]);
                    device::check(cudaMemcpyAsync(data->ptr,outputs[i],chunk_bytes_,cudaMemcpyHostToDevice,0));
                    cache_.emplace(missing[first+i],Entry{std::move(data),++clock_}); ++uploads_; uploaded_+=chunk_bytes_;
                }
                device::check(cudaEventRecord(staging.done,0));
            }
            device::complete();
        } catch (...) {
            // A later CPU read/allocation can fail with earlier uploads in
            // flight. Drain before staging can be reused or destroyed, and
            // discard entries whose asynchronous transfer may have failed.
            cudaStreamSynchronize(0);
            cache_.clear();
            throw;
        }
    }
    void prepare(const std::set<std::int64_t>& needed) {
        std::vector<std::int64_t> missing;
        for (auto key:needed) {
            auto it=cache_.find(key);
            if (it==cache_.end()) missing.push_back(key); else { it->second.used=++clock_; ++hits_; }
        }
        // Evict only entries outside this request. Request size was checked first.
        while ((cache_.size()+missing.size())*chunk_bytes_>budget_) {
            auto victim=cache_.end();
            for (auto it=cache_.begin();it!=cache_.end();++it)
                if (!needed.count(it->first) && (victim==cache_.end() || it->second.used<victim->second.used)) victim=it;
            if (victim==cache_.end()) throw std::runtime_error("CUDA cache budget exhausted");
            cache_.erase(victim);
        }
        upload_missing(missing);
        std::size_t capacity=1; while (capacity<needed.size()*2) capacity*=2;
        std::vector<std::int64_t> keys(capacity,-1);
        std::vector<const unsigned char*> pointers(capacity,nullptr);
        for (auto key:needed) {
            auto slot=(std::uint64_t(key)*UINT64_C(0x9e3779b97f4a7c15))&(capacity-1);
            while (keys[slot]>=0) slot=(slot+1)&(capacity-1);
            keys[slot]=key; pointers[slot]=static_cast<const unsigned char*>(cache_.at(key).data->ptr);
        }
        reserve(keys_,capacity*sizeof(std::int64_t)); reserve(pointers_,capacity*sizeof(void*));
        device::check(cudaMemcpy(keys_->ptr,keys.data(),keys.size()*sizeof(std::int64_t),cudaMemcpyHostToDevice));
        device::check(cudaMemcpy(pointers_->ptr,pointers.data(),pointers.size()*sizeof(void*),cudaMemcpyHostToDevice));
        source_.mask=capacity-1; source_.keys=static_cast<const std::int64_t*>(keys_->ptr);
        source_.pointers=static_cast<const unsigned char* const*>(pointers_->ptr);
    }
    bool route_cpu(const std::set<std::int64_t>& needed, std::size_t count, int kind, const std::string& route) {
        if (route!="auto" && route!="chunks" && route!="cpu") throw std::invalid_argument("source must be auto, chunks or cpu");
        const bool fits=needed.size()<=budget_/chunk_bytes_;
        if (route=="chunks" && !fits) throw std::invalid_argument("request exceeds CUDA chunk-cache budget; increase cache_bytes or use source='auto'");
        std::size_t missing=0; for (auto key:needed) missing+=!cache_.count(key);
        return route=="cpu" || !fits || (route=="auto" && kind==0 && missing>count/chunk_bytes_);
    }
    Result output(std::size_t n, bool floating) { return {device::allocate(n*(floating ? 4 : 1),device_),n,floating}; }
public:
    Reader(Array& array, std::size_t budget, int device) : array_(array),device_(device),budget_(budget) {
        const auto& spec=array.spec();
        if (array.ndim()!=3 || spec.itemsize!=1) throw std::invalid_argument("CUDA reader requires 3-D uint8 CT");
        device::Guard guard(device_);
        std::int64_t total=1;
        for (int a=0;a<3;++a) {
            if (spec.shape[a]>(1LL<<30) || spec.chunks[a]>(1LL<<30)) throw std::invalid_argument("CUDA reader dimensions must be at most 2**30");
            source_.shape[a]=spec.shape[a]; source_.chunks[a]=spec.chunks[a];
            source_.grid[a]=(spec.shape[a]+spec.chunks[a]-1)/spec.chunks[a];
            if (source_.grid[a] && total>INT64_MAX/source_.grid[a]) throw std::invalid_argument("chunk grid overflow");
            total*=source_.grid[a];
            if (chunk_bytes_>SIZE_MAX/std::size_t(spec.chunks[a])) throw std::invalid_argument("chunk size overflow");
            chunk_bytes_*=spec.chunks[a];
        }
        source_.fill=spec.fill[0];
    }
    int device_id() const { return device_; }
    void clear() { ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); cache_.clear(); device::complete(); }
    std::vector<std::uint64_t> stats() { ready(); std::lock_guard lock(mutex_); return {cache_.size()*chunk_bytes_,budget_,hits_,uploads_,uploaded_,cpu_routes_,gpu_routes_}; }
    Result grids(std::vector<Grid> grids, int kind, int order, bool floating, const std::string& route) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        if (grids.size()>65535 || (kind!=0 && kind!=1) || (order!=0 && order!=1) || (kind==1 && order==1 && !floating)) throw std::invalid_argument("invalid CUDA grid request");
        std::set<std::int64_t> needed;
        std::size_t total=0,largest=0;
        for (auto& g:grids) {
            g.count=1; g.offset=total;
            for (int a=0;a<3;++a) {
                if (g.shape[a]<0 || g.shape[a]>(1LL<<24) || (g.shape[a] && g.count>SIZE_MAX/4/std::size_t(g.shape[a]))) throw std::invalid_argument("invalid crop shape");
                if (g.base[a]>INT64_MAX-g.shape[a] || g.base[a]<-(1LL<<40) || g.base[a]>(1LL<<40)) throw std::invalid_argument("invalid crop origin");
                if (kind==1 && (g.base[a]<0 || g.base[a]>(1LL<<30))) throw std::invalid_argument("invalid affine integer origin");
                g.count*=g.shape[a];
                if (!std::isfinite(g.origin[a])) throw std::invalid_argument("nonfinite origin");
            }
            for (float value:g.axes) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite axes");
            if (g.count>SIZE_MAX/4-total) throw std::invalid_argument("batch size overflow");
            total+=g.count; largest=std::max(largest,g.count);
            if (!g.count) continue;
            if (kind==0) box(needed,g.base,g.shape);
            else {
                // Corner bounds in the same float32 evaluation order as sampling.
                std::int64_t lo[3],hi[3];
                for (int a=0;a<3;++a) { lo[a]=source_.shape[a]; hi[a]=-1; }
                for (unsigned corner=0;corner<8;++corner) for (int a=0;a<3;++a) {
                    float c=g.origin[a]; for (int b=0;b<3;++b) c=sampling::step(c,g.axes[3*a+b],float((corner>>b&1) ? g.shape[b]-1 : 0));
                    if (!std::isfinite(c)) throw std::invalid_argument("affine coordinates overflow float32");
                    const float bounded=std::clamp(c,float(-g.base[a]-2),float(source_.shape[a]-g.base[a]+2));
                    const auto low=std::floor(bounded);
                    auto at=g.base[a]+std::int64_t(low);
                    if (!order) { const float f=bounded-low; at+=(f>.5f || (f==.5f && (at&1))); }
                    lo[a]=std::min(lo[a],std::max<std::int64_t>(0,at));
                    hi[a]=std::max(hi[a],std::min(source_.shape[a],at+1+(order && bounded!=low)));
                }
                std::int64_t size[3]; for (int a=0;a<3;++a) size[a]=std::max<std::int64_t>(0,hi[a]-lo[a]);
                box(needed,lo,size);
            }
        }
        auto result=output(total,floating);
        if (!total) return result;
        if (route_cpu(needed,total,kind,route)) {
            ++cpu_routes_;
            std::vector<std::uint8_t> host(total*(kind==1 && order==1 ? 4 : 1));
            if (kind==0) {
                std::vector<Index> starts,sizes; std::vector<std::uint8_t*> outs;
                for (const auto& g:grids) { starts.emplace_back(g.base,g.base+3); sizes.emplace_back(g.shape,g.shape+3); outs.push_back(host.data()+g.offset); }
                array_.read_many(starts,sizes,outs);
            } else for (const auto& g:grids) {
                FloatAffine coords; for (int a=0;a<3;++a) { coords.shape[a]=g.shape[a]; coords.origin[a]=g.origin[a]; coords.offset[a]=g.base[a]; }
                std::copy_n(g.axes,9,coords.axes.begin()); array_.sample_u8_float32(coords,g.count,order,host.data()+g.offset*(order ? 4 : 1));
            }
            if (floating && (kind==0 || !order)) {
                reserve(requests_,total); device::check(cudaMemcpy(requests_->ptr,host.data(),total,cudaMemcpyHostToDevice)); cast(requests_->ptr,result.storage->ptr,total,1,4);
            } else device::check(cudaMemcpy(result.storage->ptr,host.data(),host.size(),cudaMemcpyHostToDevice));
        } else {
            ++gpu_routes_; prepare(needed); reserve(requests_,grids.size()*sizeof(Grid));
            device::check(cudaMemcpy(requests_->ptr,grids.data(),grids.size()*sizeof(Grid),cudaMemcpyHostToDevice));
            launch(source_,static_cast<const Grid*>(requests_->ptr),grids.size(),largest,nullptr,0,total,nullptr,kind,order,floating,result.storage->ptr);
        }
        device::complete(); return result;
    }
private:
    std::set<std::int64_t> discover_points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order) {
        if ((kind!=2 && kind!=3) || (bits!=32 && bits!=64) || (kind==2 && bits!=32) || (order!=0 && order!=1) || base.size()!=3 || count>SIZE_MAX/24) throw std::invalid_argument("invalid point request");
        for (auto b:base) if (b<0 || b>(1LL<<30)) throw std::invalid_argument("coordinate origin must be in [0, 2**30]");
        std::set<std::int64_t> needed;
        // Native discovery avoids Python's point enumeration/unique bottleneck.
        // The set is bounded by the touched chunks, not the number of samples.
        struct ChunkSet {
            ChunkIds ids;
            std::vector<std::int64_t> values;
            void insert(std::int64_t key) {
                const auto fresh=static_cast<std::uint32_t>(values.size());
                if (ids.find_or_insert(key,fresh)==fresh) values.push_back(key);
            }
        };
        constexpr std::size_t grain=65536;
        std::vector<ChunkSet> segments((count+grain-1)/grain);
        array_.parallel_for(segments.size(),[&](std::size_t segment) {
        auto& keys=segments[segment];
        for (std::size_t p=segment*grain;p<std::min(count,(segment+1)*grain);++p) {
            std::int64_t at[3];
            if (kind==3) {
                for (int a=0;a<3;++a) at[a]=bits==32 ? static_cast<const std::int32_t*>(points)[3*p+a] : static_cast<const std::int64_t*>(points)[3*p+a];
                add(keys,at);
            } else {
                const float* c=static_cast<const float*>(points)+3*p; bool outside=false;
                float frac[3];
                for (int a=0;a<3;++a) {
                    if (!(c[a]>float(-1-base[a]) && c[a]<float(source_.shape[a]-base[a])+(order ? 0.f : 1.f))) { outside=true; break; }
                    const float low=std::floor(c[a]); frac[a]=c[a]-low; at[a]=base[a]+std::int64_t(low);
                    if (!order) at[a]+=(frac[a]>.5f || (frac[a]==.5f && (at[a]&1)));
                }
                if (outside) continue;
                bool whole=true;
                for (int a=0;a<3;++a) whole &= at[a]>=0 && at[a]+1<source_.shape[a] && at[a]%source_.chunks[a]+1<source_.chunks[a];
                if (!order || whole) add(keys,at);
                else for (unsigned corner=0;corner<8;++corner) {
                    std::int64_t v[3]; for (int a=0;a<3;++a) v[a]=at[a]+((corner>>a&1) && frac[a]!=0);
                    add(keys,v);
                }
            }
        }
        });
        for (const auto& segment:segments) needed.insert(segment.values.begin(),segment.values.end());
        return needed;
    }
    Result execute_points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order,
                          const std::set<std::int64_t>& needed, const void* resident, const std::string& route) {
        const bool floating=kind==2 && order;
        auto result=output(count,floating); if (!count) return result;
        if (route_cpu(needed,count,kind,route)) {
            ++cpu_routes_; std::vector<std::uint8_t> host(count*(floating ? 4 : 1));
            if (kind==2) {
                FloatPoints coords; coords.data=static_cast<const float*>(points); coords.ndim=3; std::copy_n(base.begin(),3,coords.offset.begin());
                array_.sample_u8_float32(coords,count,order,host.data());
            } else {
                std::vector<std::int64_t> wide;
                const auto* ptr=static_cast<const std::int64_t*>(points);
                if (bits==32) { auto* first=static_cast<const std::int32_t*>(points); wide.assign(first,first+3*count); ptr=wide.data(); }
                array_.read_points(ptr,count,host.data());
            }
            device::check(cudaMemcpy(result.storage->ptr,host.data(),host.size(),cudaMemcpyHostToDevice));
        } else {
            ++gpu_routes_; prepare(needed);
            if (!resident) {
                const auto bytes=count*3*(bits/8); reserve(requests_,bytes);
                device::check(cudaMemcpy(requests_->ptr,points,bytes,cudaMemcpyHostToDevice));
                resident=requests_->ptr;
            }
            launch(source_,nullptr,0,0,resident,bits,count,base.data(),kind,order,floating,result.storage->ptr);
        }
        device::complete(); return result;
    }
public:
    Result points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order, const std::string& route) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        const auto needed=discover_points(points,count,bits,base,kind,order);
        return execute_points(points,count,bits,base,kind,order,needed,nullptr,route);
    }
    std::shared_ptr<PointPlan> prepare_points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        auto plan=std::make_shared<PointPlan>();
        plan->needed=discover_points(points,count,bits,base,kind,order);
        plan->owner=this; plan->count=count; plan->bits=bits; plan->base=base; plan->kind=kind; plan->order=order;
        const auto bytes=count*3*(bits/8);
        plan->host.resize(bytes); if (bytes) std::memcpy(plan->host.data(),points,bytes);
        plan->coordinates=device::allocate(bytes,device_);
        if (bytes) device::check(cudaMemcpy(plan->coordinates->ptr,points,bytes,cudaMemcpyHostToDevice));
        device::complete(); return plan;
    }
    Result read_plan(const PointPlan& plan, const std::string& route) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        if (plan.owner!=this) throw std::invalid_argument("point plan belongs to another reader");
        return execute_points(plan.host.data(),plan.count,plan.bits,plan.base,plan.kind,plan.order,plan.needed,plan.coordinates->ptr,route);

    }
};
}
