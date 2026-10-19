#pragma once
#include "../array.hpp"
#include "../cuda_buffer.hpp"
#include "types.hpp"
#include <bit>
#include <list>
#include <set>
namespace vcr::reader_cuda {
// valid: optional uint8 mask, 1 where every neighbour read lies in the volume
// (and, for source="resident", in a cached brick).
struct Result { device::Storage storage; std::size_t count; bool floating; device::Storage valid; };
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
    // The cache holds bricks: blocks of at most kBrick voxels per axis that evenly
    // divide storage chunks. Sampling reads only the bricks a request touches.
    static constexpr std::int64_t kBrick=32;
    std::size_t budget_, brick_bytes_=1, capacity_=0;
    // Bricks live in fixed slots of large device segments, in LRU order.
    struct Entry { std::uint32_t slot; std::list<std::int64_t>::iterator age; };
    std::unordered_map<std::int64_t,Entry> cache_;
    std::list<std::int64_t> lru_;
    std::vector<device::Storage> segments_;
    std::vector<std::uint32_t> free_;
    std::size_t segment_slots_=1, slots_=0;
    struct Staging {
        device::PinnedBuffer buffer;
        device::Storage device;
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
    device::Storage table_, requests_;
    device::PinnedBuffer table_host_;
    // Lookup over every cached brick for source="resident"; rebuilt after the
    // cache changes.
    device::Storage resident_table_, resident_requests_;
    Source resident_{};
    bool resident_dirty_=true;
    std::mutex mutex_;
    // Completion of this reader's latest request. Each request's stream waits
    // on it first, so reused scratch and lookup tables stay ordered when
    // asynchronous resident requests run on different streams.
    cudaEvent_t last_=nullptr;
    void chain() { if (last_) device::check(cudaStreamWaitEvent(device::stream(),last_,0)); }
    void mark() {
        if (!last_) device::check(cudaEventCreateWithFlags(&last_,cudaEventDisableTiming));
        device::check(cudaEventRecord(last_,device::stream()));
    }
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
            key=key*source_.grid[a]+at[a]/source_.brick[a];
        }
        keys.insert(key);
    }
    // Bricks overlapping a voxel box; `keep(brick start)` may reject candidates.
    template<class Keep> void box(std::set<std::int64_t>& keys, const std::int64_t* start, const std::int64_t* size, Keep&& keep) const {
        std::int64_t lo[3],hi[3];
        for (int a=0;a<3;++a) {
            if (!size[a]) return;
            lo[a]=std::max<std::int64_t>(start[a],0); hi[a]=std::min(start[a]+size[a],source_.shape[a]);
            if (lo[a]>=hi[a]) return;
            lo[a]/=source_.brick[a]; hi[a]=(hi[a]-1)/source_.brick[a];
        }
        for (auto z=lo[0];z<=hi[0];++z) for (auto y=lo[1];y<=hi[1];++y) for (auto x=lo[2];x<=hi[2];++x) {
            const std::int64_t at[3]{z*source_.brick[0],y*source_.brick[1],x*source_.brick[2]};
            if (keep(at)) keys.insert((z*source_.grid[1]+y)*source_.grid[2]+x);
        }
    }
    void box(std::set<std::int64_t>& keys, const std::int64_t* start, const std::int64_t* size) const {
        box(keys,start,size,[](const std::int64_t*) { return true; });
    }
    // Conservative separating-axis test between an affine crop's sample
    // parallelepiped and the coordinates whose interpolation can read a brick.
    // Any subset of axes is conservative; the margin dominates float32 rounding.
    struct Footprint {
        double axis[15][3], centre[15], radius[15], margin;
        int count=0;
        Footprint(const Grid& g) {
            double p[3], e[3][3], scale=0;
            for (int a=0;a<3;++a) {
                p[a]=g.origin[a];
                for (int k=0;k<3;++k) {
                    e[k][a]=double(g.axes[3*a+k])*double(std::max<std::int64_t>(g.shape[k]-1,0))/2;
                    p[a]+=e[k][a]; scale+=std::abs(e[k][a]);
                }
                scale+=std::abs(p[a]);
            }
            margin=1e-3+1e-5*scale;
            const auto push=[&](double x, double y, double z) {
                const double n=std::sqrt(x*x+y*y+z*z);
                if (!(n>1e-9)) return;
                auto& l=axis[count]; l[0]=x/n; l[1]=y/n; l[2]=z/n;
                centre[count]=l[0]*p[0]+l[1]*p[1]+l[2]*p[2]; radius[count]=0;
                for (int k=0;k<3;++k) radius[count]+=std::abs(l[0]*e[k][0]+l[1]*e[k][1]+l[2]*e[k][2]);
                ++count;
            };
            for (int a=0;a<3;++a) push(a==0,a==1,a==2);
            for (int k=0;k<3;++k) {
                const auto& u=e[k]; const auto& v=e[(k+1)%3];
                push(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);
            }
            for (int a=0;a<3;++a) for (int k=0;k<3;++k) {
                const double u[3]{double(a==0),double(a==1),double(a==2)}; const auto& v=e[k];
                push(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);
            }
        }
        // Local coordinates c can read voxel floor(c)+{0,1} (linear) or a rounded
        // neighbour (nearest), so a brick [lo, lo+size) needs c in [lo-1, lo+size].
        bool touches(const double* lo, const std::int64_t* size) const {
            double q[3], h[3];
            for (int a=0;a<3;++a) { q[a]=lo[a]+double(size[a]-1)/2; h[a]=double(size[a]+1)/2+margin; }
            for (int i=0;i<count;++i) {
                const auto& l=axis[i];
                const double extent=radius[i]+h[0]*std::abs(l[0])+h[1]*std::abs(l[1])+h[2]*std::abs(l[2]);
                if (std::abs(l[0]*q[0]+l[1]*q[1]+l[2]*q[2]-centre[i])>extent) return false;
            }
            return true;
        }
    };
    const unsigned char* slot_pointer(std::uint32_t slot) const {
        return static_cast<const unsigned char*>(segments_[slot/segment_slots_]->ptr)+std::size_t(slot%segment_slots_)*brick_bytes_;
    }
    std::uint32_t take_slot() {
        if (free_.empty()) {
            const auto n=std::min(segment_slots_,capacity_-slots_);
            segments_.push_back(device::allocate(n*brick_bytes_,device_));
            for (std::size_t i=n;i-->0;) free_.push_back(std::uint32_t(slots_+i));
            slots_+=n;
        }
        const auto slot=free_.back(); free_.pop_back(); return slot;
    }
    void drop_all() {
        resident_dirty_=true;
        cache_.clear(); lru_.clear(); free_.clear();
        for (std::size_t i=slots_;i-->0;) free_.push_back(std::uint32_t(i));
    }
    struct Miss { std::int64_t key; std::uint32_t slot; };
    // Uploads misses in order. Two bounded pinned buffers: CPU reading batch
    // N+1 overlaps the default-stream upload and scatter of N; each slot's event
    // prevents reusing its host/device staging early. Bricks are read
    // contiguously, uploaded with one copy, then scattered into their slots.
    // ready(n) runs once the first n misses are enqueued; work it enqueues on
    // the default stream is ordered after their transfer.
    template<class Ready> void upload(const std::vector<Miss>& missing, Ready&& ready) {
        if (missing.empty()) return;
        constexpr std::size_t staging_limit=64ull<<20;
        const auto batch=std::max<std::size_t>(1,(staging_limit/2)/(brick_bytes_+sizeof(void*)));
        const auto table=[&](std::size_t n) { return (n*brick_bytes_+15)/16*16; };
        for (std::size_t slot=0;slot<2 && slot*batch<missing.size();++slot) {
            const auto n=std::min(batch,missing.size()-slot*batch), bytes=table(n)+n*sizeof(void*);
            staging_[slot].reserve(bytes);
            if (!staging_[slot].device || staging_[slot].device->bytes<bytes) staging_[slot].device=device::allocate(bytes,device_);
        }
        for (std::size_t first=0;first<missing.size();first+=batch) {
            auto& staging=staging_[(first/batch)%2];
            device::check(cudaEventSynchronize(staging.done));
            auto n=std::min(batch,missing.size()-first);
            std::vector<Index> starts, sizes(n,Index(source_.brick,source_.brick+3));
            std::vector<std::uint8_t*> outputs;
            auto* targets=reinterpret_cast<const unsigned char**>(staging.buffer.data()+table(n));
            for (std::size_t i=0;i<n;++i) {
                auto idx=index(missing[first+i].key); for (int a=0;a<3;++a) idx[a]*=source_.brick[a];
                starts.push_back(std::move(idx)); outputs.push_back(staging.buffer.data()+i*brick_bytes_);
                targets[i]=slot_pointer(missing[first+i].slot);
            }
            array_.read_many(starts,sizes,outputs,nullptr,true);
            auto* device=static_cast<unsigned char*>(staging.device->ptr);
            device::check(cudaMemcpyAsync(device,staging.buffer.data(),table(n)+n*sizeof(void*),cudaMemcpyHostToDevice,device::stream()));
            scatter(device,reinterpret_cast<void* const*>(device+table(n)),n,brick_bytes_);
            device::check(cudaEventRecord(staging.done,device::stream()));
            uploads_+=n; uploaded_+=n*brick_bytes_;
            ready(first+n);
        }
    }
    // Refreshes a request's hits, evicts the oldest bricks outside it, and
    // assigns slots to its misses, which are entered now and filled by
    // upload(). Publishes the request's device lookup table.
    std::vector<Miss> admit(const std::set<std::int64_t>& needed) {
        std::vector<std::int64_t> missing;
        for (auto key:needed) {
            auto it=cache_.find(key);
            if (it==cache_.end()) missing.push_back(key);
            else { lru_.splice(lru_.end(),lru_,it->second.age); ++hits_; }
        }
        // Hits are now youngest: evict the oldest entries outside this request.
        // Request size was checked first.
        while (cache_.size()+missing.size()>capacity_) {
            if (lru_.empty() || needed.count(lru_.front())) throw std::runtime_error("CUDA cache budget exhausted");
            auto victim=cache_.find(lru_.front());
            free_.push_back(victim->second.slot); cache_.erase(victim); lru_.pop_front();
        }
        std::vector<Miss> misses;
        misses.reserve(missing.size());
        resident_dirty_ |= !missing.empty();
        for (auto key:missing) {
            const auto slot=take_slot();
            lru_.push_back(key); cache_.emplace(key,Entry{slot,std::prev(lru_.end())});
            misses.push_back({key,slot});
        }
        std::size_t capacity=1; while (capacity<needed.size()*2) capacity*=2;
        const auto bytes=capacity*sizeof(Slot);
        table_host_.reserve(bytes); reserve(table_,bytes);
        auto* table=reinterpret_cast<Slot*>(table_host_.data());
        fill_table(table,capacity,needed);
        // The previous request completed before returning, so the pinned table is free.
        device::check(cudaMemcpyAsync(table_->ptr,table_host_.data(),bytes,cudaMemcpyHostToDevice,device::stream()));
        source_.mask=capacity-1; source_.table=static_cast<const Slot*>(table_->ptr);
        return misses;
    }
    template<class Keys> void fill_table(Slot* table, std::size_t capacity, const Keys& keys) const {
        std::fill_n(table,capacity,Slot{-1,nullptr});
        for (const auto& item:keys) {
            const std::int64_t key=[&] { if constexpr (std::is_integral_v<std::decay_t<decltype(item)>>) return item; else return item.first; }();
            auto slot=home_slot(key,capacity-1);
            while (table[slot].key>=0) slot=(slot+1)&(capacity-1);
            table[slot]={key,slot_pointer(cache_.at(key).slot)};
        }
    }
    // Runs device work through the final synchronization. A failure, including
    // an asynchronous one, may leave admitted bricks unfilled: drain and drop them.
    template<class Work> void guarded(Work&& work) {
        try {
            work();
            device::complete();
        } catch (...) {
            cudaStreamSynchronize(device::stream());
            drop_all();
            throw;
        }
    }
    bool route_cpu(const std::set<std::int64_t>& needed, std::size_t count, int kind, const std::string& route) {
        if (route!="auto" && route!="chunks" && route!="cpu") throw std::invalid_argument("source must be auto, chunks or cpu");
        const bool fits=needed.size()<=capacity_;
        if (route=="chunks" && !fits) throw std::invalid_argument("request exceeds CUDA chunk-cache budget; increase cache_bytes or use source='auto'");
        std::size_t missing=0; for (auto key:needed) missing+=!cache_.count(key);
        return route=="cpu" || !fits || (route=="auto" && kind==0 && missing>count/brick_bytes_);
    }
    Result output(std::size_t n, bool floating, bool valid=false) {
        return {device::allocate(n*(floating ? 4 : 1),device_),n,floating,valid ? device::allocate(n,device_) : nullptr};
    }
    static void* mask(const Result& r) { return r.valid ? r.valid->ptr : nullptr; }
public:
    Reader(Array& array, std::size_t budget, int device) : array_(array),device_(device),budget_(budget) {
        const auto& spec=array.spec();
        if (array.ndim()!=3 || spec.itemsize!=1) throw std::invalid_argument("CUDA reader requires 3-D uint8 CT");
        device::Guard guard(device_);
        std::int64_t total=1;
        for (int a=0;a<3;++a) {
            if (spec.shape[a]>(1LL<<30) || spec.chunks[a]>(1LL<<30)) throw std::invalid_argument("CUDA reader dimensions must be at most 2**30");
            auto brick=spec.chunks[a];
            while (brick>kBrick && brick%2==0) brick/=2;
            source_.shape[a]=spec.shape[a]; source_.brick[a]=brick;
            source_.shift[a]=std::has_single_bit(std::uint64_t(brick)) ? std::countr_zero(std::uint64_t(brick)) : -1;
            source_.grid[a]=(spec.shape[a]+brick-1)/brick;
            if (source_.grid[a] && total>INT64_MAX/source_.grid[a]) throw std::invalid_argument("chunk grid overflow");
            total*=source_.grid[a];
            if (brick_bytes_>SIZE_MAX/std::size_t(brick)) throw std::invalid_argument("chunk size overflow");
            brick_bytes_*=brick;
        }
        source_.fill=spec.fill[0];
        capacity_=std::min<std::size_t>(budget_/brick_bytes_,UINT32_MAX);
        segment_slots_=std::max<std::size_t>(1,(256ull<<20)/brick_bytes_);
    }
    ~Reader() {
        if (!last_ || pid_!=getpid()) return;
        int previous; if (cudaGetDevice(&previous)!=cudaSuccess || cudaSetDevice(device_)!=cudaSuccess) return;
        cudaEventSynchronize(last_); cudaEventDestroy(last_); cudaSetDevice(previous);
    }
    int device_id() const { return device_; }
    void clear() {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        cache_.clear(); lru_.clear(); free_.clear(); segments_.clear(); slots_=0; resident_dirty_=true; device::complete();
    }
    std::vector<std::uint64_t> stats() { ready(); std::lock_guard lock(mutex_); return {cache_.size()*brick_bytes_,budget_,hits_,uploads_,uploaded_,cpu_routes_,gpu_routes_}; }
    Result grids(std::vector<Grid> grids, int kind, int order, bool floating, const std::string& route, bool valid=false, device::Storage destination={}) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        if (grids.size()>65535 || (kind!=0 && kind!=1) || (order!=0 && order!=1) || (kind==1 && order==1 && !floating)) throw std::invalid_argument("invalid CUDA grid request");
        std::set<std::int64_t> needed;
        std::vector<std::vector<std::int64_t>> footprints(grids.size());
        std::vector<std::pair<std::array<std::int64_t,3>,std::array<std::int64_t,3>>> bounds(grids.size());
        std::size_t total=0;
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
            total+=g.count;
            if (!g.count) continue;
            auto& [lo,size]=bounds[&g-grids.data()];
            if (kind==0) { std::copy_n(g.base,3,lo.begin()); std::copy_n(g.shape,3,size.begin()); }
            else {
                // Corner bounds in the same float32 evaluation order as sampling.
                std::int64_t hi[3];
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
                for (int a=0;a<3;++a) size[a]=std::max<std::int64_t>(0,hi[a]-lo[a]);
            }
        }
        // Brick footprints are independent per crop; enumerate them in parallel.
        array_.parallel_for(grids.size(),[&](std::size_t j) {
            const auto& g=grids[j]; const auto& [lo,size]=bounds[j];
            if (!g.count) return;
            std::set<std::int64_t> keys;
            if (kind==0) box(keys,lo.data(),size.data());
            else {
                const Footprint footprint(g);
                box(keys,lo.data(),size.data(),[&](const std::int64_t* at) {
                    const double local[3]{double(at[0]-g.base[0]),double(at[1]-g.base[1]),double(at[2]-g.base[2])};
                    return footprint.touches(local,source_.brick);
                });
            }
            footprints[j].assign(keys.begin(),keys.end());
        });
        {
            std::vector<std::int64_t> all;
            for (const auto& f:footprints) all.insert(all.end(),f.begin(),f.end());
            std::sort(all.begin(),all.end()); all.erase(std::unique(all.begin(),all.end()),all.end());
            needed.insert(all.begin(),all.end());
        }
        if (destination && (destination->bytes!=total*(floating ? 4 : 1) || destination->device!=device_))
            throw std::invalid_argument("output size/device mismatch");
        auto result=destination ? Result{std::move(destination),total,floating,valid ? device::allocate(total,device_) : nullptr}
                                : output(total,floating,valid);
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
                reserve(requests_,total); device::check(cudaMemcpyAsync(requests_->ptr,host.data(),total,cudaMemcpyHostToDevice,device::stream())); cast(requests_->ptr,result.storage->ptr,total,1,4);
            } else device::check(cudaMemcpyAsync(result.storage->ptr,host.data(),host.size(),cudaMemcpyHostToDevice,device::stream()));
            if (valid) {
                // CPU-sampled crops: geometric validity from the same descriptors.
                std::size_t widest=0; for (const auto& g:grids) widest=std::max(widest,g.count);
                reserve(requests_,grids.size()*sizeof(Grid));
                device::check(cudaMemcpyAsync(requests_->ptr,grids.data(),grids.size()*sizeof(Grid),cudaMemcpyHostToDevice,device::stream()));
                validity(source_,static_cast<const Grid*>(requests_->ptr),grids.size(),widest,nullptr,0,total,nullptr,kind,order,false,mask(result));
            }
        } else {
            ++gpu_routes_;
            guarded([&] {
                auto missing=admit(needed);
                // Upload misses in order of the first crop needing them; each
                // crop samples as soon as its bricks' transfers are enqueued.
                std::unordered_map<std::int64_t,std::size_t> position;
                for (std::size_t i=0;i<missing.size();++i) position.emplace(missing[i].key,i);
                std::vector<Miss> ordered; ordered.reserve(missing.size());
                std::vector<std::size_t> after(grids.size());
                for (std::size_t j=0;j<grids.size();++j) {
                    for (auto key:footprints[j]) {
                        auto it=position.find(key);
                        if (it!=position.end()) { ordered.push_back(missing[it->second]); position.erase(it); }
                    }
                    after[j]=ordered.size();
                }
                reserve(requests_,grids.size()*sizeof(Grid));
                device::check(cudaMemcpyAsync(requests_->ptr,grids.data(),grids.size()*sizeof(Grid),cudaMemcpyHostToDevice,device::stream()));
                const auto* device_grids=static_cast<const Grid*>(requests_->ptr);
                std::size_t launched=0;
                const auto sample=[&](std::size_t uploaded) {
                    std::size_t end=launched, widest=0;
                    for (;end<grids.size() && after[end]<=uploaded;++end) widest=std::max(widest,grids[end].count);
                    if (widest) launch(source_,device_grids+launched,end-launched,widest,nullptr,0,total,nullptr,kind,order,floating,result.storage->ptr,mask(result));
                    launched=end;
                };
                sample(0);
                upload(ordered,sample);
            });
            return result;
        }
        device::complete(); return result;
    }
private:
    const Source& resident_source() {
        if (!resident_dirty_) return resident_;
        std::size_t capacity=1; while (capacity<std::max<std::size_t>(1,cache_.size())*2) capacity*=2;
        std::vector<Slot> table(capacity); fill_table(table.data(),capacity,cache_);
        reserve(resident_table_,capacity*sizeof(Slot));
        // Synchronous: earlier resident kernels may still read the old table.
        device::complete();
        device::check(cudaMemcpyAsync(resident_table_->ptr,table.data(),capacity*sizeof(Slot),cudaMemcpyHostToDevice,device::stream()));
        resident_=source_; resident_.mask=capacity-1; resident_.table=static_cast<const Slot*>(resident_table_->ptr);
        resident_dirty_=false; return resident_;
    }
    // Validates crop shapes/origins and assigns offsets; returns the total count.
    static std::size_t layout(std::vector<Grid>& grids) {
        std::size_t total=0;
        for (auto& g:grids) {
            g.count=1; g.offset=total;
            for (int a=0;a<3;++a) {
                if (g.shape[a]<0 || g.shape[a]>(1LL<<24) || (g.shape[a] && g.count>SIZE_MAX/4/std::size_t(g.shape[a]))) throw std::invalid_argument("invalid crop shape");
                g.count*=g.shape[a];
            }
            if (g.count>SIZE_MAX/4-total) throw std::invalid_argument("batch size overflow");
            total+=g.count;
        }
        return total;
    }
public:
    // Loads the bricks of voxel boxes without producing output.
    void warm(std::vector<Grid> boxes) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        std::set<std::int64_t> needed;
        for (const auto& g:boxes) box(needed,g.base,g.shape);
        if (needed.size()>capacity_) throw std::invalid_argument("warm regions exceed CUDA chunk-cache budget; increase cache_bytes");
        guarded([&] { upload(admit(needed),[](std::size_t) {}); });
    }
    // Device-coordinate sampling from resident bricks only: no planning, no
    // host transfer of coordinates. Missing bricks read as fill.
    Result points_resident(const void* points, std::size_t count, int bits, const Index& base, int kind, int order, bool valid, bool sync=true) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        if ((kind!=2 && kind!=3) || (kind==2 && bits!=32) || (bits!=32 && bits!=64) || (order!=0 && order!=1)) throw std::invalid_argument("invalid point request");
        const bool floating=kind==2 && order; auto result=output(count,floating,valid);
        if (!count) return result;
        launch(resident_source(),nullptr,0,0,points,bits,count,base.data(),kind,order,floating,result.storage->ptr,mask(result));
        mark(); if (sync) device::complete(); return result;
    }
    // Affine crops of one shape from device origins/axes, resident bricks only.
    // Returns samples and, with valid, their validity (residency included).
    Result affine_resident(const void* origins, int bits, const float* axes, std::size_t jobs,
            const std::int64_t* shape, const std::int64_t* coordinate_origin, int order, bool floating, bool valid, bool sync=true) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        if (jobs>65535 || (order!=0 && order!=1) || (order==1 && !floating)) throw std::invalid_argument("invalid CUDA affine request");
        std::size_t count=1; for (int a=0;a<3;++a) { if (shape[a]<0 || shape[a]>(1LL<<24)) throw std::invalid_argument("invalid crop shape"); count*=std::size_t(shape[a]); }
        if (jobs && count>SIZE_MAX/4/jobs) throw std::invalid_argument("batch size overflow");
        auto result=output(jobs*count,floating,valid);
        if (!jobs || !count) return result;
        const auto& source=resident_source();
        reserve(resident_requests_,jobs*sizeof(Grid));
        auto* grids=static_cast<Grid*>(resident_requests_->ptr);
        device_grids(origins,bits,axes,jobs,shape,coordinate_origin,grids);
        launch(source,grids,jobs,count,nullptr,0,jobs*count,nullptr,1,order,floating,result.storage->ptr,mask(result));
        mark(); if (sync) device::complete(); return result;
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
                for (int a=0;a<3;++a) whole &= at[a]>=0 && at[a]+1<source_.shape[a] && at[a]%source_.brick[a]+1<source_.brick[a];
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
                          const std::set<std::int64_t>& needed, const void* resident, const std::string& route, bool valid) {
        const bool floating=kind==2 && order;
        auto result=output(count,floating,valid); if (!count) return result;
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
            device::check(cudaMemcpyAsync(result.storage->ptr,host.data(),host.size(),cudaMemcpyHostToDevice,device::stream()));
            if (valid) {
                const auto bytes=count*3*(bits/8); reserve(requests_,bytes);
                if (!resident) device::check(cudaMemcpyAsync(requests_->ptr,points,bytes,cudaMemcpyHostToDevice,device::stream()));
                validity(source_,nullptr,0,0,resident ? resident : requests_->ptr,bits,count,base.data(),kind,order,false,mask(result));
            }
        } else {
            ++gpu_routes_;
            guarded([&] {
                upload(admit(needed),[](std::size_t) {});
                if (!resident) {
                    const auto bytes=count*3*(bits/8); reserve(requests_,bytes);
                    device::check(cudaMemcpyAsync(requests_->ptr,points,bytes,cudaMemcpyHostToDevice,device::stream()));
                    resident=requests_->ptr;
                }
                launch(source_,nullptr,0,0,resident,bits,count,base.data(),kind,order,floating,result.storage->ptr,mask(result));
            });
            return result;
        }
        device::complete(); return result;
    }
public:
    Result points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order, const std::string& route, bool valid=false) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        const auto needed=discover_points(points,count,bits,base,kind,order);
        return execute_points(points,count,bits,base,kind,order,needed,nullptr,route,valid);
    }
    std::shared_ptr<PointPlan> prepare_points(const void* points, std::size_t count, int bits, const Index& base, int kind, int order) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        auto plan=std::make_shared<PointPlan>();
        plan->needed=discover_points(points,count,bits,base,kind,order);
        plan->owner=this; plan->count=count; plan->bits=bits; plan->base=base; plan->kind=kind; plan->order=order;
        const auto bytes=count*3*(bits/8);
        plan->host.resize(bytes); if (bytes) std::memcpy(plan->host.data(),points,bytes);
        plan->coordinates=device::allocate(bytes,device_);
        if (bytes) device::check(cudaMemcpyAsync(plan->coordinates->ptr,points,bytes,cudaMemcpyHostToDevice,device::stream()));
        device::complete(); return plan;
    }
    Result read_plan(const PointPlan& plan, const std::string& route, bool valid=false) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_);
        if (plan.owner!=this) throw std::invalid_argument("point plan belongs to another reader");
        chain();
        return execute_points(plan.host.data(),plan.count,plan.bits,plan.base,plan.kind,plan.order,plan.needed,plan.coordinates->ptr,route,valid);

    }
};
}
