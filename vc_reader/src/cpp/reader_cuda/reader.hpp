#pragma once
#include "../array.hpp"
#include "../cuda_buffer.hpp"
#include "types.hpp"
#include <bit>
#include <list>
#include <mutex>
#include <set>
namespace vcr::reader_cuda {
// valid: optional uint8 mask, 1 where every neighbour read lies in the volume
// (and, for source="resident", in a cached brick).
struct Result { device::Storage storage; std::size_t count; bool floating; device::Storage valid; };
// One cache per array/device. Requests serialize so eviction cannot invalidate a
// running kernel. Outputs own independent storage and survive cache/reader close.
class Reader;
// Device brick slots shared by readers on one device (e.g. the pyramid levels of a
// volume): one byte budget, one least-recently-used order across all their bricks and
// one lock, so a level's bricks make room for another's. Readers keep their pool alive.
class Pool {
public:
    // A slot holds one brick: up to 32^3 uint8 voxels unless slot_bytes says otherwise.
    static constexpr std::size_t kSlotBytes=32*32*32;
    Pool(std::size_t budget, int device, std::size_t slot_bytes=kSlotBytes)
        : budget_(budget),slot_bytes_(std::max<std::size_t>(1,slot_bytes)),device_(device) {
        capacity_=std::min<std::size_t>(budget_/slot_bytes_,UINT32_MAX);
        segment_slots_=std::max<std::size_t>(1,(256ull<<20)/slot_bytes_);
    }
    Pool(const Pool&)=delete;
    Pool& operator=(const Pool&)=delete;
    std::size_t budget() const { return budget_; }
    std::size_t slot_bytes() const { return slot_bytes_; }
    std::size_t capacity() const { return capacity_; }
    int device() const { return device_; }
    // Bytes of resident bricks (slots in use), across the pool's readers.
    std::size_t resident_bytes() { std::lock_guard lock(mutex_); return lru_.size()*slot_bytes_; }
private:
    friend class Reader;
    struct Age { Reader* reader; std::int64_t key; };
    std::recursive_mutex mutex_;
    std::size_t budget_, slot_bytes_, capacity_=0, segment_slots_=1, slots_=0;
    int device_;
    std::list<Age> lru_;
    std::vector<device::Storage> segments_;
    std::vector<std::uint32_t> free_;
    const unsigned char* pointer(std::uint32_t slot) const {
        return static_cast<const unsigned char*>(segments_[slot/segment_slots_]->ptr)+std::size_t(slot%segment_slots_)*slot_bytes_;
    }
    std::uint32_t take() {
        if (free_.empty()) {
            const auto n=std::min(segment_slots_,capacity_-slots_);
            segments_.push_back(device::allocate(n*slot_bytes_,device_));
            for (std::size_t i=n;i-->0;) free_.push_back(std::uint32_t(slots_+i));
            slots_+=n;
        }
        const auto slot=free_.back(); free_.pop_back(); return slot;
    }
    // With no brick resident, gives the device memory back.
    void release_if_unused() {
        if (!lru_.empty()) return;
        segments_.clear(); free_.clear(); slots_=0;
    }
};
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
    std::size_t brick_bytes_=1;
    // Bricks live in the pool's fixed slots, in the pool's LRU order.
    std::shared_ptr<Pool> pool_;
    struct Entry { std::uint32_t slot; std::list<Pool::Age>::iterator age; };
    std::unordered_map<std::int64_t,Entry> cache_;
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
    std::recursive_mutex& mutex_;  // the pool's: requests of all its readers serialize
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
    const unsigned char* slot_pointer(std::uint32_t slot) const { return pool_->pointer(slot); }
    std::size_t capacity() const { return pool_->capacity_; }
    // Gives this reader's slots back to the pool.
    void drop_all() {
        resident_dirty_=true;
        for (auto& [key,entry]:cache_) { pool_->lru_.erase(entry.age); pool_->free_.push_back(entry.slot); }
        cache_.clear();
    }
    void evict(const Pool::Age& victim) {
        auto& owner=*victim.reader;
        auto it=owner.cache_.find(victim.key);
        pool_->free_.push_back(it->second.slot);
        owner.cache_.erase(it);
        owner.resident_dirty_=true;
        pool_->lru_.pop_front();
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
    // upload(). Publishes the request's device lookup table unless told not to.
    std::vector<Miss> admit(const std::set<std::int64_t>& needed, bool publish=true) {
        std::vector<std::int64_t> missing;
        for (auto key:needed) {
            auto it=cache_.find(key);
            if (it==cache_.end()) missing.push_back(key);
            else { pool_->lru_.splice(pool_->lru_.end(),pool_->lru_,it->second.age); ++hits_; }
        }
        // Hits are now youngest: evict the oldest bricks of the pool outside this
        // request, whichever reader holds them. Request size was checked first.
        auto& lru=pool_->lru_;
        std::set<Reader*> others;
        while (lru.size()+missing.size()>capacity()) {
            if (lru.empty() || (lru.front().reader==this && needed.count(lru.front().key)))
                throw std::runtime_error("CUDA cache budget exhausted");
            if (lru.front().reader!=this) others.insert(lru.front().reader);
            evict(lru.front());
        }
        // Another reader's last work may still read the slots it gave up.
        for (auto* other:others)
            if (other->last_) device::check(cudaStreamWaitEvent(device::stream(),other->last_,0));
        std::vector<Miss> misses;
        misses.reserve(missing.size());
        resident_dirty_ |= !missing.empty();
        for (auto key:missing) {
            const auto slot=pool_->take();
            lru.push_back({this,key}); cache_.emplace(key,Entry{slot,std::prev(lru.end())});
            misses.push_back({key,slot});
        }
        if (!publish) return misses;
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
        const bool fits=needed.size()<=capacity();
        if (route=="chunks" && !fits) throw std::invalid_argument("request exceeds CUDA chunk-cache budget; increase cache_bytes or use source='auto'");
        std::size_t missing=0; for (auto key:needed) missing+=!cache_.count(key);
        return route=="cpu" || !fits || (route=="auto" && kind==0 && missing>count/brick_bytes_);
    }
    Result output(std::size_t n, bool floating, bool valid=false) {
        return {device::allocate(n*(floating ? 4 : 1),device_),n,floating,valid ? device::allocate(n,device_) : nullptr};
    }
    static void* mask(const Result& r) { return r.valid ? r.valid->ptr : nullptr; }
public:
    // The bytes of one brick of `array` (its slots must hold at least this many).
    static std::size_t brick_bytes(const Array& array) {
        const auto& spec=array.spec();
        std::size_t bytes=std::max<std::size_t>(1,spec.itemsize);
        for (std::size_t a=0;a<spec.chunks.size();++a) {
            auto brick=spec.chunks[a];
            while (brick>kBrick && brick%2==0) brick/=2;
            bytes*=std::size_t(std::max<std::int64_t>(1,brick));
        }
        return bytes;
    }
    static std::shared_ptr<Pool> pooled(std::shared_ptr<Pool> pool) {
        if (!pool) throw std::invalid_argument("a CUDA reader needs a pool");
        return pool;
    }
    // A reader with a pool of its own.
    Reader(Array& array, std::size_t budget, int device)
        : Reader(array,std::make_shared<Pool>(budget,device,brick_bytes(array))) {}
    // A reader sharing `pool` (on the pool's device) with other readers.
    Reader(Array& array, std::shared_ptr<Pool> pool)
        : array_(array),device_(pool ? pool->device() : 0),pool_(pooled(std::move(pool))),mutex_(pool_->mutex_) {
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
        if (brick_bytes_>pool_->slot_bytes()) throw std::invalid_argument("the array's bricks do not fit the CUDA pool's slots");
    }
    ~Reader() {
        if (pid_!=getpid()) return;
        std::lock_guard lock(mutex_);
        // This reader's last work may still read its slots: finish it before giving them back.
        if (last_) {
            int previous;
            if (cudaGetDevice(&previous)==cudaSuccess && cudaSetDevice(device_)==cudaSuccess) {
                cudaEventSynchronize(last_); cudaEventDestroy(last_); cudaSetDevice(previous);
            }
        }
        drop_all();
        pool_->release_if_unused();
    }
    int device_id() const { return device_; }
    const std::shared_ptr<Pool>& pool() const { return pool_; }
    // Readers holding the same lock (one pool) share it; take several in this order.
    const void* lock_id() const { return &mutex_; }
    // Runs f(source) with the table of resident bricks, under this reader's lock, on its
    // device and the current stream. f enqueues device work that reads the bricks and
    // may return with it in flight: the work is ordered (chain/mark) before any later
    // request that evicts or replaces them, so the lock is not held while it runs.
    // The caller completes its stream before reading the work's results.
    template<class F> void with_resident(F&& f) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        f(resident_source());
        mark();
    }
    // Makes the bricks with these keys (see Source: z-major brick-grid indices)
    // resident, reading them from the array. Throws when they exceed the cache budget.
    // Once `cancel` is set while reading, the bricks read in full are still made
    // resident, then Cancelled is thrown.
    void load_bricks(const std::vector<std::int64_t>& keys, const std::atomic<bool>* cancel=nullptr) {
        ready();
        std::set<std::int64_t> needed;
        const std::int64_t total=source_.grid[0]*source_.grid[1]*source_.grid[2];
        for (auto key:keys) if (key>=0 && key<total) needed.insert(key);
        if (needed.size()>capacity()) throw std::invalid_argument("request exceeds CUDA brick-cache budget; increase cache_bytes");
        // Before locking, read the bricks this reader lacks and stage them on the device:
        // with_resident promises a frame the bricks already here, so the lock is held
        // only to assign slots and enqueue their scatter, never across a read or a sync.
        std::vector<std::int64_t> staged;
        {
            std::lock_guard lock(mutex_);
            for (auto key:needed) if (!cache_.count(key)) staged.push_back(key);
        }
        device::Guard guard(device_);
        device::Storage bricks;
        bool cancelled=false;
        if (!staged.empty()) {
            std::vector<std::uint8_t> host(staged.size()*brick_bytes_);
            std::vector<Index> starts, sizes(staged.size(),Index(source_.brick,source_.brick+3));
            std::vector<std::uint8_t*> outputs;
            for (std::size_t i=0;i<staged.size();++i) {
                auto idx=index(staged[i]); for (int a=0;a<3;++a) idx[a]*=source_.brick[a];
                starts.push_back(std::move(idx)); outputs.push_back(host.data()+i*brick_bytes_);
            }
            std::vector<std::uint8_t> complete;
            try { array_.read_many(starts,sizes,outputs,nullptr,false,cancel,&complete); }
            catch (const Cancelled&) { cancelled=true; }
            if (cancelled) {
                // Keep the bricks read in full (in order, so staged stays sorted); the
                // others are no longer needed by this load.
                std::size_t kept=0;
                for (std::size_t i=0;i<staged.size();++i) {
                    if (!complete[i]) { needed.erase(staged[i]); continue; }
                    if (kept!=i) {
                        staged[kept]=staged[i];
                        std::memcpy(host.data()+kept*brick_bytes_,host.data()+i*brick_bytes_,brick_bytes_);
                    }
                    ++kept;
                }
                staged.resize(kept); host.resize(kept*brick_bytes_);
                if (staged.empty()) throw Cancelled("read cancelled");
            }
            bricks=device::allocate(host.size(),device_);
            // Pageable: the host buffer is free once this returns.
            device::check(cudaMemcpyAsync(bricks->ptr,host.data(),host.size(),cudaMemcpyHostToDevice,device::stream()));
        }
        std::lock_guard lock(mutex_); chain();
        try {
            // Staged bricks another reader made resident meanwhile scatter into a spare
            // brick; bricks evicted since staging upload as before.
            std::unordered_map<std::int64_t,const unsigned char*> target;
            std::vector<Miss> rest;
            for (const auto& miss:admit(needed,false)) {
                if (std::binary_search(staged.begin(),staged.end(),miss.key)) target.emplace(miss.key,slot_pointer(miss.slot));
                else rest.push_back(miss);
            }
            if (!target.empty()) {
                const auto spare=device::allocate(brick_bytes_,device_);
                std::vector<const unsigned char*> targets(staged.size(),static_cast<const unsigned char*>(spare->ptr));
                for (std::size_t i=0;i<staged.size();++i)
                    if (auto it=target.find(staged[i]);it!=target.end()) targets[i]=it->second;
                const auto table=device::allocate(targets.size()*sizeof(void*),device_);
                device::check(cudaMemcpyAsync(table->ptr,targets.data(),targets.size()*sizeof(void*),cudaMemcpyHostToDevice,device::stream()));
                scatter(bricks->ptr,static_cast<void* const*>(table->ptr),staged.size(),brick_bytes_);
                uploads_+=target.size(); uploaded_+=target.size()*brick_bytes_;
            }
            upload(rest,[](std::size_t) {});
        } catch (...) {
            cudaStreamSynchronize(device::stream());
            drop_all();
            throw;
        }
        mark();
        if (cancelled) throw Cancelled("read cancelled");
    }
    void clear() {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        device::complete();
        drop_all(); pool_->release_if_unused();
    }
    std::vector<std::uint64_t> stats() { ready(); std::lock_guard lock(mutex_); return {cache_.size()*brick_bytes_,pool_->budget(),hits_,uploads_,uploaded_,cpu_routes_,gpu_routes_}; }
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
                    if (!order) at+=nearest_rounds_up(bounded-low);
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
        // A new table: earlier resident kernels may still read the old one, which is
        // freed on this stream, after them (callers chain() first).
        auto published=device::allocate(capacity*sizeof(Slot),device_);
        device::check(cudaMemcpyAsync(published->ptr,table.data(),capacity*sizeof(Slot),cudaMemcpyHostToDevice,device::stream()));
        if (resident_table_) resident_table_->stream=device::stream();
        resident_table_=std::move(published);
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
        if (needed.size()>capacity()) throw std::invalid_argument("warm regions exceed CUDA chunk-cache budget; increase cache_bytes");
        guarded([&] { upload(admit(needed),[](std::size_t) {}); });
    }
    // Device-coordinate sampling from resident bricks only: no planning, no
    // host transfer of coordinates. Missing bricks read as fill.
    Result points_resident(const void* points, std::size_t count, int bits, const Index& base, int kind, int order, bool valid, bool sync=true) {
        ready(); std::lock_guard lock(mutex_); device::Guard guard(device_); chain();
        if ((kind!=2 && kind!=3) || (bits!=32 && bits!=64) || (order!=0 && order!=1)) throw std::invalid_argument("invalid point request");
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
        if ((kind!=2 && kind!=3) || (bits!=32 && bits!=64) || (order!=0 && order!=1) || base.size()!=3 || count>SIZE_MAX/24) throw std::invalid_argument("invalid point request");
        for (auto b:base) if (b<0 || b>(1LL<<30)) throw std::invalid_argument("coordinate origin must be in [0, 2**30]");
        std::set<std::int64_t> needed;
        // C++ chunk discovery avoids Python's point enumeration/unique bottleneck.
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
        const auto discover = [&]<class Coord>() {
        array_.parallel_for(segments.size(),[&](std::size_t segment) {
        auto& keys=segments[segment];
        for (std::size_t p=segment*grain;p<std::min(count,(segment+1)*grain);++p) {
            std::int64_t at[3];
            if (kind==3) {
                for (int a=0;a<3;++a) at[a]=bits==32 ? static_cast<const std::int32_t*>(points)[3*p+a] : static_cast<const std::int64_t*>(points)[3*p+a];
                add(keys,at);
            } else {
                const Coord* c=static_cast<const Coord*>(points)+3*p; bool outside=false;
                Coord frac[3];
                for (int a=0;a<3;++a) {
                    if (!(c[a]>Coord(-1-base[a]) && c[a]<Coord(source_.shape[a]-base[a])+(order ? 0.f : 1.f))) { outside=true; break; }
                    const Coord low=std::floor(c[a]); frac[a]=c[a]-low; at[a]=base[a]+std::int64_t(low);
                    if (!order) at[a]+=nearest_rounds_up(frac[a]);
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
        };
        if (kind==2 && bits==64) discover.template operator()<double>();
        else discover.template operator()<float>();
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
                const auto sample = [&]<class Coord>() {
                    RebasedCoordinates<PointCoordinates<Coord>> coords;
                    coords.data=static_cast<const Coord*>(points); coords.ndim=3;
                    std::copy_n(base.begin(),3,coords.offset.begin());
                    array_.sample_u8_float32(coords,count,order,host.data());
                };
                if (bits==64) sample.template operator()<double>();
                else sample.template operator()<float>();
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
