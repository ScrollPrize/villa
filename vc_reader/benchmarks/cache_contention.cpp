// No I/O or decoding: stress the production decoded-cache bookkeeping.
#include <vc_reader/native/memory_cache.hpp>
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <numeric>
#ifdef __linux__
#include <sched.h>
#endif

using Clock=std::chrono::steady_clock;
constexpr size_t keysCount=65536, chunkBytes=256, capacity=4096;
size_t cacheShards=vcr::MemoryCache::default_shards();
size_t fixedOperations=0;
size_t maintenanceWorkers=0;
size_t candidateScan=0;
double operationsPerSecond=0;
size_t cores() {
#ifdef __linux__
    cpu_set_t set;
    if(sched_getaffinity(0,sizeof(set),&set)==0) return CPU_COUNT(&set);
#endif
    return std::max(1u,std::thread::hardware_concurrency());
}
struct alignas(64) Count {
    uint64_t operations=0,hits=0,shrinks=0,reads=0,puts=0,overshoot=0;
    std::vector<std::uint64_t> put_ns;
};

void trial(const std::string& mode,size_t workers,int repeat,double duration) {
    std::vector<std::string> keys;
    std::vector<vcr::ChunkPtr> payloads;
    for(size_t i=0;i<keysCount;++i) {
        keys.push_back("volume:synthetic/chunk/"+std::to_string(i)+"/channel/0");
        payloads.push_back(std::make_shared<vcr::Chunk>(chunkBytes,uint8_t(i)));
    }
    // Kept payload references exclude allocator/destructor bandwidth. The
    // allocate_evict case instead creates a fresh payload for each insertion.
    std::atomic<uint64_t> available{capacity*chunkBytes*2};
    vcr::MemoryCache::AutoOptions options;
    options.available=[&]{return available.load(std::memory_order_relaxed);};
    options.smoothing_seconds=0; options.refresh_interval=std::chrono::hours(24);
    auto cache=mode=="mixed_shrink" ? std::make_unique<vcr::MemoryCache>(options,cacheShards)
        : std::make_unique<vcr::MemoryCache>((mode=="hits"||mode=="hot"||mode=="replace"||mode=="insert") ? keysCount*chunkBytes*2 : capacity*chunkBytes,cacheShards);
    if(mode!="insert") for(size_t i=0;i<(mode=="hits"||mode=="hot"||mode=="replace" ? keysCount : capacity);++i)
        cache->put(keys[i],payloads[i]);
    if(maintenanceWorkers)cache->enable_maintenance(maintenanceWorkers,candidateScan);
    std::vector<Count> counts(workers);
    for(auto& count:counts)count.put_ns.reserve(2048);
    std::mutex mutex; std::condition_variable cv;
    size_t ready=0; bool start=false,abort=false;
    Clock::time_point deadline;
    Clock::time_point begin;
    std::vector<std::thread> threads;
    try {
        for(size_t w=0;w<workers;++w) threads.emplace_back([&,w] {
            {std::unique_lock lock(mutex); if(++ready==workers)cv.notify_all();cv.wait(lock,[&]{return start;});if(abort)return;}
            auto& count=counts[w]; uint64_t rng=w+1;
            size_t next=w; auto nextShrink=Clock::now();
            auto nextReport=Clock::now();
            do {
                const double age=std::chrono::duration<double>(Clock::now()-begin).count();
                const bool burst=mode=="burst" && age>duration/3 && age<2*duration/3;
                for(int block=0;block<64;++block) {
                    if(fixedOperations && count.operations>=fixedOperations/workers+(w<fixedOperations%workers))return;
                    rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
                    size_t k=rng%keysCount;
                    if(mode=="insert") {if(next>=keysCount)return;k=next;next+=workers;}
                    if(mode=="hot")k%=16;
                    if(mode=="hits"||mode=="hot"||mode=="miss"||((mode=="mixed"||mode=="mixed_shrink"||mode=="burst")&&(rng%10<(burst?1:8)))) {
                        if(mode=="mixed"||mode=="mixed_shrink"||mode=="burst")k%=1024;
                        auto value=cache->get(mode=="miss"?keys[capacity+k%(keysCount-capacity)]:keys[k]);
                        count.hits+=value.first;
                        ++count.reads;
                    } else {
                        const bool sample=(count.puts++%64)==0;
                        const auto start=sample?Clock::now():Clock::time_point{};
                        if(mode=="allocate_evict")cache->put(keys[k],std::make_shared<vcr::Chunk>(chunkBytes,uint8_t(k)));
                        else cache->put(keys[k],payloads[k]);
                        if(sample)count.put_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());
                    }
                    ++count.operations;
                }
                const auto bytes=cache->nbytes(),budget=cache->max_bytes();
                count.overshoot=std::max(count.overshoot,bytes>budget?bytes-budget:0);
                if(mode=="burst" && w==0 && Clock::now()>=nextReport) {
                    const auto stats=cache->maintenance_stats();
                    std::cerr<<"headroom,"<<age<<','<<stats.headroom_bytes<<','<<bytes<<','<<stats.bypasses<<','<<stats.evicted_bytes<<'\n';
                    nextReport=Clock::now()+std::chrono::milliseconds(10);
                }
                if(mode=="mixed_shrink" && w==0 && Clock::now()>=nextShrink) {
                    available.store(count.shrinks%2 ? capacity*chunkBytes*2 : 0,std::memory_order_relaxed);
                    cache->refresh_budget();++count.shrinks;
                    nextShrink=Clock::now()+std::chrono::milliseconds(10);
                }
                if(operationsPerSecond>0)
                    std::this_thread::sleep_until(begin+std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(count.operations*workers/operationsPerSecond)));
            } while(mode=="insert" || fixedOperations || Clock::now()<deadline);
        });
    } catch(const std::system_error& error) {
        {std::lock_guard lock(mutex);abort=true;start=true;}cv.notify_all();
        for(auto& thread:threads)thread.join();
        std::cout<<"UNAVAILABLE,"<<workers<<","<<threads.size()<<","<<error.what()<<std::endl;
        throw;
    }
    {std::unique_lock lock(mutex);cv.wait(lock,[&]{return ready==workers;});}
    begin=Clock::now();
    {std::lock_guard lock(mutex);deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(duration));start=true;}cv.notify_all();
    for(auto& thread:threads)thread.join();
    const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
    // Optional headroom replenishment after foreground work is not measured.
    const double drain_seconds=0;
    uint64_t ops=0,hits=0,shrinks=0,reads=0,puts=0,overshoot=0;
    std::vector<std::uint64_t> latencies;
    for(auto& c:counts){ops+=c.operations;hits+=c.hits;shrinks+=c.shrinks;reads+=c.reads;puts+=c.puts;
        overshoot=std::max(overshoot,c.overshoot);latencies.insert(latencies.end(),c.put_ns.begin(),c.put_ns.end());}
    std::sort(latencies.begin(),latencies.end());
    const auto quantile=[&](double q){return latencies.empty()?0:latencies[std::size_t(q*(latencies.size()-1))];};
    const auto stats=cache->maintenance_stats();
    if((mode=="hits"||mode=="hot")&&hits!=ops)throw std::runtime_error("hit-only workload missed");
    if(mode=="insert"&&cache->size()!=keysCount)throw std::runtime_error("insert-only lost entries");
    if(cache->nbytes()>cache->max_bytes())throw std::runtime_error("cache exceeded its budget");
    std::cout<<mode<<','<<workers<<','<<repeat<<','<<elapsed<<','<<ops<<','<<ops/elapsed<<','<<ops/elapsed/workers<<','<<hits<<','<<shrinks
        <<','<<reads<<','<<puts<<','<<stats.bypasses<<','<<stats.admitted_bytes<<','<<stats.evicted_bytes<<','<<stats.headroom_bytes
        <<','<<quantile(.5)<<','<<quantile(.95)<<','<<quantile(.99)<<','<<overshoot<<','<<drain_seconds<<','<<stats.batches
        <<','<<stats.assist_puts<<','<<stats.assist_bytes<<','<<stats.assist_batches
        <<','<<stats.selection_scans<<','<<stats.reclaim_claims<<','<<stats.evicted_entries<<std::endl;
}
void shrinkTrial(int repeat) {
    vcr::MemoryCache::AutoOptions options;
    uint64_t available=capacity*chunkBytes*2;
    options.available=[&]{return available;}; options.smoothing_seconds=0;
    options.refresh_interval=std::chrono::hours(24);
    vcr::MemoryCache cache(options,cacheShards);
    std::vector<std::string> keys;std::vector<vcr::ChunkPtr> values;
    for(size_t i=0;i<capacity;++i){keys.push_back(std::to_string(i));values.push_back(std::make_shared<vcr::Chunk>(chunkBytes));}
    double elapsed=0;size_t evicted=0;
    for(int cycle=0;cycle<32;++cycle) {
        available=capacity*chunkBytes*2;cache.refresh_budget();
        for(size_t i=0;i<capacity;++i)cache.put(keys[i],values[i]);
        available=0;const auto before=cache.size();const auto begin=Clock::now();
        cache.refresh_budget();elapsed+=std::chrono::duration<double>(Clock::now()-begin).count();
        evicted+=before-cache.size();
    }
    if(evicted!=32*capacity/2)throw std::runtime_error("shrink did not evict half the cache");
    std::cout<<"shrink_only,1,"<<repeat<<','<<elapsed<<','<<evicted<<','<<evicted/elapsed<<','<<evicted/elapsed<<",0,32"<<std::endl;
}
int main(int argc,char** argv) {
    if(argc==2&&std::string(argv[1])=="cores"){std::cout<<cores()<<'\n';return 0;}
    const size_t workers=argc>1?std::stoul(argv[1]):1;
    const int repeats=argc>2?std::stoi(argv[2]):3;
    const double duration=argc>3?std::stod(argv[3]):.25;
    const std::string selected=argc>4?argv[4]:"all";
    if(argc>5)cacheShards=std::stoul(argv[5]);
    if(argc>6)fixedOperations=std::stoul(argv[6]);
    if(argc>7)maintenanceWorkers=std::stoul(argv[7]);
    if(argc>8)operationsPerSecond=std::stod(argv[8]);
    if(argc>9)candidateScan=std::stoul(argv[9]);
    if(!workers||workers>10*cores()||repeats<1||repeats>5||duration<=0||duration>2)return 2;
    std::cout<<"case,workers,repeat,seconds,operations,chunks_per_second,chunks_per_second_per_thread,hits,refreshes,reads,puts,bypasses,admitted_bytes,evicted_bytes,headroom_bytes,put_p50_ns,put_p95_ns,put_p99_ns,peak_sampled_overshoot,drain_seconds,batches,assist_puts,assist_bytes,assist_batches,selection_scans,reclaim_claims,evicted_entries\n";
    try {
        if(workers==1&&(selected=="all"||selected=="shrink_only"))for(int r=0;r<repeats;++r)shrinkTrial(r);
        for(const auto* mode:{"hits","hot","miss","insert","replace","evict","allocate_evict","mixed","mixed_shrink","burst"})
            if(selected=="all"||selected==mode)
            for(int r=0;r<repeats;++r)trial(mode,workers,r,duration);
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
