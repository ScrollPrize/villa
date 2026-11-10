// Entry-count-realistic cache stress; only the immutable payloads are shared.
#ifdef VCR_BENCH_PRE_READER
#include "pre_reader/cache_adapter.hpp"
#else
#include <vc_reader/native/memory_cache.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using Clock=std::chrono::steady_clock;
constexpr std::size_t chunk_bytes=16*16*16;
std::string key(std::uint64_t id) { return "volume:synthetic/chunk/"+std::to_string(id)+"/channel/0"; }
struct alignas(64) Counts {
    std::uint64_t ops=0,reads=0,hits=0,puts=0;
    std::vector<std::uint64_t> latency;
};
std::uint64_t peak_rss_bytes() {
#if defined(__linux__) || defined(__APPLE__)
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    return std::uint64_t(usage.ru_maxrss)*
#ifdef __APPLE__
        1;
#else
        1024;
#endif
#else
    return 0;
#endif
}
int main(int argc,char** argv) {
    if(argc<6||argc>10) { std::cerr<<"usage: bench_cache_scale ENTRIES WORKERS SHARDS MAINTENANCE SECONDS [SCAN [PROFILE [WINDOWS [full]]]]\n";return 2; }
    const auto entries=std::stoull(argv[1]),workers=std::stoull(argv[2]),shards=std::stoull(argv[3]),maintenance=std::stoull(argv[4]);
    const double seconds=std::stod(argv[5]);
    const std::size_t scan=argc>6?std::stoull(argv[6]):0;
    const bool profile=argc>7 && std::stoi(argv[7]);
    const int windows=argc>8?std::stoi(argv[8]):6;
    const bool full=argc>9 && std::string_view(argv[9])=="full";
    const char* setup_env=std::getenv("VCR_BENCH_PREFILL_SECONDS");
    const double setup_seconds=setup_env?std::stod(setup_env):60;
    if(!(setup_seconds>0&&setup_seconds<=600))return 2;
    if(!entries||!workers||workers>1920||!shards||seconds<=0||seconds>2||windows<1||windows>15)return 2;
    const auto full_budget=entries*chunk_bytes;
    std::atomic<std::size_t> target_budget{full_budget};
    vcr::MemoryCache* cache_ptr=nullptr;
    vcr::MemoryCache::AutoOptions options;
    options.smoothing_seconds=0;options.refresh_interval=std::chrono::hours(24);
    options.available=[&] {
        const auto resident=cache_ptr?cache_ptr->nbytes():0;
        return 2*target_budget.load()-std::min(2*target_budget.load(),resident);
    };
    auto owner=full?std::make_unique<vcr::MemoryCache>(options,shards):std::make_unique<vcr::MemoryCache>(full_budget,shards);
    auto& cache=*owner;cache_ptr=&cache;
    // One reference-count block per producer, not a single contended shared_ptr.
    std::vector<vcr::ChunkPtr> payloads;
    for(std::size_t w=0;w<workers;++w)payloads.push_back(std::make_shared<vcr::Chunk>(chunk_bytes,std::uint8_t(w)));
    std::atomic<std::size_t> done{0};
    std::atomic<bool> cancel{false};
    std::atomic<std::int64_t> prefill_elapsed_ns{0};
    auto begin=Clock::now();
    std::vector<std::thread> pool;
    for(std::size_t w=0;w<workers;++w)pool.emplace_back([&,w]{
        for(std::size_t first=w*1024;first<entries && !cancel.load();first+=workers*1024) {
            const auto end=std::min<std::size_t>(entries,first+1024);
            for(auto i=first;i<end;++i)cache.put(key(i),payloads[w]);
            done.fetch_add(end-first,std::memory_order_relaxed);
        }
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
        auto old=prefill_elapsed_ns.load();
        while(old<elapsed&&!prefill_elapsed_ns.compare_exchange_weak(old,elapsed)){}
    });
    while(done.load()<entries) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const double age=std::chrono::duration<double>(Clock::now()-begin).count();
        std::cerr<<"prefill,"<<done.load()<<','<<entries<<','<<age<<std::endl;
        if(age>=setup_seconds) {cancel.store(true);break;}
    }
    for(auto& t:pool)t.join();pool.clear();
    const double prefill_seconds=prefill_elapsed_ns.load()/1e9;
    std::cerr<<"prefill_complete,"<<done.load()<<','<<prefill_seconds<<std::endl;
    if(done.load()!=entries) {std::cerr<<"INCOMPLETE: setup exceeded "<<setup_seconds<<" seconds; no capacity result claimed"<<std::endl;std::_Exit(3);}
    std::cerr<<"prefill_peak_rss_bytes,"<<peak_rss_bytes()<<std::endl;
    if(maintenance)cache.enable_maintenance(maintenance,scan);
    cache.enable_profiling(profile);
    // Keep phases within one populated cache. Phase 0 is a read-only control;
    // each write uses a fresh ID, so subsequent phases still cause eviction.
    std::cout<<"entries,workers,shards,maintenance,mode,repeat,seconds,ops_per_second,reads,hits,puts,p50_ns,p95_ns,p99_ns,evicted_entries,batches,scans,claims,assist_puts,headroom_bytes,resident_bytes,drain_seconds,get_wait_ns,put_wait_ns,get_hold_ns,put_hold_ns,end_resident_bytes,end_reclaiming_bytes,peak_sampled_overshoot\n";
    std::uint64_t next_base=entries;
    const std::vector<std::string_view> modes=full
        ?std::vector<std::string_view>{"hits","hot","miss","replace","mixed","evict","burst","paced","adaptive","mixed_shrink","allocate_evict","shrink_only"}
        :std::vector<std::string_view>{"hits","mixed","evict"};
    if(full)std::cerr<<"insert_result,"<<entries/prefill_seconds<<','<<prefill_seconds<<std::endl;
    for(const auto mode:modes)for(int repeat=0;repeat<(mode=="shrink_only"?1:mode=="hits"&&!full?1:windows);++repeat) {
        const bool shrink=mode=="mixed_shrink";
        if(full) {target_budget.store(full_budget);cache.refresh_budget();}
        // Previous background headroom work may evict initial low-number keys.
        // Hot-key contention must measure hits, not accidental absent keys.
        if(mode=="hot")for(std::size_t i=0;i<16;++i)cache.put(key(i),payloads[i%workers]);
        if(mode=="shrink_only") {
            const auto resident_before=cache.nbytes();
            const auto target=resident_before-resident_before/20;
            const auto shrink_begin=Clock::now();
            target_budget.store(target);cache.refresh_budget();
            while(cache.nbytes()>target && Clock::now()-shrink_begin<std::chrono::seconds(28))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            const auto elapsed=std::chrono::duration<double>(Clock::now()-shrink_begin).count();
            const auto reclaimed=(resident_before-cache.nbytes())/chunk_bytes;
            std::cerr<<"shrink_result,"<<reclaimed/elapsed<<','<<elapsed<<','<<reclaimed<<std::endl;
            if(cache.nbytes()>target){std::cerr<<"shrink did not complete within 28s"<<std::endl;std::_Exit(6);}
            continue;
        }
        std::vector<Counts> counts(workers);
        std::barrier start(static_cast<std::ptrdiff_t>(workers+1));
        Clock::time_point deadline;
        const auto before=cache.maintenance_stats();
        const auto prof_before=cache.profile();
        std::atomic<std::size_t> overshoot{0};
        for(std::size_t w=0;w<workers;++w)pool.emplace_back([&,w]{
            auto& c=counts[w];c.latency.reserve(4096);
            std::uint64_t rng=w+1,next=next_base+w;
            const auto insert=[&](const std::string& k) {
                const bool sample=(c.puts%64)==0;const auto t=sample?Clock::now():Clock::time_point{};
                if(mode=="allocate_evict") {
                    cache.put(k,std::make_shared<vcr::Chunk>(chunk_bytes,std::uint8_t(w)));
                    // Keep real allocation/free work, but never accumulate real
                    // payloads across the virtual-capacity fixture. Each request
                    // includes a second put replacing the new payload by a shared
                    // dummy. Other cases retain their original operation counts.
                    cache.put(k,payloads[w]);++c.puts;
                }
                else cache.put(k,payloads[w]);
                ++c.puts;
                if(sample)c.latency.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count());
            };
            start.arrive_and_wait();
            bool shrunk=false;
            const double rate=mode=="paced"?1000000:mode=="adaptive"?250000:0;
            do {
                const auto age=std::chrono::duration<double>(Clock::now()-begin).count();
                const bool burst=(mode=="burst"||mode=="adaptive")&&age>seconds/3&&age<2*seconds/3;
                for(int block=0;block<64;++block) {
                    rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
                    const bool mixed=mode=="mixed"||mode=="mixed_shrink"||mode=="burst"||mode=="paced"||mode=="adaptive";
                    if(mode=="replace") {
                        insert(key(rng%entries));
                    } else if(mode=="hits"||mode=="hot"||mode=="miss"||(mixed&&rng%10<(burst?1:8))) {
                        const auto id=mode=="miss"?UINT64_MAX-rng%entries:mode=="hot"?rng%16:mode=="hits"?rng%entries:rng%std::max<std::size_t>(1,entries/4);
                        const auto k=key(id);const auto hit=cache.get(k).first;
                        c.hits+=hit;++c.reads;
                        // A real reader refills a missed hot chunk. Without this,
                        // permanent losses make later windows artificially easy.
                        if(!hit && mixed)insert(k);
                    } else {
                        insert(key(next));next+=workers;
                    }
                    ++c.ops;
                }
                const auto used=cache.nbytes();
                if(used>full_budget) {
                    auto old=overshoot.load();
                    while(old<used-full_budget && !overshoot.compare_exchange_weak(old,used-full_budget)){}
                }
                if(shrink && w==0 && !shrunk && age>seconds/2) {
                    target_budget.store(full_budget-full_budget/20);cache.refresh_budget();shrunk=true;
                }
                if(rate)std::this_thread::sleep_until(begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(c.ops*workers/rate)));
            }while(Clock::now()<deadline);
        });
        begin=Clock::now();deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        start.arrive_and_wait();for(auto& t:pool)t.join();pool.clear();
        const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
        const auto end_resident=cache.nbytes(),end_reclaiming=cache.maintenance_stats().reclaiming_bytes;
        // No headroom-drain barrier: maintain pressure across sampling windows.
        const double drain_seconds=0;
        const auto after=cache.maintenance_stats();
        const auto prof_after=cache.profile();
        Counts sum;std::uint64_t max_puts=0;
        for(auto& c:counts) {sum.ops+=c.ops;sum.reads+=c.reads;sum.hits+=c.hits;sum.puts+=c.puts;max_puts=std::max(max_puts,c.puts);sum.latency.insert(sum.latency.end(),c.latency.begin(),c.latency.end());}
        if(mode=="hot" && sum.hits!=sum.reads){std::cerr<<"hot-key fixture lost a key"<<std::endl;std::_Exit(7);}
        next_base+=max_puts*workers;
        std::sort(sum.latency.begin(),sum.latency.end());
        const auto q=[&](double x){return sum.latency.empty()?0:sum.latency[std::size_t(x*(sum.latency.size()-1))];};
        std::cout<<entries<<','<<workers<<','<<shards<<','<<maintenance<<','<<mode<<','<<repeat<<','<<elapsed<<','<<sum.ops/elapsed<<','<<sum.reads<<','<<sum.hits<<','<<sum.puts<<','<<q(.5)<<','<<q(.95)<<','<<q(.99)<<','<<after.evicted_entries-before.evicted_entries<<','<<after.batches-before.batches<<','<<after.selection_scans-before.selection_scans<<','<<after.reclaim_claims-before.reclaim_claims<<','<<after.assist_puts-before.assist_puts<<','<<after.headroom_bytes<<','<<cache.nbytes()<<','<<drain_seconds<<','<<prof_after.get.wait_ns-prof_before.get.wait_ns<<','<<prof_after.put.wait_ns-prof_before.put.wait_ns<<','<<prof_after.get.hold_ns-prof_before.get.hold_ns<<','<<prof_after.put.hold_ns-prof_before.put.hold_ns<<','<<end_resident<<','<<end_reclaiming<<','<<overshoot.load()<<std::endl;
        if(maintenance && overshoot.load()) {std::cerr<<"hard capacity violation"<<std::endl;std::_Exit(5);}
        std::cerr<<"peak_rss_bytes,"<<mode<<','<<peak_rss_bytes()<<std::endl;
    }
    // This standalone, memory-only benchmark can hold 134 million entries.
    // Let the OS reclaim the large fixture instead of timing serial destructors.
    // Small smoke tests still exercise normal destruction and thread joining.
    if(entries>1000000) {std::cout.flush();std::cerr.flush();std::_Exit(0);}
}
