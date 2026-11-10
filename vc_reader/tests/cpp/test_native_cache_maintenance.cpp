#include <vc_reader/native/memory_cache.hpp>
#include <iostream>
#include <stdexcept>

void require(bool condition,const char* message) { if(!condition)throw std::runtime_error(message); }
int main() {
    try {
        for(auto scan:{0,8})for(auto workers:{1,2,4,8}) {
            std::atomic<std::uint64_t> available{2*1024*1024};
            vcr::MemoryCache::AutoOptions options;
            options.available=[&]{return available.load();};
            options.smoothing_seconds=0;
            options.refresh_interval=std::chrono::hours(24);
            vcr::MemoryCache cache(options,32);
            cache.enable_maintenance(workers,scan);
            std::vector<std::thread> producers;
            for(int t=0;t<8;++t)producers.emplace_back([&,t]{
                for(int i=0;i<10000;++i) {
                    const auto key=std::to_string(t*10000+i);
                    cache.put(key,std::make_shared<vcr::Chunk>(256,42));
                    const auto result=cache.get(key);
                    if(result.first && (!result.second || result.second->at(0)!=42))std::abort();
                }
            });
            for(auto& thread:producers)thread.join();
            require(cache.wait_for_maintenance(),"maintenance did not drain");
            require(cache.nbytes()<=cache.max_bytes(),"admission exceeded byte budget");
            require(cache.nbytes()==cache.size()*256,"settled byte accounting mismatch");
            const auto stats=cache.maintenance_stats();
            require(stats.bypasses==0,"fitting chunks must be admitted rather than bypassed");
            require(stats.evicted_bytes>0 && stats.batches>0,"no background eviction");
            require(stats.headroom_bytes<=cache.max_bytes()/20,"headroom exceeds 5% cap");
            available=0;cache.refresh_budget();
            require(cache.wait_for_maintenance(),"budget shrink did not drain");
            require(cache.nbytes()<=cache.max_bytes(),"shrink exceeded new budget");
            cache.clear();
            require(cache.wait_for_maintenance() && cache.nbytes()==0,"clear accounting mismatch");
        }
        for(auto scan:{0,8}) {
            std::atomic<std::uint64_t> free{512*1024};
            vcr::MemoryCache::AutoOptions options;
            options.available=[&]{return free.load();};options.smoothing_seconds=0;
            options.refresh_interval=std::chrono::hours(24);
            // Sampling is not a global-LRU guarantee. Verify its chosen-shard
            // ordering here; multi-shard hit retention is a benchmark metric.
            vcr::MemoryCache cache(options,scan?1:32);
            for(int i=0;i<800;++i)cache.put(std::to_string(i),std::make_shared<vcr::Chunk>(256));
            cache.enable_maintenance(4,scan);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            for(int i=0;i<32;++i)require(cache.get(std::to_string(i)).first,"hot setup missing");
            free=0;cache.refresh_budget();
            require(cache.wait_for_maintenance(),"recency test did not drain");
            for(int i=0;i<32;++i)require(cache.get(std::to_string(i)).first,scan?"sampled scan evicted a hot chunk while cold chunks remained":"cohort scan evicted a hot chunk while cold chunks remained");
        }
        // An outstanding reader retains a payload after its cache entry leaves.
        vcr::MemoryCache cache(4096,8);
        cache.enable_maintenance(2);
        auto held=std::make_shared<vcr::Chunk>(256,93);
        cache.put("held",held);
        auto retained=cache.get("held").second;
        require(retained!=nullptr,"initial admission failed");
        for(int i=0;i<10000;++i)cache.put(std::to_string(i),std::make_shared<vcr::Chunk>(256));
        require(cache.wait_for_maintenance() && retained->at(0)==93,"eviction invalidated reader payload");
        cache.put("large",std::make_shared<vcr::Chunk>(4096,19));
        require(cache.maintenance_stats().bypasses==0,"large fitting chunk cannot bypass admission");
        std::cout<<"Async admission, accounting, budget shrink, headroom and retained-reader tests passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
