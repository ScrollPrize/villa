#include "vc/core/render/ChunkCache.hpp"
#include "../src/render/SamplingWaitProfile.hpp"
#include <nlohmann/json.hpp>
#include <atomic>
#include <barrier>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

using namespace vc::render;
using Json=nlohmann::json;
namespace {
class MemoryFetcher final:public IChunkFetcher {
public:
    std::atomic<size_t> count{0};
    ChunkFetchResult fetch(const ChunkKey& key) override {
        ++count;ChunkFetchResult r;r.status=ChunkFetchStatus::Found;
        r.bytes.assign(4096,std::byte(1+key.ix%200));return r;
    }
};
}
int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"Usage: bench_chunk_cache_contention MAX_WORKERS OUTPUT\n";return 2;}
    try {
        const size_t maximum=std::stoul(argv[1]);if(!maximum)throw std::invalid_argument("workers must be positive");
        auto service=std::make_shared<ChunkCacheService>(ChunkCacheService::batchOptions(maximum,maximum*(1<<20)));
        Json results=Json::array();
        for(size_t workers:{size_t(1),size_t(4),size_t(16),maximum}){
            if(workers>maximum)continue;
            for(bool local:{false,true})for(bool overlapping:{false,true})for(int repeat=0;repeat<3;++repeat){
                auto fetcher=std::make_shared<MemoryFetcher>();
                ChunkCache::LevelInfo info;info.shape={16,16,int(workers*4*16)};info.chunkShape={16,16,16};
                const auto name="cache-bench-"+std::to_string(results.size());
                auto shared=service->acquireSource(name,{info},{fetcher},0,ChunkDtype::UInt8,{});
                std::vector<std::shared_ptr<ChunkCacheService>> partitions;
                std::vector<std::shared_ptr<ChunkCache>> sources;
                for(size_t w=0;w<workers;++w){
                    if(local){auto partition=service->createDecodedPartition(1<<20);sources.push_back(shared->inService(partition));partitions.push_back(partition);}
                    else sources.push_back(shared);
                    for(int i=0;i<4;++i)sources.back()->getChunkBlocking(0,0,0,int(overlapping?i:w*4+i));
                }
                const auto initialReads=fetcher->count.load();
                std::barrier start(int(workers+1)),finish(int(workers+1));
                std::vector<std::thread> threads;std::vector<Json> metrics(workers);
                constexpr size_t rounds=5000;
                for(size_t w=0;w<workers;++w)threads.emplace_back([&,w]{
                    start.arrive_and_wait();
                    for(size_t i=0;i<rounds;++i){
                        const auto result=sources[w]->getChunkBlocking(0,0,0,int((overlapping?0:w*4)+(i%4)));
                        if(result.status!=ChunkStatus::Data)std::terminate();
                    }
                    const auto& counters=detail::samplingWaitStats();
                    metrics[w]={{"blocking_total",counters.seconds[0]},{"blocking_lock",counters.seconds[1]},
                        {"payload_wait",counters.seconds[9]},{"entry_budget",counters.seconds[10]},{"shared_budget",counters.seconds[11]}};
                    finish.arrive_and_wait();
                });
                const auto began=std::chrono::steady_clock::now();start.arrive_and_wait();finish.arrive_and_wait();
                const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
                for(auto& thread:threads)thread.join();
                if(fetcher->count!=initialReads)throw std::runtime_error("warm cache benchmark unexpectedly fetched");
                Json row={{"workers",workers},{"mode",local?"worker-local":"shared"},{"overlap",overlapping},
                    {"repeat",repeat},{"requests",workers*rounds},{"seconds",elapsed},{"initial_fetches",initialReads},{"thread_metrics",metrics}};
                results.push_back(row);std::cout<<row.dump()<<std::endl;
            }
        }
        std::ofstream out(argv[2]);out<<results.dump(2)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
