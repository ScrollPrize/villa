// Bounded-memory scratch-lease contention, not a decompression benchmark.
#include <vc_reader/native/scratch_pool.hpp>
#include <nlohmann/json.hpp>
#include <barrier>
#include <future>
#include <iostream>
#include <numeric>
int main(int argc,char** argv){try{
    const size_t users=argc>1?std::stoul(argv[1]):32;
    const double seconds=argc>2?std::stod(argv[2]):.5;
    if(!users||users>10000||seconds<=0||seconds>3)return 2;
    using Clock=std::chrono::steady_clock;
    struct alignas(64) Result {uint64_t ops=0,errors=0;};
    std::vector<Result> result(users);std::vector<std::thread> workers;
    std::promise<void> gate;auto open=gate.get_future().share();std::atomic<bool> abort=false,stop=false;
    std::barrier ready(ptrdiff_t(users+1)),done(ptrdiff_t(users+1));
    try{for(size_t user=0;user<users;++user)workers.emplace_back([&,user]{
        open.wait();if(abort)return;
        for(size_t window=0;window<4;++window){ready.arrive_and_wait();auto& r=result[user];r.ops=r.errors=0;
            while(!stop.load(std::memory_order_relaxed)){
                auto lease=vcr::workerScratch().lease();
                for(size_t i=0;i<4;++i){auto b=lease.bytes(i,i==3?16:4096*(i+1));
                    b.front()=uint8_t(r.ops);b.back()=uint8_t(user);if(b.front()!=uint8_t(r.ops)||b.back()!=uint8_t(user))++r.errors;}
                ++r.ops;
            }
            done.arrive_and_wait();
        }
    });}catch(...){abort=true;gate.set_value();for(auto& t:workers)t.join();throw;}
    gate.set_value();std::vector<double> rates;uint64_t ops=0,errors=0,growthStart=0;
    auto stats=vcr::scratchMaintenance().stats;
    for(size_t window=0;window<4;++window){stop=false;const auto start=Clock::now();ready.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::duration<double>(window?seconds:.25));stop=true;done.arrive_and_wait();
        const auto elapsed=std::chrono::duration<double>(Clock::now()-start).count();uint64_t count=0;
        for(auto r:result){count+=r.ops;errors+=r.errors;}
        if(window){ops+=count;rates.push_back(count/elapsed);}else growthStart=stats->allocations.load();
    }
    for(auto& t:workers)t.join();std::sort(rates.begin(),rates.end());
    std::cout<<nlohmann::json{{"users",users},{"repetitions",3},{"seconds",seconds},{"operations",ops},
        {"leases_per_second",rates[1]},{"mean_leases_per_second",std::accumulate(rates.begin(),rates.end(),0.)/3},
        {"min_leases_per_second",rates.front()},{"max_leases_per_second",rates.back()},
        {"measured_growth_allocations",stats->allocations.load()-growthStart},{"peak_bytes",stats->peak.load()},
        {"errors",errors}}.dump()<<'\n';return errors?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
