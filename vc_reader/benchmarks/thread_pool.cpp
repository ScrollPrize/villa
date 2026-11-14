// Scheduler throughput with concurrent callers; no volume I/O or cache work.
#include "../src/cpp/thread_pool.hpp"
#include <nlohmann/json.hpp>
#include <barrier>
#include <future>
#include <iostream>
#include <numeric>
#include <string>

int main(int argc,char** argv) { try {
    const std::string mode=argc>1?argv[1]:"tiny";
    const size_t users=argc>2?std::stoul(argv[2]):32;
    const size_t helpers=argc>3?std::stoul(argv[3]):16;
    const double duration=argc>4?std::stod(argv[4]):.3;
    if(!users||users>1920||helpers>192||duration<=0||duration>3||
       (mode!="tiny"&&mode!="bulk"&&mode!="mixed"))return 2;
    using Clock=std::chrono::steady_clock;
    vcr::ThreadPool pool(helpers);
    struct alignas(64) Result {uint64_t batches=0,errors=0;};
    std::vector<Result> results(users);std::vector<std::thread> callers;
    std::atomic<bool> stop=false,abort=false;std::promise<void> gate;
    auto open=gate.get_future().share();std::barrier ready(ptrdiff_t(users+1)),done(ptrdiff_t(users+1));
    const size_t count=argc>5?std::stoul(argv[5]):mode=="bulk"?256:mode=="mixed"?64:16;
    if(!count||count>1048576)return 2;
    try { for(size_t user=0;user<users;++user)callers.emplace_back([&,user]{
        open.wait();if(abort)return;std::vector<uint64_t> output(count);
        for(size_t window=0;window<4;++window){
            ready.arrive_and_wait();auto& result=results[user];result={};
            while(!stop.load(std::memory_order_relaxed)){
                pool.parallel_for(count,[&](size_t i){
                    if(mode=="mixed"&&i%16==0)std::this_thread::sleep_for(std::chrono::microseconds(20));
                    output[i]=uint64_t(i)*17+user;
                });
                if(output.front()!=user||output.back()!=(count-1)*17+user)++result.errors;
                ++result.batches;
            }
            done.arrive_and_wait();
        }
    }); } catch(...) {abort=true;gate.set_value();for(auto& t:callers)t.join();throw;}
    gate.set_value();std::vector<double> rates;uint64_t errors=0;
    for(size_t window=0;window<4;++window){
        stop=false;ready.arrive_and_wait();auto start=Clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(window?duration:.15));
        stop=true;done.arrive_and_wait();
        const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();uint64_t batches=0;
        for(auto r:results){batches+=r.batches;errors+=r.errors;}
        if(window)rates.push_back(batches/elapsed);
    }
    for(auto& t:callers)t.join();std::sort(rates.begin(),rates.end());
    std::cout<<nlohmann::json{{"mode",mode},{"callers",users},{"helpers",helpers},{"items_per_batch",count},
        {"repetitions",3},{"seconds_per_window",duration},{"errors",errors},{"queue_lock_free",pool.queue_lock_free()},
        {"batches_per_second",rates[1]},{"items_per_second",rates[1]*count},
        {"mean_batches_per_second",std::accumulate(rates.begin(),rates.end(),0.)/3},
        {"min_batches_per_second",rates.front()},{"max_batches_per_second",rates.back()}}.dump()<<'\n';
    return errors?1:0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
