// Sustained, bounded-memory coordination workloads, including oversubscription.
#include <vc_reader/array.hpp>
#include <nlohmann/json.hpp>
#include <barrier>
#include <chrono>
#include <iostream>
#include <thread>

using Clock=std::chrono::steady_clock;
int main(int argc,char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"independent";
        const size_t workers=argc>2?std::stoul(argv[2]):32;
        const double seconds=argc>3?std::stod(argv[3]):1.;
        if(!workers||workers>10000||seconds<=0||seconds>3)return 2;
        const bool integrated=mode=="whole"||mode=="selected";
        const bool shared=mode=="shared"||mode=="failures";
        if(!integrated&&!shared&&mode!="independent"&&mode!="collision"&&mode!="probe")return 2;
        vcr::SingleFlight<int> flights;
        std::vector<std::string> keys;
        for(size_t i=0;keys.size()<std::max(workers,size_t(64));++i) {
            const auto key="chunk-"+std::to_string(i);
            // Former stripes hashed the physical key alone. Registration now
            // uses the full quality/array namespace, as actual reader loads do.
            if(mode!="collision"||std::hash<std::string>{}(key)%256==0)keys.push_back("array|"+key);
        }
        std::vector<vcr::SingleFlight<int>::Claim> held;
        if(mode=="probe")for(size_t i=0;i<64;++i)held.push_back(flights.acquire(keys[i]));
        std::atomic<uint64_t> producers=0;
        vcr::ArraySpec spec;spec.identity="singleflight-contention";spec.shape={4,1,1,4096*64};
        spec.chunks={4,1,1,64};spec.itemsize=1;spec.fill={0};spec.threads=0;
        spec.compute=[&](const auto&,auto* out){producers.fetch_add(1,std::memory_order_relaxed);std::fill_n(out,256,uint8_t(42));return true;};
        auto cache=std::make_shared<vcr::MemoryCache>(64*1024);
        vcr::Array reader(spec,cache);if(mode=="selected")reader.request_channel_cache({2});
        struct alignas(64) Result {uint64_t ops=0,errors=0;std::vector<double> latency;};
        std::vector<Result> results(workers);
        std::atomic<bool> stop=false,abort=false;
        std::promise<void> gate;auto opened=gate.get_future().share();
        std::barrier ready(ptrdiff_t(workers+1)),done(ptrdiff_t(workers+1));
        constexpr size_t windows=4; // one warmup, three measured windows
        std::vector<std::thread> team;
        try {
            for(size_t w=0;w<workers;++w)team.emplace_back([&,w]{
                opened.wait();if(abort)return;
                auto& result=results[w];result.latency.reserve(4096);
                uint64_t sequence=w*17;
                for(size_t window=0;window<windows;++window){
                    ready.arrive_and_wait();result.ops=result.errors=0;result.latency.clear();
                    while(!stop.load(std::memory_order_relaxed)) {
                        const bool timed=(result.ops%128==0)&&result.latency.size()<4096;
                        const auto start=timed?Clock::now():Clock::time_point{};
                        try {
                            if(integrated) {
                                const auto idx=(sequence++*17)%4096;
                                if(mode=="selected") {double p[]{2.,0.,0.,double(idx*64)};uint8_t value=0;
                                    reader.sample_nearest(p,1,&value);if(value!=42)++result.errors;}
                                else {auto value=reader.chunk({0,0,0,int64_t(idx)});if(!value||value->front()!=42)++result.errors;}
                            } else {
                                auto claim=flights.acquire(keys[shared?sequence++%8:mode=="probe"?sequence++%64:w]);
                                if(mode=="probe") {if(claim.owner()||claim.ready())++result.errors;}
                                else {
                                    const auto value=claim.resolve([&] {
                                        producers.fetch_add(1,std::memory_order_relaxed);
                                        if(shared)std::this_thread::sleep_for(std::chrono::microseconds(50));
                                        if(mode=="failures")throw std::runtime_error("expected source failure");
                                        return 42;
                                    });
                                    if(value!=42||mode=="failures")++result.errors;
                                }
                            }
                        } catch(const std::runtime_error&) {if(mode!="failures")++result.errors;}
                        if(timed)result.latency.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count());
                        ++result.ops;
                    }
                    done.arrive_and_wait();
                }
            });
        } catch(...) {abort=true;gate.set_value();for(auto& t:team)t.join();throw;}
        gate.set_value();
        std::vector<double> rates,elapsed,latencies;uint64_t errors=0,ops=0,loads=0;
        for(size_t window=0;window<windows;++window) {
            stop=false;const auto before=producers.load();const auto start=Clock::now();ready.arrive_and_wait();
            std::this_thread::sleep_for(std::chrono::duration<double>(window?seconds:.25));
            stop=true;done.arrive_and_wait();const auto duration=std::chrono::duration<double>(Clock::now()-start).count();
            uint64_t count=0;for(auto& r:results){count+=r.ops;errors+=r.errors;
                if(window)latencies.insert(latencies.end(),r.latency.begin(),r.latency.end());}
            if(window){ops+=count;loads+=producers.load()-before;rates.push_back(count/duration);elapsed.push_back(duration);}
        }
        for(auto& thread:team)thread.join();
        std::sort(rates.begin(),rates.end());std::sort(latencies.begin(),latencies.end());
        auto percentile=[&](double p){return latencies.empty()?0.:latencies[size_t((latencies.size()-1)*p)];};
        std::cout<<nlohmann::json{{"case",mode},{"workers",workers},{"cores",std::thread::hardware_concurrency()},
            {"repetitions",3},{"window_seconds",seconds},{"actual_seconds",elapsed},
            {"ops_per_second",rates[1]},{"mean_ops_per_second",std::accumulate(rates.begin(),rates.end(),0.)/3},
            {"min_ops_per_second",rates.front()},{"max_ops_per_second",rates.back()},
            {"latency_us_p50",percentile(.5)},{"latency_us_p95",percentile(.95)},{"latency_us_max",percentile(1)},
            {"operations",ops},{"source_calls",loads},{"errors",errors}}.dump()<<'\n';
        return errors?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
