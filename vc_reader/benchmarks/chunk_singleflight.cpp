// General read/decode coordination workloads. Fixtures are disposable; no user
// source is modified. Warm filesystem-cache results are not device bandwidth.
#include <vc_reader/array.hpp>
#include <nlohmann/json.hpp>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
using Json=nlohmann::json;
int main(int argc,char** argv) {
    const size_t workers=argc>1?std::stoul(argv[1]):32;
    const size_t waves=argc>2?std::stoul(argv[2]):8;
    if(!workers||workers>256||!waves)return 2;
    const auto root=fs::temp_directory_path()/("vcr-singleflight-"+std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    struct Cleanup {fs::path path;~Cleanup(){std::error_code e;fs::remove_all(path,e);}} cleanup{root};
    try {
        constexpr size_t plane=64*64*64,bytes=4*plane;
        std::vector<uint8_t> reference(bytes);uint32_t random=42;
        for(size_t i=0;i<bytes;++i){random^=random<<13;random^=random>>17;random^=random<<5;
            reference[i]=uint8_t((i/64+i/plane*31+(random&15))%256);}
        std::vector<size_t> collisions;
        for(size_t i=0;collisions.size()<workers;++i)
            if(std::hash<std::string>{}("0.0.0."+std::to_string(i))%256==0)collisions.push_back(i);
        const auto maximum=std::max(workers,collisions.back()+1);
        vcr::EncodeOptions encoding;encoding.cname="zstd";encoding.level=3;
        const auto encoded=vcr::encode(vcr::Codec::Blosc,encoding,reference.data(),reference.size(),1,{4,64,64,64});
        for(const auto& codec:{std::string("blosc"),std::string("raw")}) {
            fs::create_directories(root/codec);
            std::set<size_t> keys(collisions.begin(),collisions.end());
            for(size_t i=0;i<workers;++i)keys.insert(i);
            const auto& data=codec=="blosc"?encoded:reference;
            for(auto i:keys){std::ofstream f(root/codec/("0.0.0."+std::to_string(i)),std::ios::binary);
                f.write(reinterpret_cast<const char*>(data.data()),data.size());}
        }
        const auto deadline=Clock::now()+std::chrono::seconds(25);
        for(const std::string name:{"file_same_full","file_same_selected","file_distinct_full","file_distinct_selected",
            "file_collision_full","file_collision_selected","compute_same_full","compute_same_selected",
            "hot_full","hot_selected","nocache_same_full","nocache_same_selected","raw_range_distinct"}) {
            if(Clock::now()>=deadline)throw std::runtime_error("benchmark exceeded 25-second budget");
            const bool selected=name.find("selected")!=std::string::npos;
            const bool computed=name.find("compute")!=std::string::npos;
            const bool hot=name.find("hot")!=std::string::npos;
            const bool raw=name=="raw_range_distinct";
            const bool noCache=name.find("nocache")!=std::string::npos||computed||raw;
            const bool distinct=name.find("distinct")!=std::string::npos||name.find("collision")!=std::string::npos;
            const bool collision=name.find("collision")!=std::string::npos;
            vcr::ArraySpec spec;spec.identity=(root/name).string();spec.shape={4,64,64,int64_t(maximum*64)};
            spec.chunks={4,64,64,64};spec.itemsize=1;spec.fill={0};spec.threads=0;
            std::atomic<size_t> computes=0;
            if(computed)spec.compute=[&](const auto&,auto* out){++computes;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));std::memcpy(out,reference.data(),bytes);return true;};
            else spec.layers.push_back({"array",(root/(raw?"raw":"blosc")).string(),raw?"raw":"blosc","",false});
            auto cache=noCache?std::shared_ptr<vcr::MemoryCache>{}:std::make_shared<vcr::MemoryCache>(512*1024*1024);
            vcr::Array reader(spec,cache);if(selected)reader.request_channel_cache({2});
            const auto read=[&](size_t worker){
                const auto key=distinct?(collision?collisions[worker]:worker):0;
                if(selected||raw){const double point[]{2.,0.,0.,double(key*64)};uint8_t value=0;
                    reader.sample_nearest(point,1,&value);
                    if(value!=reference[2*plane])throw std::runtime_error("selected/range value changed");
                } else {const auto value=reader.chunk({0,0,0,int64_t(key)});
                    if(!value||value->size()!=bytes||(*value)[0]!=reference[0]||value->back()!=reference.back())
                        throw std::runtime_error("whole chunk value changed");}
            };
            if(hot)read(0);
            const auto before=reader.stats().disk_hits.load();std::vector<double> times;
            std::vector<std::thread> team;std::vector<std::exception_ptr> errors(workers);
            std::barrier ready(ptrdiff_t(workers+1)),done(ptrdiff_t(workers+1));
            constexpr size_t repetitions=3;
            for(size_t worker=0;worker<workers;++worker)team.emplace_back([&,worker]{
                for(size_t i=0;i<repetitions*waves;++i){ready.arrive_and_wait();
                    try{read(worker);}catch(...){errors[worker]=std::current_exception();}done.arrive_and_wait();}
            });
            for(size_t rep=0;rep<repetitions;++rep) {
                double elapsed=0.;
                for(size_t wave=0;wave<waves;++wave){if(cache&&!hot)cache->clear();
                    const auto start=Clock::now();ready.arrive_and_wait();done.arrive_and_wait();
                    elapsed+=std::chrono::duration<double>(Clock::now()-start).count();}
                times.push_back(elapsed);
            }
            for(auto& thread:team)thread.join();for(auto error:errors)if(error)std::rethrow_exception(error);
            const auto mean=std::accumulate(times.begin(),times.end(),0.)/times.size();std::sort(times.begin(),times.end());
            std::cout<<Json{{"case",name},{"workers",workers},{"waves",waves},{"repetitions",repetitions},
                {"mean_seconds",mean},{"p50_seconds",times[1]},{"p95_seconds",times.back()},
                {"min_seconds",times.front()},{"max_seconds",times.back()},
                {"reads_per_second",workers*waves/times[1]},
                {"disk_loads",reader.stats().disk_hits.load()-before},{"compute_calls",computes.load()},
                {"chunk_bytes",bytes},{"encoded_bytes",encoded.size()}}.dump()<<'\n';
        }
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
