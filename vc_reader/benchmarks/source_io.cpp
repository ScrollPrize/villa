// Matched source-filesystem experiment. Stage a bounded subset once, then run
// identical metadata or selected-channel work against either root. No input edits.
#include <vc_reader/native/channel_decode.hpp>
#include <vc_reader/native/chunk_io.hpp>
#include <vc_reader/native/thread_pool.hpp>
#include <barrier>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/resource.h>

namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr size_t plane=64*64*64,total=7*plane;

std::vector<fs::path> children(const fs::path& root) {
    std::vector<fs::path> out;
    for(const auto& e:fs::directory_iterator(root))out.push_back(e.path());
    std::sort(out.begin(),out.end());return out;
}
std::vector<std::string> paths(const fs::path& manifest) {
    std::ifstream input(manifest);if(!input)throw std::runtime_error("missing subset manifest");
    std::vector<std::string> out;std::string line;
    while(std::getline(input,line))if(!line.empty())out.push_back(line);
    if(out.empty())throw std::runtime_error("empty subset");return out;
}
double cpuTime(bool system=false) {
#if defined(__linux__)
    rusage use{};if(getrusage(RUSAGE_THREAD,&use))throw std::runtime_error("getrusage failed");
    const auto t=system?use.ru_stime:use.ru_utime;
    return double(t.tv_sec)+double(t.tv_usec)*1e-6;
#else
    (void)system;return 0;
#endif
}
int main(int argc,char** argv) {try {
    if(argc==4&&std::string(argv[1])=="stage") {
        const fs::path source=argv[2],cache=argv[3];
        if(!fs::create_directory(cache))throw std::runtime_error("cache must be a new directory");
        struct Cleanup{fs::path path;bool keep=false;~Cleanup(){if(!keep){std::error_code ec;fs::remove_all(path,ec);}}} cleanup{cache};
        const auto begin=Clock::now();std::vector<std::string> files;
        for(int z:{10,50,100,150,200,250}) {
            const auto start=files.size();const auto dir=source/"0"/std::to_string(z);
            if(!fs::exists(dir))continue;
            for(const auto& y:children(dir)) {
                if(!fs::is_directory(y))continue;
                for(const auto& x:children(y)) {
                    // Exclude almost-empty compressed chunks; this is only a
                    // benchmark fixture selection, never a reader/filter policy.
                    if(fs::is_regular_file(x)&&fs::file_size(x)>=32768)
                        files.push_back(x.lexically_relative(source).generic_string());
                    if(files.size()-start>=128)break;
                }
                if(files.size()-start>=128)break;
            }
        }
        if(files.empty())throw std::runtime_error("no suitable chunk files");
        const auto listed=Clock::now();std::atomic<uint64_t> bytes=0;
        // Make directories once, before parallel copy.
        for(const auto& p:files)fs::create_directories((cache/p).parent_path());
        vcr::ThreadPool workers(31);
        workers.parallel_for(files.size(),[&](size_t i) {
            const auto from=source/files[i],to=cache/files[i];fs::copy_file(from,to);
            const auto size=fs::file_size(from);if(fs::file_size(to)!=size)throw std::runtime_error("copy size mismatch");
            bytes.fetch_add(size,std::memory_order_relaxed);
        });
        std::ofstream manifest(cache/"subset.txt");for(const auto& p:files)manifest<<p<<'\n';
        manifest.close();if(!manifest)throw std::runtime_error("subset manifest write failed");
        cleanup.keep=true;
        std::cout<<nlohmann::json{{"stage_files",files.size()},{"stage_bytes",bytes.load()},
            {"selection_seconds",std::chrono::duration<double>(listed-begin).count()},
            {"copy_seconds",std::chrono::duration<double>(Clock::now()-listed).count()}}.dump()<<'\n';return 0;
    }
    if(argc!=5)throw std::runtime_error("usage: bench_source_io stage SOURCE_ROOT CACHE_ROOT | ROOT SUBSET_MANIFEST WORKERS metadata|decode");
    const auto relative=paths(argv[2]);const size_t workers=std::stoul(argv[3]);const std::string mode=argv[4];
    if(!workers||workers>1920||(mode!="metadata"&&mode!="decode"))throw std::runtime_error("invalid workers/mode");
    std::vector<std::string> files;for(const auto& p:relative)files.push_back((fs::path(argv[1])/p).string());
    nlohmann::json rows=nlohmann::json::array();std::vector<double> rates;
    for(int rep=-1;rep<3;++rep) {
        const double seconds=rep<0?.2:.7;
        std::barrier start(static_cast<std::ptrdiff_t>(workers));
        struct Result {uint64_t calls=0,bytes=0,check=0;double user=0,system=0,wall=0;std::exception_ptr error;};
        std::vector<Result> results(workers);std::vector<std::thread> threads;
        for(size_t t=0;t<workers;++t)threads.emplace_back([&,t]{
            start.arrive_and_wait();const auto begin=Clock::now();const auto u=cpuTime(),s=cpuTime(true);auto& r=results[t];
            try {
                const std::vector<size_t> channels{4,5,6};
                for(size_t i=t;Clock::now()-begin<std::chrono::duration<double>(seconds);i+=workers) {
                    const auto& path=files[i%files.size()];auto file=vcr::File::open_read(path);
                    if(!file)throw std::runtime_error("missing staged source file");
                    vcr::platform::FileStat st;
                    if(vcr::platform::fstat(file.fd(),st))throw std::runtime_error("stat failed");
                    if(mode=="decode") {
                        auto decoded=vcr::decode_channels(vcr::ChunkCodec{vcr::Codec::Blosc},total,plane,channels,channels,size_t(st.size),
                            [&](auto off,auto n,auto* dst){vcr::pread_exact(file.fd(),dst,n,int64_t(off),path);});
                        r.bytes+=decoded.bytesRead;
                        for(const auto& c:decoded.channels)r.check+=(*c.bytes)[0]+(*c.bytes)[plane-1];
                    }
                    ++r.calls;
                }
            }catch(...){r.error=std::current_exception();}
            r.wall=std::chrono::duration<double>(Clock::now()-begin).count();r.user=cpuTime()-u;r.system=cpuTime(true)-s;
        });
        for(auto& thread:threads)thread.join();
        double wall=0,user=0,system=0;uint64_t calls=0,bytes=0,check=0;
        for(const auto& r:results){if(r.error)std::rethrow_exception(r.error);wall=std::max(wall,r.wall);
            user+=r.user;system+=r.system;calls+=r.calls;bytes+=r.bytes;check+=r.check;}
        if(rep<0)continue;rates.push_back(calls/wall);
        rows.push_back({{"seconds",wall},{"calls",calls},{"calls_per_second",calls/wall},
            {"read_bytes",bytes},{"user_core_seconds",user},{"system_core_seconds",system},{"check",check}});
    }
    std::sort(rates.begin(),rates.end());
    std::cout<<nlohmann::json{{"root",argv[1]},{"workers",workers},{"mode",mode},{"files",files.size()},
        {"runs",rows},{"mean_per_second",std::accumulate(rates.begin(),rates.end(),0.)/rates.size()},
        {"median_per_second",rates[1]},{"min_per_second",rates.front()},{"max_per_second",rates.back()}}.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
