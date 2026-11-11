#include <vc_reader/array.hpp>
#include <vc_reader/native/codecs.hpp>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Exercise a domain policy without importing any domain-specific reader code.
struct MaskedPolicy {
    static constexpr bool custom=true,clamp_boundary=true;
    static constexpr std::size_t components=2;
    using real_type=float;
    std::vector<float>& output;
    std::int64_t offset(std::size_t c,std::size_t a) const {return a==0?2*std::int64_t(c):0;}
    template<class Cell> void prepare_cell(Cell& cell) const {
        cell.k=3;cell.axes[0]=3;cell.axes[1]=2;cell.axes[2]=1;
    }
    template<class Cell,class Read> float evaluate(const Cell& cell,Read read) const {
        const auto weights=vcr::sampling::cornerWeights(std::array<float,3>{cell.frac[3],cell.frac[2],cell.frac[1]});
        float sum=0.f,total=0.f;
        for(std::size_t k=0;k<8;++k)if(read(k,0)>0) {sum+=weights[k]*read(k,1);total+=weights[k];}
        return total>0?sum/total:-1.f;
    }
    void store(std::size_t p,float v)const{output[p]=v;}
    void outside(std::size_t p)const{output[p]=-1.f;}
};
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    fs::path root;
    vcr::ArraySpec spec;
    std::size_t first_channel_cached_planes = 1;
    static constexpr std::size_t plane = 512;
    Fixture(const fs::path& base, const std::string& codec) : root(base / codec)
    {
        fs::create_directories(root);
        spec.identity = root.string();
        spec.shape = {5,8,8,8};
        spec.chunks = {4,8,8,8};
        spec.itemsize = 1;
        spec.fill = {9};
        spec.threads = 2;
        spec.layers.push_back({"array",root.string(),codec,"",false});
        for (int block = 0; block < 2; ++block) {
            auto decoded = chunk(block);
            vcr::EncodeOptions options;
            options.blocksize = 128;
            const auto encoded = vcr::encode(vcr::parse_codec(codec), options,
                decoded.data(),decoded.size(),1,spec.chunks);
            if (codec == "zstd") first_channel_cached_planes = 2;
            if (codec == "blosc" && block == 0 && !(encoded[2] & BLOSC_MEMCPYED)) {
                std::size_t raw_size = 0, stored_size = 0, block_size = 0;
                blosc_cbuffer_sizes(encoded.data(),&raw_size,&stored_size,&block_size);
                // Blosc may increase a tiny requested block size. A relevant
                // channel covered by the required physical blocks is retained.
                if ((3*plane-1)/block_size <= (plane-1)/block_size)
                    first_channel_cached_planes = 2;
            }
            std::ofstream file(root / (std::to_string(block)+".0.0.0"),std::ios::binary);
            file.write(reinterpret_cast<const char*>(encoded.data()),encoded.size());
        }
    }
    static std::uint8_t value(int c, int z, int y, int x)
    {
        return std::uint8_t(c*37+z*11+y*3+x);
    }
    static vcr::Chunk chunk(int block)
    {
        vcr::Chunk data(4*plane);
        for (int c=0;c<4;++c) for (int z=0;z<8;++z) for (int y=0;y<8;++y) for (int x=0;x<8;++x)
            data[c*plane+z*64+y*8+x] = value(block*4+c,z,y,x);
        return data;
    }
    static vcr::Chunk read_channel(vcr::Array& reader, int channel)
    {
        vcr::Chunk out(plane);
        reader.read_many({{channel,0,0,0}},{{1,8,8,8}},{out.data()});
        for (int z=0;z<8;++z) for (int y=0;y<8;++y) for (int x=0;x<8;++x)
            require(out[z*64+y*8+x] == value(channel,z,y,x),"channel values");
        return out;
    }
};

void check_codec(const fs::path& path, const std::string& codec)
{
    Fixture f(path,codec);
    auto memory = std::make_shared<vcr::MemoryCache>(1<<20);
    vcr::Array reader(f.spec,memory);
    auto unrelated = f.spec;
    unrelated.identity += "-baseline";
    vcr::Array baseline(unrelated,std::make_shared<vcr::MemoryCache>(1<<20));
    reader.request_channel_cache({0,2,4});
    for(int baseChannel:{0,2}) {
        const std::vector<float> xyz{0,0,0,1.25f,2.5f,3.75f,7,7,7,7.1f,0,0,-1,0,0,NAN,0,0};
        const vcr::PointCoordinates<float> coordinates{xyz.data(),3};
        const vcr::ChannelCoordinates selected{coordinates,baseChannel};
        std::vector<float> actual(6),expected(6,-1.f);
        const MaskedPolicy policy{actual};
        reader.sample_interpolated_coordinates(selected,actual.size(),vcr::ElementType{'u',1,false},policy);
        for(size_t p=0;p<3;++p) {
            // PointCoordinates uses the array's ZYX axis order.
            const float z=xyz[3*p],y=xyz[3*p+1],x=xyz[3*p+2];
            vcr::sampling::Cell<4,float> cell;
            const float c[]{float(baseChannel),z,y,x};
            vcr::sampling::cellOf(c,4,cell,[](auto){return int64_t(0);});policy.prepare_cell(cell);
            expected[p]=policy.evaluate(cell,[&](size_t k,size_t component){
                const auto xx=std::min(7,int(std::floor(x))+int(k&1));
                const auto yy=std::min(7,int(std::floor(y))+int(k>>1&1));
                const auto zz=std::min(7,int(std::floor(z))+int(k>>2&1));
                return float(Fixture::value(baseChannel+2*component,zz,yy,xx));
            });
        }
        require(actual==expected,"custom policy uses direct and split interpolation with masked channels");
        memory->clear();
        actual.assign(6,123.f);
        std::array<uint8_t,6> missing{};
        const auto readsBefore=reader.stats().disk_hits.load();
        reader.sample_interpolated_coordinates(selected,actual.size(),vcr::ElementType{'u',1,false},policy,missing.data());
        for(size_t p=0;p<3;++p)
            require(missing[p]==1&&actual[p]==123.f,"cache-only policy miss preserves interior and split outputs");
        for(size_t p=3;p<6;++p)
            require(missing[p]==0&&actual[p]==-1.f,"policy handles invalid coordinates without source misses");
        require(reader.stats().disk_hits.load()==readsBefore,"cache-only policies perform no source reads");
        memory->clear();
    }
    Fixture::read_channel(reader,0);
    const auto one_read_bytes = memory->nbytes();
    if (one_read_bytes != f.first_channel_cached_planes*Fixture::plane)
        throw std::runtime_error(codec + ": first channel cached " + std::to_string(one_read_bytes) +
            " bytes; expected " + std::to_string(f.first_channel_cached_planes*Fixture::plane));
    require(memory->size() == f.first_channel_cached_planes,"per-channel entry count");
    Fixture::read_channel(reader,4);
    require(memory->nbytes() == one_read_bytes+Fixture::plane,"partial last channel chunk caches only valid requested channel");
    std::array<std::uint8_t,8> a{},b{};
    reader.read_many({{0,1,1,1},{2,2,2,2}},{{1,2,2,2},{1,2,2,2}},{a.data(),b.data()});
    require(a[0] == Fixture::value(0,1,1,1) && b[7] == Fixture::value(2,3,3,3),"disjoint-channel batch region");
    // Arbitrary channel fractions exercise interpolation across channel planes,
    // including padding boundaries, clipping, and the planner's grouped runs.
    std::vector<double> coordinates;
    for (int i=0;i<160;++i) {
        coordinates.insert(coordinates.end(),{double(i%5)+.25,double(i%8)+.125,
            double((i/3)%8)+.5,double((i/7)%8)+.25});
    }
    for (int i=0;i<128;++i)
        coordinates.insert(coordinates.end(),{1.25,double(i%6)+.125,
            double((i/7)%6)+.5,double((i/11)%6)+.25});
    coordinates.insert(coordinates.end(),{-1.,2.,2.,2.,5.,2.,2.,2.,0.,-1.,0.,0.});
    const auto count = coordinates.size()/4;
    vcr::Chunk actual(count),expected(count);
    reader.sample_nearest(coordinates.data(),count,actual.data());
    baseline.sample_nearest(coordinates.data(),count,expected.data());
    require(actual == expected,"nearest channel cache equals full chunk");
    std::vector<float> actual_linear(count),expected_linear(count);
    reader.sample_linear(coordinates.data(),count,{'u',1,false},false,actual_linear.data());
    baseline.sample_linear(coordinates.data(),count,{'u',1,false},false,expected_linear.data());
    require(actual_linear == expected_linear,"linear channel cache equals full chunk bit for bit");
    // Float64 integral coordinates follow nearest; integer reads share the same path.
    const std::int64_t points[]{4,7,7,7,2,3,2,1,0,0,0,0};
    std::array<std::uint8_t,3> values{};
    reader.read_points(points,3,values.data());
    require(values[0] == Fixture::value(4,7,7,7) && values[1] == Fixture::value(2,3,2,1)
        && values[2] == Fixture::value(0,0,0,0),"integer point read");
    auto whole = reader.chunk({1,0,0,0});
    require(whole && *whole == Fixture::chunk(1),"full chunk includes stored channel padding");
    auto hit = reader.cached_chunk({1,0,0,0});
    require(hit.first && hit.second && *hit.second == *whole,"full cached chunk");

    memory->clear();
    const double point[]{2,2,2,2};
    std::uint8_t result = 123,missing = 0;
    reader.sample_nearest_cached(point,1,&result,&missing);
    require(missing == 1 && result == 123,"cache-only source miss leaves output unchanged");
    Fixture::read_channel(reader,2);
    reader.sample_nearest_cached(point,1,&result,&missing);
    require(!missing && result == Fixture::value(2,2,2,2),"cache-only selected hit");
    memory->clear();
    reader.read_many({{0,0,0,0}},{{1,1,1,1}},{nullptr});
    require(reader.cached_chunk({0,0,0,0}).first,"null-output prefill retains full requested chunk");

    // Every same-identity reader sees the growing policy, independently of cache ownership.
    auto second_memory = std::make_shared<vcr::MemoryCache>(1<<20);
    vcr::Array second(f.spec,second_memory);
    require(second.cache_channels() == vcr::Index({0,2,4}),"shared policy across readers");
    second.request_channel_cache({1,2});
    require(reader.cache_channels() == vcr::Index({0,1,2,4}),"relevant channels accumulate as a set");
    Fixture::read_channel(second,1);
    require(second_memory->nbytes() < 4*Fixture::plane,"separate cache obeys same policy");
    vcr::Array same_cache(f.spec,second_memory);
    const auto reads = same_cache.stats().disk_hits.load();
    Fixture::read_channel(same_cache,1);
    require(same_cache.stats().disk_hits == reads && same_cache.stats().memory_hits > 0,
        "shared memory cache reuses selected channel");
    bool rejected = false;
    try { second.request_channel_cache({7}); } catch (const std::out_of_range&) { rejected = true; }
    require(rejected && reader.cache_channels() == vcr::Index({0,1,2,4}),"invalid request cannot mutate policy");
    second_memory->clear();
    std::vector<std::future<void>> tasks;
    for (int i=0;i<8;++i) tasks.push_back(std::async(std::launch::async,[&] {
        for (int j=0;j<20;++j) Fixture::read_channel(second,2);
    }));
    for (auto& task : tasks) task.get();
    require(second_memory->nbytes() <= 3*Fixture::plane,"concurrent reads do not duplicate cache entries");
}

int main()
{
    const auto root = fs::temp_directory_path() / ("vcr-channel-cache-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code e; fs::remove_all(path,e); } } cleanup{root};
    try {
        for (const auto& codec : {"raw","blosc","zstd"}) check_codec(root,codec);
        Fixture migration(root / "migration","raw");
        auto memory = std::make_shared<vcr::MemoryCache>(1<<20);
        vcr::Array reader(migration.spec,memory);
        Fixture::read_channel(reader,0);
        require(memory->nbytes() == 4*Fixture::plane,"default whole-chunk cache");
        reader.request_channel_cache({0,2});
        Fixture::read_channel(reader,0);
        require(memory->nbytes() == 2*Fixture::plane && memory->size() == 2,
            "enabling selected mode replaces full cache entry without duplicate storage");
        {
            auto policy = vcr::ChannelPolicy::for_array(migration.spec.identity);
            std::promise<void> locked,release;
            auto released = release.get_future().share();
            auto holder = std::async(std::launch::async,[&] {
                std::lock_guard hold((*policy->reads)[std::hash<std::string>{}("0.0.0.0")%policy->reads->size()]);
                locked.set_value();
                released.wait();
            });
            locked.get_future().wait();
            auto reads = std::async(std::launch::async,[&] {
                Fixture::read_channel(reader,0); // resident hits must bypass source-I/O lock
                const double absent[]{3,1,1,1};
                std::uint8_t output = 123,missing = 0;
                reader.sample_nearest_cached(absent,1,&output,&missing);
                require(missing == 1 && output == 123,"busy source read cannot block cache-only miss");
            });
            const auto ready = reads.wait_for(std::chrono::seconds(1));
            release.set_value(); // release even if a regression blocked the reader
            holder.get();
            reads.get();
            require(ready == std::future_status::ready,"resident/cache-only reads waited on source-I/O stripe");
#ifndef _WIN32
            std::lock_guard parent_hold(policy->mutex);
            const auto child = fork();
            require(child >= 0,"fork channel-policy regression");
            if (child == 0) {
                alarm(2); // a regression must fail, not hang the test suite
                auto fresh = vcr::ChannelPolicy::for_array(migration.spec.identity);
                if (fresh.get() == policy.get() || fresh->enabled || !fresh->mutex.try_lock()) _exit(1);
                fresh->mutex.unlock();
                _exit(0);
            }
            int status = 0;
            while (waitpid(child,&status,0) < 0) {
                if (errno != EINTR) throw std::runtime_error("waitpid channel-policy regression");
            }
            require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "fork inherited locked channel policy instead of a fresh process registry");
#endif
        }
        Fixture corrupt(root / "corrupt","raw");
        fs::resize_file(corrupt.root / "0.0.0.0",1);
        vcr::Array broken(corrupt.spec,std::make_shared<vcr::MemoryCache>(1<<20));
        broken.request_channel_cache({0,2});
        std::uint8_t out = 0;
        broken.read_many({{0,0,0,0}},{{1,1,1,1}},{&out});
        require(out == 9 && broken.stats().corrupt == 1,"corrupt source reads as fill and is counted");
        std::uint8_t miss = 1;
        const double point[]{0,0,0,0};
        broken.sample_nearest_cached(point,1,&out,&miss);
        require(!miss && out == 9,"cached absent selected channel is fill, not cache miss");
        std::cout << "Channel cache codec parity, sharing, superset, batching, edges, sampling, prefill, corruption and concurrency passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
