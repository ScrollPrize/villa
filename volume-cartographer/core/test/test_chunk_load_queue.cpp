#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/Volume.hpp"
#include "utils/zarr.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <filesystem>
#include <random>
using namespace vc::render;
using namespace std::chrono_literals;
namespace {
class Fetcher final:public IChunkFetcher {
public:
    std::atomic<int> calls{0},active{0},peak{0},value{31};
    std::shared_future<void> gate;
    ChunkFetchResult fetch(const ChunkKey&) override {
        ++calls;const int now=++active;int old=peak;
        while(now>old&&!peak.compare_exchange_weak(old,now)){}
        const int v=value;
        if(gate.valid())gate.wait();
        std::this_thread::sleep_for(1ms);--active;
        if(v==-2)throw std::runtime_error("synthetic queued fetch error");
        ChunkFetchResult r;r.status=v<0?ChunkFetchStatus::Missing:ChunkFetchStatus::Found;
        if(v>=0)r.bytes.assign(64,std::byte(v));return r;
    }
};
auto service(size_t window=4,size_t bytes=1<<20){
    auto o=ChunkCacheService::batchOptions(4,bytes);o.loaderBackend=ChunkCacheService::Options::LoaderBackend::Queued;o.queuedChunkWindow=window;
    return std::make_shared<ChunkCacheService>(o);
}
auto source(const std::shared_ptr<ChunkCacheService>& s,const std::shared_ptr<Fetcher>& f,const std::string& name="data"){
    ChunkCache::LevelInfo info;info.shape={4,4,128};info.chunkShape={4,4,4};
    return s->acquireSource(name,{info},{f},0,ChunkDtype::UInt8,{});
}
template<class F>bool until(F predicate){auto end=std::chrono::steady_clock::now()+3s;while(std::chrono::steady_clock::now()<end){if(predicate())return true;std::this_thread::sleep_for(1ms);}return false;}
}

TEST_CASE("queued batches coalesce and bound concurrent unique keys") {
    auto s=service(4);auto f=std::make_shared<Fetcher>();std::promise<void> gate;f->gate=gate.get_future().share();auto a=source(s,f);
    std::vector<ChunkKey> keys;for(int x=0;x<8;++x)keys.push_back({0,0,0,x});
    std::vector<std::future<std::vector<ChunkResult>>> reads;
    for(int i=0;i<10;++i)reads.push_back(std::async(std::launch::async,[&]{return a->getChunksBlocking(keys);}));
    const bool combined=until([&]{return s->loaderStats().coalesced>=72;});
    gate.set_value();CHECK(combined);
    for(auto& read:reads){auto results=read.get();REQUIRE(results.size()==8);for(auto& r:results)CHECK(r.bytes->at(0)==std::byte(31));}
    CHECK(f->calls==8);CHECK(f->peak<=4);CHECK(f->peak>1);
    CHECK(s->loaderStats().peakInFlight==4);CHECK(s->loaderStats().completed==8);
    CHECK(until([&]{return a->stats().blockingReaders==0;}));
    CHECK_THROWS(s->createDecodedPartition(1024));
}

TEST_CASE("queued reservations survive tiny cache and invalidation") {
    auto s=service(3,1);auto f=std::make_shared<Fetcher>();auto a=source(s,f);
    std::vector<ChunkKey> keys;for(int x=0;x<32;++x)keys.push_back({0,0,0,x});
    auto results=a->getChunksBlocking(keys);CHECK(f->calls==32);
    for(auto& r:results){REQUIRE(r.bytes);CHECK(r.bytes->at(0)==std::byte(31));}
    CHECK(until([&]{return a->stats().blockingReaders==0;}));
    std::promise<void> gate;f->gate=gate.get_future().share();a->invalidate();
    const int before=f->calls;
    auto read=std::async(std::launch::async,[&]{return a->getChunkBlocking(0,0,0,0);});
    const bool started=until([&]{return f->calls>before;});
    f->value=42;a->invalidate();gate.set_value();CHECK(started);
    auto result=read.get();REQUIRE(result.bytes);CHECK(result.bytes->at(0)==std::byte(42));
    CHECK(f->calls>=before+2);CHECK(until([&]{return a->stats().blockingReaders==0;}));
}

TEST_CASE("queued resident reads bypass admission and mixed batches preserve order") {
    auto s=service();auto f=std::make_shared<Fetcher>();auto a=source(s,f);
    REQUIRE(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Data);
    REQUIRE(until([&]{return a->stats().blockingReaders==0;}));
    const auto requests=s->loaderStats().requests;
    a->prefetchChunks({{0,0,0,0}},true);
    auto hits=a->getChunksBlocking({{0,0,0,0},{0,0,0,0}});
    REQUIRE(hits.size()==2);CHECK(hits[0].bytes==hits[1].bytes);
    CHECK(s->loaderStats().requests==requests);CHECK(f->calls==1);
    f->value=42;
    auto mixed=a->getChunksBlocking({{0,0,0,0},{0,0,0,1},{0,0,0,0},{0,0,0,1}});
    REQUIRE(mixed.size()==4);
    CHECK(mixed[0].bytes->at(0)==std::byte(31));CHECK(mixed[2].bytes==mixed[0].bytes);
    CHECK(mixed[1].bytes->at(0)==std::byte(42));CHECK(mixed[3].bytes==mixed[1].bytes);
    CHECK(s->loaderStats().requests==requests+1);CHECK(f->calls==2);
    REQUIRE(until([&]{return a->stats().blockingReaders==0;}));
    a->invalidate();f->value=-2;
    REQUIRE(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Error);
    const auto afterError=s->loaderStats().requests;
    CHECK(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Error);
    CHECK(s->loaderStats().requests==afterError);
}

TEST_CASE("queued hints use ordinary caches and preserve missing fill errors") {
    auto s=service();auto af=std::make_shared<Fetcher>(),bf=std::make_shared<Fetcher>();auto a=source(s,af,"a"),b=source(s,bf,"b");
    ChunkRequestContext request;request.channelHints=std::make_shared<ChunkChannelHints>(std::vector<std::shared_ptr<IChunkedArray>>{b});
    REQUIRE(a->getChunkBlocking(0,0,0,0,request).status==ChunkStatus::Data);
    REQUIRE(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::Data);CHECK(bf->calls==1);
    for(auto [value,status]:std::vector<std::pair<int,ChunkStatus>>{{-1,ChunkStatus::Missing},{0,ChunkStatus::AllFill},{-2,ChunkStatus::Error}}){
        bf->value=value;b->invalidate();CHECK(b->getChunkBlocking(0,0,0,0).status==status);
    }
    CHECK(b->getChunkBlocking(0,0,0,-1).status==ChunkStatus::AllFill);
    CHECK(a->getChunksBlocking({}).empty());
}

TEST_CASE("queue stop resolves pending and active requests without blocking IO") {
    ChunkLoadQueue queue(1);std::atomic<int> acquired=0,released=0,unsubscribed=0;
    ChunkLoadQueue::Source source;source.id=1;
    source.acquire=[&](const auto& keys){acquired+=keys.size();};
    source.poll=[&](const auto& queries){return std::vector<ChunkResult>(queries.size());};
    source.release=[&](const auto& keys){released+=keys.size();};
    source.subscribe=[&](auto){return [&]{++unsubscribed;};};
    auto future=queue.submit(source,{{0,0,0,0},{0,0,0,1}},false,0);
    REQUIRE(until([&]{return acquired==1;}));queue.stop();
    CHECK_THROWS(future.get());CHECK(released==1);CHECK(unsubscribed==1);
    CHECK_THROWS(queue.submit(source,{{0,0,0,0}},false,0));
}

TEST_CASE("queued completion cannot be hidden by a throwing user listener") {
    auto s=service();auto f=std::make_shared<Fetcher>();auto a=source(s,f);
    auto id=a->addChunkReadyListener([]{throw std::runtime_error("bad user listener");});
    auto read=std::async(std::launch::async,[&]{return a->getChunkBlocking(0,0,0,0);});
    const auto status=read.wait_for(3s);
    a->removeChunkReadyListener(id);
    if(status!=std::future_status::ready)a->invalidate();
    CHECK(status==std::future_status::ready);CHECK(read.get().status==ChunkStatus::Data);
}

TEST_CASE("queue cancellation releases only that source and allows subsequent work") {
    ChunkLoadQueue queue(4);std::atomic<int> acquired=0,released=0;
    ChunkLoadQueue::Source source;source.id=1;
    source.acquire=[&](const auto& keys){acquired+=keys.size();};
    source.poll=[](const auto& queries){return std::vector<ChunkResult>(queries.size());};
    source.release=[&](const auto& keys){released+=keys.size();};
    source.subscribe=[](auto){return []{};};
    auto other=source;other.id=2;
    auto one=queue.submit(source,{{0,0,0,0}},false,0);
    auto two=queue.submit(other,{{0,0,0,0}},false,0);
    REQUIRE(until([&]{return acquired==2;}));queue.cancelSource(1);
    CHECK_THROWS(one.get());CHECK(released==1);CHECK(two.wait_for(1ms)==std::future_status::timeout);
    queue.stop();CHECK_THROWS(two.get());CHECK(released==2);
}

TEST_CASE("failed acquisition is not released and cleanup exceptions do not strand promises") {
    ChunkLoadQueue queue(4);std::atomic<int> releases=0,unsubscribed=0;
    ChunkLoadQueue::Source source;source.id=1;
    source.acquire=[](const auto&){throw std::runtime_error("acquisition failed atomically");};
    source.poll=[](const auto& q){return std::vector<ChunkResult>(q.size());};
    source.release=[&](const auto&){++releases;throw std::runtime_error("release failure");};
    source.subscribe=[&](auto){return [&]{++unsubscribed;throw std::runtime_error("unsubscribe failure");};};
    auto result=queue.submit(source,{{0,0,0,0}},false,0);
    CHECK_THROWS(result.get());queue.stop();CHECK(releases==0);CHECK(unsubscribed==1);
}

TEST_CASE("queued packed ROI reads preserve offsets and fill padding") {
    auto options=ChunkCacheService::batchOptions(2,1<<20);options.loaderBackend=ChunkCacheService::Options::LoaderBackend::Queued;
    configureProcessChunkCacheService(options);
    namespace fs=std::filesystem;
    auto path=fs::temp_directory_path()/("queued_roi_"+std::to_string(std::mt19937_64(std::random_device{}())()));
    struct Cleanup {fs::path path;~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{path};
    utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;meta.shape={2,4,4,8};meta.chunks={2,4,4,4};
    meta.dtype=utils::ZarrDtype::uint8;meta.compressor_id.clear();meta.fill_value=7;
    auto array=std::make_shared<utils::ZarrArray>(utils::ZarrArray::create(path,meta));
    std::vector<std::byte> data(128,std::byte(20));std::fill(data.begin()+64,data.end(),std::byte(90));
    array->write_chunk(std::vector<size_t>{0,0,0,0},data);
    auto a=acquireProcessZarrChannel(path.string(),array,0),b=acquireProcessZarrChannel(path.string(),array,1);
    ChunkRequestContext request;request.channelHints=std::make_shared<ChunkChannelHints>(std::vector<std::shared_ptr<IChunkedArray>>{b});
    Array3D<uint8_t> out({6,6,10});Volume::readZYX(out,{-1,-1,-1},*a,0,request);
    // Preserve the existing ROI reader's exterior-zero policy; an absent chunk
    // INSIDE the logical volume uses its declared fill value instead.
    CHECK(out(0,0,0)==0);CHECK(out(1,1,1)==20);CHECK(out(4,4,4)==20);
    CHECK(out(1,1,5)==7);CHECK(out(1,1,8)==7);CHECK(out(5,5,9)==0);
    auto results=b->getChunksBlocking({{0,0,0,0},{0,0,0,0}});
    REQUIRE(results.size()==2);CHECK(results[0].byteOffset==64);CHECK(results[0].bytes->at(64)==std::byte(90));
    CHECK(results[0].bytes==results[1].bytes);
    auto cache=acquireProcessChunkCache(path.string(),array);CHECK(cache->stats().decodeAttempts==1);
}
