#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/DecodedChunkCacheBudget.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "utils/zarr.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <random>
#include <thread>

using namespace vc::render;
namespace {
class Fetcher final:public IChunkFetcher {
public:
    std::atomic<int> calls{0},value{19};size_t bytes=64;
    ChunkFetchResult fetch(const ChunkKey&) override {
        ++calls;ChunkFetchResult r;
        if(value==-2)throw std::runtime_error("hinted companion error");
        if(value<0){r.status=ChunkFetchStatus::Missing;return r;}
        r.status=ChunkFetchStatus::Found;r.bytes.assign(bytes,std::byte(value.load()));return r;
    }
};
auto service(){return std::make_shared<ChunkCacheService>(ChunkCacheService::batchOptions(2,1<<20));}
auto source(const std::shared_ptr<ChunkCacheService>& s,const std::string& name,
    const std::shared_ptr<Fetcher>& f,int chunk) {
    ChunkCache::LevelInfo info;info.shape={8,8,8};info.chunkShape={chunk,chunk,chunk};
    f->bytes=size_t(chunk)*chunk*chunk;
    return s->acquireSource(name,{info},{f},0,ChunkDtype::UInt8,{});
}
bool cached(const std::shared_ptr<IChunkedArray>& cache,const std::vector<ChunkKey>& keys){
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(std::chrono::steady_clock::now()<deadline){
        bool done=true;for(const auto& k:keys)done &= cache->getChunkIfCached(k.level,k.iz,k.iy,k.ix).status!=ChunkStatus::MissQueued;
        if(done)return true;std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}
}

TEST_CASE("read hint fills ordinary companion caches with differently chunked channels") {
    auto s=service();auto af=std::make_shared<Fetcher>(),bf=std::make_shared<Fetcher>();
    auto a=source(s,"a",af,4),b=source(s,"b",bf,2);
    ChunkRequestContext request;request.foreground=true;
    request.channelHints=std::make_shared<ChunkChannelHints>(std::vector<std::shared_ptr<IChunkedArray>>{b});
    REQUIRE(a->getChunkBlocking(0,0,0,0,request).status==ChunkStatus::Data);
    std::vector<ChunkKey> keys;for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x)keys.push_back({0,z,y,x});
    REQUIRE(cached(b,keys));CHECK(af->calls==1);CHECK(bf->calls==8);
    for(const auto& k:keys)CHECK(b->getChunkBlocking(k.level,k.iz,k.iy,k.ix).status==ChunkStatus::Data);
    CHECK(bf->calls==8);CHECK(b->getChunkIfCached(0,2,0,0).status==ChunkStatus::MissQueued);
}

TEST_CASE("channel hints reject incompatible grids and do not recursively fan out") {
    auto s=service();auto f=std::make_shared<Fetcher>();auto a=source(s,"a",f,4);
    CHECK_THROWS(ChunkChannelHints({}));CHECK_THROWS(ChunkChannelHints({a,nullptr}));
    ChunkCache::LevelInfo info;info.shape={4,4,4};info.chunkShape={4,4,4};
    auto b=s->acquireSource("bad",{info},{f},0,ChunkDtype::UInt8,{});
    CHECK_THROWS(ChunkChannelHints({a,b}));
    ChunkRequestContext request;request.channelHints=std::make_shared<ChunkChannelHints>(std::vector<std::shared_ptr<IChunkedArray>>{a,a});
    a->prefetchChunks({{0,0,0,0}},true,0,request);CHECK(f->calls==1);
}

TEST_CASE("hinted missing fill and errors remain visible to ordinary reads") {
    auto s=service();auto af=std::make_shared<Fetcher>(),bf=std::make_shared<Fetcher>();
    auto a=source(s,"a",af,4),b=source(s,"b",bf,4);
    ChunkRequestContext request;
    request.channelHints=std::make_shared<ChunkChannelHints>(std::vector<std::shared_ptr<IChunkedArray>>{b});
    bf->value=-1;a->getChunkBlocking(0,0,0,0,request);REQUIRE(cached(b,{{0,0,0,0}}));
    CHECK(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::Missing);CHECK(bf->calls==1);
    bf->value=0;b->invalidate();a->getChunkBlocking(0,0,0,0,request);REQUIRE(cached(b,{{0,0,0,0}}));
    CHECK(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::AllFill);CHECK(bf->calls==2);
    bf->value=-2;b->invalidate();a->getChunkBlocking(0,0,0,0,request);REQUIRE(cached(b,{{0,0,0,0}}));
    CHECK(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::Error);CHECK(bf->calls==3);
}

TEST_CASE("partition teardown does not drain unrelated parent work") {
    class BlockingFetcher final:public IChunkFetcher {
    public:
        std::promise<void> entered,release;
        std::shared_future<void> ready=release.get_future().share();
        ChunkFetchResult fetch(const ChunkKey&) override {
            entered.set_value();ready.wait();ChunkFetchResult r;r.status=ChunkFetchStatus::Missing;return r;
        }
    };
    auto s=service();auto f=std::make_shared<BlockingFetcher>();
    ChunkCache::LevelInfo info;info.shape=info.chunkShape={4,4,4};
    auto a=s->acquireSource("blocked",{info},{f},0,ChunkDtype::UInt8,{});
    auto partition=s->createDecodedPartition(4096);
    auto reading=std::async(std::launch::async,[&]{return a->getChunkBlocking(0,0,0,0);});
    f->entered.get_future().wait();
    auto closing=std::async(std::launch::async,[p=std::move(partition)]() mutable {p.reset();});
    const auto status=closing.wait_for(std::chrono::seconds(1));
    f->release.set_value();closing.get();
    CHECK(status==std::future_status::ready);CHECK(reading.get().status==ChunkStatus::Missing);
}

TEST_CASE("decoded partitions share pool capacities but isolate data and propagate invalidation") {
    auto s=service();auto f=std::make_shared<Fetcher>();auto a=source(s,"a",f,4);
    auto partition=s->createDecodedPartition(4096);auto b=a->inService(partition);
    CHECK(a->physicalCacheIdentity()!=b->physicalCacheIdentity());
    CHECK(a->sourceId()!=b->sourceId());
    CHECK(partition->localReadWorkers()==s->localReadWorkers());
    CHECK(partition->decodeWorkerCapacity()==s->decodeWorkerCapacity());
    s->configureDecodeConcurrency(1);CHECK(partition->decodeConcurrency()==1);
    partition->configureDecodeConcurrency(2);CHECK(s->decodeConcurrency()==2);
    CHECK_THROWS(a->inService(service()));
    REQUIRE(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Data);
    REQUIRE(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::Data);CHECK(f->calls==2);
    CHECK(a->stats().decodedBytes==64);CHECK(b->stats().decodedBytes==64);
    CHECK(partition->decodedByteBudget()->maximumBytes()==4096);
    f->value=-1;s->invalidateSource("a");
    CHECK(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Missing);
    CHECK(b->getChunkBlocking(0,0,0,0).status==ChunkStatus::Missing);
    f->value=31;b->invalidate();
    CHECK(a->getChunkBlocking(0,0,0,0).bytes->at(0)==std::byte(31));
    CHECK(b->getChunkBlocking(0,0,0,0).bytes->at(0)==std::byte(31));CHECK(f->calls==6);
    b.reset();partition.reset();CHECK(a->getChunkBlocking(0,0,0,0).status==ChunkStatus::Data);CHECK(f->calls==6);
}

TEST_CASE("batch resource policy uses granted cores") {
    for(size_t cores:{1,4,16,48,224}){
        auto o=ChunkCacheService::batchOptions(cores,12345);
        CHECK(o.localReadWorkers==std::max<size_t>(16,2*cores));CHECK(o.decodeWorkerCapacity==cores);CHECK(o.decodedByteCapacity==12345);
    }
    CHECK_THROWS(ChunkCacheService::batchOptions(0,1024));CHECK_THROWS(ChunkCacheService::batchOptions(2,0));
}

TEST_CASE("packed hinted channels retain ordinary offsets and share one physical decode") {
    namespace fs=std::filesystem;
    const auto path=fs::temp_directory_path()/("channel_hints_"+std::to_string(std::mt19937_64(std::random_device{}())()));
    struct Cleanup {fs::path p;~Cleanup(){std::error_code ec;fs::remove_all(p,ec);}} cleanup{path};
    utils::ZarrMetadata meta;meta.version=utils::ZarrVersion::v2;meta.shape={3,4,4,4};meta.chunks=meta.shape;
    meta.dtype=utils::ZarrDtype::uint8;meta.compressor_id.clear();meta.fill_value=0;
    auto array=std::make_shared<utils::ZarrArray>(utils::ZarrArray::create(path,meta));
    std::vector<std::byte> data(192);for(size_t c=0;c<3;++c)std::fill(data.begin()+c*64,data.begin()+(c+1)*64,std::byte(20+c));
    array->write_chunk(std::vector<size_t>{0,0,0,0},data);
    std::vector<std::shared_ptr<IChunkedArray>> channels;
    for(size_t c=0;c<3;++c)channels.push_back(acquireProcessZarrChannel(path.string(),array,c));
    auto cache=acquireProcessChunkCache(path.string(),array);
    ChunkRequestContext request;request.channelHints=std::make_shared<ChunkChannelHints>(channels);
    auto first=channels[0]->getChunkBlocking(0,0,0,0,request);REQUIRE(first.bytes);
    for(size_t c=0;c<3;++c){auto r=channels[c]->getChunkBlocking(0,0,0,0);CHECK(r.bytes==first.bytes);CHECK(r.byteOffset==64*c);CHECK(r.bytes->at(r.byteOffset)==std::byte(20+c));}
    CHECK(cache->stats().decodeAttempts==1);
}
