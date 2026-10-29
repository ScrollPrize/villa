#include "vc_reader/chunks.hpp"
#include "vc_reader/zarr.hpp"
#include <filesystem>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <atomic>
using namespace vcr::chunks;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
vcr::ChunkPtr bytes(int x) { return std::make_shared<const vcr::Chunk>(64, std::uint8_t(x)); }
int main() try {
    Service service({.decodedBytes=128, .remoteWorkers=1, .localWorkers=1});
    std::mutex mutex;
    std::condition_variable cv;
    bool entered=false, release=false;
    std::vector<int> order;
    auto source=service.addSource([&](auto, auto, auto, auto x) {
        std::unique_lock lock(mutex);
        if (x==0) { entered=true; cv.notify_all(); cv.wait(lock,[&]{return release;}); }
        order.push_back(int(x));
        return bytes(int(x));
    }, 3);
    std::atomic<int> ready=0;
    service.addReadyListener(source,[&](const Key&){ std::lock_guard lock(mutex); ++ready; cv.notify_all(); });
    service.request({source,0,0,0,0});
    { std::unique_lock lock(mutex); cv.wait(lock,[&]{return entered;}); }
    service.replaceViewDemand(1,1,{0,0}, {{{source,0,0,0,1},{0,0},0}, {{source,2,0,0,2},{5,5},2}, {{source,0,0,0,3},{1,1},0}});
    service.replaceViewDemand(1,2,{0,0}, {{{source,0,0,0,1},{0,0},0}, {{source,2,0,0,2},{5,5},2}});
    service.replaceViewDemand(1,1,{0,0}, {{{source,0,0,0,3},{0,0},0}}); // stale
    { std::lock_guard lock(mutex); release=true; } cv.notify_all();
    service.drain(source);
    { std::unique_lock lock(mutex); cv.wait(lock,[&]{return ready==3;}); }
    check(order==std::vector<int>({0,2,1}),"coarse priority and stale demand cancellation");
    check(service.probe({source,0,0,0,0}).status==Status::Unavailable,"budget eviction");
    check(service.probe({source,2,0,0,2}).status==Status::Data,"resident probe");
    service.get({source,0,0,0,4});
    check(service.probe({source,2,0,0,2}).status==Status::Unavailable,"probe does not promote");
    check(service.stats().decodedBytes<=128,"shared byte budget");
    auto bulk=service.addSource([](auto,auto,auto,auto x){return bytes(int(x));},1,{.preferSelfEviction=true});
    service.get({bulk,0,0,0,1});
    check(service.probe({source,0,0,0,4}).status==Status::Data,"bulk preserves viewer chunks");
    auto missing=service.addSource([](auto,auto,auto,auto)->vcr::ChunkPtr{return nullptr;},1);
    check(service.get({missing,0,0,0,0}).status==Status::Fill,"missing record");
    auto failing=service.addSource([](auto,auto,auto,auto)->vcr::ChunkPtr{throw std::runtime_error("failure");},1);
    check(service.get({failing,0,0,0,0}).error=="failure","fetch errors returned");
    service.removeSource(source);
    check(service.probe({source,0,0,0,4}).status==Status::Unavailable,"source removal");
    // Invalidating during a fetch must not let its old result repopulate memory.
    entered=false; release=false;
    auto changing=service.addSource([&](auto,auto,auto,auto) {
        std::unique_lock lock(mutex); entered=true; cv.notify_all();
        cv.wait(lock,[&]{return release;}); return bytes(8);
    },1);
    service.request({changing,0,0,0,0});
    { std::unique_lock lock(mutex); cv.wait(lock,[&]{return entered;}); }
    service.invalidate(changing);
    { std::lock_guard lock(mutex); release=true; } cv.notify_all();
    service.drain(changing);
    check(service.probe({changing,0,0,0,0}).status==Status::Unavailable,"in-flight invalidation");
    // A real writer/opener pipeline exercises decoded data and all-fill detection.
    const auto root = std::filesystem::temp_directory_path() /
        ("vcr-chunks-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{root};
    vcr::zarr::CreateOptions options;
    options.shape={4,4,12}; options.chunks={4,4,4}; options.dtype="|u1";
    options.compressor="zstd"; options.fillValue=7;
    auto writer=vcr::zarr::Writer::create(root,options);
    auto data=bytes(9), fill=bytes(7);
    writer.writeChunk({0,0,0},data->data(),data->size());
    writer.writeChunk({0,0,1},fill->data(),fill->size());
    auto opened=vcr::zarr::openArray(root.string(),{});
    service.configure(512,1,false);
    auto arraySource=service.addSource(std::vector<std::shared_ptr<vcr::Array>>{std::shared_ptr<vcr::Array>(std::move(opened.array))});
    check(*service.get({arraySource,0,0,0,0}).bytes==*data,"Zarr decoded chunk");
    check(service.get({arraySource,0,0,0,1}).status==Status::Fill,"stored all-fill chunk");
    check(service.get({arraySource,0,0,0,2}).status==Status::Fill,"absent Zarr chunk");
    std::atomic<int> activity=0;
    auto remote=service.addSource([](auto,auto,auto,auto){return bytes(1);},1,{.remote=true});
    service.addFetchListener(remote,[&](const Key&,bool){++activity; throw std::runtime_error("listener");});
    check(service.get({remote,0,0,0,0}).status==Status::Data && activity==2,"remote activity and listener isolation");
    std::cout << "chunk service tests passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
