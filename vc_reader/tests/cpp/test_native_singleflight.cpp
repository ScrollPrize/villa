#include <vc_reader/array.hpp>
#include <vc_reader/native/single_flight.hpp>
#include <barrier>
#include <future>
#include <iostream>
#include <thread>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
using namespace std::chrono_literals;
void primitive() {
    vcr::SingleFlight<int> flights(1);
    auto first=flights.acquire("first");auto follower=flights.acquire("first");
    require(first.owner()&&!follower.owner(),"one owner per exact key");
    auto unrelated=flights.acquire("second");
    require(unrelated.owner(),"another key proceeds even in the same metadata shard");
    require(unrelated.resolve([]{return 2;})==2,"unrelated producer");
    require(!follower.ready(),"in-flight result not ready");
    require(first.resolve([]{return 1;})==1&&follower.resolve([]{return 99;})==1,"share exact result");
    auto failed=flights.acquire("failure");auto observer=flights.acquire("failure");
    bool threw=false;try{failed.resolve([]()->int{throw std::runtime_error("source failed");});}catch(...){threw=true;}
    require(threw,"producer error");threw=false;
    try{observer.resolve([]{return 0;});}catch(const std::runtime_error& e){threw=std::string(e.what())=="source failed";}
    require(threw,"same error reaches followers");
    require(flights.acquire("failure").resolve([]{return 3;})==3,"retry after failure");
    std::optional<vcr::SingleFlight<int>::Claim> abandoned;
    {auto owner=flights.acquire("abandoned");abandoned.emplace(flights.acquire("abandoned"));}
    threw=false;try{abandoned->resolve([]{return 0;});}catch(const std::runtime_error&){threw=true;}
    require(threw,"abandoned owner wakes followers with an error");
}

vcr::ArraySpec source(const std::string& identity) {
    vcr::ArraySpec spec;spec.identity=identity;spec.shape={4,2,2,2};spec.chunks={4,2,2,2};
    spec.itemsize=1;spec.fill={0};spec.threads=0;return spec;
}
uint8_t sample(vcr::Array& array,int channel=2,size_t chunk=0) {
    const double p[]{double(channel),0,0,double(chunk*2)};uint8_t out=0;array.sample_nearest(p,1,&out);return out;
}
void shared_computation(bool channels) {
    auto spec=source(channels?"flight-selected":"flight-whole");
    std::promise<void> entered,release;auto released=release.get_future().share();std::atomic<int> calls=0;
    spec.compute=[&](const auto&,auto* out){if(++calls==1)entered.set_value();released.wait();std::fill_n(out,32,uint8_t(42));return true;};
    vcr::Array a(spec,{}),b(spec,{});if(channels)a.request_channel_cache({2});
    auto owner=std::async(std::launch::async,[&]{return sample(a);});entered.get_future().wait();
    auto follower=std::async(std::launch::async,[&]{return sample(b);});
    auto cached=std::async(std::launch::async,[&]{
        const auto chunk=b.cached_chunk({0,0,0,0});return !chunk.first;
    });
    const auto cachedReady=cached.wait_for(1s);const auto waiting=follower.wait_for(20ms);
    release.set_value();
    require(cachedReady==std::future_status::ready&&cached.get(),"cache-only reads never wait on source");
    require(waiting==std::future_status::timeout,"same-key follower waits rather than recomputes");
    require(owner.get()==42&&follower.get()==42&&calls==1,"same chunk decoded once, even without a cache or shared Array instance");
}
void local_loads_join(bool channels) {
    auto spec=source(channels?"local-selected":"local-whole");
    // An existing operation supplies a known result. A path that bypasses the
    // coordinator would consult this absent source and return fill instead.
    spec.layers.push_back({"array","/nonexistent-vcr-singleflight-source","blosc","",false});
    vcr::Array reader(spec,{});if(channels)reader.request_channel_cache({2});
    auto policy=vcr::ChannelPolicy::for_array(spec.identity);
    auto claim=policy->reads.acquire(spec.identity+"|0.0.0.0");
    auto request=std::async(std::launch::async,[&]{return sample(reader);});
    const auto waiting=request.wait_for(20ms);
    claim.resolve([] {
        auto result=std::make_shared<vcr::ChunkLoad>();result->complete=true;
        result->whole=std::make_shared<const vcr::Chunk>(32,uint8_t(73));
        return std::shared_ptr<const vcr::ChunkLoad>(result);
    });
    require(waiting==std::future_status::timeout&&request.get()==73,"local layer reads use the same physical-chunk coordinator");
    require(reader.stats().disk_hits==0,"joined local load did not read/decode another copy");
}
void namespaces() {
    auto a=source("lossless"),b=source("lossy");a.channel_identity=b.channel_identity="same-source";
    std::promise<void> entered,release;auto released=release.get_future().share();
    a.compute=[&](const auto&,auto* out){entered.set_value();released.wait();std::fill_n(out,32,11);return true;};
    b.compute=[](const auto&,auto* out){std::fill_n(out,32,99);return true;};
    vcr::Array lossless(a,{}),lossy(b,{});
    auto first=std::async(std::launch::async,[&]{return sample(lossless);});entered.get_future().wait();
    auto second=std::async(std::launch::async,[&]{return sample(lossy);});
    auto ready=second.wait_for(1s);release.set_value();
    require(ready==std::future_status::ready&&second.get()==99&&first.get()==11,"quality namespaces cannot exchange decoded values");
}
void independent_chunks(bool channels) {
    auto spec=source(channels?"independent-selected":"independent-whole");
    const auto stripe=std::hash<std::string>{}("0.0.0.0")%256;size_t other=1;
    while(std::hash<std::string>{}("0.0.0."+std::to_string(other))%256!=stripe)++other;
    spec.shape[3]=int64_t((other+1)*2);
    std::promise<void> entered,release;auto released=release.get_future().share();
    spec.compute=[&](const auto& idx,auto* out){if(idx[3]==0){entered.set_value();released.wait();}
        std::fill_n(out,32,uint8_t(42));return true;};
    vcr::Array reader(spec,{});if(channels)reader.request_channel_cache({2});
    auto first=std::async(std::launch::async,[&]{return sample(reader);});entered.get_future().wait();
    auto second=std::async(std::launch::async,[&]{return sample(reader,2,other);});
    const auto ready=second.wait_for(1s);release.set_value();
    require(ready==std::future_status::ready&&first.get()==42&&second.get()==42,
        "unrelated chunks must not wait through a matching hash stripe's I/O");
}
void failures(bool channels) {
    auto spec=source(channels?"failure-selected":"failure-whole");
    std::promise<void> entered,release;auto released=release.get_future().share();
    std::atomic<int> calls=0;std::atomic<bool> fail=true;
    spec.compute=[&](const auto&,auto* out){if(++calls==1)entered.set_value();released.wait();
        if(fail)throw std::runtime_error("read failed");std::fill_n(out,32,uint8_t(42));return true;};
    vcr::Array reader(spec,{});if(channels)reader.request_channel_cache({2});
    auto read=[&]{try{sample(reader);}catch(const std::runtime_error& e){return std::string(e.what())=="read failed";}return false;};
    auto first=std::async(std::launch::async,read);entered.get_future().wait();
    auto second=std::async(std::launch::async,read);second.wait_for(20ms);release.set_value();
    require(first.get()&&second.get()&&calls==1,"one failed source operation notifies every waiter");
    fail=false;require(sample(reader)==42&&calls==2,"source failure does not poison later retries");
}
void caller_thread() {
    auto spec=source("thread-bound");spec.caller_thread_only=true;std::atomic<int> calls=0;
    spec.compute=[&](const auto&,auto* out){++calls;std::this_thread::sleep_for(20ms);std::fill_n(out,32,uint8_t(42));return true;};
    vcr::Array reader(spec,{});
    auto first=std::async(std::launch::async,[&]{return sample(reader);});
    auto second=std::async(std::launch::async,[&]{return sample(reader);});
    require(first.get()==42&&second.get()==42&&calls==2,"thread-bound sources retain caller-thread execution");
}
void partial_followers() {
    auto spec=source("partial-followers");
    std::promise<void> entered,release;auto released=release.get_future().share();std::atomic<int> calls=0;
    spec.compute=[&](const auto&,auto* out){if(++calls==1){entered.set_value();released.wait();}
        for(int c=0;c<4;++c)std::fill_n(out+c*8,8,uint8_t(40+c));return true;};
    vcr::Array a(spec,{}),b(spec,{});a.request_channel_cache({2});
    auto first=std::async(std::launch::async,[&]{return sample(a);});entered.get_future().wait();
    auto full=std::async(std::launch::async,[&]{return b.chunk({0,0,0,0});});
    const auto waiting=full.wait_for(20ms);release.set_value();
    const auto result=full.get();
    require(waiting==std::future_status::timeout&&first.get()==42&&calls==2,
        "whole follower reuses partial result then loads remaining planes once, even without cache");
    for(int c=0;c<4;++c)for(int i=0;i<8;++i)
        require((*result)[c*8+i]==40+c,"whole follower assembles correct channel positions");
}
int main(){try{primitive();for(bool selected:{false,true}){shared_computation(selected);local_loads_join(selected);independent_chunks(selected);failures(selected);}namespaces();caller_thread();partial_followers();}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
