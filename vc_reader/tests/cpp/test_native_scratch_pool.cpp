#include <vc_reader/native/scratch_pool.hpp>
#include <future>
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#endif
using namespace std::chrono_literals;
using Clock=vcr::ScratchPool::Clock;
Clock::time_point fakeTime{};
Clock::time_point fakeNow(){return fakeTime;}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){try{
    auto stats=std::make_shared<vcr::ScratchStats>();
    {
        vcr::ScratchPool pool(stats,{2s,10s},fakeNow);uint8_t* original=nullptr;
        {auto lease=pool.lease();auto bytes=lease.bytes(0,1024);original=bytes.data();bytes[0]=71;
            {auto nested=pool.lease();auto other=nested.bytes(0,64);other[0]=99;
                require(other.data()!=bytes.data()&&bytes[0]==71,"nested lease cannot overwrite parent");}
            fakeTime+=100s;
            auto trim=std::async(std::launch::async,[&]{pool.trim(fakeTime);});trim.get();
            require(bytes[0]==71&&stats->bytes==1024,"idle trim skips active frame, drops only inactive nested frame");
        }
        fakeTime+=1s;
        {auto lease=pool.lease();auto bytes=lease.bytes(0,1024);require(bytes.data()==original,"reuse capacity");}
        const auto allocations=stats->allocations.load();
        fakeTime+=9s;
        {auto lease=pool.lease();lease.bytes(0,16);require(stats->bytes==1024,"do not shrink immediately after smaller demand");}
        fakeTime+=1s;
        {auto lease=pool.lease();lease.bytes(0,16);require(stats->bytes==16&&stats->shrinks==1,"sustained smaller demand expires high capacity");}
        require(stats->allocations==allocations+1,"warm reuse does not allocate");
        try{auto lease=pool.lease();lease.bytes(1,128);throw std::runtime_error("test");}catch(const std::runtime_error&){}
        {auto lease=pool.lease();lease.bytes(0,16);}
        fakeTime+=3s;pool.trim(fakeTime);require(stats->bytes==0,"idle expiry releases buffers");
        {auto lease=pool.lease();lease.bytes(0,1024);lease.bytes(1,16);}
        fakeTime+=1s;{auto lease=pool.lease();lease.bytes(1,16);}
        fakeTime+=2s;{auto lease=pool.lease();lease.bytes(1,16);}
        require(stats->bytes==16,"unused slot expires while other slots remain busy");
        struct State {int value=0;};
        {auto outer=pool.lease();outer.resource<State>().value=17;
            {auto inner=pool.lease();require(inner.resource<State>().value==0,"nested resource isolated");}
            require(outer.resource<State>().value==17,"nested resource preserves parent");}
        {auto lease=pool.lease();require(lease.resource<State>().value==17,"resource reused");}
        fakeTime+=3s;pool.trim(fakeTime);
        {auto lease=pool.lease();require(lease.resource<State>().value==0,"idle resource released");}
    }
    require(stats->bytes==0,"pool destruction accounts all bytes");
    auto processStats=vcr::scratchMaintenance().stats;
    {auto lease=vcr::workerScratch().lease();lease.bytes(0,65536);}
    require(processStats->bytes>=65536,"track retained scratch");
    const auto deadline=Clock::now()+4s;
    while(processStats->bytes&&Clock::now()<deadline)std::this_thread::sleep_for(20ms);
    require(processStats->bytes==0,"idle scavenger expires a live but unused caller's buffers");
    std::vector<std::thread> workers;
    for(int i=0;i<32;++i)workers.emplace_back([]{
        for(int j=0;j<100;++j){auto lease=vcr::workerScratch().lease();auto data=lease.bytes(0,4096);data[0]=uint8_t(j);}
    });
    for(auto& t:workers)t.join();
#ifndef _WIN32
    const auto child=fork();if(child<0)throw std::runtime_error("fork failed");
    if(child==0){try{auto lease=vcr::workerScratch().lease();lease.bytes(0,16);_exit(0);}catch(...){_exit(1);}}
    int status=0;waitpid(child,&status,0);require(WIFEXITED(status)&&WEXITSTATUS(status)==0,"forked child creates its own scratch service");
#endif
    std::cout<<"Scratch reuse, nested leases, delayed shrink, idle expiry, concurrency and fork passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
