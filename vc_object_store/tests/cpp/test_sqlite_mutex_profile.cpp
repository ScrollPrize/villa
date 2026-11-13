#include "../../apps/support/SqliteMutexProfile.hpp"
#include <future>
#include <iostream>
#include <vector>

namespace profile=vc::objects::benchmark::sqlite_profile;
int main()
{
    try {
        profile::install(VC_SQLITE_PROFILE_STACKS);
        if(sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0)!=SQLITE_OK || sqlite3_initialize()!=SQLITE_OK)
            throw std::runtime_error("SQLite initialization failed");
        auto* mutex=sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
        if(!mutex)throw std::runtime_error("static mutex unavailable");
        int shared=0;
        std::vector<std::future<nlohmann::json>> threads;
        for(int thread=0;thread<4;++thread)threads.push_back(std::async(std::launch::async,[&]{
            for(int i=0;i<4096;++i){sqlite3_mutex_enter(mutex);++shared;sqlite3_mutex_leave(mutex);}
            auto* recursive=sqlite3_mutex_alloc(SQLITE_MUTEX_RECURSIVE);
            sqlite3_mutex_enter(recursive);sqlite3_mutex_enter(recursive);
            sqlite3_mutex_leave(recursive);sqlite3_mutex_leave(recursive);sqlite3_mutex_free(recursive);
#if VC_SQLITE_PROFILE_STACKS
            auto* pmem=sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_PMEM);
            for(int i=0;i<65536;++i){sqlite3_mutex_enter(pmem);sqlite3_mutex_leave(pmem);}
#endif
            return profile::snapshot();
        }));
        for(auto& thread:threads){
            const auto result=thread.get();
            if(result.at("static_app1").at("entry_calls")!=4096 ||
               result.at("static_app1").at("samples")!=4 ||
               result.at("dynamic_or_unknown").at("entry_calls")!=2)
                throw std::runtime_error("thread-local mutex counters differ");
#if VC_SQLITE_PROFILE_STACKS
            if(result.at("static_pmem").at("stacks").empty())
                throw std::runtime_error("sampled PMEM stack missing");
#endif
        }
        if(shared!=4*4096)throw std::runtime_error("mutex forwarding did not preserve exclusion");
        if(profile::snapshot().contains("static_app1"))
            throw std::runtime_error("worker counters leaked to calling thread");
        sqlite3_shutdown();
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
