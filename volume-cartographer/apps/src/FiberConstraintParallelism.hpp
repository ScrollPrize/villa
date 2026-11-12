#pragma once
#include <charconv>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <sched.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

inline int fiberConstraintWorkersForPhysicalCores(int physical) {
    if(physical<1)throw std::invalid_argument("Physical CPU count must be positive");
    return physical;
}

// Count physical cores available to this process, not SMT hardware threads.
// Do not guess an SMT factor when OS topology is unavailable: explicit threads
// remain usable on such platforms and in restrictive containers.
inline int defaultFiberConstraintWorkers() {
#if defined(__linux__)
    cpu_set_t allowed;
    if(sched_getaffinity(0,sizeof(allowed),&allowed)==0) {
        std::set<std::string> cores;
        std::error_code error;
        const std::filesystem::directory_iterator entries("/sys/devices/system/cpu",error);
        if(!error)for(const auto& entry:entries) {
            const auto name=entry.path().filename().string();
            if(!name.starts_with("cpu")||name.size()==3)continue;
            int cpu=-1;
            const auto parsed=std::from_chars(name.data()+3,name.data()+name.size(),cpu);
            if(parsed.ec!=std::errc{}||parsed.ptr!=name.data()+name.size()||cpu<0||cpu>=CPU_SETSIZE||!CPU_ISSET(cpu,&allowed))continue;
            std::ifstream topology(entry.path()/"topology/thread_siblings_list");
            std::string siblings;
            if(!std::getline(topology,siblings)||siblings.empty())
                throw std::runtime_error("Cannot read physical CPU topology; set extraction threads explicitly");
            cores.insert(std::move(siblings));
        }
        if(!cores.empty())return fiberConstraintWorkersForPhysicalCores(int(cores.size()));
    }
#elif defined(__APPLE__)
    int physical=0;size_t bytes=sizeof(physical);
    if(sysctlbyname("hw.physicalcpu",&physical,&bytes,nullptr,0)==0&&physical>0)
        return fiberConstraintWorkersForPhysicalCores(physical);
#endif
    throw std::runtime_error("Cannot determine physical CPU topology; set extraction threads explicitly");
}
