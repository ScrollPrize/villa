#pragma once
#include <nlohmann/json.hpp>
#ifndef _WIN32
#include <sys/resource.h>
#endif
namespace vc::objects::benchmark {
inline nlohmann::json peakRssBytes()
{
#ifndef _WIN32
    rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)==0){
#ifdef __APPLE__
        return int64_t(usage.ru_maxrss);
#else
        return int64_t(usage.ru_maxrss)*1024;
#endif
    }
#endif
    return nullptr;
}
}
