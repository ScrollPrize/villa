#pragma once
#include <algorithm>
#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>

namespace vc::objects {
// Coarse independent native tasks, shared by bulk readers and preparation.
// A callback owns its task's data; connections, when used, remain task-confined.
template<class F>void parallelFor(size_t count,size_t workers,const F& work)
{
    if(!workers)throw std::invalid_argument("parallel worker count must be positive");
    if(!count)return;
    if(workers==1||count==1){for(size_t i=0;i<count;++i)work(i);return;}
    std::atomic<size_t> next{0};std::atomic<bool> failed{false};std::vector<std::future<void>> jobs;
    for(size_t i=0;i<std::min(count,workers);++i)jobs.push_back(std::async(std::launch::async,[&]{
        try{for(size_t n;!failed.load()&&(n=next.fetch_add(1))<count;)work(n);}catch(...){failed.store(true);throw;}
    }));
    std::exception_ptr error;for(auto& job:jobs)try{job.get();}catch(...){if(!error)error=std::current_exception();}
    if(error)std::rethrow_exception(error);
}
}
