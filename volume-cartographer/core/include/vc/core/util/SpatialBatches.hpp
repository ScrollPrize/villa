#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vc::core::util {

// Large read batches share a fixed CPU budget. With only a few ROIs, callers
// can use the remaining budget inside each batch instead of running serially.
// Independent caller threads allow the sampler's OpenMP loops to use that
// budget without nested OpenMP teams or changing process-global settings.
template<class Fn>
void parallelSpatialBatches(std::size_t batches,int workers,Fn fn) {
    if(workers<1)throw std::invalid_argument("Invalid spatial worker budget");
    if(!batches)return;
    const auto active=std::min<std::size_t>(batches,workers);
    const auto inner=std::max(1,workers/static_cast<int>(active));
    std::atomic<std::size_t> next{0};std::atomic<bool> failed{false};
    const auto consume=[&](std::size_t worker){
        try {
            for(std::size_t job;!failed.load(std::memory_order_relaxed)&&
                (job=next.fetch_add(1,std::memory_order_relaxed))<batches;)
                fn(job,worker,inner);
        }catch(...){failed.store(true,std::memory_order_relaxed);throw;}
    };
    if(active==1){consume(0);return;}
    std::vector<std::future<void>> jobs; jobs.reserve(active);
    for(std::size_t worker=0;worker<active;++worker)
        jobs.push_back(std::async(std::launch::async,consume,worker));
    std::exception_ptr failure;
    for(auto& job:jobs)try{job.get();}catch(...){if(!failure)failure=std::current_exception();}
    if(failure)std::rethrow_exception(failure);
}

// Shared deterministic partition sort/merge for spatial plans and source keys.
template<class T,class Less>
void parallelSort(std::vector<T>& values,Less less,int workers,std::size_t grain=131072) {
    if(workers<1||!grain)throw std::invalid_argument("Invalid parallel sort settings");
    const auto count=values.size();
    const auto parts=std::max<std::size_t>(1,std::min<std::size_t>(workers,std::max<std::size_t>(1,count/grain)));
    std::vector<std::size_t> cuts(parts+1);
    std::vector<std::exception_ptr> errors(parts);
    for(std::size_t p=0;p<=parts;++p)cuts[p]=p*(count/parts)+std::min(p,count%parts);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(parts)) if(parts>1)
#endif
    for(std::ptrdiff_t p=0;p<static_cast<std::ptrdiff_t>(parts);++p)try {
        std::sort(values.begin()+cuts[p],values.begin()+cuts[p+1],less);
    }catch(...){errors[p]=std::current_exception();}
    for(const auto& error:errors)if(error)std::rethrow_exception(error);
    for(std::size_t stride=1;stride<parts;stride*=2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(parts)) if(parts>1)
#endif
        for(std::ptrdiff_t p=0;p<static_cast<std::ptrdiff_t>(parts);p+=2*stride)
            if(std::size_t(p)+stride<parts)try {
                std::inplace_merge(values.begin()+cuts[p],values.begin()+cuts[p+stride],
                    values.begin()+cuts[std::min(std::size_t(p)+2*stride,parts)],less);
            }catch(...){errors[p]=std::current_exception();}
        for(const auto& error:errors)if(error)std::rethrow_exception(error);
    }
}

struct SpatialBatchPlan {
    struct Batch { std::size_t begin, end, weight; };
    std::vector<std::size_t> order;
    std::vector<Batch> batches;
    std::size_t regions=0,maximumWeight=0;
};

// Partition individual positions, never source/UUID/fiber-pair ordering.
// Index tie-breaking and deterministic merges make the plan worker-independent.
template<class Position>
SpatialBatchPlan spatialBatches(std::size_t count,double regionSize,std::size_t target,
                               int workers,Position position) {
    if(!std::isfinite(regionSize)||regionSize<0||!target||workers<1)
        throw std::invalid_argument("Invalid spatial batch settings");
    SpatialBatchPlan result;
    if(!count)return result;
    using Region=std::array<double,3>;
    struct Hash {
        std::size_t operator()(const Region& region)const {
            std::size_t value=0;
            for(auto x:region)value^=std::hash<double>{}(x)+0x9e3779b9u+(value<<6)+(value>>2);
            return value;
        }
    };
#ifdef _OPENMP
    const auto parts=std::min<std::size_t>(workers,std::max<std::size_t>(1,count/16384));
#else
    const std::size_t parts=1;
#endif
    const auto cut=[&](std::size_t p){return p*(count/parts)+std::min(p,count%parts);};
    std::vector<Region> regions(count);
    std::vector<std::unordered_map<Region,std::size_t,Hash>> histograms(parts);
    std::vector<std::exception_ptr> errors(parts);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(parts)) if(parts>1)
#endif
    for(std::ptrdiff_t p=0;p<static_cast<std::ptrdiff_t>(parts);++p)try {
      auto& histogram=histograms[p];
      for(auto i=cut(p);i<cut(p+1);++i) {
        auto& region=regions[i];const auto point=position(i);
        for(int d=0;d<3;++d) {
            const auto value=regionSize>0?std::floor(point[2-d]/regionSize):0.;
            if(!std::isfinite(value)) {
                // Invalid sample positions retain their original values and
                // validity handling; only their scheduling key is isolated.
                region.fill(INFINITY);break;
            }
            region[d]=value;
        }
        ++histogram[region];
      }
    }catch(...){errors[p]=std::current_exception();}
    for(const auto& error:errors)if(error)std::rethrow_exception(error);
    // Sort only distinct ROI keys, not millions of individual sample points.
    // Prefixes visit source blocks in order, preserving the original stable
    // index tie-break independently of hash iteration and worker count.
    std::map<Region,std::vector<std::pair<std::size_t,std::size_t>>> owners;
    for(std::size_t p=0;p<parts;++p)
        for(const auto& [region,n]:histograms[p])owners[region].emplace_back(p,n);
    result.regions=owners.size();result.order.resize(count);
    std::size_t offset=0;
    for(const auto& [region,blocks]:owners) {
        const auto first=offset;
        for(const auto& [p,n]:blocks){histograms[p].at(region)=offset;offset+=n;}
        for(auto begin=first;begin<offset;) {
            const auto end=begin+std::min(target,offset-begin);
            result.batches.push_back({begin,end,end-begin});
            result.maximumWeight=std::max(result.maximumWeight,end-begin);begin=end;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(parts)) if(parts>1)
#endif
    for(std::ptrdiff_t p=0;p<static_cast<std::ptrdiff_t>(parts);++p)
        for(auto i=cut(p);i<cut(p+1);++i)result.order[histograms[p].find(regions[i])->second++]=i;
    return result;
}
} // namespace vc::core::util
