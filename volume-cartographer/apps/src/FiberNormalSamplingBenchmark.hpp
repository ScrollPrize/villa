#pragma once
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <ctime>
#include <numeric>
#include <nlohmann/json.hpp>

// Isolate normal sampling on real imported geometry. Both implementations use
// one warmed decoded cache, identical batches and the same outer worker pool.
inline nlohmann::json benchmarkFiberNormalSampling(const nlohmann::json& config,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaDataset& dataset,vc::lasagna::LasagnaNormalSamplerOptions options)
{
    using Json=nlohmann::json;
    using Clock=std::chrono::steady_clock;
    using Sample=vc::lasagna::LasagnaNormalSampler::FloatNormalSample;
    const auto count=config.value("points",size_t{131072});
    const auto unique=config.value("unique_points",size_t{4096});
    const auto batch=config.value("batch_points",size_t{2048});
    const auto workers=config.value("workers",size_t{1});
    const auto repetitions=config.value("repetitions",size_t{3});
    const double minimumSeconds=config.value("minimum_measurement_seconds",0.25);
    if(!count||!unique||!batch||!workers||workers>1024||!repetitions||repetitions>100||
       !std::isfinite(minimumSeconds)||minimumSeconds<=0||minimumSeconds>5.)
        throw std::invalid_argument("invalid normal sampling benchmark sizes");
    std::vector<cv::Vec3f> seed;
    for(const auto& line:lines) {
        for(const auto& p:line.pointsBaseXYZ){seed.emplace_back(p);if(seed.size()==unique)break;}
        if(seed.size()==unique)break;
    }
    if(seed.empty())throw std::invalid_argument("normal sampling benchmark needs nonempty geometry");
    std::vector<cv::Vec3f> points;points.reserve(count);
    for(size_t i=0;i<count;++i)points.push_back(seed[i%seed.size()]);
    if(!options.memoryCache)options.memoryCache=std::make_shared<vcr::MemoryCache>(options.maxCachedBytes);
    options.pointInterpolation=false;
    vc::lasagna::LasagnaNormalSampler reference(dataset,options);
    options.pointInterpolation=true;
    vc::lasagna::LasagnaNormalSampler fused(dataset,options);
    const auto channels=fused.regionChannels();
    if(!channels[0].packed||channels[0].cuda||channels[0].array!=channels[1].array||channels[0].array!=channels[2].array)
        throw std::invalid_argument("normal policy benchmark requires one CPU packed channel array");
    // The caller participates, so workers-1 helpers gives this outer width.
    auto pool=vcr::sharedPool(workers-1);
    std::vector<Sample> expected(count),actual(count);
    const auto run=[&](const auto& sampler,auto& out) {
        const auto start=Clock::now();const auto cpu=std::clock();
        pool->parallel_for((count+batch-1)/batch,[&](size_t job){
            const auto begin=job*batch,end=std::min(count,begin+batch);
            std::vector<cv::Vec3f> positions(points.begin()+begin,points.begin()+end);
            std::vector<Sample> values;
            (void)sampler.sampleNormalBatch(positions,1,values);
            std::copy(values.begin(),values.end(),out.begin()+begin);
        });
        return std::pair{std::chrono::duration<double>(Clock::now()-start).count(),
            double(std::clock()-cpu)/CLOCKS_PER_SEC};
    };
    const auto check=[&]{
        for(size_t i=0;i<count;++i){
            bool same=actual[i].valid==expected[i].valid;
            for(int d=0;d<3;++d)same&=std::bit_cast<uint32_t>(actual[i].normal[d])==std::bit_cast<uint32_t>(expected[i].normal[d]);
            if(!same)throw std::runtime_error("normal interpolation differs at sample "+std::to_string(i));
        }
    };
    run(reference,expected);run(fused,actual);check(); // Warmup is separate.
    Json runs=Json::array();std::vector<double> oldTimes,newTimes;
    const auto measure=[&](const auto& sampler,auto& values,const char* method,size_t repetition) {
        size_t passes=0;double wall=0,cpu=0;
        do {const auto time=run(sampler,values);wall+=time.first;cpu+=time.second;++passes;}
        while(wall<minimumSeconds);
        runs.push_back({{"method",method},{"repetition",repetition},{"seconds",wall},
            {"cpu_seconds",cpu},{"passes",passes},{"seconds_per_pass",wall/passes}});
        return wall/passes;
    };
    for(size_t i=0;i<repetitions;++i) {
        const auto old=[&]{oldTimes.push_back(measure(reference,expected,"corners",i));};
        const auto next=[&]{newTimes.push_back(measure(fused,actual,"policy",i));};
        if(i%2){next();old();}else{old();next();}check();
    }
    const auto summary=[&](auto times){
        const auto mean=std::accumulate(times.begin(),times.end(),0.)/times.size();
        std::sort(times.begin(),times.end());
        return Json{{"min_seconds",times.front()},{"median_seconds",times[times.size()/2]},
            {"max_seconds",times.back()},{"mean_seconds",mean},{"median_points_per_second",count/times[times.size()/2]}};
    };
    const auto valid=std::count_if(actual.begin(),actual.end(),[](const auto& n){return n.valid;});
    return {{"points",count},{"unique_points",seed.size()},{"batch_points",batch},{"workers",workers},
        {"repetitions",repetitions},{"minimum_measurement_seconds",minimumSeconds},
        {"valid_normals",valid},{"bit_exact",true},{"cache_bytes",options.memoryCache->nbytes()},
        {"channel_cache",options.channelCache},{"corners",summary(oldTimes)},{"policy",summary(newTimes)},{"runs",runs}};
}
