#include "common.hpp"
#include "history.hpp"
#include <vc_reader/components.hpp>
#include <vc_reader/distance_transform.hpp>
#include "../transforms/cuda/select_indices.cuh"
#include "../transforms/cuda/sort_keys.cuh"
namespace vcr::thinning {
namespace tc=transforms::cuda;
namespace {
struct PathInfo {std::uint32_t offset,first,second,seed;};
struct Counts {unsigned long long traces,steps,candidates;};
__global__ void seedFlags(const std::uint8_t* distance,std::uint8_t* flags,unsigned rows,unsigned cols) {
    const auto n=std::size_t(rows)*cols;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x) {
        const auto v=distance[i];bool maximum=v!=0;
        const int y=int(i/cols),x=int(i%cols);
        for(int dy=-1;dy<=1 && maximum;++dy)for(int dx=-1;dx<=1;++dx){
            const int yy=y+dy,xx=x+dx;
            if(yy>=0 && xx>=0 && yy<int(rows) && xx<int(cols) && distance[std::size_t(yy)*cols+xx]>v){maximum=false;break;}
        }
        flags[i]=maximum;
    }
}
__global__ void seedKeys(const std::uint32_t* seeds,const std::int32_t* labels,unsigned long long* keys,unsigned count) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(gridDim.x)*blockDim.x)
        keys[i]=(static_cast<unsigned long long>(labels[seeds[i]])<<32)|seeds[i];
}
__global__ void groupFlags(const unsigned long long* keys,std::uint8_t* flags,unsigned count) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(gridDim.x)*blockDim.x)
        flags[i]=!i || (keys[i]>>32)!=(keys[i-1]>>32);
}
__global__ void componentSizes(const std::int32_t* labels,std::uint32_t* sizes,std::size_t n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        if(labels[i])atomicAdd(sizes+labels[i]-1,1u);
}
__device__ unsigned nextPoint(unsigned current,const std::uint8_t* distance,unsigned rows,unsigned cols,
                             detail::PointHistory& history,const detail::Penalties* table) {
    const int x=int(current%cols),y=int(current/cols);history.push_back({x,y});
    const int dx[8]={-1,0,1,-1,1,-1,0,1},dy[8]={-1,-1,-1,0,0,1,1,1};
    float best=-3.402823466e38f;unsigned next=UINT32_MAX;
    for(int k=0;k<8;++k) {
        const int xx=x+dx[k],yy=y+dy[k];if(xx<0||yy<0||xx>=int(cols)||yy>=int(rows))continue;
        const unsigned at=unsigned(yy)*cols+unsigned(xx);
        const float score=float(distance[at])+table->values[history.count-1][history.code][k];
        if(score>best){best=score;next=at;}
    }
    return next;
}
// Propose short deterministic path prefixes without touching the shared visited
// mask. The ordered component pass accepts only the prefix allowed by that mask.
__global__ void proposals(const std::uint8_t* distance,unsigned rows,unsigned cols,const unsigned long long* keys,
                          unsigned seedCount,const detail::Penalties* table,unsigned* plans,unsigned proposalSteps) {
    for(std::size_t j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;j<seedCount;j+=std::size_t(gridDim.x)*blockDim.x) {
        const unsigned seed=unsigned(keys[j]);unsigned second=UINT32_MAX;
        for(unsigned direction=0;direction<2;++direction) {
            detail::PointHistory history;if(direction && second!=UINT32_MAX)history.push_back({int(second%cols),int(second/cols)});
            unsigned current=seed;
            for(unsigned k=0;k<proposalSteps;++k) {
                auto next=current==UINT32_MAX?UINT32_MAX:nextPoint(current,distance,rows,cols,history,table);
                if(next!=UINT32_MAX && !distance[next])next=UINT32_MAX;
                plans[(std::size_t(j)*2+direction)*proposalSteps+k]=next;
                if(!direction && !k)second=next;
                current=next;
            }
        }
    }
}
struct PathResult {unsigned steps=0,second=UINT32_MAX;unsigned long long candidates=0;};
__device__ PathResult path(unsigned seed,const std::uint8_t* distance,std::uint8_t* visited,
                          unsigned rows,unsigned cols,detail::PointHistory history,
                          const detail::Penalties* table,std::uint32_t* records,const unsigned* proposal,unsigned proposalSteps) {
    const unsigned lane=threadIdx.x&7,mask=0xffu<<((threadIdx.x&31u)&~7u);
    PathResult result;unsigned current=seed;
    const int dx[8]={-1,0,1,-1,1,-1,0,1},dy[8]={-1,-1,-1,0,0,1,1,1};
    for(;;) {
        if(!lane) {
            if(current!=seed)visited[current]=std::uint8_t(result.steps<254?result.steps+1:255);
            if(records)records[result.steps]=current;
        }
        __syncwarp(mask);
        ++result.steps;if(result.steps==2)result.second=current;
        const int x=int(current%cols),y=int(current/cols);history.push_back({x,y});
        unsigned next;
        if(result.steps<=proposalSteps) {
            result.candidates+=unsigned((1+(x>0)+(x+1<int(cols)))*(1+(y>0)+(y+1<int(rows)))-1);
            next=proposal[result.steps-1];
        } else {
            const int xx=x+dx[lane],yy=y+dy[lane];
            const bool valid=xx>=0 && yy>=0 && xx<int(cols) && yy<int(rows);
            result.candidates+=__popc(__ballot_sync(mask,valid));
            next=valid?unsigned(yy)*cols+unsigned(xx):UINT32_MAX;
            float best=valid?float(distance[next])+table->values[history.count-1][history.code][lane]:-3.402823466e38f;
            for(int offset=4;offset;offset/=2) {
                const auto value=__shfl_down_sync(mask,best,offset,8);
                const auto index=__shfl_down_sync(mask,next,offset,8);
                if(lane+offset<8 && (value>best || (value==best && index<next))){best=value;next=index;}
            }
            next=__shfl_sync(mask,next,0,8);
        }
        if(next==UINT32_MAX || !distance[next] || (visited[next] && next!=seed))break;
        current=next;
    }
    return result;
}
__global__ void traceGroups(const std::uint8_t* distance,std::uint8_t* visited,unsigned rows,unsigned cols,
                           const unsigned long long* keys,const std::uint32_t* groups,unsigned groupCount,unsigned seedCount,
                           const detail::Penalties* table,const std::uint32_t* sizes,unsigned long long* allocation,
                           std::uint32_t* records,PathInfo* info,Counts* counts,const unsigned* plans,unsigned proposalSteps) {
    const unsigned lane=threadIdx.x&7,mask=0xffu<<((threadIdx.x&31u)&~7u);
    for(std::size_t g=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/8;g<groupCount;g+=std::size_t(gridDim.x)*blockDim.x/8) {
        const unsigned begin=groups[g],end=g+1<groupCount?groups[g+1]:seedCount;
        // Components are disjoint. Within each component seeds retain scan order.
        unsigned long long offset=0;
        if(records && !lane)offset=atomicAdd(allocation,2ull*sizes[unsigned(keys[begin]>>32)-1]);
        offset=__shfl_sync(mask,offset,0,8);
        Counts local{};
        for(unsigned j=begin;j<end;++j) {
            const unsigned seed=unsigned(keys[j]);
            if(visited[seed]) {if(info && !lane)info[j]={0,0,0,seed};continue;}
            if(!lane)visited[seed]=1;__syncwarp(mask);++local.traces;
            const auto first=path(seed,distance,visited,rows,cols,{},table,records?records+offset:nullptr,proposalSteps?plans+std::size_t(j)*2*proposalSteps:nullptr,proposalSteps);
            detail::PointHistory history;
            if(first.second!=UINT32_MAX)history.push_back({int(first.second%cols),int(first.second/cols)});
            const auto second=path(seed,distance,visited,rows,cols,history,table,records?records+offset+first.steps:nullptr,proposalSteps?plans+(std::size_t(j)*2+(first.steps>1))*proposalSteps:nullptr,proposalSteps);
            if(info && !lane)info[j]={unsigned(offset),first.steps,second.steps,seed};
            offset+=first.steps+second.steps;
            local.steps+=first.steps+second.steps;local.candidates+=first.candidates+second.candidates;
        }
        if(counts && !lane){atomicAdd(&counts->traces,local.traces);atomicAdd(&counts->steps,local.steps);atomicAdd(&counts->candidates,local.candidates);}
    }
}
void run(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols,Traces* traces,Stats* stats) {
    const auto n=detail::validate(input,rows,cols,1);
    if(n && !output)throw std::invalid_argument("null trace output");
    // Packed component/seed keys use uint32 indices. Recorded path offsets use
    // twice the foreground count; reject overflow before allocating buffers.
    if(n>UINT32_MAX || (traces && n>UINT32_MAX/2))throw std::invalid_argument("CUDA trace grid exceeds index capacity");
    if(!n){if(traces)traces->clear();return;}
    if(tc::BatchQueue::current)tc::BatchQueue::current->flush();
    const auto stream=device::stream();Stats timings;
    tc::Buffer<std::uint8_t> distance(n),flags(n);
    tc::Buffer<std::int32_t> labels(n);tc::Buffer<std::uint32_t> seeds(n),count(1);
    const auto dtStart=std::chrono::steady_clock::now();
    distance::cuda::manhattanDevice(input,distance.ptr,{rows,cols});
    timings.distanceTransformSeconds=detail::elapsed(dtStart);
    const auto seedStart=std::chrono::steady_clock::now();
    const std::size_t grid[]{rows,cols};components::cuda::label(distance.ptr,grid,8,labels.ptr);
    seedFlags<<<tc::blocks(n),256,0,stream>>>(distance.ptr,flags.ptr,unsigned(rows),unsigned(cols));tc::check(cudaGetLastError());
    tc::select_flagged_indices(flags,seeds,count);unsigned seedCount=0;count.download(&seedCount);
    tc::check(cudaMemsetAsync(output,0,n,stream));
    timings.seedCount=seedCount;
    if(!seedCount){if(traces)traces->clear();tc::check(cudaStreamSynchronize(stream));if(stats)stats->accumulate(timings);return;}
    tc::Buffer<unsigned long long> keys(seedCount);
    seedKeys<<<tc::blocks(seedCount),256,0,stream>>>(seeds.ptr,labels.ptr,keys.ptr,seedCount);tc::check(cudaGetLastError());
    tc::sort_keys(keys,seedCount);
    if(tc::BatchQueue::current)tc::BatchQueue::current->flush();
    tc::Buffer<std::uint8_t> groupMask(seedCount);tc::Buffer<std::uint32_t> groups(seedCount);
    groupFlags<<<tc::blocks(seedCount),256,0,stream>>>(keys.ptr,groupMask.ptr,seedCount);tc::check(cudaGetLastError());
    tc::select_flagged_indices(groupMask,groups,count);unsigned groupCount=0;count.download(&groupCount);
    timings.seedDetectionSeconds=detail::elapsed(seedStart);
    tc::Buffer<detail::Penalties> table(1);table.upload(&detail::penalties());
    tc::Buffer<std::uint32_t> sizes(traces?n:0),records(traces?2*n:0);
    tc::Buffer<unsigned long long> allocation(traces?1:0);
    tc::Buffer<PathInfo> info(traces?seedCount:0);tc::Buffer<Counts> counts(stats?1:0);
    if(traces) {
        tc::check(cudaMemsetAsync(sizes.ptr,0,n*sizeof(std::uint32_t),stream));tc::check(cudaMemsetAsync(allocation.ptr,0,sizeof(unsigned long long),stream));
        componentSizes<<<tc::blocks(n),256,0,stream>>>(labels.ptr,sizes.ptr,n);tc::check(cudaGetLastError());
    }
    if(stats)tc::check(cudaMemsetAsync(counts.ptr,0,sizeof(Counts),stream));
    // Bound speculation storage independently of image size; long prefixes
    // reduce serialized work within large components without changing paths.
    constexpr std::size_t proposalBudget=256*1024*1024;
    const unsigned proposalSteps=unsigned(std::min<std::size_t>(8,proposalBudget/(std::size_t(seedCount)*2*sizeof(unsigned))));
    tc::Buffer<unsigned> plans(std::size_t(seedCount)*2*proposalSteps);
    const auto traceStart=std::chrono::steady_clock::now();
    if(proposalSteps)proposals<<<tc::blocks(seedCount),256,0,stream>>>(distance.ptr,unsigned(rows),unsigned(cols),keys.ptr,seedCount,table.ptr,plans.ptr,proposalSteps);tc::check(cudaGetLastError());
    traceGroups<<<unsigned((std::size_t(groupCount)+7)/8),64,0,stream>>>(distance.ptr,output,unsigned(rows),unsigned(cols),keys.ptr,groups.ptr,
        groupCount,seedCount,table.ptr,sizes.ptr,allocation.ptr,records.ptr,info.ptr,counts.ptr,plans.ptr,proposalSteps);
    tc::check(cudaGetLastError());tc::check(cudaStreamSynchronize(stream));
    timings.tracePathsSeconds=detail::elapsed(traceStart);
    if(stats){Counts c;counts.download(&c);timings.traceSteps=c.steps;timings.candidateEvaluations=c.candidates;timings.traceCount=traces?c.traces:0;}
    if(traces) {
        std::vector<PathInfo> paths(seedCount);info.download(paths.data());
        unsigned long long used=0;allocation.download(&used);
        std::vector<std::uint32_t> points(used);
        tc::check(cudaMemcpyAsync(points.data(),records.ptr,used*sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream));tc::check(cudaStreamSynchronize(stream));
        std::sort(paths.begin(),paths.end(),[](auto a,auto b){return a.seed<b.seed;});
        traces->clear();
        for(auto p:paths)if(p.first) {
            auto& line=traces->emplace_back();line.reserve(p.first+p.second-1);
            for(unsigned j=p.second;j-->0;){auto v=points[p.offset+p.first+j];line.push_back({int(v%cols),int(v/cols)});}
            for(unsigned j=1;j<p.first;++j){auto v=points[p.offset+j];line.push_back({int(v%cols),int(v/cols)});}
        }
        timings.tracesKept=traces->size();
    }
    if(stats)stats->accumulate(timings);
}
}
namespace cuda {
void traceDevice(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols) {run(input,output,rows,cols,nullptr,nullptr);}
}
namespace detail {
void traceCudaHost(const std::uint8_t* input,std::size_t rows,std::size_t cols,std::uint8_t* output,Traces* traces,Stats* stats,PruneParams params,std::size_t stride) {
    const auto n=rows*cols;tc::Buffer<std::uint8_t> data(n),result(n);
    tc::check(cudaMemcpy2DAsync(data.ptr,cols,input,stride,cols,rows,cudaMemcpyHostToDevice,device::stream()));
    Stats local;run(data.ptr,result.ptr,rows,cols,traces,stats?&local:nullptr);
    if(output)result.download(output);
    if(traces && params.minLengthPx>0) {
        local.tracesKept=0;prune(*traces,params,&local);
        if(output){std::fill_n(output,n,0);for(const auto& line:*traces)for(auto p:line)output[std::size_t(p.y)*cols+p.x]=255;}
    }
    if(stats)stats->accumulate(local);
}
}
}
