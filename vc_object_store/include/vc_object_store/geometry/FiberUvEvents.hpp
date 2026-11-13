#pragma once
#include "vc_object_store/ParallelFor.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace vc::geometry {

struct StridedUvArc {
    const double* data{};
    ptrdiff_t stride{1};
    double operator[](size_t i)const{return data[i*stride];}
};

struct FiberUvEventInputs {
    std::span<const int64_t> a,b,graphToCatalog,family;
    StridedUvArc arcA,arcB;
    std::span<const bool> parallel,validA,validB,validConnector,nodeMask;
};

// Final solver buffers, not a serialized representation. Native ownership is
// retained by the language binding; no temporary per-event object is built.
struct FiberUvEvents {
    size_t nodes{},events{};
    std::unique_ptr<int64_t[]> selected,a,b;
    std::unique_ptr<double[]> arcA,arcB;
    std::unique_ptr<bool[]> parallel;
};

inline FiberUvEvents prepareFiberUvEvents(const FiberUvEventInputs& in,size_t workers=32)
{
    const size_t m=in.a.size(),n=in.family.size();
    if(!workers)throw std::invalid_argument("UV workers must be positive");
    if(in.b.size()!=m||in.parallel.size()!=m||in.arcA.stride<1||in.arcB.stride<1||
       (m&&(!in.arcA.data||!in.arcB.data)))
        throw std::invalid_argument("UV event columns differ");
    for(const auto mask:{in.validA,in.validB,in.validConnector})
        if(!mask.empty()&&mask.size()!=m)throw std::invalid_argument("UV validity column differs");
    if(!in.nodeMask.empty()&&in.nodeMask.size()!=n)
        throw std::invalid_argument("UV node selection differs");
    constexpr size_t eventBlock=65536,nodeBlock=4096;
    const size_t eventBlocks=(m+eventBlock-1)/eventBlock,nodeBlocks=(n+nodeBlock-1)/nodeBlock;
    vc::objects::parallelFor(nodeBlocks,workers,[&](size_t block){
        for(size_t i=block*nodeBlock;i<std::min(n,(block+1)*nodeBlock);++i)
            if(in.family[i]!=0&&in.family[i]!=1)throw std::invalid_argument("UV family must be H or V");
    });
    auto valid=std::make_unique_for_overwrite<uint8_t[]>(m);
    std::vector<uint8_t> supported(n,0);
    auto mapped=[&](int64_t node){
        if(node<0)throw std::invalid_argument("negative graph node");
        if(in.graphToCatalog.empty()){
            if(uint64_t(node)>=n)throw std::invalid_argument("UV node outside catalog");
            return node;
        }
        if(uint64_t(node)>=in.graphToCatalog.size())throw std::invalid_argument("UV node outside graph mapping");
        const auto value=in.graphToCatalog[node];
        if(value < -1 || (value>=0&&uint64_t(value)>=n))
            throw std::invalid_argument("UV graph mapping outside catalog");
        return value;
    };
    auto marked=[&](std::span<const bool> mask,size_t k){return mask.empty()||mask[k];};
    // First pass admits exactly the ordinary event_batches masks and family
    // relation. Only perpendicular-admitted nodes establish UV support.
    vc::objects::parallelFor(eventBlocks,workers,[&](size_t block){
        const size_t end=std::min(m,(block+1)*eventBlock);
        for(size_t k=block*eventBlock;k<end;++k){
            const auto a=mapped(in.a[k]),b=mapped(in.b[k]);
            const bool keep=a>=0&&b>=0&&marked(in.validA,k)&&marked(in.validB,k)&&
                marked(in.validConnector,k)&&marked(in.nodeMask,a)&&marked(in.nodeMask,b)&&
                in.parallel[k]==(in.family[a]==in.family[b]);
            valid[k]=keep;
            if(keep&&!in.parallel[k]){
                for(const auto node:{a,b}){
                    std::atomic_ref<uint8_t> flag(supported[node]);
                    if(!flag.load(std::memory_order_relaxed))flag.store(1,std::memory_order_relaxed);
                }
            }
        }
    });
    std::vector<size_t> nodeOffsets(nodeBlocks+1,0);
    vc::objects::parallelFor(nodeBlocks,workers,[&](size_t block){
        const size_t end=std::min(n,(block+1)*nodeBlock);size_t count=0;
        for(size_t i=block*nodeBlock;i<end;++i)count+=supported[i]!=0;
        nodeOffsets[block+1]=count;
    });
    for(size_t i=1;i<nodeOffsets.size();++i)nodeOffsets[i]+=nodeOffsets[i-1];
    FiberUvEvents out;out.nodes=nodeOffsets.back();
    out.selected=std::make_unique_for_overwrite<int64_t[]>(out.nodes);
    auto remap=std::make_unique_for_overwrite<int64_t[]>(n);
    vc::objects::parallelFor(nodeBlocks,workers,[&](size_t block){
        const size_t end=std::min(n,(block+1)*nodeBlock);size_t at=nodeOffsets[block];
        for(size_t i=block*nodeBlock;i<end;++i){
            remap[i]=-1;
            if(supported[i]){remap[i]=int64_t(at);out.selected[at++]=int64_t(i);}
        }
    });
    std::vector<size_t> eventOffsets(eventBlocks+1,0);
    vc::objects::parallelFor(eventBlocks,workers,[&](size_t block){
        const size_t end=std::min(m,(block+1)*eventBlock);size_t count=0;
        for(size_t k=block*eventBlock;k<end;++k)
            if(valid[k])count+=remap[mapped(in.a[k])]>=0&&remap[mapped(in.b[k])]>=0;
        eventOffsets[block+1]=count;
    });
    for(size_t i=1;i<eventOffsets.size();++i)eventOffsets[i]+=eventOffsets[i-1];
    out.events=eventOffsets.back();
    out.a=std::make_unique_for_overwrite<int64_t[]>(out.events);
    out.b=std::make_unique_for_overwrite<int64_t[]>(out.events);
    out.arcA=std::make_unique_for_overwrite<double[]>(out.events);
    out.arcB=std::make_unique_for_overwrite<double[]>(out.events);
    out.parallel=std::make_unique_for_overwrite<bool[]>(out.events);
    vc::objects::parallelFor(eventBlocks,workers,[&](size_t block){
        const size_t end=std::min(m,(block+1)*eventBlock);size_t at=eventOffsets[block];
        for(size_t k=block*eventBlock;k<end;++k){
            if(!valid[k])continue;
            const auto a=remap[mapped(in.a[k])],b=remap[mapped(in.b[k])];
            if(a<0||b<0)continue;
            out.a[at]=a;out.b[at]=b;out.arcA[at]=in.arcA[k];out.arcB[at]=in.arcB[k];
            out.parallel[at]=in.parallel[k];++at;
        }
    });
    return out;
}
} // namespace vc::geometry
