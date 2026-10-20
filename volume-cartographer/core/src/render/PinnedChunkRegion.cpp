#include "vc/core/render/PinnedChunkRegion.hpp"
#include <sstream>
namespace vc::render {
namespace {thread_local const PinnedChunkRegion* active=nullptr;}
const PinnedChunkRegion* activePinnedChunkRegion(){return active;}
ScopedPinnedChunkRegion::ScopedPinnedChunkRegion(const PinnedChunkRegion* region):previous_(active){active=region;}
ScopedPinnedChunkRegion::~ScopedPinnedChunkRegion(){active=previous_;}
const ChunkResult* borrowedPinnedChunk(const IChunkedArray* array,int l,int z,int y,int x){
    return active?&active->get(array,l,z,y,x):nullptr;
}
void PinnedChunkRegion::pin(std::shared_ptr<IChunkedArray> source,const std::vector<ChunkKey>& keys){
    if(active || sealed_) throw std::logic_error("cannot populate immutable read region");
    auto& a=arrays_[source.get()];a.source=source;
    source->prefetchChunks(keys,false);
    for(const auto& k:keys){
        auto key=std::array{k.level,k.iz,k.iy,k.ix};if(a.chunks.contains(key))continue;
        auto value=source->getChunkBlocking(k.level,k.iz,k.iy,k.ix);
        if(value.status==ChunkStatus::Error)throw std::runtime_error(value.error);
        if(value.status==ChunkStatus::MissQueued)throw std::runtime_error("unresolved pinned dependency");
        a.chunks.emplace(key,std::move(value));
    }
}
const ChunkResult& PinnedChunkRegion::get(const IChunkedArray* source,int l,int z,int y,int x)const{
    const auto a=arrays_.find(source);
    if(a==arrays_.end())throw std::logic_error("unregistered channel in pinned region");
    const auto it=a->second.chunks.find({l,z,y,x});
    if(it==a->second.chunks.end()) {
        std::ostringstream detail;
        detail<<"array="<<source<<" level="<<l<<" zyx="<<z<<','<<y<<','<<x;
        throw PinnedRegionMiss(detail.str());
    }
    return it->second;
}
size_t PinnedChunkRegion::bytes()const{
    std::unordered_set<const void*> seen;size_t n=0;
    for(const auto& [_,a]:arrays_)for(const auto& [__,c]:a.chunks)
        if(c.bytes && seen.insert(c.bytes.get()).second)n+=c.bytes->size();
    return n;
}
}
