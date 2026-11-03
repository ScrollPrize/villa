#include "vc/core/render/PinnedChunkRegion.hpp"
#include <sstream>
#include <cmath>
#include <limits>
namespace vc::render {
PinnedChunkGrid::PinnedChunkGrid(std::shared_ptr<const PinnedChunkRegion> owner,
    const std::shared_ptr<IChunkedArray>& source,int level,
    std::array<int,3> lo,std::array<int,3> hi,const PinnedChunkGrid* peer):owner_(std::move(owner))
{
    if(!owner_ || !owner_->sealed_ || !source)throw std::invalid_argument("grid requires sealed owned region");
    const auto shape=source->shape(level),chunk=source->chunkShape(level);
    dtype_=source->dtype();fill_=source->fillValue();
    if(dtype_!=ChunkDtype::UInt8 && dtype_!=ChunkDtype::UInt16 && dtype_!=ChunkDtype::Float32)
        throw std::invalid_argument("unsupported decoded grid dtype");
    if(dtype_!=ChunkDtype::Float32 && (!std::isfinite(fill_) || fill_<0 ||
       fill_>(dtype_==ChunkDtype::UInt8?255:65535) || std::trunc(fill_)!=fill_))
        throw std::invalid_argument("invalid integer fill value");
    size_t total=1;
    for(int a=0;a<3;++a){
        volume_[a]=shape[2-a];chunk_[a]=chunk[2-a];
        if(chunk_[a]<=0 || volume_[a]<=0 || hi[a]<lo[a])throw std::invalid_argument("invalid chunk grid bounds");
        const int lower=std::max(0,lo[a]),upper=std::min(volume_[a]-1,hi[a]);
        if(upper<lower)return; // Entire requested box is outside the physical data.
        first_[a]=lower/chunk_[a];count_[a]=upper/chunk_[a]-first_[a]+1;
        if(size_t(count_[a])>std::numeric_limits<size_t>::max()/total)throw std::length_error("chunk grid too large");
        total*=count_[a];
    }
    entries_.resize(total);
    const size_t elementBytes=dtype_==ChunkDtype::UInt8?1:(dtype_==ChunkDtype::UInt16?2:4);
    for(int z=0;z<count_[2];++z)for(int y=0;y<count_[1];++y)for(int x=0;x<count_[0];++x){
        const auto& c=owner_->get(source.get(),level,z+first_[2],y+first_[1],x+first_[0]);
        auto& e=entries_[(size_t(z)*count_[1]+y)*count_[0]+x];e.shape=c.shape;
        if(c.status==ChunkStatus::Missing || c.status==ChunkStatus::AllFill){
            e.shape=chunk;continue; // Logical full-chunk strides for fill entries.
        }
        if(c.status!=ChunkStatus::Data || c.dtype!=dtype_ || !c.bytes)
            throw std::runtime_error("invalid retained chunk data");
        size_t bytes=elementBytes;
        for(int a=0;a<3;++a){
            const int coord=a==0?z+first_[2]:(a==1?y+first_[1]:x+first_[0]);
            if(c.shape[a]<std::min(chunk[a],shape[a]-coord*chunk[a]) || c.shape[a]>chunk[a])
                throw std::runtime_error("invalid retained chunk shape");
            if(size_t(c.shape[a])>std::numeric_limits<size_t>::max()/bytes)throw std::length_error("chunk too large");
            bytes*=c.shape[a];
        }
        if(c.byteOffset>c.bytes->size() || bytes>c.bytes->size()-c.byteOffset)
            throw std::runtime_error("short retained chunk payload");
        e.data=c.bytes->data()+c.byteOffset;
    }
    if(peer && volume_==peer->volume_ && chunk_==peer->chunk_ &&
       first_==peer->first_ && count_==peer->count_ && entries_.size()==peer->entries_.size()) {
        bool compatible=true;
        for(size_t i=0;i<entries_.size();++i)compatible&=entries_[i].shape==peer->entries_[i].shape;
        if(compatible)layoutIdentity_=peer->layoutIdentity_;
    }
}
PinnedChunkLookup::PinnedChunkLookup(const PinnedChunkRegion& region, const IChunkedArray* source)
    : region_(&region), source_(source)
{
    if (!region.sealed_)
        throw std::logic_error("sampling requires a sealed pinned region");
    const auto it = region.arrays_.find(source);
    if (it == region.arrays_.end())
        throw std::logic_error("unregistered channel in pinned region");
    array_ = &it->second;
}
const ChunkResult& PinnedChunkLookup::resolve(const ChunkKey& key)
{
    for (size_t i = 0; i < size_; ++i) {
        if (keys_[i] == key) {
            lastKey_ = key;
            last_ = results_[i];
            return *last_;
        }
    }
    const auto it = array_->chunks.find({key.level, key.iz, key.iy, key.ix});
    if (it == array_->chunks.end())
        return region_->get(source_, key.level, key.iz, key.iy, key.ix); // detailed error
    lastKey_ = key;
    last_ = &it->second;
    const size_t slot = size_ < keys_.size() ? size_++ : next_++ % keys_.size();
    keys_[slot] = key;
    results_[slot] = last_;
    return *last_;
}
void PinnedChunkRegion::pin(std::shared_ptr<IChunkedArray> source,const std::vector<ChunkKey>& keys){
    if(sealed_) throw std::logic_error("cannot populate immutable read region");
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
