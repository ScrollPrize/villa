#include "vc/core/render/IChunkedArray.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace vc::render {
ChunkChannelHints::ChunkChannelHints(std::vector<std::shared_ptr<IChunkedArray>> channels)
    :channels_(std::move(channels))
{
    if(channels_.empty())throw std::invalid_argument("channel hint requires channels");
    for(const auto& channel:channels_){
        if(!channel)throw std::invalid_argument("null hinted channel");
        if(channel->numLevels()!=channels_[0]->numLevels())throw std::invalid_argument("hinted channel levels differ");
        for(int l=0;l<channel->numLevels();++l){
            const auto a=channel->levelTransform(l),b=channels_[0]->levelTransform(l);
            if(channel->shape(l)!=channels_[0]->shape(l)||a.scaleFromLevel0!=b.scaleFromLevel0||a.offsetFromLevel0!=b.offsetFromLevel0)
                throw std::invalid_argument("hinted channels must share a logical voxel grid");
        }
    }
}

void ChunkChannelHints::prefetch(const IChunkedArray& origin,const std::vector<ChunkKey>& keys,
    int priority,const ChunkRequestContext& request) const
{
    // A caller may list only companion channels, or reuse one list containing
    // all channels for every read. Validate an unlisted origin too.
    if(std::none_of(channels_.begin(),channels_.end(),[&](const auto& c){return c.get()==&origin;})){
        if(origin.numLevels()!=channels_[0]->numLevels())throw std::invalid_argument("hinted origin levels differ");
        for(int l=0;l<origin.numLevels();++l){
            const auto a=origin.levelTransform(l),b=channels_[0]->levelTransform(l);
            if(origin.shape(l)!=channels_[0]->shape(l)||a.scaleFromLevel0!=b.scaleFromLevel0||a.offsetFromLevel0!=b.offsetFromLevel0)
                throw std::invalid_argument("hinted origin must share the logical voxel grid");
        }
    }
    auto plain=request;plain.channelHints.reset(); // no recursive fanout
    std::unordered_set<const void*> seen{origin.physicalCacheIdentity()};
    for(const auto& channel:channels_){
        if(!seen.insert(channel->physicalCacheIdentity()).second)continue;
        std::vector<ChunkKey> companions;
        std::unordered_set<ChunkKey,ChunkKeyHash> unique;
        for(const auto& key:keys){
            if(key.level<0||key.level>=origin.numLevels())continue;
            const auto src=origin.chunkShape(key.level),dst=channel->chunkShape(key.level),shape=origin.shape(key.level);
            const std::array<int,3> index{key.iz,key.iy,key.ix};
            std::array<int,3> lo{},hi{};bool valid=true;
            for(int a=0;a<3;++a){
                if(src[a]<=0||dst[a]<=0)throw std::invalid_argument("invalid hinted chunk shape");
                const auto start=int64_t(index[a])*src[a],end=start+src[a]-1;
                valid &= end>=0 && start<shape[a];
                lo[a]=int(std::max<int64_t>(0,start)/dst[a]);
                hi[a]=int(std::min<int64_t>(shape[a]-1,end)/dst[a]);
            }
            if(!valid)continue;
            for(int z=lo[0];z<=hi[0];++z)for(int y=lo[1];y<=hi[1];++y)for(int x=lo[2];x<=hi[2];++x){
                ChunkKey other{key.level,z,y,x};if(unique.insert(other).second)companions.push_back(other);
            }
        }
        if(!companions.empty())channel->prefetchChunks(companions,false,priority,plain);
    }
}
}
