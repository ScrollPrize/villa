#pragma once
#include "vc/core/render/IChunkedArray.hpp"
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace vc::render {
// Immutable ownership view, not a cache: pins keep payloads alive until all
// synchronous users finish, even if the regular cache evicts its own entries.
struct PinnedRegionMiss : std::runtime_error {
    explicit PinnedRegionMiss(const std::string& detail):std::runtime_error("sampling exceeded pinned chunk region: "+detail){}
};
class PinnedChunkRegion {
    friend class PinnedChunkLookup;
    friend class PinnedChunkGrid;
    struct Array {std::shared_ptr<IChunkedArray> source;std::map<std::array<int,4>,ChunkResult> chunks;};
    std::unordered_map<const IChunkedArray*,Array> arrays_;
    std::unordered_set<const void*> readyOwners_;
    bool sealed_=false;
public:
    void pin(std::shared_ptr<IChunkedArray>,const std::vector<ChunkKey>&);
    const ChunkResult& get(const IChunkedArray*,int,int,int,int) const;
    void markReady(const void* owner){if(sealed_)throw std::logic_error("sealed pinned region");readyOwners_.insert(owner);}
    // Call once after all channels are pinned, before sharing with seed jobs.
    void seal(){sealed_=true;}
    bool ready(const void* owner) const {return readyOwners_.contains(owner);}
    size_t bytes() const;
};

// Immutable dense table of retained decoded chunks. Ownership is acquired once,
// not per sample. No voxel copies, cache/map lookup, locks or refcount operations
// in corners(). Packed channel byte offsets and edge-chunk strides are resolved
// at construction. Coordinates/order: XYZ input, C-order ZYX decoded payload.
class PinnedChunkGrid {
public:
    // Borrowed immutable addressing plan. Keep a participating grid alive while
    // using it. Only prepareCorners can populate offsets; no payload ownership or
    // shared_ptr operations are performed per sample.
    class CornerAddress {
        friend class PinnedChunkGrid;
        const void* layout_ = nullptr;
        std::array<size_t,8> entries_{}, offsets_{};
        bool single_ = false;
    public:
        CornerAddress() = default;
    };
    PinnedChunkGrid(std::shared_ptr<const PinnedChunkRegion>,
        const std::shared_ptr<IChunkedArray>&,int level,
        std::array<int,3> loXYZ,std::array<int,3> hiXYZ,
        const PinnedChunkGrid* layoutPeer = nullptr);
    size_t tableBytes() const {return entries_.capacity()*sizeof(Entry);}
    bool accepts(const CornerAddress& address) const noexcept {
        return address.layout_ && address.layout_==layoutIdentity_.get();
    }
    CornerAddress prepareCorners(const std::array<int,3>& lo,const std::array<int,3>& hi) const {
        std::array<int,3> c0,c1,l0,l1;
        for(int a=0;a<3;++a){
            if(lo[a]<0 || hi[a]<lo[a] || hi[a]>=volume_[a])
                throw PinnedRegionMiss("corner outside physical volume");
            c0[a]=lo[a]/chunk_[a];c1[a]=hi[a]/chunk_[a];
            l0[a]=lo[a]-c0[a]*chunk_[a];l1[a]=hi[a]-c1[a]*chunk_[a];
            c0[a]-=first_[a];c1[a]-=first_[a];
            if(c0[a]<0 || c1[a]>=count_[a])throw PinnedRegionMiss("corner outside chunk grid");
        }
        const bool single=c0==c1;
        const auto index=[&](int x,int y,int z){return(size_t(z)*count_[1]+y)*count_[0]+x;};
        const size_t common=index(c0[0],c0[1],c0[2]);
        CornerAddress address;address.layout_=layoutIdentity_.get();address.single_=single;
        for(size_t k=0;k<8;++k){
            const size_t entry=single?common:index((k&1)?c1[0]:c0[0],(k&2)?c1[1]:c0[1],(k&4)?c1[2]:c0[2]);
            const auto& e=entries_[entry];address.entries_[k]=entry;
            const int x=(k&1)?l1[0]:l0[0],y=(k&2)?l1[1]:l0[1],z=(k&4)?l1[2]:l0[2];
            address.offsets_[k]=(size_t(z)*e.shape[1]+y)*e.shape[2]+x;
        }
        return address;
    }
    template<class T>
    std::array<T,8> corners(const CornerAddress& address) const {
        static_assert(std::is_same_v<T,uint8_t> || std::is_same_v<T,double>);
        if(!accepts(address))throw std::invalid_argument("corner address belongs to incompatible chunk layout");
        if constexpr(std::is_same_v<T,uint8_t>)
            if(dtype_!=ChunkDtype::UInt8)throw std::logic_error("byte corners require uint8");
        const Entry* common=address.single_?&entries_[address.entries_[0]]:nullptr;
        std::array<T,8> values;
        for(size_t k=0;k<8;++k){
            const auto& e=address.single_?*common:entries_[address.entries_[k]];
            if(!e.data){values[k]=static_cast<T>(fill_);continue;}
            const size_t i=address.offsets_[k];
            if constexpr(std::is_same_v<T,uint8_t>)values[k]=std::to_integer<uint8_t>(e.data[i]);
            else {
                if(dtype_==ChunkDtype::UInt8)values[k]=std::to_integer<uint8_t>(e.data[i]);
                else if(dtype_==ChunkDtype::UInt16){uint16_t v;std::memcpy(&v,e.data+i*2,2);values[k]=v;}
                else {float v;std::memcpy(&v,e.data+i*4,4);values[k]=v;}
            }
        }
        return values;
    }
    template<class T>
    std::array<T,8> corners(const std::array<int,3>& lo,const std::array<int,3>& hi) const {
        return corners<T>(prepareCorners(lo,hi));
    }
private:
    struct Entry {const std::byte* data=nullptr;std::array<int,3> shape{};};
    std::shared_ptr<const PinnedChunkRegion> owner_;
    std::vector<Entry> entries_;
    // Shared only at construction after exact geometry/stride comparison. Dtype,
    // byte offset and fill may differ; prepared offsets are in elements, not bytes.
    std::shared_ptr<const char> layoutIdentity_=std::make_shared<const char>(0);
    std::array<int,3> first_{},count_{},chunk_{},volume_{};
    ChunkDtype dtype_;
    double fill_=0;
};
// Per-operation, per-array lookup. Region must be sealed and outlive this
// object and all borrowed results. No TLS, I/O, locks or shared_ptr increments.
// The tiny pointer window preserves locality across chunk boundaries.
class PinnedChunkLookup {
    const PinnedChunkRegion* region_;
    const IChunkedArray* source_;
    const PinnedChunkRegion::Array* array_;
    ChunkKey lastKey_{};
    const ChunkResult* last_ = nullptr;
    std::array<ChunkKey, 8> keys_{};
    std::array<const ChunkResult*, 8> results_{};
    size_t size_ = 0;
    size_t next_ = 0;
    const ChunkResult& resolve(const ChunkKey&);
public:
    PinnedChunkLookup(const PinnedChunkRegion&, const IChunkedArray*);
    const ChunkResult& get(const ChunkKey& key) {
        if (last_ && lastKey_ == key)
            return *last_;
        return resolve(key);
    }
};
}
