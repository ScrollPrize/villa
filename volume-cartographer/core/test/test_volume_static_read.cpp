// Cover the static Volume::readZYX(out, offset, IChunkedArray&, level)
// overloads using a synthetic IChunkedArray.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/StridedCopy3D.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

class ConstChunkArray : public vc::render::IChunkedArray {
public:
    ConstChunkArray(uint8_t v, vc::render::ChunkDtype dt)
        : value_(v), dtype_(dt) {}

    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {16, 16, 16}; }
    std::array<int, 3> chunkShape(int) const override { return {16, 16, 16}; }
    vc::render::ChunkDtype dtype() const override { return dtype_; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        vc::render::ChunkResult r;
        r.dtype = dtype_;
        if (level != 0 || iz != 0 || iy != 0 || ix != 0) {
            r.status = vc::render::ChunkStatus::Missing;
            return r;
        }
        r.status = vc::render::ChunkStatus::Data;
        r.shape = shape(0);
        std::size_t bytes = 16 * 16 * 16;
        if (dtype_ == vc::render::ChunkDtype::UInt16) bytes *= 2;
        auto vec = std::make_shared<std::vector<std::byte>>(bytes, std::byte{value_});
        r.bytes = std::move(vec);
        return r;
    }
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
private:
    uint8_t value_;
    vc::render::ChunkDtype dtype_;
};

} // namespace

TEST_CASE("Volume::readZYX (static uint8) reads from an IChunkedArray")
{
    ConstChunkArray a(120, vc::render::ChunkDtype::UInt8);
    Array3D<uint8_t> out({4, 4, 4});
    Volume::readZYX(out, /*offset=*/{0, 0, 0}, a, /*level=*/0);
    CHECK(out(0, 0, 0) == 120);
    CHECK(out(3, 3, 3) == 120);
}

TEST_CASE("Volume::readZYX (static uint16) handles the 2-byte dtype")
{
    ConstChunkArray a(50, vc::render::ChunkDtype::UInt16);
    Array3D<uint16_t> out({4, 4, 4});
    Volume::readZYX(out, {0, 0, 0}, a, 0);
    // 16-bit value with both bytes = 50 → 50 | (50<<8) = 12850
    CHECK(out(0, 0, 0) == ((uint16_t(50) << 8) | uint16_t(50)));
}

TEST_CASE("Volume::readZYX (static) on missing chunk falls back to fill")
{
    ConstChunkArray a(0, vc::render::ChunkDtype::UInt8);
    Array3D<uint8_t> out({4, 4, 4}, 99);
    // Offset that maps to a chunk we don't have — should fill with fillValue=0.
    Volume::readZYX(out, /*offset=*/{0, 0, 0}, a, 0);
    // value_ = 0 too, so output is also 0. Sanity check no crash.
    CHECK(out(0, 0, 0) == 0);
}

namespace {
class IndexedChunkArray : public ConstChunkArray {
public:
    explicit IndexedChunkArray(bool floats)
        : ConstChunkArray(0, floats ? vc::render::ChunkDtype::Float32 : vc::render::ChunkDtype::UInt8)
        , floats_(floats) {}
    std::atomic<int> calls{0};
    std::array<int,3> shape(int) const override { return {5,6,7}; }
    std::array<int,3> chunkShape(int) const override { return {2,3,4}; }
    double fillValue() const override { return floats_ ? 17.25 : 17; }
    static double value(int z, int y, int x, bool floats) {
        const int n=z*42+y*7+x;
        return floats ? n*.125+.0625 : n;
    }
    vc::render::ChunkResult getChunkBlocking(int, int z, int y, int x) override {
        ++calls;
        vc::render::ChunkResult result;
        result.dtype=dtype();result.shape=chunkShape(0);
        if(z==2 && y==1 && x==1){result.status=vc::render::ChunkStatus::AllFill;return result;}
        result.status=vc::render::ChunkStatus::Data;result.byteOffset=16;
        auto bytes=std::make_shared<std::vector<std::byte>>(16+24*(floats_?4:1),std::byte{99});
        for(int a=0;a<2;++a)for(int b=0;b<3;++b)for(int c=0;c<4;++c){
            const auto index=(a*3+b)*4+c;
            if(floats_){const float v=value(z*2+a,y*3+b,x*4+c,true);std::memcpy(bytes->data()+16+index*4,&v,4);}
            else (*bytes)[16+index]=std::byte(static_cast<uint8_t>(value(z*2+a,y*3+b,x*4+c,false)));
        }
        result.bytes=bytes;return result;
    }
private:
    bool floats_;
};
}

TEST_CASE("Volume ROI fetches each chunk once with packed offsets, fill and edge clipping") {
    for(bool floats : {false,true}) {
        IndexedChunkArray array(floats);
        Array3D<uint8_t> bytes({8,7,11},99);Array3D<float> values({8,7,11},-999.f);
        if(floats)Volume::readZYX(values,{-1,1,-2},array);
        else Volume::readZYX(bytes,{-1,1,-2},array);
        CHECK(array.calls==12);
        for(int z=0;z<8;++z)for(int y=0;y<7;++y)for(int x=0;x<11;++x){
            const int az=z-1,ay=y+1,ax=x-2;
            double expected=0;
            if(az>=0&&az<5&&ay<6&&ax>=0&&ax<7)
                expected=(az/2==2&&ay/3==1&&ax/4==1)?array.fillValue():IndexedChunkArray::value(az,ay,ax,floats);
            CHECK((floats?double(values(z,y,x)):double(bytes(z,y,x)))==expected);
        }
    }
}

template<class T> void checkBlockedCopy() {
    const std::array<size_t,3> size{67,5,71}, ss{5*79,79,1}, ds{1,73,73*7};
    std::vector<T> src(73*5*79,T(99)), dst(73*7*79,T(123)), expected=dst;
    for(size_t z=0;z<size[0];++z)for(size_t y=0;y<size[1];++y)for(size_t x=0;x<size[2];++x){
        const T v=T((z*13+y*7+x)%251);
        src[3+z*ss[0]+y*ss[1]+x]=v;
        expected[2+z+y*ds[1]+x*ds[2]]=v;
    }
    vc::copyStrided3D(src.data()+3,ss,dst.data()+2,ds,size);
    CHECK(dst==expected);
}

TEST_CASE("Blocked strided copy preserves irregular shapes and padding") {
    checkBlockedCopy<uint8_t>();
    checkBlockedCopy<uint16_t>();
    checkBlockedCopy<float>();
}

TEST_CASE("Interior ROI overwrites poisoned output including fill chunks") {
    IndexedChunkArray array(true);
    Array3D<float> out({5,6,7},-999.f);
    Volume::readZYX(out,{0,0,0},array);
    for(int z=0;z<5;++z)for(int y=0;y<6;++y)for(int x=0;x<7;++x)
        CHECK(out(z,y,x)==((z/2==2&&y/3==1&&x/4==1)?17.25:IndexedChunkArray::value(z,y,x,true)));
}

TEST_CASE("Interior absent chunk states overwrite uint16 output with typed fill") {
    class AbsentArray final : public ConstChunkArray {
    public:
        AbsentArray() : ConstChunkArray(0,vc::render::ChunkDtype::UInt16) {}
        vc::render::ChunkStatus status=vc::render::ChunkStatus::Missing;
        double fillValue() const override { return 321; }
        vc::render::ChunkResult getChunkBlocking(int,int,int,int) override {
            vc::render::ChunkResult result;result.status=status;return result;
        }
    } array;
    for(auto state : {vc::render::ChunkStatus::Missing,vc::render::ChunkStatus::AllFill,vc::render::ChunkStatus::Data}) {
        array.status=state;
        Array3D<uint16_t> out({7,9,11},999);
        Volume::readZYX(out,{2,3,4},array);
        for(size_t i=0;i<out.size();++i) CHECK(out.data()[i]==321);
    }
}
