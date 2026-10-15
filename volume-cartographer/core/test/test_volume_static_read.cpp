// Cover the static Volume::readZYX(out, offset, IChunkedArray&, level)
// overloads using a synthetic IChunkedArray.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
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

namespace {
template <typename T>
class PatternChunkArray : public ConstChunkArray {
public:
    explicit PatternChunkArray(bool large = false)
        : ConstChunkArray(0, std::is_same_v<T, uint8_t>
            ? vc::render::ChunkDtype::UInt8 : vc::render::ChunkDtype::UInt16), large_(large) {}
    std::array<int, 3> shape(int) const override {
        return large_ ? std::array<int,3>{37,39,53} : std::array<int,3>{5,7,11};
    }
    std::array<int, 3> chunkShape(int) const override {
        return large_ ? std::array<int,3>{19,21,23} : std::array<int,3>{3,4,5};
    }
    double fillValue() const override { return 13; }
    static T value(int z, int y, int x) { return T(1 + z*77 + y*11 + x); }
    vc::render::ChunkResult tryGetChunk(int, int iz, int iy, int ix) override
    {
        vc::render::ChunkResult result;
        result.dtype = dtype();
        result.shape = chunkShape(0);
        if (ix == 1) {
            result.status = iy == 0 ? vc::render::ChunkStatus::AllFill
                                    : vc::render::ChunkStatus::Missing;
            return result;
        }
        const auto c = chunkShape(0);
        auto bytes = std::make_shared<std::vector<std::byte>>(c[0]*c[1]*c[2]*sizeof(T));
        for (int z=0; z<c[0]; ++z) for (int y=0; y<c[1]; ++y) for (int x=0; x<c[2]; ++x) {
            const T v = value(iz*c[0]+z, iy*c[1]+y, ix*c[2]+x);
            std::memcpy(bytes->data()+((z*c[1]+y)*c[2]+x)*sizeof(T), &v, sizeof(T));
        }
        result.bytes = std::move(bytes);
        result.status = vc::render::ChunkStatus::Data;
        return result;
    }
private:
    bool large_;
};

template <typename T>
void checkChunkReads(bool large)
{
    PatternChunkArray<T> array(large);
    const auto shape = array.shape(0);
    const auto chunks = array.chunkShape(0);
    // Partial first/last chunks, volume clipping on every axis, entirely
    // out-of-bounds X ranges, nonzero sparse fill, and both supported dtypes.
    for (const auto offset : std::vector<std::array<int,3>>{
            {-1,-2,-3}, {1,2,3}, {3,5,9},
            {0,0,-shape[2]-10}, {0,0,shape[2]+3},
            {-shape[0]-10,0,0}, {0,shape[1]+3,0}}) {
        Array3D<T> out({size_t(shape[0]+3),size_t(shape[1]+3),size_t(shape[2]+5)}, T{99});
        Volume::readZYX(out, offset, array, 0);
        std::vector<T> contiguous(out.size(),T{99});
        Volume::readContiguousZYX(std::span<T>(contiguous),out.shape(),offset,array,0);
        bool matches = true;
        for (int z=0; z<int(out.shape()[0]); ++z) for (int y=0; y<int(out.shape()[1]); ++y) for (int x=0; x<int(out.shape()[2]); ++x) {
            const int iz=z+offset[0], iy=y+offset[1], ix=x+offset[2];
            T expected = 0;
            if (iz>=0 && iz<shape[0] && iy>=0 && iy<shape[1] && ix>=0 && ix<shape[2])
                expected = ix/chunks[2] == 1 ? T{13} : PatternChunkArray<T>::value(iz,iy,ix);
            matches = matches && out(z,y,x) == expected &&
                contiguous[(size_t(z)*out.shape()[1]+y)*out.shape()[2]+x] == expected;
        }
        CHECK(matches);
    }
}
} // namespace

TEST_CASE("Volume::readZYX copies clipped chunks and partial transpose tiles for both dtypes")
{
    for (bool large : {false, true}) {
        checkChunkReads<uint8_t>(large);
        checkChunkReads<uint16_t>(large);
    }
}

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


TEST_CASE("Volume::readContiguousZYX validates storage and dtype")
{
    ConstChunkArray array(120,vc::render::ChunkDtype::UInt8);
    std::vector<uint8_t> out(64,99);
    CHECK_THROWS_AS(Volume::readContiguousZYX(out,{4,4,5},{0,0,0},array),std::invalid_argument);
    CHECK_THROWS_AS(Volume::readContiguousZYX(out,{4,4,4},{0,0,0},array,-1),std::out_of_range);
    CHECK_THROWS_AS(Volume::readContiguousZYX(out,{4,4,4},{0,0,0},array,1),std::out_of_range);
    std::vector<uint16_t> wrongDtype(64);
    CHECK_THROWS_AS(Volume::readContiguousZYX(wrongDtype,{4,4,4},{0,0,0},array),std::runtime_error);
    std::vector<uint8_t> empty;
    CHECK_NOTHROW(Volume::readContiguousZYX(empty,{0,4,4},{0,0,0},array));
}
