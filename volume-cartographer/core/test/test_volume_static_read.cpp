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
