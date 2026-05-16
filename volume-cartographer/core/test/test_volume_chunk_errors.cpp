#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace {

class SyntheticChunkArray final : public vc::render::IChunkedArray
{
public:
    int numLevels() const override { return 1; }

    std::array<int, 3> shape(int) const override { return {4, 4, 4}; }

    std::array<int, 3> chunkShape(int) const override { return {2, 2, 2}; }

    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }

    double fillValue() const override { return 0.0; }

    LevelTransform levelTransform(int) const override { return {}; }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        return getChunkBlocking(level, iz, iy, ix);
    }

    vc::render::ChunkResult getChunkBlocking(int, int iz, int iy, int ix) override
    {
        if (iz == 0 && iy == 0 && ix == 0) {
            return {vc::render::ChunkStatus::Error,
                    vc::render::ChunkDtype::UInt8,
                    chunkShape(0),
                    nullptr,
                    "synthetic chunk error"};
        }

        auto bytes = std::make_shared<std::vector<std::byte>>(8);
        std::fill(bytes->begin(), bytes->end(), std::byte{7});
        return {vc::render::ChunkStatus::Data,
                vc::render::ChunkDtype::UInt8,
                chunkShape(0),
                bytes,
                {}};
    }

    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int = 0) override {}

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 1; }

    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

} // namespace

TEST_CASE("Volume::readZYX reports chunk errors outside OpenMP workers")
{
    SyntheticChunkArray array;
    Array3D<uint8_t> out({4, 4, 4});

    bool threw = false;
    try {
        Volume::readZYX(out, {0, 0, 0}, array, 0);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        CHECK(message.find("Volume::read failed fetching chunk 0/0/0/0") != std::string::npos);
        CHECK(message.find("synthetic chunk error") != std::string::npos);
    }

    CHECK(threw);
}
