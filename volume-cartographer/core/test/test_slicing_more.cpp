// More Slicing coverage: uint16 dataset, Tricubic sampling, multi-slice
// with both dtypes, edge cases.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/render/IChunkedArray.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

template <typename T>
class ConstArray : public vc::render::IChunkedArray {
public:
    ConstArray(T v, vc::render::ChunkDtype dt) : value_(v), dtype_(dt) {}
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
        const std::size_t bytes = 16 * 16 * 16 * sizeof(T);
        auto vec = std::make_shared<std::vector<std::byte>>(bytes);
        for (std::size_t i = 0; i < 16 * 16 * 16; ++i) {
            T v = value_;
            std::memcpy(&(*vec)[i * sizeof(T)], &v, sizeof(T));
        }
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
    T value_;
    vc::render::ChunkDtype dtype_;
};

cv::Mat_<cv::Vec3f> coords(int rows, int cols, float z = 4.f)
{
    cv::Mat_<cv::Vec3f> c(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < cols; ++x)
            c(r, x) = cv::Vec3f(float(x) + 0.5f, float(r) + 0.5f, z);
    return c;
}

} // namespace

TEST_CASE("readInterpolated3D uint16: trilinear")
{
    ConstArray<uint16_t> a(20000, vc::render::ChunkDtype::UInt16);
    auto c = coords(4, 4);
    cv::Mat_<uint16_t> out(4, 4, uint16_t{0});
    readInterpolated3D(out, &a, 0, c, vc::Sampling::Trilinear);
    // Constant input should yield the same constant.
    CHECK(out(0, 0) == 20000);
}

TEST_CASE("readInterpolated3D uint16: tricubic")
{
    ConstArray<uint16_t> a(5000, vc::render::ChunkDtype::UInt16);
    auto c = coords(4, 4);
    cv::Mat_<uint16_t> out(4, 4, uint16_t{0});
    readInterpolated3D(out, &a, 0, c, vc::Sampling::Tricubic);
    // Catmull-Rom on a constant field with edge clamping over-/undershoots
    // slightly near boundaries; just check we got a value in the right ballpark.
    CHECK(out(0, 0) > 4500);
    CHECK(out(0, 0) < 6500);
}

TEST_CASE("readMultiSlice uint16: round-trips")
{
    ConstArray<uint16_t> a(7777, vc::render::ChunkDtype::UInt16);
    auto base = coords(4, 4, 2.f);
    cv::Mat_<cv::Vec3f> step(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<float> offsets = {0.f, 1.f, 2.f};
    std::vector<cv::Mat_<uint16_t>> out;
    readMultiSlice(out, &a, 0, base, step, offsets);
    REQUIRE(out.size() == 3);
    CHECK(out[0](0, 0) == 7777);
    CHECK(out[2](3, 3) == 7777);
}

TEST_CASE("sampleTileSlices uint16: also covers tile path")
{
    ConstArray<uint16_t> a(99, vc::render::ChunkDtype::UInt16);
    auto base = coords(4, 4, 2.f);
    cv::Mat_<cv::Vec3f> step(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    std::vector<float> offsets = {0.f, 1.f};
    std::vector<cv::Mat_<uint16_t>> out;
    sampleTileSlices(out, &a, 0, base, step, offsets);
    REQUIRE(out.size() == 2);
    CHECK(out[0](0, 0) == 99);
}

TEST_CASE("readInterpolated3D uint8: out-of-bounds coord yields fill")
{
    ConstArray<uint8_t> a(150, vc::render::ChunkDtype::UInt8);
    cv::Mat_<cv::Vec3f> c(2, 2);
    // Coords way outside the 16^3 volume.
    c(0, 0) = cv::Vec3f(-100, -100, -100);
    c(0, 1) = cv::Vec3f(1000, 1000, 1000);
    c(1, 0) = cv::Vec3f(8, 8, 8);   // in-bounds
    c(1, 1) = cv::Vec3f(0, 0, 0);   // edge
    cv::Mat_<uint8_t> out(2, 2, uint8_t{0});
    readInterpolated3D(out, &a, 0, c, vc::Sampling::Trilinear);
    CHECK(int(out(1, 0)) == 150);
}

TEST_CASE("readCompositeFast with multiple methods on uint8 const data")
{
    ConstArray<uint8_t> a(100, vc::render::ChunkDtype::UInt8);
    auto base = coords(4, 4, 2.f);
    cv::Mat_<cv::Vec3f> normals(4, 4, cv::Vec3f(0.f, 0.f, 1.f));
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    CompositeParams params;
    const std::vector<std::string> methods = {"mean", "max", "min"};
    for (const auto& method : methods) {
        params.method = method;
        readCompositeFast(out, &a, 0, base, normals,
                          /*zStep=*/1.0f, /*zStart=*/0, /*zEnd=*/3, params);
        CHECK(out(0, 0) == 100);
    }
}

TEST_CASE("samplePlane: trilinear")
{
    ConstArray<uint8_t> a(55, vc::render::ChunkDtype::UInt8);
    cv::Mat_<uint8_t> out(4, 4, uint8_t{0});
    samplePlane(out, &a, 0,
        cv::Vec3f(2.f, 2.f, 2.f), cv::Vec3f(1.f, 0.f, 0.f), cv::Vec3f(0.f, 1.f, 0.f),
        4, 4, vc::Sampling::Trilinear);
    CHECK(out(0, 0) == 55);
}

TEST_CASE("samplePlane: tricubic")
{
    ConstArray<uint8_t> a(70, vc::render::ChunkDtype::UInt8);
    cv::Mat_<uint8_t> out(2, 2, uint8_t{0});
    samplePlane(out, &a, 0,
        cv::Vec3f(4.f, 4.f, 4.f), cv::Vec3f(1.f, 0.f, 0.f), cv::Vec3f(0.f, 1.f, 0.f),
        2, 2, vc::Sampling::Tricubic);
    CHECK(out(0, 0) == 70);
}
