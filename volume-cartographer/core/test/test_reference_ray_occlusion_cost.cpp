// ReferenceRayOcclusionCost and ReferenceRayOcclusionAnalyticCost on a
// synthetic volume holding one occluding slab: a ray that crosses the slab
// costs something, a free ray costs nothing, the analytic cost returns the
// functor's residual bit for bit, and its Jacobian matches central
// differences wherever the march does not change between the two probes.
//
// ceres pulls in glog, whose CHECK macro replaces the test framework's and
// aborts on failure, so the assertions here use REQUIRE.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/ReferenceRayOcclusionCost.hpp"

#include "vc/core/render/IChunkedArray.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

constexpr int kSide = 64;
constexpr int kChunk = passTroughComputor::CHUNK_SIZE;
constexpr double kThreshold = 128.0;

// The slab: unit normal n through the volume centre, value 255 within 2 voxels
// of the mid plane, falling linearly to 0 at 6 voxels, rounded to uint8.
const cv::Vec3d kNormal = cv::normalize(cv::Vec3d(0.48, 0.60, 0.64));
const cv::Vec3d kCentre(32.0, 32.0, 32.0);

std::uint8_t slab_value(int x, int y, int z)
{
    const double d = std::abs((cv::Vec3d(x, y, z) - kCentre).dot(kNormal));
    const double v = d <= 2.0 ? 255.0 : std::max(0.0, 255.0 * (6.0 - d) / 4.0);
    return static_cast<std::uint8_t>(std::lround(v));
}

class SlabArray : public vc::render::IChunkedArray {
public:
    int numLevels() const override { return 1; }
    std::array<int, 3> shape(int) const override { return {kSide, kSide, kSide}; }
    std::array<int, 3> chunkShape(int) const override { return {kChunk, kChunk, kChunk}; }
    vc::render::ChunkDtype dtype() const override { return vc::render::ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int) const override { return {}; }

    vc::render::ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override
    {
        vc::render::ChunkResult r;
        r.dtype = vc::render::ChunkDtype::UInt8;
        const int n = kSide / kChunk;
        if (level != 0 || iz < 0 || iy < 0 || ix < 0 || iz >= n || iy >= n || ix >= n) {
            r.status = vc::render::ChunkStatus::Missing;
            return r;
        }
        r.status = vc::render::ChunkStatus::Data;
        r.shape = chunkShape(0);
        auto bytes = std::make_shared<std::vector<std::byte>>(
            static_cast<std::size_t>(kChunk) * kChunk * kChunk);
        std::size_t i = 0;
        for (int z = 0; z < kChunk; ++z)
            for (int y = 0; y < kChunk; ++y)
                for (int x = 0; x < kChunk; ++x)
                    (*bytes)[i++] = std::byte{slab_value(ix * kChunk + x, iy * kChunk + y, iz * kChunk + z)};
        r.bytes = std::move(bytes);
        return r;
    }
    vc::render::ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override
    {
        return tryGetChunk(level, iz, iy, ix);
    }
    void prefetchChunks(const std::vector<vc::render::ChunkKey>&, bool, int) override {}
    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback) override { return 0; }
    void removeChunkReadyListener(ChunkReadyCallbackId) override {}
};

struct SlabVolume {
    SlabArray array;
    passTroughComputor compute;
    Chunked3d<std::uint8_t, passTroughComputor> volume{compute, std::array<int, 3>{kSide, kSide, kSide}, &array, 0};
};

double functor_residual(SlabVolume& s, const cv::Vec3d& start, const cv::Vec3d& target)
{
    ReferenceRayOcclusionCost functor(&s.volume, target, kThreshold, 1.0, 1.0, 0.0);
    const double x[3] = {start[0], start[1], start[2]};
    double r = -1.0;
    REQUIRE(functor(x, &r));
    return r;
}

double analytic_residual(SlabVolume& s, const cv::Vec3d& start, const cv::Vec3d& target, double* J = nullptr)
{
    ReferenceRayOcclusionAnalyticCost cost(&s.volume, target, kThreshold, 1.0, 1.0, 0.0);
    double x[3] = {start[0], start[1], start[2]};
    double* params[1] = {x};
    double* jac[1] = {J};
    double r = -1.0;
    REQUIRE(cost.Evaluate(params, &r, J ? jac : nullptr));
    return r;
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

cv::Vec3i hit_cell(const ReferenceRayOcclusionAnalyticCost::March& m)
{
    return cv::Vec3i(static_cast<int>(std::floor(m.point[0])),
                     static_cast<int>(std::floor(m.point[1])),
                     static_cast<int>(std::floor(m.point[2])));
}

bool same_march(const ReferenceRayOcclusionAnalyticCost::March& a,
                const ReferenceRayOcclusionAnalyticCost::March& b)
{
    return a.hit == b.hit && a.k == b.k && a.n == b.n && hit_cell(a) == hit_cell(b);
}

} // namespace

TEST_CASE("reference ray occlusion: a ray through the slab costs, a free ray does not")
{
    SlabVolume s;
    const cv::Vec3d crossing_start = kCentre - 15.0 * kNormal;
    const cv::Vec3d crossing_target = kCentre + 15.0 * kNormal;
    const cv::Vec3d free_start = kCentre + 10.0 * kNormal;
    const cv::Vec3d free_target = kCentre + 20.0 * kNormal;

    const double crossing = functor_residual(s, crossing_start, crossing_target);
    const double free_ray = functor_residual(s, free_start, free_target);
    REQUIRE(crossing > 0.0);
    REQUIRE(free_ray == 0.0);

    REQUIRE(same_bits(analytic_residual(s, crossing_start, crossing_target), crossing));
    REQUIRE(same_bits(analytic_residual(s, free_start, free_target), free_ray));
}

TEST_CASE("reference ray occlusion: analytic Jacobian against central differences off the kinks")
{
    SlabVolume s;
    std::mt19937 rng(20260924);
    std::uniform_real_distribution<double> before(-18.0, -7.0);
    std::uniform_real_distribution<double> after(7.0, 18.0);
    std::uniform_real_distribution<double> lateral(-8.0, 8.0);

    // Two unit vectors in the slab plane, for the lateral offsets.
    cv::Vec3d u = kNormal.cross(cv::Vec3d(1.0, 0.0, 0.0));
    u = cv::normalize(u);
    const cv::Vec3d w = kNormal.cross(u);

    const double h = 1e-4;
    const double rel = 1e-6;
    int accepted = 0;
    int attempts = 0;
    int residual_bits_differ = 0;
    int over_bound = 0;
    double worst = 0.0;

    while (accepted < 1000 && attempts < 200000) {
        ++attempts;
        const cv::Vec3d start = kCentre + before(rng) * kNormal + lateral(rng) * u + lateral(rng) * w;
        const cv::Vec3d target = kCentre + after(rng) * kNormal + lateral(rng) * u + lateral(rng) * w;

        ReferenceRayOcclusionAnalyticCost cost(&s.volume, target, kThreshold, 1.0, 1.0, 0.0);
        ReferenceRayOcclusionAnalyticCost::March at;
        const double x[3] = {start[0], start[1], start[2]};
        const double r = cost.march(x, &at);
        if (!at.hit || r <= 0.0)
            continue;

        bool kink = false;
        double fd[3];
        for (int i = 0; i < 3 && !kink; ++i) {
            double xp[3] = {x[0], x[1], x[2]};
            double xm[3] = {x[0], x[1], x[2]};
            xp[i] += h;
            xm[i] -= h;
            ReferenceRayOcclusionAnalyticCost::March mp, mm;
            const double rp = cost.march(xp, &mp);
            const double rm = cost.march(xm, &mm);
            kink = !same_march(at, mp) || !same_march(at, mm);
            fd[i] = (rp - rm) / (2.0 * h);
        }
        if (kink)
            continue;

        double J[3] = {0.0, 0.0, 0.0};
        const double r_eval = analytic_residual(s, start, target, J);
        residual_bits_differ += !same_bits(r_eval, functor_residual(s, start, target));

        const double scale = std::max({std::abs(J[0]), std::abs(J[1]), std::abs(J[2])});
        REQUIRE(scale > 0.0);
        for (int i = 0; i < 3; ++i) {
            const double e = std::abs(J[i] - fd[i]) / scale;
            worst = std::max(worst, e);
            over_bound += e > rel;
        }
        ++accepted;
    }

    MESSAGE("rays drawn " << attempts << ", accepted " << accepted << ", residual bits differ "
            << residual_bits_differ << ", entries over 1e-6 " << over_bound << ", worst relative " << worst);
    REQUIRE(accepted == 1000);
    REQUIRE(residual_bits_differ == 0);
    REQUIRE(over_bound == 0);
}
