// Regression test for concurrent QuadSurface::gen().
//
// The renderer (vc_render_tifxyz, Render.cpp) calls gen() from OpenMP tile
// workers on a *single* surface instance. gen() previously wrote into shared
// mutable scratch members (_genCoordsScratch/_genNormalsScratch/
// _genValidScratch) and lazily built shared caches (_normalCache,
// _validMaskCache) without guarding them. Under concurrency one thread's
// Mat::create() reallocated a buffer another thread was reading -> data race
// and SIGSEGV (issue #1046 / #1054).
//
// This test fails (mismatch or crash) on the unfixed code and passes once the
// scratch buffers are thread-local and the cache builds are guarded.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// A wavy, partly-sparse grid: the z-variation makes normals non-trivial (so the
// _normalCache path matters) and the -1 sentinels force the validity-mask path
// (so _validMaskCache / the per-pixel invalidation pass run too).
cv::Mat_<cv::Vec3f> makeWavyGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float z = 50.f + 6.f * std::sin(0.20f * c) * std::cos(0.17f * r);
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
        }
    }
    // Punch a few invalid cells so validMask() is not all-valid.
    for (int r = 3; r < rows; r += 11)
        for (int c = 5; c < cols; c += 13)
            m(r, c) = cv::Vec3f(-1.f, -1.f, -1.f);
    return m;
}

// Element-wise equality with NaN==NaN (invalid pixels are quiet NaN).
bool vec3fMatsEqual(const cv::Mat_<cv::Vec3f>& a, const cv::Mat_<cv::Vec3f>& b)
{
    if (a.size() != b.size()) return false;
    for (int r = 0; r < a.rows; ++r) {
        const cv::Vec3f* pa = a[r];
        const cv::Vec3f* pb = b[r];
        for (int c = 0; c < a.cols; ++c) {
            for (int k = 0; k < 3; ++k) {
                const float x = pa[c][k], y = pb[c][k];
                const bool bothNan = std::isnan(x) && std::isnan(y);
                if (!bothNan && x != y) return false;
            }
        }
    }
    return true;
}

struct TileParams {
    cv::Vec3f offset;
    float scale;
};

// A spread of pan/zoom requests, mimicking the renderer issuing many gen()
// calls for different viewport tiles of the same surface.
std::vector<TileParams> makeTiles(int n)
{
    std::vector<TileParams> tiles;
    tiles.reserve(n);
    for (int i = 0; i < n; ++i) {
        const float ox = static_cast<float>((i * 7) % 40);
        const float oy = static_cast<float>((i * 5) % 32);
        const float scale = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? 0.5f : 2.0f;
        tiles.push_back({cv::Vec3f(ox, oy, 0.f), scale});
    }
    return tiles;
}

} // namespace

TEST_CASE("QuadSurface::gen is consistent under concurrent calls")
{
    const int rows = 96, cols = 128;
    const cv::Size tileSize(80, 64);
    const auto tiles = makeTiles(64);

    auto pts = makeWavyGrid(rows, cols);

    // --- serial reference (fresh surface, caches cold at start) ----------
    std::vector<cv::Mat_<cv::Vec3f>> refCoords(tiles.size());
    std::vector<cv::Mat_<cv::Vec3f>> refNormals(tiles.size());
    {
        QuadSurface ref(pts, cv::Vec2f(1.f, 1.f));
        for (size_t i = 0; i < tiles.size(); ++i) {
            cv::Mat_<cv::Vec3f> coords, normals;
            ref.gen(&coords, &normals, tileSize, cv::Vec3f(0, 0, 0),
                    tiles[i].scale, tiles[i].offset);
            // gen() returns views into (thread-local) scratch; clone to snapshot
            // before the buffer is reused by the next call.
            refCoords[i] = coords.clone();
            refNormals[i] = normals.clone();
        }
    }

    // --- concurrent run on a single shared surface ----------------------
    QuadSurface shared(pts, cv::Vec2f(1.f, 1.f));
    std::vector<cv::Mat_<cv::Vec3f>> outCoords(tiles.size());
    std::vector<cv::Mat_<cv::Vec3f>> outNormals(tiles.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(tiles.size()); ++i) {
        cv::Mat_<cv::Vec3f> coords, normals;
        shared.gen(&coords, &normals, tileSize, cv::Vec3f(0, 0, 0),
                   tiles[i].scale, tiles[i].offset);
        outCoords[i] = coords.clone();
        outNormals[i] = normals.clone();
    }

    int mismatches = 0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (!vec3fMatsEqual(outCoords[i], refCoords[i])) ++mismatches;
        if (!vec3fMatsEqual(outNormals[i], refNormals[i])) ++mismatches;
    }
    CHECK(mismatches == 0);
}
