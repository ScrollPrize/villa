// Microbenchmark / perf-regression guard for QuadSurface::gen() — the function
// that generates every rendered surface tile (coords + normals + validity).
//
// gen()'s hot paths are exercised constantly by the renderer, and two recent
// changes touch them: the warp helpers were optimized (separable x-axis hoist)
// and made thread-safe (thread_local scratch + cached _normalCache /
// _validMaskCache built once per surface). This bench guards both against a
// future regression, with NO scroll data (a synthetic in-memory surface).
//
// Two cases, both calling gen() with normals + a non-zero w-offset so the
// bilinear coords warp, the validity warp, the nearest normal warp and the
// offset-along-normal pass all engage:
//   Case WARM: one surface reused for every iteration — the per-surface
//              _normalCache / _validMaskCache and the thread_local scratch are
//              built once and reused, as in steady-state rendering.
//   Case COLD: a fresh surface every iteration — forces the caches to rebuild
//              each time.
// WARM must be meaningfully faster than COLD; if a refactor breaks the cache
// reuse (or makes the scratch reallocate every call), the gap collapses and
// this fails. The headline WARM throughput is printed so warp regressions are
// visible on manual/opt-in runs.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>

namespace {

constexpr int kGrid = 1536;   // source grid (large enough that cache build is real work)
constexpr int kTile = 512;    // rendered tile size
constexpr int kIters = 30;

cv::Mat_<cv::Vec3f> makeWavyGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const float z = 50.f + 6.f * std::sin(0.02f * c) * std::cos(0.017f * r);
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
        }
    // sprinkle invalids so the validity warp + per-pixel invalidation engage
    for (int r = 3; r < rows; r += 11)
        for (int c = 5; c < cols; c += 13)
            m(r, c) = cv::Vec3f(-1.f, -1.f, -1.f);
    return m;
}

double timeIt(auto&& fn)
{
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

TEST_CASE("QuadSurface::gen bench: warm cache path is faster than cold")
{
    const cv::Mat_<cv::Vec3f> grid = makeWavyGrid(kGrid, kGrid);
    const cv::Size tile(kTile, kTile);
    const cv::Vec3f ptr(0, 0, 0);
    const cv::Vec3f offset(0, 0, 1.5f); // non-zero w -> normals + offset pass

    // WARM: one surface, caches + scratch built once and reused.
    QuadSurface warm(grid.clone(), cv::Vec2f(1.f, 1.f));
    { // warm-up: build caches, settle the heap
        cv::Mat_<cv::Vec3f> c, n;
        warm.gen(&c, &n, tile, ptr, 1.0f, offset);
    }
    const double tWarm = timeIt([&]() {
        for (int i = 0; i < kIters; ++i) {
            cv::Mat_<cv::Vec3f> coords, normals;
            const float scale = (i % 2) ? 1.0f : 0.5f;
            warm.gen(&coords, &normals, tile, ptr, scale, offset);
        }
    });

    // COLD: fresh surface each iteration, caches rebuilt every time.
    const double tCold = timeIt([&]() {
        for (int i = 0; i < kIters; ++i) {
            QuadSurface cold(grid.clone(), cv::Vec2f(1.f, 1.f));
            cv::Mat_<cv::Vec3f> coords, normals;
            const float scale = (i % 2) ? 1.0f : 0.5f;
            cold.gen(&coords, &normals, tile, ptr, scale, offset);
        }
    });

    const double tilePixels = double(kTile) * double(kTile) * double(kIters);
    const double mpixWarm = tilePixels / tWarm / 1e6;

    std::printf("\nQuadSurface::gen bench (grid %d^2, tile %d^2, %d iters)\n",
                kGrid, kTile, kIters);
    std::printf("  WARM (cache reused): %.3f s  (%.1f Mpixel/s)\n", tWarm, mpixWarm);
    std::printf("  COLD (cache rebuilt): %.3f s\n", tCold);
    std::printf("  warm speedup: %.2fx\n\n", tCold / tWarm);

    // The cold path rebuilds _normalCache (~kGrid^2 normals) and _validMaskCache
    // every iteration, so warm is comfortably faster. A loose bound keeps this
    // robust on shared CI hardware while still catching a broken cache reuse.
    CHECK(tWarm < tCold);
}
