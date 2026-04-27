// vc_coord_bench: QuadSurface coord-generation and pointTo() throughput.
//
// The live profile shows at_int() at 3.84% and search_min_loc/pointTo at
// 1.83% for QuadSurface-based viewers. Both are called per-pixel during
// coord generation — so they scale with output resolution and are a common
// regression target when anyone touches Geometry.cpp or QuadSurface.cpp.
//
// Two benches here:
//   gen      : QuadSurface::gen over a synthetic grid, 100 iterations.
//              Measures the per-pixel cost of the coord generator (includes
//              pointTo seeding, affine remap, at_int interp of grid points).
//   at_int   : isolated at_int() calls over random (u,v) positions on a
//              synthetic grid. Directly measures bilinear interp perf.
//   pointTo  : repeated pointTo() calls at a sliding target. Covers
//              search_min_loc hot loop.
//
// Usage:
//   vc_coord_bench [--tile N] [--iters N] [--grid N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"

using Clock = std::chrono::steady_clock;

namespace {

double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Build a non-trivial synthetic Vec3f grid: a gently curved sheet in 3D
// space so at_int has real interpolation work and pointTo has a real
// local-minimum search.
cv::Mat_<cv::Vec3f> makeSyntheticGrid(int w, int h) {
    cv::Mat_<cv::Vec3f> g(h, w);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const float fx = float(x), fy = float(y);
        const float z = 100.0f + 5.0f * std::sin(fx * 0.01f) * std::cos(fy * 0.01f);
        g(y, x) = cv::Vec3f(fx * 1.1f, fy * 1.1f, z);
      }
    }
    return g;
}

}  // namespace

int main(int argc, char** argv)
{
    int tile = 512;
    int gridSize = 512;
    int iters = 200;
    int pointToIters = 2000;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](const char* w){ if(i+1>=argc){fprintf(stderr,"%s needs value\n",w);std::exit(1);} return argv[++i]; };
        if      (a == "--tile")           tile = std::atoi(need("--tile"));
        else if (a == "--grid")           gridSize = std::atoi(need("--grid"));
        else if (a == "--iters")          iters = std::atoi(need("--iters"));
        else if (a == "--pointto-iters")  pointToIters = std::atoi(need("--pointto-iters"));
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    printf("vc_coord_bench\n");
    printf("  Tile:          %d x %d\n", tile, tile);
    printf("  Grid:          %d x %d\n", gridSize, gridSize);
    printf("  Gen iters:     %d\n", iters);
    printf("  pointTo iters: %d\n", pointToIters);
    printf("\n");

    auto grid = makeSyntheticGrid(gridSize, gridSize);

    // ==== at_int throughput ====
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> ux(1.0f, float(gridSize - 2));
        std::uniform_real_distribution<float> uy(1.0f, float(gridSize - 2));
        std::vector<cv::Vec2f> pts(iters * 1000);
        for (auto& p : pts) p = cv::Vec2f(ux(rng), uy(rng));

        cv::Vec3f sink{0, 0, 0};
        const auto t0 = Clock::now();
        for (const auto& p : pts) sink += at_int(grid, p);
        const double sec = elapsed(t0, Clock::now());
        volatile float use = sink[0] + sink[1] + sink[2]; (void)use;
        printf("  at_int          %9zu calls  %7.3fs  %10.0f calls/s  %6.1f ns/call\n",
               pts.size(), sec, pts.size() / sec, sec * 1e9 / pts.size());
    }

    // ==== QuadSurface::gen throughput ====
    {
        QuadSurface surf(grid, cv::Vec2f(1.0f, 1.0f));
        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        const cv::Vec3f center(float(gridSize)/2, float(gridSize)/2, 100.0f);
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            cv::Vec3f offset(float(i % 10) * 1.0f, float(i / 10 % 10) * 1.0f, 0.0f);
            surf.gen(&coords, &normals, cv::Size(tile, tile), center, 1.0f, offset);
        }
        const double sec = elapsed(t0, Clock::now());
        const size_t pixels = size_t(iters) * tile * tile;
        printf("  QuadSurface::gen %8d tiles %7.3fs  %10.0f tiles/s  %.1f Mpix/s\n",
               iters, sec, iters / sec, pixels / sec / 1e6);
    }

    // ==== pointTo / search_min_loc throughput ====
    {
        QuadSurface surf(grid, cv::Vec2f(1.0f, 1.0f));
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> tx(50.0f, float(gridSize) - 50.0f);
        std::uniform_real_distribution<float> ty(50.0f, float(gridSize) - 50.0f);
        cv::Vec3f ptr{0,0,0};
        const auto t0 = Clock::now();
        for (int i = 0; i < pointToIters; ++i) {
            cv::Vec3f tgt(tx(rng), ty(rng), 100.0f);
            surf.pointTo(ptr, tgt, 0.5f, 100);
        }
        const double sec = elapsed(t0, Clock::now());
        volatile float use = ptr[0] + ptr[1] + ptr[2]; (void)use;
        printf("  pointTo          %8d calls  %7.3fs  %10.0f calls/s  %6.1f us/call\n",
               pointToIters, sec, pointToIters / sec, sec * 1e6 / pointToIters);
    }

    printf("\nDone.\n");
    return 0;
}
