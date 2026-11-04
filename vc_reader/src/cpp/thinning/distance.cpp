#include "common.hpp"
#include <vc_reader/distance_transform.hpp>
namespace vcr::thinning::detail {
void prepare(const std::uint8_t* input, int rows, int cols, std::size_t stride, Scratch& scratch, Stats& stats) {
    const auto start = std::chrono::steady_clock::now();
    const auto n = std::size_t(rows) * cols;
    scratch.distance.resize(n);
    auto* d = scratch.distance.data();
    if (stride != std::size_t(cols)) {
        for (int y=0;y<rows;++y) std::copy_n(input+std::size_t(y)*stride,cols,d+std::size_t(y)*cols);
        input=d;
    }
    distance::manhattan(input,d,{std::size_t(rows),std::size_t(cols)});
    stats.distanceTransformSeconds = elapsed(start);
    const auto seedStart = std::chrono::steady_clock::now();
    scratch.seedPoints.clear();
    for (int y = 0; y < rows; ++y) for (int x = 0; x < cols; ++x) {
        const auto v = d[std::size_t(y) * cols + x];
        if (!v) continue;
        bool maximum = true;
        for (int yy = std::max(0, y-1); yy <= std::min(rows-1, y+1) && maximum; ++yy)
            for (int xx = std::max(0, x-1); xx <= std::min(cols-1, x+1); ++xx)
                if (d[std::size_t(yy)*cols+xx] > v) { maximum = false; break; }
        if (maximum) scratch.seedPoints.push_back({x,y});
    }
    stats.seedCount = scratch.seedPoints.size();
    stats.seedDetectionSeconds = elapsed(seedStart);
}
}
