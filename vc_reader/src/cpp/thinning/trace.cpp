#include "common.hpp"
#include "history.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>



namespace vcr::thinning {
namespace {

using detail::PointHistory;
constexpr std::array<int,8> kNeighborDx={-1,0,1,-1,1,-1,0,1};
constexpr std::array<int,8> kNeighborDy={-1,-1,-1,0,0,1,1,1};
struct TraceResult {
    bool hasSecondPoint = false;
    Point secondPoint;
    uint64_t steps = 0;
    uint64_t candidateEvaluations = 0;
};

template <bool StorePoints>
static TraceResult tracePath(
    const Point& startPoint,
    const detail::View& distTransform,
    detail::View& visited,
    std::uint8_t* outputImage,
    PointHistory history,
    const Point& seedPoint,
    std::vector<Point>* tracedPoints)
{
    const auto& table=detail::penalties().values;
    Point currentPoint = startPoint;
    std::uint64_t visitIndex = 0;
    TraceResult result;

    if constexpr (StorePoints) {
        tracedPoints->clear();
    }

    while (true) {
        if (currentPoint != seedPoint) {
            auto* visitedRow = visited.ptr<uint8_t>(currentPoint.y);
            if (outputImage != nullptr) {
                visitedRow[currentPoint.x] = static_cast<uint8_t>(std::min<std::uint64_t>(visitIndex + 1, 255));
            } else {
                visitedRow[currentPoint.x] = 1;
            }
        }

        if constexpr (StorePoints) {
            tracedPoints->push_back(currentPoint);
        }

        ++result.steps;
        if (!result.hasSecondPoint && result.steps == 2) {
            result.hasSecondPoint = true;
            result.secondPoint = currentPoint;
        }
        ++visitIndex;

        history.push_back(currentPoint);
        const auto& scores = table[history.count-1][history.code];

        float maxDist = -std::numeric_limits<float>::max();
        Point nextPoint{-1, -1};

        for (size_t ni = 0; ni < kNeighborDx.size(); ++ni) {
            const int nx = currentPoint.x + kNeighborDx[ni];
            const int ny = currentPoint.y + kNeighborDy[ni];
            if (nx < 0 || nx >= distTransform.cols || ny < 0 || ny >= distTransform.rows) {
                continue;
            }

            ++result.candidateEvaluations;
            const auto* distRow = distTransform.ptr<uint8_t>(ny);
            const float dist = static_cast<float>(distRow[nx]);

            const float effectiveDist = dist + scores[ni];
            if (effectiveDist > maxDist) {
                maxDist = effectiveDist;
                nextPoint = Point{nx, ny};
            }
        }

        if (nextPoint.x == -1) {
            break;
        }

        const auto* distRow = distTransform.ptr<uint8_t>(nextPoint.y);
        const auto* visitedRow = visited.ptr<uint8_t>(nextPoint.y);
        if (distRow[nextPoint.x] == 0 ||
            (visitedRow[nextPoint.x] != 0 && nextPoint != seedPoint)) {
            break;
        }

        currentPoint = nextPoint;
    }

    return result;
}

} // namespace

void trace(const std::uint8_t* input, std::size_t rows, std::size_t cols,
           std::uint8_t* outputImage, Traces* traces, Scratch& scratch,
           Stats* stats, PruneParams pruneParams, Backend backend, std::size_t stride)
{
    const auto n = detail::validate(input, rows, cols, 1);
    if (!std::isfinite(pruneParams.minLengthPx) || pruneParams.minLengthPx < 0)
        throw std::invalid_argument("minLengthPx must be finite and nonnegative");
    if (pruneParams.minLengthPx > 0 && !traces)
        throw std::invalid_argument("pruning requires traces");
    if (!stride) stride = cols;
    if (stride < cols || (rows && stride > SIZE_MAX / rows))
        throw std::invalid_argument("invalid input row stride");
    if (backend != Backend::Cpu && backend != Backend::Cuda)
        throw std::invalid_argument("invalid thinning backend");
#ifndef VC_READER_CUDA
    if (backend == Backend::Cuda) throw std::runtime_error("CUDA support is not built");
#endif
    if (!n) { if (traces) traces->clear(); return; }
#ifdef VC_READER_CUDA
    if (backend == Backend::Cuda) {
        detail::traceCudaHost(input,rows,cols,outputImage,traces,stats,pruneParams,stride);return;
    }
#endif
    Stats localStats;
    detail::prepare(input, int(rows), int(cols), stride, scratch, localStats);
    detail::View distTransform{scratch.distance.data(), int(rows), int(cols)};
    if (!outputImage) scratch.visited.resize(n);
    detail::View visited{outputImage ? outputImage : scratch.visited.data(), int(rows), int(cols)};
    std::fill_n(visited.data, n, std::uint8_t(0));

    if (traces != nullptr) {
        traces->clear();
        traces->reserve(scratch.seedPoints.size());
    }

    const auto traceStart = std::chrono::steady_clock::now();

    for (const auto& seed : scratch.seedPoints) {
        auto* visitedRow = visited.ptr<uint8_t>(seed.y);
        if (visitedRow[seed.x] != 0) {
            continue;
        }
        visitedRow[seed.x] = 1;

        PointHistory initialHistory;

        if (traces != nullptr) {
            const auto firstResult = tracePath<true>(
                seed, distTransform, visited, outputImage, initialHistory, seed, &scratch.firstPath);
            localStats.traceSteps += firstResult.steps;
            localStats.candidateEvaluations += firstResult.candidateEvaluations;

            PointHistory secondHistory;
            if (firstResult.hasSecondPoint) {
                secondHistory.push_back(firstResult.secondPoint);
            }
            const auto secondResult = tracePath<true>(
                seed, distTransform, visited, outputImage, secondHistory, seed, &scratch.secondPath);
            localStats.traceSteps += secondResult.steps;
            localStats.candidateEvaluations += secondResult.candidateEvaluations;

            // Build fullTrace directly inside traces' next slot to skip
            // the fullTrace→traces move/copy step entirely.
            auto& dst = traces->emplace_back();
            dst.reserve(scratch.secondPath.size() + scratch.firstPath.size() - (scratch.firstPath.empty() ? 0 : 1));
            dst.insert(dst.end(), scratch.secondPath.rbegin(), scratch.secondPath.rend());
            if (!scratch.firstPath.empty()) {
                dst.insert(dst.end(), scratch.firstPath.begin() + 1, scratch.firstPath.end());
            }
            if (dst.empty()) {
                traces->pop_back();
            } else {
                ++localStats.traceCount;
            }
        } else {
            const auto firstResult = tracePath<false>(
                seed, distTransform, visited, outputImage, initialHistory, seed, nullptr);
            localStats.traceSteps += firstResult.steps;
            localStats.candidateEvaluations += firstResult.candidateEvaluations;

            PointHistory secondHistory;
            if (firstResult.hasSecondPoint) {
                secondHistory.push_back(firstResult.secondPoint);
            }
            const auto secondResult = tracePath<false>(
                seed, distTransform, visited, outputImage, secondHistory, seed, nullptr);
            localStats.traceSteps += secondResult.steps;
            localStats.candidateEvaluations += secondResult.candidateEvaluations;
        }
    }

    localStats.tracePathsSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - traceStart).count();

    if (traces != nullptr) {
        if (pruneParams.minLengthPx > 0.0) {
            prune(*traces, pruneParams, &localStats);

            if (outputImage != nullptr) {
                // Re-rasterise from kept traces. The visit-index encoding is
                // sacrificed when pruning is enabled in image+trace mode.
                std::fill_n(outputImage, n, std::uint8_t(0));
                for (const auto& trace : *traces) {
                    for (const auto& p : trace) {
                        outputImage[std::size_t(p.y) * cols + p.x] = 255;
                    }
                }
            }
        } else {
            localStats.tracesKept += traces->size();
        }
    }

    if (stats != nullptr) {
        stats->accumulate(localStats);
    }
}

void prune(std::vector<std::vector<Point>>& traces,
                         PruneParams params,
                         Stats* stats)
{
    if (!std::isfinite(params.minLengthPx) || params.minLengthPx < 0)
        throw std::invalid_argument("minLengthPx must be finite and nonnegative");
    uint64_t pruned = 0;
    size_t out = 0;
    for (size_t i = 0; i < traces.size(); ++i) {
        auto& trace = traces[i];

        if (trace.size() < 2) {
            ++pruned;
            continue;
        }

        double length = 0.0;
        for (size_t k = 1; k < trace.size(); ++k) {
            const double dx = double(trace[k].x) - trace[k - 1].x;
            const double dy = double(trace[k].y) - trace[k - 1].y;
            length += std::sqrt(dx * dx + dy * dy);
        }

        if (length < params.minLengthPx) {
            ++pruned;
            continue;
        }

        if (out != i) {
            traces[out] = std::move(trace);
        }
        ++out;
    }
    traces.resize(out);

    if (stats != nullptr) {
        stats->tracesPruned += pruned;
        stats->tracesKept += out;
    }
}

} // namespace vcr::thinning
