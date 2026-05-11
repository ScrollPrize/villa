#include "vc/core/util/Thinning.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kHistoryLength = 4;
constexpr std::array<int, 8> kNeighborDx = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr std::array<int, 8> kNeighborDy = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr std::array<std::array<float, kHistoryLength>, kHistoryLength> kHistoryWeights = {{
    {{1.0f, 0.0f, 0.0f, 0.0f}},
    {{1.0f, 0.25f, 0.0f, 0.0f}},
    {{1.0f, 0.625f, 0.25f, 0.0f}},
    {{1.0f, 0.75f, 0.5f, 0.25f}},
}};
constexpr std::array<float, kHistoryLength> kHistoryWeightSums = {
    1.0f,
    1.25f,
    1.875f,
    2.5f,
};

struct PointHistory {
    std::array<cv::Point, kHistoryLength> points{};
    int count = 0;
    int start = 0;

    void push_back(const cv::Point& point)
    {
        if (count < kHistoryLength) {
            points[(start + count) % kHistoryLength] = point;
            ++count;
            return;
        }

        points[start] = point;
        start = (start + 1) % kHistoryLength;
    }

    const cv::Point& at(int idx) const
    {
        return points[(start + idx) % kHistoryLength];
    }
};

struct TraceResult {
    bool hasSecondPoint = false;
    cv::Point secondPoint;
    uint64_t steps = 0;
    uint64_t candidateEvaluations = 0;
};

// Reuses `dilated` as scratch so callers can keep the buffer across
// calls; avoids the per-call slice-sized cv::Mat allocation that NMS
// would otherwise do on every customThinning invocation.
static void nonMaximumSuppression(const cv::Mat& src, cv::Mat& dst,
                                  cv::Mat& dilated, int size)
{
    cv::dilate(src, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size)));
    cv::compare(src, dilated, dst, cv::CMP_EQ);
}

template <bool StorePoints>
static TraceResult tracePath(
    const cv::Point& startPoint,
    const cv::Mat& distTransform,
    cv::Mat& visited,
    cv::Mat* outputImage,
    PointHistory history,
    const cv::Point& seedPoint,
    std::vector<cv::Point>* tracedPoints)
{
    cv::Point currentPoint = startPoint;
    int visitIndex = 0;
    TraceResult result;

    if constexpr (StorePoints) {
        tracedPoints->clear();
    }

    while (true) {
        if (currentPoint != seedPoint) {
            auto* visitedRow = visited.ptr<uint8_t>(currentPoint.y);
            if (outputImage != nullptr) {
                visitedRow[currentPoint.x] = static_cast<uint8_t>(std::min(visitIndex + 1, 255));
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

        float maxDist = -std::numeric_limits<float>::max();
        cv::Point nextPoint(-1, -1);

        for (size_t ni = 0; ni < kNeighborDx.size(); ++ni) {
            const int nx = currentPoint.x + kNeighborDx[ni];
            const int ny = currentPoint.y + kNeighborDy[ni];
            if (nx < 0 || nx >= distTransform.cols || ny < 0 || ny >= distTransform.rows) {
                continue;
            }

            ++result.candidateEvaluations;
            const auto* distRow = distTransform.ptr<uint8_t>(ny);
            const float dist = static_cast<float>(distRow[nx]);

            float penalty = 0.0f;
            if (history.count > 0) {
                const auto& weights = kHistoryWeights[history.count - 1];
                float totalWeightedDistance = 0.0f;
                for (int i = 0; i < history.count; ++i) {
                    const auto& point = history.at(i);
                    const float dx = static_cast<float>(nx - point.x);
                    const float dy = static_cast<float>(ny - point.y);
                    totalWeightedDistance += weights[i] * std::sqrt(dx * dx + dy * dy);
                }
                penalty = totalWeightedDistance / kHistoryWeightSums[history.count - 1];
            }

            const float effectiveDist = dist + penalty;
            if (effectiveDist > maxDist) {
                maxDist = effectiveDist;
                nextPoint = cv::Point(nx, ny);
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

static void customThinningImpl(
    const cv::Mat& inputImage,
    cv::Mat* outputImage,
    std::vector<std::vector<cv::Point>>* traces,
    ThinningStats* stats,
    ThinningScratch& scratch,
    const ThinningPruneParams& pruneParams)
{
    if (inputImage.empty() || inputImage.type() != CV_8UC1) {
        if (outputImage != nullptr) {
            outputImage->release();
        }
        if (traces != nullptr) {
            traces->clear();
        }
        return;
    }

    ThinningStats localStats;

    const auto dtStart = std::chrono::steady_clock::now();
    // Mat::create is a no-op when the existing buffer already matches
    // size+type, so passing scratch.* as out-params keeps the same
    // backing storage across calls — opencv writes into it instead of
    // mmapping a fresh slice-sized buffer per call.
    cv::distanceTransform(inputImage, scratch.distTransform, cv::DIST_L1, cv::DIST_MASK_5, CV_8U);
    CV_Assert(scratch.distTransform.type() == CV_8U);
    localStats.distanceTransformSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dtStart).count();

    const auto seedStart = std::chrono::steady_clock::now();
    nonMaximumSuppression(scratch.distTransform, scratch.localMaxima, scratch.dilated, 3);

    cv::bitwise_and(scratch.localMaxima, inputImage, scratch.seeds);

    cv::findNonZero(scratch.seeds, scratch.seedPoints);
    localStats.seedCount = scratch.seedPoints.size();
    localStats.seedDetectionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seedStart).count();

    cv::Mat visited;
    if (outputImage != nullptr) {
        outputImage->create(inputImage.size(), CV_8UC1);
        outputImage->setTo(0);
        visited = *outputImage;
    } else {
        scratch.visited.create(inputImage.size(), CV_8UC1);
        scratch.visited.setTo(0);
        visited = scratch.visited;
    }

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
                seed, scratch.distTransform, visited, outputImage, initialHistory, seed, &scratch.firstPath);
            localStats.traceSteps += firstResult.steps;
            localStats.candidateEvaluations += firstResult.candidateEvaluations;

            PointHistory secondHistory;
            if (firstResult.hasSecondPoint) {
                secondHistory.push_back(firstResult.secondPoint);
            }
            const auto secondResult = tracePath<true>(
                seed, scratch.distTransform, visited, outputImage, secondHistory, seed, &scratch.secondPath);
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
                seed, scratch.distTransform, visited, outputImage, initialHistory, seed, nullptr);
            localStats.traceSteps += firstResult.steps;
            localStats.candidateEvaluations += firstResult.candidateEvaluations;

            PointHistory secondHistory;
            if (firstResult.hasSecondPoint) {
                secondHistory.push_back(firstResult.secondPoint);
            }
            const auto secondResult = tracePath<false>(
                seed, scratch.distTransform, visited, outputImage, secondHistory, seed, nullptr);
            localStats.traceSteps += secondResult.steps;
            localStats.candidateEvaluations += secondResult.candidateEvaluations;
        }
    }

    localStats.tracePathsSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - traceStart).count();

    if (traces != nullptr) {
        if (pruneParams.enabled) {
            const auto pruneStart = std::chrono::steady_clock::now();
            pruneThinningTraces(*traces, scratch.distTransform, pruneParams, &localStats);
            localStats.pruneSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pruneStart).count();

            if (outputImage != nullptr) {
                // Re-rasterise from kept traces. The visit-index encoding is
                // sacrificed when pruning is enabled in image+trace mode.
                outputImage->setTo(0);
                for (const auto& trace : *traces) {
                    for (const auto& p : trace) {
                        outputImage->at<uint8_t>(p.y, p.x) = 255;
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

} // namespace

void customThinning(const cv::Mat& inputImage,
                    cv::Mat& outputImage,
                    std::vector<std::vector<cv::Point>>* traces)
{
    customThinning(inputImage, outputImage, traces, nullptr);
}

void customThinning(const cv::Mat& inputImage,
                    cv::Mat& outputImage,
                    std::vector<std::vector<cv::Point>>* traces,
                    ThinningStats* stats,
                    const ThinningPruneParams& pruneParams)
{
    ThinningScratch scratch;
    customThinningImpl(inputImage, &outputImage, traces, stats, scratch, pruneParams);
}

void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats,
                             const ThinningPruneParams& pruneParams)
{
    ThinningScratch scratch;
    customThinningImpl(inputImage, nullptr, &traces, stats, scratch, pruneParams);
}

void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats,
                             ThinningScratch& scratch,
                             const ThinningPruneParams& pruneParams)
{
    customThinningImpl(inputImage, nullptr, &traces, stats, scratch, pruneParams);
}

void pruneThinningTraces(std::vector<std::vector<cv::Point>>& traces,
                         const cv::Mat& distTransform,
                         const ThinningPruneParams& params,
                         ThinningStats* stats)
{
    CV_Assert(distTransform.type() == CV_8U);

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
            const int dx = trace[k].x - trace[k - 1].x;
            const int dy = trace[k].y - trace[k - 1].y;
            length += std::sqrt(static_cast<double>(dx * dx + dy * dy));
        }

        const cv::Point& a = trace.front();
        const cv::Point& b = trace.back();
        const double dbfA = static_cast<double>(distTransform.at<uint8_t>(a.y, a.x));
        const double dbfB = static_cast<double>(distTransform.at<uint8_t>(b.y, b.x));
        const double dbfTip = std::min(dbfA, dbfB);

        const double teasarThresh = params.scale * dbfTip + params.constant;
        const double thresh = std::max(teasarThresh, params.minLength);

        if (length < thresh) {
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
