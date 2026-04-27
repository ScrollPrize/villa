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

static void nonMaximumSuppression(const cv::Mat& src, cv::Mat& dst, int size)
{
    cv::Mat dilated;
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

    const uint64_t maxSteps = static_cast<uint64_t>(distTransform.cols + distTransform.rows);

    while (true) {
        if (result.steps >= maxSteps) {
            break;
        }
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
    ThinningStats* stats)
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
    cv::Mat distTransform;
    cv::distanceTransform(inputImage, distTransform, cv::DIST_L1, cv::DIST_MASK_5, CV_8U);
    CV_Assert(distTransform.type() == CV_8U);
    localStats.distanceTransformSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - dtStart).count();

    const auto seedStart = std::chrono::steady_clock::now();
    cv::Mat localMaxima;
    nonMaximumSuppression(distTransform, localMaxima, 3);

    cv::Mat seeds;
    cv::bitwise_and(localMaxima, inputImage, seeds);

    std::vector<cv::Point> seedPoints;
    cv::findNonZero(seeds, seedPoints);
    localStats.seedCount = seedPoints.size();
    localStats.seedDetectionSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seedStart).count();

    cv::Mat visited;
    if (outputImage != nullptr) {
        *outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);
        visited = *outputImage;
    } else {
        visited = cv::Mat::zeros(inputImage.size(), CV_8UC1);
    }

    if (traces != nullptr) {
        traces->clear();
        traces->reserve(seedPoints.size());
    }

    const auto traceStart = std::chrono::steady_clock::now();
    std::vector<cv::Point> firstPath;
    std::vector<cv::Point> secondPath;

    for (const auto& seed : seedPoints) {
        auto* visitedRow = visited.ptr<uint8_t>(seed.y);
        if (visitedRow[seed.x] != 0) {
            continue;
        }
        visitedRow[seed.x] = 1;

        PointHistory initialHistory;

        if (traces != nullptr) {
            const auto firstResult = tracePath<true>(
                seed, distTransform, visited, outputImage, initialHistory, seed, &firstPath);
            localStats.traceSteps += firstResult.steps;
            localStats.candidateEvaluations += firstResult.candidateEvaluations;

            PointHistory secondHistory;
            if (firstResult.hasSecondPoint) {
                secondHistory.push_back(firstResult.secondPoint);
            }
            const auto secondResult = tracePath<true>(
                seed, distTransform, visited, outputImage, secondHistory, seed, &secondPath);
            localStats.traceSteps += secondResult.steps;
            localStats.candidateEvaluations += secondResult.candidateEvaluations;

            std::vector<cv::Point> fullTrace;
            fullTrace.reserve(secondPath.size() + firstPath.size() - (firstPath.empty() ? 0 : 1));
            fullTrace.insert(fullTrace.end(), secondPath.rbegin(), secondPath.rend());
            if (!firstPath.empty()) {
                fullTrace.insert(fullTrace.end(), firstPath.begin() + 1, firstPath.end());
            }
            if (!fullTrace.empty()) {
                traces->push_back(std::move(fullTrace));
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
                    ThinningStats* stats)
{
    customThinningImpl(inputImage, &outputImage, traces, stats);
}

void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats)
{
    customThinningImpl(inputImage, nullptr, &traces, stats);
}

