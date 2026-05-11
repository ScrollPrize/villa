#pragma once

#include <cstdint>

#include <opencv2/core.hpp>
#include <vector>

struct ThinningStats {
    double distanceTransformSeconds = 0.0;
    double seedDetectionSeconds = 0.0;
    double tracePathsSeconds = 0.0;
    double pruneSeconds = 0.0;
    uint64_t seedCount = 0;
    uint64_t traceCount = 0;        // raw traces produced before pruning
    uint64_t traceSteps = 0;
    uint64_t candidateEvaluations = 0;
    uint64_t tracesPruned = 0;      // traces dropped by pruning pass
    uint64_t tracesKept = 0;        // traces retained after pruning

    void accumulate(const ThinningStats& other) {
        distanceTransformSeconds += other.distanceTransformSeconds;
        seedDetectionSeconds += other.seedDetectionSeconds;
        tracePathsSeconds += other.tracePathsSeconds;
        pruneSeconds += other.pruneSeconds;
        seedCount += other.seedCount;
        traceCount += other.traceCount;
        traceSteps += other.traceSteps;
        candidateEvaluations += other.candidateEvaluations;
        tracesPruned += other.tracesPruned;
        tracesKept += other.tracesKept;
    }
};

// TEASAR-style branch pruning. A trace is dropped when
//   length < max(scale * DBF(tip) + constant, minLength)
// where length is the polyline length (sum of 8-connected step distances),
// DBF is the distance-from-boundary value at the trace endpoint (the thinner
// endpoint is used), and minLength is an unconditional floor. When `enabled`
// is false the pruning pass is skipped entirely.
struct ThinningPruneParams {
    bool enabled = false;
    double scale = 0.0;
    double constant = 0.0;
    double minLength = 0.0;
};

// Reusable per-thread scratch for customThinning. Letting one thread
// keep these across calls avoids ~5 slice-sized cv::Mat allocations
// per invocation; the kernel page-fault tax of the freshly-mmapped
// output buffers was the dominant non-CPU cost in profiles.
struct ThinningScratch {
    cv::Mat distTransform;
    cv::Mat dilated;
    cv::Mat localMaxima;
    cv::Mat seeds;
    cv::Mat visited;
    std::vector<cv::Point> seedPoints;
    std::vector<cv::Point> firstPath;
    std::vector<cv::Point> secondPath;
};

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<std::vector<cv::Point>>* traces = nullptr);
void customThinning(const cv::Mat& inputImage,
                    cv::Mat& outputImage,
                    std::vector<std::vector<cv::Point>>* traces,
                    ThinningStats* stats,
                    const ThinningPruneParams& pruneParams = ThinningPruneParams{});
void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats = nullptr,
                             const ThinningPruneParams& pruneParams = ThinningPruneParams{});
void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats,
                             ThinningScratch& scratch,
                             const ThinningPruneParams& pruneParams = ThinningPruneParams{});

// Drop traces shorter than max(scale*DBF(tip) + constant, minLength).
// Runs unconditionally — caller decides whether to invoke. distTransform
// must be the uint8 L1 distance field used by customThinning.
void pruneThinningTraces(std::vector<std::vector<cv::Point>>& traces,
                         const cv::Mat& distTransform,
                         const ThinningPruneParams& params,
                         ThinningStats* stats = nullptr);
