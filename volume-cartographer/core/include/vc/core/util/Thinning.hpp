#pragma once

#include <cstdint>

#include <opencv2/core.hpp>
#include <vector>

struct ThinningStats {
    double distanceTransformSeconds = 0.0;
    double seedDetectionSeconds = 0.0;
    double tracePathsSeconds = 0.0;
    uint64_t seedCount = 0;
    uint64_t traceCount = 0;
    uint64_t traceSteps = 0;
    uint64_t candidateEvaluations = 0;

    void accumulate(const ThinningStats& other) {
        distanceTransformSeconds += other.distanceTransformSeconds;
        seedDetectionSeconds += other.seedDetectionSeconds;
        tracePathsSeconds += other.tracePathsSeconds;
        seedCount += other.seedCount;
        traceCount += other.traceCount;
        traceSteps += other.traceSteps;
        candidateEvaluations += other.candidateEvaluations;
    }
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
                    ThinningStats* stats);
void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats = nullptr);
void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats,
                             ThinningScratch& scratch);
