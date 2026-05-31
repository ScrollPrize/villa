#pragma once

#include <opencv2/core/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vc::lasagna {

struct NormalSample {
    cv::Vec3d normal{0.0, 0.0, 0.0};
    bool valid = false;
    std::string reason;
};

struct NormalSampleWithDerivative {
    NormalSample sample;
    cv::Matx33d dNormalDVolume = cv::Matx33d::zeros();
    bool hasDerivative = false;
};

struct NormalPrefetchReport {
    uint64_t requestedChunks = 0;
    uint64_t chunksRead = 0;
};

class NormalSampler {
public:
    virtual ~NormalSampler() = default;
    [[nodiscard]] virtual NormalSample sampleNormal(const cv::Vec3d& volumePoint) const = 0;
    [[nodiscard]] virtual NormalSampleWithDerivative sampleNormalWithDerivative(
        const cv::Vec3d& volumePoint) const
    {
        return {sampleNormal(volumePoint), cv::Matx33d::zeros(), false};
    }
    [[nodiscard]] virtual NormalPrefetchReport prefetchNormalSamples(
        const std::vector<cv::Vec3d>& /*volumePoints*/,
        bool /*withDerivative*/) const
    {
        return {};
    }
};

struct LinePoint {
    cv::Vec3d position{0.0, 0.0, 0.0};
    NormalSample sampledNormal;
    bool valid = true;
};

struct SegmentNormalSample {
    double t = 0.0;
    cv::Vec3d position{0.0, 0.0, 0.0};
    NormalSample sampledNormal;
};

struct LineSegmentSamples {
    std::vector<SegmentNormalSample> samples;
};

struct LineModel {
    std::vector<LinePoint> points;
    std::vector<LineSegmentSamples> segmentSamples;
    // Preferred optimized control-point index for generated cut-plane roll.
    // Negative values fall back to the middle seed index.
    int displayFrameAnchorIndex = -1;
};

struct LineOptimizationConfig {
    enum class TangentGuideMode {
        None,
        ProjectVectorOntoTangentPlane,
        CrossVectorWithNormal,
    };

    enum class LinearSolver {
        DenseQR,
        DenseNormalCholesky,
        SparseNormalCholesky,
        DenseSchur,
        SparseSchur,
        IterativeSchur,
        CGNR,
    };

    int segmentsPerSide = 100;
    double segmentLength = 16.0;
    double straightnessWeight = 1.0;
    double tangentStraightnessWeight = -1.0;
    double normalStraightnessWeight = -1.0;
    double normalAlignmentWeight = 1.0;
    double distanceWeight = 1.0;
    bool useInitialTangent = false;
    cv::Vec3d initialTangent{0.0, 0.0, 0.0};
    double initialTangentWeight = 1.0;
    TangentGuideMode tangentGuideMode = TangentGuideMode::None;
    cv::Vec3d tangentGuideVector{0.0, 0.0, 0.0};
    double tangentGuideWeight = 1.0;
    // Number of equal intervals evaluated per segment. A value of 1 stores
    // only segment endpoints, with no interior sub-samples.
    int samplesPerSegment = 1;
    int maxIterations = 50;
    bool differentiableNormalSampling = false;
    bool runGlobalOptimization = true;
    LinearSolver linearSolver = LinearSolver::SparseNormalCholesky;
    int numThreads = 1;
    bool printSolverProgress = true;
};

} // namespace vc::lasagna
