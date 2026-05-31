#pragma once

#include <opencv2/core/types.hpp>

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

class NormalSampler {
public:
    virtual ~NormalSampler() = default;
    [[nodiscard]] virtual NormalSample sampleNormal(const cv::Vec3d& volumePoint) const = 0;
    [[nodiscard]] virtual NormalSampleWithDerivative sampleNormalWithDerivative(
        const cv::Vec3d& volumePoint) const
    {
        return {sampleNormal(volumePoint), cv::Matx33d::zeros(), false};
    }
    virtual void prefetchNormalSamples(const std::vector<cv::Vec3d>& /*volumePoints*/,
                                       bool /*withDerivative*/) const
    {
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

    int segmentsPerSide = 10;
    double segmentLength = 50.0;
    double straightnessWeight = 1.0;
    double normalAlignmentWeight = 1.0;
    double distanceWeight = 1.0;
    bool useInitialTangent = false;
    cv::Vec3d initialTangent{0.0, 0.0, 0.0};
    double initialTangentWeight = 1.0;
    TangentGuideMode tangentGuideMode = TangentGuideMode::None;
    cv::Vec3d tangentGuideVector{0.0, 0.0, 0.0};
    double tangentGuideWeight = 1.0;
    // Number of equal intervals evaluated per segment. A value of 4 stores
    // endpoints plus 3 intermediate samples, for 5 samples per segment.
    int samplesPerSegment = 4;
    int maxIterations = 50;
    bool differentiableNormalSampling = false;
    LinearSolver linearSolver = LinearSolver::SparseNormalCholesky;
    int numThreads = 1;
    bool printSolverProgress = true;
};

} // namespace vc::lasagna
