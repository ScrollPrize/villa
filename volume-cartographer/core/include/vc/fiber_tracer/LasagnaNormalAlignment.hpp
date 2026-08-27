#pragma once

#include "vc/fiber_tracer/BinaryBeliefPropagation.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct LasagnaNormalLattice {
    std::array<std::int64_t, 3> beginXYZ{};
    std::array<std::size_t, 3> shapeXYZ{};
    double spacingBaseVoxels = 0.0;
    std::vector<cv::Vec3f> positionsBaseXYZ;
};

[[nodiscard]] LasagnaNormalLattice makeLasagnaNormalLattice(const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ, double spacingBaseVoxels);

[[nodiscard]] std::vector<BinaryPairwiseFactor> makeLasagnaNormalLatticeFactors(
    const LasagnaNormalLattice& lattice, std::span<const std::size_t> nodeByLatticeSample, std::span<const cv::Vec3f> retainedNormals, int neighborRadius);

struct LasagnaNormalAlignmentConfig {
    BinaryBeliefPropagationConfig beliefPropagation;
};

struct LasagnaNormalAlignmentReport {
    std::vector<cv::Vec3f> alignedNormals;
    std::vector<double> flipProbability;
    std::vector<BinaryBeliefState> fixedStates;
    std::vector<std::size_t> componentByNode;
    std::size_t connectedComponents = 0;
    std::size_t isolatedSamples = 0;
    std::size_t flippedSamples = 0;
    BinaryBeliefPropagationReport beliefPropagation;
};

struct AlignedLasagnaNormalSample {
    cv::Vec3f normal{0.0F, 0.0F, 0.0F};
    std::size_t component = 0;
    std::size_t node = 0;
};

struct LasagnaNormalAlignmentField {
    LasagnaNormalLattice lattice;
    std::vector<std::size_t> nodeByLatticeSample;
    std::vector<cv::Vec3f> positionsBaseXYZ;
    std::vector<cv::Vec3f> rawNormals;
    LasagnaNormalAlignmentReport alignment;
    std::size_t candidateSamples = 0;
    double prefetchMilliseconds = 0.0;
    double materializeMilliseconds = 0.0;

    [[nodiscard]] std::optional<AlignedLasagnaNormalSample> nearest(
        const cv::Vec3d& pointBaseXYZ) const;
};

[[nodiscard]] std::optional<BinaryPairwiseFactor> makeLasagnaNormalAlignmentFactor(std::size_t a, std::size_t b, const cv::Vec3f& normalA, const cv::Vec3f& normalB);

[[nodiscard]] LasagnaNormalAlignmentReport alignLasagnaNormalSamples(
    std::span<const cv::Vec3f> normals, std::span<const BinaryPairwiseFactor> neighborhoodFactors, const LasagnaNormalAlignmentConfig& config = {});

[[nodiscard]] LasagnaNormalAlignmentField sampleAndAlignLasagnaNormalLattice(
    const vc::lasagna::LasagnaNormalSampler& sampler,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ,
    double spacingBaseVoxels,
    int neighborRadius,
    int parallelThreads,
    const LasagnaNormalAlignmentConfig& config = {});

struct NormalGlyphObjConfig {
    double baseRadius = 1.0;
    double directionLength = 4.0;
};

void writeNormalGlyphObj(
    const std::filesystem::path& path, std::span<const cv::Vec3f> positionsBaseXYZ, std::span<const cv::Vec3f> normals, const NormalGlyphObjConfig& config = {});

}  // namespace vc::fiber_tracer
