#include "vc/fiber_tracer/FiberSkeletonFields.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    const double length = cv::norm(value);
    if (!std::isfinite(length) || !(length > kEpsilon))
        return {0.0, 0.0, 0.0};
    return value / length;
}

}  // namespace

FiberSkeletonFieldSample interpolateFiberSkeletonBlock(
    const FiberSkeletonCornerBlock& block, const cv::Vec3d& pointBaseXYZ)
{
    if (!(block.spacingBase > 0.0) || !std::isfinite(block.spacingBase))
        throw std::invalid_argument("corner block spacing must be positive");

    // The cell is fixed for the evaluation; clamping keeps a query that has
    // drifted a fraction of a voxel outside it on the same trilinear form.
    const cv::Vec3d local =
        (pointBaseXYZ - block.cellOriginBaseXYZ) / block.spacingBase;
    const double tx = std::clamp(local[0], 0.0, 1.0);
    const double ty = std::clamp(local[1], 0.0, 1.0);
    const double tz = std::clamp(local[2], 0.0, 1.0);

    FiberSkeletonFieldSample sample;
    cv::Vec3d direction{0.0, 0.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 0.0};
    for (std::size_t corner = 0; corner < 8; ++corner) {
        const double wx = (corner & 1U) != 0 ? tx : 1.0 - tx;
        const double wy = (corner & 2U) != 0 ? ty : 1.0 - ty;
        const double wz = (corner & 4U) != 0 ? tz : 1.0 - tz;
        const double weight = wx * wy * wz;
        sample.presence += weight * block.presence[corner];
        direction += weight * block.direction[corner];
        normal += weight * block.normal[corner];
        if (block.valid[corner] != 0)
            sample.valid = true;
        if (block.normalValid[corner] != 0)
            sample.normalValid = true;
    }
    sample.direction = normalizedOrZero(direction);
    sample.normal = normalizedOrZero(normal);
    if (sample.direction.dot(sample.direction) <= 0.0)
        sample.valid = false;
    if (sample.normal.dot(sample.normal) <= 0.0)
        sample.normalValid = false;
    sample.presence = std::clamp(sample.presence, 0.0, 1.0);
    return sample;
}

void sampleFiberSkeletonFields(
    FiberSkeletonFieldProvider& provider,
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const std::vector<cv::Vec3d>& referenceDirections,
    std::vector<FiberSkeletonFieldSample>& samples)
{
    if (pointsBaseXYZ.size() != referenceDirections.size())
        throw std::invalid_argument(
            "skeleton field points and reference directions size mismatch");
    std::vector<FiberSkeletonCornerBlock> blocks;
    provider.loadCorners(pointsBaseXYZ, referenceDirections, blocks);
    if (blocks.size() != pointsBaseXYZ.size())
        throw std::runtime_error("skeleton field provider returned wrong size");
    samples.clear();
    samples.reserve(pointsBaseXYZ.size());
    for (std::size_t index = 0; index < blocks.size(); ++index)
        samples.push_back(
            interpolateFiberSkeletonBlock(blocks[index], pointsBaseXYZ[index]));
}

}  // namespace vc::fiber_tracer
