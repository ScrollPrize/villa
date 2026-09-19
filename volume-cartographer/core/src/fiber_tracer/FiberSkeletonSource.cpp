#include "vc/fiber_tracer/FiberSkeletonSource.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;
// The scope mask is evaluated on a coarser lattice than the crop; the radius
// is widened by the cell diagonal so it never cuts inside the exact tube.
constexpr std::int64_t kScopeCell = 8;

double segmentDistance(
    const cv::Vec3d& point, const cv::Vec3d& start, const cv::Vec3d& end)
{
    const cv::Vec3d delta = end - start;
    const double lengthSquared = delta.dot(delta);
    const double t = lengthSquared > kEpsilon
        ? std::clamp((point - start).dot(delta) / lengthSquared, 0.0, 1.0)
        : 0.0;
    return cv::norm(point - (start + delta * t));
}

// Uniform grid over polyline segments. A segment is inserted into every cell
// its radius-expanded bounding box touches, so the cell containing a query
// point holds every segment that can be within the radius of it.
class SegmentIndex
{
public:
    SegmentIndex(
        const std::vector<std::vector<cv::Vec3d>>& polylines, double radius)
        : cell_(std::max(radius, 1.0))
    {
        for (const auto& polyline : polylines) {
            for (std::size_t index = 1; index < polyline.size(); ++index) {
                const cv::Vec3d start = polyline[index - 1];
                const cv::Vec3d end = polyline[index];
                segments_.push_back({start, end});
                const std::size_t id = segments_.size() - 1;
                std::array<std::int64_t, 3> low{};
                std::array<std::int64_t, 3> high{};
                for (int axis = 0; axis < 3; ++axis) {
                    low[axis] = cellOf(
                        std::min(start[axis], end[axis]) - radius);
                    high[axis] = cellOf(
                        std::max(start[axis], end[axis]) + radius);
                }
                for (std::int64_t x = low[0]; x <= high[0]; ++x) {
                    for (std::int64_t y = low[1]; y <= high[1]; ++y) {
                        for (std::int64_t z = low[2]; z <= high[2]; ++z)
                            cells_[{x, y, z}].push_back(id);
                    }
                }
            }
        }
    }

    [[nodiscard]] bool within(const cv::Vec3d& point, double radius) const
    {
        const auto found = cells_.find(
            {cellOf(point[0]), cellOf(point[1]), cellOf(point[2])});
        if (found == cells_.end())
            return false;
        for (const auto id : found->second) {
            if (segmentDistance(point, segments_[id].first,
                                segments_[id].second) <= radius) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool empty() const noexcept { return segments_.empty(); }

private:
    [[nodiscard]] std::int64_t cellOf(double value) const
    {
        return static_cast<std::int64_t>(std::floor(value / cell_));
    }

    double cell_;
    std::vector<std::pair<cv::Vec3d, cv::Vec3d>> segments_;
    std::map<std::array<std::int64_t, 3>, std::vector<std::size_t>> cells_;
};

struct ScopeMask {
    std::array<std::size_t, 3> shapeZYX{0, 0, 0};
    std::vector<std::uint8_t> values;

    [[nodiscard]] bool inside(
        std::size_t z, std::size_t y, std::size_t x) const noexcept
    {
        const std::size_t cz = z / static_cast<std::size_t>(kScopeCell);
        const std::size_t cy = y / static_cast<std::size_t>(kScopeCell);
        const std::size_t cx = x / static_cast<std::size_t>(kScopeCell);
        if (cz >= shapeZYX[0] || cy >= shapeZYX[1] || cx >= shapeZYX[2])
            return false;
        return values[(cz * shapeZYX[1] + cy) * shapeZYX[2] + cx] != 0;
    }
};

ScopeMask buildScopeMask(
    const FiberSkeletonCrop& crop, const SegmentIndex& index, double radius)
{
    ScopeMask mask;
    for (int axis = 0; axis < 3; ++axis) {
        mask.shapeZYX[axis] =
            (crop.shapeZYX[axis] + static_cast<std::size_t>(kScopeCell) - 1) /
            static_cast<std::size_t>(kScopeCell);
    }
    mask.values.assign(
        mask.shapeZYX[0] * mask.shapeZYX[1] * mask.shapeZYX[2], 0);
    const double cellDiagonal = std::sqrt(3.0) *
        static_cast<double>(kScopeCell) * crop.predictionToBaseScale;
    const double reach = radius + cellDiagonal;
    for (std::size_t cz = 0; cz < mask.shapeZYX[0]; ++cz) {
        for (std::size_t cy = 0; cy < mask.shapeZYX[1]; ++cy) {
            for (std::size_t cx = 0; cx < mask.shapeZYX[2]; ++cx) {
                const std::array<std::size_t, 3> center{
                    cz * static_cast<std::size_t>(kScopeCell) +
                        static_cast<std::size_t>(kScopeCell) / 2,
                    cy * static_cast<std::size_t>(kScopeCell) +
                        static_cast<std::size_t>(kScopeCell) / 2,
                    cx * static_cast<std::size_t>(kScopeCell) +
                        static_cast<std::size_t>(kScopeCell) / 2};
                if (index.within(crop.baseXYZ(center), reach)) {
                    mask.values[(cz * mask.shapeZYX[1] + cy) *
                                    mask.shapeZYX[2] + cx] = 1;
                }
            }
        }
    }
    return mask;
}

}  // namespace

FiberSkeletonCrop buildFiberSkeletonCrop(
    const FiberPredictionField& predictions,
    const std::vector<std::vector<cv::Vec3d>>& tubePolylinesBaseXYZ,
    double tubeRadiusBase, double cropMarginBase, int parallelThreads)
{
    if (!(tubeRadiusBase > 0.0) || !std::isfinite(tubeRadiusBase))
        throw std::invalid_argument("tube radius must be finite and positive");
    if (!(cropMarginBase >= 0.0) || !std::isfinite(cropMarginBase))
        throw std::invalid_argument("crop margin must be finite and non-negative");
    std::size_t vertexCount = 0;
    for (const auto& polyline : tubePolylinesBaseXYZ)
        vertexCount += polyline.size();
    if (vertexCount < 2)
        throw std::invalid_argument("skeleton crop needs a tube polyline");

    const auto grid = predictions.storedGridInfo();
    const double scale = grid.predictionToBaseScale;
    if (!(scale > 0.0) || !std::isfinite(scale))
        throw std::invalid_argument("stored prediction scale is invalid");

    cv::Vec3d minimum{std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
    cv::Vec3d maximum = -minimum;
    for (const auto& polyline : tubePolylinesBaseXYZ) {
        for (const auto& point : polyline) {
            for (int axis = 0; axis < 3; ++axis) {
                minimum[axis] = std::min(minimum[axis], point[axis]);
                maximum[axis] = std::max(maximum[axis], point[axis]);
            }
        }
    }
    const double padding = tubeRadiusBase + cropMarginBase;

    FiberSkeletonCrop crop;
    crop.predictionToBaseScale = scale;
    for (int axis = 0; axis < 3; ++axis) {
        // Crop axes are Z, Y, X while base points are X, Y, Z.
        const int base = 2 - axis;
        const auto low = static_cast<std::int64_t>(
            std::floor((minimum[base] - padding) / scale));
        const auto high = static_cast<std::int64_t>(
            std::ceil((maximum[base] + padding) / scale));
        const auto limit = static_cast<std::int64_t>(grid.shapeZYX[axis]);
        const std::int64_t begin = std::clamp<std::int64_t>(low, 0, limit);
        const std::int64_t end = std::clamp<std::int64_t>(high + 1, 0, limit);
        crop.originZYX[axis] = begin;
        crop.shapeZYX[axis] = static_cast<std::size_t>(std::max<std::int64_t>(
            0, end - begin));
    }
    if (crop.voxelCount() == 0)
        throw std::runtime_error("skeleton crop is empty");
    crop.presence.assign(crop.voxelCount(), 0);

    const SegmentIndex index(tubePolylinesBaseXYZ, tubeRadiusBase);
    const auto scope = buildScopeMask(crop, index, tubeRadiusBase);

    // Presence is fetched only inside the scope, in Z-major batches so the
    // stored chunk cache sees a locality-friendly request order.
    constexpr std::size_t kBatch = 1u << 21;
    std::vector<std::array<std::size_t, 3>> indices;
    std::vector<std::size_t> targets;
    std::vector<FiberStoredPresenceSample> samples;
    indices.reserve(kBatch);
    targets.reserve(kBatch);
    const auto flush = [&]() {
        if (indices.empty())
            return;
        predictions.sampleStoredPresenceBatch(
            indices, parallelThreads, samples);
        if (samples.size() != indices.size())
            throw std::runtime_error("stored presence batch size mismatch");
        crop.scopedVoxels += samples.size();
        for (std::size_t item = 0; item < samples.size(); ++item) {
            if (!samples[item].valid) {
                ++crop.missingVoxels;
                continue;
            }
            const double value =
                std::clamp(samples[item].presence, 0.0, 1.0) * 255.0;
            crop.presence[targets[item]] =
                static_cast<std::uint8_t>(std::lround(value));
        }
        indices.clear();
        targets.clear();
    };

    const auto cellSide = static_cast<std::size_t>(kScopeCell);
    for (std::size_t z = 0; z < crop.shapeZYX[0]; ++z) {
        for (std::size_t y = 0; y < crop.shapeZYX[1]; ++y) {
            // Whole scope rows are usually empty; skipping them keeps the
            // full-box scan proportional to the scope, not to the box.
            const std::size_t row =
                ((z / cellSide) * scope.shapeZYX[1] + (y / cellSide)) *
                scope.shapeZYX[2];
            bool any = false;
            for (std::size_t cx = 0; cx < scope.shapeZYX[2] && !any; ++cx)
                any = scope.values[row + cx] != 0;
            if (!any)
                continue;
            for (std::size_t x = 0; x < crop.shapeZYX[2]; ++x) {
                if (scope.values[row + x / cellSide] == 0)
                    continue;
                indices.push_back(
                    {static_cast<std::size_t>(crop.originZYX[0]) + z,
                     static_cast<std::size_t>(crop.originZYX[1]) + y,
                     static_cast<std::size_t>(crop.originZYX[2]) + x});
                targets.push_back(crop.flatIndex(z, y, x));
                if (indices.size() >= kBatch)
                    flush();
            }
        }
    }
    flush();
    if (crop.scopedVoxels == 0 || crop.missingVoxels == crop.scopedVoxels) {
        throw std::runtime_error(
            "skeleton crop has no published prediction coverage; the stored "
            "prediction cache must be complete for the scoped region");
    }
    return crop;
}

class FiberSkeletonStoredFields::Impl
{
public:
    Impl(
        const FiberPredictionField& predictions,
        const vc::lasagna::NormalSampler& normalSampler,
        double normalWorkingToBaseScale, FiberSkeletonCrop crop,
        int parallelThreads)
        : predictions_(predictions)
        , normalSampler_(normalSampler)
        , normalScale_(normalWorkingToBaseScale)
        , crop_(std::move(crop))
        , threads_(std::max(1, parallelThreads))
    {
        if (!(normalScale_ > 0.0) || !std::isfinite(normalScale_))
            throw std::invalid_argument("normal working scale is invalid");
        if (crop_.presence.size() != crop_.voxelCount())
            throw std::invalid_argument("skeleton crop presence size mismatch");
    }

    // Grid-aligned sample of one crop voxel, memoized for the provider's life.
    const FiberSkeletonFieldSample& voxel(std::size_t flat)
    {
        const auto found = cache_.find(flat);
        if (found != cache_.end())
            return found->second;
        pending_.assign(1, flat);
        fetch(pending_);
        return cache_.at(flat);
    }

    void fetch(const std::vector<std::size_t>& flats)
    {
        std::vector<std::array<std::size_t, 3>> indices;
        std::vector<std::size_t> targets;
        std::vector<cv::Vec3d> points;
        for (const auto flat : flats) {
            if (cache_.count(flat) != 0)
                continue;
            if (flat >= crop_.voxelCount())
                continue;
            const auto zyx = crop_.voxelZYX(flat);
            indices.push_back(
                {static_cast<std::size_t>(crop_.originZYX[0]) + zyx[0],
                 static_cast<std::size_t>(crop_.originZYX[1]) + zyx[1],
                 static_cast<std::size_t>(crop_.originZYX[2]) + zyx[2]});
            targets.push_back(flat);
            points.push_back(crop_.baseXYZ(zyx) * (1.0 / normalScale_));
        }
        if (indices.empty())
            return;
        std::vector<FiberStoredPredictionSample> stored;
        predictions_.sampleStoredGridBatch(indices, threads_, stored);
        if (stored.size() != indices.size())
            throw std::runtime_error("stored prediction batch size mismatch");
        for (std::size_t item = 0; item < targets.size(); ++item) {
            FiberSkeletonFieldSample sample;
            sample.valid = stored[item].valid;
            sample.presence = std::clamp(stored[item].presence, 0.0, 1.0);
            const double length = cv::norm(stored[item].direction);
            if (sample.valid && std::isfinite(length) && length > kEpsilon)
                sample.direction = stored[item].direction / length;
            else
                sample.valid = false;
            const auto normal = normalSampler_.sampleNormal(points[item]);
            const double normalLength = cv::norm(normal.normal);
            if (normal.valid && std::isfinite(normalLength) &&
                normalLength > kEpsilon) {
                sample.normal = normal.normal / normalLength;
                sample.normalValid = true;
            }
            cache_.emplace(targets[item], sample);
        }
    }

    [[nodiscard]] const FiberSkeletonCrop& crop() const noexcept { return crop_; }
    [[nodiscard]] std::size_t cached() const noexcept { return cache_.size(); }

private:
    const FiberPredictionField& predictions_;
    const vc::lasagna::NormalSampler& normalSampler_;
    double normalScale_ = 1.0;
    FiberSkeletonCrop crop_;
    int threads_ = 1;
    std::unordered_map<std::size_t, FiberSkeletonFieldSample> cache_;
    std::vector<std::size_t> pending_;
};

FiberSkeletonStoredFields::FiberSkeletonStoredFields(
    const FiberPredictionField& predictions,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, FiberSkeletonCrop crop,
    int parallelThreads)
    : impl_(std::make_unique<Impl>(
          predictions, normalSampler, normalWorkingToBaseScale, std::move(crop),
          parallelThreads))
{
}

FiberSkeletonStoredFields::~FiberSkeletonStoredFields() = default;

void FiberSkeletonStoredFields::loadCorners(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const std::vector<cv::Vec3d>& referenceDirections,
    std::vector<FiberSkeletonCornerBlock>& blocks)
{
    if (pointsBaseXYZ.size() != referenceDirections.size())
        throw std::invalid_argument(
            "skeleton corner points and reference directions size mismatch");
    const auto& crop = impl_->crop();
    const double scale = crop.predictionToBaseScale;

    // Collect every corner of every query first, so one batch covers them all.
    std::vector<std::array<std::int64_t, 3>> cornerZYX(
        pointsBaseXYZ.size() * 8);
    std::vector<std::size_t> request;
    request.reserve(pointsBaseXYZ.size() * 8);
    for (std::size_t point = 0; point < pointsBaseXYZ.size(); ++point) {
        std::array<std::int64_t, 3> low{};
        for (int axis = 0; axis < 3; ++axis) {
            const double local =
                pointsBaseXYZ[point][2 - axis] / scale - crop.originZYX[axis];
            low[axis] = static_cast<std::int64_t>(std::floor(local));
        }
        for (std::size_t corner = 0; corner < 8; ++corner) {
            const std::array<std::int64_t, 3> zyx{
                low[0] + ((corner & 4U) != 0 ? 1 : 0),
                low[1] + ((corner & 2U) != 0 ? 1 : 0),
                low[2] + ((corner & 1U) != 0 ? 1 : 0)};
            cornerZYX[point * 8 + corner] = zyx;
            if (zyx[0] < 0 || zyx[1] < 0 || zyx[2] < 0 ||
                zyx[0] >= static_cast<std::int64_t>(crop.shapeZYX[0]) ||
                zyx[1] >= static_cast<std::int64_t>(crop.shapeZYX[1]) ||
                zyx[2] >= static_cast<std::int64_t>(crop.shapeZYX[2])) {
                continue;
            }
            request.push_back(crop.flatIndex(
                static_cast<std::size_t>(zyx[0]),
                static_cast<std::size_t>(zyx[1]),
                static_cast<std::size_t>(zyx[2])));
        }
    }
    impl_->fetch(request);

    blocks.assign(pointsBaseXYZ.size(), {});
    for (std::size_t point = 0; point < pointsBaseXYZ.size(); ++point) {
        auto& block = blocks[point];
        block.spacingBase = scale;
        block.referenceDirection = referenceDirections[point];
        const auto& origin = cornerZYX[point * 8];
        block.cellOriginBaseXYZ = {
            (static_cast<double>(origin[2]) +
             static_cast<double>(crop.originZYX[2])) * scale,
            (static_cast<double>(origin[1]) +
             static_cast<double>(crop.originZYX[1])) * scale,
            (static_cast<double>(origin[0]) +
             static_cast<double>(crop.originZYX[0])) * scale};
        block.inside = true;
        for (std::size_t corner = 0; corner < 8; ++corner) {
            const auto& zyx = cornerZYX[point * 8 + corner];
            if (zyx[0] < 0 || zyx[1] < 0 || zyx[2] < 0 ||
                zyx[0] >= static_cast<std::int64_t>(crop.shapeZYX[0]) ||
                zyx[1] >= static_cast<std::int64_t>(crop.shapeZYX[1]) ||
                zyx[2] >= static_cast<std::int64_t>(crop.shapeZYX[2])) {
                block.inside = false;
                continue;
            }
            const std::size_t flat = crop.flatIndex(
                static_cast<std::size_t>(zyx[0]),
                static_cast<std::size_t>(zyx[1]),
                static_cast<std::size_t>(zyx[2]));
            const auto& sample = impl_->voxel(flat);
            if (sample.valid) {
                block.presence[corner] = sample.presence;
                // The predicted direction is sign ambiguous; fixing the sign
                // here keeps it constant through the solver evaluation.
                block.direction[corner] =
                    sample.direction.dot(referenceDirections[point]) < 0.0
                    ? -sample.direction
                    : sample.direction;
                block.valid[corner] = 1;
            }
            if (sample.normalValid) {
                block.normal[corner] = sample.normal;
                block.normalValid[corner] = 1;
            }
        }
        // Keep the interpolated normal from cancelling: normals are sign
        // ambiguous too, so align every corner with the first valid one.
        cv::Vec3d anchor{0.0, 0.0, 0.0};
        for (std::size_t corner = 0; corner < 8; ++corner) {
            if (block.normalValid[corner] == 0)
                continue;
            if (anchor.dot(anchor) <= 0.0) {
                anchor = block.normal[corner];
                continue;
            }
            if (block.normal[corner].dot(anchor) < 0.0)
                block.normal[corner] = -block.normal[corner];
        }
    }
}

void FiberSkeletonStoredFields::loadVoxels(
    const std::vector<std::size_t>& flatCropIndices,
    std::vector<FiberSkeletonFieldSample>& samples)
{
    impl_->fetch(flatCropIndices);
    samples.clear();
    samples.reserve(flatCropIndices.size());
    for (const auto flat : flatCropIndices) {
        if (flat >= impl_->crop().voxelCount()) {
            samples.emplace_back();
            continue;
        }
        samples.push_back(impl_->voxel(flat));
    }
}

const FiberSkeletonCrop& FiberSkeletonStoredFields::crop() const noexcept
{
    return impl_->crop();
}

double FiberSkeletonStoredFields::predictionToBaseScale() const noexcept
{
    return impl_->crop().predictionToBaseScale;
}

std::size_t FiberSkeletonStoredFields::cachedVoxels() const noexcept
{
    return impl_->cached();
}

}  // namespace vc::fiber_tracer
