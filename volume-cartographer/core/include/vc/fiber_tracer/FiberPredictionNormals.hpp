#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <span>
#include <vector>

// A fiber's normal from the fiber prediction close to it: of the prediction
// samples around a point of the traced fiber, those whose predicted fiber
// direction runs along the traced fiber state the sheet normal. Each sample
// is weighted by an ellipse around the fiber point: its offset is measured
// along a normal (height) and across it (width). No current normal estimate
// of the fiber enters.
namespace vc::fiber_tracer
{

// One prediction sample near a fiber point. A zero direction or normal marks
// a sample without that prediction.
struct FiberPredictionNormalSample {
    cv::Vec3d offset{0., 0., 0.};     // sample point - fiber point, base voxels
    cv::Vec3d direction{0., 0., 0.};  // predicted fiber direction (a line), unit
    cv::Vec3d normal{0., 0., 0.};     // predicted sheet normal (a line), unit
    double presence = 0.;             // fiber presence, raw channel value
};

// height, width: semi-axes of the ellipse along and across its normal; the
// weight falls as exp(-2 r^2) to the ellipse border (r = 1) and is zero
// beyond. The presence weight is linear between floor (0) and full (1), the
// alignment |direction . tangent| linearly between minAlignment (0) and 1.
struct FiberPredictionNormalOptions {
    double height = 20., width = 80.;
    double presenceFloor = 0., presenceFull = 128.;
    double minAlignment = 0.8;
    // Share of a sample's weight that does not depend on presence: the presence
    // factor is presenceBase + (1 - presenceBase) * presence weight (1: presence ignored).
    double presenceBase = 0.;
};

// The normal the ellipse of a sample is measured along.
// Sample: the sample's own predicted normal.
// Isotropic: none; the offset length over `width`.
// Given: `axis`, the same for all samples (an earlier estimate).
enum class FiberPredictionNormalFrame { Sample, Isotropic, Given };

// The weighted outer products of the samples' normals at one fiber point.
// With agreementPower > 0 (and a non-zero axis) each sample is also weighted
// by |normal . axis|^agreementPower: the estimate condenses onto the samples
// that agree with an earlier one.
struct FiberPredictionNormalScatter {
    cv::Matx33d scatter = cv::Matx33d::zeros();
    double weight = 0.;  // summed sample weight
    double count = 0.;   // samples with weight
    FiberPredictionNormalScatter& operator+=(const FiberPredictionNormalScatter& other);
    [[nodiscard]] FiberPredictionNormalScatter scaled(double factor) const;
};

[[nodiscard]] FiberPredictionNormalScatter fiberPredictionNormalScatter(
    const cv::Vec3d& tangent, std::span<const FiberPredictionNormalSample> samples,
    const FiberPredictionNormalOptions& options = {},
    FiberPredictionNormalFrame frame = FiberPredictionNormalFrame::Sample, const cv::Vec3d& axis = {0., 0., 0.},
    double agreementPower = 0.);

// The principal axis of a scatter (sign-free; zero when it has no weight).
struct FiberPredictionNormalEstimate {
    cv::Vec3d axis{0., 0., 0.};
    double weight = 0.;     // summed sample weight
    double coherence = 0.;  // (largest - second eigenvalue) / trace
    double count = 0.;      // samples with weight
};

[[nodiscard]] FiberPredictionNormalEstimate fiberPredictionNormal(const FiberPredictionNormalScatter& scatter);

// The smallest rotation that turns unit vector `from` into unit vector `to`
// (identity for opposite vectors); carries a scatter along a bending fiber.
[[nodiscard]] cv::Matx33d fiberDirectionTurn(const cv::Vec3d& from, const cv::Vec3d& to);
[[nodiscard]] FiberPredictionNormalScatter turned(const FiberPredictionNormalScatter& scatter, const cv::Matx33d& turn);

// A running estimate along one fiber. It starts at the seed: the point with
// the most presence-weighted support. From there it moves point by point in
// both directions: the running scatter is carried to the next point (turned
// with the fiber direction, decayed by exp(-step / length)), the next point's
// samples are weighted by |normal . carried axis|^agreementPower, and their
// scatter is added. A point's estimate is the principal axis of the running
// scatter there. `arcs` ascend; one entry of `tangents` and `samples` per point.
struct FiberPredictionNormalTrackOptions {
    double length = 100.;         // decay length of the running average, base voxels
    double agreementPower = 4.;
};

[[nodiscard]] std::vector<FiberPredictionNormalEstimate> trackFiberPredictionNormals(
    std::span<const double> arcs, std::span<const cv::Vec3d> tangents,
    std::span<const std::vector<FiberPredictionNormalSample>> samples, const FiberPredictionNormalOptions& options = {},
    const FiberPredictionNormalTrackOptions& track = {}, std::size_t* seed = nullptr);

}  // namespace vc::fiber_tracer
