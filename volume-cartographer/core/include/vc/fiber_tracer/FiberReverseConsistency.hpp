#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer {

// GT-independent delayed-check scheduling shared by replay and volume tracing.
// Caller owns the committed points and performs checks on its worker. No global
// locks, nested worker pool, or prediction resampling is required by this class.
class FiberReverseCheckSchedule {
public:
    using Verify = std::function<bool(size_t checkpoint, size_t checked, bool terminal)>;
    FiberReverseCheckSchedule(double interval, size_t delay) : interval_(interval), delay_(delay) {}
    bool check(const std::vector<cv::Vec3d>& points, bool terminal, const Verify&);
    // Ordinary presence/width stop: verify from delay edges back, retaining the
    // accepted incoming-history tail. A history-only tail isn't a passed check.
    bool stopTail(const std::vector<cv::Vec3d>& points, const Verify&);
    size_t checked() const { return checked_; }
    bool historyOnly() const { return historyOnly_; }
private:
    void update(const std::vector<cv::Vec3d>&);
    void accept(const std::vector<cv::Vec3d>&, size_t);
    double interval_, unchecked_ = 0;
    size_t delay_, checked_ = 0, observed_ = 0;
    std::optional<size_t> pending_;
    bool historyOnly_ = false;
};

struct FiberReverseCheckpointResult {
    bool passed = false;
    double travelled = 0;
    bool converged = false;
    size_t historyCount = 0;
    double seconds = 0;
};
FiberReverseCheckpointResult verifyFiberReverseCheckpoint(
    const FiberPredictionSource&, const vc::lasagna::NormalSampler&,
    const FiberTraceConfig&, const std::vector<cv::Vec3d>& pointsBase,
    size_t checkpoint, size_t checked, double widthMeanTrace);

struct ReverseSpanOptions {
    // Incoming edges from the FUTURE side of the forward checkpoint, in reverse
    // travel order. Empty preserves legacy initialization.
    std::vector<cv::Vec3d> initialDirections;
    double verifiedThroughArc = 0;
    double minimumTravel = 0;
    double convergenceDistance = 4;
    double convergenceAngleDegrees = 5;
    int convergenceSteps = 4;
};
struct ReverseSpanResult {
    bool passed = false;
    bool converged = false;
    double remainingArc = 0;
    double travelled = 0;
};

std::vector<cv::Vec3d> fiberReverseFutureHistory(
    const PolylineArcGeometry&, size_t checkpoint, size_t count);

// Backend-independent verification against the actual forward prediction.
// Coordinates and radii are base voxels. Geometry must come from
// makePolylineArcGeometry and outlive this object, as must normals.
class ReverseSpanVerifier {
public:
    ReverseSpanVerifier(const PolylineArcGeometry& geometry, double startArc,
        const cv::Vec3d& startPoint, const vc::lasagna::NormalSampler& normals,
        double normalWorkingToBaseScale, double normalRadius, double tangentRadius,
        const ReverseSpanOptions& options = {});
    // Stop at false: either the start was reached, or the first check failed.
    bool observe(const cv::Vec3d& point);
    double remainingArc() const { return arc_; }
    ReverseSpanResult result() const { return {arc_ <= 1e-12 || converged_, converged_, arc_, travelled_}; }
private:
    const PolylineArcGeometry& geometry_;
    const vc::lasagna::NormalSampler& normals_;
    double scale_, normalRadius_, tangentRadius_, arc_;
    cv::Vec3d previous_;
    ReverseSpanOptions options_;
    double travelled_ = 0;
    int closeSteps_ = 0;
    bool converged_ = false;
};

// Return the earliest verified base arclength reached from startArc (zero means
// the whole span passed). No ground truth enters tracing or verification.
// Optional history seeds reverse smoothness from up to 64 forward edges and
// snaps the start to the next forward vertex. Normals use trace coordinates.
// For width gating, pass the forward result's widthMean as widthReferenceMean
// to keep the same fixed reference during the reverse check.
double verifyFiberReverseSpan(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase,
    int historySteps = 0);

ReverseSpanResult verifyFiberReverseSpanDetailed(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase,
    int historySteps, const ReverseSpanOptions& options);

} // namespace vc::fiber_tracer
