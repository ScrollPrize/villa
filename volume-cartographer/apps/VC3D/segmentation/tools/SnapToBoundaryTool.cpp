#include "SnapToBoundaryTool.hpp"

#include "../SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "../../adaptive/CAdaptiveVolumeViewer.hpp"
#include "../../CState.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"

#include <QCoreApplication>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <limits>

Q_LOGGING_CATEGORY(lcSnapTool, "vc.snap")

namespace {

// Snap step + range. Tuned for typical scroll voxel sizes; adjust later if
// surfaces are routinely off by more than ~25 voxels.
constexpr float kSnapStep = 0.5f;        // voxels per ray sample
constexpr float kMaxSnapDistance = 25.0f;  // voxels in either direction
constexpr float kMinDragLengthVoxels = 1.0f;  // ignore micro-drags

// Returns the largest signed iso-crossing offset (along the normal) found
// inside [-kMaxSnapDistance, +kMaxSnapDistance] that best matches preferSign.
// preferSign in {-1, 0, +1}. Returns 0 if no crossing found.
//
// `samples` are intensity values along the ray, indexed [0..numSamples-1],
// with sample i corresponding to offset = (i - midIdx) * step.
float pickBestCrossing(const float* samples,
                       int numSamples,
                       int midIdx,
                       float step,
                       float iso,
                       int preferSign)
{
    // Find all crossings: sample[i-1] and sample[i] straddle iso.
    // Each crossing is recorded as a fractional offset where the linear
    // interpolation hits iso.
    float bestOffset = 0.0f;
    float bestScore = std::numeric_limits<float>::max();
    bool found = false;

    for (int i = 1; i < numSamples; ++i) {
        const float a = samples[i - 1];
        const float b = samples[i];
        const bool aHigh = a >= iso;
        const bool bHigh = b >= iso;
        if (aHigh == bHigh) continue;

        // Linear interp to iso between samples[i-1] and samples[i].
        const float denom = (b - a);
        const float frac = (denom != 0.0f) ? (iso - a) / denom : 0.5f;
        const float offsetIdx = float(i - 1) + frac;
        const float offset = (offsetIdx - float(midIdx)) * step;

        // Score: prefer the crossing nearest current point in the requested
        // sign direction. Wrong-sign crossings are heavily penalized.
        float score;
        if (preferSign == 0) {
            score = std::fabs(offset);
        } else {
            const bool sameSign = (preferSign > 0 && offset >= 0.0f) ||
                                  (preferSign < 0 && offset <= 0.0f);
            score = std::fabs(offset) + (sameSign ? 0.0f : 1e6f);
        }

        if (score < bestScore) {
            bestScore = score;
            bestOffset = offset;
            found = true;
        }
    }

    return found ? bestOffset : 0.0f;
}

// Numerical surface normal at grid (col, row): cross product of central
// differences. Falls back to the +X / +Y face normal at boundaries.
// Returns a unit-length normal, or {0,0,0} if degenerate.
cv::Vec3f surfaceNormalAt(const cv::Mat_<cv::Vec3f>& points, int col, int row)
{
    const int rows = points.rows;
    const int cols = points.cols;
    if (col < 0 || col >= cols || row < 0 || row >= rows) {
        return {0.0f, 0.0f, 0.0f};
    }

    const cv::Vec3f& p = points(row, col);
    if (p[0] == -1.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    auto fetch = [&](int c, int r) -> const cv::Vec3f* {
        if (c < 0 || c >= cols || r < 0 || r >= rows) return nullptr;
        const cv::Vec3f& q = points(r, c);
        if (q[0] == -1.0f) return nullptr;
        return &q;
    };

    const cv::Vec3f* px0 = fetch(col - 1, row);
    const cv::Vec3f* px1 = fetch(col + 1, row);
    const cv::Vec3f* py0 = fetch(col, row - 1);
    const cv::Vec3f* py1 = fetch(col, row + 1);

    cv::Vec3f dx;
    if (px0 && px1)      dx = (*px1 - *px0) * 0.5f;
    else if (px1)        dx = *px1 - p;
    else if (px0)        dx = p - *px0;
    else                 return {0.0f, 0.0f, 0.0f};

    cv::Vec3f dy;
    if (py0 && py1)      dy = (*py1 - *py0) * 0.5f;
    else if (py1)        dy = *py1 - p;
    else if (py0)        dy = p - *py0;
    else                 return {0.0f, 0.0f, 0.0f};

    cv::Vec3f n = dx.cross(dy);
    const float len = static_cast<float>(cv::norm(n));
    if (len < 1e-6f) return {0.0f, 0.0f, 0.0f};
    return n / len;
}

}  // namespace

SnapToBoundaryTool::SnapToBoundaryTool(SegmentationModule& module,
                                       SegmentationEditManager* editManager,
                                       CState* state)
    : _module(module),
      _editManager(editManager),
      _state(state)
{}

void SnapToBoundaryTool::setDependencies(SegmentationEditManager* editManager,
                                         CState* state)
{
    _editManager = editManager;
    _state = state;
}

bool SnapToBoundaryTool::startStroke(CTiledVolumeViewer* viewer,
                                     const cv::Vec3f& worldPos)
{
    cancel();
    if (!viewer || !_editManager) return false;

    auto* surface = viewer->currentSurface();
    auto* plane = dynamic_cast<PlaneSurface*>(surface);
    if (!plane) {
        qCInfo(lcSnapTool) << "snap: ignoring stroke on non-plane viewer";
        return false;
    }

    auto vol = viewer->currentVolume();
    if (!vol) {
        qCWarning(lcSnapTool) << "snap: viewer has no volume";
        return false;
    }

    if (!_editManager->hasSession()) {
        qCWarning(lcSnapTool) << "snap: no active edit session";
        return false;
    }

    _viewer = viewer;
    _volume = std::move(vol);
    _planeNormal = plane->normal({0, 0, 0});
    _startWorld = worldPos;
    _currentWorld = worldPos;
    _previewPoints = {worldPos, worldPos};
    _active = true;
    return true;
}

void SnapToBoundaryTool::extendStroke(const cv::Vec3f& worldPos)
{
    if (!_active) return;
    _currentWorld = worldPos;
    _previewPoints = {_startWorld, _currentWorld};
}

bool SnapToBoundaryTool::applyStroke()
{
    if (!_active) return false;
    const cv::Vec3f drag = _currentWorld - _startWorld;
    const float dragLen = static_cast<float>(cv::norm(drag));
    auto status = [&](const QString& msg, int ms = 2500) {
        Q_EMIT _module.statusMessageRequested(msg, ms);
    };
    if (dragLen < kMinDragLengthVoxels) {
        qCInfo(lcSnapTool) << "snap: drag too short" << dragLen << "voxels — ignoring";
        status(QCoreApplication::translate("SnapToBoundaryTool",
                                           "Snap: drag too short (%1 voxels).")
                   .arg(QString::number(dragLen, 'f', 2)));
        cancel();
        return false;
    }

    if (!_editManager->hasSession()) {
        cancel();
        return false;
    }
    cv::Mat_<cv::Vec3f>& preview = _editManager->previewPointsMutable();
    if (preview.empty()) {
        cancel();
        return false;
    }
    cv::Mat_<cv::Vec3f>* points = &preview;

    // ROI: a disc on the slice plane centered at the click, radius = max(drag,
    // viewport extent fraction). Capped so we never snap on more than
    // ~viewport-sized regions even on a tiny drag.
    float roiRadius = std::max(dragLen * 2.0f, 32.0f);
    if (_viewer) {
        // Approximate scene-rect extent in world units → voxels via dsScale.
        // The exact viewport read happens through the camera; here we just cap
        // at a generous bound so we don't iterate the full surface.
        roiRadius = std::min(roiRadius, 1024.0f);
    }
    const float roiRadiusSq = roiRadius * roiRadius;

    // Plane-band thickness: only consider vertices within this many voxels of
    // the slice plane. Keeps the worked region a thin shell around the plane
    // so we don't drag points that are nowhere near where the user is looking.
    constexpr float kPlaneBand = 16.0f;

    // Collect candidate (col, row) cells whose world position is in the ROI
    // disc and within the plane band.
    struct Cand {
        int col;
        int row;
        cv::Vec3f world;
        cv::Vec3f normal;
    };
    std::vector<Cand> candidates;
    candidates.reserve(4096);

    const int rows = points->rows;
    const int cols = points->cols;
    for (int r = 0; r < rows; ++r) {
        const cv::Vec3f* rowPtr = &(*points)(r, 0);
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = rowPtr[c];
            if (p[0] == -1.0f) continue;

            // Plane-band test: |((p - start) · planeNormal)| <= kPlaneBand
            const cv::Vec3f delta = p - _startWorld;
            const float planeDist = std::fabs(delta.dot(_planeNormal));
            if (planeDist > kPlaneBand) continue;

            // ROI disc test: distance squared in the slice plane.
            const cv::Vec3f inPlane = delta - _planeNormal * delta.dot(_planeNormal);
            const float distSq = inPlane.dot(inPlane);
            if (distSq > roiRadiusSq) continue;

            cv::Vec3f n = surfaceNormalAt(*points, c, r);
            if (n.dot(n) < 1e-6f) continue;

            candidates.push_back({c, r, p, n});
        }
    }

    if (candidates.empty()) {
        qCInfo(lcSnapTool) << "snap: no candidate vertices in ROI";
        status(QCoreApplication::translate("SnapToBoundaryTool",
                                           "Snap: no surface vertices in ROI (%1 voxel band around the slice plane).")
                   .arg(QString::number(kPlaneBand)));
        cancel();
        return false;
    }

    qCInfo(lcSnapTool) << "snap:" << candidates.size() << "candidates, dragLen" << dragLen;

    // Iso threshold = the rendering pipeline's preprocessing iso cutoff
    // (the same slider that "trims off the low value air noise").
    float iso = 0.0f;
    if (_viewer) {
        iso = static_cast<float>(_viewer->compositeRenderSettings().params.isoCutoff);
    }
    std::fprintf(stderr, "[snap] iso=%.2f (from compositeRenderSettings.isoCutoff)\n", iso);
    qCInfo(lcSnapTool) << "snap: iso=" << iso;
    if (iso <= 0.0f) {
        qCWarning(lcSnapTool) << "snap: iso cutoff is 0 — every voxel is 'papyrus', no crossings to find";
        status(QCoreApplication::translate("SnapToBoundaryTool",
                                           "Snap iso=0 (raise the preprocessing iso cutoff above 0)"));
        cancel();
        return false;
    }

    // Build one big coords Mat: rows = candidates, cols = ray samples.
    const int numSamples = static_cast<int>(std::round((2.0f * kMaxSnapDistance) / kSnapStep)) + 1;
    const int midIdx = numSamples / 2;
    cv::Mat_<cv::Vec3f> coords(static_cast<int>(candidates.size()), numSamples);
    for (size_t k = 0; k < candidates.size(); ++k) {
        const auto& cand = candidates[k];
        cv::Vec3f* outRow = &coords(static_cast<int>(k), 0);
        for (int s = 0; s < numSamples; ++s) {
            const float t = (s - midIdx) * kSnapStep;
            outRow[s] = cand.world + cand.normal * t;
        }
    }

    cv::Mat_<uint8_t> values(coords.size());
    vc::SampleParams sp;
    sp.level = 0;
    _volume->sample(values, coords, sp);

    // For each candidate, find best iso crossing along its normal.
    int snapped = 0;
    for (size_t k = 0; k < candidates.size(); ++k) {
        const auto& cand = candidates[k];

        // Per-vertex prefer-sign: the user's drag projected onto this vertex's
        // normal gives a hint about which side of the iso surface to snap to.
        const float dragDot = drag.dot(cand.normal);
        int preferSign = 0;
        if (dragDot > 1e-3f)      preferSign = +1;
        else if (dragDot < -1e-3f) preferSign = -1;

        // Convert uint8 row to float for crossing detection.
        std::vector<float> sampleBuf(numSamples);
        for (int s = 0; s < numSamples; ++s) {
            sampleBuf[s] = static_cast<float>(values(static_cast<int>(k), s));
        }

        const float offset = pickBestCrossing(sampleBuf.data(), numSamples,
                                              midIdx, kSnapStep, iso, preferSign);
        if (offset == 0.0f) continue;

        const cv::Vec3f newPos = cand.world + cand.normal * offset;
        (*points)(cand.row, cand.col) = newPos;
        ++snapped;
    }

    qCInfo(lcSnapTool) << "snap: moved" << snapped << "of" << candidates.size() << "candidates";
    status(QCoreApplication::translate("SnapToBoundaryTool",
                                       "Snap: moved %1 of %2 vertices (iso=%3, drag=%4 vox).")
               .arg(snapped)
               .arg(static_cast<int>(candidates.size()))
               .arg(QString::number(iso, 'f', 1))
               .arg(QString::number(dragLen, 'f', 1)));

    if (snapped > 0) {
        // Sync the undo snapshot to match the just-mutated preview, then notify
        // the rest of the app the surface changed (mirrors SegmentationLineTool).
        _editManager->applyPreview();
        if (_state) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
        }
        _module.refreshOverlay();
        _module.emitPendingChanges();
        _module.markAutosaveNeeded();
    }

    cancel();
    return snapped > 0;
}

void SnapToBoundaryTool::cancel()
{
    _active = false;
    _viewer = nullptr;
    _volume.reset();
    _previewPoints.clear();
}
