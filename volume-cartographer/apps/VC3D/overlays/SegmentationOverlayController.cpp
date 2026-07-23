#include "SegmentationOverlayController.hpp"

#include "../CState.hpp"
#include "SegmentationIntersectionInvalidation.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../ViewerManager.hpp"
#include "../segmentation/tools/SegmentationEditManager.hpp"

#include <QColor>
#include <QDebug>
#include <QElapsedTimer>
#include <QPainter>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgcodecs.hpp>

#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

namespace
{
constexpr const char* kOverlayGroupKey = "segmentation_vertex_edit";
const QColor kActiveFill = QColor(255, 215, 0, 220);
const QColor kActiveBorder = QColor(255, 180, 0, 255);
const QColor kNeighborFill = QColor(0, 220, 255, 160);
const QColor kNeighborBorder = QColor(255, 255, 255, 210);
const QColor kGrowthFill = QColor(140, 255, 160, 210);
const QColor kGrowthBorder = QColor(80, 190, 90, 230);
const QColor kHoverFill = QColor(255, 255, 255, 200);
const QColor kHoverBorder = QColor(255, 255, 255, 240);
constexpr qreal kMarkerRadius = 5.0;
constexpr qreal kMarkerPenWidth = 2.0;
constexpr qreal kActivePenWidth = 2.5;
constexpr qreal kMaskZ = 60.0;
constexpr qreal kApprovalMaskZ = 50.0;  // Below mask path (60) but above most other overlays
constexpr qreal kSurfaceOverlapZ = 45.0;  // Below approval mask
constexpr qreal kMarkerZ = 95.0;
constexpr qreal kRadiusCircleZ = 80.0;
constexpr float kManualAddPlaneDistanceTolerance = 10.0f;
constexpr qreal kManualAddIntersectionZ = 121.0;
constexpr float kManualAddIntersectionOpacityScale = 1.2f;
constexpr float kManualAddIntersectionWidthScale = 1.3f;
constexpr float kManualAddIntersectionMinWidthDelta = 0.75f;
constexpr int kSurfaceOverlapSampleStride = 1;

// Full opacity for mask pixels - the slider controls overall opacity via QGraphicsPixmapItem::setOpacity
constexpr int kApprovalMaskAlpha = 255;

bool validSurfaceOverlapPoint(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
           p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f;
}

int overlapSampleCount(int fullCount)
{
    if (fullCount <= 0) {
        return 0;
    }
    int count = (fullCount - 1) / kSurfaceOverlapSampleStride + 1;
    if ((fullCount - 1) % kSurfaceOverlapSampleStride != 0) {
        ++count;
    }
    return count;
}

int overlapSampleIndex(int sampleIndex, int sampleCount, int fullCount)
{
    if (sampleIndex + 1 == sampleCount) {
        return fullCount - 1;
    }
    return std::min(sampleIndex * kSurfaceOverlapSampleStride, fullCount - 1);
}

class OverlayProfileScope {
public:
    OverlayProfileScope(const char* event, const char* detail = "")
        : event_(event)
        , detail_(detail)
        , enabled_(ProfileLoggingEnabled())
    {
        if (!enabled_) {
            return;
        }
        timer_.start();
        Logger()->info("[vc3d-profile] {} begin{}", event_, detailSuffix());
    }

    ~OverlayProfileScope()
    {
        if (!enabled_) {
            return;
        }
        Logger()->info("[vc3d-profile] {} end elapsed_ms={}{}",
                       event_,
                       timer_.elapsed(),
                       detailSuffix());
    }

private:
    std::string detailSuffix() const
    {
        return detail_.empty() ? std::string{} : " detail='" + detail_ + "'";
    }

    const char* event_;
    std::string detail_;
    bool enabled_{false};
    QElapsedTimer timer_;
};

QColor manualAddIntersectionColor()
{
    return QColor(0, 220, 255);
}

qreal manualAddIntersectionWidth(qreal baseWidth)
{
    return std::max(baseWidth * kManualAddIntersectionWidthScale,
                    baseWidth + kManualAddIntersectionMinWidthDelta);
}

std::optional<QPointF> markerGridToScene(VolumeViewerBase* viewer,
                                         const SegmentationOverlayController::VertexMarker& marker)
{
    if (!viewer || viewer->surfName() != "segmentation") {
        return std::nullopt;
    }

    auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!quad) {
        return std::nullopt;
    }

    const cv::Vec2f scale = quad->scale();
    if (std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) {
        return std::nullopt;
    }

    const cv::Vec3f center = quad->center();
    const float surfaceX = static_cast<float>(marker.col) / scale[0] - center[0];
    const float surfaceY = static_cast<float>(marker.row) / scale[1] - center[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}
}

bool SegmentationOverlayController::State::operator==(const State& rhs) const
{
    const auto equalMarker = [](const VertexMarker& lhs, const VertexMarker& rhs) {
        return lhs.row == rhs.row &&
               lhs.col == rhs.col &&
               std::fabs(lhs.world[0] - rhs.world[0]) < 1e-4f &&
               std::fabs(lhs.world[1] - rhs.world[1]) < 1e-4f &&
               std::fabs(lhs.world[2] - rhs.world[2]) < 1e-4f &&
               lhs.isActive == rhs.isActive &&
               lhs.isGrowth == rhs.isGrowth;
    };

    const auto optionalEqual = [&](const std::optional<VertexMarker>& lhsOpt,
                                   const std::optional<VertexMarker>& rhsOpt) {
        if (!lhsOpt && !rhsOpt) {
            return true;
        }
        if (static_cast<bool>(lhsOpt) != static_cast<bool>(rhsOpt)) {
            return false;
        }
        return equalMarker(*lhsOpt, *rhsOpt);
    };

    const auto vectorEqual = [&](const std::vector<VertexMarker>& lhsVec,
                                 const std::vector<VertexMarker>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            if (!equalMarker(lhsVec[i], rhsVec[i])) {
                return false;
            }
        }
        return true;
    };

    const auto maskEqual = [&](const std::vector<cv::Vec3f>& lhsVec,
                               const std::vector<cv::Vec3f>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            const cv::Vec3f delta = lhsVec[i] - rhsVec[i];
            if (cv::norm(delta) >= 1e-4f) {
                return false;
            }
        }
        return true;
    };

    const auto floatEqual = [](float lhs, float rhs) {
        return std::fabs(lhs - rhs) < 1e-4f;
    };

    const auto vec3fOptEqual = [&](const std::optional<cv::Vec3f>& lhs, const std::optional<cv::Vec3f>& rhs) {
        if (!lhs && !rhs) return true;
        if (static_cast<bool>(lhs) != static_cast<bool>(rhs)) return false;
        return cv::norm(*lhs - *rhs) < 1e-4f;
    };

    // Helper for cv::Vec3f comparison
    const auto vec3fEqual = [](const cv::Vec3f& lhs, const cv::Vec3f& rhs) {
        return cv::norm(lhs - rhs) < 1e-4f;
    };

    return optionalEqual(activeMarker, rhs.activeMarker) &&
           vectorEqual(neighbours, rhs.neighbours) &&
           maskEqual(maskPoints, rhs.maskPoints) &&
           surfaceMaskPoints == rhs.surfaceMaskPoints &&
           maskVisible == rhs.maskVisible &&
           brushActive == rhs.brushActive &&
           brushStrokeActive == rhs.brushStrokeActive &&
           lineStrokeActive == rhs.lineStrokeActive &&
           hasLineStroke == rhs.hasLineStroke &&
           pushPullActive == rhs.pushPullActive &&
           falloff == rhs.falloff &&
           floatEqual(gaussianRadiusSteps, rhs.gaussianRadiusSteps) &&
           floatEqual(gaussianSigmaSteps, rhs.gaussianSigmaSteps) &&
           floatEqual(displayRadiusSteps, rhs.displayRadiusSteps) &&
           floatEqual(gridStepWorld, rhs.gridStepWorld) &&
           correctionDragActive == rhs.correctionDragActive &&
           vec3fEqual(correctionDragStart, rhs.correctionDragStart) &&
           vec3fEqual(correctionDragCurrent, rhs.correctionDragCurrent) &&
           approvalMaskMode == rhs.approvalMaskMode &&
           approvalStrokeActive == rhs.approvalStrokeActive &&
           approvalStrokeSegments == rhs.approvalStrokeSegments &&
           maskEqual(approvalCurrentStroke, rhs.approvalCurrentStroke) &&
           floatEqual(approvalBrushRadius, rhs.approvalBrushRadius) &&
           paintingApproval == rhs.paintingApproval &&
           approvalBrushColor == rhs.approvalBrushColor &&
           surface == rhs.surface &&
           approvalHoverSurfacePos == rhs.approvalHoverSurfacePos &&
           vec3fOptEqual(approvalHoverPlaneNormal, rhs.approvalHoverPlaneNormal) &&
           manualAddActive == rhs.manualAddActive &&
           manualAddRevision == rhs.manualAddRevision;
}

SegmentationOverlayController::SegmentationOverlayController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupKey, parent)
    , _state(state)
{
    if (_state) {
        connect(_state, &CState::surfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }

    _overlapWatcher = new QFutureWatcher<QImage>(this);
    connect(_overlapWatcher, &QFutureWatcher<QImage>::finished,
            this, &SegmentationOverlayController::handleOverlapComputeFinished);
}

void SegmentationOverlayController::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;
    refreshAll();
}

void SegmentationOverlayController::setEditManager(SegmentationEditManager* manager)
{
    _editManager = manager;
}

void SegmentationOverlayController::applyState(const State& state)
{
    State sanitized = state;
    sanitized.gaussianRadiusSteps = std::max(sanitized.gaussianRadiusSteps, 0.0f);
    sanitized.gaussianSigmaSteps = std::max(sanitized.gaussianSigmaSteps, 0.0f);
    sanitized.displayRadiusSteps = std::max(sanitized.displayRadiusSteps, 0.0f);
    sanitized.gridStepWorld = std::max(sanitized.gridStepWorld, 1e-4f);

    // skip refresh if state hasn't changed
    if (_currentState && *_currentState == sanitized) {
        return;
    }

    // throttle the overlay refresh a bit
    static constexpr auto kMinRefreshInterval = std::chrono::milliseconds(32);
    auto now = std::chrono::steady_clock::now();
    if (now - _lastRefreshTime < kMinRefreshInterval) {
        // state changed but we're throttling - save state for next refresh
        _currentState = std::move(sanitized);
        // ensure a deferred refresh fires after the throttle window
        if (!_deferredRefreshTimer) {
            _deferredRefreshTimer = new QTimer(this);
            _deferredRefreshTimer->setSingleShot(true);
            connect(_deferredRefreshTimer, &QTimer::timeout, this, [this]() {
                _lastRefreshTime = {};  // Reset so next applyState proceeds
                if (_currentState) {
                    refreshAll();
                }
            });
        }
        _deferredRefreshTimer->start(static_cast<int>(kMinRefreshInterval.count()));
        return;
    }
    _lastRefreshTime = now;

    _currentState = std::move(sanitized);
    refreshAll();
}

void SegmentationOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    _viewerCaches.erase(viewer);
    _overlapCaches.erase(viewer);
    if (_pendingOverlap.viewer == viewer) {
        _pendingOverlap.viewer = nullptr;
        if (_overlapCancel) {
            _overlapCancel->store(true, std::memory_order_relaxed);
        }
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

void SegmentationOverlayController::loadApprovalMaskImage(QuadSurface* surface)
{
    // Segment switch: flush pending approval edits to the OLD surface now,
    // before its pointer changes, so they aren't lost.
    flushPendingApprovalMaskSave();

    // Clear undo stack - old entries are for a different surface
    clearApprovalMaskUndoHistory();

    if (!surface) {
        _savedApprovalMaskImage = QImage();
        _pendingApprovalMaskImage = QImage();
        return;
    }

    cv::Mat approvalMask = surface->channel("approval", SURF_CHANNEL_NORESIZE);
    if (approvalMask.empty()) {
        // Create new empty mask
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (!points || points->empty()) {
            _savedApprovalMaskImage = QImage();
            _pendingApprovalMaskImage = QImage();
            return;
        }
        approvalMask = cv::Mat_<uint8_t>(points->size(), static_cast<uint8_t>(0));
    }

    // Convert saved mask to ARGB32_Premultiplied format
    // Supports both legacy 1-channel B&W masks and new 3-channel RGB masks
    QImage savedImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);

    if (approvalMask.channels() == 1) {
        // Legacy 1-channel mask: 0=unapproved, non-zero=approved
        // Convert approved areas to pure green (0, 255, 0)
        for (int row = 0; row < approvalMask.rows; ++row) {
            const uint8_t* maskRow = approvalMask.ptr<uint8_t>(row);
            QRgb* imageRow = reinterpret_cast<QRgb*>(savedImage.scanLine(row));
            for (int col = 0; col < approvalMask.cols; ++col) {
                uint8_t val = maskRow[col];
                if (val > 0) {
                    // Legacy B&W: convert to pure green
                    imageRow[col] = qRgba(0, 255, 0, kApprovalMaskAlpha);
                } else {
                    imageRow[col] = qRgba(0, 0, 0, 0);  // Fully transparent
                }
            }
        }
    } else if (approvalMask.channels() == 3) {
        // New 3-channel RGB mask: (0,0,0)=unapproved, any other RGB=approved with that color
        for (int row = 0; row < approvalMask.rows; ++row) {
            const cv::Vec3b* maskRow = approvalMask.ptr<cv::Vec3b>(row);
            QRgb* imageRow = reinterpret_cast<QRgb*>(savedImage.scanLine(row));
            for (int col = 0; col < approvalMask.cols; ++col) {
                const cv::Vec3b& pixel = maskRow[col];
                // OpenCV uses BGR order, Qt uses RGB
                uint8_t b = pixel[0];
                uint8_t g = pixel[1];
                uint8_t r = pixel[2];
                if (r > 0 || g > 0 || b > 0) {
                    imageRow[col] = qRgba(r, g, b, kApprovalMaskAlpha);
                } else {
                    imageRow[col] = qRgba(0, 0, 0, 0);  // Fully transparent
                }
            }
        }
    } else {
        qWarning() << "loadApprovalMaskImage: unexpected channel count:" << approvalMask.channels();
        savedImage.fill(qRgba(0, 0, 0, 0));
    }

    _savedApprovalMaskImage = savedImage;

    // Initialize empty pending mask with same dimensions
    _pendingApprovalMaskImage = QImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));  // Fully transparent

    // Invalidate all viewer caches
    ++_savedImageVersion;
    ++_pendingImageVersion;

    // Trigger re-rendering of intersection lines on plane viewers
    invalidatePlaneIntersections();
}

void SegmentationOverlayController::paintApprovalMaskDirect(
    const std::vector<std::pair<int, int>>& gridPositions,
    float radiusSteps,
    uint8_t paintValue,
    const QColor& brushColor,
    bool useRectangle,
    float widthSteps,
    float heightSteps,
    bool isAutoApproval)
{
    OverlayProfileScope profile("paintApprovalMaskDirect",
                                isAutoApproval ? "auto_approval" : "manual");
    if (_pendingApprovalMaskImage.isNull()) {
        qWarning() << "paintApprovalMaskDirect: pending image is NULL!";
        return;
    }
    if (gridPositions.empty()) {
        return;
    }

    const int radius = static_cast<int>(std::ceil(radiusSteps));

    // For rectangle mode (flattened view), use explicit width and height
    const int rectHalfWidth = useRectangle && widthSteps > 0
        ? static_cast<int>(std::ceil(widthSteps / 2.0f))
        : radius;
    const int rectHalfHeight = useRectangle && heightSteps > 0
        ? static_cast<int>(std::ceil(heightSteps / 2.0f))
        : radius;

    // Compute bounding box of affected region
    int minRow = std::numeric_limits<int>::max();
    int maxRow = std::numeric_limits<int>::min();
    int minCol = std::numeric_limits<int>::max();
    int maxCol = std::numeric_limits<int>::min();

    for (const auto& [row, col] : gridPositions) {
        minRow = std::min(minRow, row - rectHalfHeight);
        maxRow = std::max(maxRow, row + rectHalfHeight);
        minCol = std::min(minCol, col - rectHalfWidth);
        maxCol = std::max(maxCol, col + rectHalfWidth);
    }

    // Clamp to image bounds
    minRow = std::max(0, minRow);
    maxRow = std::min(_pendingApprovalMaskImage.height() - 1, maxRow);
    minCol = std::max(0, minCol);
    maxCol = std::min(_pendingApprovalMaskImage.width() - 1, maxCol);

    const int regionWidth = maxCol - minCol + 1;
    const int regionHeight = maxRow - minRow + 1;

    // Save the affected regions for undo (both pending and saved images)
    if (regionWidth > 0 && regionHeight > 0) {
        ApprovalMaskUndoEntry undoEntry;
        undoEntry.topLeft = QPoint(minCol, minRow);
        undoEntry.pendingRegion = _pendingApprovalMaskImage.copy(minCol, minRow, regionWidth, regionHeight);
        if (!_savedApprovalMaskImage.isNull()) {
            undoEntry.savedRegion = _savedApprovalMaskImage.copy(minCol, minRow, regionWidth, regionHeight);
        }
        undoEntry.isAutoApproval = isAutoApproval;
        _approvalMaskUndoStack.push_back(std::move(undoEntry));

        // Limit undo stack size
        if (_approvalMaskUndoStack.size() > kMaxUndoEntries) {
            _approvalMaskUndoStack.pop_front();
        }
    }

    // Paint directly into the images. Both masks are Format_ARGB32_Premultiplied,
    // so we write packed QRgb straight into scanLine() rows -- far cheaper than
    // setPixel() (no per-pixel detach/format dispatch). alpha is 255 (or 0), so
    // the value equals its premultiplied form; no conversion needed.
    const int imgH = _pendingApprovalMaskImage.height();
    const int imgW = _pendingApprovalMaskImage.width();
    const QRgb approveRgb = qRgba(brushColor.red(), brushColor.green(), brushColor.blue(), kApprovalMaskAlpha);
    const QRgb clearRgb = qRgba(0, 0, 0, 0);
    const float radiusSq = radiusSteps * radiusSteps;  // circle span uses the exact (float) radius

    const bool clearSaved = (paintValue == 0) && !_savedApprovalMaskImage.isNull();
    const int savedH = clearSaved ? _savedApprovalMaskImage.height() : 0;
    const int savedW = clearSaved ? _savedApprovalMaskImage.width() : 0;

    const QRgb writeRgb = (paintValue > 0) ? approveRgb : clearRgb;

    for (const auto& [centerRow, centerCol] : gridPositions) {
        const int colRange = useRectangle ? rectHalfWidth : radius;
        const int rowRange = useRectangle ? rectHalfHeight : radius;

        const int r0 = std::max(0, centerRow - rowRange);
        const int r1 = std::min(imgH - 1, centerRow + rowRange);

        for (int row = r0; row <= r1; ++row) {
            // Per-row horizontal span. For a circle, solve the column extent once
            // (dc^2 <= r^2 - dr^2) instead of testing every pixel; for a rectangle
            // the span is the full colRange. Then fill the run contiguously.
            int dcMax = colRange;
            if (!useRectangle) {
                const int dr = row - centerRow;
                const float rem = radiusSq - static_cast<float>(dr * dr);
                if (rem < 0.f) continue;  // row entirely outside the circle
                dcMax = static_cast<int>(std::sqrt(rem));
            }
            const int c0 = std::max(0, centerCol - dcMax);
            const int c1 = std::min(imgW - 1, centerCol + dcMax);
            if (c1 < c0) continue;

            QRgb* pendRow = reinterpret_cast<QRgb*>(_pendingApprovalMaskImage.scanLine(row));
            std::fill(pendRow + c0, pendRow + c1 + 1, writeRgb);

            // Unapproving also clears the saved image directly (no red preview).
            if (clearSaved && row < savedH) {
                QRgb* savedRow = reinterpret_cast<QRgb*>(_savedApprovalMaskImage.scanLine(row));
                const int sc1 = std::min(c1, savedW - 1);
                if (sc1 >= c0) {
                    std::fill(savedRow + c0, savedRow + sc1 + 1, clearRgb);
                }
            }
        }
    }

    // If unapproving, also invalidate saved version since we modified it directly
    if (paintValue == 0) {
        ++_savedImageVersion;
    }

    // Invalidate pending version since we modified the pending image
    ++_pendingImageVersion;

    // Manual painting should update interactively; auto-approval bursts are coalesced.
    if (isAutoApproval) {
        schedulePlaneIntersectionRefresh();
    } else {
        invalidatePlaneIntersections();
    }
}

void SegmentationOverlayController::composeApprovalMaskToSurface(QuadSurface* surface)
{
    OverlayProfileScope profile("composeApprovalMaskToSurface");
    if (!surface || (_savedApprovalMaskImage.isNull() && _pendingApprovalMaskImage.isNull())) {
        return;
    }

    // Merge pending approvals into saved, then convert to cv::Mat (3-channel RGB)
    // Note: Unapprovals are applied immediately to saved image in paintApprovalMaskDirect(),
    // so pending only contains colored approval pixels
    int width = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.width() : _pendingApprovalMaskImage.width();
    int height = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.height() : _pendingApprovalMaskImage.height();

    // Create 3-channel BGR mask for OpenCV (BGR order for cv::imwrite)
    cv::Mat_<cv::Vec3b> approvalMask(height, width, cv::Vec3b(0, 0, 0));

    for (int row = 0; row < height; ++row) {
        cv::Vec3b* maskRow = approvalMask.ptr<cv::Vec3b>(row);
        for (int col = 0; col < width; ++col) {
            uint8_t finalR = 0, finalG = 0, finalB = 0;

            // Check saved value first
            if (!_savedApprovalMaskImage.isNull() && row < _savedApprovalMaskImage.height() && col < _savedApprovalMaskImage.width()) {
                const QRgb* savedRow = reinterpret_cast<const QRgb*>(_savedApprovalMaskImage.constScanLine(row));
                QRgb pixel = savedRow[col];
                if (qAlpha(pixel) > 0) {
                    finalR = qRed(pixel);
                    finalG = qGreen(pixel);
                    finalB = qBlue(pixel);
                }
            }

            // Check pending - pending overwrites saved if present
            if (!_pendingApprovalMaskImage.isNull() && row < _pendingApprovalMaskImage.height() && col < _pendingApprovalMaskImage.width()) {
                const QRgb* pendingRow = reinterpret_cast<const QRgb*>(_pendingApprovalMaskImage.constScanLine(row));
                QRgb pixel = pendingRow[col];
                if (qAlpha(pixel) > 0) {
                    finalR = qRed(pixel);
                    finalG = qGreen(pixel);
                    finalB = qBlue(pixel);
                }
            }

            // Store as BGR for OpenCV
            maskRow[col] = cv::Vec3b(finalB, finalG, finalR);
        }
    }

    // Stash the composited mask onto the surface's approval channel. Disk
    // persistence is the single SegmentationModule autosave's job (saveOverwrite
    // writes x/y/z + meta + all channels incl. approval every 10s).
    surface->setChannel("approval", approvalMask);

    // Update saved image to match what we just wrote and clear pending
    for (int row = 0; row < height; ++row) {
        QRgb* savedRow = reinterpret_cast<QRgb*>(_savedApprovalMaskImage.scanLine(row));
        const cv::Vec3b* maskRow = approvalMask.ptr<cv::Vec3b>(row);
        for (int col = 0; col < width; ++col) {
            const cv::Vec3b& bgr = maskRow[col];
            if (bgr[0] > 0 || bgr[1] > 0 || bgr[2] > 0) {
                // Convert BGR back to RGB for QImage
                savedRow[col] = qRgba(bgr[2], bgr[1], bgr[0], kApprovalMaskAlpha);
            } else {
                savedRow[col] = qRgba(0, 0, 0, 0);
            }
        }
    }

    // Clear pending
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));

    // Invalidate both versions
    ++_savedImageVersion;
    ++_pendingImageVersion;

    // Trigger re-rendering of intersection lines on plane viewers
    invalidatePlaneIntersections();

    // NOTE: We intentionally do NOT clear undo history here.
    // The undo entries store pre-paint image regions, which remain valid for undo
    // even after save merges pending into saved. This allows surface edit undo
    // to also undo the corresponding auto-approval that was painted.
}

bool SegmentationOverlayController::undoLastApprovalMaskPaint()
{
    if (_approvalMaskUndoStack.empty() || _pendingApprovalMaskImage.isNull()) {
        return false;
    }

    // Pop the last entry
    ApprovalMaskUndoEntry entry = std::move(_approvalMaskUndoStack.back());
    _approvalMaskUndoStack.pop_back();

    // Restore the pending image region
    {
        QPainter painter(&_pendingApprovalMaskImage);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
        painter.drawImage(entry.topLeft, entry.pendingRegion);
        painter.end();
    }

    // Restore the saved image region (for unapprovals that modified it directly)
    if (!entry.savedRegion.isNull() && !_savedApprovalMaskImage.isNull()) {
        QPainter painter(&_savedApprovalMaskImage);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
        painter.drawImage(entry.topLeft, entry.savedRegion);
        painter.end();
        ++_savedImageVersion;
    }

    ++_pendingImageVersion;
    invalidatePlaneIntersections();
    forceRefreshAllOverlays();
    return true;
}

bool SegmentationOverlayController::undoLastAutoApproval()
{
    // Find and undo the most recent auto-approval entry
    for (auto it = _approvalMaskUndoStack.rbegin(); it != _approvalMaskUndoStack.rend(); ++it) {
        if (it->isAutoApproval) {
            // Found an auto-approval entry - restore it
            ApprovalMaskUndoEntry entry = std::move(*it);
            _approvalMaskUndoStack.erase(std::next(it).base());

            // Restore the pending image region
            {
                QPainter painter(&_pendingApprovalMaskImage);
                painter.setCompositionMode(QPainter::CompositionMode_Source);
                painter.drawImage(entry.topLeft, entry.pendingRegion);
                painter.end();
            }

            // Restore the saved image region
            if (!entry.savedRegion.isNull() && !_savedApprovalMaskImage.isNull()) {
                QPainter painter(&_savedApprovalMaskImage);
                painter.setCompositionMode(QPainter::CompositionMode_Source);
                painter.drawImage(entry.topLeft, entry.savedRegion);
                painter.end();
                ++_savedImageVersion;
            }

            ++_pendingImageVersion;
            invalidatePlaneIntersections();
            forceRefreshAllOverlays();
            return true;
        }
    }
    return false;
}

bool SegmentationOverlayController::canUndoApprovalMaskPaint() const
{
    return !_approvalMaskUndoStack.empty();
}

bool SegmentationOverlayController::canUndoAutoApproval() const
{
    for (const auto& entry : _approvalMaskUndoStack) {
        if (entry.isAutoApproval) {
            return true;
        }
    }
    return false;
}

void SegmentationOverlayController::clearApprovalMaskUndoHistory()
{
    _approvalMaskUndoStack.clear();
}

void SegmentationOverlayController::scheduleDebouncedSave(QuadSurface* surface)
{
    // Compose the edited mask onto the surface's approval channel and remember
    // the surface for the boundary flush. The caller (which owns the module)
    // triggers the single autosave; there is no separate approval timer.
    if (!surface) {
        return;
    }
    composeApprovalMaskToSurface(surface);
    _approvalSaveSurface = surface;
}

void SegmentationOverlayController::flushPendingApprovalMaskSave()
{
    // Boundary flush (segment switch / app close): write the pending approval
    // edits to disk synchronously before the surface pointer changes. Uses the
    // full saveOverwrite so it shares the per-dir lock with the autosave.
    if (!_approvalSaveSurface) {
        return;
    }
    QuadSurface* surface = _approvalSaveSurface;
    _approvalSaveSurface = nullptr;
    composeApprovalMaskToSurface(surface);
    try {
        surface->saveOverwrite();
    } catch (const std::exception& e) {
        qWarning() << "flushPendingApprovalMaskSave: save failed:" << e.what();
    }
}

bool SegmentationOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    // Enable overlay rendering if editing is enabled OR if approval mask mode is active
    // (approval mask can be viewed without editing enabled)
    const bool approvalMaskActive = _currentState && _currentState->approvalMaskMode;

    // Also enable if surface overlap overlay is active on this viewer
    bool surfaceOverlapActive = false;
    surfaceOverlapActive = viewer && viewer->surfaceOverlayEnabled() &&
                           !viewer->surfaceOverlays().empty();

    return _editingEnabled || approvalMaskActive || surfaceOverlapActive;
}

void SegmentationOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }

    // Render surface overlap overlay (independent of editing mode and state)
    buildSurfaceOverlapOverlay(viewer, builder);

    if (!_currentState) {
        return;
    }

    const State& state = *_currentState;

    if (state.manualAddActive) {
        const bool flattened = viewer->surfName() == "segmentation";
        auto* quadSurface = flattened ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
        auto gridToScene = [&](const QPointF& grid) -> QPointF {
            if (!quadSurface) {
                return {};
            }
            const cv::Vec2f surfScale = quadSurface->scale();
            const cv::Vec3f center = quadSurface->center();
            if (std::abs(surfScale[0]) < 1e-6f || std::abs(surfScale[1]) < 1e-6f) {
                return {};
            }
            const float surfX = static_cast<float>(grid.x()) / surfScale[0] - center[0];
            const float surfY = static_cast<float>(grid.y()) / surfScale[1] - center[1];
            return viewer->surfaceCoordsToScene(surfX, surfY);
        };
        auto drawLine = [&](const State::ManualAddLine& line) {
            std::vector<QPointF> points;
            if (flattened) {
                points.reserve(line.surfacePoints.size());
                for (const QPointF& grid : line.surfacePoints) {
                    const QPointF scene = gridToScene(grid);
                    if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                        points.push_back(scene);
                    }
                }
            } else if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
                Q_UNUSED(plane);
                return;
            } else {
                points = volumeToScene(viewer, line.worldPoints);
            }
            if (points.size() < 2) {
                return;
            }
            ViewerOverlayControllerBase::OverlayStyle style;
            style.penColor = line.committed ? QColor(0, 220, 255, 240) : QColor(255, 220, 0, 230);
            style.penWidth = 3.0;
            style.z = 110.0;
            builder.addLineStrip(points, false, style);
        };
        if (flattened) {
            ViewerOverlayControllerBase::OverlayStyle hoverFillStyle;
            hoverFillStyle.penColor = Qt::transparent;
            hoverFillStyle.brushColor = QColor(255, 220, 0, 55);
            hoverFillStyle.penWidth = 0.0;
            hoverFillStyle.z = 104.0;
            for (const auto& span : state.manualAddHoverFillSpans) {
                const QPointF topLeft = gridToScene(QPointF(span.colFirst - 0.5, span.row - 0.5));
                const QPointF bottomRight = gridToScene(QPointF(span.colLast + 0.5, span.row + 0.5));
                if (!std::isfinite(topLeft.x()) || !std::isfinite(topLeft.y()) ||
                    !std::isfinite(bottomRight.x()) || !std::isfinite(bottomRight.y())) {
                    continue;
                }
                builder.addRect(QRectF(topLeft, bottomRight).normalized(), true, hoverFillStyle);
            }
        }
        for (const auto& line : state.manualAddCommittedLines) {
            drawLine(line);
        }
        for (const auto& line : state.manualAddHoverLines) {
            drawLine(line);
        }
        if (flattened && state.manualAddHoverVertex) {
            const QPointF scene = gridToScene(*state.manualAddHoverVertex);
            if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                ViewerOverlayControllerBase::OverlayStyle hoverStyle;
                hoverStyle.penColor = QColor(255, 255, 255, 250);
                hoverStyle.brushColor = state.manualAddHoverCrossFill ? Qt::transparent : QColor(255, 220, 0, 220);
                hoverStyle.penWidth = 2.0;
                hoverStyle.z = 112.0;
                builder.addCircle(scene, state.manualAddHoverCrossFill ? 8.0 : 5.0, !state.manualAddHoverCrossFill, hoverStyle);
            }
        }
        if (!flattened) {
            auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface());
            if (plane) {
                ViewerOverlayControllerBase::OverlayStyle previewStyle;
                previewStyle.penColor = manualAddIntersectionColor();
                previewStyle.penColor.setAlphaF(std::clamp(viewer->intersectionOpacity() *
                                                           kManualAddIntersectionOpacityScale,
                                                           0.0f,
                                                           1.0f));
                previewStyle.penWidth = manualAddIntersectionWidth(viewer->intersectionThickness());
                previewStyle.z = kManualAddIntersectionZ;

                auto addUnique = [](std::vector<cv::Vec3f>& points, const cv::Vec3f& candidate) {
                    for (const cv::Vec3f& point : points) {
                        if (cv::norm(point - candidate) < 1e-3f) {
                            return;
                        }
                    }
                    points.push_back(candidate);
                };

                constexpr int kEdges[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
                constexpr float kPlaneEpsilon = 1e-3f;
                std::vector<std::pair<cv::Vec3f, cv::Vec3f>> segments;
                for (const auto& quad : state.manualAddPreviewQuads) {
                    std::vector<cv::Vec3f> intersections;
                    for (const auto& edge : kEdges) {
                        const cv::Vec3f& a = quad[edge[0]];
                        const cv::Vec3f& b = quad[edge[1]];
                        const float da = plane->scalarp(a);
                        const float db = plane->scalarp(b);
                        const bool aOnPlane = std::abs(da) <= kPlaneEpsilon;
                        const bool bOnPlane = std::abs(db) <= kPlaneEpsilon;
                        if (aOnPlane) {
                            addUnique(intersections, a);
                        }
                        if (bOnPlane) {
                            addUnique(intersections, b);
                        }
                        if (aOnPlane || bOnPlane || da * db > 0.0f) {
                            continue;
                        }
                        const float t = da / (da - db);
                        addUnique(intersections, a + (b - a) * t);
                    }
                    if (intersections.size() == 2) {
                        segments.push_back({intersections[0], intersections[1]});
                    } else if (intersections.size() > 2) {
                        std::pair<int, int> farthest{0, 1};
                        float farthestDist = 0.0f;
                        for (int i = 0; i < static_cast<int>(intersections.size()); ++i) {
                            for (int j = i + 1; j < static_cast<int>(intersections.size()); ++j) {
                                const float dist = cv::norm(intersections[i] - intersections[j]);
                                if (dist > farthestDist) {
                                    farthestDist = dist;
                                    farthest = {i, j};
                                }
                            }
                        }
                        segments.push_back({intersections[farthest.first], intersections[farthest.second]});
                    }
                }

                std::vector<std::vector<cv::Vec3f>> polylines;
                constexpr float kMergeEpsilon = 1e-2f;
                auto samePoint = [](const cv::Vec3f& a, const cv::Vec3f& b) {
                    return cv::norm(a - b) < kMergeEpsilon;
                };
                for (const auto& segment : segments) {
                    if (cv::norm(segment.first - segment.second) < kMergeEpsilon) {
                        continue;
                    }
                    bool merged = false;
                    for (auto& line : polylines) {
                        if (samePoint(line.back(), segment.first)) {
                            line.push_back(segment.second);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.back(), segment.second)) {
                            line.push_back(segment.first);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.front(), segment.second)) {
                            line.insert(line.begin(), segment.first);
                            merged = true;
                            break;
                        }
                        if (samePoint(line.front(), segment.first)) {
                            line.insert(line.begin(), segment.second);
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        polylines.push_back({segment.first, segment.second});
                    }
                }
                for (bool changed = true; changed;) {
                    changed = false;
                    for (std::size_t i = 0; i < polylines.size() && !changed; ++i) {
                        for (std::size_t j = i + 1; j < polylines.size(); ++j) {
                            if (samePoint(polylines[i].back(), polylines[j].front())) {
                                polylines[i].insert(polylines[i].end(), polylines[j].begin() + 1, polylines[j].end());
                            } else if (samePoint(polylines[i].back(), polylines[j].back())) {
                                polylines[i].insert(polylines[i].end(), polylines[j].rbegin() + 1, polylines[j].rend());
                            } else if (samePoint(polylines[i].front(), polylines[j].back())) {
                                polylines[i].insert(polylines[i].begin(), polylines[j].begin(), polylines[j].end() - 1);
                            } else if (samePoint(polylines[i].front(), polylines[j].front())) {
                                polylines[i].insert(polylines[i].begin(), polylines[j].rbegin(), polylines[j].rend() - 1);
                            } else {
                                continue;
                            }
                            polylines.erase(polylines.begin() + static_cast<std::ptrdiff_t>(j));
                            changed = true;
                            break;
                        }
                    }
                }

                auto lineLength = [](const std::vector<cv::Vec3f>& line) {
                    float length = 0.0f;
                    for (std::size_t i = 1; i < line.size(); ++i) {
                        length += cv::norm(line[i] - line[i - 1]);
                    }
                    return length;
                };
                auto bestLine = std::max_element(polylines.begin(), polylines.end(), [&](const auto& lhs, const auto& rhs) {
                    return lineLength(lhs) < lineLength(rhs);
                });
                if (bestLine != polylines.end() && bestLine->size() >= 2) {
                    builder.addLineStrip(volumeToScene(viewer, *bestLine), false, previewStyle);
                }
            }
            ViewerOverlayControllerBase::OverlayStyle constraintStyle;
            constraintStyle.penColor = QColor(255, 255, 120, 255);
            constraintStyle.brushColor = QColor(255, 255, 120, 220);
            constraintStyle.penWidth = 1.0;
            constraintStyle.z = 111.0;
            for (const cv::Vec3f& constraint : state.manualAddPlaneConstraints) {
                if (!plane || plane->pointDist(constraint) <= kManualAddPlaneDistanceTolerance) {
                    builder.addCircle(viewer->volumeToScene(constraint), 5.0, true, constraintStyle);
                }
            }
        }
    }

    // Render approval mask overlays regardless of editing enabled
    // (but painting requires editing to be enabled - handled in SegmentationModule)
    if (state.approvalMaskMode && state.surface) {
        buildApprovalMaskOverlay(state, viewer, builder);
    }

    // Render correction drag line (regardless of editing mode - corrections are annotations)
    if (state.correctionDragActive) {
        const QPointF startScene = viewer->volumeToScene(state.correctionDragStart);
        const QPointF currentScene = viewer->volumeToScene(state.correctionDragCurrent);

        // Draw line from start to current
        ViewerOverlayControllerBase::OverlayStyle lineStyle;
        lineStyle.penColor = QColor(255, 100, 100);  // Red-ish for correction
        lineStyle.penWidth = 3.0;
        lineStyle.z = 100.0;  // Draw on top
        builder.addLineStrip({startScene, currentScene}, false, lineStyle);

        // Draw a small circle at the start (anchor) position
        ViewerOverlayControllerBase::OverlayStyle anchorStyle;
        anchorStyle.penColor = QColor(255, 200, 100);  // Orange for anchor
        anchorStyle.brushColor = QColor(255, 200, 100, 180);
        anchorStyle.penWidth = 2.0;
        anchorStyle.z = 101.0;
        builder.addCircle(startScene, 8.0, true, anchorStyle);

        // Draw a small circle at the current (target) position
        ViewerOverlayControllerBase::OverlayStyle targetStyle;
        targetStyle.penColor = QColor(100, 255, 100);  // Green for target
        targetStyle.brushColor = QColor(100, 255, 100, 180);
        targetStyle.penWidth = 2.0;
        targetStyle.z = 101.0;
        builder.addCircle(currentScene, 8.0, true, targetStyle);
    }

    // Other overlays require editing to be enabled
    if (!_editingEnabled) {
        return;
    }

    if (shouldShowMask(state)) {
        builder.addPath(buildMaskPrimitive(state));
    }

    if (!state.surfaceMaskPoints.empty() && viewer->surfName() == "segmentation") {
        std::vector<QPointF> scenePoints;
        scenePoints.reserve(state.surfaceMaskPoints.size());
        for (const QPointF& surfacePoint : state.surfaceMaskPoints) {
            const QPointF scene = viewer->surfaceCoordsToScene(static_cast<float>(surfacePoint.x()),
                                                               static_cast<float>(surfacePoint.y()));
            if (std::isfinite(scene.x()) && std::isfinite(scene.y())) {
                scenePoints.push_back(scene);
            }
        }

        if (!scenePoints.empty()) {
            ViewerOverlayControllerBase::OverlayStyle style;
            style.penColor = QColor(255, 80, 40, 230);
            style.brushColor = QColor(255, 80, 40, 70);
            style.penWidth = 2.0;
            style.penStyle = Qt::DashLine;
            style.dashPattern = {4.0, 4.0};
            style.z = kMaskZ + 5.0;

            const QPointF center = scenePoints.back();
            const QPointF edge = viewer->surfaceCoordsToScene(
                static_cast<float>(state.surfaceMaskPoints.back().x() + state.displayRadiusSteps),
                static_cast<float>(state.surfaceMaskPoints.back().y()));
            const qreal radiusPixels = std::hypot(edge.x() - center.x(), edge.y() - center.y());
            if (radiusPixels > 1.0) {
                builder.addCircle(center, radiusPixels, false, style);
            }

            if (scenePoints.size() >= 2) {
                style.penWidth = std::max<qreal>(2.0, radiusPixels * 0.35);
                style.penStyle = Qt::SolidLine;
                builder.addLineStrip(scenePoints, false, style);
            }
        }
    }

    buildVertexMarkers(state, viewer, builder);
    buildRadiusOverlay(state, viewer, builder);
}

ViewerOverlayControllerBase::PathPrimitive SegmentationOverlayController::buildMaskPrimitive(const State& state) const
{
    ViewerOverlayControllerBase::PathPrimitive maskPath;
    maskPath.points = state.maskPoints;
    maskPath.renderMode = state.hasLineStroke ? ViewerOverlayControllerBase::PathRenderMode::LineStrip
                                              : ViewerOverlayControllerBase::PathRenderMode::Points;
    maskPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Circle;

    const float brushPixelRadius = std::clamp(state.displayRadiusSteps * 1.5f, 3.0f, 18.0f);
    const bool drawingOverlay = state.brushActive || state.brushStrokeActive || state.lineStrokeActive || state.hasLineStroke;
    const float brushOpacity = drawingOverlay ? 0.6f : 0.45f;

    maskPath.pointRadius = state.hasLineStroke ? std::max(brushPixelRadius * 0.35f, 2.0f) : brushPixelRadius;
    maskPath.lineWidth = state.hasLineStroke ? 3.0f : std::max(brushPixelRadius * 0.5f, 2.0f);
    maskPath.color = state.hasLineStroke ? QColor(80, 170, 255) : QColor(255, 140, 0);
    maskPath.opacity = state.hasLineStroke ? 0.85f : brushOpacity;
    maskPath.z = kMaskZ;

    return maskPath;
}

bool SegmentationOverlayController::shouldShowMask(const State& state) const
{
    return state.maskVisible && !state.maskPoints.empty();
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, std::shared_ptr<Surface> surface)
{
    Q_UNUSED(surface);

    // Surface changing: flush pending approval edits before the pointer dangles.
    flushPendingApprovalMaskSave();

    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::buildRadiusOverlay(const State& state,
                                                       VolumeViewerBase* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.activeMarker || !state.activeMarker->isActive) {
        return;
    }

    const float radiusWorld = state.gaussianRadiusSteps * state.gridStepWorld;
    if (radiusWorld <= 0.0f) {
        return;
    }

    const cv::Vec3f world = state.activeMarker->world;
    const QPointF sceneCenter = viewer->volumeToScene(world);

    cv::Vec3f offsetWorld = world;
    offsetWorld[0] += radiusWorld;
    const QPointF sceneEdge = viewer->volumeToScene(offsetWorld);
    const qreal radiusPixels = std::hypot(sceneEdge.x() - sceneCenter.x(),
                                          sceneEdge.y() - sceneCenter.y());

    if (radiusPixels <= 1.0e-3) {
        return;
    }

    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 255, 255, 120);
    style.penWidth = 1.5f;
    style.brushColor = QColor(Qt::transparent);
    style.z = kRadiusCircleZ;

    builder.addCircle(sceneCenter, radiusPixels, false, style);
}

void SegmentationOverlayController::buildVertexMarkers(const State& state,
                                                       VolumeViewerBase* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    const auto buildStyle = [](const VertexMarker& marker) {
        ViewerOverlayControllerBase::OverlayStyle style;
        style.penWidth = marker.isActive ? kActivePenWidth : kMarkerPenWidth;
        style.z = kMarkerZ;

        if (marker.isActive) {
            style.penColor = kActiveBorder;
            style.brushColor = kActiveFill;
        } else if (marker.isGrowth) {
            style.penColor = kGrowthBorder;
            style.brushColor = kGrowthFill;
        } else {
            style.penColor = kNeighborBorder;
            style.brushColor = kNeighborFill;
        }

        return style;
    };

    const auto appendMarker = [&](VertexMarker marker) {
        const auto directScene = markerGridToScene(viewer, marker);
        const QPointF scene = directScene ? *directScene : viewer->volumeToScene(marker.world);
        const auto style = buildStyle(marker);
        builder.addCircle(scene, kMarkerRadius, true, style);
    };

    for (const auto& marker : state.neighbours) {
        appendMarker(marker);
    }

    if (state.activeMarker) {
        auto active = *state.activeMarker;
        active.isActive = true;
        appendMarker(active);
    }
}

void SegmentationOverlayController::rebuildViewerCache(VolumeViewerBase* viewer, QuadSurface* surface) const
{
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int gridRows = points->rows;
    const int gridCols = points->cols;

    const bool hasSaved = !_savedApprovalMaskImage.isNull() &&
                         _savedApprovalMaskImage.width() == gridCols &&
                         _savedApprovalMaskImage.height() == gridRows;
    const bool hasPending = !_pendingApprovalMaskImage.isNull() &&
                           _pendingApprovalMaskImage.width() == gridCols &&
                           _pendingApprovalMaskImage.height() == gridRows;

    if (!hasSaved && !hasPending) {
        return;
    }

    // Simple approach: composite saved and pending images using QPainter (fast!)
    QImage compositeImage(gridCols, gridRows, QImage::Format_ARGB32_Premultiplied);
    compositeImage.fill(Qt::transparent);

    QPainter painter(&compositeImage);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    // Draw saved (dark green) first
    if (hasSaved) {
        painter.drawImage(0, 0, _savedApprovalMaskImage);
    }

    // Draw pending (light green) on top
    if (hasPending) {
        painter.drawImage(0, 0, _pendingApprovalMaskImage);
    }

    painter.end();

    // Calculate scene-space bounds using direct grid-to-scene formula (no pointTo!)
    // The relationship between scene coords and grid coords:
    // - scene2vol: surfLoc = scenePos / dsScale, then coord() uses (surfLoc + center) * surfScale
    // - So: gridPos = (scenePos/dsScale + center) * surfScale
    // - Inverting: scenePos = (gridPos/surfScale - center) * dsScale
    // BUT the rendering uses viewerScale (_scale), so we need to use that for overlay positioning
    const cv::Vec3f center = surface->center();
    const cv::Vec2f surfScale = surface->scale();
    const float viewerScale = viewer->getCurrentScale();

    // Formula: scenePos = (gridPos/surfScale - center) * viewerScale
    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfLocalX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfLocalY = static_cast<float>(row) / surfScale[1] - center[1];
        const float sceneX = surfLocalX * viewerScale;
        const float sceneY = surfLocalY * viewerScale;
        return QPointF(sceneX, sceneY);
    };

    // Calculate corners directly
    QPointF topLeft = gridToScene(0, 0);
    QPointF topRight = gridToScene(0, gridCols - 1);
    QPointF bottomLeft = gridToScene(gridRows - 1, 0);
    QPointF bottomRight = gridToScene(gridRows - 1, gridCols - 1);

    QRectF sceneBounds;
    sceneBounds.setLeft(std::min({topLeft.x(), topRight.x(), bottomLeft.x(), bottomRight.x()}));
    sceneBounds.setRight(std::max({topLeft.x(), topRight.x(), bottomLeft.x(), bottomRight.x()}));
    sceneBounds.setTop(std::min({topLeft.y(), topRight.y(), bottomLeft.y(), bottomRight.y()}));
    sceneBounds.setBottom(std::max({topLeft.y(), topRight.y(), bottomLeft.y(), bottomRight.y()}));

    // Calculate scale to map from grid-space to scene-space
    const qreal sceneWidth = sceneBounds.width();
    const qreal sceneHeight = sceneBounds.height();
    const qreal scaleX = sceneWidth / static_cast<qreal>(gridCols);
    const qreal scaleY = sceneHeight / static_cast<qreal>(gridRows);
    const qreal scale = std::max(scaleX, scaleY);

    // Store in cache
    ViewerImageCache& cache = _viewerCaches[viewer];
    cache.compositeImage = compositeImage;
    cache.topLeft = sceneBounds.topLeft();
    cache.scale = scale;
    cache.surface = surface;
    cache.savedImageVersion = _savedImageVersion;
    cache.pendingImageVersion = _pendingImageVersion;
}

int SegmentationOverlayController::queryApprovalStatus(int row, int col) const
{
    // Check pending first (takes priority for display)
    // Returns: 0 = not approved, 1 = saved approved, 2 = pending approved
    // Note: Unapprovals are applied immediately (no pending unapproval state)
    if (!_pendingApprovalMaskImage.isNull() &&
        row >= 0 && row < _pendingApprovalMaskImage.height() &&
        col >= 0 && col < _pendingApprovalMaskImage.width()) {
        QRgb pixel = _pendingApprovalMaskImage.pixel(col, row);
        // Any non-transparent pixel with any color is approved
        if (qAlpha(pixel) > 0 && (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0)) {
            return 2;  // Pending approval
        }
    }

    // Check saved
    if (!_savedApprovalMaskImage.isNull() &&
        row >= 0 && row < _savedApprovalMaskImage.height() &&
        col >= 0 && col < _savedApprovalMaskImage.width()) {
        QRgb pixel = _savedApprovalMaskImage.pixel(col, row);
        // Any non-transparent pixel with any color is approved
        if (qAlpha(pixel) > 0 && (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0)) {
            return 1;  // Saved
        }
    }

    return 0;  // Not approved
}

QColor SegmentationOverlayController::queryApprovalColor(int row, int col) const
{
    // Check pending first (takes priority for display)
    if (!_pendingApprovalMaskImage.isNull() &&
        row >= 0 && row < _pendingApprovalMaskImage.height() &&
        col >= 0 && col < _pendingApprovalMaskImage.width()) {
        QRgb pixel = _pendingApprovalMaskImage.pixel(col, row);
        if (qAlpha(pixel) > 0 && (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0)) {
            return QColor(qRed(pixel), qGreen(pixel), qBlue(pixel));
        }
    }

    // Check saved
    if (!_savedApprovalMaskImage.isNull() &&
        row >= 0 && row < _savedApprovalMaskImage.height() &&
        col >= 0 && col < _savedApprovalMaskImage.width()) {
        QRgb pixel = _savedApprovalMaskImage.pixel(col, row);
        if (qAlpha(pixel) > 0 && (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0)) {
            return QColor(qRed(pixel), qGreen(pixel), qBlue(pixel));
        }
    }

    return QColor();  // Invalid color if not approved
}

std::pair<int, int> SegmentationOverlayController::approvalMaskDimensions() const
{
    // Return dimensions from pending mask (preferred) or saved mask
    if (!_pendingApprovalMaskImage.isNull()) {
        return {_pendingApprovalMaskImage.height(), _pendingApprovalMaskImage.width()};
    }
    if (!_savedApprovalMaskImage.isNull()) {
        return {_savedApprovalMaskImage.height(), _savedApprovalMaskImage.width()};
    }
    return {0, 0};
}

float SegmentationOverlayController::sampleImageBilinear(const QImage& image, float row, float col)
{
    if (image.isNull()) {
        return 0.0f;
    }

    const int width = image.width();
    const int height = image.height();

    // Get the four surrounding pixel coordinates
    const int col0 = static_cast<int>(std::floor(col));
    const int col1 = col0 + 1;
    const int row0 = static_cast<int>(std::floor(row));
    const int row1 = row0 + 1;

    // Compute interpolation weights
    const float colFrac = col - static_cast<float>(col0);
    const float rowFrac = row - static_cast<float>(row0);

    // Sample the four corners (clamped to image bounds)
    auto sampleAlpha = [&](int r, int c) -> float {
        if (r < 0 || r >= height || c < 0 || c >= width) {
            return 0.0f;
        }
        return static_cast<float>(qAlpha(image.pixel(c, r)));
    };

    const float v00 = sampleAlpha(row0, col0);
    const float v01 = sampleAlpha(row0, col1);
    const float v10 = sampleAlpha(row1, col0);
    const float v11 = sampleAlpha(row1, col1);

    // Bilinear interpolation
    const float v0 = v00 * (1.0f - colFrac) + v01 * colFrac;
    const float v1 = v10 * (1.0f - colFrac) + v11 * colFrac;
    return v0 * (1.0f - rowFrac) + v1 * rowFrac;
}

float SegmentationOverlayController::queryApprovalBilinear(float row, float col, int* outStatus) const
{
    // First check pending mask (takes priority) - contains colored approvals
    // Note: Unapprovals are applied immediately (no pending unapproval state)
    const float pendingAlpha = sampleImageBilinear(_pendingApprovalMaskImage, row, col);
    if (pendingAlpha > 0.5f) {  // Threshold to determine if we're "in" pending region
        const int nearestRow = static_cast<int>(std::round(row));
        const int nearestCol = static_cast<int>(std::round(col));
        if (!_pendingApprovalMaskImage.isNull() &&
            nearestRow >= 0 && nearestRow < _pendingApprovalMaskImage.height() &&
            nearestCol >= 0 && nearestCol < _pendingApprovalMaskImage.width()) {
            QRgb pixel = _pendingApprovalMaskImage.pixel(nearestCol, nearestRow);
            // Any non-transparent pixel with any color is approved
            if (qAlpha(pixel) > 0 && (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0)) {
                if (outStatus) {
                    *outStatus = 2;  // Pending approval
                }
                return pendingAlpha / 255.0f;
            }
        }
    }

    // Check saved mask with bilinear interpolation
    const float savedAlpha = sampleImageBilinear(_savedApprovalMaskImage, row, col);
    if (savedAlpha > 0.0f) {
        if (outStatus) {
            *outStatus = 1;  // Saved
        }
        return savedAlpha / 255.0f;
    }

    if (outStatus) {
        *outStatus = 0;  // Not approved
    }
    return 0.0f;
}

bool SegmentationOverlayController::hasApprovalMaskData() const
{
    if (!_currentState.has_value() || !_currentState->approvalMaskMode) {
        return false;
    }
    return !_savedApprovalMaskImage.isNull() || !_pendingApprovalMaskImage.isNull();
}

void SegmentationOverlayController::forceRefreshAllOverlays()
{
    // Force rebuild all viewer overlays, bypassing state comparison
    refreshAll();
}

void SegmentationOverlayController::invalidatePlaneIntersections()
{
    OverlayProfileScope profile("invalidatePlaneIntersections", "immediate");
    if (!_viewerManager) {
        return;
    }

    vc3d::segmentation::invalidateApprovalPlaneIntersections(
        _viewerManager->baseViewers(),
        vc3d::segmentation::ApprovalIntersectionRefresh::Immediate);
}

void SegmentationOverlayController::schedulePlaneIntersectionRefresh()
{
    OverlayProfileScope profile("invalidatePlaneIntersections", "deferred");
    if (!_viewerManager) {
        return;
    }

    vc3d::segmentation::invalidateApprovalPlaneIntersections(
        _viewerManager->baseViewers(),
        vc3d::segmentation::ApprovalIntersectionRefresh::Deferred);
}

void SegmentationOverlayController::setApprovalMaskOpacity(int opacity)
{
    const int clamped = std::clamp(opacity, 0, 100);
    if (_approvalMaskOpacity == clamped) {
        return;
    }
    _approvalMaskOpacity = clamped;
    // Trigger redraw by invalidating all viewer caches
    ++_savedImageVersion;
    invalidatePlaneIntersections();
    refreshAll();
}

void SegmentationOverlayController::buildApprovalMaskOverlay(const State& state,
                                                              VolumeViewerBase* viewer,
                                                              ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.surface) {
        return;
    }

    // Check if this viewer is displaying a PlaneSurface (XY/XZ/YZ orthogonal view)
    // For plane viewers, the approval mask is rendered via modified intersection lines
    // in the viewer's intersection renderer, not here.
    Surface* viewerSurf = viewer->currentSurface();
    const bool isPlaneViewer = dynamic_cast<PlaneSurface*>(viewerSurf) != nullptr;

    // Draw brush reticle at hover position
    // Show when we have a hover position (which only exists in edit approval mask mode)
    // Flat cylinder model: circle in XY/XZ/YZ planes and flattened view
    if (state.approvalHoverWorld) {
        const cv::Vec3f& hoverWorld = *state.approvalHoverWorld;
        const float brushRadius = state.approvalBrushRadius;

        if (isPlaneViewer) {
            // For plane viewers: use volumeToScene which is fast (O(1) for PlaneSurface)
            // Compute center and radius consistently using the same projection
            const QPointF sceneCenter = viewer->volumeToScene(hoverWorld);

            // Convert brush radius from native voxels to scene pixels
            // Use both X and Y offsets to handle different plane orientations
            const cv::Vec3f offsetPosX = hoverWorld + cv::Vec3f(brushRadius, 0, 0);
            const cv::Vec3f offsetPosY = hoverWorld + cv::Vec3f(0, brushRadius, 0);
            const QPointF sceneOffsetX = viewer->volumeToScene(offsetPosX);
            const QPointF sceneOffsetY = viewer->volumeToScene(offsetPosY);
            const qreal radiusPixelsX = std::hypot(sceneOffsetX.x() - sceneCenter.x(),
                                                    sceneOffsetX.y() - sceneCenter.y());
            const qreal radiusPixelsY = std::hypot(sceneOffsetY.x() - sceneCenter.x(),
                                                    sceneOffsetY.y() - sceneCenter.y());
            // Use whichever axis projects into the view (the other will be ~0)
            const qreal radiusPixels = std::max(radiusPixelsX, radiusPixelsY);

            if (radiusPixels > 1.0) {
                ViewerOverlayControllerBase::OverlayStyle style;
                style.penColor = state.paintingApproval ? QColor(0, 0, 255) : QColor(255, 0, 0);
                style.penWidth = viewer->intersectionThickness();
                style.penColor.setAlphaF(std::clamp(viewer->intersectionOpacity(), 0.0f, 1.0f));

                style.brushColor = Qt::transparent;
                style.penStyle = Qt::DashLine;
                style.dashPattern = {4.0, 4.0};  // Dashed pattern
                style.z = kApprovalMaskZ + 10.0;

                // XY/XZ/YZ planes: draw a circle (cylinder cross-section)
                builder.addCircle(sceneCenter, radiusPixels, false, style);
            }
        } else {
            // For segmentation/flattened view: draw the same circular 2D brush
            // that is stamped into the approval-mask image.
            const float thisViewerScale = viewer->getCurrentScale();

            QPointF sceneCenter;
            if (state.approvalHoverSurfacePos) {
                sceneCenter = viewer->surfaceCoordsToScene(
                    static_cast<float>(state.approvalHoverSurfacePos->x()),
                    static_cast<float>(state.approvalHoverSurfacePos->y()));
            } else {
                sceneCenter = viewer->volumeToScene(hoverWorld);
            }

            // Flattened-view approval painting is stamped in approval-mask image
            // coordinates, so draw the reticle with the same mask-pixel radius.
            float surfaceScale = 1.0f;
            if (state.surface) {
                const cv::Vec2f scale = state.surface->scale();
                const float avgScale = (std::abs(scale[0]) + std::abs(scale[1])) * 0.5f;
                if (avgScale > 1.0e-6f) {
                    surfaceScale = avgScale;
                }
            }
            const qreal gridToScene = thisViewerScale / surfaceScale;

            // Add a small offset to account for painting extending to cell edges
            constexpr float gridOffset = 0.5f;
            const qreal radiusPixels = (brushRadius + gridOffset) * gridToScene;

            if (radiusPixels > 1.0) {
                ViewerOverlayControllerBase::OverlayStyle style;
                style.penColor = state.paintingApproval ? QColor(0, 0, 255) : QColor(255, 0, 0);
                style.penWidth = viewer->intersectionThickness();
                style.penColor.setAlphaF(std::clamp(viewer->intersectionOpacity(), 0.0f, 1.0f));
                style.brushColor = Qt::transparent;
                style.penStyle = Qt::DashLine;
                style.dashPattern = {4.0, 4.0};  // Dashed pattern
                style.z = kApprovalMaskZ + 10.0;

                builder.addCircle(sceneCenter, radiusPixels, false, style);
            }
        }
    }

    if (isPlaneViewer) {
        return;  // Mask image handled by renderIntersections()
    }

    // For segmentation view (QuadSurface), render as a 2D image overlay in the
    // viewer's current parameterization. The approval image pixels are addressed
    // by the same grid coordinates used by the flattened brush.
    auto* renderSurface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!renderSurface) {
        renderSurface = state.surface;
    }

    // Check if we need to rebuild the COMPOSITE IMAGE (only when masks change, not when view changes)
    auto it = _viewerCaches.find(viewer);
    bool needsRebuild = (it == _viewerCaches.end()) ||
                       (it->second.surface != renderSurface) ||
                       (it->second.savedImageVersion != _savedImageVersion) ||
                       (it->second.pendingImageVersion != _pendingImageVersion);

    if (needsRebuild) {
        rebuildViewerCache(viewer, renderSurface);
        it = _viewerCaches.find(viewer);
    }

    // If cache still doesn't exist or is empty, nothing to render
    if (it == _viewerCaches.end() || it->second.compositeImage.isNull()) {
        return;
    }

    const ViewerImageCache& cache = it->second;

    const cv::Vec2f surfScale = renderSurface->scale();

    // Grid index (row, col) -> surface coords, then to scene coords.
    // Surface space is centered: grid center = (0,0) in surface space.
    // Grid (0,0) maps to (-center.x, -center.y) in surface space.
    const cv::Vec3f center = renderSurface->center();
    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfY = static_cast<float>(row) / surfScale[1] - center[1];
        return viewer->surfaceCoordsToScene(surfX, surfY);
    };

    const QPointF grid00 = gridToScene(0, 0);
    const QPointF grid01 = gridToScene(0, 1);
    const QPointF grid10 = gridToScene(1, 0);
    const QPointF colStep = grid01 - grid00;
    const QPointF rowStep = grid10 - grid00;
    const qreal gridToSceneScaleX = std::hypot(colStep.x(), colStep.y());
    const qreal gridToSceneScaleY = std::hypot(rowStep.x(), rowStep.y());

    if (gridToSceneScaleX < 1e-6 || gridToSceneScaleY < 1e-6) {
        return;
    }

    // Keep this identical in spirit to markerGridToScene(): image pixel
    // (col,row) is drawn at the scene position for grid index (row,col).
    const qreal opacity = static_cast<qreal>(_approvalMaskOpacity) / 100.0;
    builder.addImage(cache.compositeImage, grid00,
                     gridToSceneScaleX, gridToSceneScaleY,
                     opacity, kApprovalMaskZ);
}

void SegmentationOverlayController::buildSurfaceOverlapOverlay(
    VolumeViewerBase* viewer,
    ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !viewer->surfaceOverlayEnabled()) {
        return;
    }

    const auto& overlays = viewer->surfaceOverlays();
    if (overlays.empty()) {
        return;
    }

    // Get the current surface
    Surface* viewerSurf = viewer->currentSurface();
    auto* currentQuad = dynamic_cast<QuadSurface*>(viewerSurf);
    if (!currentQuad) {
        return;
    }

    const float threshold = viewer->surfaceOverlapThreshold();
    if (threshold <= 0.0f || !_state || !_viewerManager) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* currentPts = currentQuad->rawPointsPtr();
    if (!currentPts || currentPts->empty()) {
        return;
    }
    const int rows = currentPts->rows;
    const int cols = currentPts->cols;
    if (rows <= 0 || cols <= 0) {
        return;
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex) {
        return;
    }

    const int sampleRows = overlapSampleCount(rows);
    const int sampleCols = overlapSampleCount(cols);
    if (sampleRows <= 0 || sampleCols <= 0) {
        return;
    }

    // The common interaction path reaches here every overlay refresh. Compare
    // only O(1) revisions before resolving surface IDs or querying the index.
    // SurfacePatchIndex's global generation deliberately over-invalidates when
    // an unrelated surface changes, which is preferable to walking and hashing
    // every selected surface on each pan/zoom tick.
    const std::uint64_t overlaysRevision = viewer->surfaceOverlaysRevision();
    const std::uint64_t indexGeneration = patchIndex->globalGeneration();
    auto cacheIt = _overlapCaches.find(viewer);
    bool needsRebuild = (cacheIt == _overlapCaches.end()) ||
                        (cacheIt->second.surface != currentQuad) ||
                        (cacheIt->second.overlaysRevision != overlaysRevision) ||
                        (cacheIt->second.indexGeneration != indexGeneration) ||
                        (std::abs(cacheIt->second.threshold - threshold) > 0.01f) ||
                        (cacheIt->second.sampleStride != kSurfaceOverlapSampleStride) ||
                        (cacheIt->second.image.width() != sampleCols) ||
                        (cacheIt->second.image.height() != sampleRows);

    if (needsRebuild) {
        std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
        std::unordered_map<QuadSurface*, cv::Vec3b> targetColors;
        targets.reserve(overlays.size());
        targetColors.reserve(overlays.size());

        for (const auto& [surfId, color] : overlays) {
            auto surfBase = _state->surface(surfId);
            auto overlaySurf = std::dynamic_pointer_cast<QuadSurface>(surfBase);
            if (!overlaySurf || overlaySurf.get() == currentQuad ||
                (!overlaySurf->id.empty() && overlaySurf->id == currentQuad->id)) {
                continue;
            }
            if (!patchIndex->containsSurface(overlaySurf)) {
                continue;
            }
            targets.insert(overlaySurf);
            targetColors[overlaySurf.get()] = color;
        }

        if (targets.empty()) {
            OverlapCache emptyCache;
            emptyCache.surface = currentQuad;
            emptyCache.overlaysRevision = overlaysRevision;
            emptyCache.indexGeneration = indexGeneration;
            emptyCache.threshold = threshold;
            emptyCache.sampleStride = kSurfaceOverlapSampleStride;
            _overlapCaches[viewer] = std::move(emptyCache);
            return;
        }

        // If a compute for *different* parameters is still in flight (e.g. the
        // previously shown surface), cancel it so the fresh one can start as soon
        // as the watcher fires instead of waiting out the stale pass.
        if (_overlapComputeRunning && _overlapCancel) {
            const OverlapCache& pending = _pendingOverlap.cache;
            const bool pendingMatches = _pendingOverlap.viewer == viewer &&
                                        pending.surface == currentQuad &&
                                        pending.overlaysRevision == overlaysRevision &&
                                        pending.indexGeneration == indexGeneration &&
                                        std::abs(pending.threshold - threshold) <= 0.01f &&
                                        pending.sampleStride == kSurfaceOverlapSampleStride;
            if (!pendingMatches) {
                _overlapCancel->store(true, std::memory_order_relaxed);
            }
        }

        if (!_overlapComputeRunning) {
            cv::Mat_<cv::Vec3f> sampledPoints(sampleRows, sampleCols);
            for (int sr = 0; sr < sampleRows; ++sr) {
                const int r = overlapSampleIndex(sr, sampleRows, rows);
                for (int sc = 0; sc < sampleCols; ++sc) {
                    const int c = overlapSampleIndex(sc, sampleCols, cols);
                    sampledPoints(sr, sc) = (*currentPts)(r, c);
                }
            }

            // Store cache metadata for when the result arrives
            _pendingOverlap.viewer = viewer;
            _pendingOverlap.cache = OverlapCache{};
            _pendingOverlap.cache.surface = currentQuad;
            _pendingOverlap.cache.overlaysRevision = overlaysRevision;
            _pendingOverlap.cache.indexGeneration = indexGeneration;
            _pendingOverlap.cache.threshold = threshold;
            _pendingOverlap.cache.sampleStride = kSurfaceOverlapSampleStride;

            _overlapComputeRunning = true;
            _overlapIndexReadInFlight = true;
            _viewerManager->beginIndexRead();

            auto cancel = std::make_shared<std::atomic_bool>(false);
            _overlapCancel = cancel;

            auto future = QtConcurrent::run(
                [sampledPoints = std::move(sampledPoints),
                 targets = std::move(targets),
                 targetColors = std::move(targetColors),
                 threshold,
                 patchIndex,
                 cancel]() -> QImage {
                try {
                    QElapsedTimer timer;
                    timer.start();

                    QImage sampledImage(sampledPoints.cols, sampledPoints.rows,
                                        QImage::Format_ARGB32);
                    sampledImage.fill(Qt::transparent);
                    uchar* const bits = sampledImage.bits();
                    const auto bytesPerLine = sampledImage.bytesPerLine();

                    // Raw-pointer include set: locateAny filters against it
                    // without copying (targets keeps the surfaces alive).
                    std::unordered_set<QuadSurface*> targetsRaw;
                    targetsRaw.reserve(targets.size());
                    for (const auto& surf : targets) {
                        targetsRaw.insert(surf.get());
                    }

                    // Explicit fan-out: VC3D pins the implicit pools (OpenCV/
                    // OMP) to one thread, so parallelism must be spawned here.
                    // locateAny() only takes a shared_lock, so rows can be
                    // resolved concurrently; each row owns its scanline.
                    const unsigned hw = std::thread::hardware_concurrency();
                    const std::size_t workerCount = std::max<std::size_t>(
                        1, std::min<std::size_t>(hw == 0 ? 4 : hw,
                                                 static_cast<std::size_t>(sampledPoints.rows)));
                    std::atomic_int nextRow{0};
                    std::vector<std::future<void>> workers;
                    workers.reserve(workerCount);
                    for (std::size_t w = 0; w < workerCount; ++w) {
                        workers.emplace_back(std::async(std::launch::async, [&]() {
                            SurfacePatchIndex::PointQuery query;
                            query.tolerance = threshold;
                            query.surfaces.includeRaw = &targetsRaw;

                            while (true) {
                                const int r = nextRow.fetch_add(1, std::memory_order_relaxed);
                                if (r >= sampledPoints.rows ||
                                    cancel->load(std::memory_order_relaxed)) {
                                    return;
                                }
                                auto* scanline = reinterpret_cast<QRgb*>(bits + r * bytesPerLine);
                                for (int c = 0; c < sampledPoints.cols; ++c) {
                                    const cv::Vec3f& pt = sampledPoints(r, c);
                                    if (!validSurfaceOverlapPoint(pt)) {
                                        continue;
                                    }

                                    query.worldPoint = pt;
                                    auto hit = patchIndex->locateAny(query);
                                    if (!hit || !hit->surface) {
                                        continue;
                                    }
                                    auto colorIt = targetColors.find(hit->surface.get());
                                    if (colorIt == targetColors.end()) {
                                        continue;
                                    }
                                    const cv::Vec3b& hitColor = colorIt->second;
                                    scanline[c] = qRgba(hitColor[2], hitColor[1], hitColor[0], 200);
                                }
                            }
                        }));
                    }
                    for (auto& worker : workers) {
                        worker.get();
                    }

                    const bool canceled = cancel->load(std::memory_order_relaxed);
                    if (ProfileLoggingEnabled()) {
                        Logger()->info("[vc3d-profile] surfaceOverlapCompute elapsed_ms={} "
                                       "size={}x{} targets={} workers={} canceled={}",
                                       timer.elapsed(), sampledPoints.cols, sampledPoints.rows,
                                       targetsRaw.size(), workerCount, canceled);
                    }
                    if (canceled) {
                        return {};  // canceled: result is discarded
                    }
                    return sampledImage;
                } catch (const std::exception& e) {
                    qWarning() << "Segmentation overlap worker failed:" << e.what();
                } catch (...) {
                    qWarning() << "Segmentation overlap worker failed with an unknown exception";
                }
                return {};
            });

            _overlapWatcher->setFuture(future);
        }
    }

    // Render from cache if available (stale data while recomputing is fine)
    if (cacheIt == _overlapCaches.end()) {
        return;
    }

    const OverlapCache& cache = cacheIt->second;
    if (cache.image.isNull()) {
        return;
    }

    // Position using the same coordinate math as the approval mask overlay
    const cv::Vec2f surfScale = currentQuad->scale();
    const cv::Vec3f center = currentQuad->center();

    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfY = static_cast<float>(row) / surfScale[1] - center[1];
        return viewer->surfaceCoordsToScene(surfX, surfY);
    };

    QPointF p0 = gridToScene(0, 0);
    QPointF p1 = gridToScene(0, 1);
    qreal gridToSceneScale = std::hypot(p1.x() - p0.x(), p1.y() - p0.y());

    if (gridToSceneScale < 1e-6) {
        return;
    }

    QPointF topLeft = gridToScene(0, 0);
    builder.addImage(cache.image,
                     topLeft,
                     gridToSceneScale * static_cast<qreal>(std::max(1, cache.sampleStride)),
                     0.6,
                     kSurfaceOverlapZ);
}

void SegmentationOverlayController::handleOverlapComputeFinished()
{
    _overlapComputeRunning = false;
    if (_overlapIndexReadInFlight) {
        if (_viewerManager) {
            _viewerManager->endIndexRead();
        }
        _overlapIndexReadInFlight = false;
    }

    const bool wasCanceled = _overlapCancel &&
                             _overlapCancel->load(std::memory_order_relaxed);
    _overlapCancel.reset();

    if (!_pendingOverlap.viewer) {
        return;  // Viewer was detached while computing
    }

    QImage result = _overlapWatcher->result();
    if (result.isNull() && wasCanceled) {
        // Stale compute was canceled mid-run: don't cache it. refreshAll()
        // re-enters buildSurfaceOverlapOverlay, which starts the fresh compute.
        _pendingOverlap.viewer = nullptr;
        refreshAll();
        return;
    }
    _pendingOverlap.cache.image = std::move(result);
    _overlapCaches[_pendingOverlap.viewer] = std::move(_pendingOverlap.cache);
    _pendingOverlap.viewer = nullptr;

    refreshAll();
}
