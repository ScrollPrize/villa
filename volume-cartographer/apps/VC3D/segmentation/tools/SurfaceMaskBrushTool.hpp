#pragma once

#include <QPointF>
#include <opencv2/core/types.hpp>

#include <optional>
#include <unordered_set>
#include <vector>

class QuadSurface;
class SegmentationModule;

class SurfaceMaskBrushTool
{
public:
    explicit SurfaceMaskBrushTool(SegmentationModule& module);

    void setSurface(QuadSurface* surface);
    void setActive(bool active);
    [[nodiscard]] bool active() const { return _active; }
    [[nodiscard]] bool strokeActive() const { return _strokeActive; }
    [[nodiscard]] bool hasPendingStroke() const { return !_pendingCells.empty(); }
    [[nodiscard]] const std::vector<QPointF>& overlaySurfacePoints() const { return _overlaySurfacePoints; }

    void startStroke(const QPointF& surfacePos);
    void extendStroke(const QPointF& surfacePos, bool forceSample);
    void finishStroke();
    void cancelStroke();
    void refreshFromSurface();

private:
    [[nodiscard]] std::optional<std::pair<float, float>> surfaceToGridPosition(const QPointF& surfacePos) const;
    void paintAt(int row, int col);
    void appendOverlayPoint(const QPointF& surfacePos);
    void queueVertex(int row, int col);
    void fillEnclosedStrokeArea();
    void persistSurface();
    void invalidateOverlay();
    void invalidateViewers(bool surfaceChanged);

    SegmentationModule& _module;
    QuadSurface* _surface{nullptr};
    bool _active{false};
    bool _strokeActive{false};
    bool _undoSnapshotCaptured{false};
    std::optional<std::pair<float, float>> _lastGridPosition;
    std::unordered_set<uint64_t> _paintedCells;
    std::vector<cv::Point> _pendingCells;
    std::vector<std::pair<int, int>> _strokeGridPoints;
    std::vector<QPointF> _overlaySurfacePoints;
};
