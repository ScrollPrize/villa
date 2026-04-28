#pragma once

#include <QPointF>
#include <opencv2/core.hpp>

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
    void pauseStroke();
    void finishStroke();
    void cancelStroke();

private:
    [[nodiscard]] std::optional<std::pair<int, int>> surfaceToGridIndex(const QPointF& surfacePos) const;
    void ensureMask();
    void paintAt(int row, int col);
    void queueVertex(int row, int col);
    void fillEnclosedStrokeArea();
    void persistSurface();
    [[nodiscard]] cv::Rect applyPendingCells();
    void refreshSurfacePatchIndex(const cv::Rect& changedRegion);
    void invalidateViewers();

    SegmentationModule& _module;
    QuadSurface* _surface{nullptr};
    cv::Mat_<uint8_t> _mask;
    bool _active{false};
    bool _strokeActive{false};
    std::optional<std::pair<int, int>> _lastGrid;
    std::unordered_set<uint64_t> _paintedCells;
    std::vector<std::pair<int, int>> _pendingCells;
    std::vector<std::pair<int, int>> _strokeGridPoints;
    std::vector<QPointF> _overlaySurfacePoints;
};
