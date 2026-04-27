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
    [[nodiscard]] const std::vector<cv::Vec3f>& overlayPoints() const { return _overlayPoints; }

    void startStroke(const QPointF& surfacePos);
    void extendStroke(const QPointF& surfacePos, bool forceSample);
    void finishStroke();
    void cancelStroke();

private:
    [[nodiscard]] std::optional<std::pair<int, int>> surfaceToGridIndex(const QPointF& surfacePos) const;
    void ensureMask();
    void paintAt(int row, int col);
    void addOverlayPoint(int row, int col);
    void persistMask();
    void invalidateViewers();

    SegmentationModule& _module;
    QuadSurface* _surface{nullptr};
    cv::Mat_<uint8_t> _mask;
    bool _active{false};
    bool _strokeActive{false};
    std::optional<std::pair<int, int>> _lastGrid;
    std::unordered_set<uint64_t> _paintedCells;
    std::vector<cv::Vec3f> _overlayPoints;
};
