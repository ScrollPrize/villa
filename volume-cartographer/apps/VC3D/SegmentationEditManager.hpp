#pragma once

#include <QObject>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "SegmentationInfluenceMode.hpp"

class QuadSurface;
class PlaneSurface;

class SegmentationEditManager : public QObject
{
    Q_OBJECT

public:
    explicit SegmentationEditManager(QObject* parent = nullptr);

    bool beginSession(QuadSurface* baseSurface, int downsample);
    void endSession();

    [[nodiscard]] bool hasSession() const { return static_cast<bool>(_baseSurface); }
    [[nodiscard]] QuadSurface* baseSurface() const { return _baseSurface; }
    [[nodiscard]] QuadSurface* previewSurface() const { return _previewSurface.get(); }

    void setDownsample(int value);
    void setRadius(float radius);
    void setSigma(float sigma);
    void setInfluenceMode(SegmentationInfluenceMode mode);
    void setRowColMode(SegmentationRowColMode mode);
    void setFillInvalidCells(bool enabled);
    [[nodiscard]] bool fillInvalidCells() const { return _fillInvalidCells; }
    [[nodiscard]] int downsample() const { return _downsample; }
    [[nodiscard]] float radius() const { return _radius; }
    [[nodiscard]] float sigma() const { return _sigma; }
    [[nodiscard]] SegmentationInfluenceMode influenceMode() const { return _influenceMode; }
    [[nodiscard]] SegmentationRowColMode rowColMode() const { return _rowColMode; }
    void setHoleSearchRadius(int radius);
    void setHoleSmoothIterations(int iterations);
    [[nodiscard]] int holeSearchRadius() const { return _holeSearchRadius; }
    [[nodiscard]] int holeSmoothIterations() const { return _holeSmoothIterations; }
    [[nodiscard]] bool hasPendingChanges() const { return _dirty; }
    [[nodiscard]] const cv::Mat_<cv::Vec3f>& previewPoints() const;
    bool setPreviewPoints(const cv::Mat_<cv::Vec3f>& points,
                          bool dirtyState);
    bool invalidateRegion(int centerRow, int centerCol, int radius);

    void resetPreview();
    void applyPreview();
    void refreshFromBaseSurface();

    struct Handle {
        int row;
        int col;
        cv::Vec3f originalWorld;
        cv::Vec3f currentWorld;
        bool isManual{false};
        SegmentationRowColAxis rowColAxis{SegmentationRowColAxis::Both};
        bool isGrowth{false};
    };

    [[nodiscard]] const std::vector<Handle>& handles() const { return _handles; }
    bool updateHandleWorldPosition(int row,
                                   int col,
                                   const cv::Vec3f& newWorldPos,
                                   std::optional<SegmentationRowColAxis> axisHint = std::nullopt);
    Handle* findNearestHandle(const cv::Vec3f& world, float tolerance);
    void bakePreviewToOriginal();
    std::optional<std::pair<int,int>> addHandleAtWorld(const cv::Vec3f& worldPos,
                                                       float tolerance = 40.0f,
                                                       PlaneSurface* plane = nullptr,
                                                       float planeTolerance = 0.0f,
                                                       bool allowCreate = false,
                                                       bool allowReuse = false,
                                                       std::optional<SegmentationRowColAxis> axisHint = std::nullopt);
    bool removeHandle(int row, int col);
    std::optional<cv::Vec3f> handleWorldPosition(int row, int col) const;
    std::optional<std::pair<int,int>> worldToGridIndex(const cv::Vec3f& worldPos, float* outDistance = nullptr) const;
    void markNextRefreshHandlesAsGrowth();

private:
    void regenerateHandles();
    void applyHandleInfluence(const Handle& handle);
    void applyHandleInfluenceGrid(const Handle& handle, const cv::Vec3f& delta);
    void applyHandleInfluenceGeodesic(const Handle& handle, const cv::Vec3f& delta);
    void applyHandleInfluenceRowCol(const Handle& handle, const cv::Vec3f& delta);
    Handle* findHandle(int row, int col);
    void syncPreviewFromBase();
    void reapplyAllHandles();
    [[nodiscard]] float estimateGridStepWorld() const;
    bool fillInvalidCellWithLocalSolve(const cv::Vec3f& worldPos,
                                       int seedRow,
                                       int seedCol,
                                       std::pair<int,int>& outCell,
                                       cv::Vec3f* outWorld = nullptr);
    std::vector<std::pair<int,int>> collectHoleCells(int centerRow, int centerCol, int radius) const;
    void relaxHolePatch(const std::vector<std::pair<int,int>>& holeCells,
                        const std::pair<int,int>& seedCell,
                        const cv::Vec3f& seedWorld);
    QuadSurface* _baseSurface{nullptr};
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _originalPoints;
    cv::Mat_<cv::Vec3f>* _previewPoints{nullptr};
    std::unique_ptr<QuadSurface> _previewSurface;
    std::vector<Handle> _handles;
    int _downsample{12};
    float _radius{1.0f};          // radius expressed in grid steps (Chebyshev distance)
    float _sigma{1.0f};           // strength multiplier applied to neighbouring grid points
    SegmentationInfluenceMode _influenceMode{SegmentationInfluenceMode::GridChebyshev};
    SegmentationRowColMode _rowColMode{SegmentationRowColMode::Dynamic};
    int _holeSearchRadius{6};
    int _holeSmoothIterations{25};
    bool _dirty{false};
    bool _fillInvalidCells{true};
    bool _pendingGrowthMarking{false};
    std::vector<cv::Vec3f> _autoHandleWorldSnapshot;
    std::vector<cv::Vec3f> _growthHandleWorld;
};
