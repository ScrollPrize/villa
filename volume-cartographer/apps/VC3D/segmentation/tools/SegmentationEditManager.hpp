#pragma once

#include <QObject>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class QuadSurface;
class ViewerManager;

class SegmentationEditManager : public QObject
{
    Q_OBJECT

public:
    enum class GridSearchResolution
    {
        Low,
        Medium,
        High
    };

    struct GridKey
    {
        int row{0};
        int col{0};

        bool operator==(const GridKey& other) const noexcept
        {
            return row == other.row && col == other.col;
        }
    };

    struct GridKeyHash
    {
        std::size_t operator()(const GridKey& key) const noexcept
        {
            const auto row = static_cast<std::uint32_t>(key.row);
            const auto col = static_cast<std::uint32_t>(key.col);
            return (static_cast<std::size_t>(row) << 32) ^ static_cast<std::size_t>(col);
        }
    };

    struct VertexEdit
    {
        int row{0};
        int col{0};
        cv::Vec3f originalWorld{0.0f, 0.0f, 0.0f};
        cv::Vec3f currentWorld{0.0f, 0.0f, 0.0f};
        bool isGrowth{false};
    };

    struct DragSample
    {
        int row{0};
        int col{0};
        cv::Vec3f baseWorld{0.0f, 0.0f, 0.0f};
        float distanceWorldSq{0.0f};
    };

    struct ActiveDrag
    {
        bool active{false};
        GridKey center{};
        cv::Vec3f baseWorld{0.0f, 0.0f, 0.0f};
        cv::Vec3f targetWorld{0.0f, 0.0f, 0.0f};
        std::vector<DragSample> samples;
    };

    explicit SegmentationEditManager(QObject* parent = nullptr);

    void setViewerManager(ViewerManager* manager) { _viewerManager = manager; }
    [[nodiscard]] ViewerManager* viewerManager() const { return _viewerManager; }

    bool beginSession(std::shared_ptr<QuadSurface> baseSurface);
    void endSession();

    [[nodiscard]] bool hasSession() const { return static_cast<bool>(_baseSurface); }
    [[nodiscard]] std::shared_ptr<QuadSurface> baseSurface() const { return _baseSurface; }
    [[nodiscard]] std::shared_ptr<QuadSurface> previewSurface() const { return _baseSurface; }

    // Synchronize a rectangular region with the latest base-surface data without rebuilding the session.
    bool applyExternalSurfaceUpdate(const cv::Rect& vertexRect);

    void setRadius(float radiusSteps);
    void setSigma(float sigmaSteps);

    [[nodiscard]] float radius() const { return _radiusSteps; }
    [[nodiscard]] float sigma() const { return _sigmaSteps; }

    [[nodiscard]] bool hasPendingChanges() const { return _hasPendingEdits; }
    [[nodiscard]] const cv::Mat_<cv::Vec3f>& previewPoints() const;
    [[nodiscard]] cv::Mat_<cv::Vec3f>& previewPointsMutable();
    bool setPreviewPoints(const cv::Mat_<cv::Vec3f>& points,
                          bool markAsPendingEdit,
                          std::optional<cv::Rect>* outDiffBounds = nullptr);

    void resetPreview();
    void applyPreview();
    void refreshFromBaseSurface();

    std::optional<std::pair<int, int>> worldToGridIndex(const cv::Vec3f& worldPos,
                                                        float* outDistance = nullptr,
                                                        GridSearchResolution detail =
                                                            GridSearchResolution::High) const;
    std::optional<cv::Vec3f> vertexWorldPosition(int row, int col) const;

    bool beginActiveDrag(const std::pair<int, int>& gridIndex);
    bool updateActiveDrag(const cv::Vec3f& newCenterWorld);
    bool updateActiveDragTargets(const std::vector<cv::Vec3f>& newWorldPositions);
    bool smoothRecentTouched(float strength = 0.35f, int iterations = 1);
    void commitActiveDrag();
    void cancelActiveDrag();
    void refreshActiveDragBasePositions();

    [[nodiscard]] const ActiveDrag& activeDrag() const { return _activeDrag; }
    [[nodiscard]] const std::vector<GridKey>& recentTouched() const { return _recentTouched; }
    [[nodiscard]] std::optional<cv::Rect> recentTouchedBounds() const;
    [[nodiscard]] std::vector<VertexEdit> editedVertices() const;

    void markNextEditsAsGrowth();

    void bakePreviewToOriginal();
    bool invalidateRegion(int centerRow, int centerCol, int radius);
    bool markInvalidRegion(int centerRow, int centerCol, float radiusSteps);
    void clearInvalidatedEdits();

    void resetPointerSeed();

private:
    static bool isInvalidPoint(const cv::Vec3f& value);
    void rebuildPreviewFromOriginal();
    bool buildActiveSamples(const std::pair<int, int>& gridIndex);
    void applyGaussianToSamples(const cv::Vec3f& delta);
    void recordVertexEdit(int row, int col, const cv::Vec3f& newWorld);
    void clearActiveDrag();
    float stepNormalization() const;

    std::shared_ptr<QuadSurface> _baseSurface;
    ViewerManager* _viewerManager{nullptr};
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _originalPoints;
    cv::Mat_<cv::Vec3f>* _previewPoints{nullptr};

    float _radiusSteps{3.0f};
    float _sigmaSteps{1.5f};
    bool _hasPendingEdits{false};
    bool _pendingGrowthMarking{false};

    std::unordered_map<GridKey, VertexEdit, GridKeyHash> _editedVertices;
    std::vector<GridKey> _recentTouched;
    ActiveDrag _activeDrag;
    cv::Vec2f _gridScale{1.0f, 1.0f};

    mutable bool _pointerSeedValid{false};
    mutable cv::Vec3f _pointerSeed{0.0f, 0.0f, 0.0f};
};
