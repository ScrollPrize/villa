#pragma once

#include <QObject>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <vc_object_store/objects/ObjectStoreSession.hpp>

class QuadSurface;
class ViewerManager;

class SegmentationEditManager : public QObject
{
    Q_OBJECT

public:
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
        float gaussianWeight{1.0f};
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

    // The edited surface's draft in the store session (vc::objects drafts):
    // the coordinates its points are in and the attachment it is edited in.
    // Its object is the surface's id.
    struct DraftTarget {
        std::shared_ptr<vc::objects::ObjectStoreSession> session;
        vc::objects::CoordinateJson coordinates;
        vc::objects::AttachmentIdentity attachment;
    };
    // Edits keep the draft current (the open surface's, kept by CState).
    bool beginSession(std::shared_ptr<QuadSurface> baseSurface, DraftTarget draft);
    void endSession();

    [[nodiscard]] bool hasSession() const { return static_cast<bool>(_baseSurface); }
    [[nodiscard]] std::shared_ptr<QuadSurface> baseSurface() const { return _baseSurface; }
    [[nodiscard]] std::shared_ptr<QuadSurface> previewSurface() const { return _baseSurface; }
    // For changes made to the preview surface outside this manager: publish
    // `vertices` (grid rows/columns) of it, or the whole grid (its size or
    // validity changed), to the draft.
    void publishDraft(const cv::Rect& vertices);
    void publishWholeDraft();

    // Synchronize a rectangular region with the latest base-surface data without rebuilding the session.
    bool applyExternalSurfaceUpdate(const cv::Rect& vertexRect);

    void setRadius(float radiusSteps);
    void setSigma(float sigmaSteps);
    void setEditScale(float scale);

    [[nodiscard]] float radius() const { return _radiusSteps; }
    [[nodiscard]] float sigma() const { return _sigmaSteps; }
    [[nodiscard]] float editScale() const { return _editScale; }

    [[nodiscard]] bool hasPendingChanges() const { return _hasPendingEdits; }
    [[nodiscard]] const cv::Mat_<cv::Vec3f>& previewPoints() const;
    [[nodiscard]] cv::Mat_<cv::Vec3f>& previewPointsMutable();
    bool setPreviewPoints(const cv::Mat_<cv::Vec3f>& points,
                          bool markAsPendingEdit,
                          std::optional<cv::Rect>* outDiffBounds = nullptr);
    bool setPreviewPointsOnly(const cv::Mat_<cv::Vec3f>& points,
                              const std::vector<GridKey>& editedVertices,
                              bool markAsPendingEdit,
                              std::optional<cv::Rect>* outDiffBounds = nullptr);
    bool restorePreviewSnapshot(const cv::Mat_<cv::Vec3f>& points);

    void resetPreview();
    void applyPreview();
    void refreshFromBaseSurface();

    std::optional<std::pair<int, int>> worldToGridIndex(const cv::Vec3f& worldPos,
                                                        float* outDistance = nullptr,
                                                        bool warnOnFailure = true) const;
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
    [[nodiscard]] std::vector<VertexEdit> editedVertices() const;

    void markNextEditsAsGrowth();

    void clearInvalidatedEdits();

    void resetPointerSeed();

private:
    void rebuildPreviewFromOriginal();
    bool buildActiveSamples(const std::pair<int, int>& gridIndex);
    void applyGaussianToSamples(const cv::Vec3f& delta);
    void recordVertexEdit(int row, int col, const cv::Vec3f& newWorld, bool markDraftVertex = true);
    void clearActiveDrag();
    float stepNormalization() const;
    cv::Mat_<cv::Vec3f>* previewPointsPtr();
    const cv::Mat_<cv::Vec3f>* previewPointsPtr() const;
    // Vertices changed since the draft was last published.
    void markDraft(int minRow, int maxRow, int minCol, int maxCol);
    void flushDraft();

    std::shared_ptr<QuadSurface> _baseSurface;
    ViewerManager* _viewerManager{nullptr};
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _originalPoints;
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _preResizeOriginalPoints;
    std::optional<DraftTarget> _draft;
    std::optional<cv::Rect> _draftDirty;

    float _radiusSteps{3.0f};
    float _sigmaSteps{1.5f};
    float _editScale{1.0f};
    bool _hasPendingEdits{false};
    bool _pendingGrowthMarking{false};

    std::unordered_map<GridKey, VertexEdit, GridKeyHash> _editedVertices;
    std::vector<GridKey> _recentTouched;
    ActiveDrag _activeDrag;
    cv::Vec2f _gridScale{1.0f, 1.0f};

    mutable bool _pointerSeedValid{false};
    mutable cv::Vec3f _pointerSeed{0.0f, 0.0f, 0.0f};
};
