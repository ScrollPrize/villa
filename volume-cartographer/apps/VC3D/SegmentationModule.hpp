#pragma once

#include <QObject>
#include <QPointF>
#include <QCursor>

#include <opencv2/core.hpp>

#include "SegmentationInfluenceMode.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationGrowth.hpp"

#include <functional>
#include <optional>
#include <unordered_set>
#include <vector>

class SegmentationWidget;
class SegmentationEditManager;
class SegmentationOverlayController;
class ViewerManager;
class CSurfaceCollection;
class CVolumeViewer;
class QKeyEvent;
class QuadSurface;
class PlaneSurface;
class VCCollection;

class SegmentationModule : public QObject
{
    Q_OBJECT

public:
    SegmentationModule(SegmentationWidget* widget,
                       SegmentationEditManager* editManager,
                       SegmentationOverlayController* overlay,
                       ViewerManager* viewerManager,
                       CSurfaceCollection* surfaces,
                       VCCollection* pointCollection,
                       bool editingEnabled,
                       int downsample,
                       float radius,
                       float sigma,
                       QObject* parent = nullptr);

    [[nodiscard]] bool editingEnabled() const { return _editingEnabled; }
    [[nodiscard]] bool pointAddModeEnabled() const { return _pointAddMode; }
    [[nodiscard]] int downsample() const { return _downsample; }
    [[nodiscard]] float radius() const { return _radius; }
    [[nodiscard]] float sigma() const { return _sigma; }
    [[nodiscard]] SegmentationInfluenceMode influenceMode() const { return _influenceMode; }
    [[nodiscard]] float sliceFadeDistance() const { return _sliceFadeDistance; }
    [[nodiscard]] SegmentationSliceDisplayMode sliceDisplayMode() const { return _sliceDisplayMode; }
    [[nodiscard]] SegmentationRowColMode rowColMode() const { return _rowColMode; }
    [[nodiscard]] int holeSearchRadius() const { return _holeSearchRadius; }
    [[nodiscard]] int holeSmoothIterations() const { return _holeSmoothIterations; }
    [[nodiscard]] bool handlesAlwaysVisible() const { return _showHandlesAlways; }
    [[nodiscard]] float handleDisplayDistance() const { return _handleDisplayDistance; }
    [[nodiscard]] float highlightDistance() const { return _highlightDistance; }
    [[nodiscard]] bool fillInvalidRegions() const { return _fillInvalidRegions; }

    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float radius);
    void setSigma(float sigma);
    void setInfluenceMode(SegmentationInfluenceMode mode);
    void setSliceFadeDistance(float distance);
    void setSliceDisplayMode(SegmentationSliceDisplayMode mode);
    void setRowColMode(SegmentationRowColMode mode);
    void setHoleSearchRadius(int radius);
    void setHoleSmoothIterations(int iterations);
    void setHandlesAlwaysVisible(bool value);
    void setHandleDisplayDistance(float distance);
    void setHighlightDistance(float distance);
    void setFillInvalidRegions(bool enabled);
    void setGrowthInProgress(bool running);

    void applyEdits();
    void resetEdits();
    void stopTools();

    void attachViewer(CVolumeViewer* viewer);
    void updateViewerCursors();
    void setPointAddMode(bool enabled, bool silent = false);
    void togglePointAddMode();

    bool handleKeyPress(QKeyEvent* event);

    bool beginEditingSession(QuadSurface* activeSurface);
    void endEditingSession();
    bool hasActiveSession() const;
    QuadSurface* activeBaseSurface() const;
    void refreshSessionFromSurface(QuadSurface* surface);
    void markNextHandlesFromGrowth();
    [[nodiscard]] bool hasPendingCorrections() const { return !_pendingCorrectionIds.empty(); }
    [[nodiscard]] SegmentationCorrectionsPayload buildCorrectionsPayload() const;
    void clearPendingCorrections();
    void setHandlesLocked(bool locked, bool userInitiated = false);
    [[nodiscard]] bool handlesLocked() const { return _handlesLocked; }
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

signals:
    void editingEnabledChanged(bool enabled);
    void statusMessageRequested(const QString& text, int timeoutMs);
    void pendingChangesChanged(bool pending);
    void stopToolsRequested();
    void focusPoiRequested(const cv::Vec3f& position, QuadSurface* surface);
    void growSurfaceRequested(SegmentationGrowthMethod method,
                              SegmentationGrowthDirection direction,
                              int steps);

private:
    struct DragState {
        bool active{false};
        int row{0};
        int col{0};
        CVolumeViewer* viewer{nullptr};
        cv::Vec3f startWorld{0, 0, 0};
        bool moved{false};
        void reset()
        {
            active = false;
            row = 0;
            col = 0;
            viewer = nullptr;
            startWorld = {0, 0, 0};
            moved = false;
        }
    };

    struct HoverState {
        bool valid{false};
        int row{0};
        int col{0};
        cv::Vec3f handleWorld{0, 0, 0};
        void set(int r, int c, const cv::Vec3f& world)
        {
            valid = true;
            row = r;
            col = c;
            handleWorld = world;
        }
        void clear()
        {
            valid = false;
        }
    };

    void bindWidgetSignals();
    void handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                    SegmentationGrowthDirection direction,
                                    int steps);
    void bindViewerSignals(CVolumeViewer* viewer);
    void refreshOverlay();
    void emitPendingChanges();
    void resetOverlayHandles();
    void resetInteractionState();
    [[nodiscard]] float gridStepWorld() const;
    [[nodiscard]] float radiusWorldExtent(float gridRadius) const;
    void showRadiusIndicator(CVolumeViewer* viewer,
                             const QPointF& scenePoint,
                             float radius);
    void handleMousePress(CVolumeViewer* viewer,
                          const cv::Vec3f& worldPos,
                          const cv::Vec3f& normal,
                          Qt::MouseButton button,
                          Qt::KeyboardModifiers modifiers);
    void handleMouseMove(CVolumeViewer* viewer,
                         const cv::Vec3f& worldPos,
                         Qt::MouseButtons buttons,
                         Qt::KeyboardModifiers modifiers);
    void handleMouseRelease(CVolumeViewer* viewer,
                            const cv::Vec3f& worldPos,
                            Qt::MouseButton button,
                            Qt::KeyboardModifiers modifiers);
    void handleRadiusWheel(CVolumeViewer* viewer,
                           int steps,
                           const QPointF& scenePoint,
                           const cv::Vec3f& worldPos);
    const QCursor& addCursor();
    [[nodiscard]] bool isSegmentationViewer(const CVolumeViewer* viewer) const;
    struct PullResult
    {
        bool moved{false};
        int candidateCount{0};
        int appliedCount{0};
        float averageWeight{0.0f};
    };
    [[nodiscard]] const SegmentationEditManager::Handle* screenClosestHandle(CVolumeViewer* viewer,
                                                                            const cv::Vec3f& referenceWorld,
                                                                            float maxScreenDistance) const;
    PullResult pullHandlesTowards(const cv::Vec3f& worldPos,
                                  std::optional<SegmentationRowColAxis> axis);
    [[nodiscard]] SegmentationRowColAxis rowColAxisForViewer(const CVolumeViewer* viewer) const;
    void updateCorrectionsWidget();
    void setCorrectionsAnnotateMode(bool enabled, bool userInitiated);
    void setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated);
    uint64_t createCorrectionCollection(bool announce = true);
    void handleCorrectionPointAdded(const cv::Vec3f& worldPos);
    void handleCorrectionPointRemove(const cv::Vec3f& worldPos);
    void pruneMissingCorrections();
    void onGrowthMethodChanged(SegmentationGrowthMethod method);
    void updateHandleVisibility();
    void onCorrectionsCreateRequested();
    void onCorrectionsCollectionSelected(uint64_t id);
    void onCorrectionsAnnotateToggled(bool enabled);
    void onCorrectionsZRangeChanged(bool enabled, int zMin, int zMax);
    void onMaskEditingToggled(bool active);
    void onMaskApplyRequested();
    void setMaskSampling(int step);
    void setMaskBrushRadius(int radius);
    bool beginMaskEditing();
    void cancelMaskEditing(bool silent = false);
    void handleMaskErase(const cv::Vec3f& worldPos);
    void updateMaskOverlay();
    void clearMaskOverlay();
    void updateMaskUi();
    [[nodiscard]] std::optional<cv::Vec3f> maskWorldForCell(int row,
                                                            int col,
                                                            const cv::Mat_<cv::Vec3f>& preview) const;
    [[nodiscard]] static bool maskCellInvalid(const cv::Vec3f& point);

    SegmentationWidget* _widget{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    ViewerManager* _viewerManager{nullptr};
    CSurfaceCollection* _surfaces{nullptr};
    VCCollection* _pointCollection{nullptr};

    bool _editingEnabled{false};
    int _downsample{12};
    float _radius{1.0f};
    float _sigma{1.0f};
    SegmentationInfluenceMode _influenceMode{SegmentationInfluenceMode::GridChebyshev};
    float _sliceFadeDistance{10.0f};
    SegmentationSliceDisplayMode _sliceDisplayMode{SegmentationSliceDisplayMode::Fade};
    SegmentationRowColMode _rowColMode{SegmentationRowColMode::Dynamic};
    int _holeSearchRadius{6};
    int _holeSmoothIterations{25};
    bool _showHandlesAlways{true};
    float _handleDisplayDistance{25.0f};
    float _highlightDistance{15.0f};
    bool _fillInvalidRegions{true};
    bool _correctionsAnnotateMode{false};
    bool _usingCorrectionsGrowth{false};
    uint64_t _activeCorrectionId{0};
    std::vector<uint64_t> _pendingCorrectionIds;
    std::unordered_set<uint64_t> _managedCorrectionIds;
    bool _handlesLocked{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};

    bool _pointAddMode{false};
    DragState _drag;
    HoverState _hover;
    cv::Vec3f _cursorWorld{0, 0, 0};
    bool _cursorValid{false};
    bool _growthInProgress{false};
    CVolumeViewer* _cursorViewer{nullptr};
    bool _maskEditingActive{false};
    bool _maskDirty{false};
    bool _maskOriginalDirty{false};
    bool _maskHasOriginal{false};
    bool _maskStrokeActive{false};
    cv::Mat_<cv::Vec3f> _maskOriginalPoints;
    std::vector<cv::Vec3f> _maskOverlayPoints;
    float _maskOverlayRadius{4.0f};
    int _maskSamplingStep{2};
    int _maskBrushRadius{3};
    std::optional<std::pair<int, int>> _maskLastCell;
};
