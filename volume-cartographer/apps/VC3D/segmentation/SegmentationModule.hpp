#pragma once

#include <QObject>
#include <QPointer>
#include <QSet>
#include <QLoggingCategory>

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "SegmentationGrowth.hpp"
#include "SegmentationPushPullConfig.hpp"
#include "SegmentationUndoHistory.hpp"

namespace segmentation { class CorrectionsState; }

Q_DECLARE_LOGGING_CATEGORY(lcSegModule);

inline constexpr int kStatusShort = 1500;
inline constexpr int kStatusMedium = 2000;
inline constexpr int kStatusLong = 5000;


class CSurfaceCollection;
class CVolumeViewer;
class PlaneSurface;
class Surface;
class QuadSurface;
class SegmentationEditManager;
class SegmentationOverlayController;
class SegmentationWidget;
class VCCollection;
class ViewerManager;
class QKeyEvent;
class SegmentationBrushTool;
class SegmentationLineTool;
class SegmentationPushPullTool;

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
                       QObject* parent = nullptr);
    ~SegmentationModule();

    [[nodiscard]] bool editingEnabled() const { return _editingEnabled; }
    void setEditingEnabled(bool enabled);
    void setDragRadius(float radiusSteps);
    void setDragSigma(float sigmaSteps);
    void setLineRadius(float radiusSteps);
    void setLineSigma(float sigmaSteps);
    void setPushPullRadius(float radiusSteps);
    void setPushPullSigma(float sigmaSteps);
    void setPushPullStepMultiplier(float multiplier);
    void setSmoothingStrength(float strength);
    void setSmoothingIterations(int iterations);
    void setAlphaPushPullConfig(const AlphaPushPullConfig& config);

    void applyEdits();
    void resetEdits();
    void stopTools();

    bool beginEditingSession(QuadSurface* surface);
    void endEditingSession();
    [[nodiscard]] bool hasActiveSession() const;
    [[nodiscard]] QuadSurface* activeBaseSurface() const;
    void refreshSessionFromSurface(QuadSurface* surface);

    void attachViewer(CVolumeViewer* viewer);
    void updateViewerCursors();

    bool handleKeyPress(QKeyEvent* event);
    bool handleKeyRelease(QKeyEvent* event);

    [[nodiscard]] std::optional<std::vector<SegmentationGrowthDirection>> takeShortcutDirectionOverride();

    void markNextEditsFromGrowth();
    void markNextHandlesFromGrowth() { markNextEditsFromGrowth(); }
    void setGrowthInProgress(bool running);
    [[nodiscard]] bool growthInProgress() const { return _growthInProgress; }
    [[nodiscard]] SegmentationCorrectionsPayload buildCorrectionsPayload() const;
    void clearPendingCorrections();
    [[nodiscard]] std::optional<std::pair<int, int>> correctionsZRange() const;

    struct HoverInfo
    {
        bool valid{false};
        int row{0};
        int col{0};
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
        CVolumeViewer* viewer{nullptr};
    };

    [[nodiscard]] HoverInfo hoverInfo() const;
    [[nodiscard]] bool isSegmentationViewer(const CVolumeViewer* viewer) const;

    void setRotationHandleHitTester(std::function<bool(CVolumeViewer*, const cv::Vec3f&)> tester);

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
    friend class SegmentationBrushTool;
    friend class SegmentationLineTool;
    friend class SegmentationPushPullTool;
    friend class segmentation::CorrectionsState;

    enum class FalloffTool
    {
        Drag,
        Line,
        PushPull
    };

    struct DragState
    {
        bool active{false};
        int row{0};
        int col{0};
        cv::Vec3f startWorld{0.0f, 0.0f, 0.0f};
        cv::Vec3f lastWorld{0.0f, 0.0f, 0.0f};
        QPointer<CVolumeViewer> viewer;
        bool moved{false};

        void reset();
    };

    struct HoverState
    {
        bool valid{false};
        int row{0};
        int col{0};
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
        QPointer<CVolumeViewer> viewer;

        void set(int r, int c, const cv::Vec3f& w, CVolumeViewer* v);
        void clear();
    };

    void bindWidgetSignals();
    void bindViewerSignals(CVolumeViewer* viewer);

    void emitPendingChanges();
    void refreshOverlay();
    void updateCorrectionsWidget();
    void setCorrectionsAnnotateMode(bool enabled, bool userInitiated);
    void setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated);
    uint64_t createCorrectionCollection(bool announce);
    void handleCorrectionPointAdded(const cv::Vec3f& worldPos);
    void handleCorrectionPointRemove(const cv::Vec3f& worldPos);
    void pruneMissingCorrections();
    void onCorrectionsCreateRequested();
    void onCorrectionsCollectionSelected(uint64_t id);
    void onCorrectionsAnnotateToggled(bool enabled);
    void onCorrectionsZRangeChanged(bool enabled, int zMin, int zMax);

    void handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                    SegmentationGrowthDirection direction,
                                    int steps);
    void setInvalidationBrushActive(bool active);
    void clearInvalidationBrush();
    void deactivateInvalidationBrush();
    void clearLineDragStroke();

    void handleMousePress(CVolumeViewer* viewer,
                          const cv::Vec3f& worldPos,
                          const cv::Vec3f& surfaceNormal,
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
    void handleWheel(CVolumeViewer* viewer,
                     int deltaSteps,
                     const QPointF& scenePos,
                     const cv::Vec3f& worldPos);
    void onSurfaceCollectionChanged(std::string name, Surface* surface);

    [[nodiscard]] bool captureUndoSnapshot();
    void discardLastUndoSnapshot();
    bool restoreUndoSnapshot();
    void clearUndoStack();

    [[nodiscard]] float gridStepWorld() const;

    void useFalloff(FalloffTool tool);
    void updateOverlayFalloff(FalloffTool tool);
    [[nodiscard]] float falloffRadius(FalloffTool tool) const;
    [[nodiscard]] float falloffSigma(FalloffTool tool) const;
    void beginDrag(int row, int col, CVolumeViewer* viewer, const cv::Vec3f& worldPos);
    void updateDrag(const cv::Vec3f& worldPos);
    void finishDrag();
    void cancelDrag();

    void updateHover(CVolumeViewer* viewer, const cv::Vec3f& worldPos);
    [[nodiscard]] bool isNearRotationHandle(CVolumeViewer* viewer, const cv::Vec3f& worldPos) const;

    bool startPushPull(int direction, std::optional<bool> alphaOverride = std::nullopt);
    void stopPushPull(int direction);
    void stopAllPushPull();
    bool applyPushPullStep();

    SegmentationWidget* _widget{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    ViewerManager* _viewerManager{nullptr};
    CSurfaceCollection* _surfaces{nullptr};
    VCCollection* _pointCollection{nullptr};

    bool _editingEnabled{false};
    float _dragRadiusSteps{5.75f};
    float _dragSigmaSteps{2.0f};
    float _lineRadiusSteps{5.75f};
    float _lineSigmaSteps{2.0f};
    float _pushPullRadiusSteps{5.75f};
    float _pushPullSigmaSteps{2.0f};
    FalloffTool _activeFalloff{FalloffTool::Drag};
    float _smoothStrength{0.4f};
    int _smoothIterations{2};
    bool _growthInProgress{false};
    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Tracer};
    int _growthSteps{10};
    bool _ignoreSegSurfaceChange{false};

    std::unique_ptr<segmentation::CorrectionsState> _corrections;

    DragState _drag;
    HoverState _hover;
    QSet<CVolumeViewer*> _attachedViewers;

    std::function<bool(CVolumeViewer*, const cv::Vec3f&)> _rotationHandleHitTester;

    bool _lineDrawKeyActive{false};
    std::optional<std::vector<SegmentationGrowthDirection>> _pendingShortcutDirections;

    std::unique_ptr<SegmentationBrushTool> _brushTool;
    std::unique_ptr<SegmentationLineTool> _lineTool;
    std::unique_ptr<SegmentationPushPullTool> _pushPullTool;

    segmentation::UndoHistory _undoHistory;
    bool _suppressUndoCapture{false};
};
