#pragma once

#include <QObject>
#include <QPointer>
#include <QSet>

#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include <opencv2/core.hpp>

#include "SegmentationGrowth.hpp"

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
class QTimer;

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
                       float radiusSteps,
                       float sigmaSteps,
                       QObject* parent = nullptr);

    [[nodiscard]] bool editingEnabled() const { return _editingEnabled; }
    [[nodiscard]] float radius() const { return _radiusSteps; }
    [[nodiscard]] float sigma() const { return _sigmaSteps; }

    void setEditingEnabled(bool enabled);
    void setRadius(float radiusSteps);
    void setSigma(float sigmaSteps);
    void setPushPullStepMultiplier(float multiplier);

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

    struct PushPullState
    {
        bool active{false};
        int direction{0};
    };

    void bindWidgetSignals();
    void bindViewerSignals(CVolumeViewer* viewer);

    void emitPendingChanges();
    void refreshOverlay();
    void refreshMaskOverlay();
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
    void startPaintStroke(const cv::Vec3f& worldPos);
    void extendPaintStroke(const cv::Vec3f& worldPos, bool forceSample = false);
    void finishPaintStroke();
    bool applyInvalidationBrush();

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

    [[nodiscard]] bool isSegmentationViewer(const CVolumeViewer* viewer) const;
    [[nodiscard]] float gridStepWorld() const;

    void beginDrag(int row, int col, CVolumeViewer* viewer, const cv::Vec3f& worldPos);
    void updateDrag(const cv::Vec3f& worldPos);
    void finishDrag();
    void cancelDrag();

    void updateHover(CVolumeViewer* viewer, const cv::Vec3f& worldPos);

    bool startPushPull(int direction);
    void stopPushPull(int direction);
    void stopAllPushPull();
    bool applyPushPullStep();
    void onPushPullTick();

    SegmentationWidget* _widget{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    ViewerManager* _viewerManager{nullptr};
    CSurfaceCollection* _surfaces{nullptr};
    VCCollection* _pointCollection{nullptr};

    bool _editingEnabled{false};
    float _radiusSteps{5.75f};
    float _sigmaSteps{2.0f};
    bool _growthInProgress{false};
    SegmentationGrowthMethod _growthMethod{SegmentationGrowthMethod::Tracer};
    int _growthSteps{10};
    bool _usingCorrectionsGrowth{false};
    bool _correctionsAnnotateMode{false};
    uint64_t _activeCorrectionId{0};
    std::vector<uint64_t> _pendingCorrectionIds;
    std::unordered_set<uint64_t> _managedCorrectionIds;
    bool _ignoreSegSurfaceChange{false};
    bool _correctionsZRangeEnabled{false};
    int _correctionsZMin{0};
    int _correctionsZMax{0};
    std::optional<std::pair<int, int>> _correctionsRange;

    DragState _drag;
    HoverState _hover;
    QSet<CVolumeViewer*> _attachedViewers;
    QTimer* _pushPullTimer{nullptr};
    PushPullState _pushPull;

    bool _invalidationBrushActive{false};
    bool _paintStrokeActive{false};
    std::vector<cv::Vec3f> _currentPaintStroke;
    std::vector<std::vector<cv::Vec3f>> _pendingPaintStrokes;
    std::vector<cv::Vec3f> _paintOverlayPoints;
    cv::Vec3f _lastPaintSample{0.0f, 0.0f, 0.0f};
    bool _hasLastPaintSample{false};
    float _pushPullStepMultiplier{4.00f};
    std::optional<std::vector<SegmentationGrowthDirection>> _pendingShortcutDirections;
};
