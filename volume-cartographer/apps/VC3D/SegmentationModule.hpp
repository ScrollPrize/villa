#pragma once

#include <QObject>
#include <QPointF>
#include <QCursor>

#include <opencv2/core.hpp>

#include <functional>
#include <optional>

class SegmentationWidget;
class SegmentationEditManager;
class SegmentationOverlayController;
class ViewerManager;
class CSurfaceCollection;
class CVolumeViewer;
class QKeyEvent;
class QuadSurface;
class PlaneSurface;

class SegmentationModule : public QObject
{
    Q_OBJECT

public:
    SegmentationModule(SegmentationWidget* widget,
                       SegmentationEditManager* editManager,
                       SegmentationOverlayController* overlay,
                       ViewerManager* viewerManager,
                       CSurfaceCollection* surfaces,
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

    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float radius);
    void setSigma(float sigma);

    void applyEdits();
    void resetEdits();
    void stopTools();

    void attachViewer(CVolumeViewer* viewer);
    void updateViewerCursors();
    void setPointAddMode(bool enabled, bool silent = false);
    void togglePointAddMode();

    bool handleKeyPress(QKeyEvent* event);

signals:
    void editingEnabledChanged(bool enabled);
    void statusMessageRequested(const QString& text, int timeoutMs);
    void pendingChangesChanged(bool pending);
    void stopToolsRequested();
    void focusPoiRequested(const cv::Vec3f& position, QuadSurface* surface);

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
    void bindViewerSignals(CVolumeViewer* viewer);
    void refreshOverlay();
    void emitPendingChanges();
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

    SegmentationWidget* _widget{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    ViewerManager* _viewerManager{nullptr};
    CSurfaceCollection* _surfaces{nullptr};

    bool _editingEnabled{false};
    int _downsample{12};
    float _radius{10.0f};
    float _sigma{10.0f};

    bool _pointAddMode{false};
    DragState _drag;
    HoverState _hover;
    cv::Vec3f _cursorWorld{0, 0, 0};
    bool _cursorValid{false};
};
