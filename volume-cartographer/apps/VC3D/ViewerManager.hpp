#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <vc_reader/affine.hpp>

#include "vc/core/util/Compositing.hpp"


class QMdiArea;
class QTimer;
class AxisAlignedSliceController;
class CChunkedVolumeViewer;
class CState;
struct POI;
class QWidget;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class RawPointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class InkDetectionOverlayController;
class SegmentationModule;
class ViewerOverlayControllerBase;
class VolumeViewerBase;
class Volume;
class Surface;
class QuadSurface;

class ViewerManager : public QObject
{
    Q_OBJECT

public:
    enum class ViewerRole {
        Standard,
        Annotation,
    };

    ViewerManager(CState* state,
                  VCCollection* points,
                  QObject* parent = nullptr);
    ~ViewerManager() override;

    // All live managers (one per workspace). Use to apply user preferences
    // uniformly so viewers behave the same across workspaces.
    static const std::vector<ViewerManager*>& allManagers();

    VolumeViewerBase* createViewer(const std::string& surfaceName,
                                   const QString& title,
                                   QMdiArea* mdiArea,
                                   ViewerRole role = ViewerRole::Standard);
    VolumeViewerBase* createViewerInWidget(const std::string& surfaceName,
                                           QWidget* parent,
                                           ViewerRole role = ViewerRole::Standard);
    void unregisterViewer(VolumeViewerBase* viewer);

    const std::vector<VolumeViewerBase*>& baseViewers() const { return _baseViewers; }

    void setSegmentationOverlay(SegmentationOverlayController* overlay);
    void setSegmentationModule(SegmentationModule* module);
    void setPointsOverlay(PointsOverlayController* overlay);
    void setRawPointsOverlay(RawPointsOverlayController* overlay);
    void setPathsOverlay(PathsOverlayController* overlay);
    void setBBoxOverlay(BBoxOverlayController* overlay);
    void setVectorOverlay(VectorOverlayController* overlay);
    void setVolumeOverlay(VolumeOverlayController* overlay);
    void setInkDetectionOverlay(InkDetectionOverlayController* overlay);
    InkDetectionOverlayController* inkDetectionOverlay() const { return _inkDetectionOverlay; }

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }

    void setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId);
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    std::shared_ptr<Volume> currentVolume() const;
    std::string currentVolumeId() const;

    void setOverlayOpacity(float opacity);
    float overlayOpacity() const { return _overlayOpacity; }

    void setOverlayColormap(const std::string& colormapId);
    void setOverlaySamplingMethod(int method);
    int overlaySamplingMethod() const { return _overlaySamplingMethod; }
    void setOverlayIntensityAsAlpha(bool enabled);
    bool overlayIntensityAsAlpha() const { return _overlayIntensityAsAlpha; }

    void setOverlayWindow(float low, float high);
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }
    void setOverlayMaxDisplayedResolution(int level);
    void setOverlayComposite(const OverlayCompositeSettings& settings);
    const OverlayCompositeSettings& overlayComposite() const { return _overlayComposite; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _volumeWindowLow; }
    float volumeWindowHigh() const { return _volumeWindowHigh; }

    bool resetDefaultFor(VolumeViewerBase* viewer) const;
    void setResetDefaultFor(VolumeViewerBase* viewer, bool value);

    // Registered automatically by AxisAlignedSliceController::setViewerManager.
    void setAxisAlignedSliceController(AxisAlignedSliceController* slices) { _slices = slices; }

    // --- Focus / navigation policy shared by all workspaces ---
    // Ensure the "focus" POI exists and lies inside the current volume: a
    // missing POI (or resetToCenter) is placed at the volume center, an
    // existing one is clamped to the volume bounds. overridePoint /
    // overrideNormal (already in the new volume's space) win when provided.
    // Returns false when no volume is set.
    bool resetFocusForVolumeChange(bool resetToCenter,
                                   const std::optional<cv::Vec3f>& overridePoint = std::nullopt,
                                   const std::optional<cv::Vec3f>& overrideNormal = std::nullopt);
    // Move the focus POI to a volume position and reorient the slice planes
    // around it (also nudges the segmentation viewer when the point projects
    // onto the active surface).
    bool centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId);
    // centerFocusAt() at the volume position under the mouse cursor: prefers
    // the viewer under the cursor, then scans all visible viewer viewports.
    bool centerFocusOnCursor();
    bool recenterViewersOnCurrentFocus();
    void recenterPlaneViewersOn(const cv::Vec3f& position);
    void recenterSegmentationViewerNear(const cv::Vec3f& position);
    VolumeViewerBase* segmentationViewer() const;

    // Last viewer the user interacted with (mouse press / zoom / cursor move);
    // null when none or while it is being torn down.
    VolumeViewerBase* activeViewer() const;

    // Runs before the default volume-click policy (Shift ignored, Ctrl+click
    // centers the focus); return true to consume the click.
    using VolumeClickInterceptor = std::function<bool(const cv::Vec3f& volLoc,
                                                      const cv::Vec3f& normal,
                                                      Surface* surf,
                                                      Qt::MouseButton button,
                                                      Qt::KeyboardModifiers modifiers)>;
    void setVolumeClickInterceptor(VolumeClickInterceptor interceptor) { _volumeClickInterceptor = std::move(interceptor); }
    // The id of a drawn surface within `tolerance` voxels of a volume point
    // other than the active one (Ctrl+Shift+click activation).
    using SurfacePicker = std::function<std::optional<std::string>(const cv::Vec3f&, float tolerance)>;
    void setSurfacePicker(SurfacePicker picker) { _surfacePicker = std::move(picker); }

    // --- Volume-switch policy shared by all workspaces ---
    // Snapshot of per-viewer cameras and view centers, retargetable through an
    // affine coordinate transform after a volume switch.
    struct ViewerNavigationSnapshot;
    std::shared_ptr<ViewerNavigationSnapshot> captureNavigation() const;
    void restoreNavigation(const std::shared_ptr<ViewerNavigationSnapshot>& snapshot,
                           const vcr::catalog::Affine& transform);
    // Set the volume on the state, re-derive the focus POI (volume-centered
    // when new/absent, clamped otherwise, transformed when a coordinate
    // transform is supplied) and retarget the captured viewer navigation.
    void switchVolume(std::shared_ptr<Volume> volume,
                      const std::optional<vcr::catalog::Affine>& navigationTransform = std::nullopt);

    // --- Fleet setters: apply a viewer preference to every viewer ---
    void setShowSurfaceNormals(bool show);
    void setShowCoordinateFrames(bool show);
    void setPlaneIntersectionLinesVisible(bool visible);

    // --- Reset-view-on-surface-change policy ---
    // Store the new default on every viewer, keeping the segmentation viewer
    // suppressed while a segmentation edit session is active.
    void setResetViewOnSurfaceChangeDefault(bool enabled);
    // Temporarily force the segmentation viewers off (true) or restore their
    // stored defaults (false).
    void setSegmentationResetViewSuppressed(bool suppressed);

    void setSegmentationCursorMirroring(bool enabled);
    bool segmentationCursorMirroring() const { return _mirrorCursorToSegmentation; }
    void broadcastLinkedCursor(VolumeViewerBase* source,
                               const std::optional<cv::Vec3f>& point);
    // Delivers a coalesced linked-cursor point now, if one is pending.
    void flushLinkedCursor();

    void setZScrollSensitivity(double sensitivity);
    double zScrollSensitivity() const { return _zScrollSensitivity; }

    void forEachBaseViewer(const std::function<void(VolumeViewerBase*)>& fn) const;
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }

    // Where `point` (voxels of the open volume) lies on `surface` (by its id:
    // a session draft or a surface of an attached store) within `tolerance`:
    // its pointer and distance. Nothing without a volume or hit.
    struct SurfaceLocation {
        cv::Vec3f ptr;
        float distance = 0;
    };
    std::optional<SurfaceLocation> locate(const QuadSurface& surface, const cv::Vec3f& point, float tolerance) const;
    // The same for each of `points`, in one read.
    std::vector<std::optional<SurfaceLocation>> locate(const QuadSurface& surface, const std::vector<cv::Vec3f>& points,
                                                       float tolerance) const;

signals:
    void baseViewerCreated(VolumeViewerBase* viewer);
    void baseViewerClosing(VolumeViewerBase* viewer);
    void currentVolumeChanged();
    // Emitted whenever the user explicitly places the focus (Ctrl+click,
    // focus-on-cursor key, point activation, ...).
    void focusCenteredByUser(const cv::Vec3f& position);
    // Emitted on Ctrl+Shift+click in a slice view when a patch lies under the
    // clicked point; the owner should make that patch the active segment.
    void surfaceActivationRequested(const std::string& surfaceId);
    // Aggregated per-viewer cache statistics (RAM / disk / network).
    void sharedCacheStatsChanged(const QStringList& items);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);
    void zScrollSensitivityChanged(double sensitivity);

private slots:
    void onStatusTick();
    void handleFocusPoiChanged(std::string name, POI* poi);
    void handleVolumeClicked(cv::Vec3f volLoc, cv::Vec3f normal, Surface* surf,
                             Qt::MouseButton button, Qt::KeyboardModifiers modifiers);

private:
    void registerOverlay(ViewerOverlayControllerBase* overlay);
    VolumeViewerBase* initializeChunkedViewer(CChunkedVolumeViewer* chunkedViewer,
                                              const std::string& surfaceName,
                                              ViewerRole role);

    CState* _state;
    VCCollection* _points;
    AxisAlignedSliceController* _slices{nullptr};
    SegmentationOverlayController* _segmentationOverlay{nullptr};
    // All overlay controllers that should be attached/detached from viewers.
    // Populated by the set*Overlay() methods. Does NOT include VolumeOverlayController
    // (which is not a ViewerOverlayControllerBase subclass).
    std::vector<ViewerOverlayControllerBase*> _allOverlays;
    SegmentationModule* _segmentationModule{nullptr};
    std::vector<VolumeViewerBase*> _baseViewers;
    VolumeViewerBase* _activeViewer{nullptr};
    VolumeClickInterceptor _volumeClickInterceptor;
    SurfacePicker _surfacePicker;
    // Refreshes the viewers' status every 5 s.
    QTimer* _statusClock{nullptr};
    std::unordered_map<VolumeViewerBase*, bool> _resetDefaults;
    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::shared_ptr<Volume> _overlayVolume;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    int _overlaySamplingMethod{0};
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    bool _overlayIntensityAsAlpha{false};
    int _overlayMaxDisplayedResolution{0};
    OverlayCompositeSettings _overlayComposite;
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};
    // Mirrored cursor points are coalesced to one broadcast per ~render tick:
    // each broadcast repaints every other viewer, and mouse moves arrive much
    // faster than viewers can repaint on large sessions.
    QTimer* _linkedCursorTimer{nullptr};
    VolumeViewerBase* _pendingLinkedCursorSource{nullptr};
    std::optional<cv::Vec3f> _pendingLinkedCursorPoint;
    bool _showCoordinateFrames{true};
    double _zScrollSensitivity{1.0};


    VolumeOverlayController* _volumeOverlay{nullptr};
    InkDetectionOverlayController* _inkDetectionOverlay{nullptr};
};
