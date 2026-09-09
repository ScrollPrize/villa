#pragma once

#include "SpiralPclRole.hpp"
#include "SpiralPointPlacementMode.hpp"
#include "SpiralPointCollectionEdit.hpp"
#include "SpiralPointChain.hpp"
#include "overlays/ScreenSpacePointIndex.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <QJsonDocument>
#include <QPainterPath>
#include <QPointF>
#include <QPointer>
#include <QSet>
#include <QSize>
#include <QString>
#include <QTransform>

#include <opencv2/core/types.hpp>

#include <array>
#include <memory>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

class QuadSurface;
class VolumeViewerBase;
class SpiralBrushCursorWidget;
class PointsOverlayController;

// Spiral-only drawn inputs. This deliberately does not use VC3D's annotation
// or segmentation drawing paths: brush marks are true swept-circle vector
// shapes, while control-point lines retain their ordered surface/volume samples.
//
// Brush paint and control-point lines live on the flattened preview viewer.
// Point collections (same-winding with Q, relative-winding with E) can also be
// placed on the bound plane viewers, where a click maps straight to a volume
// position; their drafts render on every bound viewer by projection.
class SpiralBrushController final : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    using PclRole = vc3d::spiral::PclRole;

    struct PreparedPatch {
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> surface;
    };
    struct PreparedPointCollections {
        QString id;
        QString role;
        QJsonDocument document;
        QString operation;
        QString targetCollectionId;
        QString baseSourceRevision;
    };

    explicit SpiralBrushController(QObject* parent = nullptr);

    void bindFlattenedViewer(VolumeViewerBase* viewer);
    // Plane viewers accept point placement and collection selection only.
    void bindPlaneViewer(VolumeViewerBase* viewer);
    void setPaintSurface(const std::shared_ptr<QuadSurface>& surface);
    void setVisiblePointCollectionIds(const QSet<QString>& ids);
    void setPclSource(PclRole role, const QJsonDocument& document,
                      double sourceToPreviewScale,
                      const QString& sourceRevision, bool editable);
    void setPclSourceVisible(PclRole role, bool visible);
    void setPclHitOverlay(PclRole role, PointsOverlayController* overlay);
    void setPointViewTolerance(double tolerance);
    void replacementConflict(const QString& id, bool discardDraft);
    void resetSession();
    bool hasUnfinalizedPaint() const;
    bool hasUnfinalizedPolylines() const;
    bool hasReadyDrafts() const;
    void markDraftsReady();
    int brushDiameter() const { return _diameterPx; }

    std::vector<PreparedPatch> preparePatches(QStringList& warnings);
    std::vector<PreparedPointCollections> preparePointCollections(QStringList& warnings);
    void finalizationSucceeded(const QString& id);
    void finalizationFailed(const QString& id);
    // The dataset now holds these finalized inputs. Their local drafts are
    // superseded by the refreshed source snapshot, so they are dropped; in
    // particular a replaced or deleted collection stops being suppressed in
    // the display overlay, which otherwise hid it for the rest of the session.
    void commitSucceeded(const QStringList& ids);
    void discardUnfinalized();

signals:
    void paintStateChanged();
    void brushDiameterChanged(int diameterPx);
    void pointPlacementRejected(const QString& message);
    // Source collections of `role` that a local draft replaces or deletes;
    // the display overlay hides them so the draft is the only rendering.
    void suppressedPclCollectionIdsChanged(vc3d::spiral::PclRole role,
                                           const QSet<QString>& ids);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    enum class GestureState { Painted, Ready, Finalizing, Finalized };
    struct Gesture {
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> source;
        QPainterPath shape;
        QPointF gridOrigin;
        QPointF columnStep;
        QPointF rowStep;
        GestureState state = GestureState::Painted;
    };
    struct PolylineGesture {
        enum class Kind { Freehand, Anchored, PointCollection };
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> source;
        std::vector<vc3d::spiral::PointChainAnchor> anchors;
        std::vector<QPointF> surfacePoints;
        std::vector<cv::Vec3f> volumePoints;
        std::optional<vc3d::spiral::EditablePclDraft> pclEdit;
        qint64 creationTime = 0;
        int sequence = 0;
        Kind kind = Kind::Freehand;
        GestureState state = GestureState::Painted;
    };
    struct EditablePclHit {
        // Source-collection hits carry the role and index into that role's
        // sources; draft hits carry the polyline index instead.
        PclRole role = PclRole::SameWinding;
        std::optional<std::size_t> sourceIndex;
        int polylineIndex = -1;
        std::size_t pointIndex = 0;
        QPointF scenePosition;
        QPointF devicePosition;
        QColor color;
        bool sourceMarker = false;
        std::uint64_t stableCollectionOrder = 0;
        std::uint64_t stablePointOrder = 0;
        VolumeViewerBase* viewer = nullptr;
    };
    struct EditablePclHitIndexState {
        std::vector<EditablePclHit> records;
        ScreenSpacePointIndex index;
        std::unordered_map<int, std::vector<cv::Vec3f>> projectionPositions;
        SurfaceProjectionContext projectionContext;
        QTransform viewportTransform;
        QSize viewportSize;
        std::uint64_t contentRevision = 0;
        bool valid = false;
    };
    // The loaded collections of one editable role, as imported from its
    // display artifact, plus what the display overlay needs from them.
    struct PclSourceSet {
        std::vector<vc3d::spiral::EditablePclDraft> sources;
        std::unordered_map<std::uint64_t, std::size_t> indexById;
        QSet<qulonglong> editableIds;
        QPointer<PointsOverlayController> hitOverlay;
        QSet<QString> suppressedIds;
        bool visible = false;
    };
    // A viewer this controller filters events on and draws a cursor cue in.
    struct BoundViewer {
        VolumeViewerBase* viewer = nullptr;
        QObject* viewObject = nullptr;
        QObject* viewport = nullptr;
        SpiralBrushCursorWidget* cursorWidget = nullptr;
    };
    enum class DragMode { None, Paint, Polyline, Erase };

    QColor nextColor();
    QPainterPath deviceDisk(const QPointF& center) const;
    QPainterPath deviceSweep(const QPointF& from, const QPointF& to) const;
    QPainterPath deviceToSurface(const QPainterPath& path) const;
    std::optional<QPointF> devicePointToSurface(const QPointF& point) const;
    std::optional<QPointF> scenePointToSurface(const QPointF& point) const;
    QPainterPath surfaceToScene(const QPainterPath& path) const;
    void beginPaint(const QPointF& devicePos);
    void beginPolyline(const QPointF& devicePos);
    void appendAnchoredPoint(const QPointF& devicePos);
    void finishAnchoredPolyline();
    void appendPointCollectionPoint(VolumeViewerBase* viewer, const QPointF& devicePos);
    std::optional<EditablePclHit> editablePclHitAt(
        VolumeViewerBase* viewer, const QPointF& devicePos);
    void updateEditablePclHover(VolumeViewerBase* viewer, const QPointF& devicePos);
    void clearEditablePclHover();
    void invalidateEditablePclHitIndex();
    void rebuildEditablePclHitIndex(VolumeViewerBase* viewer);
    std::optional<EditablePclHit> draftEditablePclHitAt(
        VolumeViewerBase* viewer, const QPointF& devicePos);
    void selectEditablePcl(const EditablePclHit& hit);
    void selectEditablePcl(PclRole role, std::size_t sourceIndex);
    void reverseActivePcl();
    void confirmDeleteActivePcl();
    const std::vector<cv::Vec3f>& pointCollectionPositions(
        const PolylineGesture& line) const;
    bool pointCollectionHasChanges(const PolylineGesture& line) const;
    void updateSuppressedPclIds();
    void finishPointCollection(bool removeIncompleteNewCollection = true);
    void deactivatePointPlacement();
    void beginErase(const QPointF& devicePos);
    void extendDrag(const QPointF& devicePos);
    void finishDrag(const QPointF& devicePos);
    void eraseWith(const QPainterPath& deviceShape);
    void updateCursor(const QPointF& devicePos);
    void updateCursorWidget();
    void sampleColor(const QPointF& scenePos);
    bool appendPolylinePoint(const QPointF& devicePos);
    bool rebuildAnchoredPolyline(PolylineGesture& gesture);
    std::optional<std::pair<QPointF, cv::Vec3f>> pointOnSurface(
        const QPointF& devicePos, const std::shared_ptr<QuadSurface>& source) const;
    std::optional<cv::Vec3f> volumePointOnSurface(
        const QPointF& surfacePos, const std::shared_ptr<QuadSurface>& source) const;
    bool surfaceSegmentValid(
        const QPointF& from, const QPointF& to,
        const std::shared_ptr<QuadSurface>& source) const;
    void resamplePolyline(PolylineGesture& gesture);
    PreparedPatch makePatch(Gesture& gesture) const;
    // The bound viewer whose view or viewport is `watched`, or nullptr.
    const BoundViewer* boundViewerFor(const QObject* watched) const;
    const BoundViewer* boundViewerFor(const VolumeViewerBase* viewer) const;
    bool isPlaneViewer(const VolumeViewerBase* viewer) const;
    void unbindViewer(BoundViewer& bound);
    void bindViewer(BoundViewer& bound, VolumeViewerBase* viewer);
    // Stable ordering of a source collection across both roles' sets, for
    // deterministic hit-test tie breaking.
    std::uint64_t sourceOrder(PclRole role, std::size_t index) const;
    PclSourceSet& sourcesFor(PclRole role) { return _pclSources[vc3d::spiral::pclRoleIndex(role)]; }
    const PclSourceSet& sourcesFor(PclRole role) const { return _pclSources[vc3d::spiral::pclRoleIndex(role)]; }
    QString pointLabel(const PolylineGesture& line, std::size_t pointIndex) const;
    bool showsPointLabels(const PolylineGesture& line, std::size_t lineIndex) const;

    // The flattened preview viewer: brush paint, control-point lines, and
    // exact-surface point placement.
    BoundViewer _flattened;
    // Plane viewers: point placement by volume position and selection only.
    std::vector<std::unique_ptr<BoundViewer>> _planeViewers;
    // Aliases of `_flattened` kept for the paint/polyline code paths.
    VolumeViewerBase* _viewer = nullptr;
    std::shared_ptr<QuadSurface> _paintSurface;
    QObject* _viewport = nullptr;
    QObject* _viewObject = nullptr;
    SpiralBrushCursorWidget* _cursorWidget = nullptr;
    std::vector<Gesture> _gestures;
    std::vector<PolylineGesture> _polylines;
    QSet<QString> _visiblePointCollectionIds;
    std::array<PclSourceSet, vc3d::spiral::kEditablePclRoles.size()> _pclSources;
    std::unordered_map<VolumeViewerBase*, EditablePclHitIndexState> _editablePclHitIndex;
    std::uint64_t _editablePclHitContentRevision = 1;
    QSet<QRgb> _usedColors;
    std::optional<QColor> _sampledColor;
    QPointF _cursorDevicePos;
    bool _cursorInside = false;
    // The bound viewer the cursor is inside (the cue is drawn there).
    const BoundViewer* _cursorBound = nullptr;
    QPointF _lastDevicePos;
    DragMode _dragMode = DragMode::None;
    int _activeGesture = -1;
    int _activePolyline = -1;
    int _nextPolylineSequence = 1;
    bool _polylineBlocked = false;
    int _diameterPx = 32;
    bool _gHeld = false;
    bool _shiftHeld = false;
    bool _controlHeld = false;
    bool _vHeld = false;
    bool _vClickConsumed = false;
    SpiralPointPlacementMode _pointPlacement;
    std::optional<EditablePclHit> _hoveredEditablePcl;
    bool _pclLeftClickConsumed = false;
    float _pointViewToleranceVoxels = 100.0f;
};
