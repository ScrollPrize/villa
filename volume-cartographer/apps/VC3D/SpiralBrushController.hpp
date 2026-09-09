#pragma once

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
class SpiralBrushController final : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
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
    void setPaintSurface(const std::shared_ptr<QuadSurface>& surface);
    void setVisiblePointCollectionIds(const QSet<QString>& ids);
    void setSameWindingSource(const QJsonDocument& document,
                              double sourceToPreviewScale,
                              const QString& sourceRevision,
                              bool editable);
    void setSameWindingSourceVisible(bool visible);
    void setSameWindingHitOverlay(PointsOverlayController* overlay);
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
    void discardUnfinalized();

signals:
    void paintStateChanged();
    void brushDiameterChanged(int diameterPx);
    void pointPlacementRejected(const QString& message);
    void suppressedSameWindingCollectionIdsChanged(const QSet<QString>& ids);

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
        std::optional<std::size_t> sourceIndex;
        int polylineIndex = -1;
        std::size_t pointIndex = 0;
        QPointF scenePosition;
        QPointF devicePosition;
        QColor color;
        bool sourceMarker = false;
        std::uint64_t stableCollectionOrder = 0;
        std::uint64_t stablePointOrder = 0;
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
    void appendPointCollectionPoint(const QPointF& devicePos);
    std::optional<EditablePclHit> editablePclHitAt(
        const QPointF& devicePos);
    void updateEditablePclHover(const QPointF& devicePos);
    void clearEditablePclHover();
    void invalidateEditablePclHitIndex();
    void rebuildEditablePclHitIndex();
    std::optional<EditablePclHit> draftEditablePclHitAt(
        const QPointF& devicePos);
    void selectEditablePcl(const EditablePclHit& hit);
    void selectEditablePcl(std::size_t sourceIndex);
    void reverseActivePcl();
    void confirmDeleteActivePcl();
    const std::vector<cv::Vec3f>& pointCollectionPositions(
        const PolylineGesture& line) const;
    bool pointCollectionHasChanges(const PolylineGesture& line) const;
    void updateSuppressedSameWindingIds();
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

    VolumeViewerBase* _viewer = nullptr;
    std::shared_ptr<QuadSurface> _paintSurface;
    QObject* _viewport = nullptr;
    QObject* _viewObject = nullptr;
    SpiralBrushCursorWidget* _cursorWidget = nullptr;
    std::vector<Gesture> _gestures;
    std::vector<PolylineGesture> _polylines;
    QSet<QString> _visiblePointCollectionIds;
    std::vector<vc3d::spiral::EditablePclDraft> _sameWindingSources;
    std::unordered_map<std::uint64_t, std::size_t> _sameWindingSourceIndexById;
    QSet<qulonglong> _editableSameWindingCollectionIds;
    QPointer<PointsOverlayController> _sameWindingHitOverlay;
    EditablePclHitIndexState _editablePclHitIndex;
    std::uint64_t _editablePclHitContentRevision = 1;
    QSet<QString> _suppressedSameWindingCollectionIds;
    bool _sameWindingSourceVisible = false;
    QSet<QRgb> _usedColors;
    std::optional<QColor> _sampledColor;
    QPointF _cursorDevicePos;
    bool _cursorInside = false;
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
