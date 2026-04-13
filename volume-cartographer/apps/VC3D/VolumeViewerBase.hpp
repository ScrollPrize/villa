#pragma once

#include <QColor>
#include <QMetaObject>
#include <QPointF>
#include <QRectF>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Include for PathPrimitive type. No circular dependency because
// ViewerOverlayControllerBase.hpp only forward-declares VolumeViewerBase.
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/Compositing.hpp"

class CVolumeViewerView;
class QGraphicsItem;
class Surface;
class QuadSurface;
class Volume;
class VCCollection;
namespace vc::cache { class TieredChunkCache; }

// Abstract interface for volume viewers (CVolumeViewer and CTiledVolumeViewer).
// Overlay controllers work through this interface to support both viewer types.
class VolumeViewerBase
{
public:
    virtual ~VolumeViewerBase() = default;

    // Active segmentation handle (shared definition for both viewer types)
    struct ActiveSegmentationHandle {
        QuadSurface* surface{nullptr};
        std::string slotName;
        QColor accentColor;
        bool viewerIsSegmentationView{false};

        bool valid() const { return surface != nullptr; }
        explicit operator bool() const { return valid(); }
        void reset()
        {
            surface = nullptr;
            slotName.clear();
            accentColor = QColor();
            viewerIsSegmentationView = false;
        }
    };

    // --- Coordinate transforms ---
    virtual QPointF volumeToScene(const cv::Vec3f& vol_point) = 0;
    virtual cv::Vec3f sceneToVolume(const QPointF& scenePoint) const = 0;

    // --- Data access ---
    virtual Surface* currentSurface() const = 0;
    virtual std::string surfName() const = 0;
    virtual std::shared_ptr<Volume> currentVolume() const = 0;
    virtual VCCollection* pointCollection() const = 0;
    virtual vc::cache::TieredChunkCache* chunkCachePtr() const = 0;

    // --- Display settings ---
    virtual float getCurrentScale() const = 0;
    virtual float dsScale() const = 0;
    virtual float normalOffset() const = 0;
    virtual int datasetScaleIndex() const = 0;
    virtual float datasetScaleFactor() const = 0;

    // --- Direction/normal hints ---
    virtual bool isShowDirectionHints() const = 0;
    virtual bool isShowSurfaceNormals() const = 0;
    virtual float normalArrowLengthScale() const = 0;
    virtual int normalMaxArrows() const = 0;

    // --- Composite settings ---
    virtual const CompositeRenderSettings& compositeRenderSettings() const = 0;
    virtual bool isCompositeEnabled() const = 0;
    virtual bool isPlaneCompositeEnabled() const = 0;

    // --- Interaction state ---
    virtual uint64_t highlightedPointId() const = 0;
    virtual uint64_t selectedPointId() const = 0;
    virtual uint64_t selectedCollectionId() const = 0;
    virtual bool isPointDragActive() const = 0;
    virtual const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const = 0;

    // --- Overlay management ---
    virtual void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items) = 0;
    virtual void clearOverlayGroup(const std::string& key) = 0;
    virtual void clearAllOverlayGroups() = 0;

    // --- BBox ---
    virtual std::vector<std::pair<QRectF, QColor>> selections() const = 0;
    virtual std::optional<QRectF> activeBBoxSceneRect() const = 0;

    // --- Intersection rendering ---
    virtual void renderIntersections() = 0;
    virtual void invalidateIntersect(const std::string& name = "") = 0;
    virtual float intersectionOpacity() const = 0;
    virtual float intersectionThickness() const = 0;
    virtual int surfacePatchSamplingStride() const = 0;

    // --- Surface overlays ---
    virtual bool surfaceOverlayEnabled() const = 0;
    virtual const std::map<std::string, cv::Vec3b>& surfaceOverlays() const = 0;
    virtual float surfaceOverlapThreshold() const = 0;

    // --- Active segmentation ---
    virtual const ActiveSegmentationHandle& activeSegmentationHandle() const = 0;

    // --- Graphics view access (for overlay controllers) ---
    virtual CVolumeViewerView* graphicsView() const = 0;

    // --- QObject access for signal connections ---
    // VolumeViewerBase is not a QObject, so these helpers bridge the gap.
    virtual QObject* asQObject() = 0;
    virtual QMetaObject::Connection connectOverlaysUpdated(
        QObject* receiver, const std::function<void()>& callback) = 0;
};
