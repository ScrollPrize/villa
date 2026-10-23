#pragma once

#include "FiberDisplayOptions.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/types/FiberView.hpp"
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <QPointF>
#include <QPointer>
#include <QString>
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

class CState;
class LineAnnotationController;

// The attached store's fibers: the catalog pages listed by the Fibers panel,
// the selected fiber, and the bounded slice overlay of every viewer. Holds no
// fiber data beyond the current page and what each viewer displays.
class FiberCollectionController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    struct Selection {
        int64_t id = 0;
        QString text, toolTip;
        // Points the selection slider walks; 0 disables it.
        int64_t pointCount = 0;
        int index = 0;
        bool openable = false;
    };
    // One listed fiber: its record (name, tags, typed summary, metadata) and links.
    struct CatalogRow {
        vc::objects::ObjectRecord record;
        // Stored units -> volume voxels; unset for a fiber not bound to the volume.
        std::optional<double> toVolume;
        int links = 0;
        int pendingLinks = 0;
        // Fibers connected through links, this one included.
        int componentSize = 0;
    };
    struct ControlPointHit {
        uint64_t fiberId = 0;
        int controlPointIndex = -1;
    };

    FiberCollectionController(CState* state, ViewerManager* manager, QObject* parent, LineAnnotationController* annotations);
    ~FiberCollectionController() override;
    [[nodiscard]] const FiberDisplayOptions& options() const { return options_; }
    void setOptions(const FiberDisplayOptions& options);
    void setLineAnnotationActive(bool active);
    void openCollection(const QString& path);
    void firstPage();
    void nextPage();
    // A listed fiber was clicked: focus it in the main views, or open it in
    // Line Annotation while that workspace is active.
    void activateFiber(int64_t id);
    void selectFiber(int64_t id, bool focusView = true, bool openAnnotation = false);
    void navigate(int index);
    void showInLineAnnotation(int64_t id);
    // Selects the trace under a plain left click that no native tool, collection
    // point or fiber annotation used. Returns true if a trace was selected.
    bool handleVolumeClick(Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    // Nearest drawn control point within maxDistancePx scene units of scenePoint.
    [[nodiscard]] std::optional<ControlPointHit> hitTestControlPoint(
        VolumeViewerBase* viewer, const QPointF& scenePoint, qreal maxDistancePx) const;
    void detachViewer(VolumeViewerBase* viewer) override;

signals:
    void catalogChanged(std::vector<FiberCollectionController::CatalogRow> rows, bool morePages);
    void statusChanged(bool attached, const QString& text);
    void selectionChanged(const FiberCollectionController::Selection& selection);
    // The selected fiber's current catalog row, for a panel whose page lacks it.
    void fiberRevealed(const FiberCollectionController::CatalogRow& row);
    // Progress of a Line Annotation open requested here; empty when idle.
    void annotationStatusChanged(const QString& text);
    // Asks the host to bring the Fibers panel into view.
    void panelRequested();

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    struct View;
    using Slice = vc::fibers::ViewRegion;
    // The last listed row: its UUID and, when bound, its volume length.
    struct CatalogCursor {
        std::string uuid;
        std::optional<double> length;
    };
    std::optional<CatalogCursor> catalogCursor_;
    CState* state_;
    ViewerManager* manager_;
    LineAnnotationController* annotations_;
    FiberDisplayOptions options_;
    int64_t totalFibers_{};
    bool annotationBusy_{false};
    bool lineAnnotationActive_{false};
    int64_t pendingAnnotation_{};
    double nativeToViewer_{1.0};
    double nativeVoxelMm_{1.0};
    int64_t selected_{0};
    std::optional<vc::objects::PinnedObject> selectedPin_;
    std::optional<double> selectedToVolume_;
    Selection selection_;
    uint64_t revision_{0}, selectionRevision_{0}, navigationRevision_{0}, annotationRevision_{0};
    bool enabled_{false}, catalogBusy_{false}, catalogPending_{false}, navigationBusy_{false};
    bool catalogPendingReset_{false};
    struct Click {
        QPointer<QObject> viewport;
        QPointF position;
    };
    std::optional<Click> press_, click_;
    std::vector<QMetaObject::Connection> annotationRenderConnections_;
    std::optional<int> pendingIndex_;
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> views_;
    void clear();
    void invalidate();
    void publishStatus(const QString& text);
    void publishSelection(Selection selection);
    void startPendingAnnotation();
    void finishAnnotationRequest();
    void watchAnnotationRender(class LineAnnotationDialog* dialog, int64_t id, uint64_t token);
    void projectChanged(bool preserveSelection = false);
    void listPage(bool reset);
    std::optional<std::string> familyFilter() const;
    bool noFamilies() const;
    void request(VolumeViewerBase* viewer, const Slice& slice);
    // Curved surfaces (flattened and segmentation views) draw the control
    // point chains of annotated fibers within the surface's bounds instead.
    void requestChains(VolumeViewerBase* viewer, const std::array<double, 6>& box);
    void collectChainPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder);
    std::optional<Slice> slice(VolumeViewerBase* viewer) const;
};
