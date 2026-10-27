#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/types/SheetCollection.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

class QDockWidget;
class QLabel;
class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QTimer;
class QMainWindow;
class ViewerManager;
class VolumeViewerBase;

// Read-only viewer for an Automated Sheet Volume (.asv): draws, on each planar CT
// view, the curves where the sheets crossing the slice meet it. Unlike the fibre
// collection viewer it does not gate on the volume's coordinate identity -- it binds
// to whatever volume is open and draws the sheets at their stored native-L0
// coordinates -- and it has no generator, no 3D view and no annotation editing.
//
// Performance: the expensive step (SQLite query + zlib decode) runs on a worker
// thread and is cached per viewport over a region a margin larger than the slice;
// each frame only intersects the cached meshes with the current plane (a bbox
// pre-check skips sheets that do not cross it), so panning within the cached region
// never touches SQLite. A slice that leaves the cached region kicks a new async
// query and keeps drawing the old cache until it arrives.
class SheetCollectionController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    SheetCollectionController(ViewerManager* manager, QMainWindow* window);
    ~SheetCollectionController() override;

    QDockWidget* dock() const { return dock_; }
    // Attach a .asv to the session (read-only). An empty/failed path clears it.
    void openCollection(const QString& path);

signals:
    // Emitted after openCollection: the attached path, or empty when cleared/failed.
    // Lets the window persist the attachment next to the project.
    void collectionOpened(const QString& path);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    void detachViewer(VolumeViewerBase* viewer) override;

private:
    using Slice = vc::sheets::ViewRegion;
    struct View;  // per-viewport cache + in-flight query (defined in the .cpp)

    std::optional<Slice> slice(VolumeViewerBase* viewer) const;
    // Start an async region query (margin-expanded) if the cache cannot serve `s`.
    void ensureRegion(VolumeViewerBase* viewer, const Slice& s);
    // Debounce queries during motion: (re)start a short timer and only query once the
    // view settles, so a drag does not fire a chain of reads.
    void scheduleQuery(VolumeViewerBase* viewer);
    void invalidateQuery();  // controls that change which sheets are fetched -> re-query
    void invalidateDraw();   // controls that change only how sheets are drawn -> repaint
    // Debounce dock-control edits: coalesce rapid spinbox steps into one apply once the
    // value settles. needQuery records whether any pending edit needs a re-query.
    void scheduleControlsApply(bool needQuery);
    // Minimum vertex count equivalent to the mm^2 size filter (area ~ verts*grid^2).
    int64_t minVertsThreshold() const;

    ViewerManager* manager_{};
    QDockWidget* dock_{};
    QLabel* status_{};
    QCheckBox* visible_{};
    QCheckBox* rainbow_{};
    QDoubleSpinBox* distance_{};    // slice half-thickness in native L0 voxels
    QDoubleSpinBox* lineWidth_{};
    QSpinBox* maxDisplayed_{};
    QDoubleSpinBox* minArea_{};     // hide sheets smaller than this (mm^2)

    std::unique_ptr<vc::sheets::SheetCollection> reader_;  // UI-thread: metadata/count only
    QString path_;
    double gridStepUm_{30.0};  // mesh grid spacing from the .asv, for the mm^2 size filter
    // Native L0 -> viewer-volume scale. 1.0 while viewing L0 (the common case); a
    // later refinement reads it from the open volume's pyramid level.
    double nativeToViewer_{1.0};
    uint64_t queryRevision_{0};  // bumped when the set of fetched sheets must change
    uint64_t drawRevision_{0};   // bumped on any change that needs a repaint
    QTimer* controlsTimer_{};    // debounces dock-control edits
    bool controlsNeedQuery_{false};

    // What was last drawn per viewport, so needsOverlayRebuild() can skip a rebuild
    // when neither the slice nor the draw revision changed.
    struct Built {
        std::optional<Slice> slice;
        uint64_t drawRevision{0};
        double dpr{1.0};  // raster resolution; rebuild if the device pixel ratio changes
    };
    std::unordered_map<VolumeViewerBase*, Built> built_;
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> views_;
};
