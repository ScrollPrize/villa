#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/types/SheetCollection.hpp"

#include <QPolygonF>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class QDockWidget;
class QLabel;
class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QTimer;
class QMainWindow;
class ViewerManager;
class VolumeViewerBase;
class CWindow;
class QuadSurface;

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

    // Context-menu seam (called from CWindow's Ctrl+right handler before its own
    // menu): if a sheet is under the click, or a stitch is in progress, pops the
    // stitch menu and returns true; otherwise returns false so VC3D's menu runs.
    bool maybeShowStitchMenu(VolumeViewerBase* viewer, QPointF scenePos, QPoint globalPos);

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

    // --- manual sheet stitching ---------------------------------------------
    // One drawn sheet's slice crossing in scene coords, kept so a click can be
    // hit-tested back to a sheet id (the overlay is render-only; we hit-test what
    // we drew, like the sibling overlays do).
    struct SheetHit { int64_t id{}; std::vector<QPolygonF> curves; };

    void installInteractions(VolumeViewerBase* viewer);  // connect mouse/menu once per viewer
    void onSheetContextMenu(VolumeViewerBase* viewer, QPointF scenePos, QPoint globalPos);
    void onSheetLeftPress(VolumeViewerBase* viewer, QPointF scenePos);
    // Nearest drawn sheet to a scene point within tolerance (px); 0 if none.
    int64_t sheetAt(VolumeViewerBase* viewer, QPointF scenePos, double tolScene) const;
    double clickToleranceScene(VolumeViewerBase* viewer) const;  // ~7 px in scene units
    void enterStitch(int64_t seedId);
    void exitStitch();
    void updateStitchBanner();     // seed id + selection count + controls
    void toggleSheet(int64_t id);
    void rebuildStitchPreview();   // re-blend the selected sheets -> pane-0 preview
    void saveStitch();             // write the blended surface as a VC3D segment
    std::shared_ptr<QuadSurface> blendSelected() const;  // nullptr if nothing usable

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

    // Stitch mode state.
    CWindow* cwin_{};                 // for banner / 4-panel layout / save, if available
    bool stitchMode_{false};
    int64_t seedId_{0};               // the sheet that started the stitch
    std::unordered_set<int64_t> stitched_;  // sheets currently in the stitch
    std::set<VolumeViewerBase*> wired_;     // viewers whose mouse/menu are connected
    std::unordered_map<VolumeViewerBase*, std::vector<SheetHit>> hits_;  // per-viewer hit geometry
};
