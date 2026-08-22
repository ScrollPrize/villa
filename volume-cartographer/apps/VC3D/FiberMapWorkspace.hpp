#pragma once

#include <QGraphicsView>
#include <QHash>
#include <QMainWindow>
#include <QPointer>
#include <QPoint>
#include <QPointF>
#include <QRectF>
#include <QString>

#include <cstdint>
#include <optional>
#include <vector>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "AnnotationFrame.hpp"
#include "FiberMapStaleness.hpp"
#include "FiberNetworkLayout.hpp"

class LineAnnotationController;
class QDockWidget;
class QEvent;
class QGraphicsItem;
class QGraphicsPathItem;
class QGraphicsScene;
class QLabel;
class QMouseEvent;
class QShowEvent;
class QTreeWidget;
class QWheelEvent;

// Pan/zoom view of the fiber map, with the same gestures as the volume viewers:
// right-drag pans, the wheel zooms. Left clicks are reported as selection
// requests; ctrl+right-click without a drag asks for the control-point menu.
class FiberMapView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit FiberMapView(QWidget* parent = nullptr);

signals:
    void clicked(QPointF scenePos);
    void controlPointMenuRequested(QPointF scenePos, QPoint globalPos);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    // Right-button press position, against which the pan is called a drag (and
    // the ctrl+right menu suppressed), plus the running pan reference.
    QPoint _pressPosition;
    QPoint _panPosition;
    bool _pressed = false;
    bool _panning = false;
    bool _panDragged = false;
    bool _menuPending = false;
};

// Interactive 2D map of every fiber, unrolled about the scroll umbilicus onto
// one plane at the winding the solver inferred for it. The layout is only
// ever rebuilt on explicit request; fiber changes just mark the current one
// stale.
class FiberMapWorkspace : public QMainWindow
{
    Q_OBJECT

public:
    explicit FiberMapWorkspace(LineAnnotationController* controller,
                               QWidget* parent = nullptr);

signals:
    void openFiberAtControlPointRequested(uint64_t fiberId, int controlPointIndex);

protected:
    // The map's colours follow the application theme, and a switch is only
    // announced by a palette change.
    void changeEvent(QEvent* event) override;
    // Catches a layout built from fiber data that has since changed.
    void showEvent(QShowEvent* event) override;

private:
    // Scene-space copy of the placed fiber (y negated once, so scroll z reads
    // upward) alongside the two path items carrying its geometry.
    struct FiberEntry {
        vc3d::fiber_map::PlacedFiber fiber;
        QGraphicsPathItem* tracedItem = nullptr;
        QGraphicsPathItem* interpolatedItem = nullptr;
    };

    void rebuildLayout();
    void rebuildScene(const QString& emptyMessage);
    void rebuildTree();
    // Puts a stale reason on the status line and records it, without latching:
    // this is how applyStaleVerdict() surfaces staleness *derived* from the
    // dependency comparison, which clears itself when its cause reverts.
    void showStale(const QString& reason);
    // The latching form, for staleness asserted rather than derived — the
    // invariant-violation defenses. Nothing in the dependency sets can prove a
    // latched reason wrong, so it survives every comparison until the layout it
    // describes is rebuilt or cleared. The reason names what actually changed
    // either way.
    void markStale(const QString& reason);
    // Drops the layout entirely, for the changes that leave it not merely out of
    // date but meaningless: geometry unrolled over one set of voxels says nothing
    // about another set, and a different package has different fibers. A differing
    // physical voxel size over the same voxel counts is *not* one of those — see
    // evaluateDependencies().
    void clearLayout(const QString& reason);
    // The decision itself lives in FiberMapStaleness.hpp as a function of two
    // dependency sets, so every arm of it is testable without a widget. Splitting
    // evaluation from application matters for one caller in particular: the fiber
    // tree's currentItemChanged handler cannot afford a synchronous clearLayout(),
    // which clears the tree and so deletes the QTreeWidgetItem the signal is still
    // being delivered with.
    using StaleVerdict = vc3d::fiber_map::StaleVerdict;
    [[nodiscard]] vc3d::fiber_map::FiberMapDependencies currentDependencies() const;
    [[nodiscard]] vc3d::fiber_map::FiberMapDependencies layoutDependencies() const;
    // Cheap enough to guard interaction — integer compares, a few volume metadata
    // reads, and stats of the umbilicus candidates. Mutates nothing.
    [[nodiscard]] StaleVerdict evaluateDependencies() const;
    // Acts on a verdict. Returns true when interaction with the map must be
    // refused — the layout was cleared or marked stale — and false when the
    // map is current (restoring the resting status if a derived stale reason
    // just reverted).
    bool applyStaleVerdict(const StaleVerdict& verdict);
    // The two together, for callers that can absorb a scene rebuild inline.
    bool refreshStaleState();
    // Appends the package's umbilicus state to a status line, resolving it at
    // most once per umbilicus fingerprint: it is filesystem work, and keying it
    // to the fibers instead threw the answer away on every save.
    [[nodiscard]] QString withCachedUmbilicusStatus(const QString& status);
    // Scene units (voxels) per centimetre, from the package's voxel size when it
    // has one and from the documented assumption otherwise. This is the only
    // route from the map's cm-valued styling constants into the voxel-space
    // scene; it is never allowed to produce displayed text, because when the
    // voxel size is unknown it is a guess.
    [[nodiscard]] double sceneVxPerCm() const;
    // A layout length (voxels) as display text: centimetres when the voxel size
    // is known, otherwise the voxel count itself, which is the one figure still
    // true when the package cannot say how big a voxel is.
    [[nodiscard]] QString formatMapLength(double valueVx) const;
    void setHighlightedFiber(uint64_t fiberId);
    void clearControlPointDots();
    void handleSceneClick(const QPointF& scenePos);
    void handleControlPointMenu(const QPointF& scenePos, const QPoint& globalPos);
    void selectFiberRow(uint64_t fiberId);
    [[nodiscard]] uint64_t fiberAt(const QPointF& scenePos) const;
    [[nodiscard]] double sceneTolerance(double viewPixels) const;

    // QPointer: the controller is owned elsewhere and dies before this widget
    // during CWindow teardown; guards keep late signals harmless.
    QPointer<LineAnnotationController> _controller;
    FiberMapView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QTreeWidget* _tree = nullptr;
    QDockWidget* _fiberDock = nullptr;
    QLabel* _statusLabel = nullptr;
    vc3d::fiber_map::GlobalResult _layout;
    QHash<uint64_t, FiberEntry> _entries;
    std::vector<QGraphicsItem*> _controlPointDots;
    // Annotation voxel size of the snapshot the current layout came from, in µm;
    // unset when the package could not say, in which case nothing physical is
    // displayed.
    std::optional<double> _voxelSizeUm;
    // Scroll top in scene z, i.e. voxels; 0 when the volume's extent is unknown.
    double _scrollZMaxVx = 0.0;
    // The scene rect keeps slack on either side so a zoomed-in view can pan
    // past the outer panels; this is the tight rect around the content, which
    // is what the first-build fit frames.
    QRectF _contentRect;
    // What the empty scene last said, so a theme change can rebuild the scene as
    // it stands rather than take a fresh snapshot to work out the message again.
    QString _emptyMessage;
    uint64_t _highlightedFiber = 0;
    bool _syncingSelection = false;
    bool _viewFitted = false;
    bool _fiberDockSized = false;
    bool _retheming = false;
    // What the current layout was built from, and whether a change has been seen
    // since; a fresh workspace is stale until its first rebuild.
    uint64_t _layoutGeneration = 0;
    vc3d::annotation::AnnotationFrame _layoutFrame;
    QString _layoutUmbilicusFingerprint;
    // Controller counters as of the build. Compared rather than observed, so that
    // this workspace existing costs annotation work nothing.
    uint64_t _layoutPackageGeneration = 0;
    uint64_t _layoutUmbilicusGeneration = 0;
    // Whether a layout has ever been built. Distinct from "the layout has no
    // fibers": an empty result is still a result, built from dependencies that
    // can go out of date, and conflating the two left a map that had found no
    // umbilicus saying so forever. Also what keeps the dependency comparison from
    // firing against a default-constructed frame before the first build.
    bool _layoutBuilt = false;
    // The stale reason currently on the status line (empty when the map is
    // current). Lives here, not in the label's text: the label also carries
    // build summaries, and reading state back out of a widget is how a summary
    // once overwrote a warning while the map stayed stale.
    QString _staleReason;
    // The latched reason, kept apart from the displayed one: a higher-priority
    // derived reason can be displayed over a latch and then revert, and the
    // latch must resurface with its own wording. Empty when nothing is
    // latched; derived reasons clear when their cause reverts, this one only
    // on rebuild or clear.
    QString _latchedReason;
    // What the status line says when nothing is stale: the last build summary,
    // or the clear reason. Restored when a derived stale reason reverts.
    QString _freshStatus;
    QString _umbilicusStatusText;
    QString _umbilicusStatusFingerprint;
    bool _umbilicusStatusValid = false;
};
