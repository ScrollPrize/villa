#pragma once

#include <QFutureWatcher>
#include <QGraphicsView>
#include <QHash>
#include <QImage>
#include <QMainWindow>
#include <QThreadPool>
#include <QPointer>
#include <QPoint>
#include <QPointF>
#include <QRectF>
#include <QString>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "AnnotationFrame.hpp"
#include "FiberMapGapField.hpp"
#include "FiberMapRebuildQueue.hpp"
#include "FiberMapStaleness.hpp"
#include "FiberNetworkLayout.hpp"
#include "OpenDataVolumeOrientation.hpp"

class FiberMapRuler;
struct FiberMapRulerModel;
struct FiberMapRulerStyle;
class LineAnnotationController;
class QCheckBox;
class QDockWidget;
class QDoubleSpinBox;
class QSpinBox;
class QEvent;
class QGraphicsItem;
class QGraphicsPathItem;
class QGraphicsScene;
class QLabel;
class QLineEdit;
class QMouseEvent;
class QPainter;
class QHideEvent;
class QPushButton;
class QShowEvent;
class QTimer;
class QTreeWidget;
class QWheelEvent;

// Pan/zoom view of the fiber map, with the same gestures as the volume viewers:
// right-drag pans, the wheel zooms. Left clicks are reported as selection
// requests; ctrl+right-click without a drag asks for the control-point menu.
// Three axes are painted over the viewport in the foreground pass - windings
// above the scroll ceiling, sheet distance below the floor, height left of
// the map - each floating at its extent edge while that is on screen and
// clamped to the viewport edge once it is not, so whatever is in view is
// labelled.
class FiberMapView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit FiberMapView(QWidget* parent = nullptr);
    ~FiberMapView() override;

    void setRulerModel(const FiberMapRulerModel& model);
    void setRulerStyle(const FiberMapRulerStyle& style);

signals:
    void clicked(QPointF scenePos);
    void controlPointMenuRequested(QPointF scenePos, QPoint globalPos);
    // The view scale changed (wheel zoom); fitInView callers refresh
    // scale-dependent state themselves.
    void zoomed();

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    // The axes, drawn in viewport coordinates over everything in the scene.
    void drawForeground(QPainter* painter, const QRectF& rect) override;
    // Tooltips over an axis band explain the axis.
    bool viewportEvent(QEvent* event) override;

private:
    // Owned; plain painter objects, not widgets.
    std::vector<std::unique_ptr<FiberMapRuler>> _rulers;
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
// one plane at the winding the solver inferred for it. The layout rebuilds on
// request, and - being cheap through the memoized Update path - automatically
// for staleness a rebuild genuinely fixes (fiber or umbilicus changes) while
// the workspace is visible; a light visible-only poll notices such changes
// even when the user is not interacting with the map. Frame and voxel-size
// staleness never rebuilds automatically: it is commonly a transiently
// displayed volume, and it heals itself when the volume switches back.
class FiberMapWorkspace : public QMainWindow
{
    Q_OBJECT

public:
    explicit FiberMapWorkspace(LineAnnotationController* controller,
                               QWidget* parent = nullptr);
    // Shuts the rebuild queue down (pending dropped, in-flight publication
    // refused) and waits out the private worker pool; the worker owns its
    // own data, so the wait is for clean pool teardown, not for safety of
    // any shared state.
    ~FiberMapWorkspace() override;

signals:
    void openFiberAtControlPointRequested(uint64_t fiberId, int controlPointIndex);

public:
    // Everything the rebuild worker consumes and produces, owned BY the job
    // so the two threads share no mutable state (public so the worker's free
    // function can see it; construction and use stay private to this class).
    struct RebuildJobResult;

protected:
    // The map's colours follow the application theme, and a switch is only
    // announced by a palette change.
    void changeEvent(QEvent* event) override;
    // Catches a layout built from fiber data that has since changed, and
    // starts/stops the visible-only staleness poll.
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    // Scene-space copy of the placed fiber (y negated once, so scroll z reads
    // upward) alongside the two path items carrying its geometry.
    struct FiberEntry {
        vc3d::fiber_map::PlacedFiber fiber;
        // Linked-network id from the layout (-1: unlinked), for the dock
        // grouping and the selection's network co-highlight.
        int networkId = -1;
        QGraphicsPathItem* tracedItem = nullptr;
        QGraphicsPathItem* interpolatedItem = nullptr;
        // Gap spans (both endpoint controls tagged break): dotted amber.
        QGraphicsPathItem* gapItem = nullptr;
        // Damaged spans: the gap's dots in the pastel pink.
        QGraphicsPathItem* damagedItem = nullptr;
        // The network emphasis: a soft semi-transparent halo behind the
        // fiber's whole geometry, created only while its network is
        // selected.
        QGraphicsPathItem* glowItem = nullptr;
    };

    // The sole rebuild entry point, for the buttons, the automatic update,
    // and pending-request dispatch alike. full = drop every cache and
    // recompute from scratch (the escape hatch for cache bugs; when the
    // inputs match the last memoized Update it verifies the output digest,
    // and a mismatch disables memoization for the session). The build runs
    // asynchronously: the snapshot is taken on the GUI thread (the
    // controller is GUI-only), everything from input conversion through the
    // solve runs on a dedicated worker, and the result is validated against
    // the world it was started in before it may publish. Requests while a
    // build is in flight coalesce through FiberMapRebuildQueue. automatic
    // = armed by a staleness gate rather than asked for: such a request,
    // pending behind a build, is dropped when the volume moved on
    // meanwhile (finishRebuild), where an asked-for one is honoured.
    void requestRebuild(bool fullRebuild = false, bool automatic = false);
    void rebuildScene(const QString& emptyMessage);
    void rebuildTree();
    // Hides every fiber row the search box does not match, and every group
    // (error, network) left without a visible row; an empty box shows all.
    void applyTreeFilter();
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
    // Drops the layout entirely, for the one change that leaves it not merely
    // out of date but meaningless: a different package has different fibers.
    // Grid and voxel-size differences are derived staleness instead — they
    // commonly mean a transiently displayed volume and heal on revert — see
    // FiberMapStaleness.hpp.
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
    // The cached catalog manifest's version for a coordinate space
    // (FiberMapDependencies::catalogManifestToken): a stat, no parse.
    [[nodiscard]] QString catalogManifestTokenFor(const std::string& coordinateSpace) const;
    // Cheap enough to guard interaction — integer compares, a few volume metadata
    // reads, and stats of the umbilicus candidates. Mutates nothing.
    [[nodiscard]] StaleVerdict evaluateDependencies() const;
    // Acts on a verdict. Returns true when interaction with the map must be
    // refused — the layout was cleared or marked stale — and false when the
    // map is current (restoring the resting status if a derived stale reason
    // just reverted).
    bool applyStaleVerdict(const StaleVerdict& verdict);
    // Update is cheap (memoized), so staleness a rebuild genuinely fixes -
    // changed fibers or a changed umbilicus - triggers one automatically:
    // queued and debounced, only while the workspace is visible, never
    // re-entrantly. Frame/voxel-size/latched staleness never auto-rebuilds
    // (commonly a transiently displayed volume; see FiberMapStaleness.hpp).
    void scheduleAutoUpdate();
    // A translucent blue bar sweeping across the triggering button as the
    // rebuild's phases complete. The rebuild blocks the event loop, so the
    // sweep advances by forced synchronous repaints at phase boundaries
    // rather than by animation.
    // Launches the worker for an already-granted queue Start. The job (see
    // RebuildJobResult above) carries the snapshot, params, and the
    // memoization cache - moved out of the workspace at start, moved back
    // only on a validated publish.
    void startRebuild(bool fullRebuild, bool automatic);
    // Watcher-delivered completion: validate against the current world,
    // publish or discard, then run the one epilogue.
    void applyRebuild(const std::shared_ptr<RebuildJobResult>& job);
    // The validated-success half of applyRebuild: watermarks, digests,
    // scene, tree, status.
    void publishRebuild(RebuildJobResult& job);
    // The single terminal path: progress cleared, buttons restored, queue
    // returned to Idle, pending request dispatched.
    void finishRebuild();
    // A translucent blue marquee sweeping the triggering button while the
    // build runs (the event loop is live now, so it really animates).
    void startRebuildProgress(QPushButton* button);
    void tickRebuildProgress();
    void clearRebuildProgress();
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
    // The gap heat map (FiberMapGapField.hpp). The field is a pure function of
    // the published layout and the toolbar's settings: it is built on the
    // rebuild worker alongside the layout, with the settings captured at job
    // start, published with it, and dropped with it. It takes part in no
    // digest. Its tiles are ordinary scene items, recreated from the retained
    // field by rebuildScene() and forgotten before every scene clear.
    [[nodiscard]] vc3d::fiber_map::gaps::GapFieldParams gapFieldParams(
        std::optional<double> voxelSizeUm) const;
    void addGapTiles();
    void setGapTilesVisible(bool visible);
    void handleGapsToggled(bool checked);
    void handleGapParamsChanged();
    void updateGapLegend();
    // The status line's heat-map suffix for the published build as the
    // toolbar stands now (empty with Gaps off), and the status text it
    // composes with the layout's own summary.
    [[nodiscard]] QString gapStatusSuffix() const;
    void refreshGapStatus();
    // Whether the published build's heat-map settings (on/off and
    // parameters, captured at its job start) are what the toolbar asks for
    // now. False before any build has published.
    [[nodiscard]] bool gapSettingsMatchPublished() const;
    // A build is running or being applied: whatever it captured of the
    // toolbar is not yet published, so no reuse decision can be made from
    // the published settings until its epilogue.
    [[nodiscard]] bool rebuildInFlight() const;
    // Tiles follow the checkbox against the published field: shown (created
    // if need be) when on and the published settings match, hidden
    // otherwise. The rebuild epilogue's last word on the heat map.
    void reconcileGapTiles();
    // A settings change that the published build does not cover: queues an
    // Update when a layout exists or a build is running (the epilogue keeps
    // a pending Update alive while the settings mismatch, see
    // finishRebuild()); before a first build with nothing running, that
    // build captures the settings itself.
    void requestGapRebuild();
    // Label chips are fixed pixel size, so zoomed far enough out they bury
    // the geometry; below the scale where one winding spans fewer screen
    // pixels than a couple of chips, they all hide.
    void updateLabelChipVisibility();
    // (Re)paints one fiber's items for its current role: the selected fiber
    // (thicker lines, raised), a member of its linked network (a gentle glow
    // behind unchanged lines), or plain.
    enum class FiberEmphasis { Plain, Network, Selected };
    void paintFiberEmphasis(FiberEntry& entry, FiberEmphasis emphasis);
    void clearControlPointDots();
    void handleSceneClick(const QPointF& scenePos);
    // Ctrl+right-click on the map: acts on the selected fiber, and only when
    // the click lands on it. "Go to control point" when a dot was hit, and
    // "Delete" always.
    void handleControlPointMenu(const QPointF& scenePos, const QPoint& globalPos);
    // Right-click on a fiber row of the dock: selects the row, then offers
    // the same Delete.
    void handleTreeContextMenu(const QPoint& pos);
    // The confirmed delete both menus end in. Runs outside the menus' nested
    // event loops; the fiber must still be loaded under the same id and file
    // name after the confirmation dialog, and nothing happens if the map's
    // dependencies moved since the menu was built.
    void confirmAndDeleteFiber(uint64_t fiberId,
                               const std::string& fileName,
                               const QString& displayName,
                               const vc3d::fiber_map::FiberMapDependencies& menuDependencies);
    // The delete itself, queued out of the confirmation dialog's signal:
    // dependencies re-checked, the fiber re-checked by id and name, then the
    // controller's deleteFibers behind a lifetime guard.
    void deleteConfirmedFiber(uint64_t fiberId,
                              const std::string& fileName,
                              const vc3d::fiber_map::FiberMapDependencies& menuDependencies);
    // Lands the tree on the fiber's row. revealHidden: a search that hides
    // the row is cleared first - for the user's own click on the map, which
    // outranks the filter; a programmatic restore (theme change) leaves the
    // search as typed.
    void selectFiberRow(uint64_t fiberId, bool revealHidden = false);
    [[nodiscard]] uint64_t fiberAt(const QPointF& scenePos) const;
    [[nodiscard]] double sceneTolerance(double viewPixels) const;

    // QPointer: the controller is owned elsewhere and dies before this widget
    // during CWindow teardown; guards keep late signals harmless.
    QPointer<LineAnnotationController> _controller;
    FiberMapView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QTreeWidget* _tree = nullptr;
    QDockWidget* _fiberDock = nullptr;
    QPushButton* _updateButton = nullptr;
    // The dock's search box: a case-insensitive substring filter over each
    // fiber row's label and annotation name, re-applied after every tree
    // rebuild.
    QLineEdit* _searchEdit = nullptr;
    QLabel* _statusLabel = nullptr;
    QCheckBox* _gapsCheck = nullptr;
    // The colour scale reads "0 [ramp] [saturation]": the spinbox IS the
    // scale's top end.
    QLabel* _gapLegendZero = nullptr;
    QLabel* _gapLegend = nullptr;
    QDoubleSpinBox* _gapSaturationSpin = nullptr;
    QCheckBox* _gapFadeCheck = nullptr;
    QSpinBox* _gapFadeWindingsSpin = nullptr;
    // The published layout's gap field (null before a build that carried
    // one) and the settings it was built with, so a toggle can tell a field
    // it may show from one that needs a rebuild.
    std::shared_ptr<const vc3d::fiber_map::gaps::GapField> _gapField;
    // What the published build was asked for (the field is null when this
    // is false, and also when the build failed with it true), and why it
    // has no field if it failed.
    bool _gapPublishedWanted = false;
    QString _gapPublishedError;
    vc3d::fiber_map::gaps::GapFieldParams _gapFieldParams;
    // Scene-owned; cleared (not deleted) whenever the scene is.
    std::vector<QGraphicsItem*> _gapTiles;
    // Tile images the worker coloured for the published field, in the theme
    // it was told (dark or light), waiting for the next addGapTiles() to
    // wrap them in pixmaps; empty once used, or when the theme has moved on
    // and they are coloured again from the field.
    std::vector<QImage> _pendingGapTiles;
    bool _pendingGapTilesDark = false;
    vc3d::fiber_map::GlobalResult _layout;
    // Memoized rebuild state: the cache, whether a verification failure
    // benched it, and the last build's input/output digests for Full
    // rebuild's check.
    vc3d::fiber_map::GlobalLayoutCache _layoutCache;
    bool _memoizationDisabled = false;
    // The catalog's say on the scroll's winding sense, consulted by the
    // worker of every rebuild (the lookup memoizes its one manifest parse
    // and locks internally; the job holds it by shared pointer so a
    // workspace torn down mid-flight cannot pull it from under the worker).
    std::shared_ptr<vc3d::opendata::CatalogVolumeOrientationLookup> _catalogOrientation;
    bool _haveLastDigests = false;
    vc3d::fiber_map::ContentDigest _lastInputsDigest;
    vc3d::fiber_map::ContentDigest _lastOutputDigest;
    QHash<uint64_t, FiberEntry> _entries;
    std::vector<QGraphicsItem*> _controlPointDots;
    // Every fiber label chip of the current scene, for the zoom-threshold
    // visibility toggle; the scale below which they hide (0: never hide).
    std::vector<QGraphicsItem*> _labelChips;
    double _chipHideScale = 0.0;
    // Annotation voxel size of the snapshot the current layout came from, in µm;
    // unset when the package could not say, in which case no measured length
    // is displayed as physical (the gap scale's top stays a cm intent and its
    // tooltip names the assumption it is converted with).
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
    // Fibers currently carrying the subtle network emphasis, so the next
    // selection change can restore exactly them.
    std::vector<uint64_t> _networkEmphasized;
    bool _syncingSelection = false;
    bool _viewFitted = false;
    bool _autoUpdateScheduled = false;
    // The rebuild lifecycle decision object (single-flight, coalescing,
    // publication epochs); the Qt glue delegates every transition to it.
    vc3d::fiber_map::FiberMapRebuildQueue _rebuildQueue;
    // Dedicated one-thread pool: no starvation from the global pool's other
    // users, and a bounded, private teardown in the destructor.
    QThreadPool _rebuildPool;
    QFutureWatcher<std::shared_ptr<RebuildJobResult>>* _rebuildWatcher = nullptr;
    QTimer* _progressMarquee = nullptr;
    QPushButton* _progressButton = nullptr;
    qint64 _progressPhase = 0;
    // Visible-only staleness poll: integer compares, one frame derivation and
    // one umbilicus-file stat per tick — the workspace still costs annotation
    // work nothing (no controller signal connections), and a hidden tab costs
    // literally nothing.
    QTimer* _stalePollTimer = nullptr;
    bool _fiberDockSized = false;
    bool _retheming = false;
    // A delete is confirmed-or-pending: from the confirmation dialog opening
    // until the queued delete has run (or the dialog was dismissed). One at a
    // time, because the controller's delete yields to the event loop while it
    // drains saves.
    bool _deleteInFlight = false;
    // What the current layout was built from, and whether a change has been seen
    // since; a fresh workspace is stale until its first rebuild.
    uint64_t _layoutGeneration = 0;
    vc3d::annotation::AnnotationFrame _layoutFrame;
    QString _layoutUmbilicusFingerprint;
    QString _layoutCatalogVolume;
    QString _layoutCatalogManifestToken;
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
    // _freshStatus without the heat-map suffix, so the suffix can follow the
    // Gaps checkbox after publication.
    QString _freshStatusBase;
    // The stylesheet that goes with _freshStatus (red while errors are ringed).
    QString _freshStatusStyle;
    // The clear reason alone, without the umbilicus suffix _freshStatus
    // froze into itself: showEvent() recomposes the suffix from the live
    // package, which can change while nothing is built.
    QString _restingReason;
    QString _umbilicusStatusText;
    QString _umbilicusStatusFingerprint;
    bool _umbilicusStatusValid = false;
};
