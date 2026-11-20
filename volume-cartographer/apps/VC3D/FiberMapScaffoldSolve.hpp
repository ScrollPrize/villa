#pragma once
// The Fiber Map's scaffold solve: the solver dock, the list of saved solves,
// and the approval ranking they feed.
//
// The solve itself is the native library (vc/fiber_graph), which reads the
// store directly: the fibers, their links with anchor positions, their piece
// approvals and the umbilicus. Nothing here does arithmetic on windings; this
// file is the Qt skin and the store-write path.
//
// A run is saved in the store it solved (ext_fiber_graph_solve rows), so the
// list of solves is the set of run rows and a selected run is read back the
// same way whether it was just computed or found in the store. The approval
// ranking is therefore always the stored one.
#include <QObject>
#include <QString>
#include <QWidget>

#include "FiberMapScaffoldRanking.hpp"

#include <vc/fiber_graph/ScaffoldNormals.hpp>
#include <vc/fiber_graph/ScaffoldRunMonitor.hpp>

#include <vc_object_store/objects/FiberConstraintApprovalExtension.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/Transactions.hpp>

#include <opencv2/core.hpp>

#include <array>
#include <functional>
#include <map>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class CState;
namespace vc::lasagna { class LasagnaNormalSampler; }
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QTreeWidget;
class QTreeWidgetItem;
class QTimer;
template <class T> class QFutureWatcher;

namespace vc3d::fiber_map
{

// What the solver dock asks for. The frame and the fallback handedness come
// from the map's own snapshot: the solve must run in the frame the fibers are
// drawn in, and the store is the first authority on the scroll's winding
// sense.
struct ScaffoldSolveRequest {
    // With no link state selected the run solves on the measured constraints
    // alone; the dock only offers that while it is measuring crossings.
    bool useLinks = true;
    std::vector<std::string> approvalStates{"approved"};
    bool includeAuto = false;
    double manualWeight = 1.0;
    bool fixApprovedSubgraphs = true;
    std::string lemonBackend = "cost_scaling";
    std::string volume;
    std::optional<int> fallbackHandedness;
    // Also measure the scaffold's own crossings: strip normals from the
    // annotation, their signs from the crossings, and the perpendicular,
    // parallel and enclosure relations they assert.
    bool measureCrossings = false;
    double perpendicularWeight = 1.0, parallelWeight = 1.0, enclosureWeight = 1.0;
    // Measure each connector's winding separation on the bound normal
    // dataset. Off keeps the crossings as enclosure evidence only.
    bool measureWindingDistances = true;
    // "geometry": the fibers' own strip normals. "fused": those pulled
    // towards the sheet axis the package's fiber-inference prediction states.
    std::string normalSource = "geometry";
    std::string signBackend = "irls";
    std::size_t workers = 1;
    // Scope: with `restricted` set, only `fiberSubset` is loaded and solved
    // (the dock's scope selector fills it from the map's selected network).
    // A restricted request with an empty subset is refused rather than run on
    // the whole scaffold.
    bool restricted = false;
    std::vector<std::string> fiberSubset;
};

// One stored run with everything the docks show, read back from the store.
struct ScaffoldRunView {
    vc::fiber_graph::ScaffoldStoredRun run;
    std::vector<vc::fiber_graph::ScaffoldStoredResidual> residuals;
    // Normal flip suspects: a map link whose inside/outside order the solved
    // normals contradict, one row per accused fiber (ScaffoldSolveRuns.hpp).
    std::vector<vc::fiber_graph::ScaffoldStoredResidual> flips;
    std::map<std::string, vc::fiber_graph::ScaffoldStoredWinding> windings;
    // The accused stretches, highest score first (rankScaffoldPieces).
    std::vector<ScaffoldRankedPiece> pieces;
};

// Runs the solve, saves it, and holds the selected run for the docks. Owns no
// widgets; the panels below read it and the workspace connects its signals.
class ScaffoldSolveController : public QObject
{
    Q_OBJECT

public:
    explicit ScaffoldSolveController(CState* state, QObject* parent = nullptr);
    ~ScaffoldSolveController() override;

    // True while a solve is in flight; the dock disables its Run button.
    [[nodiscard]] bool busy() const { return _busy; }
    [[nodiscard]] const std::vector<vc::fiber_graph::ScaffoldStoredRun>& runs() const { return _runs; }
    [[nodiscard]] const ScaffoldRunView* selected() const { return _selected ? &*_selected : nullptr; }
    [[nodiscard]] const QString& lastError() const { return _lastError; }
    // Piece approvals of every fiber of the store, loaded on demand (the
    // highlight checkboxes ask for them) and dropped when the store changes.
    [[nodiscard]] const std::map<std::string, ScaffoldApprovalSpans>& approvals() const { return _approvals; }
    [[nodiscard]] bool approvalsLoaded() const { return _approvalsLoaded; }
    void loadApprovals();
    void dropApprovals();
    // The volume the runs are solved in. A run states its anchors and
    // stretches as arc lengths in that frame, so the control arcs they are
    // placed against are measured in it too - not in each fiber's stored
    // frame, where the same fiber has another length.
    void setFrame(const std::string& volume);
    // The normals the selected run used along one fiber: its annotated line in
    // the runs' frame and, per point, the solved strip normal with the run's
    // whole-fiber sign, turned to point away from the scroll centre (the
    // annotation views' convention). nullopt, with the reason, when the
    // selected run solved no normals or the store no longer holds them for
    // the fiber as it is now.
    struct RunFiberNormals {
        std::vector<cv::Vec3f> points, normals;
    };
    [[nodiscard]] std::optional<RunFiberNormals> runFiberNormals(const std::string& fiberUuid, QString* why = nullptr) const;
    // The span of a fiber covering an anchor arc, and whether it is approved.
    // Answers from the loaded piece approvals, so it needs loadApprovals().
    [[nodiscard]] ScaffoldAnchorSpan anchorSpan(const std::string& fiberUuid, double arc) const;

    // The winding density a run measures its separations on. The workspace
    // binds the package's Lasagna normal dataset; a run that would measure
    // one without a sampler is refused rather than run without distances.
    //
    // Two steps, so that Solve never blocks the GUI: the resolver runs on the
    // GUI thread and only captures what the project says to open; the opener
    // it returns runs on the solve's worker, where a remote manifest may take
    // its seconds. A dataset that cannot be opened fails the run there.
    using WindingSamplerOpener =
        std::function<std::shared_ptr<vc::lasagna::LasagnaNormalSampler>(QString* error)>;
    using WindingSamplerResolver = std::function<WindingSamplerOpener()>;
    void setWindingSampler(WindingSamplerResolver);

    // The prediction a fused run reads its sheet axis from. Resolved like the
    // winding sampler: on this thread, before the run starts.
    // `identity` receives what the prediction is called (its manifest
    // location): the key a fused run's normals are cached under.
    using AxisSamplerOpener =
        std::function<vc::fiber_graph::ScaffoldAxisSampler(QString* error, std::string* identity)>;
    using AxisSamplerResolver = std::function<AxisSamplerOpener()>;
    void setAxisSampler(AxisSamplerResolver);

    // Approval of a *measured* constraint (a crossing). A link's approval is
    // its edge record's review state and is not touched here.
    void reloadConstraintApprovals();
    [[nodiscard]] bool constraintApproved(const vc::fiber_graph::ScaffoldStoredResidual&) const;
    // Writes the approval or its revocation; the returned message is empty
    // when the store took it.
    QString setConstraintApproval(const vc::fiber_graph::ScaffoldStoredResidual&, bool approved);

    // Re-read the store's run rows (after a solve, or when the store changed).
    void reloadRuns();
    // Whether the last reloadRuns() read a store, as opposed to finding none
    // attached. False means the empty list says nothing about the store.
    [[nodiscard]] bool runsLoaded() const { return _runsLoaded; }
    // True when a fiber, a link or the umbilicus was changed after the run
    // read the store (scaffoldAnnotationCommit): its windings answer a
    // question about an annotation that has moved since. A run's own rows are
    // extension rows, so saving a run does not make it stale.
    [[nodiscard]] bool runStale(const vc::fiber_graph::ScaffoldStoredRun&) const;
    // Read one run's windings and ranking; an empty UUID clears the selection.
    void selectRun(const std::string& runUuid);
    // Start a solve. Refused while one is running or without a fiber store.
    void startSolve(const ScaffoldSolveRequest&);
    // Asks the running solve to stop at its next checkpoint (between fibers,
    // prediction chunks, components or cycle searches). Nothing is written.
    void cancelSolve();
    // Where the running solve is; an empty stage when none is in flight.
    [[nodiscard]] vc::fiber_graph::ScaffoldRunMonitor::Progress progress() const;

    // What two stored runs disagree about, read from the store on demand.
    // `error` is empty when the comparison was made.
    ScaffoldRunDiff diffRuns(const std::string& first, const std::string& second, QString& error);

signals:
    void runsChanged();
    void selectionChanged();
    void approvalsChanged();
    void constraintApprovalsChanged();
    void busyChanged(bool busy);
    // A line for the workspace status bar; errors arrive here too.
    void statusMessage(const QString& message);

private:
    struct SolveJob;
    void finishSolve();
    [[nodiscard]] std::optional<vc::objects::AttachmentIdentity> attachment() const;

    CState* _state = nullptr;
    bool _busy = false;
    QString _lastError;
    std::vector<vc::fiber_graph::ScaffoldStoredRun> _runs;
    // The store state the run list was read at, and the commit the annotation
    // itself last changed at, for the stale verdict.
    vc::objects::StoreStatus _storeState;
    std::int64_t _lastAnnotationCommit = 0;
    std::optional<ScaffoldRunView> _selected;
    std::map<std::string, ScaffoldApprovalSpans> _approvals;
    std::string _frameVolume;
    bool _approvalsLoaded = false;
    bool _runsLoaded = false;
    std::map<vc::objects::FiberConstraintIdentity, vc::objects::FiberConstraintApproval> _constraintApprovals;
    std::shared_ptr<SolveJob> _job;
    QFutureWatcher<void>* _watcher = nullptr;
    WindingSamplerResolver _windingSampler;
    AxisSamplerResolver _axisSampler;
};

// The solver dock: what a run uses, and the Run button.
class ScaffoldSolverPanel : public QWidget
{
    Q_OBJECT

public:
    ScaffoldSolverPanel(ScaffoldSolveController* controller, QWidget* parent = nullptr);

    // The map supplies the frame it drew in and the sense the catalog states;
    // the panel does not go looking for either.
    void setFrame(const std::string& volume, std::optional<int> fallbackHandedness);
    // The fibers of the map's currently selected network, for the restricted
    // scope. Asked on the GUI thread when a run starts; an empty answer makes
    // the controller refuse the run.
    using ScopeResolver = std::function<std::vector<std::string>()>;
    void setScopeResolver(ScopeResolver);
    // Whether the map can restrict a run to a selected network. Without a
    // layout there is no selection, so the scope is the whole scaffold and
    // the chooser says why.
    void setScopeAvailable(bool available);
    [[nodiscard]] ScaffoldSolveRequest request() const;
    [[nodiscard]] bool fixApprovedSubgraphs() const;

signals:
    void solveRequested();

private:
    void updateEnabled();
    // The running solve's stage on the dock's progress line.
    void showProgress();

    ScaffoldSolveController* _controller = nullptr;
    QCheckBox* _approvedCheck = nullptr;
    QCheckBox* _pendingCheck = nullptr;
    QCheckBox* _autoCheck = nullptr;
    QCheckBox* _fixApprovedCheck = nullptr;
    QCheckBox* _crossingsCheck = nullptr;
    QCheckBox* _windingDistanceCheck = nullptr;
    QDoubleSpinBox* _manualWeightSpin = nullptr;
    QDoubleSpinBox* _perpendicularWeightSpin = nullptr;
    QDoubleSpinBox* _parallelWeightSpin = nullptr;
    QDoubleSpinBox* _enclosureWeightSpin = nullptr;
    QComboBox* _normalSourceCombo = nullptr;
    QComboBox* _signBackendCombo = nullptr;
    QComboBox* _backendCombo = nullptr;
    QComboBox* _scopeCombo = nullptr;
    QPushButton* _runButton = nullptr;
    QPushButton* _cancelButton = nullptr;
    QLabel* _statusLabel = nullptr;
    QLabel* _progressLabel = nullptr;
    QProgressBar* _progressBar = nullptr;
    QTimer* _progressTimer = nullptr;
    // The stage the ETA is being measured over, and when it started. A stage
    // counts its own work, so the estimate is per stage and restarts with it.
    std::string _progressStage;
    std::chrono::steady_clock::time_point _progressStarted;
    std::int64_t _progressStageFirstDone = 0;
    ScopeResolver _scope;
    std::string _volume;
    std::optional<int> _fallbackHandedness;
};

// The list of saved solves. Selecting one, with the switch on, lays the map
// out from it.
class ScaffoldRunsPanel : public QWidget
{
    Q_OBJECT

public:
    ScaffoldRunsPanel(ScaffoldSolveController* controller, QWidget* parent = nullptr);

    // Whether the map should follow the selected solve's links.
    [[nodiscard]] bool follow() const;

signals:
    void followChanged();

private:
    enum Column {
        // When the run was made, from its own settings.
        kRunWhenColumn = 0,
        kRunMethodColumn,
        // "geometry" or "fused", and the sign backend it ran.
        kRunNormalsColumn,
        kRunSignBackendColumn,
        kRunFibersColumn,
        kRunComponentsColumn,
        kRunArcsColumn,
        kRunObjectiveColumn,
        kRunViolatedColumn,
        kRunMaxResidualColumn,
        // Anchored fibers whose sign the crossings would have flipped
        // (sign_fixed_disagreements); empty for a run that solved no signs.
        kRunSignColumn,
        // Whether the run is behind the store it was computed from.
        kRunStaleColumn,
        // Filled only while exactly two runs are selected: the fibers this
        // run places differently from the other one.
        kRunDiffColumn,
        kRunColumnCount,
    };
    void rebuild();
    void applySelection();
    // Compares the two selected runs, or clears the diff column.
    void showDiff();

    ScaffoldSolveController* _controller = nullptr;
    QTreeWidget* _tree = nullptr;
    QCheckBox* _followCheck = nullptr;
    QLabel* _diffLabel = nullptr;
    bool _rebuilding = false;
};

// The approval dock: the selected solve's ranking of suspect stretches and
// constraints, the highlight switches, and the way into a piece.
class ScaffoldApprovalPanel : public QWidget
{
    Q_OBJECT

public:
    ScaffoldApprovalPanel(ScaffoldSolveController* controller, QWidget* parent = nullptr);

    // What the map fades: nothing, the spans whose geometry is not approved,
    // or the spans whose normals are not (which implies their geometry is,
    // since a normals approval requires one).
    enum class FadeMode { None, Geometry, Normals };
    [[nodiscard]] FadeMode fadeMode() const;
    // Fiber labels for the table; the map knows them, the store rows do not.
    using LabelLookup = std::function<QString(const std::string& fiberUuid)>;
    void setLabelLookup(LabelLookup);

signals:
    void highlightRequested(const ScaffoldHighlight& highlight);
    // The row under the mouse, or an empty highlight when it left the list:
    // shown on the map in a style of its own, without selecting or moving.
    void hoverRequested(const ScaffoldHighlight& highlight);
    // Double-click on a stretch: open it as a piece between its bracketing
    // control points.
    // `arcFrom`/`arcTo`: the stretch's own arc range, so the piece can open
    // on its ends rather than on the controls that bracket it. NaN when the
    // row names no arcs.
    void pieceActivated(const std::string& fiberUuid, std::int64_t controlFrom, std::int64_t controlTo,
                        double arcFrom, double arcTo);
    void highlightModeChanged();

private:
    enum Column {
        kKindColumn = 0,
        kFiberColumn,
        kWhereColumn,
        kOffColumn,
        kStateColumn,
        kCostColumn,
        // Cost per 1000 voxels of fiber: what the default order ranks by.
        kDensityColumn,
        kJointColumn,
        // The numbered cycle (scaffoldCycleNumbers) and its constraint count.
        kCycleNumberColumn,
        kCycleColumn,
        kLengthColumn,
        kCountColumn,
        kColumnCount,
    };
    // One table row. A constraint row carries the two covering pieces as its
    // children, which are rows of their own: that is what has to be approved
    // before the constraint can be.
    struct Row {
        enum class Kind { Constraint, Piece };
        Kind kind = Kind::Constraint;
        // Index into the selected run's residuals, -1 for a stretch or a
        // covering piece.
        std::int64_t residual = -1;
        std::string fiberUuid, otherUuid;
        std::int64_t controlFrom = -1, controlTo = -1;
        // A segment's exact ends as control positions (3.42: 42 % of the way
        // from control point 3 to 4); -1 for a row that names whole spans.
        double positionFrom = -1, positionTo = -1;
        // The stretch's own arc range (voxels); NaN when the row names none,
        // which is a covering piece of a constraint.
        double arcFrom = std::numeric_limits<double>::quiet_NaN();
        double arcTo = std::numeric_limits<double>::quiet_NaN();
        ScaffoldApprovalRow keys;
    };
    void rebuild();
    void applySelection();
    void activateRow(QTreeWidgetItem* item);
    // The row an item names, or null when the item is stale.
    [[nodiscard]] const Row* rowOf(const QTreeWidgetItem*) const;
    // What a row names on the map: its fibers and the stretches drawn.
    [[nodiscard]] ScaffoldHighlight highlightOf(const QTreeWidgetItem*) const;
    bool eventFilter(QObject* watched, QEvent* event) override;
    // The residual a row names, when the row is a constraint of the selected
    // run; null for a stretch or a row of another run.
    [[nodiscard]] const vc::fiber_graph::ScaffoldStoredResidual* residualOf(const QTreeWidgetItem*) const;
    // The two covering pieces of a measured constraint, in the order of its
    // fibers; `known` is false when a fiber's spans could not place the arc.
    [[nodiscard]] std::array<ScaffoldAnchorSpan, 2> coveringSpans(
        const vc::fiber_graph::ScaffoldStoredResidual&) const;

    ScaffoldSolveController* _controller = nullptr;
    QPushButton* _approveButton = nullptr;
    QComboBox* _fadeCombo = nullptr;
    QComboBox* _orderCombo = nullptr;
    QComboBox* _componentCombo = nullptr;
    QTreeWidget* _tree = nullptr;
    QLabel* _summaryLabel = nullptr;
    // Errors of an approval attempt, kept apart from the summary so a
    // refusal does not erase the counts.
    QLabel* _errorLabel = nullptr;
    std::vector<Row> _rows;
    LabelLookup _labels;
    bool _rebuilding = false;
};

} // namespace vc3d::fiber_map
