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

#include <vc_object_store/objects/FiberConstraintApprovalExtension.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <array>
#include <functional>
#include <map>
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
class QPushButton;
class QRadioButton;
class QTreeWidget;
class QTreeWidgetItem;
template <class T> class QFutureWatcher;

namespace vc3d::fiber_map
{

// What the solver dock asks for. The frame and the fallback handedness come
// from the map's own snapshot: the solve must run in the frame the fibers are
// drawn in, and the store is the first authority on the scroll's winding
// sense.
struct ScaffoldSolveRequest {
    std::vector<std::string> approvalStates{"approved"};
    bool includeAuto = false;
    double manualWeight = 1.0;
    bool fixApprovedSubgraphs = true;
    std::string lemonBackend = "network_simplex";
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
};

// One stored run with everything the docks show, read back from the store.
struct ScaffoldRunView {
    vc::fiber_graph::ScaffoldStoredRun run;
    std::vector<vc::fiber_graph::ScaffoldStoredResidual> residuals;
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

    // The winding density a run measures its separations on. The workspace
    // binds the package's Lasagna normal dataset; a run that would measure
    // one without a sampler is refused rather than run without distances.
    using WindingSamplerResolver =
        std::function<std::shared_ptr<vc::lasagna::LasagnaNormalSampler>(QString* error)>;
    void setWindingSampler(WindingSamplerResolver);

    // The prediction a fused run reads its sheet axis from. Resolved like the
    // winding sampler: on this thread, before the run starts.
    using AxisSamplerResolver = std::function<vc::fiber_graph::ScaffoldAxisSampler(QString* error)>;
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
    // Read one run's windings and ranking; an empty UUID clears the selection.
    void selectRun(const std::string& runUuid);
    // Start a solve. Refused while one is running or without a fiber store.
    void startSolve(const ScaffoldSolveRequest&);

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
    std::optional<ScaffoldRunView> _selected;
    std::map<std::string, ScaffoldApprovalSpans> _approvals;
    bool _approvalsLoaded = false;
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
    [[nodiscard]] ScaffoldSolveRequest request() const;
    [[nodiscard]] bool fixApprovedSubgraphs() const;

signals:
    void solveRequested();

private:
    void updateEnabled();

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
    QPushButton* _runButton = nullptr;
    QLabel* _statusLabel = nullptr;
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
        kRunMethodColumn = 0,
        kRunFibersColumn,
        kRunComponentsColumn,
        kRunObjectiveColumn,
        kRunViolatedColumn,
        kRunMaxResidualColumn,
        kRunColumnCount,
    };
    void rebuild();
    void applySelection();

    ScaffoldSolveController* _controller = nullptr;
    QTreeWidget* _tree = nullptr;
    QCheckBox* _followCheck = nullptr;
    bool _rebuilding = false;
};

// The approval dock: the selected solve's ranking of suspect stretches and
// constraints, the highlight switches, and the way into a piece.
class ScaffoldApprovalPanel : public QWidget
{
    Q_OBJECT

public:
    ScaffoldApprovalPanel(ScaffoldSolveController* controller, QWidget* parent = nullptr);

    [[nodiscard]] bool highlightApprovedFibers() const;
    [[nodiscard]] bool highlightApprovedPieces() const;
    // Fiber labels for the table; the map knows them, the store rows do not.
    using LabelLookup = std::function<QString(const std::string& fiberUuid)>;
    void setLabelLookup(LabelLookup);

signals:
    void highlightRequested(const ScaffoldHighlight& highlight);
    // Double-click on a stretch: open it as a piece between its bracketing
    // control points.
    void pieceActivated(const std::string& fiberUuid, std::int64_t controlFrom, std::int64_t controlTo);
    void highlightModeChanged();

private:
    enum Column {
        kKindColumn = 0,
        kFiberColumn,
        kWhereColumn,
        kCostColumn,
        kCycleColumn,
        kCountColumn,
        kColumnCount,
    };
    void rebuild();
    void applySelection();
    void activateRow(QTreeWidgetItem* item);
    // The residual a row names, when the row is a constraint of the selected
    // run; null for a stretch or a row of another run.
    [[nodiscard]] const vc::fiber_graph::ScaffoldStoredResidual* residualOf(const QTreeWidgetItem*) const;

    ScaffoldSolveController* _controller = nullptr;
    QPushButton* _approveButton = nullptr;
    QCheckBox* _highlightFibersCheck = nullptr;
    QCheckBox* _highlightPiecesCheck = nullptr;
    QComboBox* _orderCombo = nullptr;
    QTreeWidget* _tree = nullptr;
    QLabel* _summaryLabel = nullptr;
    LabelLookup _labels;
    bool _rebuilding = false;
};

} // namespace vc3d::fiber_map
