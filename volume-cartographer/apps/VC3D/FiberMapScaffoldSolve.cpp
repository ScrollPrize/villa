#include "FiberMapScaffoldSolve.hpp"

#include "CState.hpp"

#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/Transactions.hpp>

#include <QCheckBox>
#include <QDateTime>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QSysInfo>
#include <QThread>
#include <QThreadPool>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrent>

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

namespace vc3d::fiber_map
{
namespace
{
using namespace vc::fiber_graph;

QString oneLine(const ScaffoldStoredRun& run)
{
    const auto count = [&](const char* name) {
        const auto found = run.counts.find(name);
        return found == run.counts.end() ? 0 : found->second;
    };
    const auto metric = [&](const char* name) {
        const auto found = run.metrics.find(name);
        return found == run.metrics.end() ? 0. : found->second;
    };
    return QObject::tr("%1 fibers in %2 components, %3 turns of residual over %4 constraints (%5 violated)")
        .arg(count("fibers"))
        .arg(count("components"))
        .arg(metric("weighted_l1"), 0, 'f', 3)
        .arg(count("constraints"))
        .arg(count("violated_constraints"));
}

} // namespace

// Everything the worker owns, so the GUI and the worker share nothing mutable.
struct ScaffoldSolveController::SolveJob {
    ScaffoldSolveRequest request;
    std::optional<vc::objects::AttachmentIdentity> attachment;
    std::shared_ptr<vc::objects::ObjectStoreSession> session;
    // Resolved on the GUI thread before the job starts; null when the run
    // measures no winding separations.
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    // Outputs.
    QString error;
    QString summary;
    std::string runUuid;
    ScaffoldRunRows rows;
    QThreadPool pool;
};

ScaffoldSolveController::ScaffoldSolveController(CState* state, QObject* parent)
    : QObject(parent)
    , _state(state)
{
}

ScaffoldSolveController::~ScaffoldSolveController() = default;

void ScaffoldSolveController::setWindingSampler(WindingSamplerResolver resolver)
{
    _windingSampler = std::move(resolver);
}

std::optional<vc::objects::AttachmentIdentity> ScaffoldSolveController::attachment() const
{
    if (!_state) return std::nullopt;
    return _state->storeFor("fiber");
}

void ScaffoldSolveController::reloadRuns()
{
    _runs.clear();
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) {
        emit runsChanged();
        return;
    }
    QString error;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            _runs = readScaffoldRuns(reader);
        });
    } catch (const std::exception& problem) {
        error = tr("The saved solves could not be read: %1").arg(QString::fromUtf8(problem.what()));
    }
    // Newest last in commit order; the list shows them newest first.
    std::reverse(_runs.begin(), _runs.end());
    if (!error.isEmpty()) {
        _lastError = error;
        emit statusMessage(error);
    }
    emit runsChanged();
}

void ScaffoldSolveController::selectRun(const std::string& runUuid)
{
    if (runUuid.empty()) {
        if (_selected) {
            _selected.reset();
            emit selectionChanged();
        }
        return;
    }
    if (_selected && _selected->run.runUuid == runUuid) return;
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) return;
    ScaffoldRunView view;
    const auto found = std::find_if(_runs.begin(), _runs.end(),
                                    [&](const auto& run) { return run.runUuid == runUuid; });
    if (found == _runs.end()) return;
    view.run = *found;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            view.residuals = readScaffoldResiduals(reader, runUuid);
            view.windings = readScaffoldWindings(reader, runUuid);
        });
    } catch (const std::exception& problem) {
        _lastError = tr("The solve could not be read: %1").arg(QString::fromUtf8(problem.what()));
        emit statusMessage(_lastError);
        return;
    }
    view.pieces = rankScaffoldPieces(view.residuals);
    _selected = std::move(view);
    emit selectionChanged();
}

void ScaffoldSolveController::dropApprovals()
{
    if (!_approvalsLoaded && _approvals.empty()) return;
    _approvals.clear();
    _approvalsLoaded = false;
    emit approvalsChanged();
}

void ScaffoldSolveController::loadApprovals()
{
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) return;
    std::map<std::string, ScaffoldApprovalSpans> loaded;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            vc::objects::ObjectSelection selection;
            selection.kind = "fiber";
            for (const auto& record : reader.selectObjects(selection)) {
                const auto rows = vc::objects::readFiberPieceApprovals(reader, record.uuid);
                if (rows.empty()) continue;
                const auto revision = reader.currentObject(record.uuid);
                if (!revision) continue;
                const auto content = vc::objects::fiberPieceContent(reader, *revision);
                const auto states = vc::objects::fiberPieceApprovalStates(content, rows);
                ScaffoldApprovalSpans entry;
                entry.controlIds = content.controlIds;
                const auto spans = content.controlIds.empty() ? 0 : content.controlIds.size() - 1;
                for (std::size_t aspect = 0; aspect < 2; ++aspect) {
                    entry.approved[aspect].assign(spans, 0);
                    for (const auto span : states.approvedSpans[aspect])
                        if (span < spans) entry.approved[aspect][span] = 1;
                }
                loaded.emplace(record.uuid, std::move(entry));
            }
        });
    } catch (const std::exception& problem) {
        _lastError = tr("The piece approvals could not be read: %1").arg(QString::fromUtf8(problem.what()));
        emit statusMessage(_lastError);
        return;
    }
    _approvals = std::move(loaded);
    _approvalsLoaded = true;
    emit approvalsChanged();
}

void ScaffoldSolveController::startSolve(const ScaffoldSolveRequest& request)
{
    if (_busy) return;
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) {
        _lastError = tr("No fiber store is attached.");
        emit statusMessage(_lastError);
        return;
    }
    if (request.volume.empty()) {
        _lastError = tr("The map has no annotation volume to solve in.");
        emit statusMessage(_lastError);
        return;
    }
    if (!vc::geometry::windingMinCostFlowAvailable()) {
        _lastError = tr("This build has no LEMON min-cost-flow backend.");
        emit statusMessage(_lastError);
        return;
    }
    // A run that measures separations needs the density to measure them on,
    // and says so before it starts rather than quietly measuring none.
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    if (request.measureCrossings && request.measureWindingDistances) {
        QString error;
        if (_windingSampler) sampler = _windingSampler(&error);
        if (!sampler) {
            _lastError = error.isEmpty() ? tr("No normal dataset is bound, so no winding separation can be measured.")
                                         : error;
            emit statusMessage(_lastError);
            return;
        }
        if (!sampler->hasWindingDensity()) {
            _lastError = tr("The bound normal dataset states no winding density (grad_mag): it cannot measure a \n"
                            "winding separation.");
            emit statusMessage(_lastError);
            return;
        }
    }
    _job = std::make_shared<SolveJob>();
    _job->sampler = std::move(sampler);
    _job->request = request;
    _job->attachment = store;
    _job->session = session;
    _job->pool.setMaxThreadCount(1);
    _busy = true;
    _lastError.clear();
    emit busyChanged(true);
    emit statusMessage(tr("Solving the scaffold..."));

    if (!_watcher) {
        _watcher = new QFutureWatcher<void>(this);
        connect(_watcher, &QFutureWatcher<void>::finished, this, [this]() { finishSolve(); });
    }
    auto job = _job;
    _watcher->setFuture(QtConcurrent::run(&job->pool, [job]() {
        try {
            ScaffoldLoadOptions load;
            load.volume = job->request.volume;
            load.fallbackHandedness = job->request.fallbackHandedness;
            ScaffoldSolveOptions options;
            options.links.approvalStates = job->request.approvalStates;
            options.links.includeAuto = job->request.includeAuto;
            options.links.weight = job->request.manualWeight;
            options.fixApprovedSubgraphs = job->request.fixApprovedSubgraphs;
            options.flow.backend = vc::geometry::parseWindingFlowBackend(job->request.lemonBackend);
            options.workers = std::max<std::size_t>(1, job->request.workers);
            ScaffoldRunRecord record;
            record.runUuid = vc::objects::util::newUuid();
            job->session->readStore(*job->attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
                const auto graph = loadScaffoldGraph(reader, load);
                const ScaffoldGeometry geometry(graph);
                // The measured stages, when the run asked for them. They run
                // on this worker, in the frame the map was built in.
                std::optional<ScaffoldNormals> normals;
                std::vector<ScaffoldEvent> events;
                std::vector<ScaffoldConstraint> measured;
                ScaffoldNormalOptions normalOptions;
                ScaffoldEventOptions eventOptions;
                ScaffoldMeasuredOptions measuredOptions;
                ScaffoldSignOptions signOptions;
                ScaffoldEventReport eventReport;
                ScaffoldMeasuredReport measuredReport;
                ScaffoldSignResult signs;
                ScaffoldApprovalReport approvals;
                ScaffoldRunStages stages;
                if (job->request.measureCrossings) {
                    normalOptions.linkApprovalStates = job->request.approvalStates;
                    normalOptions.includeAutoLinks = job->request.includeAuto;
                    normalOptions.workers = options.workers;
                    eventOptions.measureWindingDistances = job->request.measureWindingDistances;
                    eventOptions.workers = options.workers;
                    measuredOptions.perpendicularWeight = job->request.perpendicularWeight;
                    measuredOptions.parallelWeight = job->request.parallelWeight;
                    measuredOptions.enclosureWeight = job->request.enclosureWeight;
                    signOptions.sign.backend = job->request.signBackend == "ceres"
                                                   ? vc::fiber_tracer::FiberSignBackend::Ceres
                                                   : vc::fiber_tracer::FiberSignBackend::Irls;
                    signOptions.sign.threads = int(options.workers);
                    ScaffoldWindingDistances distances;
                    if (auto sampler = job->sampler)
                        distances = [sampler](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
                                              double step, int workers) {
                            return sampler->normalAlignedWindingDistancesBatch(connectors, step, workers);
                        };
                    normals.emplace(graph, geometry, normalOptions);
                    events = scaffoldEvents(graph, geometry, *normals, eventOptions, distances, eventReport);
                    signs = scaffoldNormalSigns(graph, *normals, events, signOptions);
                    measured =
                        scaffoldMeasuredConstraints(graph, geometry, events, signs.sign, measuredOptions, measuredReport);
                    applyScaffoldConstraintApprovals(graph, geometry, measured,
                                                     vc::objects::readFiberConstraintApprovals(reader), approvals);
                    stages.normalOptions = &normalOptions;
                    stages.normals = &normals->report();
                    stages.eventOptions = &eventOptions;
                    stages.events = &eventReport;
                    stages.measuredOptions = &measuredOptions;
                    stages.measured = &measuredReport;
                    stages.signOptions = &signOptions;
                    stages.signs = &signs;
                }
                const auto result = solveScaffoldWinding(graph, geometry, options, measured);
                record.input = {graph.source.storeUuid, graph.source.generation, graph.source.commitSequence};
                job->rows = scaffoldSolveRunRows(graph, result, options, record, stages);
                job->runUuid = record.runUuid;
                job->summary = QObject::tr("%1 fibers in %2 components, %3 turns of residual, %4 violated of %5 \n"
                                           "constraints, %6 approved subgraphs, %7 s")
                                   .arg(graph.fibers.size())
                                   .arg(result.components)
                                   .arg(result.objective, 0, 'f', 3)
                                   .arg(result.violatedConstraints)
                                   .arg(result.constraints.size())
                                   .arg(result.approvedSubgraphs)
                                   .arg(result.seconds, 0, 'f', 3);
                if (job->request.measureCrossings)
                    job->summary += QObject::tr("; %1 crossings, %2 measured relations, %3 enclosures, %4 signs from \n"
                                                "crossings")
                                        .arg(eventReport.events)
                                        .arg(measuredReport.used)
                                        .arg(measuredReport.enclosureArcs)
                                        .arg(signs.decided);
            });
        } catch (const std::exception& problem) {
            job->error = QString::fromUtf8(problem.what());
        }
    }));
}

void ScaffoldSolveController::finishSolve()
{
    auto job = _job;
    _job.reset();
    _busy = false;
    emit busyChanged(false);
    if (!job) return;
    if (!job->error.isEmpty()) {
        _lastError = tr("The solve failed: %1").arg(job->error);
        emit statusMessage(_lastError);
        return;
    }
    // The rows are written from the GUI thread, like every other store edit
    // this workspace makes.
    try {
        auto request = job->session->newRequest(*job->attachment);
        for (const auto& table : job->rows.tables)
            request.operations.push_back(vc::objects::CreateExtensionTableOp{table});
        for (auto& [table, rows] : job->rows.rows)
            request.operations.push_back(vc::objects::AppendExtensionRowsOp{table, rows});
        job->session->submitDurably(*job->attachment, request);
    } catch (const std::exception& problem) {
        _lastError = tr("The solve could not be saved: %1").arg(QString::fromUtf8(problem.what()));
        emit statusMessage(_lastError);
        return;
    }
    reloadRuns();
    selectRun(job->runUuid);
    reloadConstraintApprovals();
    if (_approvalsLoaded) loadApprovals();
    emit statusMessage(job->summary);
}

namespace {
// The store's kind of a stored residual's constraint, or nullopt for a manual
// link (whose approval is its edge record's review state).
std::optional<vc::objects::FiberConstraintKind> storedConstraintKind(const ScaffoldStoredResidual& residual)
{
    switch (residual.kind) {
    case std::int64_t(ScaffoldConstraintKind::Perpendicular): return vc::objects::FiberConstraintKind::Perpendicular;
    case std::int64_t(ScaffoldConstraintKind::Parallel): return vc::objects::FiberConstraintKind::Parallel;
    case std::int64_t(ScaffoldConstraintKind::Enclosure): return vc::objects::FiberConstraintKind::Enclosure;
    default: return std::nullopt;
    }
}
std::optional<vc::objects::FiberConstraintIdentity> storedConstraintIdentity(const ScaffoldStoredResidual& residual)
{
    const auto kind = storedConstraintKind(residual);
    if (!kind || residual.fiberA.empty() || residual.fiberB.empty()) return std::nullopt;
    try {
        return vc::objects::fiberConstraintIdentity(residual.fiberA, residual.arcA, residual.fiberB, residual.arcB,
                                                    *kind);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}
} // namespace

void ScaffoldSolveController::reloadConstraintApprovals()
{
    _constraintApprovals.clear();
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (store && session) {
        try {
            session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
                for (auto& row : vc::objects::liveFiberConstraintApprovals(
                         vc::objects::readFiberConstraintApprovals(reader)))
                    _constraintApprovals.insert_or_assign(row.identity, row);
            });
        } catch (const std::exception& problem) {
            _lastError = tr("The constraint approvals could not be read: %1").arg(QString::fromUtf8(problem.what()));
            emit statusMessage(_lastError);
        }
    }
    emit constraintApprovalsChanged();
}

bool ScaffoldSolveController::constraintApproved(const ScaffoldStoredResidual& residual) const
{
    const auto identity = storedConstraintIdentity(residual);
    return identity && _constraintApprovals.count(*identity) != 0;
}

QString ScaffoldSolveController::setConstraintApproval(const ScaffoldStoredResidual& residual, bool approved)
{
    using namespace vc::objects;
    const auto identity = storedConstraintIdentity(residual);
    if (!identity)
        return tr("A link's approval is its review state; only a measured crossing is approved here.");
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) return tr("No fiber store is attached.");
    const auto existing = _constraintApprovals.find(*identity);
    if (!approved && existing == _constraintApprovals.end())
        return tr("This constraint carries no approval to revoke.");

    auto request = session->newRequest(*store);
    const std::string now = QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();
    const std::string host = QSysInfo::machineHostName().toStdString();
    std::vector<ExtensionRow> values;
    std::vector<ObjectRevision> revisions;
    QString error;
    try {
        session->readStore(*store, [&](const StoreReader& reader, const auto&) {
            if (!approved) {
                values.push_back(fiberConstraintApprovalRow(
                    revokeFiberConstraintApproval(existing->second, request.author, now)));
                return;
            }
            // Both fibers as the store holds them now: the approval records
            // their versions and the fingerprints of the two covering spans.
            std::array<FiberPieceContent, 2> contents;
            std::array<std::int64_t, 2> versions{0, 0};
            std::array<std::size_t, 2> spans{0, 0};
            const std::array<std::string, 2> uuids{identity->fiberA, identity->fiberB};
            const std::array<double, 2> arcs{identity->fiberA == residual.fiberA ? residual.arcA : residual.arcB,
                                             identity->fiberA == residual.fiberA ? residual.arcB : residual.arcA};
            for (std::size_t side = 0; side < 2; ++side) {
                const auto record = reader.currentObject(uuids[side]);
                if (!record) { error = tr("A fiber of this constraint is no longer in the store."); return; }
                const auto found = reader.objectsAtVersions({record->reference()});
                if (found.empty() || !found.front() || found.front()->deleted) {
                    error = tr("A fiber of this constraint cannot be read.");
                    return;
                }
                revisions.push_back(*found.front());
                contents[side] = fiberPieceContent(reader, *found.front());
                versions[side] = found.front()->version;
                const auto span = vc::fiber_graph::scaffoldSpanCoveringArc(contents[side], arcs[side]);
                if (!span) { error = tr("A fiber of this constraint has no span covering its anchor."); return; }
                spans[side] = *span;
            }
            values.push_back(fiberConstraintApprovalRow(
                approveFiberConstraint(*identity, contents[0], spans[0], versions[0], contents[1], spans[1],
                                       versions[1], request.author, now, host)));
        });
    } catch (const std::exception& problem) {
        return tr("The store could not be read: %1").arg(QString::fromUtf8(problem.what()));
    }
    if (!error.isEmpty()) return error;
    if (values.empty()) return tr("Nothing to write.");
    for (const auto& revision : revisions) expectObject(request, revision);
    request.operations.push_back(CreateExtensionTableOp{fiberConstraintApprovalTable()});
    request.operations.push_back(AppendExtensionRowsOp{kFiberConstraintApprovalTable, std::move(values)});
    try {
        session->submitDurably(*store, request);
    } catch (const std::exception& problem) {
        return tr("The approval could not be stored: %1").arg(QString::fromUtf8(problem.what()));
    }
    reloadConstraintApprovals();
    return {};
}

ScaffoldSolverPanel::ScaffoldSolverPanel(ScaffoldSolveController* controller, QWidget* parent)
    : QWidget(parent)
    , _controller(controller)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    auto* form = new QFormLayout();
    form->setContentsMargins(0, 0, 0, 0);
    form->setSpacing(2);

    auto* states = new QWidget(this);
    auto* statesLayout = new QHBoxLayout(states);
    statesLayout->setContentsMargins(0, 0, 0, 0);
    _approvedCheck = new QCheckBox(tr("approved"), states);
    _approvedCheck->setObjectName(QStringLiteral("fiberMapSolveApprovedLinks"));
    _approvedCheck->setChecked(true);
    _pendingCheck = new QCheckBox(tr("pending"), states);
    _pendingCheck->setObjectName(QStringLiteral("fiberMapSolvePendingLinks"));
    _pendingCheck->setChecked(true);
    statesLayout->addWidget(_approvedCheck);
    statesLayout->addWidget(_pendingCheck);
    statesLayout->addStretch(1);
    states->setToolTip(tr("Which links the solve uses, by their review state.\n"
                          "A rejected link is never used."));
    form->addRow(tr("Links"), states);

    _autoCheck = new QCheckBox(tr("Use automatic links"), this);
    _autoCheck->setObjectName(QStringLiteral("fiberMapSolveAutoLinks"));
    _autoCheck->setToolTip(tr("Also use links the automatic linker proposed and nobody reviewed."));
    form->addRow(QString(), _autoCheck);

    _manualWeightSpin = new QDoubleSpinBox(this);
    _manualWeightSpin->setObjectName(QStringLiteral("fiberMapSolveManualWeight"));
    _manualWeightSpin->setRange(0.0, 1000.0);
    _manualWeightSpin->setDecimals(2);
    _manualWeightSpin->setSingleStep(0.25);
    _manualWeightSpin->setValue(1.0);
    _manualWeightSpin->setToolTip(tr("Cost of one winding of error on a link. Only relative weights\n"
                                     "matter while links are the only constraints."));
    form->addRow(tr("Link weight"), _manualWeightSpin);

    _fixApprovedCheck = new QCheckBox(tr("Fix approved subgraphs"), this);
    _fixApprovedCheck->setObjectName(QStringLiteral("fiberMapSolveFixApproved"));
    _fixApprovedCheck->setChecked(true);
    _fixApprovedCheck->setToolTip(tr("Solve the approved constraints alone first and hold their relative\n"
                                     "windings fixed, so an approved relation is never traded away.\n"
                                     "Approved constraints that contradict each other stop the run."));
    form->addRow(QString(), _fixApprovedCheck);

    _backendCombo = new QComboBox(this);
    _backendCombo->setObjectName(QStringLiteral("fiberMapSolveBackend"));
    _backendCombo->addItem(tr("network simplex"), QStringLiteral("network_simplex"));
    _backendCombo->addItem(tr("cost scaling"), QStringLiteral("cost_scaling"));
    _backendCombo->setToolTip(tr("The min-cost-flow backend. Both solve the same problem exactly."));
    form->addRow(tr("Backend"), _backendCombo);

    _crossingsCheck = new QCheckBox(tr("Measure the fibers' crossings"), this);
    _crossingsCheck->setObjectName(QStringLiteral("fiberMapSolveCrossings"));
    _crossingsCheck->setToolTip(tr("Also use the crossings of the annotated fibers themselves: their strip\n"
                                   "normals, the signs those crossings decide, and the perpendicular,\n"
                                   "parallel and enclosure relations they assert. This reads the volume and\n"
                                   "takes seconds rather than milliseconds."));
    form->addRow(tr("Crossings"), _crossingsCheck);

    _windingDistanceCheck = new QCheckBox(tr("Measure winding separations"), this);
    _windingDistanceCheck->setObjectName(QStringLiteral("fiberMapSolveWindingDistances"));
    _windingDistanceCheck->setChecked(true);
    _windingDistanceCheck->setToolTip(tr("Measure how many windings apart each crossing is, on the bound\n"
                                         "Lasagna normal dataset's winding density. Without it a crossing\n"
                                         "asserts no step and only witnesses an enclosure."));
    form->addRow(QString(), _windingDistanceCheck);

    const auto weightSpin = [this](const char* name, const QString& tip) {
        auto* spin = new QDoubleSpinBox(this);
        spin->setObjectName(QString::fromLatin1(name));
        spin->setRange(0.0, 1000.0);
        spin->setDecimals(2);
        spin->setSingleStep(0.25);
        spin->setValue(1.0);
        spin->setToolTip(tip);
        return spin;
    };
    _perpendicularWeightSpin = weightSpin("fiberMapSolvePerpendicularWeight",
        tr("Cost of one winding of error on a measured perpendicular crossing."));
    form->addRow(tr("Perpendicular weight"), _perpendicularWeightSpin);
    _parallelWeightSpin = weightSpin("fiberMapSolveParallelWeight",
        tr("Cost of one winding of error on a measured parallel relation."));
    form->addRow(tr("Parallel weight"), _parallelWeightSpin);
    _enclosureWeightSpin = weightSpin("fiberMapSolveEnclosureWeight",
        tr("Cost of one winding of error on an enclosure relation: the exact step a pair's\n"
           "crossings bound from both sides. Zero leaves enclosures out."));
    form->addRow(tr("Enclosure weight"), _enclosureWeightSpin);

    _signBackendCombo = new QComboBox(this);
    _signBackendCombo->setObjectName(QStringLiteral("fiberMapSolveSignBackend"));
    _signBackendCombo->addItem(tr("IRLS"), QStringLiteral("irls"));
    _signBackendCombo->addItem(tr("Ceres"), QStringLiteral("ceres"));
    _signBackendCombo->setToolTip(tr("Which backend decides the normal signs of the fibers no anchor fixed.\n"
                                     "Both descend the same objective."));
    form->addRow(tr("Sign backend"), _signBackendCombo);
    layout->addLayout(form);

    _runButton = new QPushButton(tr("Solve"), this);
    _runButton->setObjectName(QStringLiteral("fiberMapSolveRun"));
    _runButton->setToolTip(tr("Read the store's fibers, links, approvals and umbilicus, assign every\n"
                              "fiber an integer winding, and save the run in the store."));
    layout->addWidget(_runButton);

    _statusLabel = new QLabel(this);
    _statusLabel->setObjectName(QStringLiteral("fiberMapSolveStatus"));
    _statusLabel->setWordWrap(true);
    _statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_statusLabel);
    layout->addStretch(1);

    connect(_runButton, &QPushButton::clicked, this, [this]() { emit solveRequested(); });
    for (auto* box : {_approvedCheck, _pendingCheck, _crossingsCheck}) {
        connect(box, &QCheckBox::toggled, this, [this]() { updateEnabled(); });
    }
    if (_controller) {
        connect(_controller, &ScaffoldSolveController::busyChanged, this, [this]() { updateEnabled(); });
        connect(_controller, &ScaffoldSolveController::statusMessage, this,
                [this](const QString& message) { _statusLabel->setText(message); });
    }
    updateEnabled();
}

void ScaffoldSolverPanel::setFrame(const std::string& volume, std::optional<int> fallbackHandedness)
{
    _volume = volume;
    _fallbackHandedness = fallbackHandedness;
    updateEnabled();
}

void ScaffoldSolverPanel::updateEnabled()
{
    const bool states = _approvedCheck->isChecked() || _pendingCheck->isChecked();
    const bool busy = _controller && _controller->busy();
    _runButton->setEnabled(states && !busy && !_volume.empty());
    if (_volume.empty()) _runButton->setToolTip(tr("The map has no annotation volume to solve in."));
    const bool crossings = _crossingsCheck->isChecked();
    for (auto* widget : std::initializer_list<QWidget*>{_windingDistanceCheck, _perpendicularWeightSpin,
                                                        _parallelWeightSpin, _enclosureWeightSpin, _signBackendCombo})
        widget->setEnabled(crossings);
}

ScaffoldSolveRequest ScaffoldSolverPanel::request() const
{
    ScaffoldSolveRequest request;
    request.approvalStates.clear();
    if (_approvedCheck->isChecked()) request.approvalStates.push_back("approved");
    if (_pendingCheck->isChecked()) request.approvalStates.push_back("pending");
    request.includeAuto = _autoCheck->isChecked();
    request.manualWeight = _manualWeightSpin->value();
    request.fixApprovedSubgraphs = _fixApprovedCheck->isChecked();
    request.lemonBackend = _backendCombo->currentData().toString().toStdString();
    request.volume = _volume;
    request.fallbackHandedness = _fallbackHandedness;
    request.measureCrossings = _crossingsCheck->isChecked();
    request.measureWindingDistances = _windingDistanceCheck->isChecked();
    request.perpendicularWeight = _perpendicularWeightSpin->value();
    request.parallelWeight = _parallelWeightSpin->value();
    request.enclosureWeight = _enclosureWeightSpin->value();
    request.signBackend = _signBackendCombo->currentData().toString().toStdString();
    request.workers = std::size_t(std::max(1, QThread::idealThreadCount()));
    return request;
}

bool ScaffoldSolverPanel::fixApprovedSubgraphs() const { return _fixApprovedCheck->isChecked(); }

ScaffoldRunsPanel::ScaffoldRunsPanel(ScaffoldSolveController* controller, QWidget* parent)
    : QWidget(parent)
    , _controller(controller)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    _tree = new QTreeWidget(this);
    _tree->setObjectName(QStringLiteral("fiberMapSolvesTree"));
    _tree->setColumnCount(kRunColumnCount);
    _tree->setHeaderLabels({tr("method"), tr("fibers"), tr("comp"), tr("L1"), tr("bad"), tr("max")});
    _tree->setUniformRowHeights(true);
    _tree->setRootIsDecorated(false);
    _tree->setSelectionMode(QAbstractItemView::SingleSelection);
    _tree->header()->setStretchLastSection(true);
    _tree->setToolTip(tr("Every solve saved in this store, newest first. Selecting one lays the\n"
                         "map out from it: each fiber at the winding that solve gave it.\n"
                         "L1 is the weighted residual in turns, bad the violated constraints,\n"
                         "max the largest single residual in turns."));
    layout->addWidget(_tree, 1);
    _followCheck = new QCheckBox(tr("Lay the map out from this solve"), this);
    _followCheck->setObjectName(QStringLiteral("fiberMapFollowSolve"));
    _followCheck->setToolTip(tr("Walk exactly the links the selected solve walked: the links it did\n"
                                "not use carry no constraint here, and the ones it could not satisfy\n"
                                "are drawn suspect. The map's own crossing evidence still places the\n"
                                "fibers it says nothing about."));
    layout->addWidget(_followCheck);

    if (_controller) {
        connect(_controller, &ScaffoldSolveController::runsChanged, this, [this]() { rebuild(); });
        connect(_tree, &QTreeWidget::itemSelectionChanged, this, [this]() { applySelection(); });
    }
    connect(_followCheck, &QCheckBox::toggled, this, [this]() { emit followChanged(); });
    rebuild();
}

bool ScaffoldRunsPanel::follow() const { return _followCheck && _followCheck->isChecked(); }

void ScaffoldRunsPanel::rebuild()
{
    if (!_controller) return;
    const auto previous = _controller->selected() ? _controller->selected()->run.runUuid : std::string();
    _rebuilding = true;
    _tree->clear();
    for (const auto& run : _controller->runs()) {
        const auto count = [&](const char* name) -> qlonglong {
            const auto found = run.counts.find(name);
            return found == run.counts.end() ? 0 : qlonglong(found->second);
        };
        const auto metric = [&](const char* name) {
            const auto found = run.metrics.find(name);
            return found == run.metrics.end() ? 0. : found->second;
        };
        auto* item = new QTreeWidgetItem(_tree);
        item->setText(kRunMethodColumn, QString::fromStdString(run.method));
        item->setText(kRunFibersColumn, QString::number(count("fibers")));
        item->setText(kRunComponentsColumn, QString::number(count("components")));
        item->setText(kRunObjectiveColumn, QString::number(metric("weighted_l1"), 'f', 3));
        item->setText(kRunViolatedColumn, QString::number(count("violated_constraints")));
        item->setText(kRunMaxResidualColumn, QString::number(metric("max_residual"), 'f', 3));
        item->setData(kRunMethodColumn, Qt::UserRole, QString::fromStdString(run.runUuid));
        item->setToolTip(kRunMethodColumn, oneLine(run) + QStringLiteral("\n") +
                                               QString::fromStdString(run.settingsJson));
        if (run.runUuid == previous) item->setSelected(true);
    }
    for (int column = 0; column < kRunColumnCount - 1; ++column) _tree->resizeColumnToContents(column);
    _rebuilding = false;
}

void ScaffoldRunsPanel::applySelection()
{
    if (_rebuilding || !_controller) return;
    const auto items = _tree->selectedItems();
    if (items.isEmpty()) {
        _controller->selectRun({});
        return;
    }
    _controller->selectRun(items.front()->data(kRunMethodColumn, Qt::UserRole).toString().toStdString());
}

ScaffoldApprovalPanel::ScaffoldApprovalPanel(ScaffoldSolveController* controller, QWidget* parent)
    : QWidget(parent)
    , _controller(controller)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(3);

    _highlightFibersCheck = new QCheckBox(tr("Highlight approved fibers"), this);
    _highlightFibersCheck->setObjectName(QStringLiteral("fiberMapHighlightApprovedFibers"));
    _highlightFibersCheck->setToolTip(tr("Outline the fibers whose every span carries a current geometry\n"
                                         "approval."));
    _highlightPiecesCheck = new QCheckBox(tr("Highlight approved pieces"), this);
    _highlightPiecesCheck->setObjectName(QStringLiteral("fiberMapHighlightApprovedPieces"));
    _highlightPiecesCheck->setToolTip(tr("Outline every span that carries a current geometry approval, and\n"
                                         "mark the spans whose normals are approved too."));
    layout->addWidget(_highlightFibersCheck);
    layout->addWidget(_highlightPiecesCheck);

    auto* orderRow = new QWidget(this);
    auto* orderLayout = new QHBoxLayout(orderRow);
    orderLayout->setContentsMargins(0, 0, 0, 0);
    orderLayout->addWidget(new QLabel(tr("Order by"), orderRow));
    _orderCombo = new QComboBox(orderRow);
    _orderCombo->setObjectName(QStringLiteral("fiberMapApprovalOrder"));
    _orderCombo->addItem(tr("smallest cycle"), int(ScaffoldApprovalOrder::Cycle));
    _orderCombo->addItem(tr("highest cost"), int(ScaffoldApprovalOrder::Cost));
    _orderCombo->addItem(tr("cost per hop"), int(ScaffoldApprovalOrder::Joint));
    _orderCombo->setToolTip(tr("smallest cycle: the shortest loop a violation lies on first, ties by\n"
                               "the shorter stretch - the most specific suspects.\n"
                               "highest cost: the most expensive violation first, ties by the shorter\n"
                               "cycle and stretch.\n"
                               "cost per hop: cost divided by the cycle's constraints, which favours an\n"
                               "expensive violation on a short loop."));
    orderLayout->addWidget(_orderCombo, 1);
    layout->addWidget(orderRow);

    _tree = new QTreeWidget(this);
    _tree->setObjectName(QStringLiteral("fiberMapApprovalTree"));
    _tree->setColumnCount(kColumnCount);
    _tree->setHeaderLabels({tr("what"), tr("fiber"), tr("where"), tr("cost"), tr("cycle"), tr("n")});
    _tree->setUniformRowHeights(true);
    _tree->setRootIsDecorated(false);
    _tree->setSelectionMode(QAbstractItemView::SingleSelection);
    _tree->header()->setStretchLastSection(true);
    _tree->setToolTip(tr("What the selected solve could not satisfy. A violated constraint is\n"
                         "listed with the shortest cycle it lies on; a stretch is a piece of one\n"
                         "fiber between two constraints of such a cycle, with the share of the\n"
                         "violated cost it is accused of.\n"
                         "Selecting a row shows it on the map; double-clicking a stretch opens it\n"
                         "as a piece, where its geometry and normals can be approved."));
    layout->addWidget(_tree, 1);

    _approveButton = new QPushButton(tr("Approve constraint"), this);
    _approveButton->setObjectName(QStringLiteral("fiberMapApproveConstraint"));
    _approveButton->setEnabled(false);
    _approveButton->setToolTip(tr("Record that this measured crossing is right. An approved constraint is\n"
                                  "held rigid by the next solve, and the record is invalidated if either\n"
                                  "fiber's covering span is edited. A link is not approved here: its\n"
                                  "approval is its review state in the annotation."));
    layout->addWidget(_approveButton);

    _summaryLabel = new QLabel(this);
    _summaryLabel->setObjectName(QStringLiteral("fiberMapApprovalSummary"));
    _summaryLabel->setWordWrap(true);
    _summaryLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_summaryLabel);

    connect(_approveButton, &QPushButton::clicked, this, [this]() {
        const auto* residual = residualOf(_tree->currentItem());
        if (!residual || !_controller) return;
        const bool approved = _controller->constraintApproved(*residual);
        const auto error = _controller->setConstraintApproval(*residual, !approved);
        if (!error.isEmpty()) {
            _summaryLabel->setText(error);
            return;
        }
        applySelection();
    });
    connect(_orderCombo, &QComboBox::currentIndexChanged, this, [this]() { rebuild(); });
    connect(_tree, &QTreeWidget::itemSelectionChanged, this, [this]() { applySelection(); });
    connect(_tree, &QTreeWidget::itemDoubleClicked, this,
            [this](QTreeWidgetItem* item, int) { activateRow(item); });
    for (auto* box : {_highlightFibersCheck, _highlightPiecesCheck}) {
        connect(box, &QCheckBox::toggled, this, [this](bool on) {
            if (on && _controller && !_controller->approvalsLoaded()) _controller->loadApprovals();
            emit highlightModeChanged();
        });
    }
    if (_controller) {
        connect(_controller, &ScaffoldSolveController::selectionChanged, this, [this]() { rebuild(); });
        connect(_controller, &ScaffoldSolveController::approvalsChanged, this,
                [this]() { emit highlightModeChanged(); });
        connect(_controller, &ScaffoldSolveController::constraintApprovalsChanged, this,
                [this]() { applySelection(); });
    }
    rebuild();
}

bool ScaffoldApprovalPanel::highlightApprovedFibers() const { return _highlightFibersCheck->isChecked(); }
bool ScaffoldApprovalPanel::highlightApprovedPieces() const { return _highlightPiecesCheck->isChecked(); }

void ScaffoldApprovalPanel::setLabelLookup(LabelLookup lookup)
{
    _labels = std::move(lookup);
    rebuild();
}

void ScaffoldApprovalPanel::rebuild()
{
    _rebuilding = true;
    _tree->clear();
    const auto* view = _controller ? _controller->selected() : nullptr;
    if (!view) {
        _summaryLabel->setText(tr("Select a solve to see what it could not satisfy."));
        _rebuilding = false;
        return;
    }
    const auto label = [this](const std::string& uuid) {
        if (_labels) {
            const auto text = _labels(uuid);
            if (!text.isEmpty()) return text;
        }
        return QString::fromStdString(uuid).left(8);
    };
    const auto kindName = [](std::int64_t kind) {
        switch (kind) {
        case std::int64_t(ScaffoldConstraintKind::Perpendicular): return tr("perpendicular");
        case std::int64_t(ScaffoldConstraintKind::Parallel): return tr("parallel");
        case std::int64_t(ScaffoldConstraintKind::Enclosure): return tr("enclosure");
        default: return tr("constraint");
        }
    };
    // One row per violated constraint and per accused stretch, ordered by the
    // chosen key with the documented secondary keys.
    struct Row {
        ScaffoldApprovalRow keys;
        bool piece = false;
        QStringList text;
        std::string fiberUuid, otherUuid;
        std::int64_t controlFrom = -1, controlTo = -1;
        // Index into the run's residuals for a constraint row, -1 otherwise.
        std::int64_t residual = -1;
    };
    std::vector<Row> rows;
    for (std::size_t index = 0; index < view->residuals.size(); ++index) {
        const auto& residual = view->residuals[index];
        Row row;
        row.piece = false;
        row.residual = std::int64_t(index);
        row.keys.cost = residual.weight * residual.residual;
        row.keys.cycle = residual.cycleHops.value_or(-1);
        row.keys.length = residual.cyclePieceLength;
        row.fiberUuid = residual.fiberA;
        row.otherUuid = residual.fiberB;
        row.text << (residual.edgeUuid.empty() ? kindName(residual.kind) : tr("link"))
                 << tr("%1 - %2").arg(label(residual.fiberA), label(residual.fiberB))
                 << tr("%1 turns off").arg(residual.residual, 0, 'f', 2)
                 << QString::number(row.keys.cost, 'f', 3)
                 << (row.keys.cycle > 0 ? QString::number(row.keys.cycle) : tr("none"))
                 << (residual.redundancy ? QString::number(*residual.redundancy) : QString());
        rows.push_back(std::move(row));
    }
    for (const auto& piece : view->pieces) {
        Row row;
        row.piece = true;
        row.keys.cost = piece.score;
        row.keys.cycle = piece.cycleHops;
        row.keys.length = piece.arcTo - piece.arcFrom;
        row.fiberUuid = piece.fiberUuid;
        row.controlFrom = piece.controlFrom.value_or(-1);
        row.controlTo = piece.controlTo.value_or(-1);
        const QString where = row.controlFrom >= 0
            ? tr("cp %1-%2").arg(row.controlFrom).arg(row.controlTo)
            : tr("%1 vx").arg(row.keys.length, 0, 'f', 0);
        row.text << tr("stretch") << label(piece.fiberUuid) << where
                 << QString::number(row.keys.cost, 'f', 3)
                 << (row.keys.cycle > 0 ? QString::number(row.keys.cycle) : tr("none"))
                 << QString::number(piece.constraints);
        rows.push_back(std::move(row));
    }
    const auto order = ScaffoldApprovalOrder(_orderCombo->currentData().toInt());
    std::stable_sort(rows.begin(), rows.end(),
                     [&](const Row& x, const Row& y) { return scaffoldApprovalRowBefore(x.keys, y.keys, order); });
    for (const auto& row : rows) {
        auto* item = new QTreeWidgetItem(_tree, row.text);
        item->setData(kKindColumn, Qt::UserRole, row.piece);
        item->setData(kFiberColumn, Qt::UserRole, QString::fromStdString(row.fiberUuid));
        item->setData(kWhereColumn, Qt::UserRole, QString::fromStdString(row.otherUuid));
        item->setData(kCostColumn, Qt::UserRole, QVariant::fromValue(qlonglong(row.controlFrom)));
        item->setData(kCycleColumn, Qt::UserRole, QVariant::fromValue(qlonglong(row.controlTo)));
        item->setData(kCountColumn, Qt::UserRole, QVariant::fromValue(qlonglong(row.residual)));
    }
    for (int column = 0; column < kColumnCount - 1; ++column) _tree->resizeColumnToContents(column);
    _summaryLabel->setText(tr("%1 violated constraints, %2 accused stretches")
                               .arg(view->residuals.size())
                               .arg(view->pieces.size()));
    _rebuilding = false;
}

const ScaffoldStoredResidual* ScaffoldApprovalPanel::residualOf(const QTreeWidgetItem* item) const
{
    const auto* view = _controller ? _controller->selected() : nullptr;
    if (!item || !view) return nullptr;
    const auto index = item->data(kCountColumn, Qt::UserRole).toLongLong();
    if (index < 0 || std::size_t(index) >= view->residuals.size()) return nullptr;
    return &view->residuals[std::size_t(index)];
}

void ScaffoldApprovalPanel::applySelection()
{
    if (_rebuilding) return;
    const auto items = _tree->selectedItems();
    // The approve button follows the selection: a measured crossing can be
    // approved or revoked, a link and a stretch cannot.
    const auto* residual = items.isEmpty() ? nullptr : residualOf(items.front());
    const bool measured = residual && residual->edgeUuid.empty() &&
                          residual->kind != std::int64_t(ScaffoldConstraintKind::Manual);
    const bool approved = measured && _controller && _controller->constraintApproved(*residual);
    _approveButton->setEnabled(measured);
    _approveButton->setText(approved ? tr("Revoke approval") : tr("Approve constraint"));
    if (items.isEmpty()) {
        emit highlightRequested({});
        return;
    }
    const auto* item = items.front();
    ScaffoldHighlight highlight;
    const auto fiber = item->data(kFiberColumn, Qt::UserRole).toString().toStdString();
    const auto other = item->data(kWhereColumn, Qt::UserRole).toString().toStdString();
    if (!fiber.empty()) highlight.fiberUuids.push_back(fiber);
    if (!other.empty()) highlight.fiberUuids.push_back(other);
    if (item->data(kKindColumn, Qt::UserRole).toBool()) {
        highlight.controlFrom = item->data(kCostColumn, Qt::UserRole).toLongLong();
        highlight.controlTo = item->data(kCycleColumn, Qt::UserRole).toLongLong();
    }
    emit highlightRequested(highlight);
}

void ScaffoldApprovalPanel::activateRow(QTreeWidgetItem* item)
{
    if (!item || !item->data(kKindColumn, Qt::UserRole).toBool()) return;
    const auto fiber = item->data(kFiberColumn, Qt::UserRole).toString().toStdString();
    const auto from = item->data(kCostColumn, Qt::UserRole).toLongLong();
    const auto to = item->data(kCycleColumn, Qt::UserRole).toLongLong();
    if (fiber.empty() || from < 0 || to <= from) return;
    emit pieceActivated(fiber, from, to);
}

} // namespace vc3d::fiber_map
