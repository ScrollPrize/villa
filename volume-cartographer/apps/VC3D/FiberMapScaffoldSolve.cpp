#include "FiberMapScaffoldSolve.hpp"

#include "vc/fiber_graph/ScaffoldCache.hpp"

#include "CState.hpp"

#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/Transactions.hpp>

#include <QCheckBox>
#include <QDateTime>
#include <QEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QStandardItemModel>
#include <QSysInfo>
#include <QThread>
#include <QThreadPool>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrent>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace vc3d::fiber_map
{
namespace
{
using namespace vc::fiber_graph;

// A duration as the dock states it: seconds below a minute, minutes and
// seconds below an hour, hours and minutes above.
QString scaffoldDurationText(double seconds)
{
    const auto whole = std::int64_t(seconds + 0.5);
    if (whole < 60) return QObject::tr("%1 s").arg(whole);
    if (whole < 3600) return QObject::tr("%1 min %2 s").arg(whole / 60).arg(whole % 60);
    return QObject::tr("%1 h %2 min").arg(whole / 3600).arg((whole % 3600) / 60);
}

// Arc length of each control point along the fiber's line points (voxels).
// Empty when the controls cannot be placed on the line.
std::vector<double> scaffoldControlArcs(const vc::objects::FiberPieceContent& content)
{
    auto indices = content.controlLineIndex;
    if (indices.size() != content.controlIds.size()) {
        auto copy = content;
        copy.resolveLineIndices();
        indices = copy.controlLineIndex;
    }
    if (indices.size() != content.controlIds.size() || content.linePoints.size() < 2) return {};
    std::vector<double> cumulative(content.linePoints.size(), 0.);
    for (std::size_t i = 1; i < content.linePoints.size(); ++i) {
        const auto& previous = content.linePoints[i - 1];
        const auto& point = content.linePoints[i];
        const auto dx = point[0] - previous[0], dy = point[1] - previous[1], dz = point[2] - previous[2];
        cumulative[i] = cumulative[i - 1] + std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    std::vector<double> arcs;
    arcs.reserve(indices.size());
    for (const auto index : indices) {
        if (index < 0 || std::size_t(index) >= cumulative.size()) return {};
        arcs.push_back(cumulative[std::size_t(index)]);
    }
    // Controls are stored along the line, so the arcs are nondecreasing; a
    // table that is not says the line indices do not match this content.
    if (!std::is_sorted(arcs.begin(), arcs.end())) return {};
    return arcs;
}

// Sorts by the numeric key a column stored, falling back to the text for the
// columns that carry none. A QTreeWidget compares the display text, which
// would order 10 before 9.
class RankingItem : public QTreeWidgetItem
{
public:
    using QTreeWidgetItem::QTreeWidgetItem;
    bool operator<(const QTreeWidgetItem& other) const override
    {
        const auto column = treeWidget() ? treeWidget()->sortColumn() : 0;
        const auto mine = data(column, Qt::UserRole + 1), theirs = other.data(column, Qt::UserRole + 1);
        if (mine.isValid() && theirs.isValid()) return mine.toDouble() < theirs.toDouble();
        return text(column).localeAwareCompare(other.text(column)) < 0;
    }
};

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
    // Captured on the GUI thread, opened on the worker (startSolve).
    ScaffoldSolveController::WindingSamplerOpener openSampler;
    ScaffoldSolveController::AxisSamplerOpener openAxes;
    // Null when the run measures no winding separations.
    std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    // The prediction's sheet axis, for a fused run.
    vc::fiber_graph::ScaffoldAxisSampler axes;
    // Progress and cancellation, read from the GUI thread while the worker
    // writes it.
    vc::fiber_graph::ScaffoldRunMonitor monitor;
    // Derived results this run may reuse, and what it computed for the next
    // one (vc/fiber_graph/ScaffoldCache.hpp).
    vc::fiber_graph::ScaffoldCacheSink cacheSink;
    // What the fused run's prediction is called; empty when it reads none.
    std::string predictionIdentity;
    // Outputs.
    QString error;
    bool cancelled = false;
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

void ScaffoldSolveController::setAxisSampler(AxisSamplerResolver resolver)
{
    _axisSampler = std::move(resolver);
}

std::optional<vc::objects::AttachmentIdentity> ScaffoldSolveController::attachment() const
{
    if (!_state) return std::nullopt;
    return _state->storeFor("fiber");
}

ScaffoldAnchorSpan ScaffoldSolveController::anchorSpan(const std::string& fiberUuid, double arc) const
{
    ScaffoldAnchorSpan answer;
    const auto found = _approvals.find(fiberUuid);
    if (found == _approvals.end()) return answer;
    const auto& arcs = found->second.controlArcs;
    if (arcs.size() < 2 || !std::isfinite(arc)) return answer;
    // The span between the two controls bracketing the arc, clamped to the
    // fiber's ends: an anchor beyond the last control belongs to the last
    // span, which is the span whose geometry covers it.
    auto upper = std::upper_bound(arcs.begin(), arcs.end(), arc);
    auto span = std::size_t(std::distance(arcs.begin(), upper));
    if (span == 0) span = 1;
    if (span >= arcs.size()) span = arcs.size() - 1;
    answer.known = true;
    answer.span = std::int64_t(span - 1);
    answer.controlFrom = std::int64_t(span - 1);
    answer.controlTo = std::int64_t(span);
    const auto approvedAt = [&](std::size_t aspect) {
        const auto& flags = found->second.approved[aspect];
        return std::size_t(answer.span) < flags.size() && flags[std::size_t(answer.span)] != 0;
    };
    answer.geometryApproved = approvedAt(std::size_t(vc::objects::FiberPieceAspect::Geometry));
    answer.normalsApproved = approvedAt(std::size_t(vc::objects::FiberPieceAspect::Normals));
    return answer;
}

void ScaffoldSolveController::reloadRuns()
{
    _runs.clear();
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) {
        // Nothing was read, so the caller may not take this for the store's
        // answer: the list is listed again once a store is there.
        _runsLoaded = false;
        emit runsChanged();
        return;
    }
    _runsLoaded = true;
    QString error;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            _runs = readScaffoldRuns(reader);
            _storeState = reader.status();
            // When the annotation itself last changed. Taken from the records
            // rather than the change feed: a store may retain no feed at all,
            // and a run's own rows are extension rows, which move no record.
            _lastAnnotationCommit = scaffoldAnnotationCommit(reader);
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

bool ScaffoldSolveController::runStale(const ScaffoldStoredRun& run) const
{
    // A run that recorded no input says nothing about what it read, so it is
    // not claimed to be current either.
    if (run.input.storeUuid.empty() || _storeState.storeUuid.empty()) return false;
    if (run.input.storeUuid != _storeState.storeUuid || run.input.generation != _storeState.generation) return true;
    return _lastAnnotationCommit > run.input.commitSequence;
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

ScaffoldRunDiff ScaffoldSolveController::diffRuns(const std::string& first, const std::string& second,
                                                 QString& error)
{
    error.clear();
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) {
        error = tr("No fiber store is attached.");
        return {};
    }
    std::map<std::string, ScaffoldStoredWinding> here, there;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            here = readScaffoldWindings(reader, first);
            there = readScaffoldWindings(reader, second);
        });
    } catch (const std::exception& problem) {
        error = tr("The solves could not be compared: %1").arg(QString::fromUtf8(problem.what()));
        return {};
    }
    return scaffoldRunDiff(here, there);
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
                const auto revision = reader.currentObject(record.uuid);
                if (!revision) continue;
                const auto content = vc::objects::fiberPieceContent(reader, *revision);
                const auto rows = vc::objects::readFiberPieceApprovals(reader, record.uuid);
                const auto states = vc::objects::fiberPieceApprovalStates(content, rows);
                ScaffoldApprovalSpans entry;
                entry.controlIds = content.controlIds;
                entry.controlArcs = scaffoldControlArcs(content);
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
    // A scope that resolved to nothing is a refusal, not a whole-scaffold run.
    if (request.restricted && request.fiberSubset.empty()) {
        _lastError = tr("The scope is a selected network, but the map has none selected.");
        emit statusMessage(_lastError);
        return;
    }
    if (!vc::geometry::windingMinCostFlowAvailable()) {
        _lastError = tr("This build has no LEMON min-cost-flow backend.");
        emit statusMessage(_lastError);
        return;
    }
    // What a measured run reads is captured here and opened on the worker:
    // opening a dataset (a remote manifest, say) must never block the GUI.
    ScaffoldSolveController::WindingSamplerOpener openSampler;
    if (request.measureCrossings && request.measureWindingDistances && _windingSampler) openSampler = _windingSampler();
    ScaffoldSolveController::AxisSamplerOpener openAxes;
    if (request.measureCrossings && request.normalSource == "fused" && _axisSampler) openAxes = _axisSampler();
    _job = std::make_shared<SolveJob>();
    _job->openSampler = std::move(openSampler);
    _job->openAxes = std::move(openAxes);
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
            load.fiberSubset = job->request.fiberSubset;
            load.monitor = &job->monitor;
            // Coarse progress: the stages this run will enter. Reading the
            // fibers, the constraint assembly, the winding solve and the
            // residual cycles always; a measured run adds the normals, the
            // six phases the crossing extraction reports, the sign solve and
            // the prediction axis when it is fused.
            // A run that measures separations needs the density to measure
            // them on, and a fused run its prediction: both are opened here,
            // and a run that cannot have them stops before it reads a fiber.
            if (job->request.measureCrossings &&
                (job->request.measureWindingDistances || job->request.normalSource == "fused")) {
                job->monitor.enter("opening the datasets");
                if (job->request.measureWindingDistances) {
                    QString error;
                    if (job->openSampler) job->sampler = job->openSampler(&error);
                    if (!job->sampler)
                        throw std::runtime_error(error.isEmpty()
                            ? "No normal dataset is bound, so no winding separation can be measured."
                            : error.toStdString());
                    if (!job->sampler->hasWindingDensity())
                        throw std::runtime_error("The bound normal dataset states no winding density (grad_mag): it "
                                                 "cannot measure a winding separation.");
                }
                job->monitor.checkpoint();
                if (job->request.normalSource == "fused") {
                    QString error;
                    if (job->openAxes) job->axes = job->openAxes(&error, &job->predictionIdentity);
                    if (!job->axes)
                        throw std::runtime_error(error.isEmpty()
                            ? "No fiber-inference prediction is bound, so no sheet axis can be read."
                            : error.toStdString());
                }
                job->monitor.checkpoint();
            }
            job->cacheSink.author = job->session->newRequest(*job->attachment).author;
            job->cacheSink.computedAt = QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();
            const bool opens = job->request.measureCrossings &&
                               (job->request.measureWindingDistances || job->request.normalSource == "fused");
            job->monitor.expectStages((job->request.measureCrossings
                                           ? (job->request.normalSource == "fused" ? 13 : 12)
                                           : 4) + (opens ? 1 : 0));
            ScaffoldSolveOptions options;
            options.links.useLinks = job->request.useLinks;
            options.links.approvalStates = job->request.approvalStates;
            options.links.includeAuto = job->request.includeAuto;
            options.links.weight = job->request.manualWeight;
            options.fixApprovedSubgraphs = job->request.fixApprovedSubgraphs;
            options.flow.backend = vc::geometry::parseWindingFlowBackend(job->request.lemonBackend);
            options.workers = std::max<std::size_t>(1, job->request.workers);
            options.monitor = &job->monitor;

            ScaffoldRunRecord record;
            record.runUuid = vc::objects::util::newUuid();
            record.createdAtUnix = QDateTime::currentDateTimeUtc().toSecsSinceEpoch();
            job->session->readStore(*job->attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
                const auto graph = loadScaffoldGraph(reader, load);
                // What earlier runs computed from the same geometry with the
                // same parameters: read once, before anything is solved.
                const vc::fiber_graph::ScaffoldCache cache(reader);
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
                    normalOptions.cache = &cache;
                    normalOptions.sink = &job->cacheSink;
                    normalOptions.predictionIdentity = job->predictionIdentity;
                    eventOptions.cache = &cache;
                    eventOptions.sink = &job->cacheSink;
                    normalOptions.linkApprovalStates = job->request.approvalStates;
                    normalOptions.includeAutoLinks = job->request.includeAuto;
                    normalOptions.workers = options.workers;
                    normalOptions.monitor = &job->monitor;
                    eventOptions.measureWindingDistances = job->request.measureWindingDistances;
                    eventOptions.workers = options.workers;
                    eventOptions.monitor = &job->monitor;
                    // The dock only ever rounds: an approval is a statement
                    // about an integer winding relation.
                    measuredOptions.separations = ScaffoldSeparationMode::Rounded;
                    measuredOptions.perpendicularWeight = job->request.perpendicularWeight;
                    measuredOptions.parallelWeight = job->request.parallelWeight;
                    measuredOptions.enclosureWeight = job->request.enclosureWeight;
                    signOptions.sign.backend = job->request.signBackend == "ceres"
                                                   ? vc::fiber_tracer::FiberSignBackend::Ceres
                                                   : vc::fiber_tracer::FiberSignBackend::Irls;
                    signOptions.sign.threads = int(options.workers);
                    signOptions.monitor = &job->monitor;
                    ScaffoldWindingDistances distances;
                    if (auto sampler = job->sampler)
                        distances = [sampler](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
                                              double step, int workers) {
                            return sampler->normalAlignedWindingDistancesBatch(connectors, step, workers);
                        };
                    normals.emplace(graph, geometry, normalOptions, job->axes);
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
                    stages.approvals = &approvals;
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
                // A crossing whose geometry and whose H/V tags disagree is
                // dropped, and that is a question about the tagging: say so
                // rather than leaving it in the gap between read and used.
                if (job->request.measureCrossings && measuredReport.skippedFamilyMismatch > 0)
                    job->summary += QObject::tr("; %1 crossings dropped where the pair's geometry and their H/V \n"
                                                "tags disagree")
                                        .arg(measuredReport.skippedFamilyMismatch);
            });
        } catch (const ScaffoldCancelled&) {
            // Asked for by the caller: no error, no rows.
            job->cancelled = true;
        } catch (const std::exception& problem) {
            job->error = QString::fromUtf8(problem.what());
        }
    }));
}

void ScaffoldSolveController::cancelSolve()
{
    if (!_busy || !_job) return;
    _job->monitor.cancel();
    emit statusMessage(tr("Taking the solve back..."));
}

vc::fiber_graph::ScaffoldRunMonitor::Progress ScaffoldSolveController::progress() const
{
    if (!_busy || !_job) return {};
    return _job->monitor.progress();
}

void ScaffoldSolveController::finishSolve()
{
    auto job = _job;
    _job.reset();
    _busy = false;
    emit busyChanged(false);
    if (!job) return;
    if (job->cancelled) {
        emit statusMessage(tr("The solve was taken back; nothing was saved."));
        return;
    }
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
        // The derived results of this run, so the next one reads them instead
        // of sampling the volume again.
        if (!job->cacheSink.normals.empty()) {
            std::vector<vc::objects::ExtensionRow> values;
            values.reserve(job->cacheSink.normals.size());
            for (const auto& row : job->cacheSink.normals)
                values.push_back(vc::objects::fiberScaffoldNormalsRow(row));
            request.operations.push_back(
                vc::objects::CreateExtensionTableOp{vc::objects::fiberScaffoldNormalsTable()});
            request.operations.push_back(
                vc::objects::AppendExtensionRowsOp{vc::objects::kFiberScaffoldNormalsTable, std::move(values)});
        }
        if (!job->cacheSink.crossings.empty()) {
            std::vector<vc::objects::ExtensionRow> values;
            values.reserve(job->cacheSink.crossings.size());
            for (const auto& row : job->cacheSink.crossings)
                values.push_back(vc::objects::fiberScaffoldCrossingsRow(row));
            request.operations.push_back(
                vc::objects::CreateExtensionTableOp{vc::objects::fiberScaffoldCrossingsTable()});
            request.operations.push_back(
                vc::objects::AppendExtensionRowsOp{vc::objects::kFiberScaffoldCrossingsTable, std::move(values)});
        }
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
                // A constraint is a statement about two pieces of geometry.
                // Approving it before those pieces are approved would pin a
                // solve to geometry nobody has reviewed.
                const auto states = fiberPieceApprovalStates(contents[side], readFiberPieceApprovals(reader, uuids[side]));
                const auto& approvedSpans = states.approvedSpans[std::size_t(FiberPieceAspect::Geometry)];
                if (std::find(approvedSpans.begin(), approvedSpans.end(), spans[side]) == approvedSpans.end()) {
                    error = tr("Approve the geometry of both pieces this crossing lies on first.");
                    return;
                }
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
                          "A rejected link is never used. With neither checked the run uses no\n"
                          "link at all and solves on the measured crossings alone, which it then\n"
                          "has to be measuring."));
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
    _backendCombo->addItem(tr("cost scaling"), QStringLiteral("cost_scaling"));
    _backendCombo->addItem(tr("network simplex"), QStringLiteral("network_simplex"));
    _backendCombo->setToolTip(tr("The min-cost-flow backend. Both solve the same problem exactly."));
    form->addRow(tr("Backend"), _backendCombo);

    _scopeCombo = new QComboBox(this);
    _scopeCombo->setObjectName(QStringLiteral("fiberMapSolveScope"));
    _scopeCombo->addItem(tr("the whole scaffold"), false);
    _scopeCombo->addItem(tr("the selected network"), true);
    _scopeCombo->setToolTip(tr("Which fibers the run reads. The selected network is the group the map\n"
                               "currently has selected: its fibers alone are loaded, so the links,\n"
                               "crossings and components of the run are those of that part of the\n"
                               "annotation. A run with nothing selected is refused rather than run on\n"
                               "everything."));
    form->addRow(tr("Scope"), _scopeCombo);

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

    _normalSourceCombo = new QComboBox(this);
    _normalSourceCombo->setObjectName(QStringLiteral("fiberMapSolveNormalSource"));
    _normalSourceCombo->addItem(tr("fiber geometry"), QStringLiteral("geometry"));
    _normalSourceCombo->addItem(tr("fused prediction"), QStringLiteral("fused"));
    _normalSourceCombo->setToolTip(tr("Where the fibers' normals come from. The geometry alone gives every\n"
                                      "fiber a strip normal; the fused source also pulls each node towards\n"
                                      "the sheet axis the package's fiber-inference prediction states there,\n"
                                      "weighted by its presence. Fused reads the prediction volume along\n"
                                      "every fiber, which takes a minute on a large scaffold."));
    form->addRow(tr("Normals"), _normalSourceCombo);

    _signBackendCombo = new QComboBox(this);
    _signBackendCombo->setObjectName(QStringLiteral("fiberMapSolveSignBackend"));
    _signBackendCombo->addItem(tr("IRLS"), QStringLiteral("irls"));
    _signBackendCombo->addItem(tr("Ceres"), QStringLiteral("ceres"));
    _signBackendCombo->setToolTip(tr("Which backend decides the normal signs of the fibers no anchor fixed.\n"
                                     "Both descend the same objective."));
    form->addRow(tr("Sign backend"), _signBackendCombo);
    layout->addLayout(form);

    auto* buttons = new QWidget(this);
    auto* buttonLayout = new QHBoxLayout(buttons);
    buttonLayout->setContentsMargins(0, 0, 0, 0);
    _runButton = new QPushButton(tr("Solve"), buttons);
    _runButton->setObjectName(QStringLiteral("fiberMapSolveRun"));
    _runButton->setToolTip(tr("Read the store's fibers, links, approvals and umbilicus, assign every\n"
                              "fiber an integer winding, and save the run in the store."));
    _cancelButton = new QPushButton(tr("Cancel"), buttons);
    _cancelButton->setObjectName(QStringLiteral("fiberMapSolveCancel"));
    _cancelButton->setEnabled(false);
    _cancelButton->setToolTip(tr("Take the running solve back. It stops at its next checkpoint - between\n"
                                 "fibers, between chunks of the prediction, between components or\n"
                                 "between cycle searches - and saves nothing."));
    buttonLayout->addWidget(_runButton, 1);
    buttonLayout->addWidget(_cancelButton);
    layout->addWidget(buttons);

    _progressBar = new QProgressBar(this);
    _progressBar->setObjectName(QStringLiteral("fiberMapSolveProgressBar"));
    _progressBar->setTextVisible(false);
    _progressBar->setVisible(false);
    _progressBar->setToolTip(tr("How far the running stage is through the work it announced. A stage\n"
                                "that cannot count its work shows as busy instead; the line below\n"
                                "names which stage of how many the run is in."));
    layout->addWidget(_progressBar);

    _progressLabel = new QLabel(this);
    _progressLabel->setObjectName(QStringLiteral("fiberMapSolveProgress"));
    _progressLabel->setToolTip(tr("The stage the running solve is in, the work it has done of what that\n"
                                  "stage announced, and the time left at the rate this stage has held so\n"
                                  "far. The estimate covers the running stage only: a run's later stages\n"
                                  "do not announce their work until they start."));
    layout->addWidget(_progressLabel);

    _statusLabel = new QLabel(this);
    _statusLabel->setObjectName(QStringLiteral("fiberMapSolveStatus"));
    _statusLabel->setWordWrap(true);
    _statusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_statusLabel);
    layout->addStretch(1);

    connect(_runButton, &QPushButton::clicked, this, [this]() { emit solveRequested(); });
    connect(_cancelButton, &QPushButton::clicked, this, [this]() {
        if (_controller) _controller->cancelSolve();
    });
    // The worker writes its progress into the run's monitor; the dock reads it
    // rather than the worker signalling every fiber it finished.
    _progressTimer = new QTimer(this);
    _progressTimer->setInterval(250);
    connect(_progressTimer, &QTimer::timeout, this, [this]() { showProgress(); });
    for (auto* box : {_approvedCheck, _pendingCheck, _crossingsCheck}) {
        connect(box, &QCheckBox::toggled, this, [this]() { updateEnabled(); });
    }
    if (_controller) {
        connect(_controller, &ScaffoldSolveController::busyChanged, this, [this](bool busy) {
            updateEnabled();
            if (busy) {
                _progressStage.clear();
                _progressTimer->start();
                showProgress();
            } else {
                _progressTimer->stop();
                _progressLabel->clear();
                _progressStage.clear();
                _progressBar->setVisible(false);
            }
        });
        connect(_controller, &ScaffoldSolveController::statusMessage, this,
                [this](const QString& message) { _statusLabel->setText(message); });
    }
    updateEnabled();
}

void ScaffoldSolverPanel::setScopeAvailable(bool available)
{
    if (!_scopeCombo) return;
    const auto at = _scopeCombo->findData(true);
    if (at < 0) return;
    if (!available && _scopeCombo->currentIndex() == at) _scopeCombo->setCurrentIndex(0);
    if (auto* model = qobject_cast<QStandardItemModel*>(_scopeCombo->model()))
        if (auto* item = model->item(at)) item->setEnabled(available);
    _scopeCombo->setItemData(at,
                             available ? QString()
                                       : tr("The map has no layout yet, so it has no selected network; press "
                                            "Update to restrict a run to one."),
                             Qt::ToolTipRole);
}

void ScaffoldSolverPanel::setScopeResolver(ScopeResolver resolver)
{
    _scope = std::move(resolver);
}

void ScaffoldSolverPanel::showProgress()
{
    if (!_controller) return;
    const auto progress = _controller->progress();
    if (progress.stage.empty()) {
        _progressLabel->clear();
        _progressBar->setVisible(false);
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    // A new stage restarts the measurement: it counts its own work, and the
    // work already done when it was first seen is not this measurement's.
    if (progress.stage != _progressStage) {
        _progressStage = progress.stage;
        _progressStarted = now;
        _progressStageFirstDone = progress.done;
    }
    // Coarse progress first: which stage of how many, then the stage itself.
    const auto coarse = progress.stageCount > 1
        ? tr("%1/%2 ").arg(progress.stageIndex).arg(progress.stageCount)
        : QString();
    const auto stage = coarse + QString::fromStdString(progress.stage);
    if (progress.total <= 0) {
        // Nothing to divide: a busy bar and the stage's name, which is all a
        // stage that announced no total can honestly show.
        _progressBar->setRange(0, 0);
        _progressBar->setVisible(true);
        _progressLabel->setText(tr("%1...").arg(stage));
        return;
    }
    _progressBar->setRange(0, int(std::min<std::int64_t>(progress.total, std::numeric_limits<int>::max())));
    _progressBar->setValue(int(std::min<std::int64_t>(progress.done, progress.total)));
    _progressBar->setVisible(true);
    const auto done = progress.done - _progressStageFirstDone;
    const auto seconds = std::chrono::duration<double>(now - _progressStarted).count();
    QString eta;
    // Below a second of measurement or a single step the rate says nothing;
    // the line then carries the counts alone rather than a wild number.
    if (done > 0 && seconds >= 1.) {
        const auto left = double(progress.total - progress.done) * seconds / double(done);
        if (left >= 1.) eta = tr(", %1 left").arg(scaffoldDurationText(left));
    }
    _progressLabel->setText(
        tr("%1: %2 of %3%4").arg(stage).arg(progress.done).arg(progress.total).arg(eta));
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
    const bool crossings = _crossingsCheck->isChecked();
    const bool busy = _controller && _controller->busy();
    // No link state selected is a measured-only run, which needs the
    // crossings: without either the solve would have nothing to constrain.
    _runButton->setEnabled((states || crossings) && !busy && !_volume.empty());
    _cancelButton->setEnabled(busy);
    if (_volume.empty())
        _runButton->setToolTip(tr("The map has no annotation volume to solve in."));
    else if (!states && !crossings)
        _runButton->setToolTip(tr("Nothing would constrain the windings: select a link state, or measure the\n"
                                  "fibers' crossings."));
    for (auto* widget : std::initializer_list<QWidget*>{_windingDistanceCheck, _perpendicularWeightSpin,
                                                        _parallelWeightSpin, _enclosureWeightSpin,
                                                        _normalSourceCombo, _signBackendCombo})
        widget->setEnabled(crossings);
}

ScaffoldSolveRequest ScaffoldSolverPanel::request() const
{
    ScaffoldSolveRequest request;
    request.approvalStates.clear();
    if (_approvedCheck->isChecked()) request.approvalStates.push_back("approved");
    if (_pendingCheck->isChecked()) request.approvalStates.push_back("pending");
    request.useLinks = !request.approvalStates.empty();
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
    request.normalSource = _normalSourceCombo->currentData().toString().toStdString();
    request.signBackend = _signBackendCombo->currentData().toString().toStdString();
    request.workers = std::size_t(std::max(1, QThread::idealThreadCount()));
    request.restricted = _scopeCombo->currentData().toBool();
    if (request.restricted && _scope) request.fiberSubset = _scope();
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
    _tree->setHeaderLabels({tr("when"), tr("method"), tr("normals"), tr("sign"), tr("fibers"), tr("comp"), tr("arcs"),
                            tr("L1"), tr("bad"), tr("max"), tr("sign!"), tr("state"), tr("diff")});
    _tree->setUniformRowHeights(true);
    _tree->setRootIsDecorated(false);
    // Two runs can be selected, which compares them; the first of them is
    // the one the other docks follow.
    _tree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    _tree->header()->setStretchLastSection(true);
    _tree->setToolTip(tr("Every solve saved in this store, newest first. Selecting one lays the\n"
                         "map out from it: each fiber at the winding that solve gave it.\n"
                         "L1 is the weighted residual in turns, bad the violated constraints,\n"
                         "max the largest single residual in turns, state whether the store has\n"
                         "changed since the run read it.\n"
                         "Selecting a second run compares the two: diff counts the fibers one\n"
                         "run places differently from the other."));
    layout->addWidget(_tree, 1);
    _diffLabel = new QLabel(this);
    _diffLabel->setObjectName(QStringLiteral("fiberMapSolvesDiff"));
    _diffLabel->setWordWrap(true);
    _diffLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _diffLabel->setToolTip(tr("What the two selected solves disagree about. A winding is gauged per\n"
                              "component and per run, so the comparison is relative: the fibers\n"
                              "counted are those that moved against the neighbours they share a\n"
                              "component with in both runs."));
    layout->addWidget(_diffLabel);
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
        // The run's own time, as it recorded it; a run saved before the
        // settings carried one shows nothing rather than a guess.
        const auto when = run.createdAtUnix > 0
            ? QDateTime::fromSecsSinceEpoch(run.createdAtUnix, Qt::UTC).toLocalTime()
            : QDateTime();
        item->setText(kRunWhenColumn, when.isValid() ? when.toString(QStringLiteral("MM-dd HH:mm")) : QString());
        item->setToolTip(kRunWhenColumn, when.isValid()
                                             ? when.toUTC().toString(Qt::ISODate)
                                             : tr("This run recorded no time."));
        item->setText(kRunNormalsColumn, QString::fromStdString(run.normalSource));
        item->setToolTip(kRunNormalsColumn, tr("The normals the run used: the fibers' own geometry, or those fused\n"
                                               "with the prediction's sheet axis. Empty for a links-only run."));
        item->setText(kRunSignBackendColumn, QString::fromStdString(run.signBackend));
        item->setToolTip(kRunSignBackendColumn, tr("Which backend decided the normal signs. Empty for a run that\n"
                                                   "solved none."));
        item->setText(kRunMethodColumn, QString::fromStdString(run.method));
        item->setText(kRunFibersColumn, QString::number(count("fibers")));
        item->setText(kRunComponentsColumn, QString::number(count("components")));
        item->setText(kRunArcsColumn, QString::number(count("arcs")));
        item->setText(kRunObjectiveColumn, QString::number(metric("weighted_l1"), 'f', 3));
        item->setText(kRunViolatedColumn, QString::number(count("violated_constraints")));
        item->setText(kRunMaxResidualColumn, QString::number(metric("max_residual"), 'f', 3));
        // The sign solve's own disagreement with the anchors it held.
        const auto signs = run.counts.find("sign_fixed_disagreements");
        item->setText(kRunSignColumn, signs == run.counts.end() ? QString() : QString::number(signs->second));
        item->setToolTip(kRunSignColumn,
                         tr("Anchored fibers whose normal sign the crossings would have flipped: the\n"
                            "anchors held, but the measured evidence disagreed with them. Empty for a\n"
                            "run that solved no signs."));
        const bool stale = _controller->runStale(run);
        item->setText(kRunStaleColumn, stale ? tr("stale") : tr("current"));
        item->setToolTip(kRunStaleColumn,
                         stale ? tr("A fiber, a link or the umbilicus changed after this run read the\n"
                                    "store (it read commit %1): its windings may answer a question about\n"
                                    "an annotation that has moved since.")
                                     .arg(run.input.commitSequence)
                               : tr("Nothing in the annotation has changed since this run read it."));
        item->setData(kRunMethodColumn, Qt::UserRole, QString::fromStdString(run.runUuid));
        item->setToolTip(kRunMethodColumn, oneLine(run) + QStringLiteral("\n") +
                                               QString::fromStdString(run.settingsJson));
        if (run.runUuid == previous) item->setSelected(true);
    }
    for (int column = 0; column < kRunColumnCount - 1; ++column) _tree->resizeColumnToContents(column);
    _rebuilding = false;
    showDiff();
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
    showDiff();
}

void ScaffoldRunsPanel::showDiff()
{
    if (!_controller) return;
    for (int row = 0; row < _tree->topLevelItemCount(); ++row)
        _tree->topLevelItem(row)->setText(kRunDiffColumn, QString());
    _diffLabel->clear();
    const auto items = _tree->selectedItems();
    if (items.size() != 2) return;
    const auto uuid = [](const QTreeWidgetItem* item) {
        return item->data(kRunMethodColumn, Qt::UserRole).toString().toStdString();
    };
    QString error;
    const auto diff = _controller->diffRuns(uuid(items[0]), uuid(items[1]), error);
    if (!error.isEmpty()) {
        _diffLabel->setText(error);
        return;
    }
    const auto moved = QString::number(diff.winding);
    for (auto* item : items) item->setText(kRunDiffColumn, moved);
    _diffLabel->setText(tr("%1 of %2 shared fibers at a different winding, %3 of %4 with a different normal "
                           "sign, over %5 component pairs; %6 and %7 fibers only one run placed.")
                            .arg(diff.winding)
                            .arg(diff.shared)
                            .arg(diff.sign)
                            .arg(diff.signStated)
                            .arg(diff.groups)
                            .arg(diff.onlyFirst)
                            .arg(diff.onlySecond));
}

ScaffoldApprovalPanel::ScaffoldApprovalPanel(ScaffoldSolveController* controller, QWidget* parent)
    : QWidget(parent)
    , _controller(controller)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(3);

    auto* fadeRow = new QWidget(this);
    auto* fadeLayout = new QHBoxLayout(fadeRow);
    fadeLayout->setContentsMargins(0, 0, 0, 0);
    fadeLayout->addWidget(new QLabel(tr("Fade unapproved"), fadeRow));
    _fadeCombo = new QComboBox(fadeRow);
    _fadeCombo->setObjectName(QStringLiteral("fiberMapApprovalFade"));
    _fadeCombo->addItem(tr("nothing"), int(FadeMode::None));
    _fadeCombo->addItem(tr("geometry"), int(FadeMode::Geometry));
    _fadeCombo->addItem(tr("normals"), int(FadeMode::Normals));
    _fadeCombo->setToolTip(tr("Draw the approved spans as the map always draws them and fade the rest:\n"
                              "geometry fades every span without a current geometry approval; normals\n"
                              "fades every span whose normals are not approved, which includes those\n"
                              "whose geometry is not (a normals approval requires one)."));
    fadeLayout->addWidget(_fadeCombo, 1);
    layout->addWidget(fadeRow);

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

    auto* componentRow = new QWidget(this);
    auto* componentLayout = new QHBoxLayout(componentRow);
    componentLayout->setContentsMargins(0, 0, 0, 0);
    componentLayout->addWidget(new QLabel(tr("Component"), componentRow));
    _componentCombo = new QComboBox(componentRow);
    _componentCombo->setObjectName(QStringLiteral("fiberMapApprovalComponent"));
    _componentCombo->setToolTip(tr("Show only the rows of one component of the selected solve. A component\n"
                                   "is solved on its own, so its violations can only be explained by its\n"
                                   "own annotation."));
    componentLayout->addWidget(_componentCombo, 1);
    layout->addWidget(componentRow);

    _tree = new QTreeWidget(this);
    _tree->setObjectName(QStringLiteral("fiberMapApprovalTree"));
    _tree->setColumnCount(kColumnCount);
    _tree->setHeaderLabels({tr("what"), tr("fiber"), tr("where"), tr("off"), tr("state"), tr("cost"), tr("joint"),
                            tr("loop"), tr("cycle"), tr("length"), tr("paths")});
    _tree->headerItem()->setToolTip(kOffColumn, tr("Turns of winding the solve could not satisfy on this constraint."));
    _tree->headerItem()->setToolTip(kStateColumn,
                                    tr("A constraint: approved, approvable (both pieces approved), blocked (they are\n"
                                       "not) or link (its approval is its review state). A piece: whether its\n"
                                       "geometry is approved, +n when its normals are too."));
    _tree->headerItem()->setToolTip(kCostColumn, tr("Weight times residual, in weighted turns."));
    _tree->headerItem()->setToolTip(kJointColumn, tr("Cost per constraint of the shortest cycle (weighted turns)."));
    _tree->headerItem()->setToolTip(kCycleNumberColumn,
                                    tr("Which cycle this row is about, numbered from 1 in rank order. Rows\n"
                                       "sharing a number are about the same loop; a stretch lists every loop\n"
                                       "that runs along it."));
    _tree->headerItem()->setToolTip(kCycleColumn, tr("Constraints on the shortest cycle the violation lies on."));
    _tree->headerItem()->setToolTip(kLengthColumn,
                                    tr("The stretch's length, or the cycle's total stretch length (voxels)."));
    _tree->headerItem()->setToolTip(kCountColumn,
                                    tr("A constraint: edge-disjoint alternative paths found (redundancy). A\n"
                                       "stretch: how many violated constraints accuse it."));
    _tree->setUniformRowHeights(true);
    _tree->setRootIsDecorated(true);
    _tree->setExpandsOnDoubleClick(false);
    _tree->setSortingEnabled(true);
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

    _errorLabel = new QLabel(this);
    _errorLabel->setObjectName(QStringLiteral("fiberMapApprovalError"));
    _errorLabel->setWordWrap(true);
    _errorLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(_errorLabel);

    connect(_approveButton, &QPushButton::clicked, this, [this]() {
        const auto* residual = residualOf(_tree->currentItem());
        if (!residual || !_controller) return;
        const bool approved = _controller->constraintApproved(*residual);
        const auto error = _controller->setConstraintApproval(*residual, !approved);
        _errorLabel->setText(error);
        if (!error.isEmpty()) return;
        rebuild();
    });
    for (auto* combo : {_orderCombo, _componentCombo})
        connect(combo, &QComboBox::currentIndexChanged, this, [this]() {
            if (_rebuilding) return;
            rebuild();
        });
    connect(_tree, &QTreeWidget::itemSelectionChanged, this, [this]() { applySelection(); });
    // Hovering a row shows it on the map, apart from the selection.
    _tree->setMouseTracking(true);
    _tree->viewport()->installEventFilter(this);
    connect(_tree, &QTreeWidget::itemEntered, this,
            [this](QTreeWidgetItem* item, int) { emit hoverRequested(highlightOf(item)); });
    connect(_tree, &QTreeWidget::itemDoubleClicked, this,
            [this](QTreeWidgetItem* item, int) { activateRow(item); });
    connect(_fadeCombo, &QComboBox::currentIndexChanged, this, [this]() {
        if (fadeMode() != FadeMode::None && _controller && !_controller->approvalsLoaded())
            _controller->loadApprovals();
        emit highlightModeChanged();
    });
    if (_controller) {
        connect(_controller, &ScaffoldSolveController::selectionChanged, this, [this]() { rebuild(); });
        connect(_controller, &ScaffoldSolveController::approvalsChanged, this, [this]() {
            rebuild();
            emit highlightModeChanged();
        });
        connect(_controller, &ScaffoldSolveController::constraintApprovalsChanged, this, [this]() { rebuild(); });
    }
    rebuild();
}

ScaffoldApprovalPanel::FadeMode ScaffoldApprovalPanel::fadeMode() const
{
    return FadeMode(_fadeCombo->currentData().toInt());
}

void ScaffoldApprovalPanel::setLabelLookup(LabelLookup lookup)
{
    _labels = std::move(lookup);
    rebuild();
}

std::array<ScaffoldAnchorSpan, 2> ScaffoldApprovalPanel::coveringSpans(const ScaffoldStoredResidual& residual) const
{
    if (!_controller) return {};
    return {_controller->anchorSpan(residual.fiberA, residual.arcA),
            _controller->anchorSpan(residual.fiberB, residual.arcB)};
}

void ScaffoldApprovalPanel::rebuild()
{
    _rebuilding = true;
    _tree->setSortingEnabled(false);
    _tree->clear();
    _rows.clear();
    const auto* view = _controller ? _controller->selected() : nullptr;
    if (!view) {
        _summaryLabel->setText(tr("Select a solve to see what it could not satisfy."));
        _rebuilding = false;
        return;
    }
    // The piece approvals answer the state column and gate the approve
    // button, so the dock needs them as soon as it has rows.
    if (_controller && !_controller->approvalsLoaded()) _controller->loadApprovals();
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
    // The component each row's fiber was solved in, so the dock can show one
    // component at a time. A constraint's two fibers are in the same
    // component by construction (it is an arc between them).
    const auto componentOf = [&](const std::string& uuid) -> std::int64_t {
        const auto found = view->windings.find(uuid);
        return found == view->windings.end() ? -1 : found->second.component;
    };
    // A row and the children it owns: the two pieces a constraint lies on.
    struct Entry {
        Row row;
        QStringList text;
        std::vector<std::pair<Row, QStringList>> children;
    };
    const auto spanText = [&](const ScaffoldAnchorSpan& span, double arc) {
        return span.known ? tr("cp %1-%2").arg(span.controlFrom).arg(span.controlTo)
                          : tr("%1 vx").arg(arc, 0, 'f', 0);
    };
    const auto stateText = [&](const ScaffoldAnchorSpan& span) {
        if (!span.known) return tr("unknown");
        if (span.geometryApproved) return span.normalsApproved ? tr("approved +n") : tr("approved");
        return tr("unapproved");
    };
    const auto cycleNumbers = scaffoldCycleNumbers(view->residuals);
    const auto cycleText = [](std::int64_t number) { return number > 0 ? QString::number(number) : QString(); };
    std::vector<Entry> entries;
    for (std::size_t index = 0; index < view->residuals.size(); ++index) {
        const auto& residual = view->residuals[index];
        Entry entry;
        entry.row.kind = Row::Kind::Constraint;
        entry.row.residual = std::int64_t(index);
        entry.row.keys.cost = residual.weight * residual.residual;
        entry.row.keys.cycle = residual.cycleHops.value_or(-1);
        entry.row.keys.length = residual.cyclePieceLength;
        entry.row.fiberUuid = residual.fiberA;
        entry.row.otherUuid = residual.fiberB;
        const auto measured = residual.edgeUuid.empty() &&
                              residual.kind != std::int64_t(ScaffoldConstraintKind::Manual);
        const auto spans = coveringSpans(residual);
        // Where: the two pieces the crossing lies on, which is where it can
        // be looked at; a link states its anchors the same way.
        const auto where = tr("%1 / %2").arg(spanText(spans[0], residual.arcA), spanText(spans[1], residual.arcB));
        const auto approvedConstraint = measured && _controller && _controller->constraintApproved(residual);
        const auto state = !measured
            ? tr("link")
            : (approvedConstraint ? tr("approved")
                                  : (spans[0].geometryApproved && spans[1].geometryApproved ? tr("approvable")
                                                                                            : tr("blocked")));
        entry.text << (residual.edgeUuid.empty() ? kindName(residual.kind) : tr("link"))
                   << tr("%1 - %2").arg(label(residual.fiberA), label(residual.fiberB)) << where
                   << tr("%1 turns").arg(residual.residual, 0, 'f', 2) << state
                   << QString::number(entry.row.keys.cost, 'f', 3)
                   << (entry.row.keys.cycle > 0
                           ? QString::number(entry.row.keys.cost / double(entry.row.keys.cycle), 'f', 3)
                           : QString())
                   << cycleText(cycleNumbers[index])
                   << (entry.row.keys.cycle > 0 ? QString::number(entry.row.keys.cycle) : tr("none"))
                   << QString::number(entry.row.keys.length, 'f', 0)
                   << (residual.redundancy ? QString::number(*residual.redundancy) : QString());
        // The two covering pieces as children: each is openable, and its
        // state says what still blocks the constraint.
        const std::array<std::string, 2> uuids{residual.fiberA, residual.fiberB};
        for (std::size_t side = 0; side < 2; ++side) {
            if (!spans[side].known) continue;
            Row child;
            child.kind = Row::Kind::Piece;
            child.fiberUuid = uuids[side];
            child.controlFrom = spans[side].controlFrom;
            child.controlTo = spans[side].controlTo;
            QStringList text;
            text << tr("piece") << label(uuids[side]) << spanText(spans[side], 0.) << QString()
                 << stateText(spans[side]) << QString() << QString() << QString() << QString() << QString()
                 << QString();
            entry.children.emplace_back(child, text);
        }
        entries.push_back(std::move(entry));
    }
    for (const auto& piece : view->pieces) {
        Entry entry;
        entry.row.kind = Row::Kind::Piece;
        entry.row.keys.cost = piece.score;
        entry.row.keys.cycle = piece.cycleHops;
        entry.row.keys.length = piece.arcTo - piece.arcFrom;
        entry.row.fiberUuid = piece.fiberUuid;
        entry.row.arcFrom = std::min(piece.arcFrom, piece.arcTo);
        entry.row.arcTo = std::max(piece.arcFrom, piece.arcTo);
        entry.row.controlFrom = piece.controlFrom.value_or(-1);
        entry.row.controlTo = piece.controlTo.value_or(-1);
        // A run that recorded no range, or a degenerate one, is not fatal: the
        // fiber's own control arcs place the stretch's start.
        if (entry.row.controlTo <= entry.row.controlFrom) {
            const auto resolved = _controller ? _controller->anchorSpan(piece.fiberUuid, piece.arcFrom)
                                              : ScaffoldAnchorSpan{};
            if (resolved.known) {
                entry.row.controlFrom = resolved.controlFrom;
                entry.row.controlTo = resolved.controlTo;
            }
        }
        const auto span = _controller ? _controller->anchorSpan(piece.fiberUuid, piece.arcFrom) : ScaffoldAnchorSpan{};
        const QString where = entry.row.controlFrom >= 0
            ? tr("cp %1-%2").arg(entry.row.controlFrom).arg(entry.row.controlTo)
            : tr("%1 vx").arg(entry.row.keys.length, 0, 'f', 0);
        QStringList loops;
        for (const auto number : piece.cycles) loops << QString::number(number);
        entry.text << tr("stretch") << label(piece.fiberUuid) << where << QString() << stateText(span)
                   << QString::number(entry.row.keys.cost, 'f', 3)
                   << (entry.row.keys.cycle > 0
                           ? QString::number(entry.row.keys.cost / double(entry.row.keys.cycle), 'f', 3)
                           : QString())
                   << loops.join(QStringLiteral(", "))
                   << (entry.row.keys.cycle > 0 ? QString::number(entry.row.keys.cycle) : tr("none"))
                   << QString::number(entry.row.keys.length, 'f', 0) << QString::number(piece.constraints);
        entries.push_back(std::move(entry));
    }
    // Offer every component the rows actually name, keeping the chosen one
    // when the run still has it.
    std::set<std::int64_t> components;
    for (const auto& entry : entries) {
        const auto component = componentOf(entry.row.fiberUuid);
        if (component >= 0) components.insert(component);
    }
    const auto chosen = _componentCombo->count() > 0 && _componentCombo->currentIndex() > 0
                            ? _componentCombo->currentData().toLongLong()
                            : -1;
    _componentCombo->clear();
    _componentCombo->addItem(tr("all (%1)").arg(components.size()), qlonglong(-1));
    for (const auto component : components)
        _componentCombo->addItem(QString::number(component), qlonglong(component));
    if (chosen >= 0) {
        const auto at = _componentCombo->findData(qlonglong(chosen));
        if (at > 0) _componentCombo->setCurrentIndex(at);
    }
    const auto filter = _componentCombo->currentIndex() > 0 ? _componentCombo->currentData().toLongLong() : -1;
    if (filter >= 0)
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [&](const Entry& entry) { return componentOf(entry.row.fiberUuid) != filter; }),
                      entries.end());

    const auto order = ScaffoldApprovalOrder(_orderCombo->currentData().toInt());
    std::stable_sort(entries.begin(), entries.end(), [&](const Entry& x, const Entry& y) {
        return scaffoldApprovalRowBefore(x.row.keys, y.row.keys, order);
    });
    // The rows are kept beside the items: an item carries only its index, so
    // every column is free to show what it means rather than a payload.
    const auto append = [this](QTreeWidgetItem* parent, const Row& row, const QStringList& text) {
        auto* item = new RankingItem(text);
        if (parent) parent->addChild(item); else _tree->addTopLevelItem(item);
        item->setData(kKindColumn, Qt::UserRole, qlonglong(_rows.size()));
        // Numeric sort keys for the columns a header click orders by; the
        // text of those columns is formatted for reading.
        const auto number = [&](int column, double value) {
            item->setData(column, Qt::UserRole + 1, value);
        };
        number(kCostColumn, row.keys.cost);
        number(kJointColumn, row.keys.cycle > 0 ? row.keys.cost / double(row.keys.cycle) : 0.);
        number(kCycleColumn, row.keys.cycle > 0 ? double(row.keys.cycle) : 1e18);
        number(kLengthColumn, row.keys.length);
        _rows.push_back(row);
        return item;
    };
    std::int64_t constraintRows = 0, pieceRows = 0, blocked = 0, approvedRows = 0;
    for (const auto& entry : entries) {
        if (entry.row.kind == Row::Kind::Piece) ++pieceRows; else ++constraintRows;
        auto* item = append(nullptr, entry.row, entry.text);
        if (entry.text.value(kStateColumn) == tr("blocked")) ++blocked;
        if (entry.text.value(kStateColumn) == tr("approved")) ++approvedRows;
        for (const auto& [row, text] : entry.children) append(item, row, text);
    }
    _tree->setSortingEnabled(true);
    for (int column = 0; column < kColumnCount - 1; ++column) _tree->resizeColumnToContents(column);
    const auto counts = filter >= 0
        ? tr("%1 violated constraints and %2 accused stretches of component %3, of %4 and %5 in the whole solve")
              .arg(constraintRows)
              .arg(pieceRows)
              .arg(filter)
              .arg(view->residuals.size())
              .arg(view->pieces.size())
        : tr("%1 violated constraints, %2 accused stretches")
              .arg(view->residuals.size())
              .arg(view->pieces.size());
    _summaryLabel->setText(counts + tr("; %1 approved, %2 blocked by unapproved geometry")
                                        .arg(approvedRows)
                                        .arg(blocked));
    _rebuilding = false;
    applySelection();
}

const ScaffoldApprovalPanel::Row* ScaffoldApprovalPanel::rowOf(const QTreeWidgetItem* item) const
{
    if (!item) return nullptr;
    const auto index = item->data(kKindColumn, Qt::UserRole).toLongLong();
    if (index < 0 || std::size_t(index) >= _rows.size()) return nullptr;
    return &_rows[std::size_t(index)];
}

const ScaffoldStoredResidual* ScaffoldApprovalPanel::residualOf(const QTreeWidgetItem* item) const
{
    const auto* view = _controller ? _controller->selected() : nullptr;
    const auto* row = rowOf(item);
    if (!view || !row || row->residual < 0 || std::size_t(row->residual) >= view->residuals.size()) return nullptr;
    return &view->residuals[std::size_t(row->residual)];
}

void ScaffoldApprovalPanel::applySelection()
{
    if (_rebuilding) return;
    const auto items = _tree->selectedItems();
    // The approve button follows the selection: a measured crossing can be
    // approved or revoked, and only while both pieces it lies on are
    // approved geometry. A link and a stretch cannot be approved here.
    const auto* residual = items.isEmpty() ? nullptr : residualOf(items.front());
    const bool measured = residual && residual->edgeUuid.empty() &&
                          residual->kind != std::int64_t(ScaffoldConstraintKind::Manual);
    const bool approved = measured && _controller && _controller->constraintApproved(*residual);
    bool pieces = false;
    if (measured) {
        const auto spans = coveringSpans(*residual);
        pieces = spans[0].geometryApproved && spans[1].geometryApproved;
    }
    _approveButton->setEnabled(measured && (approved || pieces));
    _approveButton->setText(approved ? tr("Revoke approval") : tr("Approve constraint"));
    if (measured && !approved && !pieces)
        _approveButton->setToolTip(tr("Both pieces this crossing lies on have to be approved for geometry\n"
                                      "first: they are the rows under it, and a double-click opens one."));
    else
        _approveButton->setToolTip(tr("Record that this measured crossing is right. An approved constraint is\n"
                                      "held rigid by the next solve, and the record is invalidated if either\n"
                                      "fiber's covering span is edited or its approval revoked. A link is not\n"
                                      "approved here: its approval is its review state in the annotation."));
    if (items.isEmpty()) {
        emit highlightRequested({});
        return;
    }
    if (!rowOf(items.front())) return;
    emit highlightRequested(highlightOf(items.front()));
}

ScaffoldHighlight ScaffoldApprovalPanel::highlightOf(const QTreeWidgetItem* item) const
{
    ScaffoldHighlight highlight;
    const auto* row = rowOf(item);
    if (!row) return highlight;
    const auto* residual = residualOf(item);
    if (!row->fiberUuid.empty()) highlight.fiberUuids.push_back(row->fiberUuid);
    if (!row->otherUuid.empty()) highlight.fiberUuids.push_back(row->otherUuid);
    if (row->kind == Row::Kind::Piece) {
        if (row->controlFrom >= 0 && row->controlTo > row->controlFrom)
            highlight.spans.push_back({row->fiberUuid, row->controlFrom, row->controlTo});
    } else if (residual) {
        // A constraint is drawn as the two pieces its anchors lie on, which
        // is what makes it visible on the map at all.
        const auto spans = coveringSpans(*residual);
        const std::array<std::string, 2> uuids{residual->fiberA, residual->fiberB};
        for (std::size_t side = 0; side < 2; ++side)
            if (spans[side].known)
                highlight.spans.push_back({uuids[side], spans[side].controlFrom, spans[side].controlTo});
    }
    return highlight;
}

bool ScaffoldApprovalPanel::eventFilter(QObject* watched, QEvent* event)
{
    // The mouse left the list: nothing is hovered any more.
    if (_tree && watched == _tree->viewport() && event->type() == QEvent::Leave) emit hoverRequested({});
    return QWidget::eventFilter(watched, event);
}

void ScaffoldApprovalPanel::activateRow(QTreeWidgetItem* item)
{
    const auto* row = rowOf(item);
    if (!row || row->kind != Row::Kind::Piece) return;
    if (row->fiberUuid.empty() || row->controlFrom < 0 || row->controlTo <= row->controlFrom) return;
    emit pieceActivated(row->fiberUuid, row->controlFrom, row->controlTo, row->arcFrom, row->arcTo);
}

} // namespace vc3d::fiber_map
