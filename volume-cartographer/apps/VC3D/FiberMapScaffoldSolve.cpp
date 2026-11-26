#include "FiberMapScaffoldSolve.hpp"

#include "vc/core/util/Logging.hpp"

#include "vc/fiber_graph/ScaffoldCache.hpp"

#include "CState.hpp"

#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
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
#include <cmath>
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

// Arc length of each line point (voxels), from the annotated line's first
// vertex (the first control) as the solve measures it: the extrapolated tail
// before it is not fiber. Empty for a line of fewer than two points.
std::vector<double> scaffoldLineArcs(const vc::objects::FiberPieceContent& content)
{
    if (content.linePoints.size() < 2) return {};
    std::vector<double> cumulative(content.linePoints.size(), 0.);
    for (std::size_t i = 1; i < content.linePoints.size(); ++i) {
        const auto& previous = content.linePoints[i - 1];
        const auto& point = content.linePoints[i];
        const auto dx = point[0] - previous[0], dy = point[1] - previous[1], dz = point[2] - previous[2];
        cumulative[i] = cumulative[i - 1] + std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    const auto origin = cumulative[vc::geometry::annotatedLineRange(content.controlPoints, content.linePoints).begin];
    for (auto& arc : cumulative) arc -= origin;
    return cumulative;
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
    const auto cumulative = scaffoldLineArcs(content);
    if (indices.size() != content.controlIds.size() || cumulative.empty()) return {};
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
    // A neighbours run: the signed strip samples next to the fiber store,
    // and the controller's opened copy of them (kept between runs).
    std::filesystem::path neighbourPath;
    std::shared_ptr<ScaffoldSolveController::NeighbourNormals> neighbours;
    vc::fiber_graph::ScaffoldNeighbourSource neighbourSource;
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
    , _neighbours(std::make_shared<NeighbourNormals>())
{
}

ScaffoldSolveController::~ScaffoldSolveController() = default;

std::filesystem::path ScaffoldSolveController::neighbourNormalsPath(const std::filesystem::path& fiberStore)
{
    return fiberStore.string() + ".neighbour_normals.vcstrip";
}

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

QString ScaffoldSolveController::anchorSpanProblem(const std::string& fiberUuid, double arc) const
{
    const auto name = QString::fromStdString(fiberUuid).left(8);
    if (!_approvalsLoaded)
        return tr("fiber %1: the fibers' control points were not read from the store").arg(name);
    const auto found = _approvals.find(fiberUuid);
    if (found == _approvals.end())
        return tr("fiber %1 is not among the %2 fibers read from the store (frame '%3')")
            .arg(name).arg(_approvals.size()).arg(QString::fromStdString(_frameVolume));
    const auto& entry = found->second;
    if (entry.controlArcs.size() < 2)
        return tr("fiber %1: its %2 control points could not be placed on its line in frame '%3' (%4 control arcs)")
            .arg(name).arg(entry.controlIds.size()).arg(QString::fromStdString(_frameVolume)).arg(entry.controlArcs.size());
    if (!std::isfinite(arc)) return tr("fiber %1: the run states no arc there").arg(name);
    return {};
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
    answer.geometryApproved = approvedAt(std::size_t(vc::objects::FiberApprovalAspect::Geometry));
    answer.normalsApproved = approvedAt(std::size_t(vc::objects::FiberApprovalAspect::Normals));
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
    // Newest last in commit order; the list shows them newest first, by
    // the time each run recorded.
    std::reverse(_runs.begin(), _runs.end());
    std::stable_sort(_runs.begin(), _runs.end(),
                     [](const auto& a, const auto& b) { return a.createdAtUnix > b.createdAtUnix; });
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
            view.flips = readScaffoldNormalFlips(reader, runUuid);
            view.crossings = readScaffoldCrossings(reader, runUuid);
            view.continuations = readScaffoldContinuations(reader, runUuid);
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

void ScaffoldSolveController::setFrame(const std::string& volume)
{
    if (volume == _frameVolume) return;
    _frameVolume = volume;
    dropApprovals();
}

std::optional<ScaffoldSolveController::RunFiberNormals> ScaffoldSolveController::runFiberNormals(
    const std::string& fiberUuid, QString* why) const
{
    const auto fail = [&](const QString& reason) -> std::optional<RunFiberNormals> {
        if (why) *why = reason;
        return std::nullopt;
    };
    const auto* view = selected();
    if (!view) return fail(tr("no solve is selected"));
    const auto winding = view->windings.find(fiberUuid);
    if (winding == view->windings.end() || !winding->second.normalSign)
        return fail(tr("the selected solve solved no normal for this fiber"));
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) return fail(tr("no object store is attached"));
    const double sign = double(*winding->second.normalSign);
    std::optional<RunFiberNormals> out;
    QString reason;
    try {
        session->readStore(*store, [&](const vc::objects::StoreReader& reader, const auto&) {
            const auto revision = reader.currentObject(fiberUuid);
            if (!revision) { reason = tr("the fiber is no longer in the store"); return; }
            auto content = vc::objects::fiberPieceContent(reader, *revision);
            if (!_frameVolume.empty() && !content.linePoints.empty()) {
                const vc::objects::GeometrySnapshot snapshot(reader);
                const auto transformed = snapshot.viewTransformedAt(*revision, _frameVolume, 1.);
                content.linePoints = transformed.points("line_points", 0, content.linePoints.size());
                content.controlPoints = transformed.points("control_points", 0, content.controlPoints.size());
            }
            // The solve reads the annotated line only; its normals are per
            // point of that.
            const auto range = vc::geometry::annotatedLineRange(content.controlPoints, content.linePoints);
            // The normals row the run itself references: no search, and an
            // error when the reference does not resolve.
            const auto gauge = vc::fiber_graph::readScaffoldRunFiberNormals(reader, view->run.runUuid, fiberUuid, range.size());
            out.emplace();
            out->points.reserve(range.size());
            out->normals.reserve(range.size());
            for (std::size_t i = 0; i < range.size(); ++i) {
                const auto& point = content.linePoints[range.begin + i];
                out->points.emplace_back(float(point[0]), float(point[1]), float(point[2]));
                // The run's signed normal: outward, as the views take it.
                out->normals.emplace_back(float(sign * gauge[3 * i]), float(sign * gauge[3 * i + 1]),
                                          float(sign * gauge[3 * i + 2]));
            }
        });
    } catch (const std::exception& problem) {
        return fail(tr("the solve's normals could not be read: %1").arg(QString::fromUtf8(problem.what())));
    }
    if (!out) return fail(reason);
    return out;
}

namespace {
// The current approved stretches of the fiber's approval layer, per aspect,
// as ranges of (fractional) line point indices. `content` holds the fiber as
// stored (its line points untransformed).
using ApprovedRanges = std::array<std::vector<std::pair<double, double>>, 2>;
ApprovedRanges approvedRangesOfLayer(const vc::objects::StoreReader& reader, const std::string& fiberUuid,
                                     const vc::objects::FiberPieceContent& content)
{
    using namespace vc::objects;
    ApprovedRanges ranges;
    const auto layer = readFiberApproval(reader, fiberUuid);
    if (!layer || layer->empty()) return ranges;
    const auto status = fiberApprovalStatus(*layer, content.linePoints);
    for (std::size_t aspect = 0; aspect < 2; ++aspect)
        ranges[aspect] = fiberStretchRanges(*layer, status, FiberApprovalAspect(aspect), FiberStretchState::Approved);
    return ranges;
}

// The line positions where broken geometry splits the fiber into parts: the
// middle of each broken stretch, as the solve splits it.
std::vector<double> splitPositionsOfLayer(const vc::objects::StoreReader& reader, const std::string& fiberUuid,
                                          const vc::objects::FiberPieceContent& content)
{
    using namespace vc::objects;
    std::vector<double> positions;
    const auto layer = readFiberApproval(reader, fiberUuid);
    if (!layer || layer->empty()) return positions;
    const auto status = fiberApprovalStatus(*layer, content.linePoints);
    for (const auto& [from, to] : fiberStretchRanges(*layer, status, FiberApprovalAspect::Geometry, FiberStretchState::Broken))
        positions.push_back(0.5 * (from + to));
    return positions;
}

// The stretches marked broken, geometry or normals, as line position ranges.
std::vector<std::pair<double, double>> brokenRangesOfLayer(const vc::objects::StoreReader& reader,
                                                           const std::string& fiberUuid,
                                                           const vc::objects::FiberPieceContent& content)
{
    using namespace vc::objects;
    std::vector<std::pair<double, double>> ranges;
    const auto layer = readFiberApproval(reader, fiberUuid);
    if (!layer || layer->empty()) return ranges;
    const auto status = fiberApprovalStatus(*layer, content.linePoints);
    for (const auto aspect : {FiberApprovalAspect::Geometry, FiberApprovalAspect::Normals})
        for (const auto& range : fiberStretchRanges(*layer, status, aspect, FiberStretchState::Broken))
            ranges.emplace_back(range.first, range.second);
    return ranges;
}

// Per span between two control points and per aspect: whether one approved
// stretch covers the whole span.
std::array<std::vector<std::uint8_t>, 2> approvedSpans(const ApprovedRanges& ranges,
                                                      const vc::objects::FiberPieceContent& content)
{
    const auto spans = content.controlIds.empty() ? 0 : content.controlIds.size() - 1;
    std::array<std::vector<std::uint8_t>, 2> approved;
    for (std::size_t aspect = 0; aspect < 2; ++aspect) {
        approved[aspect].assign(spans, 0);
        for (std::size_t span = 0; span < spans && span + 1 < content.controlLineIndex.size(); ++span) {
            const double from = double(content.controlLineIndex[span]), to = double(content.controlLineIndex[span + 1]);
            if (from < 0 || to < from) continue;
            for (const auto& [first, last] : ranges[aspect])
                if (from >= first - 1e-9 && to <= last + 1e-9) { approved[aspect][span] = 1; break; }
        }
    }
    return approved;
}

// The ranges as arcs of the line (`lineArcs`: one arc per line point).
ApprovedRanges approvedArcs(const ApprovedRanges& ranges, const std::vector<double>& lineArcs)
{
    ApprovedRanges arcs;
    if (lineArcs.empty()) return arcs;
    const auto arcAt = [&](double position) {
        position = std::clamp(position, 0., double(lineArcs.size() - 1));
        const auto index = std::min(std::size_t(position), lineArcs.size() - 2);
        const double part = position - double(index);
        return lineArcs[index] + part * (lineArcs[index + 1] - lineArcs[index]);
    };
    for (std::size_t aspect = 0; aspect < 2; ++aspect)
        for (const auto& [first, last] : ranges[aspect]) arcs[aspect].emplace_back(arcAt(first), arcAt(last));
    return arcs;
}
} // namespace

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
            const vc::objects::GeometrySnapshot snapshot(reader);
            for (const auto& record : reader.selectObjects(selection)) {
                const auto revision = reader.currentObject(record.uuid);
                if (!revision) continue;
                auto content = vc::objects::fiberPieceContent(reader, *revision);
                ScaffoldApprovalSpans entry;
                // Taken over the stored line, before it is mapped below.
                const auto ranges = approvedRangesOfLayer(reader, record.uuid, content);
                const auto splits = splitPositionsOfLayer(reader, record.uuid, content);
                ApprovedRanges brokenRanges;
                brokenRanges[0] = brokenRangesOfLayer(reader, record.uuid, content);
                entry.approved = approvedSpans(ranges, content);
                // The line in the runs' frame, read as the solve reads it:
                // the arcs below are then the arcs a run states.
                if (!_frameVolume.empty() && !content.linePoints.empty()) {
                    const auto view = snapshot.viewTransformedAt(*revision, _frameVolume, 1.);
                    content.linePoints = view.points("line_points", 0, content.linePoints.size());
                    content.controlPoints = view.points("control_points", 0, content.controlPoints.size());
                }
                entry.controlIds = content.controlIds;
                entry.controlArcs = scaffoldControlArcs(content);
                const auto lineArcs = scaffoldLineArcs(content);
                entry.approvedArcs = approvedArcs(ranges, lineArcs);
                // One range per split position, read back as its arc.
                ApprovedRanges splitRanges;
                for (const double position : splits) splitRanges[0].emplace_back(position, position);
                for (const auto& [arc, same] : approvedArcs(splitRanges, lineArcs)[0]) entry.splitArcs.push_back(arc);
                entry.brokenArcs = approvedArcs(brokenRanges, lineArcs)[0];
                loaded.emplace(record.uuid, std::move(entry));
            }
        });
    } catch (const std::exception& problem) {
        _lastError = tr("The fiber approvals could not be read: %1").arg(QString::fromUtf8(problem.what()));
        emit statusMessage(_lastError);
        return;
    }
    std::size_t unplaced = 0;
    for (const auto& [uuid, entry] : loaded)
        if (entry.controlArcs.empty()) ++unplaced;
    Logger()->info("scaffold approvals: {} fibers read in frame '{}', {} whose controls could not be placed on their line",
                   loaded.size(), _frameVolume, unplaced);
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
    if (request.measureCrossings && request.normalSource == "neighbours") {
        _job->neighbourPath = neighbourNormalsPath(session->path(*store));
        _job->neighbours = _neighbours;
    }
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
                (job->request.measureWindingDistances || job->request.normalSource != "geometry")) {
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
                if (job->request.normalSource == "neighbours") {
                    // Read once and kept: the file is the whole volume's
                    // normals, and a second run reads the same one.
                    std::error_code missing;
                    const auto written = std::filesystem::last_write_time(job->neighbourPath, missing);
                    if (missing)
                        throw std::runtime_error("No full-volume normals next to the fiber store: " +
                                                 job->neighbourPath.string() + " is missing.");
                    const std::lock_guard<std::mutex> guard(job->neighbours->mutex);
                    if (!job->neighbours->source.field || job->neighbours->path != job->neighbourPath ||
                        job->neighbours->written != written) {
                        job->neighbours->source = {};
                        job->neighbours->source = vc::fiber_graph::openScaffoldNeighbourSource(job->neighbourPath);
                        job->neighbours->path = job->neighbourPath;
                        job->neighbours->written = written;
                    }
                    job->neighbourSource = job->neighbours->source;
                }
                job->monitor.checkpoint();
            }
            job->cacheSink.author = job->session->newRequest(*job->attachment).author;
            job->cacheSink.computedAt = QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();
            const bool opens = job->request.measureCrossings &&
                               (job->request.measureWindingDistances || job->request.normalSource != "geometry");
            job->monitor.expectStages((job->request.measureCrossings
                                           ? (job->request.normalSource != "geometry" ? 13 : 12)
                                           : 4) + (opens ? 1 : 0));
            ScaffoldSolveOptions options;
            options.links.useLinks = job->request.useLinks;
            options.links.approvalStates = job->request.approvalStates;
            options.links.includeAuto = job->request.includeAuto;
            options.links.weight = job->request.manualWeight;
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
                vc::fiber_graph::ScaffoldFacingReport facing;
                std::vector<ScaffoldCrossing> crossings;
                ScaffoldRunStages stages;
                if (job->request.measureCrossings) {
                    normalOptions.cache = &cache;
                    normalOptions.sink = &job->cacheSink;
                    normalOptions.predictionIdentity = job->predictionIdentity;
                    normalOptions.neighbours.field = job->neighbourSource.field.get();
                    normalOptions.neighbours.identity = job->neighbourSource.identity;
                    normalOptions.neighbours.volume = job->neighbourSource.volume;
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
                    measuredOptions.signWeight = job->request.signWeight;
                    signOptions.sign.backend = job->request.signBackend == "ceres"
                                                   ? vc::fiber_tracer::FiberSignBackend::Ceres
                                                   : vc::fiber_tracer::FiberSignBackend::Irls;
                    signOptions.sign.threads = int(options.workers);
                    signOptions.monitor = &job->monitor;
                    // One crossing, one confidence: the winding weights take the sign stage's own term.
                    measuredOptions.confidence = signOptions.sign;
                    ScaffoldWindingDistances distances;
                    if (auto sampler = job->sampler)
                        distances = [sampler](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
                                              double step, int workers) {
                            return sampler->normalAlignedWindingDistancesBatch(connectors, step, workers);
                        };
                    normals.emplace(graph, geometry, normalOptions, job->axes);
                    events = scaffoldEvents(graph, geometry, *normals, eventOptions, distances, eventReport);
                    // The approved normals are faced first, as one object per
                    // connected set of stretches; contradictions fail the run.
                    facing = vc::fiber_graph::scaffoldFaceApprovedNormals(graph, geometry, *normals, events,
                                                                          normalOptions, eventOptions, signOptions,
                                                                          eventReport);
                    signs = scaffoldNormalSigns(graph, *normals, events, signOptions);
                    const auto stated = scaffoldCrossingApprovals(
                        graph, geometry, events, vc::objects::readFiberConstraintApprovals(reader), approvals);
                    measured = scaffoldMeasuredConstraints(graph, geometry, events, signs.sign, measuredOptions,
                                                           measuredReport, stated, &crossings);
                    stages.crossings = &crossings;
                    stages.normalOptions = &normalOptions;
                    stages.normals = &normals->report();
                    stages.normalRows = normals->cacheRows();
                    stages.eventOptions = &eventOptions;
                    stages.events = &eventReport;
                    stages.measuredOptions = &measuredOptions;
                    stages.measured = &measuredReport;
                    stages.signOptions = &signOptions;
                    stages.signs = &signs;
                    stages.approvals = &approvals;
                    stages.facing = &facing;
                }
                const auto result = solveScaffoldWinding(graph, geometry, options, measured);
                // Where the map links and the solved normals disagree about
                // which fiber is inside: the stretches to review first.
                std::vector<ScaffoldNormalFlip> flips;
                ScaffoldNormalFlipReport flipReport;
                if (normals) {
                    flips = scaffoldLinkNormalFlips(graph, geometry, *normals, signs.sign, result.constraints, flipReport);
                    stages.flips = &flips;
                    stages.flipReport = &flipReport;
                }
                record.input = {graph.source.storeUuid, graph.source.generation, graph.source.commitSequence};
                job->rows = scaffoldSolveRunRows(graph, result, options, record, stages);
                job->runUuid = record.runUuid;
                job->summary = QObject::tr("%1 fibers in %2 components, %3 turns of residual, %4 violated of %5 \n"
                                           "constraints, %6 of %7 continuation links broken, %8 s")
                                   .arg(graph.fibers.size())
                                   .arg(result.components)
                                   .arg(result.objective, 0, 'f', 3)
                                   .arg(result.violatedConstraints)
                                   .arg(result.constraints.size())
                                   .arg(result.brokenContinuations)
                                   .arg(result.continuationLinks)
                                   .arg(result.seconds, 0, 'f', 3);
                if (job->request.measureCrossings)
                    job->summary += QObject::tr("; %1 crossings, %2 measured relations, %3 enclosures, %4 signs from \n"
                                                "crossings")
                                        .arg(eventReport.events)
                                        .arg(measuredReport.used)
                                        .arg(measuredReport.enclosureArcs)
                                        .arg(signs.decided);
                if (normals && normalOptions.neighbours.field)
                    job->summary += QObject::tr("; full-volume neighbours faced %1 fibers and answered %2 of %3 \n"
                                                "samples")
                                        .arg(normals->report().neighbourFacedFibers)
                                        .arg(normals->report().neighbourSamplesAnswered)
                                        .arg(normals->report().neighbourSamples);
                // A crossing whose geometry and whose H/V tags disagree is
                // dropped, and that is a question about the tagging: say so
                // rather than leaving it in the gap between read and used.
                if (normals && normals->report().approvedNormalPoints > 0)
                    job->summary += QObject::tr("; approved normals held at %1 line points")
                                        .arg(normals->report().approvedNormalPoints);
                // A normals approval the solve could not use approves nothing
                // until it is given again: say so instead of guessing.
                if (normals && normals->report().staleNormalApprovals > 0)
                    job->summary += QObject::tr("; %1 normals approvals are stale or carry no stored normals \n"
                                                "and were not used: approve those stretches again")
                                        .arg(normals->report().staleNormalApprovals);
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
// The store's kind of a stored residual's constraint, or nullopt for a map
// link (whose approval is its edge record's review state).
std::optional<vc::objects::FiberConstraintKind> storedConstraintKind(const ScaffoldStoredResidual& residual)
{
    if (!residual.edgeUuid.empty()) return std::nullopt;
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

std::vector<ScaffoldFiberCrossing> ScaffoldSolveController::crossingsOf(const std::string& fiberUuid) const
{
    std::vector<ScaffoldFiberCrossing> out;
    if (!_selected) return out;
    for (const auto& crossing : _selected->crossings) {
        const bool first = crossing.fiberA == fiberUuid;
        if (!first && crossing.fiberB != fiberUuid) continue;
        ScaffoldFiberCrossing entry;
        entry.arc = first ? crossing.arcA : crossing.arcB;
        entry.otherUuid = first ? crossing.fiberB : crossing.fiberA;
        entry.otherArc = first ? crossing.arcB : crossing.arcA;
        entry.estimateHalves = crossing.estimateHalves;
        entry.signKnown = crossing.signKnown;
        entry.separation = crossing.separation;
        try {
            const auto found = _constraintApprovals.find(vc::objects::fiberConstraintIdentity(
                crossing.fiberA, crossing.arcA, crossing.fiberB, crossing.arcB,
                vc::objects::FiberConstraintKind::Perpendicular));
            if (found != _constraintApprovals.end())
                entry.state = found->second.distance == vc::objects::FiberConstraintDistance::Distant ? 2 : 1;
        } catch (const std::exception&) {
            continue; // not a crossing an approval can name
        }
        out.push_back(std::move(entry));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.arc < b.arc; });
    return out;
}

QString ScaffoldSolveController::setCrossingApproval(const std::string& fiberUuid, const ScaffoldFiberCrossing& crossing,
                                                     int state)
{
    using namespace vc::objects;
    if (state < 0 || state > 2) return tr("Unknown neighbour state.");
    const auto store = attachment();
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!store || !session) return tr("No fiber store is attached.");
    FiberConstraintIdentity identity;
    try {
        identity = fiberConstraintIdentity(fiberUuid, crossing.arc, crossing.otherUuid, crossing.otherArc,
                                           FiberConstraintKind::Perpendicular);
    } catch (const std::exception& problem) {
        return tr("This crossing cannot be named: %1").arg(QString::fromUtf8(problem.what()));
    }
    const auto existing = _constraintApprovals.find(identity);
    if (state == 0 && existing == _constraintApprovals.end()) return tr("This crossing carries no state to take back.");
    if (!_approvalsLoaded) loadApprovals();

    auto request = session->newRequest(*store);
    const std::string now = QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();
    const std::string host = QSysInfo::machineHostName().toStdString();
    std::vector<ExtensionRow> values;
    std::vector<ObjectRevision> revisions;
    QString error;
    try {
        session->readStore(*store, [&](const StoreReader& reader, const auto&) {
            if (state == 0) {
                values.push_back(
                    fiberConstraintApprovalRow(revokeFiberConstraintApproval(existing->second, request.author, now)));
                return;
            }
            // Both fibers as the store holds them now: the approval records
            // their versions and the fingerprints of the two covering spans.
            std::array<FiberPieceContent, 2> contents;
            std::array<std::int64_t, 2> versions{0, 0};
            std::array<std::size_t, 2> spans{0, 0};
            const std::array<std::string, 2> uuids{identity.fiberA, identity.fiberB};
            const bool ordered = identity.fiberA == fiberUuid;
            const std::array<double, 2> arcs{ordered ? crossing.arc : crossing.otherArc,
                                             ordered ? crossing.otherArc : crossing.arc};
            for (std::size_t side = 0; side < 2; ++side) {
                const auto record = reader.currentObject(uuids[side]);
                if (!record) { error = tr("A fiber of this crossing is no longer in the store."); return; }
                const auto found = reader.objectsAtVersions({record->reference()});
                if (found.empty() || !found.front() || found.front()->deleted) {
                    error = tr("A fiber of this crossing cannot be read.");
                    return;
                }
                revisions.push_back(*found.front());
                contents[side] = fiberPieceContent(reader, *found.front());
                versions[side] = found.front()->version;
                // The span the anchor lies on, as the solve places it among
                // the fiber's control points (arcs are the solve's own).
                const auto span = anchorSpan(uuids[side], arcs[side]);
                if (!span.known || std::size_t(span.span) + 1 >= contents[side].controlPoints.size()) {
                    error = tr("Fiber %1 of this crossing has changed since the solve: solve again.")
                                .arg(QString::fromStdString(uuids[side]).left(8));
                    return;
                }
                spans[side] = std::size_t(span.span);
            }
            auto row = approveFiberConstraint(identity, contents[0], spans[0], versions[0], contents[1], spans[1],
                                              versions[1],
                                              state == 2 ? FiberConstraintDistance::Distant
                                                         : FiberConstraintDistance::Direct,
                                              request.author, now, host);
            // A changed state replaces the live row of this crossing.
            if (existing != _constraintApprovals.end()) {
                row.supersedes = existing->second.rowUuid;
                row.approvalUuid = existing->second.approvalUuid;
            }
            values.push_back(fiberConstraintApprovalRow(row));
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
        return tr("The neighbour state could not be stored: %1").arg(QString::fromUtf8(problem.what()));
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

    // No settings: a run is one of the solver profiles.
    auto* form = new QFormLayout();
    form->setContentsMargins(0, 0, 0, 0);
    form->setSpacing(2);
    _profileCombo = new QComboBox(this);
    _profileCombo->setObjectName(QStringLiteral("fiberMapSolveProfile"));
    _profileCombo->addItem(tr("default scaffold"), QStringLiteral("default_scaffold"));
    _profileCombo->setToolTip(tr("The solver profile of the run.\n"
                                 "default scaffold: the whole scaffold, no links, measured crossings with\n"
                                 "winding separations, perpendicular weight 1, sign weight 3 (30 on\n"
                                 "approved normals), continuation links at weight 10, normals from the\n"
                                 "full-volume neighbours."));
    form->addRow(tr("Profile"), _profileCombo);
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
    const bool busy = _controller && _controller->busy();
    _runButton->setEnabled(!busy && !_volume.empty());
    _cancelButton->setEnabled(busy);
    if (_volume.empty()) _runButton->setToolTip(tr("The map has no annotation volume to solve in."));
}

ScaffoldSolveRequest ScaffoldSolverPanel::request() const
{
    // The one profile, "default scaffold".
    ScaffoldSolveRequest request;
    request.approvalStates.clear();
    request.useLinks = false;
    request.includeAuto = false;
    request.manualWeight = 1.0;
    request.lemonBackend = "cost_scaling";
    request.volume = _volume;
    request.fallbackHandedness = _fallbackHandedness;
    request.measureCrossings = true;
    request.measureWindingDistances = true;
    request.perpendicularWeight = 1.0;
    request.parallelWeight = 0.0;
    request.enclosureWeight = 0.0;
    request.signWeight = 3.0;
    request.normalSource = "neighbours";
    request.signBackend = "irls";
    request.workers = std::size_t(std::max(1, QThread::idealThreadCount()));
    request.restricted = false;
    return request;
}


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
    if (_controller) {
        connect(_controller, &ScaffoldSolveController::runsChanged, this, [this]() { rebuild(); });
        connect(_tree, &QTreeWidget::itemSelectionChanged, this, [this]() { applySelection(); });
        // A solve selected elsewhere (the one that just finished) is the selected row.
        connect(_controller, &ScaffoldSolveController::selectionChanged, this, [this]() {
            const auto wanted = _controller->selected() ? _controller->selected()->run.runUuid : std::string();
            const auto items = _tree->selectedItems();
            const auto shown = items.isEmpty()
                ? std::string() : items.front()->data(kRunMethodColumn, Qt::UserRole).toString().toStdString();
            if (shown == wanted) return;
            _rebuilding = true;
            _tree->clearSelection();
            for (int row = 0; row < _tree->topLevelItemCount(); ++row) {
                auto* item = _tree->topLevelItem(row);
                if (item->data(kRunMethodColumn, Qt::UserRole).toString().toStdString() != wanted) continue;
                item->setSelected(true);
                _tree->scrollToItem(item);
            }
            _rebuilding = false;
            showDiff();
        });
    }
    rebuild();
}


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
    _orderCombo->addItem(tr("highest cost"), int(ScaffoldApprovalOrder::Cost));
    _orderCombo->addItem(tr("cost per voxel"), int(ScaffoldApprovalOrder::Density));
    _orderCombo->setToolTip(tr("highest cost: the stretch with the most violated cost on it first.\n"
                               "cost per voxel: that cost over the stretch's length."));
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
    _tree->setHeaderLabels({tr("geo"), tr("nrm"), tr("approved w."), tr("winding"), tr("sign"), tr("cont"), tr("where"), tr("cost"),
                            tr("cost/kvx"), tr("length"), tr("fiber")});
    _tree->headerItem()->setToolTip(kGeometryColumn, tr("A check: the stretch is approved for geometry."));
    _tree->headerItem()->setToolTip(kNormalsColumn, tr("A check: the stretch is approved for normals."));
    _tree->headerItem()->setToolTip(kReviewedColumn,
                                    tr("Crossings on the stretch whose winding distance you approved (direct or distant),\n"
                                       "of all crossings on it."));
    _tree->headerItem()->setToolTip(kViolatedColumn,
                                    tr("Violated winding constraints on the stretch (how many windings two fibers\n"
                                       "are apart; a violated link counts too), of the winding constraints on it."));
    _tree->headerItem()->setToolTip(kSignColumn,
                                    tr("Violated sign constraints on the stretch (which fiber lies outside), of its\n"
                                       "crossings whose normals state a side."));
    _tree->headerItem()->setToolTip(kContinuationColumn,
                                    tr("Continuation links on the stretch that the solve broke (the fiber changes\n"
                                       "winding there), of the links on it."));
    _tree->headerItem()->setToolTip(kCostColumn, tr("Summed cost of the violated constraints on the stretch, in weighted turns."));
    _tree->headerItem()->setToolTip(kDensityColumn, tr("Cost per 1000 voxels of the stretch."));
    _tree->headerItem()->setToolTip(kLengthColumn, tr("The stretch's length (voxels)."));
    _tree->setUniformRowHeights(true);
    _tree->setRootIsDecorated(true);
    _tree->setExpandsOnDoubleClick(false);
    _tree->setSortingEnabled(true);
    _tree->setSelectionMode(QAbstractItemView::SingleSelection);
    _tree->header()->setStretchLastSection(true);
    _tree->setToolTip(tr("What the selected solve could not satisfy. A violated constraint is\n"
                         "listed with the shortest cycle it lies on.\n"
                         "A geometry stretch is a piece of one fiber between constraints of such\n"
                         "cycles, with the share of the violated cost it is accused of; approved\n"
                         "geometry takes no share.\n"
                         "A normals stretch reaches from the crossing of a violated sign\n"
                         "constraint to at least one other crossing of that fiber; approving its\n"
                         "normals ties those crossings to one facing.\n"
                         "Selecting a row shows it on the map; double-clicking a stretch opens it\n"
                         "in the fiber approval window."));
    layout->addWidget(_tree, 1);

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
    // Loading the approvals announces them, and that announcement asks for a
    // rebuild: the one already running is that rebuild. Without this the
    // inner one filled the list and the outer one then appended every row a
    // second time.
    if (_rebuilding) return;
    _rebuilding = true;
    // Before anything is cleared or read: the state column and the approve
    // button need them.
    if (_controller && _controller->selected() && !_controller->approvalsLoaded()) _controller->loadApprovals();
    _tree->setSortingEnabled(false);
    _tree->clear();
    _rows.clear();
    const auto* view = _controller ? _controller->selected() : nullptr;
    if (!view) {
        _summaryLabel->setText(tr("Select a solve to see what it could not satisfy."));
        _rebuilding = false;
        return;
    }
    // A solve of an annotation that has changed since names stretches that
    // may no longer exist: nothing is listed from it.
    if (_controller->runStale(view->run)) {
        _summaryLabel->setText(tr("The selected solve is outdated: a fiber, a link or the umbilicus changed after it "
                                  "ran. Nothing is listed from it. Press Solve in the scaffold solver dock."));
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
        case std::int64_t(ScaffoldConstraintKind::SignAtLeast): return tr("sign (at least)");
        case std::int64_t(ScaffoldConstraintKind::SignAtMost): return tr("sign (at most)");
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
    std::vector<Entry> entries;
    // A control position as text: a whole number on a control point, two
    // decimals between two of them - never a range the segment does not have.
    const auto positionText = [](double position) {
        const double nearest = std::round(position);
        return std::abs(position - nearest) < 5e-3 ? QString::number(qlonglong(nearest))
                                                    : QString::number(position, 'f', 2);
    };
    // A stretch row: where it lies, what it draws and what it opens. The
    // ends are constraint positions, shown as control positions; the row
    // opens between the control points bracketing them outward.
    const auto stretchEntry = [&](const std::string& uuid, double arcFrom, double arcTo, QString& where) {
        Entry entry;
        entry.row.kind = Row::Kind::Piece;
        entry.row.keys.length = arcTo - arcFrom;
        entry.row.fiberUuid = uuid;
        entry.row.arcFrom = arcFrom;
        entry.row.arcTo = arcTo;
        where = tr("%1-%2 vx").arg(arcFrom, 0, 'f', 0).arg(arcTo, 0, 'f', 0);
        if (!_controller) return entry;
        const auto found = _controller->approvals().find(uuid);
        if (found == _controller->approvals().end()) return entry;
        // A stretch outside the fiber's control points is fiber the
        // annotation no longer has: it is not placed, and selecting it says so.
        const auto& arcs = found->second.controlArcs;
        if (arcs.size() < 2 || arcFrom < arcs.front() - 1e-6 || arcTo > arcs.back() + 1e-6) return entry;
        const double positionFrom = scaffoldControlPosition(arcs, arcFrom), positionTo = scaffoldControlPosition(arcs, arcTo);
        entry.row.positionFrom = positionFrom;
        entry.row.positionTo = positionTo;
        const auto last = std::int64_t(arcs.size()) - 1;
        std::int64_t from = std::int64_t(std::floor(positionFrom));
        std::int64_t to = std::max(std::int64_t(std::ceil(positionTo)), from + 1);
        // On the fiber's last control point the covering span is the last one.
        if (to > last) { to = last; from = std::min(from, last - 1); }
        entry.row.controlFrom = from;
        entry.row.controlTo = to;
        where = arcTo > arcFrom ? tr("cp %1-%2").arg(positionText(positionFrom), positionText(positionTo))
                                : tr("cp %1").arg(positionText(positionFrom));
        return entry;
    };
    // The stretches: derived here, from what the selected solve left violated
    // and what is approved now (scaffoldStretches). A solve does not hold
    // stretches; an approval moves this list at once.
    static const std::map<std::string, ScaffoldApprovalSpans> noApprovals;
    const auto& approvals = _controller ? _controller->approvals() : noApprovals;
    std::vector<std::string> stretchUuids;
    std::map<std::string, std::int64_t> stretchIndex;
    std::vector<vc::fiber_graph::ScaffoldStretchFiber> stretchFibers;
    for (const auto& [uuid, spans] : approvals) {
        // Only the fibers this run solved, placed among their controls.
        if (!view->windings.contains(uuid) || spans.controlArcs.size() < 2) continue;
        vc::fiber_graph::ScaffoldStretchFiber fiber;
        fiber.length = spans.controlArcs.back();
        fiber.cuts = spans.splitArcs;
        fiber.broken = spans.brokenArcs;
        for (const auto& ranges : spans.approvedArcs) fiber.approved.insert(fiber.approved.end(), ranges.begin(), ranges.end());
        stretchIndex.emplace(uuid, std::int64_t(stretchFibers.size()));
        stretchUuids.push_back(uuid);
        stretchFibers.push_back(std::move(fiber));
    }
    const auto indexOf = [&](const std::string& uuid) {
        const auto found = stretchIndex.find(uuid);
        return found == stretchIndex.end() ? std::int64_t(-1) : found->second;
    };
    std::vector<vc::fiber_graph::ScaffoldStretchPlace> stretchPlaces;
    for (const auto& crossing : view->crossings) {
        const std::int64_t constraints = (crossing.signKnown ? 1 : 0) + (crossing.separation ? 1 : 0);
        stretchPlaces.push_back({indexOf(crossing.fiberA), crossing.arcA, constraints});
        stretchPlaces.push_back({indexOf(crossing.fiberB), crossing.arcB, constraints});
    }
    std::vector<vc::fiber_graph::ScaffoldStretchViolation> stretchViolations;
    for (const auto& residual : view->residuals) {
        const double cost = residual.weight * residual.residual;
        const auto a = indexOf(residual.fiberA), b = indexOf(residual.fiberB);
        stretchViolations.push_back({a, residual.arcA, cost, b, residual.arcB});
        stretchViolations.push_back({b, residual.arcB, cost, a, residual.arcA});
    }
    vc::fiber_graph::ScaffoldStretchReport stretchReport;
    const auto stretches =
        vc::fiber_graph::scaffoldStretches(stretchFibers, stretchPlaces, stretchViolations, {}, stretchReport);
    std::map<std::string, std::vector<ScaffoldFiberCrossing>> crossingsOfFiber;
    const QString check = QStringLiteral("\u2713");
    for (const auto& computed : stretches) {
        // The row's values under the names the columns below read.
        struct {
            std::string fiberA, fiberB;
            double residual = 0, weight = 0;
            std::int64_t target = 0;
        } stretch;
        stretch.fiberA = stretchUuids[std::size_t(computed.fiber)];
        if (computed.other >= 0 && std::size_t(computed.other) < stretchUuids.size())
            stretch.fiberB = stretchUuids[std::size_t(computed.other)];
        stretch.residual = computed.cost;
        stretch.weight = double(computed.constraints);
        stretch.target = computed.violations;
        const auto& piece = computed.piece;
        // Per aspect: how much of the stretch is approved, 0 to 1.
        std::array<double, 2> approved{0., 0.};
        // Violated on the stretch, by what the constraint is about: the side
        // (sign) or the winding step (a separation or a link).
        std::int64_t crossings = 0, reviewed = 0, signed_ = 0, signViolated = 0, separations = 0, windingViolated = 0;
        std::int64_t continuations = 0, continuationsBroken = 0;
        for (const auto& link : view->continuations) {
            if (link.fiber != stretch.fiberA || link.arc < piece.arcFrom || link.arc > piece.arcTo) continue;
            ++continuations;
            if (link.residual > 0) ++continuationsBroken;
        }
        for (const auto& residual : view->residuals) {
            const auto on = [&](const std::string& fiber, double arc) {
                return fiber == stretch.fiberA && arc >= piece.arcFrom && arc <= piece.arcTo;
            };
            if (!on(residual.fiberA, residual.arcA) && !on(residual.fiberB, residual.arcB)) continue;
            // A broken continuation link has its own column.
            if (residual.kind == std::int64_t(ScaffoldConstraintKind::Continuation)) continue;
            const bool sign = residual.kind == std::int64_t(ScaffoldConstraintKind::SignAtLeast) ||
                              residual.kind == std::int64_t(ScaffoldConstraintKind::SignAtMost);
            ++(sign ? signViolated : windingViolated);
        }
        if (_controller) {
            const auto found = _controller->approvals().find(stretch.fiberA);
            if (found != _controller->approvals().end()) {
                for (std::size_t aspect = 0; aspect < 2; ++aspect) {
                    double covered = 0;
                    for (const auto& [first, last] : found->second.approvedArcs[aspect])
                        covered += std::max(0., std::min(piece.arcTo, last) - std::max(piece.arcFrom, first));
                    approved[aspect] = std::min(1., covered / std::max(piece.arcTo - piece.arcFrom, 1.));
                }
            }
            auto list = crossingsOfFiber.find(stretch.fiberA);
            if (list == crossingsOfFiber.end())
                list = crossingsOfFiber.emplace(stretch.fiberA, _controller->crossingsOf(stretch.fiberA)).first;
            for (const auto& crossing : list->second) {
                if (crossing.arc < piece.arcFrom || crossing.arc > piece.arcTo) continue;
                ++crossings;
                if (crossing.state != 0) ++reviewed;
                if (crossing.signKnown) ++signed_;
                if (crossing.separation) ++separations;
            }
        }
        QString where;
        auto entry = stretchEntry(stretch.fiberA, piece.arcFrom, piece.arcTo, where);
        const double length = piece.arcTo - piece.arcFrom;
        entry.row.keys.cost = stretch.residual;
        entry.row.keys.density = stretch.residual / std::max(length, 1.);
        entry.row.otherUuid = stretch.fiberB;
        entry.row.reviewedShare = crossings > 0 ? double(reviewed) / double(crossings) : 0.;
        entry.row.violated = double(windingViolated);
        entry.row.signViolated = double(signViolated);
        entry.row.continuationsBroken = double(continuationsBroken);
        // A check for a stretch approved as a whole. A new
        // stretch only reaches a margin into an approved neighbour, which
        // is not an approval of it.
        const auto approvalText = [&](double share) {
            const double length = piece.arcTo - piece.arcFrom;
            return share * length >= length - 0.5 ? check : QString();
        };
        entry.row.approvedShare = approved;
        entry.text << approvalText(approved[0]) << approvalText(approved[1])
                   << tr("%1/%2").arg(reviewed).arg(crossings)
                   << tr("%1/%2").arg(windingViolated).arg(std::max(separations, windingViolated))
                   << tr("%1/%2").arg(signViolated).arg(std::max(signed_, signViolated))
                   << tr("%1/%2").arg(continuationsBroken).arg(continuations) << where
                   << QString::number(stretch.residual, 'f', 3)
                   << QString::number(1000. * entry.row.keys.density, 'f', 3) << QString::number(length, 'f', 0)
                   << label(stretch.fiberA);
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
        item->setData(kGeometryColumn, Qt::UserRole, qlonglong(_rows.size()));
        // Numeric sort keys for the columns a header click orders by; the
        // text of those columns is formatted for reading.
        const auto number = [&](int column, double value) {
            item->setData(column, Qt::UserRole + 1, value);
        };
        number(kCostColumn, row.keys.cost);
        number(kDensityColumn, row.keys.density);
        number(kLengthColumn, row.keys.length);
        number(kGeometryColumn, row.approvedShare[0]);
        number(kNormalsColumn, row.approvedShare[1]);
        number(kReviewedColumn, row.reviewedShare);
        number(kViolatedColumn, row.violated);
        number(kSignColumn, row.signViolated);
        number(kContinuationColumn, row.continuationsBroken);
        _rows.push_back(row);
        return item;
    };
    for (const auto& entry : entries) append(nullptr, entry.row, entry.text);
    _tree->setSortingEnabled(true);
    for (int column = 0; column < kColumnCount - 1; ++column) _tree->resizeColumnToContents(column);
    _summaryLabel->setText(filter >= 0
        ? tr("%1 stretches to approve in component %2; the solve left %3 constraints violated")
              .arg(entries.size()).arg(filter).arg(view->residuals.size())
        : tr("%1 stretches to approve; the solve left %2 constraints violated")
              .arg(entries.size()).arg(view->residuals.size()));
    _rebuilding = false;
    applySelection();
}

const ScaffoldApprovalPanel::Row* ScaffoldApprovalPanel::rowOf(const QTreeWidgetItem* item) const
{
    if (!item) return nullptr;
    const auto index = item->data(kGeometryColumn, Qt::UserRole).toLongLong();
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
    if (items.isEmpty()) {
        emit highlightRequested({});
        return;
    }
    if (!rowOf(items.front())) return;
    emit highlightRequested(highlightOf(items.front()));
}

std::vector<std::pair<double, double>> ScaffoldApprovalPanel::stretchArcs(const std::string& fiberUuid) const
{
    std::vector<std::pair<double, double>> out;
    for (const auto& row : _rows)
        if (row.kind == Row::Kind::Piece && row.fiberUuid == fiberUuid && row.arcTo > row.arcFrom)
            out.emplace_back(row.arcFrom, row.arcTo);
    return out;
}

ScaffoldHighlight ScaffoldApprovalPanel::highlightOf(const QTreeWidgetItem* item) const
{
    ScaffoldHighlight highlight;
    const auto* row = rowOf(item);
    if (!row) return highlight;
    const auto* residual = residualOf(item);
    if (!row->fiberUuid.empty()) highlight.fiberUuids.push_back(row->fiberUuid);
    if (!row->otherUuid.empty()) highlight.fiberUuids.push_back(row->otherUuid);
    // Every row names stretches. One that cannot be stated is an error with
    // its reason, never the whole fiber.
    if (row->kind == Row::Kind::Piece) {
        if (row->controlFrom >= 0 && row->controlTo > row->controlFrom) {
            highlight.spans.push_back(
                {row->fiberUuid, row->controlFrom, row->controlTo, row->positionFrom, row->positionTo});
        } else {
            auto why = _controller ? _controller->anchorSpanProblem(row->fiberUuid, row->arcFrom) : tr("no solve controller");
            if (why.isEmpty())
                why = tr("fiber %1: the stretch does not lie between the fiber's control points as they are now "
                         "(the run is older than the annotation; solve again)")
                          .arg(QString::fromStdString(row->fiberUuid).left(8));
            highlight.errors.push_back(tr("stretch at arc %1-%2 vx: %3")
                                           .arg(row->arcFrom, 0, 'f', 0).arg(row->arcTo, 0, 'f', 0).arg(why).toStdString());
        }
    } else if (residual) {
        // A constraint is drawn as the two pieces its anchors lie on, which
        // is what makes it visible on the map at all.
        const auto spans = coveringSpans(*residual);
        const std::array<std::string, 2> uuids{residual->fiberA, residual->fiberB};
        const std::array<double, 2> arcs{residual->arcA, residual->arcB};
        for (std::size_t side = 0; side < 2; ++side) {
            if (spans[side].known) {
                highlight.spans.push_back({uuids[side], spans[side].controlFrom, spans[side].controlTo});
                continue;
            }
            auto why = _controller ? _controller->anchorSpanProblem(uuids[side], arcs[side]) : tr("no solve controller");
            if (why.isEmpty()) why = tr("fiber %1: no span covers the anchor").arg(QString::fromStdString(uuids[side]).left(8));
            highlight.errors.push_back(
                tr("constraint anchor at arc %1 vx: %2").arg(arcs[side], 0, 'f', 0).arg(why).toStdString());
        }
    } else {
        highlight.errors.push_back(tr("this row names no stretch and no constraint").toStdString());
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
    if (row->fiberUuid.empty() || row->controlFrom < 0 || row->controlTo <= row->controlFrom) {
        // A stretch without a control range cannot be opened, and says why.
        const auto highlight = highlightOf(item);
        QStringList reasons;
        for (const auto& error : highlight.errors) reasons << QString::fromStdString(error);
        if (_controller)
            emit _controller->statusMessage(tr("ERROR: stretch not opened - %1")
                                                .arg(reasons.isEmpty() ? tr("it has no control range") : reasons.join(QStringLiteral("; "))));
        return;
    }
    emit pieceActivated(row->fiberUuid, row->controlFrom, row->controlTo, row->arcFrom, row->arcTo);
}

} // namespace vc3d::fiber_map
