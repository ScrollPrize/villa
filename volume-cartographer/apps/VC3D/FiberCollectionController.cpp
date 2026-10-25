#include "FiberCollectionController.hpp"
#include "ObjectStoreEvents.hpp"
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "ObjectStoreRegions.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QColor>
#include <QFutureWatcher>
#include <QGraphicsView>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QScopeGuard>
#include <QSettings>
#include <QStyleHints>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <functional>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

using vc::fibers::Bounds;
using vc::fibers::Point;
using Json = nlohmann::json;
using vc3d::store::background;
using vc3d::store::Outcome;
namespace
{
constexpr int64_t kCatalogPageSize = 200;
cv::Vec3f cvpoint(const Point& p, double scale = 1)
{
    return {float(p[0] * scale), float(p[1] * scale), float(p[2] * scale)};
}
Point point(const cv::Vec3f& p, double scale = 1)
{
    return {p[0] * scale, p[1] * scale, p[2] * scale};
}
double dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
bool inside(const Point& local, const Bounds& bounds)
{
    for (size_t i = 0; i < 3; ++i)
        if (local[i] < bounds.lower[i] || local[i] > bounds.upper[i])
            return false;
    return true;
}
// Little-endian float64 XYZ rows, as the store returns them.
Point xyzAt(const std::vector<uint8_t>& bytes, size_t row)
{
    Point p{};
    for (size_t axis = 0; axis < 3; ++axis) {
        uint64_t bits = 0;
        for (size_t byte = 0; byte < 8; ++byte)
            bits |= uint64_t(bytes[row * 24 + axis * 8 + byte]) << (8 * byte);
        p[axis] = std::bit_cast<double>(bits);
    }
    return p;
}
// Consecutive groups are separated by the golden angle; saturation and value
// bands keep unusually large sets distinct.
QColor groupColor(uint64_t id)
{
    const int hue = static_cast<int>((id * 137ULL) % 360ULL);
    const int saturation = 190 + static_cast<int>((id / 360ULL) % 3ULL) * 25;
    const int value = 255 - static_cast<int>((id / 1080ULL) % 3ULL) * 20;
    return QColor::fromHsv(hue, saturation, value);
}
// The look of one linked control point, from every link on it, by the Line
// Annotation views' marker rule: pending wins, automatic only while every
// pending link on the point is automatic, the same-H/V warning of the winning
// state, a triangle for any adjacent link.
struct LinkMark {
    bool linked = false;
    bool pending = false;
    bool humanPending = false;
    bool sameHvPending = false;
    bool sameHvApproved = false;
    bool adjacent = false;
    void add(const vc::objects::EdgeRecord& edge, bool sameHv)
    {
        const auto meta = Json::parse(edge.meta);
        const bool linkPending = edge.approvalState != "approved";
        const bool linkAdjacent = meta.value("adjacent", false);
        const bool automatic = linkPending && !linkAdjacent && meta.value("auto", false);
        linked = true;
        pending = pending || linkPending;
        humanPending = humanPending || (linkPending && !automatic);
        adjacent = adjacent || linkAdjacent;
        if (sameHv) (linkPending ? sameHvPending : sameHvApproved) = true;
    }
    bool automatic() const { return pending && !humanPending; }
    bool sameHv() const { return pending ? sameHvPending : sameHvApproved; }
};
// Whether a link joins two fibers of the same known H/V family.
std::function<bool(const vc::objects::EdgeRecord&)> sameFamilyTest(
    const vc::objects::ObjectStore& db, const vc::objects::ReadTxn& txn,
    const std::vector<vc::objects::EdgeRecord>& edges)
{
    vc::objects::ObjectSelection selection;
    for (const auto& edge : edges) {
        selection.uuids.push_back(edge.aUuid);
        selection.uuids.push_back(edge.bUuid);
    }
    std::map<std::string, std::string> families;
    if (!selection.uuids.empty())
        for (const auto& object : db.selectObjects(txn, selection, std::vector<std::string>{"fiber"}))
            if (object.fiber && !object.fiber->family.empty()) families[object.uuid] = object.fiber->family;
    return [families = std::move(families)](const vc::objects::EdgeRecord& edge) {
        const auto a = families.find(edge.aUuid), b = families.find(edge.bUuid);
        return a != families.end() && b != families.end() && a->second == b->second;
    };
}
ViewerOverlayControllerBase::OverlayStyle linkMarkStyle(const LinkMark& mark, float opacity)
{
    const auto rgb = vc3d::line_annotation::linkStateRgb(mark.pending, mark.sameHv(), mark.automatic());
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(rgb.r, rgb.g, rgb.b, 245);
    style.brushColor = QColor(rgb.r, rgb.g, rgb.b, 175);
    style.penColor.setAlphaF(style.penColor.alphaF() * opacity);
    style.brushColor.setAlphaF(style.brushColor.alphaF() * opacity);
    style.penWidth = 2.0;
    style.z = 95;
    return style;
}
struct SelectedFiber {
    FiberCollectionController::CatalogRow row;
    vc::objects::PinnedObject pin;
    int64_t pointCount = 0;
};
struct RegionResult {
    vc::fibers::RegionPage page;
    std::vector<std::string> blockUuids;
    struct Control {
        std::string uuid;
        int index = 0;
        Point position;
    };
    std::vector<Control> controls;
    struct Ring {
        Point position;
        LinkMark mark;
    };
    std::vector<Ring> rings;
    // Visible fiber -> smallest UUID of its linked component.
    std::map<std::string, std::string> linkGroups;
};
struct ChainResult {
    struct Chain {
        std::string uuid;
        std::vector<cv::Vec3f> points;
        // Per control point (unlinked when !linked).
        std::vector<LinkMark> linkMarks;
    };
    std::vector<Chain> chains;
    std::map<std::string, std::string> linkGroups;
};
// Smallest UUID of each linked component among the given fibers.
std::map<std::string, std::string> linkGroupsOf(const vc::objects::ObjectStore& db, const vc::objects::ReadTxn& txn,
                                                const std::set<std::string>& linked, const std::set<std::string>& drawn,
                                                const std::atomic_bool& cancelled)
{
    std::map<std::string, std::string> groups;
    for (const auto& uuid : linked) {
        if (groups.contains(uuid)) continue;
        const auto component = db.componentOf(txn, uuid, {"fiber_branch"}, &cancelled);
        const auto representative = *std::min_element(component.begin(), component.end());
        for (const auto& member : component)
            if (drawn.contains(member)) groups[member] = representative;
    }
    return groups;
}
// Stored units -> volume voxels of each listed object bound to the volume.
template <class Keys>
std::map<std::string, double> volumeFactors(const vc::objects::ObjectStore& db, const vc::objects::ReadTxn& txn,
                                            const Keys& objects, const std::string& volume)
{
    vc::objects::ObjectSelection selection;
    for (const auto& [uuid, value] : objects) selection.uuids.push_back(uuid);
    std::map<std::string, double> factors;
    for (const auto& object : db.selectObjects(txn, selection, std::vector<std::string>{"meta"}))
        if (const auto factor = vc::objects::geometryScaleToVolume(Json::parse(object.meta), volume))
            factors.emplace(object.uuid, *factor);
    return factors;
}
}  // namespace

struct FiberCollectionController::View {
    vc3d::store::RegionRequests region;
    std::optional<Slice> displayed;
    std::vector<vc::fibers::Block> blocks;
    struct Control {
        uint64_t fiber = 0;
        int index = 0;
        Point position;
    };
    std::vector<Control> controls;
    std::vector<RegionResult::Ring> rings;
    std::unordered_map<uint64_t, uint64_t> linkGroups;
    struct Chain {
        uint64_t fiber = 0;
        std::vector<cv::Vec3f> points;
        std::vector<LinkMark> linkMarks;
    };
    std::vector<Chain> chains;
    std::optional<std::array<double, 6>> chainBox;
    bool dirty{true};
    QRectF displayedRect;
    qreal displayedDpr{};
    FiberDisplayOptions displayedOptions;
    struct Hit {
        QPointF a, b;
        int64_t fiber;
    };
    std::vector<Hit> hits;
    struct ControlHit {
        QPointF position;
        uint64_t fiber = 0;
        int index = 0;
    };
    std::vector<ControlHit> controlHits;
};

FiberCollectionController::FiberCollectionController(CState* state, ViewerManager* manager, QObject* parent, LineAnnotationController* annotations)
    : ViewerOverlayControllerBase("fiber-collection", parent), state_(state), manager_(manager), annotations_(annotations)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    options_.minLengthMm = settings.value("fiberCollections/minLengthMm", 0.0).toDouble();
    options_.maxDisplayed = settings.value("fiberCollections/maxDisplayed", 5000).toInt();
    options_.lineWidth = settings.value("fiberCollections/lineWidth", 3.0).toDouble();
    options_.horizontal = settings.value("fiberCollections/showHorizontal", true).toBool();
    options_.vertical = settings.value("fiberCollections/showVertical", true).toBool();
    options_.rainbow = settings.value("fiberCollections/rainbowColors", false).toBool();
    options_.showFibers = settings.value("fiberCollections/showFibers", true).toBool();
    options_.showLinked = settings.value("fiberCollections/showLinked", false).toBool();
    options_.sliceDistance = settings.value("fiberCollections/sliceDistance", 3.0).toDouble();
    connect(state_, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        // Sync/lease status is not a data change. Pulled edits arrive through
        // Committed events; status-only events must preserve warm views/selection.
        if (event.kind == vc::objects::StoreSessionEvent::Kind::Hub) return;
        if (event.kind == vc::objects::StoreSessionEvent::Kind::Committed &&
            event.attachment != state_->objectStoreAttachment()) return;
        if (!vc3d::store::touches(event, {"fiber", "fiber_branch"})) return;
        projectChanged(event.kind == vc::objects::StoreSessionEvent::Kind::Committed);
    });
    connect(state_, &CState::volumeChanged, this, [this]() { projectChanged(); });
    connect(state_, &CState::volumeClosing, this, [this]() { clear(); });
    bindToViewerManager(manager);
}
FiberCollectionController::~FiberCollectionController()
{
    for (auto& [v, s] : views_)
        if (s->region.cancelled)
            s->region.cancelled->store(true);
}
void FiberCollectionController::setOptions(const FiberDisplayOptions& options)
{
    if (options == options_) return;
    const auto previous = std::exchange(options_, options);
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("fiberCollections/minLengthMm", options_.minLengthMm);
    settings.setValue("fiberCollections/maxDisplayed", options_.maxDisplayed);
    settings.setValue("fiberCollections/lineWidth", options_.lineWidth);
    settings.setValue("fiberCollections/showHorizontal", options_.horizontal);
    settings.setValue("fiberCollections/showVertical", options_.vertical);
    settings.setValue("fiberCollections/rainbowColors", options_.rainbow);
    settings.setValue("fiberCollections/showFibers", options_.showFibers);
    settings.setValue("fiberCollections/showLinked", options_.showLinked);
    settings.setValue("fiberCollections/sliceDistance", options_.sliceDistance);
    const bool filters = previous.minLengthMm != options_.minLengthMm ||
                         previous.horizontal != options_.horizontal || previous.vertical != options_.vertical;
    if (filters || previous.maxDisplayed != options_.maxDisplayed || previous.sliceDistance != options_.sliceDistance ||
        previous.showLinked != options_.showLinked || previous.showFibers != options_.showFibers)
        invalidate();
    else
        refreshAll();
    if (filters)
        listPage(true);
}
void FiberCollectionController::setLineAnnotationActive(bool active)
{
    lineAnnotationActive_ = active;
}
void FiberCollectionController::publishStatus(const QString& text)
{
    emit statusChanged(state_->objectStoreAttachment().has_value(), text);
}
void FiberCollectionController::publishSelection(Selection selection)
{
    selection_ = std::move(selection);
    emit selectionChanged(selection_);
}
void FiberCollectionController::clear()
{
    ++revision_;
    ++selectionRevision_;
    ++navigationRevision_;
    ++annotationRevision_;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    press_.reset();
    click_.reset();
    enabled_ = false;
    selected_ = 0;
    selectedPin_.reset();
    selectedToVolume_.reset();
    pendingAnnotation_ = 0;
    catalogCursor_.reset();
    for (auto& [v, s] : views_) {
        s->region.reset();
        s->blocks.clear();
        s->controls.clear();
        s->rings.clear();
        s->linkGroups.clear();
        s->chains.clear();
        s->chainBox.reset();
        s->hits.clear();
        s->controlHits.clear();
        s->displayed.reset();
        s->dirty = true;
    }
    catalogPending_ = false;
    pendingIndex_.reset();
    emit catalogChanged({}, false);
    emit annotationStatusChanged({});
    publishSelection({});
    publishStatus(tr("No object store attached."));
    refreshAll();
}
void FiberCollectionController::projectChanged(bool preserveSelection)
{
    if (!preserveSelection) clear();
    else ++revision_;
    const auto connection = state_->objectStoreSession();
    const auto attachment = state_->objectStoreAttachment();
    const auto volume = state_->currentVolume();
    if (!attachment || !annotations_) return;
    catalogCursor_.reset();
    if (preserveSelection) {
        // A commit revises the views in place: each keeps drawing what it has
        // until the replacement lands, so an edit never blanks the overlay.
        refreshRegions();
    } else {
        enabled_ = false;
        invalidate();
        publishStatus(tr("Reading fiber catalog metadata…"));
        nativeToViewer_ = 1;
        nativeVoxelMm_ = volume ? volume->voxelSize()/1000. : 1.;
    }
    const auto rev = revision_;
    background<int64_t>(this,
        [connection, attachment=*attachment] {
            int64_t count = 0;
            connection->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                count = db.objectCount(txn, "fiber");
            });
            return count;
        },
        [this, rev, preserveSelection](Outcome<int64_t> result) {
            if (rev != revision_) return;
            if (!result.error.isEmpty()) { publishStatus(result.error); return; }
            totalFibers_ = result.value;
            const bool wasEnabled = std::exchange(enabled_, true);
            listPage(true);
            if (wasEnabled) publishCount();
            else invalidate();
            if (preserveSelection && selected_ > 0) selectFiber(selected_, false, false);
        });
}
void FiberCollectionController::publishCount()
{
    publishStatus(options_.maxDisplayed
        ? tr("%1 fibers in store · up to %2 visible per view").arg(totalFibers_).arg(options_.maxDisplayed)
        : tr("%1 fibers in store · all visible fibers").arg(totalFibers_));
}
void FiberCollectionController::invalidate()
{
    if (enabled_)
        publishCount();
    for (auto& [v, s] : views_) {
        s->region.reset();
        s->displayed.reset();
        s->blocks.clear();
        s->controls.clear();
        s->rings.clear();
        s->linkGroups.clear();
        s->chains.clear();
        s->chainBox.reset();
        s->hits.clear();
        s->controlHits.clear();
        s->dirty = true;
    }
    clearPointChainProjectionCache();
    refreshAll();
}
void FiberCollectionController::refreshRegions()
{
    for (auto& [v, s] : views_) {
        s->region.refresh();
        s->chainBox.reset();
        s->dirty = true;
    }
    refreshAll();
}
std::optional<std::string> FiberCollectionController::familyFilter() const
{
    if (options_.horizontal && !options_.vertical) return std::string{"H"};
    if (options_.vertical && !options_.horizontal) return std::string{"V"};
    return std::nullopt;
}
bool FiberCollectionController::noFamilies() const
{
    return !options_.horizontal && !options_.vertical;
}
void FiberCollectionController::firstPage()
{
    listPage(true);
}
void FiberCollectionController::nextPage()
{
    listPage(false);
}
void FiberCollectionController::listPage(bool reset)
{
    if (!enabled_) return;
    if (reset) catalogCursor_.reset();
    if (catalogBusy_) { catalogPending_ = true; catalogPendingReset_ = catalogPendingReset_ || reset; return; }
    const auto rev = revision_;
    const auto volume = state_->currentVolume();
    // Longest first in the active volume's voxels, then fibers not bound to it.
    vc::objects::ObjectSelection selection;
    selection.kind = "fiber";
    selection.orderBy = "length_desc";
    selection.limit = kCatalogPageSize;
    selection.family = familyFilter();
    if (options_.minLengthMm > 0) selection.minLength = options_.minLengthMm / nativeVoxelMm_;
    if (volume) {
        selection.volume = volume->id();
        selection.includeUnbound = true;
    }
    if (catalogCursor_) {
        selection.afterUuid = catalogCursor_->uuid;
        selection.afterLength = catalogCursor_->length;
    }
    const auto filters = std::tuple{options_.minLengthMm, options_.horizontal, options_.vertical};
    catalogBusy_ = true;
    background<std::vector<CatalogRow>>(this,
        [connection=state_->objectStoreSession(), attachment=*state_->objectStoreAttachment(),
         selection, none=noFamilies()]() {
            std::vector<CatalogRow> rows;
            if (none) return rows;
            connection->read(attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
                std::vector<std::string> uuids;
                for (auto& record : db.selectObjects(txn, selection, std::vector<std::string>{"name", "tags", "fiber", "meta"})) {
                    CatalogRow row;
                    if (selection.volume)
                        row.toVolume = vc::objects::geometryScaleToVolume(Json::parse(record.meta), *selection.volume);
                    uuids.push_back(record.uuid);
                    row.record = std::move(record);
                    rows.push_back(std::move(row));
                }
                if (rows.empty()) return;
                std::map<std::string, CatalogRow*> byUuid;
                for (auto& row : rows) byUuid[row.record.uuid] = &row;
                vc::objects::EdgeSelection edges;
                edges.kind = "fiber_branch";
                edges.endpointUuids = uuids;
                for (const auto& edge : db.selectEdges(txn, edges, std::vector<std::string>{"approval_state"})) {
                    for (const auto& endpoint : {edge.aUuid, edge.bUuid}) {
                        const auto found = byUuid.find(endpoint);
                        if (found == byUuid.end()) continue;
                        ++found->second->links;
                        if (edge.approvalState != "approved") ++found->second->pendingLinks;
                    }
                }
                for (auto& row : rows)
                    if (row.links > 0)
                        row.componentSize = int(db.componentOf(txn, row.record.uuid, {"fiber_branch"}, &cancelled).size());
            });
            return rows;
        },
        [this, rev, filters, volume=bool(volume)](Outcome<std::vector<CatalogRow>> result) {
            catalogBusy_ = false;
            if (catalogPending_) {
                catalogPending_ = false;
                listPage(std::exchange(catalogPendingReset_, false));
                return;
            }
            if (rev != revision_ || filters != std::tuple{options_.minLengthMm, options_.horizontal, options_.vertical}) return;
            if (!result.error.isEmpty()) { publishStatus(result.error); return; }
            auto& rows = result.value;
            if (!rows.empty()) {
                // A bound row resumes by volume length; an unbound one by UUID alone.
                const auto& last = rows.back();
                const double stored = last.record.fiber ? last.record.fiber->length : 0.0;
                catalogCursor_ = CatalogCursor{last.record.uuid,
                    !volume ? std::optional(stored) : last.toVolume ? std::optional(stored * *last.toVolume) : std::nullopt};
            }
            const bool more = rows.size() == size_t(kCatalogPageSize);
            emit catalogChanged(std::move(rows), more);
        });
}
void FiberCollectionController::activateFiber(int64_t id)
{
    selectFiber(id, !lineAnnotationActive_, false);
}
void FiberCollectionController::selectFiber(int64_t id, bool focusView, bool openAnnotation)
{
    const auto uuid = annotations_ ? annotations_->fiberUuidForId(uint64_t(id)) : std::string{};
    if (!enabled_ || id <= 0 || uuid.empty())
        return;
    const std::optional<int> previousIndex = selected_ == id && !focusView ? std::optional(selection_.index) : std::nullopt;
    selected_ = id;
    ++navigationRevision_;
    pendingIndex_.reset();
    ++annotationRevision_;
    pendingAnnotation_ = 0;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    emit annotationStatusChanged(openAnnotation ? tr("Loading fiber #%1…").arg(id) : QString());
    const auto token = ++selectionRevision_;
    const auto rev = revision_;
    publishSelection({id, tr("Loading fiber #%1…").arg(id), {}, 0, 0, false});
    const auto volume = state_->currentVolume();
    const auto volumeId = volume ? volume->id() : std::string{};
    background<SelectedFiber>(
        this,
        [connection=state_->objectStoreSession(),attachment=*state_->objectStoreAttachment(),uuid,volumeId]() {
            SelectedFiber selected;
            connection->read(attachment,[&](const auto& db,const auto& txn,const auto&) {
                const auto revision=db.currentObjectRevision(txn,uuid);
                if (!revision || revision->record.retired || !revision->record.fiber)
                    throw std::runtime_error("Fiber is no longer available");
                auto& row=selected.row;
                selected.pin={attachment,{attachment.storeUuid,uuid},revision->uuid,revision->geometryRevision};
                row.record=revision->record;
                row.toVolume=volumeId.empty() ? std::nullopt : vc::objects::geometryScaleToVolume(Json::parse(row.record.meta),volumeId);
                const auto& fiber=*row.record.fiber;
                selected.pointCount=fiber.linePointCount ? fiber.linePointCount : fiber.controlPointCount;
            });
            return selected;
        },
        [this, id, token, rev, focusView, openAnnotation, previousIndex](const Outcome<SelectedFiber>& result) {
            if (rev != revision_ || token != selectionRevision_) return;
            if (!result.error.isEmpty()) {
                publishSelection({id, result.error, {}, 0, 0, false});
                if (openAnnotation) emit annotationStatusChanged(result.error);
                return;
            }
            const auto& row = result.value.row;
            // Region queries rank the selected fiber first; re-selecting the
            // same fiber (after a commit) leaves them as they are.
            const bool reselected = selectedPin_ && selectedPin_->object == result.value.pin.object;
            selectedPin_ = result.value.pin;
            selectedToVolume_ = row.toVolume;
            emit fiberRevealed(row);
            const auto pointCount = result.value.pointCount;
            const double length = row.toVolume ? row.record.fiber->length * *row.toVolume : -1.0;
            Selection selection;
            selection.id = id;
            selection.toolTip = QString::fromStdString(row.record.name);
            selection.text = length < 0 ? tr("#%1 · %2 points · coordinate binding required").arg(id).arg(pointCount)
                : tr("#%1 · %2 points · %3 mm\nFollow the complete stored fiber with the slider.").arg(id).arg(pointCount).arg(length * nativeVoxelMm_, 0, 'f', 2);
            selection.pointCount = pointCount;
            const int last = int(std::min<int64_t>(pointCount - 1, std::numeric_limits<int>::max()));
            selection.index = previousIndex ? std::clamp(*previousIndex, 0, std::max(last, 0)) : std::max(last, 0) / 2;
            selection.openable = annotations_ != nullptr;
            publishSelection(selection);
            if (!reselected) refreshRegions();
            if (focusView)
                navigate(selection.index);
            if (openAnnotation)
                showInLineAnnotation(id);
        });
}
void FiberCollectionController::navigate(int index)
{
    if (!selected_ || !enabled_ || !selectedPin_) return;
    selection_.index = index;
    if (!selectedToVolume_) selectedToVolume_ = annotations_->fiberCoordinateScale(uint64_t(selected_), true);
    if (!selectedToVolume_) return;
    ++navigationRevision_;
    pendingIndex_ = index;
    if (navigationBusy_)
        return;
    navigationBusy_ = true;
    pendingIndex_.reset();
    const auto rev = revision_;
    const auto token = navigationRevision_;
    background<Point>(
        this,
        [connection=state_->objectStoreSession(),pin=*selectedPin_,index]() {
            Point point{};
            connection->read(pin.attachment,[&](const auto& db,const auto& txn,const auto&) {
                if(!pin.geometryRevision) throw std::runtime_error("Fiber has no geometry");
                const auto geometry=db.geometryRevision(txn,*pin.geometryRevision);
                const auto lines=std::find_if(geometry->arrays.begin(),geometry->arrays.end(),[](const auto& a){return a.name=="line_points";});
                const auto name=lines!=geometry->arrays.end() && lines->shape[0]>0 ? "line_points" : "control_points";
                point=xyzAt(db.readGeometryArray(txn,*pin.geometryRevision,name,int64_t(index)*24,24),0);
            });return point;
        },
        [this, rev, token, factor=*selectedToVolume_](const Outcome<Point>& result) {
            navigationBusy_ = false;
            if (pendingIndex_) {
                const int next = *pendingIndex_;
                pendingIndex_.reset();
                navigate(next);
                return;
            }
            if (rev != revision_ || token != navigationRevision_)
                return;
            if (!result.error.isEmpty()) {
                auto selection = selection_;
                selection.text = result.error;
                publishSelection(selection);
                return;
            }
            manager_->centerFocusAt(cvpoint(result.value, factor), {0, 0, 1}, "fiber-collection");
        });
}
std::optional<FiberCollectionController::Slice> FiberCollectionController::slice(VolumeViewerBase* viewer) const
{
    if (!dynamic_cast<PlaneSurface*>(viewer->currentSurface()))
        return std::nullopt;
    const auto rect = visibleSceneRect(viewer);
    if (rect.isEmpty())
        return std::nullopt;
    return vc3d::store::viewRegion(point(viewer->sceneToVolume(rect.topLeft()), 1 / nativeToViewer_),
                                   point(viewer->sceneToVolume(rect.topRight()), 1 / nativeToViewer_),
                                   point(viewer->sceneToVolume(rect.bottomLeft()), 1 / nativeToViewer_),
                                   options_.sliceDistance);
}
bool FiberCollectionController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && enabled_ && options_.showFibers;
}
bool FiberCollectionController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto found = views_.find(viewer);
    if (found == views_.end()) return true;
    const auto& v = *found->second;
    const auto current = slice(viewer);
    if (!current) return true;
    return v.dirty || v.displayed != current || v.displayedRect != visibleSceneRect(viewer) ||
        v.displayedOptions != options_ ||
        v.displayedDpr != viewer->graphicsView()->viewport()->devicePixelRatioF();
}
void FiberCollectionController::request(VolumeViewerBase* viewer, const Slice& s)
{
    auto& ptr = views_[viewer];
    if (!ptr) {
        ptr = std::make_unique<View>();
        viewer->graphicsView()->viewport()->installEventFilter(this);
    }
    auto& v = *ptr;
    const auto started = v.region.start(s);
    if (!started) return;
    const auto query = *started;
    v.dirty = true;
    auto cancel = v.region.cancelled;
    const auto rev = revision_;
    const double min = options_.minLengthMm / nativeVoxelMm_;
    const auto family = familyFilter();
    const auto maximum = options_.maxDisplayed;
    const bool linked = options_.showLinked;
    background<RegionResult>(
        this,
        [connection=state_->objectStoreSession(),attachment=*state_->objectStoreAttachment(),
         volume=state_->currentVolume() ? state_->currentVolume()->id() : std::string{},
         selectedUuid=selectedPin_ ? selectedPin_->object.objectUuid : std::string{}, s=query,min,cancel,maximum,family,linked,
         none=noFamilies()]() {
            RegionResult result;
            if (none || volume.empty()) return result;
            connection->read(attachment,[&](const auto& db,const auto& txn,const auto& cancelled) {
                const auto slice = vc3d::store::regionSlice(s);
                vc::objects::RegionOptions options;
                options.kind="fiber"; options.volume=volume; options.maximumObjects=maximum; options.minLength=min;
                if (!selectedUuid.empty()) options.selected=selectedUuid;
                options.family=family;
                auto region=db.geometryRegion(txn,slice,options,[&](const auto&) { return cancel->load() || cancelled.load(); });
                result.page.complete = region.complete();
                result.page.decodedBytes = region.stats.outputBytes;
                // Runs arrive in volume voxels, merged in UUID order with the selected fiber first.
                std::map<std::string, std::optional<std::string>> drawn;
                for (auto& run : region.runs) {
                    auto& sampledRevision = drawn[run.objectUuid];
                    // Control points are annotation markers only on fibers drawn from a sampled line.
                    if (run.arrayName == "line_points") sampledRevision = run.revisionUuid;
                    result.blockUuids.push_back(run.objectUuid);
                    result.page.blocks.push_back({0,0,run.firstSegment,run.family,std::move(run.points)});
                }
                if (drawn.empty()) return;
                const auto factors = volumeFactors(db, txn, drawn, volume);
                std::vector<std::string> revisions, owners;
                for (const auto& [uuid, sampledRevision] : drawn)
                    if (sampledRevision && factors.contains(uuid)) { revisions.push_back(*sampledRevision); owners.push_back(uuid); }
                if (!revisions.empty()) {
                    const auto controls = db.readGeometryArrays(txn, revisions, "control_points");
                    if (controls.dtype != "<f8") throw std::runtime_error("fiber control points must be float64");
                    for (size_t i = 0; i < owners.size(); ++i)
                        for (uint64_t row = controls.offsets[i]; row < controls.offsets[i + 1]; ++row) {
                            auto position = xyzAt(controls.data, row);
                            for (auto& coordinate : position) coordinate *= factors.at(owners[i]);
                            result.controls.push_back({owners[i], int(row - controls.offsets[i]), position});
                        }
                }
                if (!linked) return;
                vc::objects::EdgeSelection edges;
                edges.kind = "fiber_branch";
                edges.endpointUuids.emplace();
                for (const auto& [uuid, fiber] : drawn) edges.endpointUuids->push_back(uuid);
                std::set<std::string> linkedUuids;
                const auto links = db.selectEdges(txn, edges);
                const auto sameFamily = sameFamilyTest(db, txn, links);
                // One mark per linked control point, over every link on it.
                std::map<std::pair<std::string, Point>, LinkMark> marks;
                for (const auto& edge : links) {
                    const bool sameHv = sameFamily(edge);
                    for (const auto& [endpoint, anchor] : {std::pair{edge.aUuid, edge.aAnchor}, std::pair{edge.bUuid, edge.bAnchor}}) {
                        const auto factor = factors.find(endpoint);
                        if (factor == factors.end()) continue;
                        linkedUuids.insert(endpoint);
                        if (!anchor.position) continue;
                        Point position{(*anchor.position)[0], (*anchor.position)[1], (*anchor.position)[2]};
                        for (auto& coordinate : position) coordinate *= factor->second;
                        marks[{endpoint, position}].add(edge, sameHv);
                    }
                }
                for (const auto& [key, mark] : marks) result.rings.push_back({key.second, mark});
                std::set<std::string> drawnUuids;
                for (const auto& [uuid, fiber] : drawn) drawnUuids.insert(uuid);
                result.linkGroups = linkGroupsOf(db, txn, linkedUuids, drawnUuids, cancelled);
            });
            return result;
        },
        [this, viewer, rev, cancel, s, query](Outcome<RegionResult> result) {
            auto found = views_.find(viewer);
            if (found == views_.end()) return;
            auto& v = *found->second;
            using Finished = vc3d::store::RegionRequests::Finished;
            const auto finished = v.region.finish(cancel, rev == revision_, s, query, result.value.page.complete, result.error);
            if (finished == Finished::Stale) return;
            v.dirty = true;
            if (finished == Finished::Install && result.error.isEmpty()) {
                auto& region = result.value;
                for (size_t i = 0; i < region.page.blocks.size(); ++i)
                    region.page.blocks[i].fiberId = int64_t(annotations_->fiberIdForUuid(region.blockUuids[i]));
                v.controls.clear();
                for (const auto& control : region.controls)
                    v.controls.push_back({annotations_->fiberIdForUuid(control.uuid), control.index, control.position});
                v.linkGroups.clear();
                for (const auto& [uuid, group] : region.linkGroups)
                    v.linkGroups[annotations_->fiberIdForUuid(uuid)] = annotations_->fiberIdForUuid(group);
                v.rings = std::move(region.rings);
                v.blocks = std::move(region.page.blocks);
            }
            refreshViewer(viewer);
        });
}
// A linked control point: a circle, or a triangle for an adjacent link.
void FiberCollectionController::addLinkMark(OverlayBuilder& builder, const QPointF& center,
                                            const OverlayStyle& style, bool adjacent)
{
    constexpr qreal radius = 6.25;
    if (!adjacent) {
        builder.addPoint(center, radius, style);
        return;
    }
    QPainterPath triangle;
    triangle.moveTo(center.x(), center.y() - radius);
    triangle.lineTo(center.x() + radius * 0.8660254, center.y() + radius * 0.5);
    triangle.lineTo(center.x() - radius * 0.8660254, center.y() + radius * 0.5);
    triangle.closeSubpath();
    builder.addPainterPath(triangle, style);
}
void FiberCollectionController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto current = slice(viewer);
    if (!current) {
        // The base clears this layer when no planar projection is supported.
        // Returning to the same former plane must rebuild it, not reuse a stamp.
        if (auto found = views_.find(viewer); found != views_.end()) {
            found->second->displayed.reset();
            found->second->dirty = true;
            found->second->hits.clear();
            found->second->controlHits.clear();
        }
        collectChainPrimitives(viewer, builder);
        return;
    }
    request(viewer, *current);
    auto& v = *views_.at(viewer);
    v.dirty = false;
    v.displayed = *current;
    v.displayedRect = visibleSceneRect(viewer);
    v.displayedDpr = viewer->graphicsView()->viewport()->devicePixelRatioF();
    v.displayedOptions = options_;
    v.hits.clear();
    v.controlHits.clear();
    OverlayStyle info;
    info.penColor = QColor(255, 210, 100);
    info.z = 60;
    const auto rect = visibleSceneRect(viewer);
    if (!v.region.error.isEmpty()) {
        builder.addText(rect.topLeft() + QPointF(8, 20), v.region.error, QFont{}, info);
        if (!v.region.loaded)
            return;
    }
    if (!v.region.loaded) {
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Loading fiber region…"), QFont{}, info);
        return;
    }
    // Reclip cached geometry to the current slice while replacement data is
    // fetched. Do not replace the whole overlay with a blank/loading frame.
    v.displayed = *current;
    using Runs = std::vector<QPolygonF>;
    Runs horizontal, vertical, other, selected;
    std::array<Runs, 32> rainbowPaths;
    std::map<uint64_t, Runs> linkedPaths;
    const bool useRainbow = options_.rainbow;
    const auto linkGroup = [&](int64_t fiber) -> std::optional<uint64_t> {
        if (!options_.showLinked) return std::nullopt;
        const auto found = v.linkGroups.find(uint64_t(fiber));
        return found == v.linkGroups.end() ? std::nullopt : std::optional(found->second);
    };
    std::unordered_map<int64_t, QColor> colors;
    std::unordered_set<int64_t> visibleIds;
    for (const auto& b : v.blocks) {
        // Hash the logical fiber ID, not its storage block or catalog row, so
        // every block of a fiber keeps its color across region/filter changes.
        const auto colorIndex = ((uint64_t(b.fiberId) * 0x9e3779b97f4a7c15ULL) >> 32) % rainbowPaths.size();
        const auto group = linkGroup(b.fiberId);
        auto& path = b.fiberId == selected_ ? selected
                     : group                ? linkedPaths[*group]
                     : useRainbow           ? rainbowPaths[colorIndex]
                     : b.family == "H"      ? horizontal
                     : b.family == "V"      ? vertical
                                            : other;
        colors.emplace(b.fiberId, b.fiberId == selected_ ? QColor("#ffff66")
                                  : group                ? groupColor(*group)
                                  : useRainbow           ? QColor::fromHsvF(double(colorIndex) / rainbowPaths.size(), 0.7, 1.0)
                                  : b.family == "H"      ? QColor("#40a5ff")
                                  : b.family == "V"      ? QColor("#ffad43")
                                                         : QColor("#bbbbbb"));
        if (b.points.empty()) continue;
        QPolygonF run;
        auto flush = [&]() {
            if (!run.isEmpty()) { path.push_back(std::move(run)); run = {}; }
        };
        auto previous = current->project(b.points.front());
        for (size_t i = 1; i < b.points.size(); ++i) {
            const auto a = previous, z = current->project(b.points[i]);
            previous = z;
            double lo, hi;
            if (!vc::fibers::clipSegment(a, z, current->local, lo, hi)) {
                flush();
                continue;
            }
            Point p, q;
            for (int j = 0; j < 3; ++j) {
                const double d = b.points[i][j] - b.points[i - 1][j];
                p[j] = b.points[i - 1][j] + lo * d;
                q[j] = b.points[i - 1][j] + hi * d;
            }
            const auto pa = viewer->volumeToScene(cvpoint(p, nativeToViewer_)), pb = viewer->volumeToScene(cvpoint(q, nativeToViewer_));
            // Stroke contiguous segments together. Independent round caps on
            // every tiny segment make Qt's compound-path rasterizer expensive
            // in dense views. Keep every vertex and every actual slice break.
            if (!run.isEmpty() && run.back() != pa) flush();
            if (run.isEmpty()) run.push_back(pa);
            run.push_back(pb);
            v.hits.push_back({pa, pb, b.fiberId});
            visibleIds.insert(b.fiberId);
        }
        flush();
    }
    // Rasterize this layer once per actual geometry/view change. CT tile
    // arrivals can then repaint it with one blit instead of stroking thousands
    // of segments again.
    const qreal dpr = viewer->graphicsView()->viewport()->devicePixelRatioF();
    const QSize pixels(qCeil(rect.width() * dpr), qCeil(rect.height() * dpr));
    QImage raster;
    if (pixels.width() > 0 && pixels.height() > 0 &&
        qint64(pixels.width()) * pixels.height() * 4 <= 32 * 1024 * 1024) {
        raster = QImage(pixels, QImage::Format_ARGB32_Premultiplied);
        raster.setDevicePixelRatio(dpr);
        raster.fill(Qt::transparent);
    }
    QPainter painter;
    if (!raster.isNull()) {
        painter.begin(&raster);
        painter.setRenderHints(viewer->graphicsView()->renderHints());
        painter.translate(-rect.topLeft());
        painter.setBrush(Qt::NoBrush);
    }
    auto draw = [&](const Runs& runs, QColor color, qreal width) {
        if (painter.isActive()) {
            painter.setPen(QPen(color, width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            for (const auto& run : runs) painter.drawPolyline(run);
            return;
        }
        OverlayStyle style;
        style.penColor = color;
        style.penWidth = width;
        style.z = 45;
        QPainterPath path;
        for (const auto& run : runs) {
            path.moveTo(run.front());
            for (qsizetype i = 1; i < run.size(); ++i) path.lineTo(run[i]);
        }
        builder.addPainterPath(path, style);
    };
    const qreal width = options_.lineWidth;
    if (useRainbow) {
        for (size_t i = 0; i < rainbowPaths.size(); ++i)
            draw(rainbowPaths[i], QColor::fromHsvF(double(i) / rainbowPaths.size(), 0.7, 1.0), width);
    } else {
        draw(horizontal, QColor("#40a5ff"), width);
        draw(vertical, QColor("#ffad43"), width);
        draw(other, QColor("#bbbbbb"), width);
    }
    for (const auto& [group, runs] : linkedPaths)
        draw(runs, groupColor(group), width);
    draw(selected, QColor("#ffff66"), width + 2);
    if (painter.isActive()) {
        painter.end();
        builder.addImage(raster, QPixmap::fromImage(raster), rect.topLeft(), 1, 1, 1, 45);
    }
    // Control points inside the slab, then the linked ones' rings on top.
    for (const auto& control : v.controls) {
        // Only fibers drawn in this view, not ones crossing just the margin.
        if (!visibleIds.contains(int64_t(control.fiber)) ||
            !inside(current->project(control.position), current->local)) continue;
        const auto scene = viewer->volumeToScene(cvpoint(control.position, nativeToViewer_));
        const auto color = colors.find(int64_t(control.fiber));
        OverlayStyle style;
        style.penColor = color == colors.end() ? QColor("#bbbbbb") : color->second;
        style.brushColor = style.penColor;
        style.penWidth = 1.0;
        style.z = 94;
        builder.addPoint(scene, 3.0, style);
        v.controlHits.push_back({scene, control.fiber, control.index});
    }
    if (options_.showLinked) {
        for (const auto& ring : v.rings) {
            if (!inside(current->project(ring.position), current->local)) continue;
            addLinkMark(builder, viewer->volumeToScene(cvpoint(ring.position, nativeToViewer_)), linkMarkStyle(ring.mark, 1.0f), ring.mark.adjacent);
        }
    }
    const QString count = options_.maxDisplayed
        ? tr("%1 visible fibers · max %2").arg(visibleIds.size()).arg(options_.maxDisplayed)
        : tr("%1 visible fibers").arg(visibleIds.size());
    builder.addText(rect.topLeft() + QPointF(8, 40), count, QFont{}, info);
    if (v.region.busy && !(v.region.complete && v.region.coverage && v.region.coverage->contains(*current)))
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Updating fiber region…"), QFont{}, info);
    else if (!v.region.complete)
        builder.addText(rect.topLeft() + QPointF(8, 20), tr("Partial fiber view · zoom in or increase minimum length"), QFont{}, info);
}
void FiberCollectionController::requestChains(VolumeViewerBase* viewer, const std::array<double, 6>& box)
{
    auto& ptr = views_[viewer];
    if (!ptr) {
        ptr = std::make_unique<View>();
        viewer->graphicsView()->viewport()->installEventFilter(this);
    }
    auto& v = *ptr;
    if (v.chainBox == box)
        return;
    // Chains share the viewer's one query in flight with its regions.
    auto& region = v.region;
    if (region.busy) {
        if (region.cancelled) region.cancelled->store(true);
        return;
    }
    region.busy = true;
    v.chainBox = box;
    region.error.clear();
    auto cancel = std::make_shared<std::atomic_bool>(false);
    region.cancelled = cancel;
    const auto rev = revision_;
    background<ChainResult>(
        this,
        [connection=state_->objectStoreSession(),attachment=*state_->objectStoreAttachment(),
         volume=state_->currentVolume() ? state_->currentVolume()->id() : std::string{}, box, cancel,
         min=options_.minLengthMm / nativeVoxelMm_, family=familyFilter(), maximum=options_.maxDisplayed,
         linked=options_.showLinked, none=noFamilies()]() {
            ChainResult result;
            if (none || volume.empty()) return result;
            connection->read(attachment,[&](const auto& db,const auto& txn,const auto& cancelled) {
                // Fibers drawn from a sampled line carry annotation controls.
                vc::objects::ObjectSelection selection;
                selection.kind = "fiber";
                selection.volume = volume;
                selection.bbox = box;
                if (min > 0) selection.minLength = min;
                selection.family = family;
                if (maximum > 0) selection.limit = maximum;
                std::vector<std::string> uuids, revisions;
                std::vector<double> factors;
                for (const auto& object : db.selectObjects(txn, selection, std::vector<std::string>{"fiber", "meta"})) {
                    if (cancel->load() || cancelled.load()) return;
                    if (!object.fiber || object.fiber->linePointCount == 0) continue;
                    const auto head = db.currentGeometry(txn, object.uuid);
                    if (!head) continue;
                    uuids.push_back(object.uuid);
                    revisions.push_back(head->uuid);
                    factors.push_back(*vc::objects::geometryScaleToVolume(Json::parse(object.meta), volume));
                }
                if (revisions.empty()) return;
                const auto controls = db.readGeometryArrays(txn, revisions, "control_points");
                const auto ids = db.readGeometryArrays(txn, revisions, "control_point_ids");
                if (controls.dtype != "<f8" || ids.dtype != "<i8") throw std::runtime_error("unexpected fiber control point arrays");
                std::map<std::string, std::pair<size_t, std::unordered_map<int64_t, size_t>>> controlIndex;
                for (size_t i = 0; i < uuids.size(); ++i) {
                    ChainResult::Chain chain;
                    chain.uuid = uuids[i];
                    auto& index = controlIndex[uuids[i]];
                    index.first = result.chains.size();
                    for (uint64_t row = controls.offsets[i]; row < controls.offsets[i + 1]; ++row)
                        chain.points.push_back(cvpoint(xyzAt(controls.data, row), factors[i]));
                    for (uint64_t row = ids.offsets[i]; row < ids.offsets[i + 1]; ++row) {
                        uint64_t bits = 0;
                        for (size_t byte = 0; byte < 8; ++byte) bits |= uint64_t(ids.data[row * 8 + byte]) << (8 * byte);
                        index.second[std::bit_cast<int64_t>(bits)] = size_t(row - ids.offsets[i]);
                    }
                    chain.linkMarks.assign(chain.points.size(), LinkMark{});
                    result.chains.push_back(std::move(chain));
                }
                if (!linked) return;
                vc::objects::EdgeSelection edges;
                edges.kind = "fiber_branch";
                edges.endpointUuids = uuids;
                std::set<std::string> linkedUuids;
                const auto links = db.selectEdges(txn, edges);
                const auto sameFamily = sameFamilyTest(db, txn, links);
                for (const auto& edge : links) {
                    const bool sameHv = sameFamily(edge);
                    for (const auto& [endpoint, anchor] : {std::pair{edge.aUuid, edge.aAnchor}, std::pair{edge.bUuid, edge.bAnchor}}) {
                        const auto chain = controlIndex.find(endpoint);
                        if (chain == controlIndex.end()) continue;
                        linkedUuids.insert(endpoint);
                        auto& marks = result.chains[chain->second.first].linkMarks;
                        for (const auto& reference : anchor.points) {
                            if (reference.array != "control_point_ids" || reference.index.empty()) continue;
                            const auto control = chain->second.second.find(reference.index.front());
                            if (control == chain->second.second.end()) continue;
                            marks[control->second].add(edge, sameHv);
                        }
                    }
                }
                result.linkGroups = linkGroupsOf(db, txn, linkedUuids, std::set<std::string>(uuids.begin(), uuids.end()), cancelled);
            });
            return result;
        },
        [this, viewer, rev, cancel](Outcome<ChainResult> result) {
            auto found = views_.find(viewer);
            if (found == views_.end() || found->second->region.cancelled != cancel)
                return;
            auto& v = *found->second;
            v.region.busy = false;
            if (rev != revision_ || cancel->load()) {
                v.chainBox.reset();
                refreshViewer(viewer);
                return;
            }
            v.region.error = result.error;
            if (result.error.isEmpty()) {
                v.chains.clear();
                for (auto& chain : result.value.chains)
                    v.chains.push_back({annotations_->fiberIdForUuid(chain.uuid), std::move(chain.points), std::move(chain.linkMarks)});
                v.linkGroups.clear();
                for (const auto& [uuid, group] : result.value.linkGroups)
                    v.linkGroups[annotations_->fiberIdForUuid(uuid)] = annotations_->fiberIdForUuid(group);
                clearPointChainProjectionCache();
            }
            refreshViewer(viewer);
        });
}
void FiberCollectionController::collectChainPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!surface)
        return;
    const auto bounds = surface->bbox();
    const double d = options_.sliceDistance;
    requestChains(viewer, {bounds.low[0] - d, bounds.low[1] - d, bounds.low[2] - d,
                           bounds.high[0] + d, bounds.high[1] + d, bounds.high[2] + d});
    auto& v = *views_.at(viewer);
    v.controlHits.clear();
    v.displayedOptions = options_;
    v.dirty = false;
    struct Ring {
        QPointF position;
        LinkMark mark;
        float opacity;
    };
    std::vector<Ring> rings;
    for (const auto& chain : v.chains) {
        const auto group = options_.showLinked ? v.linkGroups.find(chain.fiber) : v.linkGroups.end();
        const QColor color = groupColor(group != v.linkGroups.end() ? group->second : chain.fiber);
        PointChainStyle style;
        style.color = color;
        style.pointBorderColor = color;
        style.lineOpacity = 0.75f;
        style.distanceTolerance = float(options_.sliceDistance);
        FilteredPoints filtered;
        std::vector<float> opacities;
        renderPointChain(viewer, builder, chain.points, style, std::nullopt, &filtered, &opacities);
        for (size_t i = 0; i < filtered.scenePoints.size() && i < filtered.sourceIndices.size(); ++i) {
            const float opacity = i < opacities.size() ? opacities[i] : 1.0f;
            // Boundary projections are line vertices only, without a dot.
            if (opacity <= 0.0f) continue;
            const size_t source = filtered.sourceIndices[i];
            v.controlHits.push_back({filtered.scenePoints[i], chain.fiber, int(source)});
            if (options_.showLinked && source < chain.linkMarks.size() && chain.linkMarks[source].linked)
                rings.push_back({filtered.scenePoints[i], chain.linkMarks[source], opacity});
        }
    }
    // Rings paint after every chain.
    for (const auto& ring : rings)
        addLinkMark(builder, ring.position, linkMarkStyle(ring.mark, ring.opacity), ring.mark.adjacent);
}
std::optional<FiberCollectionController::ControlPointHit> FiberCollectionController::hitTestControlPoint(
    VolumeViewerBase* viewer, const QPointF& scenePoint, qreal maxDistancePx) const
{
    const auto found = views_.find(viewer);
    if (!isOverlayEnabledFor(viewer) || found == views_.end()) return std::nullopt;
    std::optional<ControlPointHit> best;
    qreal bestDistanceSq = maxDistancePx * maxDistancePx;
    for (const auto& hit : found->second->controlHits) {
        const QPointF delta = hit.position - scenePoint;
        const qreal distanceSq = delta.x() * delta.x() + delta.y() * delta.y();
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            best = ControlPointHit{hit.fiber, hit.index};
        }
    }
    return best;
}
bool FiberCollectionController::eventFilter(QObject* obj, QEvent* event)
{
    // Only observe the mouse here, so native tools receive every press, drag
    // and release. Picking waits for handleVolumeClick().
    if (event->type() != QEvent::MouseButtonPress && event->type() != QEvent::MouseButtonRelease)
        return false;
    const auto* mouse = static_cast<QMouseEvent*>(event);
    if (mouse->button() != Qt::LeftButton)
        return false;
    if (event->type() == QEvent::MouseButtonPress) {
        press_ = Click{obj, mouse->position()};
        click_.reset();
    } else {
        const bool clicked = press_ && press_->viewport == obj &&
                             (mouse->position() - press_->position).manhattanLength() <= QGuiApplication::styleHints()->startDragDistance();
        click_ = clicked ? std::optional<Click>(Click{obj, mouse->position()}) : std::nullopt;
        press_.reset();
    }
    return false;
}
bool FiberCollectionController::handleVolumeClick(Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    const auto clicked = std::exchange(click_, std::nullopt);
    if (button != Qt::LeftButton || modifiers != Qt::NoModifier || !clicked || !clicked->viewport || !enabled_ ||
        !options_.showFibers)
        return false;
    for (auto& [viewer, v] : views_) {
        if (viewer->graphicsView()->viewport() != clicked->viewport)
            continue;
        const auto* chunked = qobject_cast<CChunkedVolumeViewer*>(viewer->asQObject());
        if (chunked && chunked->lineAnnotationPlacementPreviewEnabled())
            return false;
        const auto current = slice(viewer);
        if (!current || !v->displayed || *current != *v->displayed)
            return false;
        const auto* view = viewer->graphicsView();
        const auto click = clicked->position;
        double best = 8 * 8;
        int64_t id = 0;
        for (const auto& hit : v->hits) {
            const QPointF a = view->mapFromScene(hit.a), b = view->mapFromScene(hit.b), d = b - a;
            const double len = dot({d.x(), d.y(), 0}, {d.x(), d.y(), 0});
            const double t = len ? std::clamp(((click.x() - a.x()) * d.x() + (click.y() - a.y()) * d.y()) / len, 0.0, 1.0) : 0;
            const auto delta = click - (a + t * d);
            const double dist = delta.x() * delta.x() + delta.y() * delta.y();
            if (dist < best) {
                best = dist;
                id = hit.fiber;
            }
        }
        if (!id)
            return false;
        selectFiber(id, false);
        emit panelRequested();
        return true;
    }
    return false;
}
void FiberCollectionController::showInLineAnnotation(int64_t id)
{
    if (!annotations_ || !enabled_)
        return;
    pendingAnnotation_ = id;
    ++annotationRevision_;
    for (const auto& connection : annotationRenderConnections_)
        disconnect(connection);
    annotationRenderConnections_.clear();
    emit annotationStatusChanged(tr("Loading fiber #%1…").arg(id));
    // Paint the requested ID before the native importer/view construction runs.
    QTimer::singleShot(0, this, &FiberCollectionController::startPendingAnnotation);
}
void FiberCollectionController::startPendingAnnotation()
{
    if(annotationBusy_ || !pendingAnnotation_ || !enabled_) return;
    const auto id=std::exchange(pendingAnnotation_,0);
    annotationBusy_=true;
    const auto finish=qScopeGuard([this] { finishAnnotationRequest(); });
    QString error;
    auto* dialog=annotations_->openCollectionFiber(uint64_t(id),&error);
    if(dialog) watchAnnotationRender(dialog,id,annotationRevision_);
    else emit annotationStatusChanged(error);
}
void FiberCollectionController::finishAnnotationRequest()
{
    annotationBusy_ = false;
    if (pendingAnnotation_)
        QTimer::singleShot(0, this, &FiberCollectionController::startPendingAnnotation);
}
void FiberCollectionController::watchAnnotationRender(LineAnnotationDialog* dialog, int64_t id, uint64_t token)
{
    emit annotationStatusChanged(tr("Loading CT for fiber #%1…").arg(id));
    auto waiting = std::make_shared<std::unordered_set<CChunkedVolumeViewer*>>();
    for (const auto& pane : dialog->panes()) {
        const QString name = QString::fromStdString(pane.surfaceName);
        if (pane.viewer && (name.endsWith("_line_surface") || name.endsWith("_line_side_slice")))
            waiting->insert(pane.viewer.data());
    }
    const auto ready = [this, token, id, waiting](CChunkedVolumeViewer* viewer) {
        if (token != annotationRevision_ || !enabled_)
            return;
        if (!viewer->hasDisplayedRenderFrame() ||
            viewer->displayedSurfaceGeometryEpoch() != viewer->surfaceGeometryEpoch())
            return;
        waiting->erase(viewer);
        if (waiting->empty()) {
            emit annotationStatusChanged(tr("Fiber #%1 displayed.").arg(id));
            for (const auto& connection : annotationRenderConnections_)
                disconnect(connection);
            annotationRenderConnections_.clear();
        }
    };
    const auto strips = *waiting;
    for (auto* viewer : strips) {
        annotationRenderConnections_.push_back(connect(viewer, &CChunkedVolumeViewer::renderFrameCompleted,
            this, [ready, viewer](std::uint64_t, double) { ready(viewer); }));
    }
    for (auto* viewer : strips)
        ready(viewer);
    annotationRenderConnections_.push_back(connect(dialog, &QObject::destroyed, this, [this, token]() {
        if (token == annotationRevision_)
            emit annotationStatusChanged({});
    }));
}
void FiberCollectionController::detachViewer(VolumeViewerBase* viewer)
{
    if (auto it = views_.find(viewer); it != views_.end()) {
        if (it->second->region.cancelled)
            it->second->region.cancelled->store(true);
        views_.erase(it);
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}
