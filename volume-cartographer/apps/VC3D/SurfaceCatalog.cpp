#include "SurfaceCatalog.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include "ObjectStoreRegions.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/objects/CatalogStore.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/Uuid.hpp>
#include <nlohmann/json.hpp>

namespace {
using namespace vc::objects;

struct Listing {
    std::vector<SurfaceCatalog::Row> rows;
    // The surfaces read: those missing from rows left the listing.
    std::optional<std::vector<std::string>> requested;
};

// The rows of `requested` (all when nullopt) in the volume, a surface from the
// first source holding it.
Listing list(ObjectStoreSession& session, const std::vector<AttachmentIdentity>& listed,
             const std::optional<AttachmentIdentity>& user, const std::string& volume,
             const std::optional<std::vector<std::string>>& requested, const std::atomic_bool& cancel)
{
    Listing listing;
    listing.requested = requested;
    if (requested && requested->empty()) return listing;
    ObjectSelection selection;
    selection.kind = "surface";
    selection.volume = volume;
    selection.includeUnbound = true;
    if (requested) selection.uuids = *requested;
    std::map<AttachmentIdentity, std::shared_ptr<const VolumeTransforms>, decltype([](const auto& a, const auto& b) {
        return std::tie(a.storeUuid, a.generation) < std::tie(b.storeUuid, b.generation);
    })> transforms;
    for (auto& [attachment, record] : session.selectObjects(listed, selection)) {
        if (cancel.load()) break;
        auto& mapping = transforms[attachment];
        if (!mapping)
            session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                mapping = std::make_shared<const VolumeTransforms>(db.volumeTransforms(txn));
            });
        SurfaceCatalog::Row row;
        row.mapping = mapping->toVolume(nlohmann::json::parse(record.meta), volume);
        row.record = std::move(record);
        row.source = attachment;
        row.catalog = attachment != user;
        listing.rows.push_back(std::move(row));
    }
    return listing;
}
}

bool SurfaceCatalog::Row::imported() const
{
    return !catalog && catalogReference(record).has_value();
}

SurfaceCatalog::SurfaceCatalog(std::shared_ptr<ObjectStoreSession> session, QObject* parent)
    : QObject(parent), session_(std::move(session))
{
}

SurfaceCatalog::~SurfaceCatalog()
{
    if (cancelled_) cancelled_->store(true);
}

void SurfaceCatalog::setSource(const std::optional<AttachmentIdentity>& attachment, std::shared_ptr<Volume> volume)
{
    if (attachment_ == attachment && volume_ == volume) return;
    attachment_ = attachment;
    volume_ = std::move(volume);
    restart();
}

void SurfaceCatalog::setCatalogSources(std::vector<AttachmentIdentity> sources)
{
    if (catalogSources_ == sources) return;
    catalogSources_ = std::move(sources);
    restart();
}

void SurfaceCatalog::readNow(const std::vector<std::string>& uuids)
{
    if (!volume_) return;
    const std::atomic_bool notCancelled{false};
    auto listing = list(*session_, sources(), attachment_, volume_->id(), uuids, notCancelled);
    for (const auto& uuid : uuids) rows_.erase(uuid);
    for (auto& row : listing.rows) rows_[row.record.uuid] = std::move(row);
    emit rowsChanged();
}

std::vector<AttachmentIdentity> SurfaceCatalog::sources() const
{
    std::vector<AttachmentIdentity> sources;
    if (!volume_) return sources;
    if (attachment_) sources.push_back(*attachment_);
    sources.insert(sources.end(), catalogSources_.begin(), catalogSources_.end());
    return sources;
}

void SurfaceCatalog::restart()
{
    ++generation_;
    if (cancelled_) cancelled_->store(true);
    cancelled_.reset();
    busy_ = false;
    pendingFull_ = false;
    pending_.clear();
    rows_.clear();
    loading_ = false;
    if (!error_.isEmpty()) { error_.clear(); emit errorChanged(error_); }
    emit rowsChanged();
    read(std::nullopt);
}

void SurfaceCatalog::handle(const StoreSessionEvent& event)
{
    const auto listed = sources();
    if (event.kind != StoreSessionEvent::Kind::Committed ||
        std::find(listed.begin(), listed.end(), event.attachment) == listed.end())
        return;
    // A registration can bring any surface into (or out of) the volume.
    if (event.touches({"volume_transform", "volume_transform_choice"})) {
        read(std::nullopt);
        return;
    }
    std::vector<std::string> changed;
    for (const auto& change : event.changes)
        if (change.kind == "surface") changed.push_back(change.recordUuid);
    if (changed.empty()) return;
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    read(changed);
    emit surfacesCommitted(changed);
}

const SurfaceCatalog::Row* SurfaceCatalog::row(const std::string& uuid) const
{
    const auto found = rows_.find(uuid);
    return found == rows_.end() ? nullptr : &found->second;
}

void SurfaceCatalog::read(std::optional<std::vector<std::string>> uuids)
{
    if (!uuids) {
        pendingFull_ = true;
        pending_.clear();
    } else if (!pendingFull_) {
        pending_.insert(uuids->begin(), uuids->end());
    }
    schedule();
}

// One read at a time, so results merge in commit order.
void SurfaceCatalog::schedule()
{
    const auto listed = sources();
    if (busy_ || listed.empty() || (!pendingFull_ && pending_.empty())) return;
    std::optional<std::vector<std::string>> requested;
    if (!std::exchange(pendingFull_, false)) requested = std::vector<std::string>(pending_.begin(), pending_.end());
    else loading_ = true;
    pending_.clear();
    busy_ = true;
    cancelled_ = std::make_shared<std::atomic_bool>(false);
    vc3d::store::background<Listing>(this,
        [session = session_, listed, user = attachment_, volume = volume_->id(), requested, cancel = cancelled_]() {
            return list(*session, listed, user, volume, requested, *cancel);
        },
        [this, generation = generation_, requested](vc3d::store::Outcome<Listing> result) {
            if (generation != generation_) return;
            busy_ = false;
            if (!requested) loading_ = false;
            if (result.error != error_) { error_ = result.error; emit errorChanged(error_); }
            if (result.error.isEmpty()) {
                if (!requested) rows_.clear();
                else for (const auto& uuid : *requested) rows_.erase(uuid);
                for (auto& row : result.value.rows) rows_[row.record.uuid] = std::move(row);
                emit rowsChanged();
            }
            schedule();
        });
}

SurfaceCatalog::Opened SurfaceCatalog::open(const std::string& uuid, const std::optional<std::string>& revision) const
{
    if (!volume_) throw std::runtime_error("Open a volume first");
    if (const auto* listed = row(uuid)) return open(session_, listed->source, volume_->id(), uuid, revision);
    if (!attachment_) throw std::runtime_error("Attach an object store and open a volume first");
    return open(session_, *attachment_, volume_->id(), uuid, revision);
}

SurfaceCatalog::Opened SurfaceCatalog::open(const std::shared_ptr<ObjectStoreSession>& session,
                                            const AttachmentIdentity& attachment, const std::string& volume,
                                            const std::string& uuid, const std::optional<std::string>& at)
{
    Opened opened;
    auto& current = opened.current;
    session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        auto revision = db.currentObjectRevision(txn, uuid);
        if (!revision || revision->record.retired || revision->record.kind != "surface")
            throw std::runtime_error("The surface is no longer in the object store");
        if (at && *at != revision->uuid) {
            revision = db.objectRevision(txn, *at);
            if (!revision || revision->objectUuid != uuid || revision->deleted)
                throw std::runtime_error("The surface has no such revision");
            current = false;
        }
        if (!revision->geometryRevision) throw NoGeometry("The surface has no geometry");
        const GeometrySnapshot snapshot(db, txn);
        const auto view = current ? snapshot.viewTransformedAt(uuid, volume)
                                  : snapshot.surface(uuid, revision->geometryRevision).viewTransformedAt(volume);
        opened.surface=load_quad_from_store(db,txn,uuid,*revision->geometryRevision,true,&view);
        opened.view = view.frame();
        opened.pin = {attachment, {attachment.storeUuid, revision->objectUuid}, revision->uuid, revision->geometryRevision};
    });
    opened.surface->setChannelLoader(store_channel_loader(session, attachment, *opened.pin.geometryRevision));
    // Past revisions are viewed; edits continue from the current one.
    opened.editable = current && session->writable(attachment);
    return opened;
}

std::vector<std::vector<SurfaceProximityHit>> SurfaceCatalog::nearest(
    ObjectStoreSession& session, const std::vector<AttachmentIdentity>& sources, const std::string& volume,
    const std::vector<std::string>& uuids, const std::vector<std::array<double, 3>>& points, double tolerance)
{
    std::vector<std::vector<SurfaceProximityHit>> result(points.size());
    if (uuids.empty() || points.empty()) return result;
    SurfaceProximityQuery query;
    query.volume = volume;
    query.uuids = uuids;
    query.tolerance = tolerance;
    auto found = session.nearestSurfaces(sources, query, points);
    for (size_t i = 0; i < points.size(); ++i)
        for (auto& hit : found[i]) result[i].push_back(std::move(hit.hit));
    return result;
}

SurfaceCatalog::ToolStore SurfaceCatalog::toolStore() const
{
    if (!attachment_ || !volume_) throw std::runtime_error("Attach an object store and open a volume first.");
    if (!session_->writable(*attachment_)) throw std::runtime_error("The object store is read-only.");
    return {session_->path(*attachment_), volume_->id()};
}

SurfaceCatalog::Opened SurfaceCatalog::importCatalogSurface(const Opened& opened)
{
    const auto uuid = opened.surface->id;
    const auto* listed = row(uuid);
    if (!listed || !listed->catalog) throw std::runtime_error("not a catalog surface: " + uuid);
    if (!attachment_ || !session_->writable(*attachment_))
        throw std::runtime_error("Attach a writable object store to edit catalog surfaces");
    session_->importObject(listed->source, *attachment_, uuid);
    readNow({uuid});
    if (const auto* imported = row(uuid); !imported || imported->catalog || !imported->bound())
        throw std::runtime_error("Your object store does not place the surface in this volume");
    auto reopened = open(uuid);
    if (reopened.view.mapping != opened.view.mapping)
        throw std::runtime_error("your object store places the surface differently in this volume; reopen it");
    reopened.surface = opened.surface;
    reopened.surface->setChannelLoader(store_channel_loader(session_, reopened.pin.attachment, *reopened.pin.geometryRevision));
    return reopened;
}

std::string SurfaceCatalog::create(QuadSurface& surface, const std::string& name, const SurfaceProvenance& provenance,
                                   const std::vector<std::string>& tags)
{
    if (!attachment_ || !volume_) throw std::runtime_error("Attach an object store and open a volume first");
    const auto uuid = util::newUuid();
    session_->act(*attachment_, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        create_quad_in_store(db, txn, uuid, surface, name, tags, volumeCoordinates(volume_->id(), 1), provenance);
    });
    surface.id = uuid;
    return uuid;
}

SurfaceCatalog::Crossings SurfaceCatalog::region(ObjectStoreSession& session, const std::vector<AttachmentIdentity>& sources,
                                                 const std::string& volume, const RegionSlice& slice, RegionOptions options,
                                                 const std::function<bool()>& cancelled, bool waitForGeometry)
{
    options.kind = "surface";
    options.volume = volume;
    const auto isCancelled = [&] { return cancelled && cancelled(); };
    for (;;) {
        if (isCancelled()) return {{}, false};
        auto results = session.geometryRegion(sources, slice, options,
                                              [&](const auto&) { return isCancelled(); });
        bool fetched = false;
        for (const auto& [attachment, result] : results) {
            if (!waitForGeometry) break;
            for (const auto& uuid : result.remote) {
                if (isCancelled()) return {{}, false};
                // Region reads queue remote geometry but do not wait for it.
                // Wait outside the read transaction, then read the whole batch
                // again instead of publishing each arriving surface separately.
                session.ensureGeometry(attachment, uuid, isCancelled);
                fetched = true;
            }
        }
        if (fetched) continue;
        Crossings crossings;
        for (auto& [attachment, result] : results) {
            crossings.complete = crossings.complete && result.complete();
            for (auto& run : result.runs) crossings.runs.push_back(std::move(run));
        }
        return crossings;
    }
}

