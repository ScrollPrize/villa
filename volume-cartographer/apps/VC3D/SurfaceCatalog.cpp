#include "SurfaceCatalog.hpp"
#include "ObjectStoreCoordinates.hpp"
#include "ObjectStoreRegions.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/Uuid.hpp>
#include <nlohmann/json.hpp>

namespace {
using namespace vc::objects;

const std::vector<std::string> kRowFields{"name", "tags", "meta", "surface", "created_by", "created_at",
                                          "changed_by", "changed_at", "approval_state"};

struct Listing {
    std::vector<SurfaceCatalog::Row> rows;
    // The surfaces read: those missing from rows left the listing.
    std::optional<std::vector<std::string>> requested;
};

// The rows of `requested` (all when nullopt) in each listed source.
Listing list(ObjectStoreSession& session, const std::vector<SurfaceCatalog::Source>& listed,
             const std::optional<AttachmentIdentity>& user, const std::optional<std::vector<std::string>>& requested,
             const std::atomic_bool& cancel)
{
    Listing listing;
    listing.requested = requested;
    for (const auto& source : listed) {
        const bool catalog = source.attachment != user;
        session.read(source.attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
            ObjectSelection selection;
            selection.kind = "surface";
            selection.volume = source.volume;
            // Catalog stores list only what they hold for this volume.
            selection.includeUnbound = !catalog;
            if (requested) {
                if (requested->empty()) return;
                selection.uuids = *requested;
            }
            const auto transforms = db.volumeTransforms(txn);
            for (auto& record : db.selectObjects(txn, selection, kRowFields)) {
                if (cancel.load() || cancelled.load()) return;
                SurfaceCatalog::Row row;
                row.mapping = transforms.toVolume(nlohmann::json::parse(record.meta), source.volume, source.scale);
                row.record = std::move(record);
                row.source = source;
                row.catalog = catalog;
                listing.rows.push_back(std::move(row));
            }
        });
    }
    return listing;
}
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

void SurfaceCatalog::setCatalogSources(std::vector<Source> sources)
{
    if (catalogSources_ == sources) return;
    catalogSources_ = std::move(sources);
    restart();
}

void SurfaceCatalog::readNow(const std::vector<std::string>& uuids)
{
    const std::atomic_bool notCancelled{false};
    auto listing = list(*session_, sources(), attachment_, uuids, notCancelled);
    for (const auto& uuid : uuids) rows_.erase(uuid);
    for (auto& row : listing.rows) rows_[row.record.uuid] = std::move(row);
    emit rowsChanged();
}

std::function<void()> SurfaceCatalog::fetchJob(const std::string& uuid) const
{
    const auto* listed = row(uuid);
    if (!listed || !listed->catalog || !fetcher_) throw std::runtime_error("The surface has no geometry");
    const auto store = session_->path(listed->source.attachment);
    return [fetcher = fetcher_, store, uuid]() { fetcher(store, uuid); };
}

std::vector<SurfaceCatalog::Source> SurfaceCatalog::sources() const
{
    std::vector<Source> sources;
    if (!volume_) return sources;
    if (attachment_) sources.push_back({*attachment_, volume_->id(), 1});
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
        std::none_of(listed.begin(), listed.end(), [&](const Source& source) { return source.attachment == event.attachment; }))
        return;
    if(std::any_of(event.changes.begin(),event.changes.end(),[](const auto& c){return c.kind=="volume_transform"||c.kind=="volume_transform_choice";})){read(std::nullopt);return;}
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
        [session = session_, listed, user = attachment_, requested, cancel = cancelled_]() {
            return list(*session, listed, user, requested, *cancel);
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

SurfaceCatalog::Opened SurfaceCatalog::open(const std::string& uuid) const
{
    if (const auto* listed = row(uuid))
        return open(session_, listed->source.attachment, listed->source.volume, uuid, listed->source.scale);
    if (!attachment_ || !volume_) throw std::runtime_error("Attach an object store and open a volume first");
    return open(session_, *attachment_, volume_->id(), uuid);
}

SurfaceCatalog::Opened SurfaceCatalog::open(const std::shared_ptr<ObjectStoreSession>& session,
                                            const AttachmentIdentity& attachment, const std::string& volume,
                                            const std::string& uuid, double scale)
{
    Opened opened;
    session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto revision = db.currentObjectRevision(txn, uuid);
        if (!revision || revision->record.retired || revision->record.kind != "surface")
            throw std::runtime_error("The surface is no longer in the object store");
        if (!revision->geometryRevision) throw NoGeometry("The surface has no geometry");
        const auto view=GeometrySnapshot(db,txn).viewTransformedAt(uuid,volume,scale);
        opened.surface=load_quad_from_store(db,txn,uuid,*revision->geometryRevision,true,1,&view);
        opened.view = {volume, scale, view.mapping()};
        opened.pin = {attachment, {attachment.storeUuid, revision->objectUuid}, revision->uuid, revision->geometryRevision};
    });
    opened.surface->setChannelLoader(store_channel_loader(session, attachment, *opened.pin.geometryRevision));
    opened.editable = session->writable(attachment);
    return opened;
}

void SurfaceCatalog::editRecord(const std::string& uuid, const std::function<void(ObjectRecord&)>& change)
{
    if (!attachment_) throw std::runtime_error("Attach an object store first");
    std::optional<std::string> revision;
    session_->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        if (const auto head = db.currentObjectRevision(txn, uuid)) revision = head->uuid;
    });
    if (!revision) throw std::runtime_error("The surface is no longer in the object store");
    session_->editKeepingRelationships(*attachment_, {{uuid, revision}}, {}, [&](auto& db, auto& txn, const auto&) {
        auto record = *db.objectByUuid(txn, uuid);
        change(record);
        db.upsertObject(txn, record);
        db.keepRecordedRelationshipsOf(txn, uuid);
    });
}

void SurfaceCatalog::setTags(const std::string& uuid, const std::vector<std::string>& tags)
{
    editRecord(uuid, [&](ObjectRecord& record) { record.tags = tags; });
}

void SurfaceCatalog::rename(const std::string& uuid, const std::string& name)
{
    editRecord(uuid, [&](ObjectRecord& record) { record.name = name; });
}

void SurfaceCatalog::updateMeta(const std::string& uuid, const std::function<void(nlohmann::json&)>& change)
{
    editRecord(uuid, [&](ObjectRecord& record) {
        auto meta = nlohmann::json::parse(record.meta);
        change(meta);
        record.meta = meta.dump();
    });
}

void SurfaceCatalog::retire(const std::vector<std::string>& uuids)
{
    if (!attachment_) throw std::runtime_error("Attach an object store first");
    RevisionSelection pins;
    session_->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& uuid : uuids)
            if (const auto head = db.currentObjectRevision(txn, uuid)) pins[uuid] = head->uuid;
    });
    session_->editKeepingRelationships(*attachment_, pins, {}, [&](auto& db, auto& txn, const auto&) {
        for (const auto& [uuid, revision] : pins) {
            db.retireObject(txn, uuid);
            db.keepRecordedRelationshipsOf(txn, uuid);
        }
    });
}

std::string SurfaceCatalog::copy(const std::string& uuid, const std::string& name)
{
    if (!attachment_) throw std::runtime_error("Attach an object store first");
    if (const auto* listed = row(uuid); listed && listed->catalog) {
        // A catalog surface becomes the user's: its geometry in the open volume's
        // voxels, bound to that volume, so it can be edited.
        auto opened = open(uuid);
        const nlohmann::json provenance{{"tool", "vc3d_copy"},
                                        {"parameters", {{"store_uuid", listed->source.attachment.storeUuid}}},
                                        {"sources", {uuid}}};
        return create(*opened.surface, name, provenance);
    }
    const auto copied = session_->copyObject(*attachment_, *attachment_, uuid);
    rename(copied, name);
    return copied;
}

std::string SurfaceCatalog::create(QuadSurface& surface, const std::string& name, const nlohmann::json& provenance,
                                   const std::vector<std::string>& tags)
{
    if (!attachment_ || !volume_) throw std::runtime_error("Attach an object store and open a volume first");
    auto meta = vc3d::coordinateBindingForVolume(*volume_, 1);
    meta["provenance"] = provenance;
    const auto uuid = util::newUuid();
    session_->act(*attachment_, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        create_quad_in_store(db, txn, uuid, surface, name, tags, meta.dump());
    });
    surface.id = uuid;
    return uuid;
}

std::vector<SurfaceProximityHit> SurfaceCatalog::nearest(const std::array<double, 3>& point, double tolerance,
                                                         const std::optional<std::vector<std::string>>& uuids) const
{
    std::vector<SurfaceProximityHit> hits;
    for (const auto& source : sources()) {
        // In the open volume's voxels: the source volume at the open level.
        SurfaceProximityQuery query;
        query.point = point;
        query.tolerance = tolerance;
        query.uuids = uuids;
        query.volume = source.volume;
        query.targetScale = source.scale;
        session_->read(source.attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (auto hit : db.nearestSurfaces(txn, query)) hits.push_back(std::move(hit));
        });
    }
    std::stable_sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
    return hits;
}
