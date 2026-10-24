#include "SurfaceCatalog.hpp"
#include "ObjectStoreCoordinates.hpp"
#include "ObjectStoreRegions.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
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
    if (!attachment_ || event.attachment != *attachment_ || event.kind != StoreSessionEvent::Kind::Committed) return;
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
    if (busy_ || !attachment_ || !volume_ || (!pendingFull_ && pending_.empty())) return;
    std::optional<std::vector<std::string>> requested;
    if (!std::exchange(pendingFull_, false)) requested = std::vector<std::string>(pending_.begin(), pending_.end());
    else loading_ = true;
    pending_.clear();
    busy_ = true;
    cancelled_ = std::make_shared<std::atomic_bool>(false);
    vc3d::store::background<Listing>(this,
        [session = session_, attachment = *attachment_, volume = volume_->id(), requested, cancel = cancelled_]() {
            Listing listing;
            listing.requested = requested;
            session->read(attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
                ObjectSelection selection;
                selection.kind = "surface";
                selection.volume = volume;
                selection.includeUnbound = true;
                if (requested) {
                    if (requested->empty()) return;
                    selection.uuids = *requested;
                }
                for (auto& record : db.selectObjects(txn, selection, kRowFields)) {
                    if (cancel->load() || cancelled.load()) return;
                    SurfaceCatalog::Row row;
                    row.toVolume = geometryScaleToVolume(nlohmann::json::parse(record.meta), volume);
                    row.record = std::move(record);
                    listing.rows.push_back(std::move(row));
                }
            });
            return listing;
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
    if (!attachment_ || !volume_) throw std::runtime_error("Attach an object store and open a volume first");
    Opened opened;
    const auto volume = volume_->id();
    session_->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        const auto revision = db.currentObjectRevision(txn, uuid);
        if (!revision || revision->record.retired || revision->record.kind != "surface")
            throw std::runtime_error("The surface is no longer in the object store");
        const auto factor = geometryScaleToVolume(nlohmann::json::parse(revision->record.meta), volume);
        if (!factor) throw std::runtime_error("The surface is not bound to volume " + volume);
        if (!revision->geometryRevision) throw std::runtime_error("The surface has no geometry");
        opened.surface = load_quad_from_store(db, txn, uuid, *revision->geometryRevision, true, *factor);
        opened.pin = {*attachment_, {attachment_->storeUuid, revision->objectUuid}, revision->uuid, revision->geometryRevision};
        opened.toVolume = *factor;
    });
    opened.surface->setChannelLoader(store_channel_loader(session_, *attachment_, *opened.pin.geometryRevision));
    opened.editable = opened.toVolume == 1 && session_->writable(*attachment_);
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

void SurfaceCatalog::retire(const std::vector<std::string>& uuids)
{
    if (!attachment_) throw std::runtime_error("Attach an object store first");
    RevisionSelection pins;
    session_->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& uuid : uuids)
            if (const auto head = db.currentObjectRevision(txn, uuid)) pins[uuid] = head->uuid;
    });
    session_->editKeepingRelationships(*attachment_, pins, {}, [&](auto& db, auto& txn, const auto&) {
        for (const auto& [uuid, revision] : pins) db.retireObject(txn, uuid);
    });
}

std::string SurfaceCatalog::copy(const std::string& uuid, const std::string& name)
{
    if (!attachment_) throw std::runtime_error("Attach an object store first");
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
