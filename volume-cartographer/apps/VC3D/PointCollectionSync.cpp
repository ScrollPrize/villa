#include "PointCollectionSync.hpp"

#include "CState.hpp"
#include "vc/core/StoredCollection.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/ui/VCCollection.hpp"

#include <vc_object_store/objects/AttachedDocument.hpp>

#include <nlohmann/json.hpp>

#include <limits>

namespace vc3d {
namespace {
// Edits arriving within this window are written in one transaction.
constexpr int kDebounceMs = 500;
// The kind of the store objects collections live in.
constexpr const char* kKind = "point_collection";
}

PointCollectionSync::PointCollectionSync(CState* state) : QObject(state), state_(state)
{
    debounce_.setSingleShot(true);
    debounce_.setInterval(kDebounceMs);
    connect(&debounce_, &QTimer::timeout, this, &PointCollectionSync::flush);
    auto* points = state_->pointCollection();
    connect(points, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>& ids) {
        for (const auto id : ids) markChanged(id);
    });
    connect(points, &VCCollection::collectionChanged, this, &PointCollectionSync::markChanged);
    connect(points, &VCCollection::collectionRemoved, this, &PointCollectionSync::markRemoved);
    connect(points, &VCCollection::pointAdded, this, [this](const ColPoint& p) { markChanged(p.collectionId); });
    connect(points, &VCCollection::pointsAdded, this, [this](const std::vector<ColPoint>& added) {
        for (const auto& p : added) markChanged(p.collectionId);
    });
    connect(points, &VCCollection::pointChanged, this, [this](const ColPoint& p) { markChanged(p.collectionId); });
    connect(points, &VCCollection::pointRemoved, this, [this](vc::PointRef p) { markChanged(p.collectionId); });
    connect(points, &VCCollection::pointsRemoved, this, [this](const std::vector<vc::PointRef>& removed) {
        for (const auto& p : removed) markChanged(p.collectionId);
    });
    connect(state_, &CState::volumeChanged, this, [this] { reload(); });
    connect(state_, &CState::objectStoreChanged, this, &PointCollectionSync::onStoreEvent);
    connect(state_, &CState::storeRoutesChanged, this, &PointCollectionSync::reload);
}

bool PointCollectionSync::anchored(uint64_t collectionId) const
{
    const auto& all = state_->pointCollection()->getAllCollections();
    const auto found = all.find(collectionId);
    return found != all.end() && found->second.anchor2d.has_value();
}

void PointCollectionSync::markChanged(uint64_t collectionId)
{
    if (installing_ || !attachment_ || anchored(collectionId)) return;
    const auto& all = state_->pointCollection()->getAllCollections();
    const auto found = all.find(collectionId);
    if (found == all.end()) return;
    uuids_[collectionId] = found->second.uuid;
    changed_.insert(collectionId);
    debounce_.start();
}

void PointCollectionSync::markRemoved(uint64_t collectionId)
{
    if (installing_ || !attachment_) return;
    // The sentinel for "all removed".
    if (collectionId == std::numeric_limits<uint64_t>::max()) {
        for (const auto& [id, uuid] : uuids_) removed_.insert(uuid);
        uuids_.clear();
        changed_.clear();
    } else if (const auto found = uuids_.find(collectionId); found != uuids_.end()) {
        removed_.insert(found->second);
        uuids_.erase(found);
        changed_.erase(collectionId);
    } else {
        return;
    }
    debounce_.start();
}

bool PointCollectionSync::flush()
{
    debounce_.stop();
    if (!attachment_ || (changed_.empty() && removed_.empty())) return true;
    const auto& all = state_->pointCollection()->getAllCollections();
    std::vector<vc::objects::AttachedCollection> changed;
    for (const auto id : changed_) {
        const auto found = all.find(id);
        if (found == all.end() || found->second.anchor2d) continue;
        utils::Json json;
        to_json(json, found->second);
        auto stored = vc::storedCollection(json);
        // Linked windings by UUID: runtime ids do not survive a reload.
        auto linked = nlohmann::json::array();
        for (const auto other : found->second.windings_linked)
            if (const auto uuid = uuids_.find(other); uuid != uuids_.end()) linked.push_back(uuid->second);
        stored.source["windings_linked"] = std::move(linked);
        changed.push_back(std::move(stored));
    }
    const std::vector<std::string> removed(removed_.begin(), removed_.end());
    try {
        vc::objects::writeVolumeCollections(*state_->objectStoreSession(), *attachment_, volume_, changed, removed);
        std::vector<std::string> uuids(removed);
        for (const auto& collection : changed) uuids.push_back(collection.uuid);
        noteRevisions(uuids);
        changed_.clear();
        removed_.clear();
        return true;
    } catch (const std::exception& error) {
        Logger()->warn("Point collections: could not save to the object store: {}", error.what());
        return false;
    }
}

void PointCollectionSync::noteRevisions(const std::vector<std::string>& uuids)
{
    if (!attachment_ || uuids.empty()) return;
    state_->objectStoreSession()->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& uuid : uuids)
            if (const auto head = db.currentObjectRevision(txn, uuid)) revisions_[uuid] = head->uuid;
            else revisions_.erase(uuid);
    });
}

void PointCollectionSync::unload()
{
    if (!flush()) return;
    attachment_.reset();
    volume_ = {};
    uuids_.clear();
    installing_ = true;
    state_->pointCollection()->replaceUnanchoredCollections({});
    installing_ = false;
}

void PointCollectionSync::reload()
{
    if (!flush()) return;
    const auto attachment = state_->storeFor("point_collection");
    const auto& volume = state_->annotationVolume();
    std::vector<vc::objects::AttachedCollection> stored;
    if (attachment && volume) {
        try {
            stored = vc::objects::volumeCollections(*state_->objectStoreSession(), *attachment, *volume);
        } catch (const std::exception& error) {
            Logger()->warn("Point collections: could not read the object store: {}", error.what());
        }
    }
    attachment_ = attachment && volume ? attachment : std::nullopt;
    volume_ = volume.value_or(vc::project::AnnotationVolume{});
    std::vector<utils::Json> collections;
    std::map<std::string, std::vector<std::string>> linked;
    for (auto& collection : stored) {
        for (const auto& uuid : collection.source.value("windings_linked", nlohmann::json::array()))
            if (uuid.is_string()) linked[collection.uuid].push_back(uuid.get<std::string>());
        collection.source.erase("windings_linked");
        collections.push_back(vc::collectionJson(collection));
    }
    auto* points = state_->pointCollection();
    installing_ = true;
    points->replaceUnanchoredCollections(collections);
    uuids_.clear();
    std::map<std::string, uint64_t> ids;
    for (const auto& [id, collection] : points->getAllCollections())
        if (!collection.anchor2d) {
            uuids_[id] = collection.uuid;
            ids[collection.uuid] = id;
        }
    std::vector<std::string> loaded;
    for (const auto& collection : stored) loaded.push_back(collection.uuid);
    revisions_.clear();
    noteRevisions(loaded);
    for (const auto& [uuid, others] : linked) {
        std::vector<uint64_t> otherIds;
        for (const auto& other : others)
            if (const auto found = ids.find(other); found != ids.end()) otherIds.push_back(found->second);
        if (const auto found = ids.find(uuid); found != ids.end()) points->setCollectionWindingsLinked(found->second, otherIds);
    }
    installing_ = false;
}

void PointCollectionSync::onStoreEvent(const vc::objects::StoreSessionEvent& event)
{
    using Kind = vc::objects::StoreSessionEvent::Kind;
    if (event.kind == Kind::Attached || event.kind == Kind::Detached) {
        reload();
        return;
    }
    if (event.kind != Kind::Committed || !attachment_ || event.attachment.storeUuid != attachment_->storeUuid ||
        !event.touches({kKind}))
        return;
    // Our own writes coming back are no reason to reload: a collection is
    // someone else's change when its revision is not the one we wrote or read.
    std::vector<std::string> touched;
    for (const auto& change : event.changes)
        if (change.kind == kKind) touched.push_back(change.recordUuid);
    bool external = false;
    state_->objectStoreSession()->read(*attachment_, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& uuid : touched) {
            const auto head = db.currentObjectRevision(txn, uuid);
            const auto known = revisions_.find(uuid);
            if ((head ? head->uuid : std::string{}) != (known == revisions_.end() ? std::string{} : known->second))
                external = true;
        }
    });
    if (external) reload();
}

} // namespace vc3d
