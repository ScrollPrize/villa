#include "FetchQueue.hpp"
#include <set>
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "hub/HubWorker.hpp"
#include "internal/SurfaceDrafts.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/sftp/Sftp.hpp"
#include "vc_object_store/Uuid.hpp"
#include <algorithm>
#include <stdexcept>

namespace vc::objects {
struct ObjectStoreSession::Attachment {
    explicit Attachment(ObjectStore value) : store(std::move(value)) {}
    ObjectStore store;
    AttachmentIdentity identity;
    bool writable = true;
    std::atomic_bool cancelled{false};
    // Admission, guarded by the session mutex.
    bool accepting = true;
    size_t readers = 0, writers = 0;
    // writer serializes accepted actions; journal guards journalPosition.
    std::mutex writer, journal;
    int64_t journalPosition = 0;
    // Hosted replicas only; declared after store so it stops first.
    std::unique_ptr<detail::HubWorker> hub;
    // Fetched attachments only (attachFetched): fetches published geometry.
    std::unique_ptr<detail::FetchQueue> fetches;
};

ObjectStoreSession::~ObjectStoreSession()
{
    // Retirement drains before releasing a handle, so no worker can still own
    // a transaction when its store closes.
    std::lock_guard transition(transition_);
    for (;;) {
        std::shared_ptr<Attachment> attachment;
        {
            std::lock_guard lock(mutex_);
            if (attached_.empty()) break;
            attachment = attached_.back();
        }
        try { retire(attachment); } catch (...) {
            std::lock_guard lock(mutex_);
            std::erase(attached_, attachment);
        }
    }
}

std::shared_ptr<ObjectStoreSession::Attachment> ObjectStoreSession::open(const std::filesystem::path& path,
    const OpenOptions& options, const HubSessionOptions& hubOptions)
{
    auto next = std::make_shared<Attachment>(ObjectStore::open(path, options));
    // Exercise write access without changing records. Opening a read connection
    // successfully is not sufficient validation for an editing session.
    next->writable = !options.readOnly;
    if (next->writable) { auto writable = next->store.write(RevisionSelection{}); writable.commit(); }
    auto txn = next->store.read();
    next->identity.storeUuid = next->store.uuid(txn);
    next->journalPosition = next->store.journalPosition(txn);
    return next;
}

AttachmentIdentity ObjectStoreSession::attach(const std::filesystem::path& path, const OpenOptions& options,
                                              const HubSessionOptions& hubOptions)
{
    std::lock_guard transition(transition_);
    auto next = open(path, options, hubOptions);
    {
        std::lock_guard lock(mutex_);
        if (std::any_of(attached_.begin(), attached_.end(),
                        [&](const auto& a) { return a->identity.storeUuid == next->identity.storeUuid; }))
            throw std::runtime_error("object store is already attached to this session");
        next->identity.generation = ++generation_;
        attached_.push_back(next);
        events_.push_back({StoreSessionEvent::Kind::Attached, next->identity, next->journalPosition, {}});
    }
    if (next->writable && vc::objects::hubStatus(next->store))
        next->hub = std::make_unique<detail::HubWorker>(next->store, hubOptions,
            [this, owner = next.get()] { hubChanged(owner); });
    return next->identity;
}

AttachmentIdentity ObjectStoreSession::replace(const AttachmentIdentity& identity, const std::filesystem::path& path,
    const OpenOptions& options, const HubSessionOptions& hubOptions)
{
    std::lock_guard transition(transition_);
    const auto previous = find(identity);
    if (!previous) throw std::runtime_error("object-store attachment is missing or superseded");
    auto next = open(path, options, hubOptions);
    {
        std::lock_guard lock(mutex_);
        if (std::any_of(attached_.begin(), attached_.end(), [&](const auto& a) {
                return a != previous && a->identity.storeUuid == next->identity.storeUuid; }))
            throw std::runtime_error("object store is already attached to this session");
    }
    retire(previous);
    {
        std::lock_guard lock(mutex_);
        next->identity.generation = ++generation_;
        attached_.push_back(next);
        events_.push_back({StoreSessionEvent::Kind::Attached, next->identity, next->journalPosition, {}});
    }
    if (next->writable && vc::objects::hubStatus(next->store))
        next->hub = std::make_unique<detail::HubWorker>(next->store, hubOptions,
            [this, owner = next.get()] { hubChanged(owner); });
    return next->identity;
}

void ObjectStoreSession::retire(const std::shared_ptr<Attachment>& attachment)
{
    std::unique_lock lock(mutex_);
    attachment->accepting = false;
    attachment->cancelled.store(true);
    idle_.wait(lock, [&] { return attachment->readers == 0 && attachment->writers == 0; });
    lock.unlock();
    try {
        // Its last commits are delivered before it is reported detached.
        observe(*attachment);
    } catch (...) {
        lock.lock();
        attachment->cancelled.store(false);
        attachment->accepting = true;
        throw;
    }
    lock.lock();
    std::erase(attached_, attachment);
    events_.push_back({StoreSessionEvent::Kind::Detached, attachment->identity, attachment->journalPosition, {}});
    // Stop the replica's sync and the catalog fetcher outside the lock their
    // status reports take.
    lock.unlock();
    attachment->hub.reset();
    attachment->fetches.reset();
}

void ObjectStoreSession::detach(const AttachmentIdentity& identity)
{
    std::lock_guard transition(transition_);
    if (const auto attachment = find(identity)) retire(attachment);
}

std::vector<AttachmentIdentity> ObjectStoreSession::attachments() const
{
    std::lock_guard lock(mutex_);
    std::vector<AttachmentIdentity> result;
    for (const auto& attachment : attached_)
        if (attachment->accepting) result.push_back(attachment->identity);
    return result;
}

std::shared_ptr<ObjectStoreSession::Attachment> ObjectStoreSession::find(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment->identity == identity) return attachment;
    return nullptr;
}

bool ObjectStoreSession::writable(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    return std::any_of(attached_.begin(), attached_.end(), [&](const auto& a) {
        return a->identity == identity && a->accepting && a->writable; });
}

bool ObjectStoreSession::current(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    return std::any_of(attached_.begin(), attached_.end(), [&](const auto& a) {
        return a->identity == identity && a->accepting; });
}

std::shared_ptr<ObjectStoreSession::Attachment> ObjectStoreSession::acquire(
    const AttachmentIdentity& identity, bool write)
{
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(attached_.begin(), attached_.end(),
                                    [&](const auto& a) { return a->identity == identity && a->accepting; });
    if (found == attached_.end()) throw std::runtime_error("object-store attachment is missing or superseded");
    if (write && !(*found)->writable) throw std::runtime_error("Object store is attached read-only");
    ++(write ? (*found)->writers : (*found)->readers);
    return *found;
}

void ObjectStoreSession::release(Attachment& attachment, bool write)
{
    std::lock_guard lock(mutex_);
    --(write ? attachment.writers : attachment.readers);
    idle_.notify_all();
}

void ObjectStoreSession::read(const AttachmentIdentity& identity, const Read& callback)
{
    auto attachment = acquire(identity, false);
    try {
        auto txn = attachment->store.read();
        callback(attachment->store, txn, attachment->cancelled);
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
}

RevisionSelection ObjectStoreSession::act(const AttachmentIdentity& identity,
    const RevisionSelection& expected, const Action& callback)
{
    RevisionSelection committed;
    auto attachment = acquire(identity, true);
    try {
        std::lock_guard writer(attachment->writer);
        auto txn = attachment->store.write(expected);
        callback(attachment->store, txn);
        txn.commit();
        committed = txn.committedRevisions();
        if (attachment->hub) attachment->hub->requestSync(true);
    } catch (...) { release(*attachment, true); throw; }
    release(*attachment, true);
    return committed;
}

std::string ObjectStoreSession::copyObject(const AttachmentIdentity& from, const AttachmentIdentity& to,
                                           const std::string& objectUuid, const std::string& name)
{
    std::optional<ObjectCopy> copy;
    read(from, [&](const auto& store, const auto& txn, const auto&) { copy = readObjectCopy(store, txn, objectUuid); });
    if (!name.empty()) copy->record.name = name;
    act(to, {{copy->record.uuid, std::nullopt}}, [&](auto& store, auto& txn) { writeObjectCopy(store, txn, *copy); });
    return copy->record.uuid;
}

void ObjectStoreSession::importObject(const AttachmentIdentity& from, const AttachmentIdentity& to,
                                      const std::string& objectUuid)
{
    ensureGeometry(from, objectUuid);
    std::optional<ObjectCopy> copy;
    read(from, [&](const auto& store, const auto& txn, const auto&) {
        copy = readObjectCopy(store, txn, objectUuid, util::canonicalUuid(objectUuid));
    });
    act(to, {{copy->record.uuid, std::nullopt}}, [&](auto& store, auto& txn) { writeObjectCopy(store, txn, *copy); });
}

void ObjectStoreSession::restoreImported(const AttachmentIdentity& identity, const std::string& objectUuid)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::vector<ObjectRevision> history;
    read(identity, [&](const auto& store, const auto& txn, const auto&) { history = store.objectHistory(txn, uuid); });
    if (history.empty()) throw std::runtime_error("object has no history: " + uuid);
    editCurrent(identity, {uuid}, [&](auto& store, auto& txn, const auto&) { store.restoreObject(txn, uuid, history.front().uuid); });
}

void ObjectStoreSession::editCurrent(const AttachmentIdentity& identity, const std::vector<std::string>& objectUuids,
                                     const std::function<void(ObjectStore&, WriteTxn&, const std::string&)>& edit)
{
    RevisionSelection pins;
    read(identity, [&](const auto& store, const auto& txn, const auto&) {
        for (const auto& objectUuid : objectUuids) {
            const auto uuid = util::canonicalUuid(objectUuid);
            const auto current = store.currentObjectRevision(txn, uuid);
            if (!current) throw std::runtime_error("object is not in the store: " + uuid);
            pins[uuid] = current->uuid;
        }
    });
    if (pins.empty()) return;
    editKeepingRelationships(identity, pins, {}, [&](auto& store, auto& txn, const auto&) {
        for (const auto& [uuid, revision] : pins) {
            edit(store, txn, uuid);
            store.keepRecordedRelationshipsOf(txn, uuid);
        }
    });
}

void ObjectStoreSession::editRecords(const AttachmentIdentity& identity, const std::vector<std::string>& objectUuids,
                                     const std::function<void(ObjectRecord&)>& change)
{
    editCurrent(identity, objectUuids, [&](auto& store, auto& txn, const auto& uuid) {
        auto record = *store.objectByUuid(txn, uuid);
        change(record);
        store.upsertObject(txn, record);
    });
}

void ObjectStoreSession::tagObjects(const AttachmentIdentity& identity, const std::vector<std::string>& objectUuids,
                                    const std::vector<std::string>& add, const std::vector<std::string>& remove)
{
    editRecords(identity, objectUuids, [&](ObjectRecord& record) {
        for (const auto& tag : add)
            if (std::find(record.tags.begin(), record.tags.end(), tag) == record.tags.end()) record.tags.push_back(tag);
        for (const auto& tag : remove) std::erase(record.tags, tag);
    });
}

AttachmentIdentity ObjectStoreSession::attachFetched(const std::filesystem::path& path, const OpenOptions& options,
                                                     GeometryFetcher fetch)
{
    std::lock_guard transition(transition_);
    auto opened = options;
    opened.readOnly = false;
    auto queue = std::make_unique<detail::FetchQueue>();
    auto* requests = queue.get();
    opened.remoteGeometryReached = [requests](const std::vector<std::string>& uuids) { requests->request(uuids); };
    auto next = std::make_shared<Attachment>(ObjectStore::open(path, opened));
    next->writable = !options.readOnly;  // fetching and local edits share the writer lock
    {
        auto txn = next->store.read();
        next->identity.storeUuid = next->store.uuid(txn);
        next->journalPosition = next->store.journalPosition(txn);
    }
    {
        std::lock_guard lock(mutex_);
        if (std::any_of(attached_.begin(), attached_.end(),
                        [&](const auto& a) { return a->identity.storeUuid == next->identity.storeUuid; }))
            throw std::runtime_error("object store is already attached to this session");
        next->identity.generation = ++generation_;
        attached_.push_back(next);
        events_.push_back({StoreSessionEvent::Kind::Attached, next->identity, next->journalPosition, {}});
    }
    const auto identity = next->identity;
    requests->start([this, identity, fetch = std::move(fetch)](const std::string& uuid) {
        auto attachment = acquire(identity, false);
        try {
            fetch(attachment->store, uuid, [&](const RevisionSelection& expected, const Action& action) {
                return fetchWrite(identity, expected, action);
            });
        } catch (...) { release(*attachment, false); throw; }
        release(*attachment, false);
    });
    next->fetches = std::move(queue);
    return identity;
}

RevisionSelection ObjectStoreSession::fetchWrite(const AttachmentIdentity& identity, const RevisionSelection& expected,
                                                 const Action& callback)
{
    RevisionSelection committed;
    auto attachment = acquire(identity, false);
    try {
        if (!attachment->fetches) throw std::runtime_error("not a fetched attachment");
        std::lock_guard writer(attachment->writer);
        auto txn = attachment->store.write(expected);
        callback(attachment->store, txn);
        txn.commit();
        committed = txn.committedRevisions();
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
    return committed;
}

void ObjectStoreSession::refetch(const AttachmentIdentity& identity, const std::vector<std::string>& objectUuids)
{
    auto attachment = acquire(identity, false);
    if (attachment->fetches) attachment->fetches->refetch(objectUuids);
    release(*attachment, false);
}

void ObjectStoreSession::ensureGeometry(const AttachmentIdentity& identity, const std::string& objectUuid,
                                        const std::function<bool()>& cancel)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    auto attachment = acquire(identity, false);
    try {
        bool local = false;
        {
            auto txn = attachment->store.read();
            local = attachment->store.currentGeometry(txn, uuid).has_value();
        }
        if (!local) {
            if (!attachment->fetches) throw GeometryUnavailable("object has no geometry: " + uuid);
            std::string error;
            if (!attachment->fetches->fetchNow(uuid, [&] { return attachment->cancelled.load() || (cancel && cancel()); },
                                               &error))
                throw GeometryUnavailable(error.empty() ? "fetching " + uuid + " was cancelled"
                                                        : "cannot fetch " + uuid + ": " + error);
        }
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
}

namespace {
// The UUIDs among `uuids` that one of `attachments` holds.
template <class Session>
std::set<std::string> heldBy(Session& session, const std::vector<AttachmentIdentity>& attachments,
                             const std::set<std::string>& uuids)
{
    std::set<std::string> held;
    if (uuids.empty()) return held;
    for (const auto& attachment : attachments)
        session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
            for (const auto& uuid : uuids)
                if (!held.contains(uuid) && store.objectByUuid(txn, uuid)) held.insert(uuid);
        });
    return held;
}
}

std::vector<AttachedRecord> ObjectStoreSession::selectObjects(const std::vector<AttachmentIdentity>& order,
                                                              const ObjectSelection& selection)
{
    std::vector<AttachedRecord> result;
    std::vector<AttachmentIdentity> earlier;
    for (const auto& attachment : order) {
        std::vector<ObjectRecord> records;
        read(attachment, [&](const auto& store, const auto& txn, const auto&) { records = store.selectObjects(txn, selection); });
        std::set<std::string> uuids;
        for (const auto& record : records) uuids.insert(record.uuid);
        const auto hidden = heldBy(*this, earlier, uuids);
        for (auto& record : records)
            if (!hidden.contains(record.uuid)) result.push_back({attachment, std::move(record)});
        earlier.push_back(attachment);
    }
    return result;
}

namespace {
// The attachment a draft's results report.
AttachmentIdentity draftAttachment(const detail::SurfaceDraft& draft)
{
    return draft.attachment.value_or(AttachmentIdentity{});
}
}

ObjectStoreSession::Drafts ObjectStoreSession::drafts() const
{
    std::lock_guard lock(mutex_);
    return drafts_;
}

void ObjectStoreSession::draftChanged(const std::string& objectUuid, const std::optional<AttachmentIdentity>& attachment)
{
    JournalEntry change;
    change.recordUuid = objectUuid;
    change.kind = "surface";
    change.changeKind = "draft";
    events_.push_back({StoreSessionEvent::Kind::Draft, attachment.value_or(AttachmentIdentity{}), 0, {std::move(change)}});
}

void ObjectStoreSession::setDraft(const std::string& objectUuid, const vc::geometry::Surface& surface,
                                  const CoordinateJson& coordinates, const std::optional<AttachmentIdentity>& attachment)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    auto draft = detail::makeSurfaceDraft(surface, coordinates, attachment);
    std::lock_guard lock(mutex_);
    drafts_[uuid] = std::move(draft);
    draftChanged(uuid, attachment);
}

void ObjectStoreSession::updateDraft(const std::string& objectUuid, int64_t row, int64_t column, int64_t rows,
                                     int64_t columns, std::span<const vc::geometry::SurfacePoint> points,
                                     std::span<const uint8_t> validity)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::shared_ptr<const detail::SurfaceDraft> current;
    {
        std::lock_guard lock(mutex_);
        const auto found = drafts_.find(uuid);
        if (found == drafts_.end()) throw std::invalid_argument("no draft of surface " + uuid);
        current = found->second;
    }
    // Edits of one draft come from its editor, one at a time; build outside the lock.
    auto edited = detail::editSurfaceDraft(*current, row, column, rows, columns, points, validity);
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(uuid);
    if (found == drafts_.end() || found->second != current)
        throw std::logic_error("surface draft " + uuid + " changed during an edit");
    found->second = std::move(edited);
    draftChanged(uuid, current->attachment);
}

void ObjectStoreSession::clearDraft(const std::string& objectUuid)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(uuid);
    if (found == drafts_.end()) return;
    const auto attachment = found->second->attachment;
    drafts_.erase(found);
    draftChanged(uuid, attachment);
}

bool ObjectStoreSession::hasDraft(const std::string& objectUuid) const
{
    std::lock_guard lock(mutex_);
    return drafts_.contains(util::canonicalUuid(objectUuid));
}

namespace {
// Volume transforms for placing drafts, read once per attachment and query.
class DraftTransforms {
public:
    DraftTransforms(ObjectStoreSession& session, const std::vector<AttachmentIdentity>& order,
                    std::shared_ptr<const VolumeTransforms> given)
        : session_(session), order_(order), given_(std::move(given)) {}
    const VolumeTransforms* operator()(const detail::SurfaceDraft& draft) {
        if (given_) return given_.get();
        std::optional<AttachmentIdentity> source;
        if (draft.attachment && session_.current(*draft.attachment)) source = draft.attachment;
        else if (!order_.empty()) source = order_.front();
        if (!source) return nullptr;
        for (const auto& [attachment, transforms] : read_)
            if (attachment == *source) return transforms.get();
        auto transforms = std::make_shared<VolumeTransforms>();
        session_.read(*source, [&](const auto& store, const auto& txn, const auto&) { *transforms = store.volumeTransforms(txn); });
        read_.emplace_back(*source, transforms);
        return transforms.get();
    }
private:
    ObjectStoreSession& session_;
    const std::vector<AttachmentIdentity>& order_;
    std::shared_ptr<const VolumeTransforms> given_;
    std::vector<std::pair<AttachmentIdentity, std::shared_ptr<VolumeTransforms>>> read_;
};
bool admits(const std::optional<std::vector<std::string>>& uuids, const std::string& uuid)
{
    return !uuids || std::any_of(uuids->begin(), uuids->end(), [&](const auto& u) { return util::canonicalUuid(u) == uuid; });
}
}

std::vector<AttachedRegion> ObjectStoreSession::geometryRegion(const std::vector<AttachmentIdentity>& order,
    const RegionSlice& slice, const RegionOptions& options, const RegionCancel& cancel)
{
    const auto drafts = options.kind == "surface" ? this->drafts() : Drafts{};
    std::vector<AttachedRegion> result;
    std::vector<AttachmentIdentity> earlier;
    for (const auto& attachment : order) {
        RegionResult region;
        read(attachment, [&](const auto& store, const auto& txn, const auto&) {
            region = store.geometryRegion(txn, slice, options, cancel);
        });
        std::set<std::string> uuids(region.remote.begin(), region.remote.end());
        for (const auto& run : region.runs) uuids.insert(run.objectUuid);
        const auto hidden = heldBy(*this, earlier, uuids);
        const auto replaced = [&](const std::string& uuid) { return hidden.contains(uuid) || drafts.contains(uuid); };
        std::erase_if(region.runs, [&](const VisibleRun& run) { return replaced(run.objectUuid); });
        std::erase_if(region.remote, replaced);
        result.push_back({attachment, std::move(region)});
        earlier.push_back(attachment);
    }
    // Drafts, in UUID order with the selected one first, each attachment's
    // within its own budgets and object limit like a stored attachment's.
    DraftTransforms transforms(*this, order, options.transforms);
    const auto selected = options.maximumObjects && options.selected
        ? std::optional(util::canonicalUuid(*options.selected)) : std::nullopt;
    std::vector<const Drafts::value_type*> ordered;
    for (const auto& entry : drafts) ordered.push_back(&entry);
    std::stable_partition(ordered.begin(), ordered.end(), [&](const auto* entry) { return entry->first == selected; });
    std::vector<AttachedRegion> drafted;
    for (const auto* entry : ordered) {
        const auto& [uuid, draft] = *entry;
        if (!admits(options.uuids, uuid)) continue;
        const auto placement = detail::placeDraft(*draft, options.coordinates, options.volume, options.targetScale,
                                                  options.toTarget, options.volume ? transforms(*draft) : nullptr);
        if (!placement) continue;
        const auto attachment = draftAttachment(*draft);
        auto group = std::find_if(drafted.begin(), drafted.end(), [&](const auto& g) { return g.attachment == attachment; });
        if (group == drafted.end()) group = drafted.insert(drafted.end(), AttachedRegion{attachment, {}});
        auto& region = group->result;
        if (!region.complete()) continue;
        detail::draftRegion(uuid, *draft, *placement, slice, options, cancel, region);
        if (region.complete() && options.maximumObjects && region.intersectingObjects == options.maximumObjects)
            region.stop = RegionStop::ObjectLimit;
    }
    for (auto& group : drafted) result.push_back(std::move(group));
    return result;
}

std::vector<AttachedHit> ObjectStoreSession::nearestSurfaces(const std::vector<AttachmentIdentity>& order,
                                                             const SurfaceProximityQuery& query)
{
    return std::move(nearestSurfaces(order, query, {query.point}).front());
}

std::vector<std::vector<AttachedHit>> ObjectStoreSession::nearestSurfaces(const std::vector<AttachmentIdentity>& order,
                                                                          const SurfaceProximityQuery& query,
                                                                          const std::vector<std::array<double, 3>>& points)
{
    const auto drafts = this->drafts();
    std::vector<std::vector<AttachedHit>> result(points.size());
    std::vector<AttachmentIdentity> earlier;
    for (const auto& attachment : order) {
        std::vector<std::vector<SurfaceProximityHit>> hits(points.size());
        read(attachment, [&](const auto& store, const auto& txn, const auto&) {
            auto around = query;
            if (around.volume && !around.transforms)
                around.transforms = std::make_shared<VolumeTransforms>(store.volumeTransforms(txn));
            for (std::size_t i = 0; i < points.size(); ++i) {
                around.point = points[i];
                hits[i] = store.nearestSurfaces(txn, around);
            }
        });
        std::set<std::string> uuids;
        for (const auto& near : hits)
            for (const auto& hit : near) uuids.insert(hit.objectUuid);
        const auto hidden = heldBy(*this, earlier, uuids);
        for (std::size_t i = 0; i < points.size(); ++i)
            for (auto& hit : hits[i])
                if (!hidden.contains(hit.objectUuid) && !drafts.contains(hit.objectUuid))
                    result[i].push_back({attachment, std::move(hit)});
        earlier.push_back(attachment);
    }
    DraftTransforms transforms(*this, order, query.transforms);
    for (const auto& [uuid, draft] : drafts) {
        if (!admits(query.uuids, uuid)) continue;
        const auto placement = detail::placeDraft(*draft, query.coordinates, query.volume, query.targetScale,
                                                  query.toTarget, query.volume ? transforms(*draft) : nullptr);
        if (!placement) continue;
        for (std::size_t i = 0; i < points.size(); ++i)
            if (auto hit = detail::draftNearest(uuid, *draft, *placement, points[i], query.tolerance))
                result[i].push_back({draftAttachment(*draft), std::move(*hit)});
    }
    for (auto& near : result)
        std::stable_sort(near.begin(), near.end(), [](const AttachedHit& a, const AttachedHit& b) {
            return a.hit.distance != b.hit.distance ? a.hit.distance < b.hit.distance : a.hit.objectUuid < b.hit.objectUuid;
        });
    return result;
}

std::filesystem::path ObjectStoreSession::artifactDirectory(const AttachmentIdentity& identity, const std::string& objectUuid,
                                                           bool create)
{
    const auto object = util::canonicalUuid(objectUuid);
    std::filesystem::path base;
    auto attachment = acquire(identity, false);
    try {
        if (const auto hub = vc::objects::hubStatus(attachment->store)) base = hub->hub;
        else base = attachment->store.path();
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
    if (sftp::isSftpUrl(base.string()))
        throw std::runtime_error("an sftp:// hub has no shared artifact directory; mount the share instead");
    auto directory = std::filesystem::path(base.string() + ".artifacts") / object;
    if (create) std::filesystem::create_directories(directory);
    return directory;
}

std::filesystem::path ObjectStoreSession::path(const AttachmentIdentity& identity)
{
    auto attachment = acquire(identity, false);
    const auto result = attachment->store.path();
    release(*attachment, false);
    return result;
}

RevisionSelection ObjectStoreSession::editKeepingRelationships(const AttachmentIdentity& identity,
    const RevisionSelection& pins, const RevisionSelection& related,
    const std::function<void(ObjectStore&, WriteTxn&, const KeepRecorded&)>& edit)
{
    // Objects pinned without a revision are created by the edit: none to lease.
    std::vector<std::string> existing;
    for (const auto& [uuid, revision] : pins)
        if (revision) existing.push_back(uuid);
    const auto leases = holdLeases(identity, existing);
    auto expected = pins;
    for (const auto& [uuid, revision] : related)
        if (!expected.emplace(uuid, revision).second && expected.at(uuid) != revision)
            throw std::invalid_argument("object store: conflicting revisions for " + uuid);
    read(identity, [&](const auto& store, const auto& txn, const auto&) {
        // Objects expected absent have no relationships yet.
        const auto touched = expected;
        for (const auto& [uuid, revision] : touched)
            if (revision) for (const auto& edge : store.edgesOf(txn, uuid, true))
                for (const auto& endpoint : {edge.aUuid, edge.bUuid})
                    if (!expected.contains(endpoint))
                        if (const auto head = store.currentObjectRevision(txn, endpoint)) expected[endpoint] = head->uuid;
    });
    return act(identity, expected, [&](auto& store, auto& txn) {
        const KeepRecorded keep = [&](const std::vector<EdgeRecord>& affected) {
            std::vector<GeometryEdgeEdit> result;
            for (const auto& edge : affected) {
                GeometryEdgeEdit decision;
                decision.edgeUuid = edge.uuid;
                decision.expectedGeneration = edge.generation;
                decision.action = "keep_recorded";
                if (const auto r = store.currentGeometry(txn, edge.aUuid)) decision.aRevision = r->uuid;
                if (const auto r = store.currentGeometry(txn, edge.bUuid)) decision.bRevision = r->uuid;
                result.push_back(std::move(decision));
            }
            return result;
        };
        edit(store, txn, keep);
    });
}

void ObjectStoreSession::finalize()
{
    std::lock_guard transition(transition_);
    std::unique_lock lock(mutex_);
    for (const auto& attachment : attached_) attachment->accepting = false;
    idle_.wait(lock, [&] {
        return std::all_of(attached_.begin(), attached_.end(), [](const auto& a) { return a->writers == 0; });
    });
    for (const auto& attachment : attached_) attachment->accepting = true;
}

void ObjectStoreSession::observe(Attachment& attachment)
{
    std::lock_guard journal(attachment.journal);
    auto txn = attachment.store.read();
    auto changes = attachment.store.journalSince(txn, attachment.journalPosition);
    if (changes.empty()) return;
    const auto position = changes.back().seq;
    std::lock_guard lock(mutex_);
    events_.push_back({StoreSessionEvent::Kind::Committed, attachment.identity, position, std::move(changes)});
    attachment.journalPosition = position;
}

std::vector<StoreSessionEvent> ObjectStoreSession::takeEvents()
{
    // Observation participates in each attachment's barrier; no event of a
    // retired attachment can arrive after its Detached notification.
    std::vector<std::shared_ptr<Attachment>> current;
    {
        std::lock_guard lock(mutex_);
        for (const auto& attachment : attached_)
            if (attachment->accepting) { ++attachment->readers; current.push_back(attachment); }
    }
    std::exception_ptr error;
    for (const auto& attachment : current) {
        if (!error) try { observe(*attachment); } catch (...) { error = std::current_exception(); }
        release(*attachment, false);
    }
    if (error) std::rethrow_exception(error);
    std::lock_guard lock(mutex_);
    std::vector<StoreSessionEvent> result;
    result.swap(events_);
    return result;
}

bool StoreSessionEvent::touches(std::initializer_list<std::string_view> kinds) const
{
    if (kind != Kind::Committed && kind != Kind::Draft) return true;
    return std::any_of(changes.begin(), changes.end(), [&](const auto& change) {
        return std::find(kinds.begin(), kinds.end(), change.kind) != kinds.end();
    });
}

LeaseHold ObjectStoreSession::holdForEditing(const PinnedObject& pin)
{
    auto lease = holdLeases(pin.attachment, {pin.object.objectUuid});
    read(pin.attachment, [&](const auto& store, const auto& txn, const auto&) {
        const auto head = store.currentObjectRevision(txn, pin.object.objectUuid);
        if (!head || head->uuid != pin.objectRevision)
            throw std::runtime_error("It changed in the object store since it was opened; reopen it to edit");
    });
    return lease;
}

LeaseHold ObjectStoreSession::holdLeases(const AttachmentIdentity& identity, const std::vector<std::string>& objects)
{
    auto attachment = acquire(identity, true);
    LeaseHold hold;
    try {
        if (attachment->hub) hold = attachment->hub->hold(objects);
    } catch (...) { release(*attachment, true); throw; }
    release(*attachment, true);
    return hold;
}

std::optional<HubSessionStatus> ObjectStoreSession::hubStatus(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment->identity == identity && attachment->accepting && attachment->hub) return attachment->hub->status();
    return std::nullopt;
}

void ObjectStoreSession::syncHub(const AttachmentIdentity& identity)
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment->identity == identity && attachment->hub) attachment->hub->requestSync(false);
}

std::optional<HubSessionStatus> ObjectStoreSession::syncHubAndWait(const AttachmentIdentity& identity,
                                                                   std::chrono::milliseconds timeout)
{
    // Admission keeps the worker alive: detaching waits for the sync.
    auto attachment = acquire(identity, false);
    std::optional<HubSessionStatus> status;
    try {
        if (attachment->hub) status = attachment->hub->syncAndWait(timeout);
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
    return status;
}

void ObjectStoreSession::resolveHubConflicts(const AttachmentIdentity& identity, const std::string& keep)
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment->identity == identity && attachment->hub) attachment->hub->requestResolve(keep);
}

void ObjectStoreSession::hubChanged(const Attachment* owner)
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment.get() == owner) events_.push_back({StoreSessionEvent::Kind::Hub, attachment->identity, 0, {}});
}
}
