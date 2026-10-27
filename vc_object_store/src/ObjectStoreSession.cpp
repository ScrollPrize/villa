#include "FetchQueue.hpp"
#include <set>
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "hub/HubWorker.hpp"
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
    // The queue reports through the attachment once it is registered.
    auto owner = std::make_shared<AttachmentIdentity>();
    auto queue = std::make_unique<detail::FetchQueue>(
        [this, owner](const std::string& uuid, GeometryState state, const std::string& error) {
            std::lock_guard lock(mutex_);
            StoreSessionEvent event{StoreSessionEvent::Kind::Fetch, *owner, 0, {}};
            event.objectUuid = uuid;
            event.fetchState = state;
            event.error = error;
            events_.push_back(std::move(event));
        });
    auto* requests = queue.get();
    opened.remoteGeometryReached = [requests](const std::vector<std::string>& uuids) { requests->request(uuids); };
    auto next = std::make_shared<Attachment>(ObjectStore::open(path, opened));
    next->writable = false;  // callers read; fetched geometry is written by the session
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
        *owner = next->identity;
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

GeometryState ObjectStoreSession::geometryState(const AttachmentIdentity& identity, const std::string& objectUuid)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    auto attachment = acquire(identity, false);
    GeometryState state = GeometryState::Remote;
    try {
        bool local = false;
        {
            auto txn = attachment->store.read();
            local = attachment->store.currentGeometry(txn, uuid).has_value();
        }
        const auto fetching = attachment->fetches ? attachment->fetches->state(uuid) : std::nullopt;
        if (fetching && (*fetching == GeometryState::Fetching || (!local && *fetching == GeometryState::Failed)))
            state = *fetching;
        else if (local) state = GeometryState::Local;
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
    return state;
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

void ObjectStoreSession::cancelFetches(const AttachmentIdentity& identity)
{
    auto attachment = acquire(identity, false);
    if (attachment->fetches) attachment->fetches->cancelQueued();
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

std::vector<AttachedRegion> ObjectStoreSession::geometryRegion(const std::vector<AttachmentIdentity>& order,
    const RegionSlice& slice, const RegionOptions& options, const RegionCancel& cancel)
{
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
        std::erase_if(region.runs, [&](const VisibleRun& run) { return hidden.contains(run.objectUuid); });
        std::erase_if(region.remote, [&](const std::string& uuid) { return hidden.contains(uuid); });
        result.push_back({attachment, std::move(region)});
        earlier.push_back(attachment);
    }
    return result;
}

std::vector<AttachedHit> ObjectStoreSession::nearestSurfaces(const std::vector<AttachmentIdentity>& order,
                                                             const SurfaceProximityQuery& query)
{
    std::vector<AttachedHit> result;
    std::vector<AttachmentIdentity> earlier;
    for (const auto& attachment : order) {
        std::vector<SurfaceProximityHit> hits;
        read(attachment, [&](const auto& store, const auto& txn, const auto&) { hits = store.nearestSurfaces(txn, query); });
        std::set<std::string> uuids;
        for (const auto& hit : hits) uuids.insert(hit.objectUuid);
        const auto hidden = heldBy(*this, earlier, uuids);
        for (auto& hit : hits)
            if (!hidden.contains(hit.objectUuid)) result.push_back({attachment, std::move(hit)});
        earlier.push_back(attachment);
    }
    std::stable_sort(result.begin(), result.end(), [](const AttachedHit& a, const AttachedHit& b) {
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
    // Catalog fetches are of surfaces.
    if (kind == Kind::Fetch) return std::find(kinds.begin(), kinds.end(), "surface") != kinds.end();
    if (kind != Kind::Committed) return true;
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
