#include "FetchQueue.hpp"
#include "HostedSession.hpp"
#include "DraftCheckpoints.hpp"
#include "internal/Artifacts.hpp"
#include <set>
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "internal/SurfaceDrafts.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include <vc_reader/native/platform_io.hpp>
#include <vc_reader/sftp.hpp>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace vc::objects {
struct ObjectStoreSession::Attachment {
    explicit Attachment(ObjectStore value) : local(std::move(value)) {}
    explicit Attachment(std::unique_ptr<detail::HostedAttachment> value) : remote(std::move(value)) {}
    // Exactly one backend: a store this process opened, or a hosted one.
    std::optional<ObjectStore> local;
    std::unique_ptr<detail::HostedAttachment> remote;
    ObjectStore& store()
    {
        if (!local)
            throw std::invalid_argument("object store attachment is hosted: it has no local database; "
                                        "use the session's typed operations");
        return *local;
    }
    AttachmentIdentity identity;
    // Attribution of newRequest() requests.
    std::string author;
    bool writable = true;
    std::atomic_bool cancelled{false};
    // Admission, guarded by the session mutex.
    bool accepting = true;
    size_t readers = 0, writers = 0;
    // writer serializes accepted actions; journal guards journalPosition.
    std::mutex writer, journal;
    int64_t journalPosition = 0;
    // Current-only stores have no journal; compare their committed state instead.
    std::optional<StoreStatus> observedState;
    // Writable attachments: fetches the geometry of objects published elsewhere
    // (setGeometryFetcher).
    std::unique_ptr<detail::FetchQueue> fetches;
    std::shared_ptr<ObjectPrefetchJob> prefetch; // guarded by session mutex
};

template <class F>
auto ObjectStoreSession::withAttachment(const AttachmentIdentity& identity, bool write, F&& f)
{
    auto attachment = acquire(identity, write);
    struct Admitted {
        ObjectStoreSession& session;
        Attachment& attachment;
        bool write;
        ~Admitted() { session.release(attachment, write); }
    } admitted{*this, *attachment, write};
    return f(*attachment);
}

ObjectStoreSession::ObjectStoreSession() : durable_(std::make_unique<detail::DurableDrafts>()) {}

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
    const OpenOptions& options)
{
    // Reads that reach objects without local geometry queue their fetch.
    std::unique_ptr<detail::FetchQueue> fetches;
    auto opened = options;
    if (!options.readOnly) {
        fetches = std::make_unique<detail::FetchQueue>();
        opened.remoteGeometryReached = [requests = fetches.get()](const std::vector<std::string>& uuids) {
            requests->request(uuids);
        };
    }
    auto next = std::make_shared<Attachment>(ObjectStore::open(path, opened));
    next->fetches = std::move(fetches);
    // Exercise write access without changing records. Opening a read connection
    // successfully is not sufficient validation for an editing session.
    next->writable = !options.readOnly;
    if (next->writable) { auto writable = next->store().write(); writable.commit(); }
    next->author = next->store().author();
    auto txn = next->store().read();
    next->identity.storeUuid = next->store().uuid(txn);
    next->observedState = next->store().status(txn);
    next->journalPosition = next->observedState->historyEnabled
        ? next->store().journalPosition(txn) : next->observedState->commitSequence;
    return next;
}

AttachmentIdentity ObjectStoreSession::attach(const std::filesystem::path& path, const OpenOptions& options)
{
    std::lock_guard transition(transition_);
    auto next = open(path, options);
    {
        std::lock_guard lock(mutex_);
        if (std::any_of(attached_.begin(), attached_.end(),
                        [&](const auto& a) { return a->identity.storeUuid == next->identity.storeUuid; }))
            throw std::runtime_error("object store is already attached to this session");
        next->identity.generation = ++generation_;
        attached_.push_back(next);
        events_.push_back({StoreSessionEvent::Kind::Attached, next->identity, next->journalPosition, {}});
    }
    startFetching(*next);
    return next->identity;
}

AttachmentIdentity ObjectStoreSession::replace(const AttachmentIdentity& identity, const std::filesystem::path& path,
    const OpenOptions& options)
{
    std::lock_guard transition(transition_);
    const auto previous = find(identity);
    if (!previous) throw std::runtime_error("object-store attachment is missing or superseded");
    auto next = open(path, options);
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
    startFetching(*next);
    return next->identity;
}

void ObjectStoreSession::retire(const std::shared_ptr<Attachment>& attachment)
{
    std::unique_lock lock(mutex_);
    attachment->accepting = false;
    attachment->cancelled.store(true);
    idle_.wait(lock, [&] { return attachment->readers == 0 && attachment->writers == 0; });
    lock.unlock();
    // A hosted attachment's poller delivers nothing after this.
    if (attachment->remote) attachment->remote->stop();
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
    // Stop the catalog fetcher outside the lock.
    lock.unlock();
    attachment->fetches.reset();
}

void ObjectStoreSession::detach(const AttachmentIdentity& identity)
{
    std::lock_guard transition(transition_);
    if (const auto attachment = find(identity)) retire(attachment);
}

AttachmentIdentity ObjectStoreSession::attachLocation(const std::string& location, const OpenOptions& options)
{
    if (location.starts_with("http://") || location.starts_with("https://") || vcr::sftp::isSftpUrl(location)) {
        HostedSessionOptions hosted;
        hosted.author = options.author;
        return attachHosted(location, hosted);
    }
    return attach(location, options);
}

AttachmentIdentity ObjectStoreSession::attachHosted(const std::string& endpoint, const HostedSessionOptions& options)
{
    // Authentication may ask the GUI for input. Do not hold the attachment
    // barrier while waiting: that thread may need to finalize existing edits.
    auto next = std::make_shared<Attachment>(std::make_unique<detail::HostedAttachment>(endpoint, options));
    std::lock_guard transition(transition_);
    next->identity.storeUuid = next->remote->storeUuid();
    next->author = options.author;
    {
        std::lock_guard lock(mutex_);
        if (std::any_of(attached_.begin(), attached_.end(),
                        [&](const auto& a) { return a->identity.storeUuid == next->identity.storeUuid; }))
            throw std::runtime_error("object store is already attached to this session");
        next->identity.generation = ++generation_;
        next->journalPosition = next->remote->freshness().status->commitSequence;
        attached_.push_back(next);
        events_.push_back({StoreSessionEvent::Kind::Attached, next->identity, next->journalPosition, {}});
    }
    next->remote->start(next->identity, [this](StoreSessionEvent event) {
        std::lock_guard lock(mutex_);
        events_.push_back(std::move(event));
    });
    return next->identity;
}

bool ObjectStoreSession::hosted(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    return std::any_of(attached_.begin(), attached_.end(), [&](const auto& a) {
        return a->identity == identity && a->accepting && a->remote; });
}

std::optional<HostedFreshness> ObjectStoreSession::freshness(const AttachmentIdentity& identity) const
{
    std::shared_ptr<Attachment> attachment;
    {
        std::lock_guard lock(mutex_);
        for (const auto& a : attached_)
            if (a->identity == identity && a->accepting) attachment = a;
    }
    if (!attachment) throw std::runtime_error("object-store attachment is missing or superseded");
    return attachment->remote ? std::optional(attachment->remote->freshness()) : std::nullopt;
}

void ObjectStoreSession::refresh(const AttachmentIdentity& identity)
{
    withAttachment(identity, false, [](Attachment& attachment) {
        if (attachment.remote) attachment.remote->refresh();
    });
}

StoreStatus ObjectStoreSession::status(const AttachmentIdentity& identity)
{
    return withAttachment(identity, false, [](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->status();
        auto txn = attachment.store().read();
        return attachment.store().status(txn);
    });
}

PinnedObject pinOf(const AttachmentIdentity& attachment, const ObjectRevision& revision, const std::string& activeGeneration)
{
    return {attachment, {attachment.storeUuid, revision.objectUuid}, revision.generation, revision.version,
            revision.geometryRevision, activeGeneration};
}

TransactionRequest ObjectStoreSession::newRequest(const AttachmentIdentity& identity)
{
    const auto author = withAttachment(identity, false, [](Attachment& attachment) { return attachment.author; });
    if (author.empty()) throw std::invalid_argument("object store attachment has no author for its requests");
    const auto current = status(identity);
    TransactionRequest request;
    request.requestUuid = util::newUuid();
    request.storeUuid = current.storeUuid;
    request.generation = current.generation;
    request.author = author;
    return request;
}

TransactionReceipt ObjectStoreSession::submit(const AttachmentIdentity& identity, const TransactionRequest& request)
{
    return withAttachment(identity, true, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->submit(request);
        std::lock_guard writer(attachment.writer);
        return attachment.store().submit(request);
    });
}

std::optional<TransactionReceipt> ObjectStoreSession::receipt(const AttachmentIdentity& identity, const std::string& requestUuid)
{
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->client().receipt(requestUuid);
        auto txn = attachment.store().read();
        return attachment.store().receipt(txn, requestUuid);
    });
}

std::optional<ObjectRevision> ObjectStoreSession::currentObject(const AttachmentIdentity& identity, const std::string& objectUuid)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->currentObject(uuid);
        auto txn = attachment.store().read();
        return attachment.store().currentObjectRevision(txn, uuid);
    });
}

std::vector<std::optional<ObjectRevision>> ObjectStoreSession::objectsAtVersions(const AttachmentIdentity& identity,
                                                                                 const std::vector<ObjectVersion>& references)
{
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->objectsAtVersions(references);
        auto txn = attachment.store().read();
        return attachment.store().objectsAtVersions(txn, references);
    });
}

std::optional<GeometryRevision> ObjectStoreSession::geometryRevision(const AttachmentIdentity& identity,
                                                                     const std::string& revisionUuid)
{
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->geometryRevision(revisionUuid);
        auto txn = attachment.store().read();
        return attachment.store().geometryRevision(txn, revisionUuid);
    });
}

std::vector<uint8_t> ObjectStoreSession::readGeometryArray(const AttachmentIdentity& identity, const std::string& revisionUuid,
                                                           const std::string& arrayName)
{
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return *attachment.remote->geometryArray(revisionUuid, arrayName);
        auto txn = attachment.store().read();
        return attachment.store().readGeometryArray(txn, revisionUuid, arrayName);
    });
}

VolumeTransforms ObjectStoreSession::volumeTransforms(const AttachmentIdentity& identity)
{
    return withAttachment(identity, false, [](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->volumeTransforms();
        auto txn = attachment.store().read();
        return attachment.store().volumeTransforms(txn);
    });
}

std::vector<ArtifactFile> ObjectStoreSession::artifacts(const AttachmentIdentity& identity, const std::string& objectUuid,
                                                       const std::optional<int64_t>& version, const std::optional<std::string>& generation)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::optional<ObjectRevision> revision;
    if (version) {
        const auto status = this->status(identity);
        revision = objectsAtVersions(identity, {{status.storeUuid, generation.value_or(status.generation), uuid, *version}}).front();
    } else {
        revision = currentObject(identity, uuid);
    }
    if (!revision || revision->deleted) throw std::invalid_argument("object version not found: " + uuid);
    return artifactManifest(revision->record);
}

std::vector<ArtifactFile> ObjectStoreSession::materializeArtifacts(const AttachmentIdentity& identity, const std::string& objectUuid,
                                                                  const std::filesystem::path& destination,
                                                                  const std::optional<int64_t>& version, const std::optional<std::string>& generation)
{
    const auto files = artifacts(identity, objectUuid, version, generation);
    for (const auto& file : files) writeArtifact(identity, file, destination / std::filesystem::path(file.name));
    return files;
}

void ObjectStoreSession::writeArtifact(const AttachmentIdentity& identity, const ArtifactFile& file,
                                       const std::filesystem::path& target)
{
    withAttachment(identity, false, [&](Attachment& attachment) {
        std::filesystem::create_directories(target.parent_path());
        if (attachment.remote) { attachment.remote->writeArtifact(file, target); return 0; }
        auto& store = attachment.store();
        detail::writeArtifactFile({file.name, file.sha256, file.length}, target, uint64_t(32) << 20,
                                  [&](uint64_t offset, uint64_t count) { return store.readArtifact(file.sha256, offset, count); });
        return 0;
    });
}

std::optional<std::filesystem::path> ObjectStoreSession::artifactFile(const AttachmentIdentity& identity,
                                                                      const std::string& objectUuid, const std::string& name,
                                                                      const std::filesystem::path& cacheDirectory,
                                                                      const std::optional<int64_t>& version, const std::optional<std::string>& generation)
{
    const auto files = artifacts(identity, objectUuid, version, generation);
    const auto file = std::find_if(files.begin(), files.end(), [&](const auto& f) { return f.name == name; });
    if (file == files.end()) return std::nullopt;
    // Content-addressed: a complete file at its path is that content (writes are atomic and verified).
    const auto target = cacheDirectory / file->sha256 / std::filesystem::path(file->name).filename();
    if (!detail::artifactFileMatches({file->name, file->sha256, file->length}, target))
        writeArtifact(identity, *file, target);
    return target;
}

TransactionReceipt ObjectStoreSession::publishArtifacts(const AttachmentIdentity& identity, const std::string& objectUuid,
                                                        int64_t expectedVersion,
                                                        const std::vector<std::pair<std::string, std::filesystem::path>>& files,
                                                        const std::string& author)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    TransactionRequest request;
    if (author.empty()) {
        request = newRequest(identity);
    } else {
        const auto status = this->status(identity);
        request.requestUuid = util::newUuid();
        request.storeUuid = status.storeUuid;
        request.generation = status.generation;
        request.author = author;
    }
    // Read the manifest from the expected head, whose publication generation
    // can predate the active history. The request still guards both the head
    // version and the active generation across the uploads below.
    const auto head = currentObject(identity, uuid);
    const auto actual = head && !head->deleted ? std::optional(head->version) : std::nullopt;
    if (actual != expectedVersion)
        throw TransactionConflict(TransactionConflict::Reason::Version, {{"object", uuid, expectedVersion, actual}},
                                  "artifact publication: expected version changed for " + uuid);
    std::map<std::string, ArtifactFile> manifest;
    for (auto& file : artifactManifest(head->record)) manifest[file.name] = std::move(file);
    // Bytes first: the request may only name what the store holds.
    withAttachment(identity, true, [&](Attachment& attachment) {
        for (const auto& [name, path] : files) {
            const auto hash = attachment.remote ? attachment.remote->client().putArtifact(path) : attachment.store().putArtifact(path);
            manifest[name] = ArtifactFile{name, hash, uint64_t(std::filesystem::file_size(path))};
        }
        return 0;
    });
    std::vector<ArtifactFile> listed;
    for (auto& [name, file] : manifest) listed.push_back(std::move(file));
    request.expectedObjects = {{uuid, expectedVersion}};
    request.operations = {SetMetaOp{uuid, kArtifactsMetaKey, artifactManifestJson(std::move(listed))}};
    return submit(identity, request);
}

std::set<std::string> ObjectStoreSession::holds(const AttachmentIdentity& identity, const std::set<std::string>& uuids)
{
    if (uuids.empty()) return {};
    return withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) return attachment.remote->holds(uuids);
        std::set<std::string> held;
        auto txn = attachment.store().read();
        for (const auto& uuid : uuids)
            if (attachment.store().objectByUuid(txn, uuid)) held.insert(uuid);
        return held;
    });
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

std::shared_ptr<ObjectPrefetchJob> ObjectStoreSession::objectPrefetch(const AttachmentIdentity& identity) const
{
    std::lock_guard lock(mutex_);
    for (const auto& attachment : attached_)
        if (attachment->identity == identity) return attachment->prefetch;
    return {};
}

std::shared_ptr<ObjectPrefetchJob> ObjectStoreSession::prefetchObjects(const AttachmentIdentity& identity)
{
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(attached_.begin(), attached_.end(), [&](const auto& attachment) {
        return attachment->identity == identity && attachment->accepting;
    });
    if (found == attached_.end() || !(*found)->remote) throw std::runtime_error("Hosted store is unavailable");
    auto attachment = *found;
    if (attachment->prefetch && !attachment->prefetch->snapshot().done) return attachment->prefetch;
    auto job = std::make_shared<ObjectPrefetchJob>();
    attachment->prefetch = job;
    // Owning the attachment keeps transport alive without admitting a long-lived
    // session reader. Detachment cancels without waiting for this transfer.
    std::thread([attachment, job] { attachment->remote->prefetch(*job, attachment->cancelled); }).detach();
    return job;
}

void ObjectStoreSession::readStore(const AttachmentIdentity& identity, const StoreRead& callback)
{
    withAttachment(identity, false, [&](Attachment& attachment) {
        if (attachment.remote) {
            const detail::HostedStoreReader reader(*attachment.remote);
            callback(reader, attachment.cancelled);
            return;
        }
        auto txn = attachment.store().read();
        const LocalStoreReader reader(attachment.store(), txn);
        callback(reader, attachment.cancelled);
    });
}

std::string ObjectStoreSession::copyObject(const AttachmentIdentity& from, const AttachmentIdentity& to,
                                           const std::string& objectUuid, const std::string& name)
{
    std::optional<ObjectCopy> copy;
    readStore(from, [&](const StoreReader& reader, const auto&) { copy = readObjectCopy(reader, objectUuid); });
    if (!name.empty()) copy->record.name = name;
    submitCopy(from, to, *copy);
    return copy->record.uuid;
}

void ObjectStoreSession::importObject(const AttachmentIdentity& from, const AttachmentIdentity& to,
                                      const std::string& objectUuid)
{
    ensureGeometry(from, objectUuid);
    std::optional<ObjectCopy> copy;
    readStore(from, [&](const StoreReader& reader, const auto&) {
        copy = readObjectCopy(reader, objectUuid, util::canonicalUuid(objectUuid));
    });
    submitCopy(from, to, *copy);
}

void ObjectStoreSession::submitCopy(const AttachmentIdentity& from, const AttachmentIdentity& to, const ObjectCopy& copy)
{
    auto request = newRequest(to);
    request.expectedObjects[copy.record.uuid] = std::nullopt;
    request.operations = objectCopyOperations(copy);
    // The manifest belongs to the version read above. Transfer its immutable
    // blobs before publishing the copy, without rereading the source's head.
    const auto files = artifactManifest(copy.record);
    if (from != to && !files.empty()) {
        withAttachment(to, true, [&](Attachment& destination) {
            struct Staging {
                std::filesystem::path path;
                ~Staging() {
                    if (!path.empty()) { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
                }
            } staging;
            for (const auto& file : files) {
                const auto held = destination.remote ? destination.remote->client().artifactLength(file.sha256)
                                                     : destination.store().artifactLength(file.sha256);
                if (held == file.length) continue;
                if (staging.path.empty())
                    staging.path = vcr::platform::make_temp_directory(
                        (std::filesystem::temp_directory_path() / "vc-object-copy-XXXXXX").string());
                const auto path = staging.path / file.sha256;
                writeArtifact(from, file, path);
                const auto hash = destination.remote ? destination.remote->client().putArtifact(path)
                                                     : destination.store().putArtifact(path, true);
                if (hash != file.sha256) throw std::runtime_error("copied artifact hash mismatch: " + file.name);
                std::filesystem::remove(path);
            }
        });
    }
    submit(to, request);
}

std::optional<TransactionReceipt> ObjectStoreSession::editCurrent(const AttachmentIdentity& identity,
                                                                  const std::vector<std::string>& objectUuids,
                                                                  const EditOperations& operations)
{
    if (objectUuids.empty()) return std::nullopt;
    auto request = newRequest(identity);
    readStore(identity, [&](const StoreReader& reader, const auto&) {
        // Guard the versions before callbacks read or compute their edits. A
        // hosted poll during a callback must not advance these expectations.
        editCurrentRequest(reader, request, objectUuids, {});
        std::vector<TransactionOperation> edits;
        for (const auto& objectUuid : objectUuids)
            for (auto& operation : operations(reader, util::canonicalUuid(objectUuid))) edits.push_back(std::move(operation));
        // Relationship decisions still run after the edits they describe.
        for (auto& operation : request.operations) edits.push_back(std::move(operation));
        request.operations = std::move(edits);
    });
    return submit(identity, request);
}

std::optional<TransactionReceipt> ObjectStoreSession::editRecords(const AttachmentIdentity& identity,
                                                                  const std::vector<std::string>& objectUuids,
                                                                  const std::function<void(ObjectRecord&)>& change)
{
    return editCurrent(identity, objectUuids, [&](const StoreReader& reader, const std::string& uuid) {
        const auto current = reader.currentObject(uuid);
        if (!current) throw std::runtime_error("object is not in the store: " + uuid);
        auto record = current->record;
        change(record);
        return std::vector<TransactionOperation>{UpdateObjectOp{objectRecordJson(record)}};
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

void ObjectStoreSession::setGeometryFetcher(const std::string& metaKey, GeometryFetcher fetch)
{
    std::lock_guard lock(mutex_);
    if (fetch) fetchers_[metaKey] = std::move(fetch);
    else fetchers_.erase(metaKey);
}

void ObjectStoreSession::startFetching(Attachment& attachment)
{
    if (!attachment.fetches) return;
    attachment.fetches->start([this, identity = attachment.identity](const std::string& uuid) {
        auto held = acquire(identity, false);
        try {
            nlohmann::json meta = nlohmann::json::object();
            {
                auto txn = held->store().read();
                if (const auto record = held->store().objectByUuid(txn, uuid)) meta = nlohmann::json::parse(record->meta);
            }
            GeometryFetcher fetch;
            {
                std::lock_guard lock(mutex_);
                for (const auto& [key, fetcher] : fetchers_)
                    if (!fetch && meta.is_object() && meta.contains(key)) fetch = fetcher;
            }
            if (!fetch) throw std::runtime_error("no geometry fetcher for " + uuid);
            fetch(held->store(), uuid, [&](const VersionSelection& expected, const Action& action) {
                return fetchWrite(identity, expected, action);
            });
        } catch (...) { release(*held, false); throw; }
        release(*held, false);
    });
}

TransactionReceipt ObjectStoreSession::fetchWrite(const AttachmentIdentity& identity, const VersionSelection& expected,
                                                  const Action& callback)
{
    TransactionReceipt committed;
    auto attachment = acquire(identity, false);
    try {
        if (!attachment->fetches) throw std::runtime_error("Object store is attached read-only");
        std::lock_guard writer(attachment->writer);
        auto txn = attachment->store().write(expected);
        callback(attachment->store(), txn);
        txn.commit();
        committed = *txn.receipt();
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
    if (attachment->remote) { release(*attachment, false); return; } // its geometry is on the host
    try {
        bool local = false;
        {
            auto txn = attachment->store().read();
            local = attachment->store().currentGeometry(txn, uuid).has_value();
        }
        if (!local) {
            if (!attachment->fetches) throw GeometryUnavailable("object has no geometry and its store is attached read-only: " + uuid);
            std::string error;
            if (!attachment->fetches->fetchNow(uuid, [&] { return attachment->cancelled.load() || (cancel && cancel()); },
                                               &error, [&] {
                                                   auto txn = attachment->store().read();
                                                   return attachment->store().currentGeometry(txn, uuid).has_value();
                                               }))
                throw GeometryUnavailable(error.empty() ? "fetching " + uuid + " was cancelled"
                                                        : "cannot fetch " + uuid + ": " + error);
        }
    } catch (...) { release(*attachment, false); throw; }
    release(*attachment, false);
}

std::set<std::string> ObjectStoreSession::heldBy(const std::vector<AttachmentIdentity>& attachments,
                                                 const std::set<std::string>& uuids)
{
    std::set<std::string> held;
    for (const auto& attachment : attachments) {
        std::set<std::string> rest;
        std::set_difference(uuids.begin(), uuids.end(), held.begin(), held.end(), std::inserter(rest, rest.end()));
        held.merge(holds(attachment, rest));
    }
    return held;
}

std::vector<AttachedRecord> ObjectStoreSession::selectObjects(const std::vector<AttachmentIdentity>& order,
                                                              const ObjectSelection& selection)
{
    std::vector<AttachedRecord> result;
    std::vector<AttachmentIdentity> earlier;
    for (const auto& attachment : order) {
        auto page = selection;
        int64_t kept = 0;
        for (;;) {
            const auto records = withAttachment(attachment, false, [&](Attachment& held) {
                if (held.remote) return held.remote->selectObjects(page);
                auto txn = held.store().read();
                return held.store().selectObjects(txn, page);
            });
            std::set<std::string> uuids;
            for (const auto& record : records) uuids.insert(record.uuid);
            const auto hidden = heldBy(earlier, uuids);
            for (const auto& record : records) {
                if (hidden.contains(record.uuid) || (selection.limit && kept == *selection.limit)) continue;
                result.push_back({attachment, record});
                ++kept;
            }
            // A full page that lost records to earlier attachments: read on.
            if (!selection.limit || kept == *selection.limit || int64_t(records.size()) < *selection.limit) break;
            page.offset += int64_t(records.size());
        }
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
                                  const CoordinateJson& coordinates, const std::optional<AttachmentIdentity>& attachment,
                                  const std::optional<ObjectVersion>& base)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    auto draft = std::const_pointer_cast<detail::SurfaceDraft>(detail::makeSurfaceDraft(surface, coordinates, attachment));
    std::lock_guard lock(mutex_);
    draft->revision = ++draftRevision_;
    drafts_[uuid] = std::move(draft);
    auto& meta = durable_->meta[uuid];
    meta.storeUuid = attachment ? std::optional(attachment->storeUuid) : std::nullopt;
    if (base) { meta.base = base; meta.conflict.clear(); }
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
    auto edited = std::const_pointer_cast<detail::SurfaceDraft>(
        detail::editSurfaceDraft(*current, row, column, rows, columns, points, validity));
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(uuid);
    if (found == drafts_.end() || found->second != current)
        throw std::logic_error("surface draft " + uuid + " changed during an edit");
    edited->revision = ++draftRevision_;
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
    durable_->meta.erase(uuid);
    draftChanged(uuid, attachment);
}

bool ObjectStoreSession::hasDraft(const std::string& objectUuid) const
{
    return draftRevision(objectUuid) != 0;
}

uint64_t ObjectStoreSession::draftRevision(const std::string& objectUuid) const
{
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(util::canonicalUuid(objectUuid));
    return found == drafts_.end() ? 0 : found->second->revision;
}

std::optional<DraftState> ObjectStoreSession::draftState(const std::string& objectUuid) const
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(uuid);
    if (found == drafts_.end()) return std::nullopt;
    DraftState state;
    state.revision = found->second->revision;
    if (const auto meta = durable_->meta.find(uuid); meta != durable_->meta.end()) {
        state.base = meta->second.base;
        state.submittedRevision = meta->second.submittedRevision;
        state.pendingRequest = meta->second.pendingRequest;
        state.conflict = meta->second.conflict;
        state.saved = meta->second.savedRevision == state.revision;
    }
    const auto written = durable_->checkpointed.find(uuid);
    state.checkpointed = state.saved || (written != durable_->checkpointed.end() && written->second == state.revision);
    return state;
}

std::optional<DraftContents> ObjectStoreSession::draftContents(const std::string& objectUuid) const
{
    std::shared_ptr<const detail::SurfaceDraft> draft;
    {
        std::lock_guard lock(mutex_);
        const auto found = drafts_.find(util::canonicalUuid(objectUuid));
        if (found == drafts_.end()) return std::nullopt;
        draft = found->second;
    }
    return DraftContents{detail::draftGrid(*draft), draft->coordinates, draft->attachment, draft->revision};
}

void ObjectStoreSession::markDraftSaved(const std::string& objectUuid, uint64_t revision)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::lock_guard lock(mutex_);
    const auto found = drafts_.find(uuid);
    if (found == drafts_.end() || found->second->revision != revision) return;
    durable_->meta[uuid].savedRevision = revision;
}

DraftRecovery ObjectStoreSession::useDraftDirectory(const std::filesystem::path& path)
{
    std::lock_guard writing(durable_->writing);
    {
        std::lock_guard lock(mutex_);
        if (durable_->directory) throw std::logic_error("the session already uses a draft directory");
        if (!drafts_.empty()) throw std::logic_error("choose the draft directory before making drafts");
    }
    auto directory = std::make_unique<detail::DraftDirectory>(path);
    auto loaded = directory->load();
    DraftRecovery recovery;
    recovery.checkpoint = loaded.sequence;
    std::vector<std::pair<std::string, std::shared_ptr<detail::SurfaceDraft>>> installed;
    for (const auto& draft : loaded.drafts) {
        std::optional<AttachmentIdentity> attachment;
        {
            std::lock_guard lock(mutex_);
            for (const auto& a : attached_)
                if (a->accepting && draft.meta.storeUuid && a->identity.storeUuid == *draft.meta.storeUuid) attachment = a->identity;
        }
        installed.emplace_back(draft.objectUuid, std::const_pointer_cast<detail::SurfaceDraft>(
            detail::makeSurfaceDraft(draft.surface, draft.coordinates, attachment)));
    }
    std::lock_guard lock(mutex_);
    for (std::size_t i = 0; i < installed.size(); ++i) {
        auto& [uuid, draft] = installed[i];
        draft->revision = ++draftRevision_;
        directory->adopt(uuid, draft->revision);
        durable_->checkpointed[uuid] = draft->revision;
        auto meta = loaded.drafts[i].meta;
        // Revisions are per session: a pending save of exactly this grid now
        // names its new revision; one edited after the save names none.
        if (meta.submittedRevision) {
            meta.submittedRevision = loaded.drafts[i].submittedAsStored ? draft->revision : 0;
            for (auto& pending : loaded.pending)
                if (pending.drafts.contains(uuid)) pending.drafts[uuid] = *meta.submittedRevision;
        }
        durable_->meta[uuid] = std::move(meta);
        const auto attachment = draft->attachment;
        drafts_[uuid] = std::move(draft);
        draftChanged(uuid, attachment);
        recovery.drafts.push_back(uuid);
    }
    for (const auto& pending : loaded.pending) {
        durable_->pending[pending.requestUuid] = pending;
        PendingSubmission listed{pending.requestUuid, pending.storeUuid, {}};
        for (const auto& [uuid, revision] : pending.drafts) listed.drafts.push_back(uuid);
        recovery.pending.push_back(std::move(listed));
    }
    durable_->directory = std::move(directory);
    return recovery;
}

uint64_t ObjectStoreSession::checkpointDrafts()
{
    std::lock_guard writing(durable_->writing);
    std::vector<detail::DraftDirectory::Capture> captures;
    std::vector<detail::PendingRecord> pending;
    {
        std::lock_guard lock(mutex_);
        if (!durable_->directory) throw std::logic_error("the session has no draft directory");
        for (const auto& [uuid, draft] : drafts_) {
            const auto meta = durable_->meta.find(uuid);
            // A saved draft has nothing to recover (unless a save of it is pending).
            if (meta != durable_->meta.end() && meta->second.savedRevision == draft->revision &&
                !meta->second.submittedRevision)
                continue;
            captures.push_back({uuid, draft, meta == durable_->meta.end() ? detail::DraftMeta{} : meta->second});
        }
        for (const auto& [request, record] : durable_->pending) pending.push_back(record);
    }
    uint64_t sequence = 0;
    try {
        sequence = durable_->directory->write(captures, pending);
    } catch (const std::exception& error) {
        std::lock_guard lock(mutex_);
        durable_->checkpointError = error.what();
        throw;
    }
    std::lock_guard lock(mutex_);
    durable_->checkpointError.clear();
    durable_->checkpointed.clear();
    for (const auto& capture : captures) durable_->checkpointed[capture.objectUuid] = capture.draft->revision;
    return sequence;
}

std::string ObjectStoreSession::draftCheckpointError() const
{
    std::lock_guard lock(mutex_);
    return durable_->checkpointError;
}

std::vector<PendingSubmission> ObjectStoreSession::pendingSubmissions() const
{
    std::lock_guard lock(mutex_);
    std::vector<PendingSubmission> result;
    for (const auto& [request, record] : durable_->pending) {
        PendingSubmission listed{request, record.storeUuid, {}};
        for (const auto& [uuid, revision] : record.drafts) listed.drafts.push_back(uuid);
        result.push_back(std::move(listed));
    }
    return result;
}

TransactionReceipt ObjectStoreSession::submitDurably(const AttachmentIdentity& attachment, const TransactionRequest& request,
                                                     const std::vector<std::string>& drafts)
{
    DurableSave save;
    for (const auto& uuid : drafts) save.drafts.push_back({uuid, 0});
    return submitDurably(attachment, request, save);
}

TransactionReceipt ObjectStoreSession::submitDurably(const AttachmentIdentity& attachment, const TransactionRequest& request,
                                                     const DurableSave& save)
{
    if (request.requestUuid.empty()) throw std::invalid_argument("a durable submission needs a request UUID");
    const auto requestUuid = util::canonicalUuid(request.requestUuid);
    bool durable = false;
    {
        std::lock_guard lock(mutex_);
        if (durable_->pending.contains(requestUuid)) throw std::invalid_argument("request " + requestUuid + " is already pending");
        detail::PendingRecord record{requestUuid, attachment.storeUuid, {}, save.keepDrafts};
        for (const auto& saved : save.drafts) {
            const auto uuid = util::canonicalUuid(saved.objectUuid);
            const auto found = drafts_.find(uuid);
            if (found == drafts_.end()) throw std::invalid_argument("no draft of " + uuid);
            if (saved.revision > found->second->revision)
                throw std::invalid_argument("draft " + uuid + " has no revision " + std::to_string(saved.revision));
            record.drafts[uuid] = saved.revision ? saved.revision : found->second->revision;
        }
        for (const auto& [uuid, revision] : record.drafts) {
            auto& meta = durable_->meta[uuid];
            meta.submittedRevision = revision;
            meta.pendingRequest = requestUuid;
        }
        durable_->pending[requestUuid] = std::move(record);
        durable = durable_->directory != nullptr;
    }
    if (durable) {
        try {
            {
                std::lock_guard writing(durable_->writing);
                TransactionPayloads payloads;
                const auto canonical = transactionRequestToJson(request, &payloads);
                durable_->directory->storeRequest(requestUuid, canonical, payloads);
            }
            checkpointDrafts();
        } catch (...) {
            conclude(requestUuid, nullptr, "");
            throw;
        }
    }
    return sendPending(attachment, request, requestUuid);
}

TransactionReceipt ObjectStoreSession::resolvePending(const AttachmentIdentity& attachment, const std::string& requestUuid)
{
    const auto uuid = util::canonicalUuid(requestUuid);
    {
        std::lock_guard lock(mutex_);
        const auto found = durable_->pending.find(uuid);
        if (found == durable_->pending.end()) throw std::invalid_argument("no pending submission " + uuid);
        if (found->second.storeUuid != attachment.storeUuid)
            throw std::invalid_argument("pending submission " + uuid + " belongs to store " + found->second.storeUuid);
    }
    if (const auto saved = receipt(attachment, uuid)) {
        conclude(uuid, &*saved, "");
        return *saved;
    }
    TransactionRequest request;
    {
        std::lock_guard writing(durable_->writing);
        if (!durable_->directory) throw std::runtime_error("pending submission " + uuid + " has no stored request");
        request = durable_->directory->loadRequest(uuid);
    }
    return sendPending(attachment, request, uuid);
}

TransactionReceipt ObjectStoreSession::sendPending(const AttachmentIdentity& attachment, const TransactionRequest& request,
                                                   const std::string& requestUuid)
{
    TransactionReceipt result;
    try {
        result = submit(attachment, request);
    } catch (const TransactionConflict& conflict) {
        conclude(requestUuid, nullptr, conflict.what());
        throw;
    } catch (const std::invalid_argument&) {
        conclude(requestUuid, nullptr, "");
        throw;
    } catch (...) {
        // A hosted store that could not be reached: unknown outcome, still
        // pending. A local write that failed committed nothing.
        if (!hosted(attachment)) conclude(requestUuid, nullptr, "");
        throw;
    }
    conclude(requestUuid, &result, "");
    return result;
}

void ObjectStoreSession::conclude(const std::string& requestUuid, const TransactionReceipt* receipt, const std::string& conflict)
{
    bool durable = false;
    {
        std::lock_guard lock(mutex_);
        const auto found = durable_->pending.find(requestUuid);
        if (found == durable_->pending.end()) return;
        const auto record = found->second;
        durable_->pending.erase(found);
        for (const auto& [uuid, submitted] : record.drafts) {
            const auto meta = durable_->meta.find(uuid);
            const auto draft = drafts_.find(uuid);
            if (meta == durable_->meta.end() || draft == drafts_.end()) continue; // discarded meanwhile
            if (meta->second.pendingRequest != requestUuid) continue;
            meta->second.submittedRevision.reset();
            meta->second.pendingRequest.reset();
            if (!receipt) {
                if (!conflict.empty()) meta->second.conflict = conflict;
                continue;
            }
            // Committed: clear what was saved (or, kept for an open editor,
            // mark it saved); later edits stay on the new base.
            const bool unchanged = draft->second->revision == submitted;
            if (unchanged && !record.keepDrafts) {
                const auto attachment = draft->second->attachment;
                drafts_.erase(draft);
                durable_->meta.erase(meta);
                draftChanged(uuid, attachment);
                continue;
            }
            if (const auto version = receipt->objects.find(uuid); version != receipt->objects.end() && version->second) {
                meta->second.base = ObjectVersion{receipt->storeUuid, receipt->generation, uuid, *version->second};
                meta->second.conflict.clear();
            }
            if (unchanged) meta->second.savedRevision = submitted;
        }
        durable = durable_->directory != nullptr;
    }
    // Record the outcome; a failure here is reported, not thrown: the
    // submission's own outcome is already decided.
    if (durable) try { checkpointDrafts(); } catch (const std::exception&) {}
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
        auto transforms = std::make_shared<VolumeTransforms>(session_.volumeTransforms(*source));
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
// A volume target places a draft from another volume through registrations.
bool needsTransforms(const detail::SurfaceDraft& draft, const std::optional<std::string>& volume)
{
    const auto& source = draft.coordinates.at("volume_uuid");
    return volume && !(source.is_string() && source.get<std::string>() == *volume);
}
// Every object a query asks for is drafted: the stores have nothing to add.
template <class Drafts>
bool onlyDrafts(const std::optional<std::vector<std::string>>& uuids, const Drafts& drafts)
{
    return uuids && std::all_of(uuids->begin(), uuids->end(),
                                [&](const auto& u) { return drafts.contains(util::canonicalUuid(u)); });
}
}

std::vector<AttachedRegion> ObjectStoreSession::geometryRegion(const std::vector<AttachmentIdentity>& order,
    const RegionSlice& slice, const RegionOptions& options, const RegionCancel& cancel)
{
    const auto drafts = options.kind == "surface" ? this->drafts() : Drafts{};
    std::vector<AttachedRegion> result;
    std::vector<AttachmentIdentity> earlier;
    const bool stored = !(options.kind == "surface" && onlyDrafts(options.uuids, drafts));
    for (const auto& attachment : stored ? order : std::vector<AttachmentIdentity>{}) {
        // A hosted store answers the whole query on the host (no cancellation).
        auto region = withAttachment(attachment, false, [&](Attachment& held) {
            if (held.remote) return held.remote->client().geometryRegion(slice, options);
            auto txn = held.store().read();
            return held.store().geometryRegion(txn, slice, options, cancel);
        });
        std::set<std::string> uuids(region.remote.begin(), region.remote.end());
        for (const auto& run : region.runs) uuids.insert(run.objectUuid);
        const auto hidden = heldBy(earlier, uuids);
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
                                                  options.toTarget,
                                                  needsTransforms(*draft, options.volume) ? transforms(*draft) : nullptr);
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
    for (const auto& attachment : onlyDrafts(query.uuids, drafts) ? std::vector<AttachmentIdentity>{} : order) {
        std::vector<std::vector<SurfaceProximityHit>> hits(points.size());
        withAttachment(attachment, false, [&](Attachment& held) {
            auto around = query;
            if (held.remote) {
                // The host resolves volume targets through its recorded transforms.
                for (std::size_t i = 0; i < points.size(); ++i) {
                    around.point = points[i];
                    hits[i] = held.remote->client().nearestSurfaces(around);
                }
                return;
            }
            auto txn = held.store().read();
            if (around.volume && !around.transforms)
                around.transforms = std::make_shared<VolumeTransforms>(held.store().volumeTransforms(txn));
            for (std::size_t i = 0; i < points.size(); ++i) {
                around.point = points[i];
                hits[i] = held.store().nearestSurfaces(txn, around);
            }
        });
        std::set<std::string> uuids;
        for (const auto& near : hits)
            for (const auto& hit : near) uuids.insert(hit.objectUuid);
        const auto hidden = heldBy(earlier, uuids);
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
                                                  query.toTarget,
                                                  needsTransforms(*draft, query.volume) ? transforms(*draft) : nullptr);
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

std::string ObjectStoreSession::location(const AttachmentIdentity& identity)
{
    return withAttachment(identity, false, [](Attachment& attachment) {
        return attachment.remote ? attachment.remote->client().endpoint() : attachment.store().path().string();
    });
}

std::filesystem::path ObjectStoreSession::path(const AttachmentIdentity& identity)
{
    return withAttachment(identity, false, [](Attachment& attachment) { return attachment.store().path(); });
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
    if (attachment.remote) return; // its poller delivers its commits
    std::lock_guard journal(attachment.journal);
    auto txn = attachment.store().read();
    const auto state = attachment.store().status(txn);
    const bool changedLineage = attachment.observedState &&
        (state.generation != attachment.observedState->generation || state.historyEnabled != attachment.observedState->historyEnabled);
    if (!state.historyEnabled || changedLineage) {
        if (attachment.observedState && state.generation == attachment.observedState->generation &&
            state.commitSequence == attachment.observedState->commitSequence) return;
        std::lock_guard lock(mutex_);
        events_.push_back({StoreSessionEvent::Kind::Committed, attachment.identity, state.commitSequence, {}, true});
        attachment.observedState = state;
        attachment.journalPosition = state.historyEnabled ? attachment.store().journalPosition(txn) : state.commitSequence;
        return;
    }
    auto changes = attachment.store().journalSince(txn, attachment.journalPosition);
    if (changes.empty()) {
        if(attachment.observedState && attachment.observedState->commitSequence != state.commitSequence) {
            std::lock_guard lock(mutex_);
            events_.push_back({StoreSessionEvent::Kind::Committed, attachment.identity, state.commitSequence, {}, true});
        }
        attachment.observedState = state;
        return;
    }
    const auto position = changes.back().seq;
    std::lock_guard lock(mutex_);
    events_.push_back({StoreSessionEvent::Kind::Committed, attachment.identity, position, std::move(changes)});
    attachment.journalPosition = position;
    attachment.observedState = state;
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
    if ((kind != Kind::Committed && kind != Kind::Draft) || revalidate) return true;
    return std::any_of(changes.begin(), changes.end(), [&](const auto& change) {
        return change.kind.empty() || std::find(kinds.begin(), kinds.end(), change.kind) != kinds.end();
    });
}

void ObjectStoreSession::requireCurrent(const PinnedObject& pin)
{
    readStore(pin.attachment, [&](const StoreReader& reader, const auto&) {
        const auto head = reader.currentObject(pin.object.objectUuid);
        if (reader.status().generation != pin.writeGeneration ||
            !head || head->version != pin.version || head->generation != pin.generation)
            throw std::runtime_error("It changed in the object store since it was opened; reopen it to edit");
    });
}
}
