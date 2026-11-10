#include "HostedSession.hpp"
#include "geometry/ArrayLayout.hpp"
#include "internal/Artifacts.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/native/disk_cache.hpp>
#include <fstream>

#include <algorithm>
#include <cstring>

#include <chrono>

namespace vc::objects::detail {
namespace {
int64_t nowUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Cache cost of a record: its metadata and fixed overheads.
std::size_t cost(const ObjectRevision& revision) { return revision.record.meta.size() + 1024; }
std::size_t cost(const GeometryRevision& geometry) { return geometry.attributes.size() + 512 + 64 * geometry.arrays.size(); }

std::string headKey(const std::string& object) { return "head/" + object; }
// Conversion preserves logical identities, but schema-20 cached JSON has the
// old generation fields. Version metadata separately; raw payload keys stay
// unchanged so a schema upgrade never duplicates geometry/artifact blocks.
std::string metadataKey(const std::string& key) { return "schema21/" + key; }
std::string versionKey(const ObjectVersion& v) { return metadataKey("version/" + v.storeUuid + "/" + v.generation + "/" + v.objectUuid + "/" + std::to_string(v.version)); }
bool isTransformKind(const std::string& kind)
{
    return kind.empty() || kind == "volume_transform" || kind == "volume_transform_choice";
}
} // namespace

HostedAttachment::HostedAttachment(const std::string& endpoint, const HostedSessionOptions& options)
    : client_(endpoint, options.client), interval_(options.pollInterval), cache_(options.cacheBytes)
{
    const auto status = client_.status();
    storeUuid_ = status.storeUuid;
    cursor_ = FeedCursor{status.storeUuid, status.generation, status.commitSequence, 0};
    freshness_.status = status;
    freshness_.checkedAtUs = nowUs();
    if (!options.cacheDirectory.empty()) {
        try {
            disk_ = vcr::zarr::diskBudget(options.cacheDirectory.string(), options.diskCacheBytes);
            disk_->set_min_free_bytes(options.minFreeBytes);
            diskRoot_ = std::filesystem::weakly_canonical(options.cacheDirectory) / "objects";
        } catch (const std::exception&) {
            // Interactive reads remain available when the cache cannot be opened.
            disk_.reset();
            diskRoot_.clear();
        }
    }
}

namespace {
constexpr int64_t payloadBlockBytes = 1 << 20;
struct PrefetchAudit {
    std::map<std::filesystem::path, std::pair<size_t, std::string>> entries;
    uint64_t downloaded = 0;
};
thread_local PrefetchAudit* prefetchAudit = nullptr;
std::vector<uint8_t> bytesOf(const std::string& value) { return {value.begin(), value.end()}; }
std::string textOf(const std::vector<uint8_t>& value) { return {value.begin(), value.end()}; }
}

std::string HostedAttachment::payloadKey(const std::string& key) const
{
    std::lock_guard lock(stateMutex_);
    return storeUuid_ + "/" + freshness_.status->generation + "/" + key;
}

std::optional<std::vector<uint8_t>> HostedAttachment::readCachedPayload(const std::string& key, std::optional<size_t> length)
{
    if (!disk_) return std::nullopt;
    const auto path = diskRoot_ / payloadHash(bytesOf(key));
    // One atomic envelope carries its checksum and payload. No budget lock is
    // needed to read; a concurrently evicted file remains open until this ends.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return std::nullopt;
    const auto size = file.tellg();
    if (size < 64 || (length && uint64_t(size) != *length + 64)) return std::nullopt;
    file.seekg(0);
    std::string hash(64, '\0');
    file.read(hash.data(), 64);
    // Large metadata is valid, but verify its checksum before allocating from
    // an untrusted on-disk length. Geometry/artifact blocks have fixed bounds.
    if (!length && uint64_t(size) > (uint64_t(64) << 20)) {
        PayloadHasher hasher;
        std::array<char, 64 * 1024> buffer;
        while (file) {
            file.read(buffer.data(), buffer.size());
            hasher.update(buffer.data(), size_t(file.gcount()));
        }
        if (hasher.finish() != hash) return std::nullopt;
        file.clear();
        file.seekg(64);
    }
    std::vector<uint8_t> bytes(size_t(size) - 64);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file || payloadHash(bytes) != hash) return std::nullopt;
    if (prefetchAudit) prefetchAudit->entries[path] = {bytes.size() + 64, payloadHash(bytes)};
    return bytes;
}

void HostedAttachment::publishPayload(const std::string& key, const std::vector<uint8_t>& bytes)
{
    if (!disk_ || bytes == bytesOf("null")) return;
    try {
        auto envelope = bytesOf(payloadHash(bytes));
        envelope.insert(envelope.end(), bytes.begin(), bytes.end());
        const auto path = diskRoot_ / payloadHash(bytesOf(key));
        disk_->write(path.string(), envelope.data(), envelope.size(), true);
    } catch (const std::exception&) { /* Interactive reads survive cache write failures. */ }
}

std::vector<uint8_t> HostedAttachment::cachedPayload(const std::string& key,
    const std::function<std::vector<uint8_t>()>& fetch, std::optional<size_t> length)
{
    struct InteractiveTurn {
        HostedAttachment& owner;
        bool interactive = !prefetchAudit;
        explicit InteractiveTurn(HostedAttachment& value) : owner(value) { if (interactive) ++owner.interactivePayloads_; }
        ~InteractiveTurn() { if (interactive) { --owner.interactivePayloads_; owner.priorityWake_.notify_all(); } }
    } turn(*this);
    if (!disk_) return fetch();
    const auto memoryKey = "payload/" + key;
    if (length && !prefetchAudit) {
        std::shared_ptr<const std::vector<uint8_t>> cached;
        {
            std::lock_guard lock(cacheMutex_);
            if (const auto* entry = cache_.get(memoryKey); entry) cached = entry->bytes;
        }
        if (cached) return *cached;
    }
    const auto remember = [&](const std::vector<uint8_t>& bytes) {
        if (length && !prefetchAudit) {
            auto shared = std::make_shared<const std::vector<uint8_t>>(bytes);
            std::lock_guard lock(cacheMutex_);
            cache_.put(memoryKey, Entry{std::nullopt, std::nullopt, shared}, shared->size() + 128);
        }
    };
    const auto path = diskRoot_ / payloadHash(bytesOf(key));
    const auto read = [&] { return readCachedPayload(key, length); };
    const auto audit = [&](const std::vector<uint8_t>& bytes) {
        if (prefetchAudit) prefetchAudit->entries[path] = {bytes.size() + 64, payloadHash(bytes)};
    };
    if (auto bytes = read()) { audit(*bytes); remember(*bytes); return std::move(*bytes); }
    std::shared_ptr<std::promise<std::vector<uint8_t>>> promise;
    std::shared_future<std::vector<uint8_t>> future;
    {
        std::lock_guard lock(missesMutex_);
        if (const auto found = misses_.find(key); found != misses_.end()) future = found->second;
        else {
            promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
            future = promise->get_future().share();
            misses_.emplace(key, future);
        }
    }
    if (!promise) { auto bytes = future.get(); audit(bytes); remember(bytes); return bytes; }
    try {
        auto bytes = read(); // another caller may have published before admission
        if (!bytes) {
            bytes = fetch();
            if (prefetchAudit) prefetchAudit->downloaded += bytes->size();
            if (length && bytes->size() != *length) throw std::runtime_error("Hosted payload length mismatch");
            publishPayload(key, *bytes);
        }
        promise->set_value(*bytes);
    } catch (...) { promise->set_exception(std::current_exception()); }
    {
        std::lock_guard lock(missesMutex_);
        misses_.erase(key);
    }
    auto bytes = future.get();
    audit(bytes);
    remember(bytes);
    return bytes;
}

std::vector<uint8_t> HostedAttachment::artifactRange(const ArtifactFile& file, uint64_t offset, uint64_t count)
{
    if (offset > file.length || count > file.length - offset) throw std::out_of_range("artifact range");
    std::vector<uint8_t> result;
    result.reserve(count);
    for (uint64_t start = offset / payloadBlockBytes * payloadBlockBytes; start < offset + count; start += payloadBlockBytes) {
        const auto length = std::min<uint64_t>(payloadBlockBytes, file.length - start);
        auto block = cachedPayload("artifact/" + file.sha256 + "/" + std::to_string(file.length) + "/" + std::to_string(start),
            [&] { return client_.readArtifact(file.sha256, start, length); }, length);
        const auto begin = std::max(start, offset), end = std::min(start + length, offset + count);
        result.insert(result.end(), block.begin() + (begin - start), block.begin() + (end - start));
    }
    return result;
}

void HostedAttachment::writeArtifact(const ArtifactFile& file, const std::filesystem::path& path)
{
    if (!disk_) { client_.fetchArtifact(file, path); return; }
    detail::writeArtifactFile({file.name, file.sha256, file.length}, path, payloadBlockBytes,
        [&](uint64_t offset, uint64_t count) { return artifactRange(file, offset, count); });
}

void HostedAttachment::prefetch(ObjectPrefetchJob& job, const std::atomic_bool& detached)
{
    PrefetchAudit audit;
    prefetchAudit = &audit;
    const auto cancelled = [&] { return job.cancelled_.load() || detached.load(); };
    const auto check = [&] {
        if (cancelled()) throw std::runtime_error("Download cancelled");
        // Yield between bounded transfers to interactive cache misses. At most
        // one prefetch request occupies the attachment's shared transport pool.
        std::unique_lock lock(priorityMutex_);
        while (interactivePayloads_.load() && !cancelled()) priorityWake_.wait_for(lock, std::chrono::milliseconds(10));
        if (cancelled()) throw std::runtime_error("Download cancelled");
    };
    const auto error = [&](const std::string& message) {
        std::lock_guard lock(job.mutex_);
        job.progress_.errors.push_back(message);
    };
    try {
        if (!disk_) throw std::runtime_error("Persistent cache is unavailable");
        check();
        // A listing is a single server snapshot and carries numbered references.
        // Subsequent payload reads stay pinned even if new commits arrive.
        StoreStatus captured;
        const auto records = client_.selectAllObjects({}, std::vector<std::string>{}, 1000, 3, &captured);
        std::vector<ObjectVersion> versions;
        for (const auto& record : records) {
            if (!record.version) throw std::runtime_error("Object listing has no version");
            versions.push_back(record.reference());
        }
        {
            std::lock_guard lock(job.mutex_);
            job.progress_.captured = captured;
            job.progress_.total = versions.size();
            job.progress_.stage = "loading";
        }
        // Associated relationships are selected against exactly these versions.
        for (size_t first = 0; first < versions.size(); first += 256) {
            check();
            relationshipsAt(std::vector<ObjectVersion>(versions.begin() + first,
                versions.begin() + std::min(versions.size(), first + 256)));
        }
        for (const auto& version : versions) {
            check();
            try {
                const auto revision = objectsAtVersions({version}).at(0);
                if (!revision) throw std::runtime_error("Captured object revision is unavailable");
                if (revision->geometryRevision) {
                    const auto geometry = geometryRevision(*revision->geometryRevision);
                    if (!geometry) throw std::runtime_error("Store does not hold geometry");
                    for (const auto& array : geometry->arrays) {
                        for (int64_t offset = 0; offset < array.byteLength; offset += payloadBlockBytes) {
                            check();
                            geometryArrayRange(geometry->uuid, array.name, offset, std::min(payloadBlockBytes, array.byteLength - offset));
                            { std::lock_guard lock(job.mutex_); job.progress_.downloadedBytes = audit.downloaded; }
                            std::this_thread::yield();
                        }
                        if (geometry->kind != "surface" && (array.name == "control_points" || array.name == "points" || array.name == "line_points")) {
                            check();
                            geometryPointAnnotations(geometry->uuid, array.name);
                        }
                    }
                } else if (revision->record.kind == "surface" || revision->record.kind == "fiber" ||
                           revision->record.kind == "point_collection" || revision->record.kind == "track") {
                    error(version.objectUuid + ": no store-held geometry (external placeholders are not downloaded)");
                }
                for (const auto& file : artifactManifest(revision->record)) {
                    PayloadHasher hash;
                    for (uint64_t offset = 0; offset < file.length; offset += payloadBlockBytes) {
                        check();
                        auto bytes = artifactRange(file, offset, std::min<uint64_t>(payloadBlockBytes, file.length - offset));
                        hash.update(bytes.data(), bytes.size());
                        { std::lock_guard lock(job.mutex_); job.progress_.downloadedBytes = audit.downloaded; }
                        std::this_thread::yield();
                    }
                    if (hash.finish() != file.sha256) throw std::runtime_error("Artifact checksum mismatch: " + file.name);
                }
            } catch (const std::exception& ex) { if (cancelled()) throw; error(version.objectUuid + ": " + ex.what()); }
            std::lock_guard lock(job.mutex_);
            ++job.progress_.completed;
            job.progress_.downloadedBytes = audit.downloaded;
        }
        // A successful write may evict an earlier block. Verify coverage at the
        // end, without fetching evicted data again and creating an eviction loop.
        { std::lock_guard lock(job.mutex_); job.progress_.stage = "checking cache coverage"; }
        for (const auto& [path, expected] : audit.entries) {
            check();
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != std::streamoff(expected.first)) {
                error("Cache coverage incomplete: disk limit, insufficient free space, or cache write failure");
                break;
            }
            file.seekg(0);
            std::string hash(64, '\0'); file.read(hash.data(), hash.size());
            PayloadHasher hasher;
            std::array<char, 64 * 1024> buffer;
            while (file) {
                file.read(buffer.data(), buffer.size());
                hasher.update(buffer.data(), size_t(file.gcount()));
            }
            if (hash != expected.second || hasher.finish() != hash) { error("Cache coverage incomplete: invalid entry"); break; }
        }
    } catch (const std::exception& ex) { if (!cancelled()) error(ex.what()); }
    prefetchAudit = nullptr;
    std::lock_guard lock(job.mutex_);
    job.progress_.downloadedBytes = audit.downloaded;
    job.progress_.cancelled = cancelled();
    job.progress_.complete = !cancelled() && job.progress_.errors.empty();
    job.progress_.done = true;
    job.progress_.stage = cancelled() ? "cancelled" : job.progress_.complete ? "cached" : "incomplete";
}

HostedAttachment::~HostedAttachment() { stop(); }

void HostedAttachment::start(const AttachmentIdentity& identity, Deliver deliver)
{
    {
        std::lock_guard lock(stateMutex_);
        identity_ = identity;
        deliver_ = std::move(deliver);
        stopping_ = false;
    }
    poller_ = std::thread([this] { poll(); });
}

void HostedAttachment::stop()
{
    {
        std::lock_guard lock(stateMutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (poller_.joinable()) poller_.join();
}

void HostedAttachment::poll()
{
    std::unique_lock lock(stateMutex_);
    while (!stopping_) {
        if (wake_.wait_for(lock, interval_, [&] { return stopping_; })) break;
        lock.unlock();
        check();
        lock.lock();
    }
}

void HostedAttachment::refresh() { check(); }

HostedFreshness HostedAttachment::freshness() const
{
    std::lock_guard lock(stateMutex_);
    return freshness_;
}

void HostedAttachment::invalidate(const std::vector<std::string>& objectUuids)
{
    std::lock_guard lock(cacheMutex_);
    ++invalidations_;
    for (const auto& uuid : objectUuids) cache_.put(headKey(uuid), Entry{}, 0);
}

void HostedAttachment::check()
{
    std::lock_guard checking(checking_);
    AttachmentIdentity identity;
    Deliver deliver;
    {
        std::lock_guard lock(stateMutex_);
        identity = identity_;
        deliver = deliver_;
    }
    std::vector<StoreSessionEvent> events;
    try {
        const auto status = client_.status();
        if (status.storeUuid != storeUuid_)
            throw std::runtime_error("hosted store: " + client_.endpoint() + " now serves another store");
        auto cursor = *cursor_;
        // Keep each feed page's commit once: copying it per change duplicates
        // its entire change vector and makes bulk edits quadratic in memory.
        std::vector<FeedCommit> commits;
        bool revalidate = !status.historyEnabled &&
            (cursor.generation != status.generation || cursor.sequence != status.commitSequence || cursor.offset);
        if (revalidate) cursor = {status.storeUuid, status.generation, status.commitSequence, 0};
        while (cursor.generation != status.generation || cursor.sequence < status.commitSequence || cursor.offset) {
            auto page = client_.changesSince(cursor);
            if (!page.cursorValid) {
                revalidate = true;
                cursor = page.next;
                break;
            }
            for (auto& commit : page.commits) commits.push_back(std::move(commit));
            cursor = page.next;
            if (!page.more) break;
        }
        if (revalidate) {
            // The position is gone (another generation, a restored store):
            // every cached head and the transform context may be stale.
            std::lock_guard lock(cacheMutex_);
            cache_.clear();
            ++invalidations_;
            transforms_.reset();
            ++transformInvalidations_;
            StoreSessionEvent event{StoreSessionEvent::Kind::Committed, identity, cursor.sequence, {}};
            event.revalidate = true;
            events.push_back(std::move(event));
        } else if (!commits.empty()) {
            // Kinds for consumers' touches(): objects from their versions (a
            // deletion's version keeps the record it removed), relationships
            // from their current or last seen record.
            std::vector<ObjectVersion> objects;
            std::vector<std::string> edges, touched;
            for (const auto& commit : commits) for (const auto& change : commit.changes) {
                if (change.recordKind == "object") {
                    objects.push_back({status.storeUuid, status.generation, change.uuid, change.version});
                    touched.push_back(change.uuid);
                } else {
                    edges.push_back(change.uuid);
                }
            }
            invalidate(touched);
            const auto revisions = objectsAtVersions(objects);
            std::map<std::string, std::string> kinds;
            for (const auto& revision : revisions)
                if (revision) kinds[revision->objectUuid] = revision->record.kind;
            if (!edges.empty()) {
                EdgeSelection selection;
                selection.uuids = edges;
                selection.includeRetired = true;
                for (const auto& edge : client_.selectAllEdges(selection, std::vector<std::string>{"kind"})) {
                    std::lock_guard lock(cacheMutex_);
                    edgeKinds_[edge.uuid] = edge.kind;
                }
                std::lock_guard lock(cacheMutex_);
                for (const auto& edge : edges)
                    if (const auto found = edgeKinds_.find(edge); found != edgeKinds_.end()) kinds[edge] = found->second;
            }
            StoreSessionEvent event{StoreSessionEvent::Kind::Committed, identity, cursor.sequence, {}};
            bool transformsChanged = false;
            for (const auto& commit : commits) for (const auto& change : commit.changes) {
                JournalEntry entry;
                entry.seq = commit.sequence;
                entry.recordUuid = change.uuid;
                if (const auto found = kinds.find(change.uuid); found != kinds.end()) entry.kind = found->second;
                entry.author = commit.author;
                entry.hostname = commit.host;
                entry.changeKind = change.deleted ? "deleted" : "published";
                transformsChanged |= change.recordKind == "object" && isTransformKind(entry.kind);
                event.changes.push_back(std::move(entry));
            }
            if (transformsChanged) {
                std::lock_guard lock(cacheMutex_);
                transforms_.reset();
                ++transformInvalidations_;
            }
            events.push_back(std::move(event));
        }
        cursor_ = cursor;
        std::lock_guard lock(stateMutex_);
        freshness_.status = status;
        freshness_.checkedAtUs = nowUs();
        freshness_.error.clear();
    } catch (const std::exception& error) {
        std::lock_guard lock(stateMutex_);
        freshness_.error = error.what();
        return;
    }
    if (deliver)
        for (auto& event : events) deliver(std::move(event));
}

StoreStatus HostedAttachment::status() { return client_.status(); }

TransactionReceipt HostedAttachment::submit(const TransactionRequest& request)
{
    auto receipt = client_.submit(request);
    std::vector<std::string> touched;
    for (const auto& [uuid, version] : receipt.objects) touched.push_back(uuid);
    invalidate(touched);
    std::lock_guard lock(cacheMutex_);
    transforms_.reset();
    ++transformInvalidations_;
    return receipt;
}

std::optional<ObjectRevision> HostedAttachment::currentObject(const std::string& objectUuid)
{
    const auto key = headKey(objectUuid);
    uint64_t epoch = 0;
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->revision) return entry->revision;
        epoch = invalidations_;
    }
    auto revision = client_.currentObjectRevision(objectUuid);
    if (revision) {
        std::lock_guard lock(cacheMutex_);
        if (epoch == invalidations_) cache_.put(key, Entry{revision, std::nullopt, nullptr}, cost(*revision));
    }
    return revision;
}

std::vector<std::optional<ObjectRevision>> HostedAttachment::objectsAtVersions(const std::vector<ObjectVersion>& references)
{
    std::vector<std::optional<ObjectRevision>> result(references.size());
    std::vector<ObjectVersion> missing;
    std::vector<std::size_t> positions;
    {
        std::lock_guard lock(cacheMutex_);
        for (std::size_t i = 0; i < references.size(); ++i) {
            if (const auto* entry = cache_.get(versionKey(references[i])); entry && entry->revision && !prefetchAudit) result[i] = entry->revision;
            else { missing.push_back(references[i]); positions.push_back(i); }
        }
    }
    if (disk_) {
        std::vector<ObjectVersion> absent;
        std::vector<size_t> absentPositions;
        for (size_t i = 0; i < missing.size(); ++i) {
            const auto bytes = readCachedPayload(versionKey(missing[i]), std::nullopt);
            if (bytes) {
                if (textOf(*bytes) != "null") result[positions[i]] = objectRevisionFromJson(textOf(*bytes));
                if (result[positions[i]]) {
                    std::lock_guard lock(cacheMutex_);
                    cache_.put(versionKey(missing[i]), Entry{result[positions[i]], std::nullopt, nullptr}, cost(*result[positions[i]]));
                }
            } else {
                absent.push_back(missing[i]); absentPositions.push_back(positions[i]);
            }
        }
        missing = std::move(absent); positions = std::move(absentPositions);
    }
    if (missing.empty()) return result;
    if (!disk_) {
        const auto fetched = client_.objectsAtVersions(missing);
        std::lock_guard lock(cacheMutex_);
        for (size_t i = 0; i < fetched.size(); ++i) {
            result[positions[i]] = fetched[i];
            if (fetched[i]) cache_.put(versionKey(missing[i]), Entry{fetched[i], std::nullopt, nullptr}, cost(*fetched[i]));
        }
        return result;
    }
    // Register each missing numbered record independently, then fetch the ones
    // we own together. Overlapping batches/prefetches share their promises.
    std::vector<std::shared_future<std::vector<uint8_t>>> futures;
    std::map<size_t, std::shared_ptr<std::promise<std::vector<uint8_t>>>> owned;
    {
        std::lock_guard lock(missesMutex_);
        for (size_t i = 0; i < missing.size(); ++i) {
            const auto key = versionKey(missing[i]);
            if (const auto found = misses_.find(key); found != misses_.end()) futures.push_back(found->second);
            else {
                auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
                futures.push_back(promise->get_future().share());
                misses_.emplace(key, futures.back());
                owned.emplace(i, std::move(promise));
            }
        }
    }
    std::set<size_t> completed;
    try {
        std::vector<ObjectVersion> fetch;
        std::vector<size_t> indices;
        for (const auto& [i, promise] : owned) {
            if (const auto bytes = readCachedPayload(versionKey(missing[i]), std::nullopt)) {
                promise->set_value(*bytes); completed.insert(i);
            } else { fetch.push_back(missing[i]); indices.push_back(i); }
        }
        if (!fetch.empty()) {
            const auto fetched = client_.objectsAtVersions(fetch);
            if (fetched.size() != fetch.size()) throw std::runtime_error("Hosted versions response length mismatch");
            for (size_t j = 0; j < fetched.size(); ++j) {
                const auto i = indices[j];
                auto bytes = bytesOf(fetched[j] ? objectRevisionToJson(*fetched[j]) : "null");
                if (prefetchAudit) prefetchAudit->downloaded += bytes.size();
                publishPayload(versionKey(missing[i]), bytes);
                owned.at(i)->set_value(std::move(bytes)); completed.insert(i);
            }
        }
    } catch (...) {
        for (const auto& [i, promise] : owned)
            if (!completed.contains(i)) promise->set_exception(std::current_exception());
    }
    {
        std::lock_guard lock(missesMutex_);
        for (const auto& [i, promise] : owned) misses_.erase(versionKey(missing[i]));
    }
    for (size_t i = 0; i < missing.size(); ++i) {
        const auto bytes = futures[i].get();
        if (textOf(bytes) == "null") continue;
        if (prefetchAudit) prefetchAudit->entries[diskRoot_ / payloadHash(bytesOf(versionKey(missing[i])))] =
            {bytes.size() + 64, payloadHash(bytes)};
        result[positions[i]] = objectRevisionFromJson(textOf(bytes));
        std::lock_guard lock(cacheMutex_);
        cache_.put(versionKey(missing[i]), Entry{result[positions[i]], std::nullopt, nullptr}, cost(*result[positions[i]]));
    }
    return result;
}

std::vector<SelectedRelationship> HostedAttachment::relationshipsAt(const std::vector<ObjectVersion>& versions)
{
    if (!disk_) return client_.relationshipsAt(versions);
    const auto query = objectVersionsRequestToJson(versions);
    const auto bytes = cachedPayload(payloadKey(metadataKey("relationships/" + payloadHash(bytesOf(query)))), [&] {
        StoreStatus asOf;
        const auto relationships = client_.relationshipsAt(versions, &asOf);
        return bytesOf(relationshipQueryResultToJson(asOf, relationships));
    });
    return relationshipQueryResultFromJson(textOf(bytes), nullptr);
}

std::optional<GeometryRevision> HostedAttachment::geometryRevision(const std::string& revisionUuid)
{
    const auto key = payloadKey(metadataKey("geometry/" + revisionUuid));
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->geometry && !prefetchAudit) return entry->geometry;
    }
    std::optional<GeometryRevision> geometry;
    if (!disk_) geometry = client_.geometryRevision(revisionUuid);
    else {
        const auto bytes = cachedPayload(key, [&] {
            const auto value = client_.geometryRevision(revisionUuid);
            return bytesOf(value ? geometryRevisionToJson(*value) : "null");
        });
        if (textOf(bytes) != "null") geometry = geometryRevisionFromJson(textOf(bytes));
    }
    if (geometry) {
        std::lock_guard lock(cacheMutex_);
        cache_.put(key, Entry{std::nullopt, geometry, nullptr}, cost(*geometry));
    }
    return geometry;
}

std::shared_ptr<const std::vector<uint8_t>> HostedAttachment::geometryArray(const std::string& revisionUuid, const std::string& array)
{
    const auto key = payloadKey("array/" + revisionUuid + "/" + array);
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->bytes) return entry->bytes;
    }
    auto bytes = std::make_shared<const std::vector<uint8_t>>(disk_ ? geometryArrayRange(revisionUuid, array, 0, std::nullopt) : client_.readGeometryArray(revisionUuid, array));
    std::lock_guard lock(cacheMutex_);
    cache_.put(key, Entry{std::nullopt, std::nullopt, bytes}, bytes->size() + 128);
    return bytes;
}

std::vector<uint8_t> HostedAttachment::geometryArrayRange(const std::string& revisionUuid, const std::string& array,
                                                        int64_t offset, const std::optional<int64_t>& count)
{
    auto rangeLength = [&](int64_t size) {
        if (offset < 0 || offset > size || (count && (*count < 0 || *count > size - offset)))
            throw std::out_of_range("geometry: byte range outside array " + array);
        return count.value_or(size - offset);
    };
    auto slice = [&](const std::vector<uint8_t>& bytes) {
        const auto length = rangeLength(int64_t(bytes.size()));
        return std::vector<uint8_t>(bytes.begin() + offset, bytes.begin() + offset + length);
    };
    std::shared_ptr<const std::vector<uint8_t>> cached;
    const auto arrayKey = payloadKey("array/" + revisionUuid + "/" + array);
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(arrayKey)) cached = entry->bytes;
    }
    if (cached && !prefetchAudit) return slice(*cached);
    const auto descriptor = geometryRevision(revisionUuid);
    if (!descriptor) throw std::invalid_argument("geometry revision not found: " + revisionUuid);
    const auto found = std::find_if(descriptor->arrays.begin(), descriptor->arrays.end(),
                                    [&](const auto& a) { return a.name == array; });
    if (found == descriptor->arrays.end()) throw std::invalid_argument("geometry: array not found: " + array);
    const auto size = found->byteLength;
    const auto length = rangeLength(size);
    if (disk_) {
        std::vector<uint8_t> result;
        result.reserve(size_t(length));
        for (int64_t start = offset / payloadBlockBytes * payloadBlockBytes; start < offset + length; start += payloadBlockBytes) {
            const auto blockLength = std::min(payloadBlockBytes, size - start);
            const auto key = payloadKey("block/" + revisionUuid + "/" + array + "/" + std::to_string(size) + "/" + std::to_string(start));
            auto block = cachedPayload(key, [&] { return client_.readGeometryArray(revisionUuid, array, start, blockLength); }, blockLength);
            const auto begin = std::max(start, offset), end = std::min(start + blockLength, offset + length);
            result.insert(result.end(), block.begin() + (begin - start), block.begin() + (end - start));
        }
        return result;
    }
    // An array larger than the cache would be downloaded again for every
    // range. Keep these reads bounded all the way to the HTTP client.
    if (cache_.limit() < 128 || uint64_t(size) > cache_.limit() - 128)
        return client_.readGeometryArray(revisionUuid, array, offset, length);
    return slice(*geometryArray(revisionUuid, array));
}

std::shared_ptr<const nlohmann::json> HostedAttachment::geometryPointAnnotations(const std::string& revisionUuid,
                                                                                  const std::string& array)
{
    // Immutable per geometry revision, cached parsed.
    const auto key = payloadKey("annotations/" + revisionUuid + "/" + array);
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->json && !prefetchAudit) return entry->json;
    }
    auto json = std::make_shared<const nlohmann::json>(disk_ ? nlohmann::json::parse(textOf(
        cachedPayload(key, [&] { return bytesOf(client_.geometryPointAnnotations(revisionUuid, array).dump()); })))
        : client_.geometryPointAnnotations(revisionUuid, array));
    // Cost: an estimate per annotated point (the parsed form is not measured).
    const auto cost = 1024 + 512 * json->size();
    std::lock_guard lock(cacheMutex_);
    cache_.put(key, Entry{std::nullopt, std::nullopt, nullptr, json}, cost);
    return json;
}

VolumeTransforms HostedAttachment::volumeTransforms()
{
    for (int attempt = 0; attempt < 4; ++attempt) {
        uint64_t epoch;
        {
            std::lock_guard lock(cacheMutex_);
            if (transforms_) return *transforms_;
            epoch = transformInvalidations_;
        }
        // Recorded registrations and choices only: a shared store never applies
        // a client's unrecorded catalog registrations.
        ObjectSelection registrations;
        registrations.kind = "volume_transform";
        std::vector<std::pair<ObjectRecord, std::string>> recorded;
        for (auto& record : client_.selectAllObjects(registrations)) {
            const auto head = client_.currentObjectRevision(record.uuid);
            if (!head) continue; // removed since the listing; the next check invalidates
            recorded.emplace_back(std::move(record), head->uuid);
        }
        ObjectSelection choices;
        choices.kind = "volume_transform_choice";
        auto transforms = volumeTransformsFrom(recorded, client_.selectAllObjects(choices), nullptr);
        std::lock_guard lock(cacheMutex_);
        // A poll or submission may have invalidated the snapshot while HTTP was in flight.
        if (epoch != transformInvalidations_) continue;
        transforms_ = transforms;
        return transforms;
    }
    throw StoreBusy("hosted store: transforms kept changing while reading; retry later");
}

std::set<std::string> HostedAttachment::holds(const std::set<std::string>& uuids)
{
    std::set<std::string> held;
    if (uuids.empty()) return held;
    ObjectSelection selection;
    selection.uuids.assign(uuids.begin(), uuids.end());
    selection.includeRetired = true;
    for (const auto& record : client_.selectAllObjects(selection, std::vector<std::string>{})) held.insert(record.uuid);
    return held;
}

std::vector<ObjectRecord> HostedAttachment::selectObjects(const ObjectSelection& selection,
                                                          const std::optional<std::vector<std::string>>& fields)
{
    return selection.limit ? client_.selectObjects(selection, fields) : client_.selectAllObjects(selection, fields);
}

std::vector<EdgeRecord> HostedAttachment::selectEdges(const EdgeSelection& selection,
                                                      const std::optional<std::vector<std::string>>& fields)
{
    return selection.limit ? client_.selectEdges(selection, fields) : client_.selectAllEdges(selection, fields);
}

std::optional<ObjectRevision> HostedStoreReader::currentObject(const std::string& objectUuid) const
{
    auto revision = attachment_->currentObject(objectUuid);
    if (revision && revision->deleted) return std::nullopt;
    return revision;
}

std::vector<ObjectRevision> HostedStoreReader::objectHistory(const std::string& objectUuid) const
{
    return attachment_->client().objectHistory(objectUuid);
}

std::optional<GeometryRevision> HostedStoreReader::currentGeometry(const std::string& objectUuid) const
{
    const auto object = currentObject(objectUuid);
    if (!object || !object->geometryRevision) return std::nullopt;
    return geometryRevision(*object->geometryRevision);
}

std::vector<uint8_t> HostedStoreReader::readGeometryArray(const std::string& revisionUuid, const std::string& arrayName,
                                                          int64_t byteOffset, const std::optional<int64_t>& byteCount) const
{
    return attachment_->geometryArrayRange(revisionUuid, arrayName, byteOffset, byteCount);
}

std::vector<uint8_t> HostedStoreReader::readGeometryRectangle(const std::string& revisionUuid, const std::string& arrayName,
                                                              int64_t row, int64_t column, int64_t rows,
                                                              int64_t columns) const
{
    const auto descriptor = geometryRevision(revisionUuid);
    if (!descriptor) throw std::invalid_argument("geometry revision not found: " + revisionUuid);
    const auto found = std::find_if(descriptor->arrays.begin(), descriptor->arrays.end(),
                                    [&](const auto& a) { return a.name == arrayName; });
    if (found == descriptor->arrays.end()) throw std::invalid_argument("geometry: array not found: " + arrayName);
    const auto& shape = found->shape;
    if (descriptor->kind != "surface" || (shape.size() != 2 && shape.size() != 3))
        throw std::invalid_argument("geometry: rectangle requires an aligned surface array");
    if (row < 0 || column < 0 || rows < 0 || columns < 0 || row > shape[0] || column > shape[1] ||
        rows > shape[0] - row || columns > shape[1] - column)
        throw std::out_of_range("geometry: invalid rectangle");
    const auto pixel = surfacePixelBytes(found->dtype, shape);
    if (found->byteLength != shape[0] * shape[1] * pixel)
        throw std::runtime_error("geometry: array bytes do not match its shape: " + arrayName);
    if (column == 0 && columns == shape[1])
        return readGeometryArray(revisionUuid, arrayName, row * columns * pixel, rows * columns * pixel);
    std::vector<uint8_t> result(static_cast<std::size_t>(rows * columns * pixel));
    for (int64_t r = 0; r < rows; ++r) {
        const auto bytes = readGeometryArray(revisionUuid, arrayName, ((row + r) * shape[1] + column) * pixel, columns * pixel);
        std::copy(bytes.begin(), bytes.end(), result.begin() + r * columns * pixel);
    }
    return result;
}

} // namespace vc::objects::detail
