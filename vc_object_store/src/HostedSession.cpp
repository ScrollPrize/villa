#include "HostedSession.hpp"
#include "geometry/ArrayLayout.hpp"

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
std::string versionKey(const ObjectVersion& v) { return "version/" + v.storeUuid + "/" + v.generation + "/" + v.objectUuid + "/" + std::to_string(v.version); }
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
        bool revalidate = false;
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
            if (const auto* entry = cache_.get(versionKey(references[i])); entry && entry->revision) result[i] = entry->revision;
            else { missing.push_back(references[i]); positions.push_back(i); }
        }
    }
    if (missing.empty()) return result;
    const auto fetched = client_.objectsAtVersions(missing);
    std::lock_guard lock(cacheMutex_);
    for (std::size_t i = 0; i < fetched.size(); ++i) {
        result[positions[i]] = fetched[i];
        if (fetched[i]) cache_.put(versionKey(missing[i]), Entry{fetched[i], std::nullopt, nullptr}, cost(*fetched[i]));
    }
    return result;
}

std::optional<GeometryRevision> HostedAttachment::geometryRevision(const std::string& revisionUuid)
{
    const auto key = "geometry/" + revisionUuid;
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->geometry) return entry->geometry;
    }
    auto geometry = client_.geometryRevision(revisionUuid);
    if (geometry) {
        std::lock_guard lock(cacheMutex_);
        cache_.put(key, Entry{std::nullopt, geometry, nullptr}, cost(*geometry));
    }
    return geometry;
}

std::shared_ptr<const std::vector<uint8_t>> HostedAttachment::geometryArray(const std::string& revisionUuid, const std::string& array)
{
    const auto key = "array/" + revisionUuid + "/" + array;
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->bytes) return entry->bytes;
    }
    auto bytes = std::make_shared<const std::vector<uint8_t>>(client_.readGeometryArray(revisionUuid, array));
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
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get("array/" + revisionUuid + "/" + array)) cached = entry->bytes;
    }
    if (cached) return slice(*cached);
    const auto descriptor = geometryRevision(revisionUuid);
    if (!descriptor) throw std::invalid_argument("geometry revision not found: " + revisionUuid);
    const auto found = std::find_if(descriptor->arrays.begin(), descriptor->arrays.end(),
                                    [&](const auto& a) { return a.name == array; });
    if (found == descriptor->arrays.end()) throw std::invalid_argument("geometry: array not found: " + array);
    const auto size = found->byteLength;
    const auto length = rangeLength(size);
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
    const auto key = "annotations/" + revisionUuid + "/" + array;
    {
        std::lock_guard lock(cacheMutex_);
        if (const auto* entry = cache_.get(key); entry && entry->json) return entry->json;
    }
    auto json = std::make_shared<const nlohmann::json>(client_.geometryPointAnnotations(revisionUuid, array));
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
