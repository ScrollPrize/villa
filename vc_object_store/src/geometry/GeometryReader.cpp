#include "geometry/GeometryReader.hpp"
#include "geometry/GeometryDelta.hpp"
#include "geometry/SharedGeometry.hpp"
#include "internal/ReferenceResolver.hpp"
#include "internal/ObjectScan.hpp"
#include "internal/VisibleRows.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"

namespace vc::objects {
using namespace detail;
namespace detail {
// One instance per leased reader connection: no shared mutable geometry or
// cross-snapshot lookups between concurrent readers. The SQLite data_version is
// connection-local and changes on commits by any *other* connection, including
// this store's writer. Writer connections never receive this cache.
class GeometryReadCache {
public:
    explicit GeometryReadCache(size_t budget) : budget_(budget) {}
    void begin(int version) {
        if (version != version_) {
            readers_.clear(); recency_.clear(); charge_ = 0; version_ = version;
        }
    }
    std::shared_ptr<GeometryReader> take(const std::string& revision) {
        const auto found = readers_.find(revision);
        if (found == readers_.end()) return {};
        auto reader = std::move(found->second.reader);
        charge_ -= found->second.charge;
        recency_.erase(found->second.recency);
        readers_.erase(found);
        return reader;
    }
    void retain(const std::string& revision, std::shared_ptr<GeometryReader> reader) {
        const auto charge = reader->retainedCharge();
        if (charge > budget_) return;
        // Also bound revision/node overhead, including empty geometries: one
        // reader per MiB of budget, at least 64.
        while (charge_ > budget_ - charge || readers_.size() >= std::max<size_t>(64, budget_ >> 20)) {
            const auto old = readers_.find(recency_.front());
            charge_ -= old->second.charge;
            readers_.erase(old); recency_.pop_front();
        }
        recency_.push_back(revision);
        try {
            readers_.emplace(revision, Entry{std::move(reader), charge, std::prev(recency_.end())});
        } catch (...) { recency_.pop_back(); throw; }
        charge_ += charge;
    }
private:
    struct Entry {
        std::shared_ptr<GeometryReader> reader;
        size_t charge;
        std::list<std::string>::iterator recency;
    };
    size_t budget_, charge_ = 0;
    int version_ = -1;
    std::map<std::string, Entry> readers_;
    std::list<std::string> recency_;
};

void beginReadCaches(sqlite3* db, StatementPool& pool, size_t geometryBudget, size_t spatialBudget)
{
    const int version = pragmaInt(db, "data_version");
    // Components stay valid while no connection has committed since.
    if (pool.componentsVersion != version) {
        pool.components.clear(); pool.componentNodes = 0; pool.componentsVersion = version;
    }
    if (!geometryBudget) pool.geometryCache.reset();
    else {
        if (!pool.geometryCache) pool.geometryCache = std::make_shared<GeometryReadCache>(geometryBudget);
        pool.geometryCache->begin(version);
    }
    beginSpatialRead(pool, spatialBudget, version);
}

namespace {
std::shared_ptr<GeometryReadCache> connectionGeometryCache(sqlite3* db)
{
    const auto lease = activeStatementPools.find(db);
    return lease == activeStatementPools.end() || !lease->second->readSnapshot ? nullptr : lease->second->geometryCache;
}
}

RetainedGeometryReader::RetainedGeometryReader(sqlite3* db, std::string revision)
    : db_(db), revision_(std::move(revision)), exceptions_(std::uncaught_exceptions())
{
    const auto cache = connectionGeometryCache(db_);
    reader_ = cache ? cache->take(revision_) : nullptr;
    if (!reader_) reader_ = std::make_shared<GeometryReader>(db_, revision_);
}

RetainedGeometryReader::~RetainedGeometryReader()
{
    if (std::uncaught_exceptions() != exceptions_) return;
    try {
        if (const auto cache = connectionGeometryCache(db_)) cache->retain(revision_, std::move(reader_));
    } catch (...) {} // retention is an optimization
}

template<class F> auto withGeometryReader(const Txn& txn, const std::string& revision, F&& read)
{
    RetainedGeometryReader reader(ObjectStoreImplAccess::conn(txn), revision);
    return read(*reader);
}

GeometryReader::GeometryReader(const Txn& txn, const std::string& revision, bool cacheChunks)
    : GeometryReader(ObjectStoreImplAccess::conn(txn), revision, cacheChunks) {}

GeometryReader::GeometryReader(sqlite3* db, const std::string& revision, bool cacheChunks, bool allowReverse)
    : db_(db), cacheChunks_(cacheChunks)
{
    auto record=resolveGeometryRecord(db,revision);
    if (!record) throw std::invalid_argument("geometry: revision not found");
    db_=record->source;
    reference_=canonicalGeometryToken(revision);
    objectUuid_=record->revision.objectUuid;
    revision_=parseUuid(revision);
    // Charge JSON nodes/strings, layouts and cache bookkeeping conservatively,
    // in addition to the separately accounted retained payload bytes.
    metadataCharge_ = record->manifest.size() * 32 + 4096;
    contentHash_ = record->revision.contentHash;
    manifest_ = std::move(record->descriptor);
    std::optional<nlohmann::json> currentDescriptor;
    if(record->archived) {
        Statement object(db_,"SELECT geometry_manifest,data_checksum FROM objects WHERE uuid=? AND storage_ref=? AND state<>2 AND geometry_manifest IS NOT NULL",
            parseUuid(record->revision.objectUuid),revision);
        if(object.row()) {
            if(object.text(1)!=contentHash_)
                throw std::runtime_error("geometry: current checksum disagrees with archive revision");
            currentDescriptor=verifiedGeometryManifest(object.text(0),object.text(1));
            metadataCharge_+=object.text(0).size()*32;
        }
    }
    if (manifest_.contains("delta")) {
        const auto target = manifest_;
        const auto object = parseUuid(record->revision.objectUuid);
        baseRevision_ = target.at("delta").at("base").get<std::string>();
        Statement current(db_, "SELECT manifest FROM geometry_materializations WHERE object_uuid=? AND revision_uuid=?",
            object, revision_);
        if (current.row()) {
            const auto full = current.text(0);
            auto materialized=verifiedGeometryManifest(full,target.at("delta").at("result_hash"));
            if(currentDescriptor && *currentDescriptor!=materialized)
                throw std::runtime_error("geometry: current descriptor disagrees with materialized archive");
            manifest_=currentDescriptor ? *currentDescriptor : std::move(materialized);
            auto compact = manifest_;
            for (auto& a : compact.at("arrays")) a.erase("chunks");
            auto expected = target; expected.erase("delta");
            if (compact != expected) throw std::runtime_error("geometry delta: materialized layout mismatch");
            materializedObject_ = formatUuid(object);
            parentReference_=record->revision.parentUuid.value_or("");
            parentHash_=target.at("delta").at("parent_hash").get<std::string>();
            if(parentReference_.empty()||parentReference_==reference_)
                throw std::runtime_error("geometry delta: invalid parent reference");
        } else {
            auto descriptor = target;
            auto id = revision;
            auto parent = record->revision.parentUuid.value_or("");
            std::set<std::string> seen;
            while (descriptor.contains("delta")) {
                if (!seen.insert(id).second || parent.empty())
                    throw std::runtime_error("geometry delta: cyclic or missing parent");
                deltas_.emplace_back(id, descriptor);
                const auto previous=resolveGeometryRecord(db_,parent);
                if (!previous || previous->revision.objectUuid != record->revision.objectUuid ||
                    previous->revision.contentHash != descriptor.at("delta").at("parent_hash").get<std::string>())
                    throw std::runtime_error("geometry delta: invalid parent");
                id = parent;
                descriptor = previous->descriptor;
                parent = previous->revision.parentUuid.value_or("");
            }
            if (id != baseRevision_) throw std::runtime_error("geometry delta: invalid base");
            manifest_ = descriptor;
            std::reverse(deltas_.begin(), deltas_.end());
            for (const auto& [deltaId, delta] : deltas_) {
                if (delta.at("delta").at("base") != baseRevision_)
                    throw std::runtime_error("geometry delta: inconsistent base");
                auto compact = manifest_;
                for (auto& a : compact.at("arrays")) a.erase("chunks");
                auto expected = delta; expected.erase("delta");
                if (compact != expected) throw std::runtime_error("geometry delta: layout changed");
                for (const auto& c : delta.at("delta").at("chunks")) {
                    auto& hash = manifest_.at("arrays").at(c.at("array").get<std::string>()).at("chunks").at(c.at("index").get<int64_t>());
                    if (hash != c.at("before")) throw std::runtime_error("geometry delta: parent hash mismatch");
                    hash = c.at("after");
                }
                if (hashText(manifest_.dump()) != delta.at("delta").at("result_hash").get<std::string>())
                    throw std::runtime_error("geometry delta: manifest result mismatch");
                metadataCharge_ += delta.dump().size() * 32 + 256;
            }
            // Historical manifests are verified against their immutable base.
            // For payloads, reversing a shorter current-head suffix avoids
            // replaying a long prefix. Stop at branches/explicit replacements.
            Statement tip(db_, "SELECT revision_uuid FROM geometry_materializations WHERE object_uuid=?", object);
            if (allowReverse && tip.row()) {
                const auto head = formatUuid(tip.blob16(0));
                auto cursor = head;
                std::vector<std::pair<std::string, nlohmann::json>> reverse;
                while (cursor != revision && reverse.size() + 1 < deltas_.size()) {
                    const auto step=resolveGeometryRecord(db_,cursor);
                    if (!step || !step->revision.parentUuid || step->revision.objectUuid!=record->revision.objectUuid) break;
                    auto m = step->descriptor;
                    if (!m.contains("delta") || m.at("delta").at("base") != baseRevision_) break;
                    reverse.emplace_back(cursor, std::move(m));
                    cursor = *step->revision.parentUuid;
                }
                if (cursor == revision && !reverse.empty()) {
                    reverseSource_ = std::make_unique<GeometryReader>(db_, head, false);
                    deltas_ = std::move(reverse);
                    metadataCharge_ += reverseSource_->retainedCharge();
                }
            }
        }
        metadataCharge_ += manifest_.dump().size() * 32;
    }
    if(currentDescriptor) {
        if(manifest_!=*currentDescriptor)
            throw std::runtime_error("geometry: current descriptor disagrees with archive");
        manifest_=std::move(*currentDescriptor);
    }
    for (const auto& [name, array] : manifest_.at("arrays").items())
        layouts_.emplace(name, geometryLayout(array));
}

std::vector<uint8_t> GeometryReader::chunk(const std::string& name, int64_t index) {
    // References can span databases; physical row ordering cannot detect cycles.
    static thread_local std::set<std::string> active;
    if(!active.insert(reference_).second)throw std::runtime_error("geometry: cyclic payload reference");
    struct Guard {std::set<std::string>& active;const std::string& ref;~Guard(){active.erase(ref);}} guard{active,reference_};
    std::vector<uint8_t> result;
    visitChunk(name, index, [&](auto bytes) { result.assign(bytes.begin(), bytes.end()); });
    return result;
}

void GeometryReader::visitChunk(const std::string& name, int64_t index,
    const std::function<void(std::span<const uint8_t>)>& visit)
{
    const ChunkKey key{name, index};
    auto found = chunks_.find(key);
    if (found != chunks_.end()) {
        requestServed(found->second.bytes.size(), &RegionStats::cacheBytes);
        recency_.splice(recency_.end(), recency_, found->second.recency);
        visit(found->second.bytes);
        return;
    }
    const auto expectedSize = layouts_.at(name).chunkBytes(index);
    const auto& expectedHash = manifest_.at("arrays").at(name).at("chunks").at(index).get_ref<const std::string&>();
    std::vector<uint8_t> owned;
    std::string storedHash;
    if (!materializedObject_.empty()) {
        if (readRequest) {
            Statement size(db_, "SELECT length(data) FROM geometry_current_chunks WHERE object_uuid=? AND array_name=? AND chunk_index=?",
                parseUuid(materializedObject_), name, index);
            if (size.row()) requestBytes(size.i64(0), &RegionStats::payloadBytes);
        }
        Statement current(db_, "SELECT data,hash FROM geometry_current_chunks WHERE object_uuid=? AND array_name=? AND chunk_index=?",
            parseUuid(materializedObject_), name, index);
        if (current.row()) {
            const auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(current.value, 0));
            if (geometryWork) geometryWork->readBytes[name] += sqlite3_column_bytes(current.value, 0);
            owned.assign(p, p + sqlite3_column_bytes(current.value, 0));
            storedHash = readDigest(current,1);
        } else {
            // Materialized rows contain modified chunks. An unchanged chunk
            // is the parent's chunk, irrespective of its physical location.
            if(!parentSource_) {
                const auto parent=resolveGeometryRecord(db_,parentReference_);
                if(!parent||parent->revision.objectUuid!=objectUuid_||parent->revision.contentHash!=parentHash_)
                    throw std::runtime_error("geometry delta: invalid parent");
                parentSource_=std::make_unique<GeometryReader>(db_,parentReference_,false,false);
                metadataCharge_+=parentSource_->retainedCharge();
            }
            owned=parentSource_->chunk(name,index);
            storedHash=parentSource_->manifest().at("arrays").at(name).at("chunks").at(index).get<std::string>();
        }
    }
    if (reverseSource_) {
        owned = reverseSource_->chunk(name, index);
        storedHash = reverseSource_->manifest().at("arrays").at(name).at("chunks").at(index).get<std::string>();
    } else if (storedHash.empty()) {
        const auto& descriptor = manifest_.at("arrays").at(name);
        if (descriptor.contains("source_revision")) {
            const auto source = descriptor.at("source_revision").get<std::string>();
            // Follow immutable ancestry, not SQLite row ordering: ancestors
            // can live in another pinned store just like their payloads.
            auto ancestor=resolveGeometryRecord(db_,formatUuid(revision_));
            std::set<std::string> seen;
            while(ancestor && ancestor->revision.uuid!=source) {
                if(!seen.insert(ancestor->revision.uuid).second || !ancestor->revision.parentUuid) {
                    ancestor.reset();break;
                }
                const auto object=ancestor->revision.objectUuid;
                ancestor=resolveGeometryRecord(db_,*ancestor->revision.parentUuid);
                if(ancestor&&ancestor->revision.objectUuid!=object)ancestor.reset();
            }
            if(!ancestor||seen.empty())throw std::runtime_error("geometry: invalid array source");
            GeometryReader inherited(db_, source, false);
            const auto& original = inherited.manifest().at("arrays").at(name);
            auto layout = descriptor; layout.erase("chunks"); layout.erase("source_revision");
            auto sourceLayout = original; sourceLayout.erase("chunks"); sourceLayout.erase("source_revision");
            if (layout != sourceLayout) throw std::runtime_error("geometry: array source layout mismatch");
            owned = inherited.chunk(name, index);
            storedHash = original.at("chunks").at(index).get<std::string>();
        } else {
            storedHash = descriptor.at("chunks").at(index).get<std::string>();
            // The descriptor is the ordered chunk index. For an archived
            // delta chain, start from the earliest pre-change payload.
            bool foundBase=false;
            for(const auto& [id,delta]:deltas_) {
                for(const auto& c:delta.at("delta").at("chunks"))
                    if(c.at("array")==name&&c.at("index")==index) {
                        storedHash=c.at("before").get<std::string>();foundBase=true;break;
                    }
                if(foundBase)break;
            }
            if (readRequest) requestBytes(geometryPayloadSize(db_,storedHash), &RegionStats::payloadBytes);
            owned=readGeometryPayload(db_,storedHash);
            if (geometryWork) geometryWork->readBytes[name] += owned.size();
            if (!deltas_.empty()) verifyGeometryChunk(owned, storedHash, storedHash, expectedSize, name);
        }
    }
    for (const auto& [id, delta] : deltas_) {
        const auto& chunks = delta.at("delta").at("chunks");
        if (std::none_of(chunks.begin(), chunks.end(), [&](const auto& c) {
            return c.at("array") == name && c.at("index") == index;
        })) continue;
        for (const auto& c : decodeGeometryDelta(db_, id, delta)) if (c.key == key) {
            applyGeometryDelta(owned, c, bool(reverseSource_));
            storedHash = reverseSource_ ? c.beforeHash : c.afterHash;
        }
    }
    const auto bytes = std::span<const uint8_t>(owned);
    verifyGeometryChunk(bytes, storedHash, expectedHash, expectedSize, name);
    if (!cacheChunks_) { visit(bytes); return; }
    // Bound bytes, not chunk count: tiled chunks are smaller than linear ones.
    constexpr size_t budget = 8 * geometryChunkSize;
    while (cachedBytes_ + bytes.size() > budget && !recency_.empty()) {
        const auto old = chunks_.find(recency_.front());
        cachedBytes_ -= old->second.bytes.size();
        chunks_.erase(old); recency_.pop_front();
    }
    recency_.push_back(key);
    auto inserted = chunks_.emplace(key, CachedChunk{std::move(owned), std::prev(recency_.end())}).first;
    cachedBytes_ += bytes.size();
    visit(inserted->second.bytes);
}

std::vector<uint8_t> GeometryReader::read(const std::string& name, int64_t offset, std::optional<int64_t> count)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& layout = found->second;
    const auto length = layout.length;
    if (offset < 0 || offset > length) throw std::out_of_range("geometry: invalid byte offset");
    const int64_t size = count.value_or(length - offset);
    if (size < 0 || size > length - offset) throw std::out_of_range("geometry: invalid byte count");
    requestBytes(size, &RegionStats::reconstructedBytes);
    std::vector<uint8_t> result(size);
    if (!size) return result;
    if (offset == 0 && size == length) {
        for (int64_t index = 0; index < layout.chunkCount(); ++index)
            visitChunk(name, index, [&](auto bytes) { copyGeometryChunk(layout, index, bytes, result.data()); });
        return result;
    }
    if (!layout.tile) {
        for (int64_t index = offset / geometryChunkSize; index <= (offset + size - 1) / geometryChunkSize; ++index) {
            const int64_t base = index * geometryChunkSize;
            visitChunk(name, index, [&](auto bytes) {
                const auto begin = std::max(base, offset), end = std::min(base + int64_t(bytes.size()), offset + size);
                std::memcpy(result.data() + begin - offset, bytes.data() + begin - base, end - begin);
            });
        }
        return result;
    }
    const auto rowBytes = layout.cols * layout.pixelBytes;
    const auto firstRow = offset / rowBytes, lastRow = (offset + size - 1) / rowBytes;
    const auto firstColTile = firstRow == lastRow ? offset % rowBytes / layout.pixelBytes / layout.tile : 0;
    const auto lastColTile = firstRow == lastRow ? (offset + size - 1) % rowBytes / layout.pixelBytes / layout.tile : layout.tileColumns() - 1;
    for (int64_t ty = firstRow / layout.tile; ty <= lastRow / layout.tile; ++ty)
        for (int64_t tx = firstColTile; tx <= lastColTile; ++tx) {
            const auto index = ty * layout.tileColumns() + tx;
            const auto [y, x, h, w] = layout.rectangle(index);
            const auto beginRow = std::max(y, firstRow), endRow = std::min(y + h, lastRow + 1);
            auto overlap = [&](int64_t row) {
                const auto base = (row * layout.cols + x) * layout.pixelBytes;
                return std::array<int64_t, 2>{std::max(base, offset), std::min(base + w * layout.pixelBytes, offset + size)};
            };
            bool needed = false;
            for (auto row = beginRow; row < endRow && !needed; ++row) {
                const auto [begin, end] = overlap(row); needed = begin < end;
            }
            if (!needed) continue;
            visitChunk(name, index, [&](auto bytes) {
                for (auto row = beginRow; row < endRow; ++row) {
                    const auto [begin, end] = overlap(row);
                    if (begin < end)
                        std::memcpy(result.data() + begin - offset,
                            bytes.data() + (row - y) * w * layout.pixelBytes + begin - (row * layout.cols + x) * layout.pixelBytes, end - begin);
                }
            });
        }
    return result;
}

std::vector<std::vector<uint8_t>> GeometryReader::ranges(const std::string& name,
    const std::vector<std::array<int64_t, 2>>& ranges)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& layout = found->second;
    struct Copy { size_t output; int64_t source, destination, count; };
    std::map<int64_t, std::vector<Copy>> plan;
    std::vector<std::vector<uint8_t>> result;
    result.reserve(ranges.size());
    for (const auto& range : ranges) {
        const auto [offset, size] = range;
        if (offset < 0 || offset > layout.length || size < 0 || size > layout.length - offset)
            throw std::out_of_range("geometry: invalid byte range");
        const auto output = result.size();
        result.emplace_back(static_cast<size_t>(size));
        if (!size) continue;
        if (!layout.tile) {
            for (auto chunk = offset / geometryChunkSize; chunk <= (offset + size - 1) / geometryChunkSize; ++chunk) {
                const int64_t base = chunk * geometryChunkSize;
                const auto begin = std::max(base, offset);
                const auto end = std::min(base + layout.chunkBytes(chunk), offset + size);
                plan[chunk].push_back({output, begin - base, begin - offset, end - begin});
            }
        } else {
            // Split logical row-major bytes into physical tile spans, including
            // partial XYZ elements and ranges crossing row/tile boundaries.
            const auto rowBytes = layout.cols * layout.pixelBytes;
            for (auto row = offset / rowBytes; row <= (offset + size - 1) / rowBytes; ++row) {
                const auto first = std::max(offset, row * rowBytes);
                const auto last = std::min(offset + size, (row + 1) * rowBytes);
                for (auto col = (first % rowBytes) / (layout.tile * layout.pixelBytes);
                     col <= (last - 1 - row * rowBytes) / (layout.tile * layout.pixelBytes); ++col) {
                    const auto chunk = row / layout.tile * layout.tileColumns() + col;
                    const auto [y, x, h, w] = layout.rectangle(chunk);
                    const auto base = row * rowBytes + x * layout.pixelBytes;
                    const auto begin = std::max(base, first), end = std::min(base + w * layout.pixelBytes, last);
                    plan[chunk].push_back({output, (row - y) * w * layout.pixelBytes + begin - base,
                                          begin - offset, end - begin});
                }
            }
        }
    }
    // Each touched chunk is loaded and verified once, independently of caller
    // order or cache capacity. Duplicates/overlaps scatter to every destination.
    for (const auto& [chunk, copies] : plan)
        visitChunk(name, chunk, [&](auto bytes) {
            for (const auto& copy : copies)
                std::memcpy(result[copy.output].data() + copy.destination,
                            bytes.data() + copy.source, copy.count);
        });
    return result;
}

std::vector<uint8_t> GeometryReader::rectangle(const std::string& name, int64_t row, int64_t column,
                                             int64_t rows, int64_t columns)
{
    const auto found = layouts_.find(name);
    if (found == layouts_.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& array = manifest_.at("arrays").at(name);
    const auto shape = array.at("shape").get<std::vector<int64_t>>();
    if (manifest_.at("kind") != "surface" || (shape.size() != 2 && shape.size() != 3))
        throw std::invalid_argument("geometry: rectangle requires an aligned surface array");
    if (row < 0 || column < 0 || rows < 0 || columns < 0 || row > shape[0] || column > shape[1] ||
        rows > shape[0] - row || columns > shape[1] - column)
        throw std::out_of_range("geometry: invalid rectangle");
    const auto pixelBytes = surfacePixelBytes(array.at("dtype"), shape);
    requestBytes(rows * columns * pixelBytes, &RegionStats::reconstructedBytes);
    std::vector<uint8_t> result(static_cast<size_t>(rows * columns * pixelBytes));
    if (!rows || !columns) return result;
    const auto& layout = found->second;
    if (!layout.tile) {
        for (int64_t r = 0; r < rows; ++r) {
            const auto bytes = read(name, ((row + r) * shape[1] + column) * pixelBytes, columns * pixelBytes);
            std::memcpy(result.data() + r * columns * pixelBytes, bytes.data(), bytes.size());
        }
        return result;
    }
    for (int64_t ty = row / layout.tile; ty <= (row + rows - 1) / layout.tile; ++ty)
        for (int64_t tx = column / layout.tile; tx <= (column + columns - 1) / layout.tile; ++tx) {
            const auto index = ty * layout.tileColumns() + tx;
            const auto [y, x, h, w] = layout.rectangle(index);
            const auto left = std::max(column, x), right = std::min(column + columns, x + w);
            visitChunk(name, index, [&](auto bytes) {
                for (auto r = std::max(row, y); r < std::min(row + rows, y + h); ++r)
                    std::memcpy(result.data() + ((r - row) * columns + left - column) * pixelBytes,
                        bytes.data() + ((r - y) * w + left - x) * pixelBytes, (right - left) * pixelBytes);
            });
        }
    return result;
}

} // namespace detail

std::vector<uint8_t> ObjectStore::readGeometryArray(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, int64_t byteOffset, const std::optional<int64_t>& byteCount) const
{
    (void)connection(txn);
    if (byteCount && *byteCount <= detail::geometryChunkSize)
        return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
            return reader.read(arrayName, byteOffset, byteCount);
        });
    return detail::GeometryReader(txn, revisionUuid, false).read(arrayName, byteOffset, byteCount);
}

ObjectStore::GeometryArrayCounts ObjectStore::geometryArrayCounts(const Txn& txn,
    const std::string& kind, const std::string& name, const std::optional<int64_t>& limit) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    GeometryArrayCounts result;
    std::vector<std::string> revisions;
    std::vector<size_t> positions;
    std::vector<std::pair<std::string,std::string>> expected;
    const auto sources=querySources(db);
    const bool hasArchive=std::any_of(sources.begin(),sources.end(),[](auto* source){return historyEnabled(source);});
    const auto flush=[&]{
        if(revisions.empty())return;
        auto records=resolveGeometryRecords(db,revisions);
        for(size_t i=0;i<records.size();++i){
            if(!records[i])throw std::invalid_argument("geometry: revision not found");
            if(records[i]->revision.objectUuid!=expected[i].first||
               records[i]->revision.contentHash!=expected[i].second)
                throw std::runtime_error("geometry: current descriptor disagrees with its reference");
            const auto& arrays=records[i]->descriptor.at("arrays");
            if(arrays.contains(name)) {
                const auto layout=geometryLayout(arrays.at(name));
                (void)layout;
                const auto shape=arrays.at(name).at("shape").get<std::vector<int64_t>>();
                if(shape.empty())throw std::invalid_argument("geometry: array has no leading dimension");
                result.counts[positions[i]]=shape[0];
            }
        }
        revisions.clear();positions.clear();expected.clear();
    };
    if(limit&&*limit<0)throw std::invalid_argument("geometry catalog limit must be nonnegative");
    ObjectSelection selection;selection.kind=kind;
    // The common composed cursor supplies visibility and UUID ordering. Read
    // the current descriptor in that SAME selected row, not another lookup per
    // fiber or even another lookup per payload batch. An absent selected head
    // remains absent, even if an older archive/base still retains its payload.
    VisibleObjectRows rows(db,[&](auto* source){return objectSelectionQuery(source,selection,
        "id,uuid,storage_kind,storage_ref,uuid,NULL,data_checksum,geometry_manifest,version,changed_by,changed_at,changed_host");},1);
    while((!limit||result.counts.size()<size_t(*limit))&&rows.row()){
        auto& row=rows.values();
        result.uuids.push_back(row.blob16(1));result.counts.push_back(0);
        if(row.text(2)!="sqlite"||row.isNull(3)||row.isNull(7))continue;
        const auto record=decodeGeometryRecord(row,3);
        for(const auto& array:record.revision.arrays)if(array.name==name){
            (void)geometryLayout(record.descriptor.at("arrays").at(name));
            if(array.shape.empty())throw std::invalid_argument("geometry: array has no leading dimension");
            result.counts.back()=array.shape[0];break;
        }
        // Current descriptors are authoritative, but retained archives must
        // agree just as in currentGeometry(). Resolve those checks in bulk;
        // history-free imports need no second descriptor lookup.
        if(hasArchive){
            positions.push_back(result.counts.size()-1);revisions.push_back(record.revision.uuid);
            expected.emplace_back(record.revision.objectUuid,record.revision.contentHash);
        }
        if(revisions.size()==1024)flush();
    }
    flush();return result;
}

namespace {
PackedGeometryArray readPackedGeometryArrays(sqlite3* db,
    const std::vector<std::string>& revisions, const std::string& name,
    std::span<const ObjectRevision> selected={})
{
    if(!selected.empty()&&selected.size()!=revisions.size())throw std::invalid_argument("geometry: selected payload count differs");
    PackedGeometryArray result;
    struct ArrayPlan {
        GeometryLayout layout;
        std::vector<std::string> hashes;
        size_t offset;
        std::unique_ptr<GeometryReader> reader;
    };
    std::vector<ArrayPlan> plan;
    plan.reserve(revisions.size());
    size_t totalBytes = 0;
    constexpr size_t window = 256;
    // The storage backend resolves both single and batched references. Keep
    // batched payload I/O for full snapshots; deltas use their normal reader.
    for(size_t begin=0;begin<revisions.size();begin+=window) {
        const auto count=std::min(window,revisions.size()-begin);
        auto records=resolveGeometryRecords(db,std::span(revisions).subspan(begin,count));
        for(size_t i=0;i<count;++i) {
            if(!records[i])throw std::invalid_argument("geometry: revision not found");
            if(!selected.empty()){
                const auto& expected=selected[begin+i];
                if(records[i]->revision.objectUuid!=expected.objectUuid||
                   (expected.record.dataChecksum&&records[i]->revision.contentHash!=*expected.record.dataChecksum))
                    throw std::runtime_error("geometry: selected descriptor disagrees with its reference");
            }
            auto manifest=std::move(records[i]->descriptor);
            std::unique_ptr<GeometryReader> reader;
            if(manifest.contains("delta") || (manifest.at("arrays").contains(name) &&
                manifest.at("arrays").at(name).contains("source_revision"))) {
                reader=std::make_unique<GeometryReader>(db,revisions[begin+i],false);
                manifest=reader->manifest();
            }
            const auto& descriptors=manifest.at("arrays");
            if(!descriptors.contains(name))throw std::invalid_argument("geometry: array not found: "+name);
            GeometryLayout layout;
            for(const auto& [key,array]:descriptors.items()) {
                const auto validated=geometryLayout(array);
                if(key==name)layout=validated;
            }
            const auto& array=descriptors.at(name);
            const auto shape=array.at("shape").get<std::vector<int64_t>>();
            const auto dtype=array.at("dtype").get<std::string>();
            if(result.offsets.size()==1){result.dtype=dtype;result.shape=shape;result.shape[0]=0;}
            if(dtype!=result.dtype||shape.size()!=result.shape.size()||
                !std::equal(shape.begin()+1,shape.end(),result.shape.begin()+1))
                throw std::invalid_argument("geometry: packed arrays require matching dtype and trailing dimensions");
            if(shape[0]>std::numeric_limits<int64_t>::max()-result.shape[0]||
                uint64_t(layout.length)>result.data.max_size()-totalBytes)
                throw std::overflow_error("geometry: packed array size overflow");
            result.shape[0]+=shape[0];result.offsets.push_back(uint64_t(result.shape[0]));
            plan.push_back({layout,array.at("chunks").get<std::vector<std::string>>(),totalBytes,std::move(reader)});
            totalBytes+=size_t(layout.length);
        }
    }
    result.data.resize(totalBytes);
    struct ChunkRequest {size_t array;int64_t index;};
    std::vector<ChunkRequest> requested;
    const auto flush=[&]{
        if(requested.empty())return;
        std::vector<std::string> hashes;
        hashes.reserve(requested.size());
        for(const auto& q:requested)hashes.push_back(plan[q.array].hashes.at(q.index));
        visitGeometryPayloads(db,hashes,[&](size_t i,std::span<const uint8_t> bytes){
            const auto& q=requested.at(i);
            const auto& array=plan[q.array];
            const auto& hash=hashes[i];
            if(geometryWork)geometryWork->readBytes[name]+=bytes.size();
            verifyGeometryChunk(bytes,hash,hash,array.layout.chunkBytes(q.index),name);
            copyGeometryChunk(array.layout,q.index,bytes,result.data.data()+array.offset);
        });
        requested.clear();
    };
    for(size_t i=0;i<plan.size();++i) {
        if(plan[i].reader) {
            const auto bytes=plan[i].reader->read(name,0);
            std::copy(bytes.begin(),bytes.end(),result.data.begin()+plan[i].offset);
            continue;
        }
        for(int64_t j=0;j<plan[i].layout.chunkCount();++j) {
            requested.push_back({i,j});
            if(requested.size()==window)flush();
        }
    }
    flush();
    return result;
}
}

PackedGeometryArray ObjectStore::readGeometryArrays(const Txn& txn,
    const std::vector<std::string>& revisions, const std::string& name) const
{
    return readPackedGeometryArrays(static_cast<sqlite3*>(connection(txn)),revisions,name);
}

PackedGeometryArray ObjectStore::readSelectedGeometryArrays(const Txn& txn,
    std::span<const ObjectRevision> selected,const std::string& name,const std::optional<std::string>& volume) const
{
    std::vector<std::string> revisions;revisions.reserve(selected.size());
    for(const auto& object:selected){
        if(object.deleted||!object.geometryRevision)throw std::invalid_argument("geometry: selected object has no stored geometry");
        revisions.push_back(*object.geometryRevision);
    }
    auto packed=readPackedGeometryArrays(static_cast<sqlite3*>(connection(txn)),revisions,name,selected);
    if(!volume)return packed;
    if(name!="line_points"||std::any_of(selected.begin(),selected.end(),[](const auto& o){return o.record.kind!="fiber";}))
        throw std::invalid_argument("mapped packed geometry currently supports float64 fiber line_points only");
    if(selected.empty()){packed.dtype="<f8";packed.shape={0,3};return packed;}
    if(packed.dtype!="<f8"||packed.shape.size()!=2||packed.shape[1]!=3)
        throw std::invalid_argument("mapped packed geometry currently supports float64 fiber line_points only");
    std::optional<VolumeTransforms> transforms;
    for(size_t item=0;item<selected.size();++item){
        const auto meta=nlohmann::json::parse(selected[item].record.meta);
        const auto scale=geometryScaleToVolume(meta,*volume);
        std::optional<VolumeAffine> mapping;
        if(!scale){
            if(!transforms)transforms=volumeTransforms(txn);
            mapping=transforms->toVolume(meta,*volume);
            if(!mapping)throw std::invalid_argument("selected geometry has no unambiguous mapping to target volume");
        }
        for(uint64_t point=packed.offsets[item];point<packed.offsets[item+1];++point){
            const auto offset=size_t(point)*3*sizeof(double);
            std::array<double,3> value;
            detail::copyScalars<double>(std::span<const uint8_t>(packed.data).subspan(offset,3*sizeof(double)),value);
            if(scale){for(auto& x:value)x*=*scale;}else value=mapping->apply(value);
            // Keep little-endian packed bytes and the same scalar arithmetic as
            // the prior NumPy scale / native transformed-view paths.
            detail::copyScalarBytes<double>(value,std::span<uint8_t>(packed.data).subspan(offset,3*sizeof(double)));
        }
    }
    return packed;
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRanges(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        return reader.ranges(arrayName, ranges);
    });
}

std::vector<uint8_t> ObjectStore::readGeometryRectangle(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, int64_t row, int64_t column, int64_t rows, int64_t columns) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        return reader.rectangle(arrayName, row, column, rows, columns);
    });
}

detail::SurfaceTile detail::surfaceTile(GeometryReader& reader, int64_t row, int64_t column, int64_t rows, int64_t columns)
{
    SurfaceTile tile;
    tile.points = xyzValues<float>(reader.rectangle("points", row, column, rows, columns));
    tile.valid.resize(tile.points.size());
    const auto channel = reader.manifest().at("arrays").contains(surfaceValidChannel)
        ? reader.rectangle(surfaceValidChannel, row, column, rows, columns) : std::vector<uint8_t>{};
    for (size_t i = 0; i < tile.points.size(); ++i)
        tile.valid[i] = vc::geometry::validSurfacePoint(tile.points[i]) && (channel.empty() || channel[i]);
    boundSurfaceTileBands(tile, columns);
    return tile;
}

void detail::boundSurfaceTileBands(SurfaceTile& tile, int64_t columns)
{
    tile.bands.clear();
    const int64_t rows = columns ? int64_t(tile.points.size()) / columns : 0;
    for (int64_t first = 0; first + 1 < rows; first += SurfaceTile::bandCells) {
        const int64_t last = std::min(first + SurfaceTile::bandCells, rows - 1);
        std::optional<std::array<double, 6>> box;
        for (int64_t r = first; r <= last; ++r)
            for (int64_t c = 0; c < columns; ++c) {
                const auto i = size_t(r * columns + c);
                if (!tile.valid[i]) continue;
                const auto& p = tile.points[i];
                if (!box) box = std::array<double, 6>{p[0], p[1], p[2], p[0], p[1], p[2]};
                else for (int k = 0; k < 3; ++k) {
                    (*box)[k] = std::min((*box)[k], double(p[k]));
                    (*box)[k + 3] = std::max((*box)[k + 3], double(p[k]));
                }
            }
        tile.bands.push_back(box);
    }
}

std::shared_ptr<const detail::SurfaceTile> detail::surfaceTile(SurfaceTileCache* cache, const std::string& revision,
                                                               GeometryReader& reader, int64_t row, int64_t column,
                                                               int64_t rows, int64_t columns)
{
    // Current history-off references keep the same object version throughout
    // one transaction. Distinct edits therefore need distinct cache keys even
    // while the reference is unchanged; use the existing descriptor checksum.
    const auto& key=revision;
    if (cache)
        if (auto tile = cache->find(key, row, column, rows, columns)) return tile;
    auto tile = std::make_shared<const SurfaceTile>(surfaceTile(reader, row, column, rows, columns));
    if (cache) cache->insert(key, row, column, rows, columns, tile);
    return tile;
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRectangles(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 4>>& rectangles) const
{
    (void)connection(txn);
    return detail::withGeometryReader(txn, revisionUuid, [&](auto& reader) {
        std::vector<std::vector<uint8_t>> result;
        result.reserve(rectangles.size());
        for (const auto& r : rectangles) result.push_back(reader.rectangle(arrayName, r[0], r[1], r[2], r[3]));
        return result;
    });
}

} // namespace vc::objects
