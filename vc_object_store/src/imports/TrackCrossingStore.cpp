#include "vc_object_store/objects/TrackCrossingStore.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "vc_object_store/objects/imports/ParallelImportStream.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/EdgeLayout.hpp"
#include "internal/Extensions.hpp"
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>

namespace vc::objects {
namespace {
using Json = nlohmann::json;
// Fixed namespace of crossing-cache edge UUIDs. Never change it: the reader
// checks every edge against the name UUID of its endpoints.
constexpr TrackCrossingUuid kEdgeNamespace{0x6b,0x1f,0x0c,0x52,0x8e,0x0d,0x4c,0x1b,0x9a,0x37,0x2d,0x64,0xf1,0x5a,0x90,0xc3};
constexpr size_t kTaskRows = 65536;

TrackCrossingUuid uuidAt(std::span<const uint8_t> bytes, size_t row)
{
    TrackCrossingUuid value;
    std::memcpy(value.data(), bytes.data() + 16 * row, 16);
    return value;
}
bool storable(double value) { return std::isfinite(value) && !std::signbit(value); }

void requireShape(const ExtensionTableInfo& info)
{
    const auto expected = trackCrossingExtension({});
    if (info.extension != expected.extension || info.formatVersion != kTrackCrossingFormat || info.columns != expected.columns
        || info.primaryKey != expected.primaryKey || info.keySpace != expected.keySpace || info.versioned)
        throw std::invalid_argument("unsupported track crossing table format");
}
TrackCrossingStoreInfo infoOf(const ExtensionTableInfo& table)
{
    const auto provisioning = Json::parse(table.provisioning);
    TrackCrossingStoreInfo info;
    info.algorithm = provisioning.at("algorithm").get<std::string>();
    info.parameters = provisioning.at("parameters").dump();
    info.key = provisioning.at("key").get<std::string>();
    info.checksum = provisioning.at("checksum").get<std::string>();
    info.tracks = provisioning.at("tracks").get<int64_t>();
    info.crossings = provisioning.at("crossings").get<int64_t>();
    info.basedOn = table.basedOn;
    return info;
}

// The data shards' table descriptors, in partition order (nullopt: the
// manifest-only root of a multi-shard store, which holds no rows). Rejects
// every collection that is not exactly one crossing cache.
std::vector<std::optional<ExtensionTableInfo>> cacheShape(const ObjectCollection& store, const CollectionTxn& txn,
    const std::vector<detail::CollectionScanPartition>& parts)
{
    if (!store.supportsTrustedGeometryReads(txn))
        throw std::invalid_argument("track crossing store must be sealed and history-off");
    const auto members = store.members();
    size_t roots = 0;
    for (const auto& member : members) {
        if (member.storeUuid != members.front().storeUuid)
            throw std::invalid_argument("track crossing store must not reference other stores");
        roots += !member.partition;
    }
    if (roots != 1 || parts.size() != members.size())
        throw std::invalid_argument("track crossing store must be one root with its own partitions");
    std::vector<std::optional<ExtensionTableInfo>> tables(parts.size());
    const ExtensionTableInfo* first = nullptr;
    size_t manifests = 0;
    for (size_t n = 0; n < parts.size(); ++n) {
        tables[n] = detail::extensionTable(parts[n].db, kTrackCrossingTable);
        if (!tables[n]) {
            // Only the manifest of several shards may lack the table, and it holds no edges.
            detail::Statement count(parts[n].db, "SELECT count(*) FROM edges");
            count.row();
            if (parts.size() == 1 || ++manifests > 1 || count.i64(0) != 0)
                throw std::invalid_argument("store has no track crossing table in every data shard");
            continue;
        }
        requireShape(*tables[n]);
        if (first && (Json::parse(first->provisioning) != Json::parse(tables[n]->provisioning) || first->basedOn != tables[n]->basedOn))
            throw std::invalid_argument("track crossing shards disagree on their provisioning");
        if (!first) first = &*tables[n];
    }
    if (!first) throw std::invalid_argument("store has no track crossing table");
    infoOf(*first); // a malformed provisioning object is rejected here
    return tables;
}
} // namespace

ExtensionTable trackCrossingExtension(const TrackCrossingStoreInfo& info)
{
    auto parameters = Json::parse(info.parameters.empty() ? std::string("{}") : info.parameters);
    if (!parameters.is_object()) throw std::invalid_argument("track crossing parameters must be a JSON object");
    ExtensionTable table;
    table.name = kTrackCrossingTable;
    table.extension = kTrackCrossingExtension;
    table.formatVersion = kTrackCrossingFormat;
    // Explicit small descriptor: identity, interpretation and counts only.
    table.provisioning = Json{{"algorithm", info.algorithm}, {"parameters", std::move(parameters)},
        {"units", "stored_voxels"}, {"key", info.key}, {"checksum", info.checksum},
        {"tracks", info.tracks}, {"crossings", info.crossings}}.dump();
    table.basedOn = info.basedOn;
    table.columns = {{"edge_uuid", ExtensionType::Blob}, {"local_a", ExtensionType::Integer},
        {"local_b", ExtensionType::Integer}, {"position_a", ExtensionType::Real},
        {"position_b", ExtensionType::Real}, {"clearance", ExtensionType::Real}};
    table.primaryKey = {"edge_uuid"};
    table.keySpace = "edges";
    return table;
}

std::string trackCrossingEdgeUuid(const TrackCrossingUuid& a, const TrackCrossingUuid& b)
{
    const bool ordered = a < b;
    char name[32];
    std::memcpy(name, (ordered ? a : b).data(), 16);
    std::memcpy(name + 16, (ordered ? b : a).data(), 16);
    return util::nameUuid(kEdgeNamespace, std::string_view(name, sizeof name));
}

InitialImportResult writeTrackCrossingStore(const std::filesystem::path& destination,
    const TrackCrossingColumnsView& view, TrackCrossingStoreInfo info, const TrackCrossingWriteOptions& write)
{
    const size_t rows = view.rows;
    if (view.a.size() != 16 * rows || view.b.size() != 16 * rows || view.localA.size() != rows || view.localB.size() != rows
        || view.positionA.size() != rows || view.positionB.size() != rows || view.clearance.size() != rows)
        throw std::invalid_argument("track crossing columns do not hold one value per row");
    if (!write.workers || write.workers > 32) throw std::invalid_argument("track crossing workers must be in 1..32");
    if (info.algorithm.empty() || info.key.empty()) throw std::invalid_argument("track crossing store needs an algorithm and a key");
    if (rows > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("too many track crossings for one store");
    const size_t tasks = (rows + kTaskRows - 1) / kTaskRows;
    // Reject bad input before anything is created, and name every pair's edge.
    std::vector<TrackCrossingUuid> ids(rows);
    parallelFor(tasks, write.workers, [&](size_t task) {
        for (size_t i = task * kTaskRows; i < std::min(rows, (task + 1) * kTaskRows); ++i) {
            if (!std::memcmp(view.a.data() + 16 * i, view.b.data() + 16 * i, 16))
                throw std::invalid_argument("track crossing has one track at both ends");
            if (view.localA[i] < 0 || view.localB[i] < 0)
                throw std::invalid_argument("track crossing point index is negative");
            if (!storable(view.positionA[i]) || !storable(view.positionB[i]) || !storable(view.clearance[i]))
                throw std::invalid_argument("track crossing position or clearance is negative or not finite");
            ids[i] = detail::parseUuid(trackCrossingEdgeUuid(uuidAt(view.a, i), uuidAt(view.b, i)));
        }
    });
    // Rows are emitted in edge-UUID order. Both tables are keyed by that
    // UUID, so every shard's writer appends near the end of its B-trees
    // instead of inserting at random pages (name UUIDs are uniform).
    std::vector<uint32_t> order(rows);
    {
        std::array<size_t,257> bucket{};
        for (const auto& id : ids) ++bucket[size_t(id[0]) + 1];
        for (size_t i = 0; i < 256; ++i) bucket[i + 1] += bucket[i];
        auto next = bucket;
        for (size_t i = 0; i < rows; ++i) order[next[ids[i][0]]++] = uint32_t(i);
        parallelFor(256, write.workers, [&](size_t byte) {
            std::sort(order.begin() + bucket[byte], order.begin() + bucket[byte + 1],
                      [&](uint32_t left, uint32_t right) { return ids[left] < ids[right]; });
        });
    }
    info.crossings = int64_t(rows);

    InitialImportOptions options;
    options.store.author = "track-crossings";
    options.store.edgeColumns = std::vector<std::string>{}; // slim: uuid, endpoints, kind
    options.store.edgeAdjacencyIndex = false;
    options.shards.resize(write.workers);
    // No objects are stored, but shards still need disjoint ID reservations.
    const auto stride = std::numeric_limits<int64_t>::max() / int64_t(write.workers);
    for (size_t i = 0; i < write.workers; ++i)
        options.shards[i].objectIds = {1 + int64_t(i) * stride, int64_t(i + 1) * stride};
    options.extensions = {trackCrossingExtension(info)};
    options.edges.sqlBatchRows = std::max<size_t>(1, write.sqlBatchRows);
    PackedStoreOptions packed;
    if (write.memoryFirst) {
        packed.disposableBulk = true; packed.memoryFirst = true; packed.backgroundFlush = false;
        packed.residentBytes = write.residentBytes;
    }
    options.packed = packed;

    imports::ParallelImportOptions queues;
    queues.shards = write.workers;
    queues.readers = write.workers;
    queues.batchRows = 4096;
    imports::ParallelImportStream stream(queues);
    std::atomic<size_t> emitted{0};
    stream.start(0, tasks, {}, [&](size_t, size_t task, imports::EdgeEmitter& emitter) {
        const size_t begin = task * kTaskRows, end = std::min(rows, begin + kTaskRows);
        for (size_t slot = begin; slot < end; ++slot) {
            const size_t i = order[slot];
            auto a = uuidAt(view.a, i), b = uuidAt(view.b, i);
            const bool ordered = a < b; // the other order swaps the endpoints' values too
            if (!ordered) std::swap(a, b);
            InitialCollectionEdge edge;
            edge.record.uuid = detail::formatUuid(ids[i]);
            edge.record.kind = "crossing";
            edge.record.aUuid = detail::formatUuid(a);
            edge.record.bUuid = detail::formatUuid(b);
            edge.a = {"", edge.record.aUuid};
            edge.b = {"", edge.record.bUuid};
            edge.extensions.push_back({kTrackCrossingTable, {
                std::vector<uint8_t>(ids[i].begin(), ids[i].end()),
                int64_t(ordered ? view.localA[i] : view.localB[i]), int64_t(ordered ? view.localB[i] : view.localA[i]),
                ordered ? view.positionA[i] : view.positionB[i], ordered ? view.positionB[i] : view.positionA[i],
                view.clearance[i]}, {}});
            const auto bytes = sizeof(InitialCollectionEdge) + edge.record.uuid.size() + edge.record.aUuid.size() * 2
                + edge.record.bUuid.size() * 2 + extensionPayloadBytes(edge.extensions);
            const auto uuid = edge.record.uuid;
            emitter.emit(std::move(edge), bytes, uuid);
        }
        emitted.fetch_add(end - begin);
    });
    options.beforePublish = [&] {
        stream.finish();
        if (emitted.load() != rows) throw std::runtime_error("track crossing import lost rows");
    };
    try {
        return importInitialCollection(destination, options, [](size_t, InitialShardImport&) {},
            [&](size_t shard, InitialShardImport& writer) {
                writer.importEdges([&](std::vector<InitialCollectionEdge>& batch) {
                    return stream.popEdges(shard, batch, [&] { writer.checkCancelled(); });
                });
            });
    } catch (...) {
        stream.fail(std::current_exception());
        throw;
    }
}

TrackCrossingStoreInfo trackCrossingStoreInfo(const ObjectCollection& store, const CollectionTxn& txn)
{
    const auto parts = detail::CollectionScanAccess::partitions(store, txn);
    for (const auto& table : cacheShape(store, txn, parts)) if (table) return infoOf(*table);
    throw std::invalid_argument("store has no track crossing table");
}

TrackCrossingColumns readTrackCrossingColumns(const ObjectCollection& store, const CollectionTxn& txn, size_t workers,
    const std::function<void(const char*,size_t,size_t)>& progress)
{
    using detail::Statement;
    if (!workers) throw std::invalid_argument("track crossing read workers must be positive");
    const auto parts = detail::CollectionScanAccess::partitions(store, txn);
    const auto tables = cacheShape(store, txn, parts);
    const auto crossing = edgeKindCode("crossing");
    std::vector<TrackCrossingColumns> data(parts.size());
    std::mutex progressMutex;
    std::atomic<size_t> read{0};
    // One worker per pinned connection; nothing else uses the collection meanwhile.
    parallelFor(parts.size(), workers, [&](size_t n) {
        if (!tables[n]) return;
        auto* db = parts[n].db;
        auto& out = data[n];
        Statement count(db, "SELECT count(*) FROM edges");
        count.row();
        const auto expected = size_t(count.i64(0));
        out.a.reserve(16 * expected); out.b.reserve(16 * expected);
        for (auto* column : {&out.localA, &out.localB}) column->reserve(expected);
        for (auto* column : {&out.positionA, &out.positionB, &out.clearance}) column->reserve(expected);
        // Both tables are keyed by the edge UUID: walk them together, so a
        // row missing on either side is seen (a join would hide it).
        Statement edges(db, "SELECT uuid,a_uuid,b_uuid,kind," + detail::edgeReadColumn(db, "state") + " FROM edges ORDER BY uuid");
        Statement rows(db, std::string("SELECT edge_uuid,local_a,local_b,position_a,position_b,clearance FROM ")
            + kTrackCrossingTable + " ORDER BY edge_uuid");
        const auto blob = [](const Statement& row, int column) {
            if (sqlite3_column_type(row.value, column) != SQLITE_BLOB || sqlite3_column_bytes(row.value, column) != 16)
                throw std::invalid_argument("track crossing store holds an invalid UUID");
            return row.blob16(column);
        };
        const auto local = [](const Statement& row, int column) {
            if (sqlite3_column_type(row.value, column) != SQLITE_INTEGER)
                throw std::invalid_argument("track crossing point index is not an integer");
            const auto value = row.i64(column);
            if (value < 0 || value > std::numeric_limits<int32_t>::max())
                throw std::invalid_argument("track crossing point index is out of range");
            return int32_t(value);
        };
        const auto real = [](const Statement& row, int column) {
            // SQLite may keep an integral REAL as an integer; NULL and text are refused.
            const auto type = sqlite3_column_type(row.value, column);
            if (type != SQLITE_FLOAT && type != SQLITE_INTEGER)
                throw std::invalid_argument("track crossing position or clearance is not a number");
            const auto value = row.f64(column);
            if (!storable(value)) throw std::invalid_argument("track crossing position or clearance is negative or not finite");
            return value;
        };
        for (;;) {
            const bool edge = edges.row(), row = rows.row();
            if (edge != row) throw std::invalid_argument("track crossing edges and measurements do not correspond");
            if (!edge) break;
            const auto id = blob(edges, 0), a = blob(edges, 1), b = blob(edges, 2);
            if (blob(rows, 0) != id) throw std::invalid_argument("track crossing edges and measurements do not correspond");
            if (sqlite3_column_type(edges.value, 3) != SQLITE_INTEGER || edges.i64(3) != crossing || edges.i64(4) != 0)
                throw std::invalid_argument("track crossing store holds an edge that is no active crossing");
            if (!(a < b) || detail::parseUuid(trackCrossingEdgeUuid(a, b)) != id)
                throw std::invalid_argument("track crossing edge does not identify its pair");
            const auto hash = util::uuidHash64(id);
            for (const auto& route : parts[n].uuidRoutes)
                if (route.keySpace == "edges.uuid" && hash % route.divisor != route.remainder)
                    throw std::invalid_argument("track crossing edge is in another shard than its UUID routes to");
            out.a.insert(out.a.end(), a.begin(), a.end());
            out.b.insert(out.b.end(), b.begin(), b.end());
            out.localA.push_back(local(rows, 1)); out.localB.push_back(local(rows, 2));
            out.positionA.push_back(real(rows, 3)); out.positionB.push_back(real(rows, 4));
            out.clearance.push_back(real(rows, 5));
        }
        out.rows = out.localA.size();
        if (progress) { std::lock_guard lock(progressMutex); progress("crossings", read.fetch_add(out.rows) + out.rows, n); }
    });
    TrackCrossingColumns result;
    for (const auto& part : data) result.rows += part.rows;
    for (const auto& table : tables)
        if (table && infoOf(*table).crossings != int64_t(result.rows))
            throw std::invalid_argument("track crossing store does not hold the rows it declares");
    result.a.reserve(16 * result.rows); result.b.reserve(16 * result.rows);
    for (auto* column : {&result.localA, &result.localB}) column->reserve(result.rows);
    for (auto* column : {&result.positionA, &result.positionB, &result.clearance}) column->reserve(result.rows);
    for (auto& part : data) {
        result.a.insert(result.a.end(), part.a.begin(), part.a.end());
        result.b.insert(result.b.end(), part.b.begin(), part.b.end());
        result.localA.insert(result.localA.end(), part.localA.begin(), part.localA.end());
        result.localB.insert(result.localB.end(), part.localB.begin(), part.localB.end());
        result.positionA.insert(result.positionA.end(), part.positionA.begin(), part.positionA.end());
        result.positionB.insert(result.positionB.end(), part.positionB.begin(), part.positionB.end());
        result.clearance.insert(result.clearance.end(), part.clearance.begin(), part.clearance.end());
        part = {};
    }
    return result;
}
} // namespace vc::objects
