#include "vc_object_store/objects/ObjectStore.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/Extensions.hpp"
#include "internal/RecordSql.hpp"
#include "internal/History.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/GeometryInternal.hpp"
#include <nlohmann/json.hpp>
#include <set>

namespace vc::objects {
namespace {
using namespace detail;
using Json = nlohmann::json;
constexpr const char* snapshotDomain = "@extension-rows-v1";

Json manifest(const std::string& metadata)
{
    const auto meta = Json::parse(metadata);
    const auto found = meta.find(extensionManifestKey);
    if (found == meta.end()) return Json::object();
    if (!found->is_object()) throw std::runtime_error("extension: invalid owner snapshot manifest");
    return *found;
}

ExtensionTableInfo versionedTable(sqlite3* db, const std::string& name, const char* ownerTable = nullptr)
{
    const auto info = extensionTable(db, name);
    if (!info || !info->versioned || info->keySpace.empty())
        throw std::invalid_argument("extension: operation requires a versioned owner table");
    if (ownerTable && info->keySpace != ownerTable)
        throw std::invalid_argument("extension: version reference has the wrong owner kind");
    return *info;
}

std::vector<ExtensionRow> currentRows(sqlite3* db, const std::string& name, const std::string& owner)
{
    std::vector<ExtensionRow> rows;
    const auto key = extensionUuid(owner);
    detail::scanExtensionRows(db, name, [&](const auto& row) { rows.push_back(row); return true; }, std::span(&key, 1));
    return rows;
}

std::vector<uint8_t> encodeRows(std::span<const ExtensionRow> rows)
{
    Json encoded = Json::array();
    for (const auto& row : rows) {
        Json cells = Json::array();
        for (const auto& value : row) {
            Json cell = Json::array();
            cell.push_back(value.index());
            std::visit([&](const auto& item) {
                using T = std::decay_t<decltype(item)>;
                if constexpr (std::is_same_v<T, std::monostate>) cell.push_back(nullptr);
                else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) cell.push_back(Json::binary(item));
                else cell.push_back(item);
            }, value);
            cells.push_back(std::move(cell));
        }
        encoded.push_back(std::move(cells));
    }
    return Json::to_cbor(encoded);
}

std::string snapshotSchema(const ExtensionTableInfo& table)
{
    Json columns = Json::array();
    for (const auto& column : table.columns)
        columns.push_back({column.name, int(column.type), column.nullable});
    const Json schema{{"name", table.name}, {"extension", table.extension},
        {"format_version", table.formatVersion}, {"columns", std::move(columns)},
        {"primary_key", table.primaryKey}, {"key_space", table.keySpace},
        {"versioned", table.versioned}, {"provisioning", Json::parse(table.provisioning)}};
    const auto bytes = Json::to_cbor(schema);
    return geometryPayloadHash("@extension-schema-v1", bytes);
}

Json storeSnapshot(sqlite3* db, const ExtensionTableInfo& table, std::span<const ExtensionRow> rows)
{
    const auto bytes = encodeRows(rows);
    Json chunks = Json::array();
    for (size_t offset = 0; offset < bytes.size(); offset += geometryChunkSize) {
        const auto length = std::min(geometryChunkSize, bytes.size() - offset);
        chunks.push_back(storeSharedPayload(db, snapshotDomain, std::span(bytes).subspan(offset, length)));
    }
    return {{"schema_hash", snapshotSchema(table)}, {"format_version", table.formatVersion},
            {"encoding", "cbor-rows-v1"}, {"chunks", std::move(chunks)},
            {"bytes", bytes.size()}, {"count", rows.size()}};
}

std::vector<ExtensionRow> readSnapshot(sqlite3* db, const ExtensionTableInfo& table,
                                     const std::string& owner, const Json& descriptor)
{
    if (descriptor.at("schema_hash").get<std::string>() != snapshotSchema(table) ||
        descriptor.at("format_version").get<int64_t>() != table.formatVersion ||
        descriptor.at("encoding") != "cbor-rows-v1")
        throw std::runtime_error("extension: historical snapshot schema mismatch");
    const auto expectedBytes = descriptor.at("bytes").get<uint64_t>();
    std::vector<uint8_t> bytes;
    for (const auto& hash : descriptor.at("chunks")) {
        const auto chunk = readSharedPayload(db, snapshotDomain, hash.get<std::string>());
        if (chunk.empty() || chunk.size() > geometryChunkSize ||
            bytes.size() > expectedBytes || chunk.size() > expectedBytes - bytes.size())
            throw std::runtime_error("extension: invalid snapshot chunk size");
        bytes.insert(bytes.end(), chunk.begin(), chunk.end());
    }
    if (bytes.size() != expectedBytes) throw std::runtime_error("extension: incomplete snapshot");
    const auto encoded = Json::from_cbor(bytes);
    if (!encoded.is_array() || encoded.size() != descriptor.at("count").get<uint64_t>())
        throw std::runtime_error("extension: invalid snapshot row count");
    const auto keyColumn = std::find_if(table.columns.begin(), table.columns.end(),
        [&](const auto& column) { return column.name == table.primaryKey.front(); });
    const auto keyIndex = size_t(keyColumn - table.columns.begin());
    const auto ownerKey = extensionUuid(owner);
    std::vector<ExtensionRow> rows;
    rows.reserve(encoded.size());
    for (const auto& cells : encoded) {
        if (!cells.is_array() || cells.size() != table.columns.size())
            throw std::runtime_error("extension: invalid snapshot columns");
        ExtensionRow row;
        row.reserve(cells.size());
        for (const auto& cell : cells) {
            if (!cell.is_array() || cell.size() != 2) throw std::runtime_error("extension: invalid snapshot cell");
            switch (cell.at(0).get<unsigned>()) {
            case 0: row.emplace_back(std::monostate{}); break;
            case 1: row.emplace_back(cell.at(1).get<int64_t>()); break;
            case 2: row.emplace_back(cell.at(1).get<double>()); break;
            case 3: row.emplace_back(cell.at(1).get<std::string>()); break;
            case 4: {
                const auto& data = cell.at(1).get_binary();
                row.emplace_back(std::vector<uint8_t>(data.begin(), data.end()));
                break;
            }
            default: throw std::runtime_error("extension: invalid snapshot cell type");
            }
        }
        if (row.at(keyIndex) != ownerKey) throw std::runtime_error("extension: snapshot owner mismatch");
        rows.push_back(std::move(row));
    }
    return rows;
}

void scanSnapshot(sqlite3* db, const ExtensionTableInfo& table, const std::string& owner,
                  const std::string& meta, const std::function<bool(const ExtensionRow&)>& visitor)
{
    if (!visitor) throw std::invalid_argument("extension: empty scan visitor");
    const auto descriptors = manifest(meta);
    const auto found = descriptors.find(table.name);
    if (found == descriptors.end()) return;
    for (const auto& row : readSnapshot(db, table, owner, *found)) if (!visitor(row)) break;
}
} // namespace

void ObjectStore::replaceExtensionRows(WriteTxn& txn, const std::string& name, const std::string& owner,
                                       std::span<const ExtensionRow> rows)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto table = versionedTable(db, name);
    Statement existing(db, "SELECT id,meta FROM " + table.keySpace + " WHERE uuid=? AND state<>2", parseUuid(owner));
    if (!existing.row()) throw std::invalid_argument("extension: owner is not present in the writable store");
    const auto id = existing.i64(0);
    auto meta = Json::parse(existing.text(1));
    const auto oldRows = currentRows(db, name, owner);
    Savepoint save(db);
    detail::replaceCurrentExtensionRows(db, name, owner, rows);
    // SQL validates the proposed data and canonicalizes its key order. A no-op
    // rolls back even the table's mutation stamp, not just the owner version.
    const auto orderedRows = currentRows(db, name, owner);
    if (oldRows == orderedRows) return;
    auto descriptors = manifest(meta.dump());
    if (orderedRows.empty()) descriptors.erase(name);
    else descriptors[name] = storeSnapshot(db, table, orderedRows);
    if (descriptors.empty()) meta.erase(extensionManifestKey);
    else meta[extensionManifestKey] = std::move(descriptors);
    if (table.keySpace == "objects") beforeCurrentObjectEdit(db, owner);
    execute(db, "UPDATE " + table.keySpace + " SET meta=? WHERE id=?", meta.dump(), id);
    stampContentChange(db, author(), hostname(), table.keySpace.c_str(), id, parseUuid(owner), "extension");
    if (table.keySpace == "edges") recordRelationshipValidation(db, *edgeByUuid(txn, owner));
    save.release();
}

void ObjectStore::scanExtensionRowsAtVersion(const Txn& txn, const std::string& name, const ObjectVersion& version,
                                             const std::function<bool(const ExtensionRow&)>& visitor) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto table = versionedTable(db, name, "objects");
    const auto revision = objectAtVersion(txn, version);
    if (!revision) throw std::invalid_argument("extension: object version is unavailable");
    if (revision->deleted) return;
    scanSnapshot(db, table, version.objectUuid, revision->record.meta, visitor);
}

void ObjectStore::scanExtensionRowsAtVersion(const Txn& txn, const std::string& name, const RelationshipVersion& version,
                                             const std::function<bool(const ExtensionRow&)>& visitor) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto table = versionedTable(db, name, "edges");
    const auto revision = relationshipAtVersion(txn, version);
    if (!revision) throw std::invalid_argument("extension: relationship version is unavailable");
    scanSnapshot(db, table, version.relationshipUuid, revision->record.meta, visitor);
}

void detail::checkExtensionMetadata(const std::string& previous, const std::string& replacement)
{
    if (!mayContainExtensionManifest(previous) && !mayContainExtensionManifest(replacement)) return;
    const auto proposed = Json::parse(replacement);
    const auto old = Json::parse(previous);
    const auto before = old.find(extensionManifestKey), after = proposed.find(extensionManifestKey);
    if ((before == old.end()) != (after == proposed.end()) ||
        (before != old.end() && *before != *after))
        throw std::invalid_argument("extension: vcobj_extensions is reserved; use replaceExtensionRows");
}

void detail::restoreCurrentExtensionRows(sqlite3* db, const char* ownerTable, const std::string& owner,
                                         const std::string& sourceMeta)
{
    const auto descriptors = manifest(sourceMeta);
    for (const auto& [name, descriptor] : descriptors.items())
        (void)versionedTable(db, name, ownerTable);
    for (const auto& table : extensionTables(db)) {
        if (!table.versioned || table.keySpace != ownerTable) continue;
        const auto found = descriptors.find(table.name);
        const auto rows = found == descriptors.end() ? std::vector<ExtensionRow>{} : readSnapshot(db, table, owner, *found);
        if (currentRows(db, table.name, owner) != rows) replaceCurrentExtensionRows(db, table.name, owner, rows);
    }
}

void detail::clearCurrentExtensionRows(sqlite3* db, const char* ownerTable, const std::string& owner)
{
    for (const auto& table : extensionTables(db))
        if (table.versioned && table.keySpace == ownerTable && !currentRows(db, table.name, owner).empty())
            replaceCurrentExtensionRows(db, table.name, owner, {});
}

std::vector<std::string> detail::currentExtensionChunks(sqlite3* db)
{
    if (!hasExtensionTables(db)) return {};
    std::set<std::string> hashes;
    for (const auto* ownerTable : {"objects", "edges"}) {
        Statement owners(db, std::string("SELECT uuid,meta FROM ") + ownerTable +
            " WHERE state<>2 AND json_type(meta,'$.vcobj_extensions') IS NOT NULL");
        while (owners.row()) {
          const auto descriptors = manifest(owners.text(1));
          for (const auto& [name, descriptor] : descriptors.items()) {
            const auto table = versionedTable(db, name, ownerTable);
            // Validate before removing any historical chunks. These also work
            // for chunks routed to an immutable linked store.
            (void)readSnapshot(db, table, formatUuid(owners.blob16(0)), descriptor);
            for (const auto& hash : descriptor.at("chunks")) hashes.insert(hash.get<std::string>());
          }
        }
    }
    return {hashes.begin(), hashes.end()};
}

void detail::materializeExtensionOwner(sqlite3* source, ObjectStore& target, WriteTxn& txn,
                                       const char* ownerTable, const std::string& owner,
                                       const std::string& sourceMeta)
{
    const auto descriptors = manifest(sourceMeta);
    for (const auto& [name, descriptor] : descriptors.items()) {
        const auto table = versionedTable(source, name, ownerTable);
        const auto rows = readSnapshot(source, table, owner, descriptor);
        ExtensionTable spec;
        spec.name = table.name; spec.extension = table.extension; spec.formatVersion = table.formatVersion;
        spec.provisioning = table.provisioning; spec.basedOn = table.basedOn; spec.columns = table.columns;
        spec.primaryKey = table.primaryKey; spec.keySpace = table.keySpace; spec.versioned = true;
        spec.indexes = table.indexes;
        target.createExtensionTable(txn, spec);
        target.replaceExtensionRows(txn, name, owner, rows);
    }
}
} // namespace vc::objects
