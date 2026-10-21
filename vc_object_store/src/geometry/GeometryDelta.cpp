#include "geometry/GeometryDelta.hpp"
#include <zstd.h>

namespace vc::objects::detail {
namespace {
constexpr size_t maxDeltaBytes = 256 * 1024 * 1024;
void number(std::vector<uint8_t>& out, uint64_t n) {
    do { out.push_back(uint8_t(n & 127) | (n > 127 ? 128 : 0)); n >>= 7; } while (n);
}
uint64_t number(std::span<const uint8_t> bytes, size_t& pos) {
    uint64_t result = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (pos == bytes.size()) throw std::runtime_error("geometry delta: truncated integer");
        const auto b = bytes[pos++];
        if (shift == 63 && b > 1) throw std::runtime_error("geometry delta: integer overflow");
        result |= uint64_t(b & 127) << shift;
        if (!(b & 128)) return result;
    }
    throw std::runtime_error("geometry delta: invalid integer");
}
}
nlohmann::json loadGeometryManifest(sqlite3* db, const std::string& revision,
                                    const std::string& text, const std::string& contentHash) {
    if (geometryWork) geometryWork->manifestBytesRead += text.size();
    auto manifest = nlohmann::json::parse(text);
    if (!manifest.contains("delta")) return verifiedGeometryManifest(text, contentHash);
    Statement checksum(db, "SELECT manifest_hash FROM geometry_deltas WHERE revision_uuid=?", parseUuid(revision));
    if (!checksum.row() || hashText(text) != checksum.text(0) ||
        manifest.at("delta").at("version") != 1 || manifest.at("delta").at("result_hash") != contentHash)
        throw std::runtime_error("geometry delta: corrupt or unsupported revision manifest");
    return manifest;
}
EncodedGeometryDelta encodeGeometryDelta(const std::vector<GeometryChunkChange>& chunks) {
    std::vector<uint8_t> raw;
    auto descriptions = nlohmann::json::array();
    for (const auto& c : chunks) {
        descriptions.push_back({{"array", c.key.first}, {"index", c.key.second},
                                {"before", c.beforeHash}, {"after", c.afterHash}});
        number(raw, c.changes.size());
        int64_t end = 0;
        for (const auto& change : c.changes) {
            number(raw, change.offset - end); number(raw, change.before.size());
            if (change.before.size() > (maxDeltaBytes - std::min(maxDeltaBytes, raw.size())) / 2)
                throw std::invalid_argument("geometry delta: edit exceeds 256 MiB encoded limit; split the edit");
            raw.insert(raw.end(), change.before.begin(), change.before.end());
            for (size_t i = 0; i < change.before.size(); ++i) raw.push_back(change.before[i] ^ change.after[i]);
            end = change.offset + change.before.size();
        }
    }
    if (raw.size() > maxDeltaBytes) throw std::invalid_argument("geometry delta: edit exceeds encoded limit");
    std::vector<uint8_t> compressed(ZSTD_compressBound(raw.size()));
    const auto size = ZSTD_compress(compressed.data(), compressed.size(), raw.data(), raw.size(), 1);
    if (ZSTD_isError(size)) throw std::runtime_error("geometry delta: compression failed");
    const bool useCompression = size < raw.size();
    compressed.resize(size);
    EncodedGeometryDelta result;
    result.description = {{"version", 1}, {"codec", useCompression ? "zstd" : "raw"},
        {"raw_size", raw.size()}, {"raw_hash", geometryHash(raw)}, {"chunks", descriptions}};
    result.bytes = useCompression ? std::move(compressed) : std::move(raw);
    result.description["payload_hash"] = geometryHash(result.bytes);
    return result;
}
std::vector<GeometryChunkChange> decodeGeometryDelta(sqlite3* db, const std::string& revision,
                                                    const nlohmann::json& manifest) {
    const auto& d = manifest.at("delta");
    const auto rawSize = d.at("raw_size").get<int64_t>();
    if (d.at("version") != 1 || rawSize <= 0 || uint64_t(rawSize) > maxDeltaBytes)
        throw std::runtime_error("geometry delta: unsupported encoding or invalid size");
    if (readRequest) {
        Statement size(db, "SELECT length(data) FROM geometry_deltas WHERE revision_uuid=?", parseUuid(revision));
        if (!size.row()) throw std::runtime_error("geometry delta: missing payload");
        requestBytes(size.i64(0), &FiberRegionStats::payloadBytes);
        requestBytes(rawSize, &FiberRegionStats::reconstructedBytes);
    }
    Statement payload(db, "SELECT data FROM geometry_deltas WHERE revision_uuid=?", parseUuid(revision));
    if (!payload.row()) throw std::runtime_error("geometry delta: missing payload");
    auto bytes = std::span(static_cast<const uint8_t*>(sqlite3_column_blob(payload.value, 0)),
                          size_t(sqlite3_column_bytes(payload.value, 0)));
    if (geometryHash(bytes) != d.at("payload_hash").get<std::string>()) throw std::runtime_error("geometry delta: corrupt payload");
    std::vector<uint8_t> raw;
    if (d.at("codec") == "zstd") {
        if (ZSTD_getFrameContentSize(bytes.data(), bytes.size()) != uint64_t(rawSize) ||
            ZSTD_findFrameCompressedSize(bytes.data(), bytes.size()) != bytes.size())
            throw std::runtime_error("geometry delta: invalid compressed frame");
        raw.resize(size_t(rawSize));
        const auto size = ZSTD_decompress(raw.data(), raw.size(), bytes.data(), bytes.size());
        if (ZSTD_isError(size) || size != raw.size()) throw std::runtime_error("geometry delta: decompression failed");
        bytes = raw;
    } else if (d.at("codec") != "raw") throw std::runtime_error("geometry delta: unsupported codec");
    if (bytes.size() != size_t(rawSize) || geometryHash(bytes) != d.at("raw_hash").get<std::string>())
        throw std::runtime_error("geometry delta: corrupt decoded payload");
    size_t pos = 0;
    std::vector<GeometryChunkChange> result;
    std::set<GeometryChunkKey> seen;
    for (const auto& entry : d.at("chunks")) {
        GeometryChunkChange c{{entry.at("array"), entry.at("index")}, entry.at("before"), entry.at("after"), {}};
        if (!seen.insert(c.key).second) throw std::runtime_error("geometry delta: duplicate chunk");
        // Descriptors omit chunk hashes, but retain shape and physical layout.
        const auto layout = geometryLayout(manifest.at("arrays").at(c.key.first), false);
        const auto length = uint64_t(layout.chunkBytes(c.key.second));
        const auto count = number(bytes, pos);
        if (!count || count > bytes.size() / 4) throw std::runtime_error("geometry delta: invalid change count");
        uint64_t end = 0;
        for (uint64_t i = 0; i < count; ++i) {
            const auto gap = number(bytes, pos), size = number(bytes, pos);
            if (gap > length - end || !size || size > length - end - gap || size > (bytes.size() - pos) / 2)
                throw std::runtime_error("geometry delta: invalid range");
            GeometryByteChange change{int64_t(end + gap), {}, {}};
            change.before.assign(bytes.begin() + pos, bytes.begin() + pos + size); pos += size;
            change.after.resize(size);
            for (size_t j = 0; j < size; ++j) change.after[j] = change.before[j] ^ bytes[pos++];
            end += gap + size;
            c.changes.push_back(std::move(change));
        }
        result.push_back(std::move(c));
    }
    if (pos != bytes.size() || result.empty()) throw std::runtime_error("geometry delta: trailing or empty payload");
    return result;
}
void applyGeometryDelta(std::vector<uint8_t>& bytes, const GeometryChunkChange& c, bool reverse) {
    requestBytes(bytes.size(), &FiberRegionStats::reconstructedBytes);
    if (geometryPayloadHash(c.key.first, bytes) != (reverse ? c.afterHash : c.beforeHash))
        throw std::runtime_error("geometry delta: parent chunk mismatch");
    for (const auto& edit : c.changes) {
        const auto& before = reverse ? edit.after : edit.before;
        const auto& after = reverse ? edit.before : edit.after;
        if (edit.offset < 0 || uint64_t(edit.offset) > bytes.size() || before.size() > bytes.size() - edit.offset ||
            !std::equal(before.begin(), before.end(), bytes.begin() + edit.offset))
            throw std::runtime_error("geometry delta: original bytes mismatch");
        std::copy(after.begin(), after.end(), bytes.begin() + edit.offset);
    }
    if (geometryPayloadHash(c.key.first, bytes) != (reverse ? c.beforeHash : c.afterHash))
        throw std::runtime_error("geometry delta: result chunk mismatch");
}
void clearGeometryMaterialization(sqlite3* db, const std::string& object) {
    for (const auto* table : {"geometry_current_chunks", "geometry_materializations"}) {
        execute(db, std::string("DELETE FROM ") + table + " WHERE object_uuid=?", parseUuid(object));
    }
}
void materializeGeometry(sqlite3* db, const std::string& object, const std::string& revision) {
    Statement row(db, "SELECT manifest,content_hash FROM geometry_revisions WHERE uuid=?", parseUuid(revision));
    if (!row.row()) throw std::runtime_error("geometry: missing revision");
    const auto descriptor = loadGeometryManifest(db, revision, row.text(0), row.text(1));
    clearGeometryMaterialization(db, object);
    if (!descriptor.contains("delta")) return;
    GeometryReader reader(db, revision, false);
    GeometryReader base(db, descriptor.at("delta").at("base").get<std::string>(), false);
    const auto& manifest = reader.manifest();
    Statement insert(db, "INSERT INTO geometry_current_chunks VALUES(?,?,?,?,?)");
    for (const auto& [name, a] : manifest.at("arrays").items())
        for (int64_t index = 0; index < geometryLayout(a).chunkCount(); ++index) {
            const std::string hash = a.at("chunks").at(index);
            if (hash == base.manifest().at("arrays").at(name).at("chunks").at(index).get<std::string>()) continue;
            auto bytes = reader.chunk(name, index);
            insert.run(parseUuid(object), name, index, hash, bytes);
        }
    execute(db, "INSERT INTO geometry_materializations VALUES(?,?,?)", parseUuid(object), parseUuid(revision),
        manifest.dump());
}
}
