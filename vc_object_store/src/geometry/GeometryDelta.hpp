#pragma once
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
using GeometryChunkKey = std::pair<std::string, int64_t>;
// A run of changed bytes in a chunk; its old and new bytes are the next
// `length` bytes of the chunk change's `before` and `after`.
struct GeometryByteChange {
    int64_t offset, length;
};
struct GeometryChunkChange {
    GeometryChunkKey key;
    std::string beforeHash, afterHash;
    std::vector<GeometryByteChange> changes;
    std::vector<uint8_t> before, after; // the runs' bytes, concatenated in order
};
struct EncodedGeometryDelta {
    nlohmann::json description;
    std::vector<uint8_t> bytes;
};
inline std::string hashText(const std::string& text) {
    return geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}
// The largest raw (uncompressed) delta encoded or decoded.
inline constexpr size_t maxGeometryDeltaBytes = 256 * 1024 * 1024;
// The raw size encodeGeometryDelta would produce, without encoding.
size_t geometryDeltaRawSize(const std::vector<GeometryChunkChange>&);
EncodedGeometryDelta encodeGeometryDelta(const std::vector<GeometryChunkChange>&);
nlohmann::json loadGeometryManifest(sqlite3*, const std::string& revision,
                                   const std::string& text, const std::string& contentHash);
std::vector<GeometryChunkChange> decodeGeometryDelta(sqlite3*, const std::string& revision,
                                                    const nlohmann::json& manifest);
void applyGeometryDelta(std::vector<uint8_t>&, const GeometryChunkChange&, bool reverse = false);
void clearGeometryMaterialization(sqlite3*, const std::string& objectUuid);
void materializeGeometry(sqlite3*, const std::string& objectUuid, const std::string& revision);
}
