#pragma once
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
using GeometryChunkKey = std::pair<std::string, int64_t>;
struct GeometryByteChange {
    int64_t offset;
    std::vector<uint8_t> before, after;
};
struct GeometryChunkChange {
    GeometryChunkKey key;
    std::string beforeHash, afterHash;
    std::vector<GeometryByteChange> changes;
};
struct EncodedGeometryDelta {
    nlohmann::json description;
    std::vector<uint8_t> bytes;
};
inline std::string hashText(const std::string& text) {
    return geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}
EncodedGeometryDelta encodeGeometryDelta(const std::vector<GeometryChunkChange>&);
nlohmann::json loadGeometryManifest(sqlite3*, const std::string& revision,
                                   const std::string& text, const std::string& contentHash);
std::vector<GeometryChunkChange> decodeGeometryDelta(sqlite3*, const std::string& revision,
                                                    const nlohmann::json& manifest);
void applyGeometryDelta(std::vector<uint8_t>&, const GeometryChunkChange&, bool reverse = false);
void clearGeometryMaterialization(sqlite3*, const std::string& objectUuid);
void materializeGeometry(sqlite3*, const std::string& objectUuid, const std::string& revision);
}
