#pragma once
#include "vc/core/objects/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <set>
#include <span>
#include <cstring>
#include <stdexcept>

namespace vc::objects::detail {
inline constexpr size_t geometryChunkSize = 1024 * 1024;
inline std::string geometryHash(std::span<const uint8_t> bytes)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest, &size, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("geometry: SHA-256 failed");
    std::string result;
    for (unsigned int i = 0; i < size; ++i) {
        result += "0123456789abcdef"[digest[i] >> 4];
        result += "0123456789abcdef"[digest[i] & 15];
    }
    return result;
}
inline nlohmann::json verifiedGeometryManifest(const std::string& text, const std::string& hash)
{
    if (geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size())) != hash)
        throw std::runtime_error("geometry: corrupt revision manifest");
    return nlohmann::json::parse(text);
}
inline void verifyGeometryChunk(std::span<const uint8_t> bytes, const std::string& storedHash,
                                const std::string& expectedHash, int64_t expectedSize)
{
    if (static_cast<int64_t>(bytes.size()) != expectedSize || storedHash != expectedHash || geometryHash(bytes) != expectedHash)
        throw std::runtime_error("geometry: corrupt payload chunk");
}
inline int64_t geometryByteLength(const std::string& dtype, const std::vector<int64_t>& shape)
{
    int64_t bytes;
    if (dtype == "|u1") bytes = 1;
    else if (dtype == "<i4" || dtype == "<f4") bytes = 4;
    else if (dtype == "<i8" || dtype == "<u8" || dtype == "<f8") bytes = 8;
    else throw std::invalid_argument("geometry: unsupported dtype " + dtype);
    if (shape.empty() || shape.size() > 8) throw std::invalid_argument("geometry: invalid array rank");
    for (auto dimension : shape) {
        if (dimension < 0 || (dimension && bytes > std::numeric_limits<int64_t>::max() / dimension))
            throw std::invalid_argument("geometry: invalid/overflowing shape");
        bytes *= dimension;
    }
    return bytes;
}
// Storage layout is independent of logical geometry format_version. Missing
// storage metadata denotes the original contiguous 1 MiB chunk representation.
struct GeometryLayout {
    int64_t length = 0, rows = 0, cols = 0, tile = 0;
    int64_t tileColumns() const { return cols / tile + (cols % tile != 0); }
    int64_t chunkCount() const {
        return tile ? (rows / tile + (rows % tile != 0)) * tileColumns()
                    : length / geometryChunkSize + (length % geometryChunkSize != 0);
    }
    std::array<int64_t, 4> rectangle(int64_t index) const {
        const auto y = index / tileColumns() * tile, x = index % tileColumns() * tile;
        return {y, x, std::min(tile, rows - y), std::min(tile, cols - x)};
    }
    int64_t chunkBytes(int64_t index) const {
        if (index < 0 || index >= chunkCount()) throw std::runtime_error("geometry: invalid chunk index");
        if (!tile) return std::min<int64_t>(geometryChunkSize, length - index * geometryChunkSize);
        const auto rect = rectangle(index);
        return rect[2] * rect[3] * 12;
    }
};
// Copy one physical chunk into its full logical row-major array destination.
inline void copyGeometryChunk(const GeometryLayout& layout, int64_t index,
                              std::span<const uint8_t> bytes, uint8_t* destination)
{
    if (!layout.tile) {
        std::memcpy(destination + index * geometryChunkSize, bytes.data(), bytes.size());
    } else {
        const auto [y, x, h, w] = layout.rectangle(index);
        for (int64_t row = 0; row < h; ++row)
            std::memcpy(destination + ((y + row) * layout.cols + x) * 12,
                        bytes.data() + row * w * 12, w * 12);
    }
}
inline void validatePatchTileSize(int64_t size) {
    if (size != 0 && size != 32 && size != 64 && size != 128)
        throw std::invalid_argument("patch tile size must be 0 (linear), 32, 64, or 128");
}
inline GeometryLayout geometryLayout(const nlohmann::json& array)
{
    GeometryLayout result;
    const auto shape = array.at("shape").get<std::vector<int64_t>>();
    result.length = geometryByteLength(array.at("dtype"), shape);
    if (result.length != array.at("byte_length").get<int64_t>())
        throw std::runtime_error("geometry: inconsistent array byte length");
    if (array.contains("storage")) {
        const auto& storage = array.at("storage");
        if (storage.at("version") != 1 || storage.at("layout") != "tiles" ||
            array.at("dtype") != "<f4" || shape.size() != 3 || shape[2] != 3)
            throw std::runtime_error("geometry: unsupported storage layout");
        result.tile = storage.at("tile_size").get<int64_t>();
        validatePatchTileSize(result.tile);
        if (!result.tile) throw std::runtime_error("geometry: invalid tiled layout");
        result.rows = shape[0]; result.cols = shape[1];
    }
    if (!array.at("chunks").is_array() || array.at("chunks").size() != static_cast<size_t>(result.chunkCount()))
        throw std::runtime_error("geometry: inconsistent chunk count");
    return result;
}

// Enumerate physical bytes in manifest order, reusing one tile-sized buffer.
// Both hashing and publication use this helper so layout cannot diverge.
template<class Visit>
void forEachGeometryChunk(const GeometryArray& array, int64_t tile, Visit&& visit)
{
    if (!tile) {
        for (size_t offset = 0, index = 0; offset < array.bytes.size(); offset += geometryChunkSize, ++index)
            visit(index, std::span(array.bytes).subspan(offset, std::min(geometryChunkSize, array.bytes.size() - offset)));
        return;
    }
    const auto rows = array.shape.at(0), cols = array.shape.at(1);
    std::vector<uint8_t> buffer;
    int64_t index = 0;
    for (int64_t y = 0; rows && cols && y < rows; y += tile)
        for (int64_t x = 0; x < cols; x += tile) {
            const auto h = std::min(tile, rows - y), w = std::min(tile, cols - x);
            buffer.resize(h * w * 12);
            for (int64_t r = 0; r < h; ++r)
                std::memcpy(buffer.data() + r * w * 12, array.bytes.data() + ((y + r) * cols + x) * 12, w * 12);
            visit(index++, std::span<const uint8_t>(buffer));
        }
}

inline nlohmann::json geometryManifest(const GeometryData& data, int64_t patchTileSize = GeometryStorageOptions{}.patchTileSize)
{
    using Json = nlohmann::json;
    validatePatchTileSize(patchTileSize);
    if (data.kind != "patch" && data.kind != "fiber" && data.kind != "point_collection" && data.kind != "track")
        throw std::invalid_argument("geometry: unsupported object kind " + data.kind);
    if (data.formatVersion != 1 || data.coordinateSystem != "source_xyz")
        throw std::invalid_argument("geometry: unsupported format version or coordinate system");
    auto attributes = Json::parse(data.attributes);
    if (!attributes.is_object()) throw std::invalid_argument("geometry: attributes must be a JSON object");
    Json arrays = Json::object();
    for (const auto& array : data.arrays) {
        if (array.name.empty() || arrays.contains(array.name)) throw std::invalid_argument("geometry: empty/duplicate array name");
        if (geometryByteLength(array.dtype, array.shape) != static_cast<int64_t>(array.bytes.size()))
            throw std::invalid_argument("geometry: array shape does not match byte length");
        arrays[array.name] = {{"dtype", array.dtype}, {"shape", array.shape}, {"byte_length", array.bytes.size()}, {"chunks", Json::array()}};
    }
    auto require = [&](const char* name, const char* dtype, size_t rank, bool xyz) {
        if (!arrays.contains(name)) throw std::invalid_argument(std::string("geometry: missing array ") + name);
        const auto& a = arrays.at(name);
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        if (a.at("dtype") != dtype || shape.size() != rank || (xyz && shape.back() != 3))
            throw std::invalid_argument(std::string("geometry: invalid layout for ") + name);
    };
    if (data.kind == "patch") {
        require("points", "<f4", 3, true);
        if (!attributes.contains("scale") || !attributes.at("scale").is_array() || attributes.at("scale").size() != 2)
            throw std::invalid_argument("geometry: patch attributes require two-axis scale");
        for (const auto& value : attributes.at("scale"))
            if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() <= 0)
                throw std::invalid_argument("geometry: patch scale must be finite and positive");
    }
    if (data.kind == "fiber") {
        require("control_points", "<f8", 2, true);
        require("line_points", "<f8", 2, true);
    }
    if (data.kind == "track") require("points", "<i4", 2, true);
    if (data.kind == "point_collection") {
        require("points", "<f8", 2, true);
        require("point_ids", "<i8", 1, false);
        if (arrays["points"]["shape"][0] != arrays["point_ids"]["shape"][0])
            throw std::invalid_argument("geometry: point IDs and coordinates differ in length");
    }
    for (const auto& array : data.arrays) {
        auto& descriptor = arrays[array.name];
        const auto tile = data.kind == "patch" && array.name == "points" ? patchTileSize : 0;
        if (tile) descriptor["storage"] = {{"version", 1}, {"layout", "tiles"}, {"tile_size", tile}};
        forEachGeometryChunk(array, tile, [&](int64_t, std::span<const uint8_t> bytes) {
            descriptor["chunks"].push_back(geometryHash(bytes));
        });
    }
    return {{"kind", data.kind}, {"format_version", data.formatVersion}, {"coordinate_system", data.coordinateSystem},
        {"attributes", attributes}, {"arrays", arrays}};
}
} // namespace vc::objects::detail
