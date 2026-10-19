#pragma once
#include "vc/core/objects/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <set>
#include <span>
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
inline nlohmann::json geometryManifest(const GeometryData& data)
{
    using Json = nlohmann::json;
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
        Json chunks = Json::array();
        for (size_t offset = 0; offset < array.bytes.size(); offset += geometryChunkSize)
            chunks.push_back(geometryHash(std::span(array.bytes).subspan(offset, std::min(geometryChunkSize, array.bytes.size() - offset))));
        arrays[array.name] = {{"dtype", array.dtype}, {"shape", array.shape}, {"byte_length", array.bytes.size()}, {"chunks", chunks}};
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
    return {{"kind", data.kind}, {"format_version", data.formatVersion}, {"coordinate_system", data.coordinateSystem},
        {"attributes", attributes}, {"arrays", arrays}};
}
} // namespace vc::objects::detail
