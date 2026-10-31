#include "vc_reader/reader.hpp"
#include "sheet_normal/sample.hpp"
#include "vc_reader/catalog.hpp"
#include "vc_reader/zarr.hpp"
#include "zarr/format.hpp"
#include "vc_reader/native/array.hpp"
#include "vc_reader/native/sheet_normal/normal.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace vcr {
using Json = nlohmann::json;

std::string normalizeZarrMetadata(const std::string& text)
{
    const auto d = Json::parse(text);
    if (d.value("zarr_format", 0) != 3) return d.dump();
    if (d.value("node_type", "") != "array" ||
        !d.value("storage_transformers", Json::array()).empty()) {
        throw std::invalid_argument("unsupported Zarr v3 array");
    }
    auto dtype = zarr::detail::v3Types.at(d.at("data_type").get<std::string>());
    const auto& grid = d.at("chunk_grid");
    if (grid.at("name") != "regular")
        throw std::invalid_argument("Zarr v3 requires regular chunks");
    const auto outer = grid.at("configuration").at("chunk_shape").get<Index>();
    auto chunks = outer;
    const auto& keys = d.at("chunk_key_encoding");
    const auto keyName = keys.at("name").get<std::string>();
    if (keyName != "default" && keyName != "v2")
        throw std::invalid_argument("unsupported Zarr v3 keys");
    const auto sep = keys.value("configuration", Json::object()).value(
        "separator", keyName == "default" ? "/" : ".");
    if (sep != "/" && sep != ".")
        throw std::invalid_argument("invalid Zarr key separator");

    auto codecs = d.at("codecs");
    Index ratios;
    bool indexEnd = true, indexCrc = false;
    if (codecs.size() == 1 && codecs[0].at("name") == "sharding_indexed") {
        const auto& shard = codecs[0].at("configuration");
        chunks = shard.at("chunk_shape").get<Index>();
        if (chunks.size() != outer.size())
            throw std::invalid_argument("invalid shard shape");
        uint64_t count = 1;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (chunks[i] <= 0 || outer[i] <= 0 || outer[i] % chunks[i])
                throw std::invalid_argument("invalid shard dimensions");
            const auto ratio = outer[i] / chunks[i];
            if (uint64_t(ratio) > ((64ull << 20) - 4) / 16 / count)
                throw std::invalid_argument("shard index exceeds 64 MiB");
            count *= ratio;
            ratios.push_back(ratio);
        }
        const auto& indexes = shard.at("index_codecs");
        if (indexes.empty() || indexes.size() > 2 || indexes[0].at("name") != "bytes" ||
            indexes[0].value("configuration", Json::object()).value("endian", "") != "little" ||
            (indexes.size() == 2 && indexes[1].at("name") != "crc32c")) {
            throw std::invalid_argument("unsupported shard index codecs");
        }
        indexCrc = indexes.size() == 2;
        const auto loc = shard.value("index_location", "end");
        if (loc != "start" && loc != "end")
            throw std::invalid_argument("invalid shard index location");
        indexEnd = loc == "end";
        // Copy before replacing codecs: shard refers to an element of codecs.
        auto innerCodecs = shard.at("codecs");
        codecs = std::move(innerCodecs);
    }
    const auto shape = d.at("shape").get<Index>();
    if (shape.empty() || shape.size() != chunks.size())
        throw std::invalid_argument("invalid Zarr shape");
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || chunks[i] <= 0)
            throw std::invalid_argument("invalid Zarr dimensions");
    }
    Json filters = nullptr;
    if (!codecs.empty() && codecs[0].at("name") == "numcodecs.packbits") {
        filters = Json::array({{{"id", "packbits"}}});
        codecs.erase(codecs.begin());
    }
    // v3 compressor names -> v2 compressor ids.
    const auto& compressors = zarr::detail::v3Compressors;
    if (codecs.empty() || codecs.size() > 2 || codecs[0].at("name") != "bytes" ||
        (codecs.size() == 2 && !compressors.contains(codecs[1].at("name").get<std::string>()))) {
        throw std::invalid_argument("unsupported Zarr v3 codec pipeline");
    }
    if (dtype.back() != '1') {
        const auto endian = codecs[0].value("configuration", Json::object()).value("endian", "");
        if (endian != "little" && endian != "big")
            throw std::invalid_argument("missing bytes endian");
        dtype[0] = endian == "little" ? '<' : '>';
    }
    Json compressor = nullptr;
    if (codecs.size() == 2) {
        compressor = codecs[1].value("configuration", Json::object());
        compressor["id"] = compressors.at(codecs[1].at("name").get<std::string>());
        if (compressor["id"] == "blosc" && compressor.contains("shuffle") && compressor["shuffle"].is_string()) {
            const std::map<std::string, int> shuffles{{"noshuffle", 0}, {"shuffle", 1}, {"bitshuffle", 2}};
            compressor["shuffle"] = shuffles.at(compressor["shuffle"].get<std::string>());
        }
    }
    return Json{
        {"zarr_format", 2}, {"shape", shape}, {"chunks", chunks}, {"dtype", dtype},
        {"fill_value", d.at("fill_value")}, {"compressor", compressor}, {"filters", filters},
        {"order", "C"}, {"dimension_separator", sep},
        {"_vc_reader_v3", {{"metadata", d}, {"prefix", keyName == "default" ? "c" + sep : ""},
            {"shard_chunks", ratios}, {"index_end", indexEnd}, {"index_crc", indexCrc}}}}.dump();
}

template<class T>
void normals(std::span<const T> patches, std::span<const std::array<float, 4>> centers,
             std::span<std::array<float, 4>> out)
{
    if (centers.size() > SIZE_MAX / (109 * 109 * 109) ||
        patches.size() != centers.size() * 109 * 109 * 109 || out.size() != centers.size()) {
        throw std::invalid_argument("sheet normal buffer sizes differ");
    }
    if (!centers.empty())
        sheet_normal::cpu(patches.data(), centers.front().data(), out.front().data(), centers.size());
}

void sheetNormalsCpu(std::span<const uint8_t> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out)
{
    normals(patches, centers, out);
}

void sheetNormalsCpu(std::span<const float> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out)
{
    normals(patches, centers, out);
}

void sheetNormals(zarr::OpenedArray& array, const double* xyz, std::size_t count, float* out)
{
    const auto& meta = array.meta;
    if (meta.shape.size() != 3) throw std::invalid_argument("sheet normals read a three-dimensional array");
    const bool bytes = meta.type.kind == 'u' && meta.itemsize() == 1;
    if (!bytes && !(meta.type.kind == 'f' && meta.itemsize() == 4)) throw std::invalid_argument("sheet normals read uint8 or float32 CT");
    const std::int64_t shape[3]{meta.shape[0], meta.shape[1], meta.shape[2]};
    if (!count) return;
    if (bytes) sheet_normal::sample<std::uint8_t>(*array.array, shape, xyz, count, out);
    else sheet_normal::sample<float>(*array.array, shape, xyz, count, out);
}
} // namespace vcr
