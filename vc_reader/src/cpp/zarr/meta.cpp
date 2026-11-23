// Zarr array metadata normalized for source chunks and decoded disk caches.
#include "vc_reader/reader.hpp"
#include "format.hpp"
#include "../element_value.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/array.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

namespace vcr::zarr {
namespace {
// Compressor ids the reader decodes, by numcodecs id (null: uncompressed).
const std::map<std::string, std::string>& supportedCompressors()
{
    static const std::map<std::string, std::string> ids{
        {"blosc", "blosc"}, {"zstd", "zstd"}, {"zlib", "zlib"}, {"gzip", "gzip"}, {"lz4", "lz4"}, {"lzw", "lzw"},
        {"imagecodecs_lzw", "lzw"}, {"vc-delta3d", "vc-delta3d"}, {"volcomp", "volcomp"}};
    return ids;
}

}  // namespace

std::size_t ArrayMeta::chunkBytes() const
{
    std::size_t n = itemsize();
    for (const auto c : chunks)
        n *= static_cast<std::size_t>(c);
    return n;
}

std::string ArrayMeta::codec() const
{
    std::string codec;
    if (!compressor.is_null()) {
        const auto id = compressor.value("id", std::string());
        const auto& ids = supportedCompressors();
        const auto it = ids.find(id);
        if (it == ids.end()) {
            std::string supported;
            for (const auto& [name, _] : ids)
                supported += ", " + name;
            throw std::invalid_argument("vc_reader cannot read arrays compressed with '" + id +
                                        "' (supported: uncompressed" + supported + ")");
        }
        codec = it->second;
    }
    bool packbits = false;
    if (!filters.is_null() && !filters.empty()) {
        packbits = filters.size() == 1 && filters[0].value("id", std::string()) == "packbits" && type.kind == 'b';
        if (!packbits)
            throw std::invalid_argument("vc_reader cannot read arrays with these numcodecs filters: " + filters.dump());
    }
    if (order != "C")
        throw std::invalid_argument("vc_reader cannot read arrays stored in Fortran order");
    return packbits ? "packbits+" + codec : codec;
}

Json ArrayMeta::decodedZarray() const
{
    Json out = raw;
    out["compressor"] = nullptr;
    out["filters"] = nullptr;
    out["order"] = "C";
    out.erase("_vc_reader_v3");
    return out;
}

Json ArrayMeta::delta3dZarray() const
{
    Json out = decodedZarray();
    out["compressor"] = {{"id", "vc-delta3d"}, {"quant", 1}};
    auto dt = out.value("dtype", std::string());
    if (!dt.empty() && dt[0] != '|')
        out["dtype"] = std::string(std::endian::native == std::endian::little ? "<" : ">") + dt.substr(1);
    return out;
}

ArrayMeta parseArrayMeta(const Json& input)
{
    Json d = input;
    if (d.value("zarr_format", 0) == 3)
        d = Json::parse(normalizeZarrMetadata(d.dump()));
    if (d.value("zarr_format", 0) != 2)
        throw std::invalid_argument("only Zarr v2/v3 arrays are supported, got zarr_format=" +
                                    (d.contains("zarr_format") ? d["zarr_format"].dump() : std::string("null")));
    ArrayMeta m;
    m.raw = d;
    m.shape = d.at("shape").get<Index>();
    m.chunks = d.at("chunks").get<Index>();
    detail::validateShape(m.shape, m.chunks);
    if (!d.at("dtype").is_string())
        throw std::invalid_argument("structured Zarr dtypes are not supported");
    const auto t = parseDtype(d.at("dtype").get<std::string>());
    m.dtype = t.str();
    m.type = t.element();
    m.fill = encodeElement(d.value("fill_value", Json()), m.type, ValueEncoding::ZarrFill);
    const auto order = d.value("order", Json());
    m.order = order.is_string() ? order.get<std::string>() : "C";
    const auto separator = d.value("dimension_separator", Json());
    m.separator = separator.is_string() ? separator.get<std::string>() : ".";
    if (m.separator != "." && m.separator != "/")
        throw std::invalid_argument("invalid Zarr dimension_separator '" + m.separator + "'");
    m.compressor = d.value("compressor", Json());
    m.filters = d.value("filters", Json());
    if (d.contains("_vc_reader_v3")) {
        const auto& v = d.at("_vc_reader_v3");
        ArrayMeta::V3 v3;
        v3.prefix = v.at("prefix").get<std::string>();
        v3.shards = v.at("shard_chunks").get<Index>();
        v3.indexEnd = v.at("index_end").get<bool>();
        v3.indexCrc = v.at("index_crc").get<bool>();
        v3.metadata = v.at("metadata");
        m.v3 = std::move(v3);
    }
    return m;
}

}  // namespace vcr::zarr
