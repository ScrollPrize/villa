// Zarr array metadata normalized for source chunks and decoded disk caches.
#include "vc_reader/reader.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/native/rounding.hpp"

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

struct Typestr {
    char order = '|', kind = 'u';
    std::size_t itemsize = 1;
};

Typestr parseTypestr(const std::string& text)
{
    Typestr t;
    std::size_t i = 0;
    if (!text.empty() && std::string_view("<>|=").find(text[0]) != std::string_view::npos)
        t.order = text[i++];
    if (i >= text.size() || std::string_view("buif").find(text[i]) == std::string_view::npos)
        throw std::invalid_argument("unsupported Zarr dtype '" + text + "'");
    t.kind = text[i++];
    try {
        t.itemsize = std::stoul(text.substr(i));
    } catch (const std::exception&) {
        throw std::invalid_argument("unsupported Zarr dtype '" + text + "'");
    }
    const bool integer = t.kind == 'u' || t.kind == 'i';
    if ((t.kind == 'b' && t.itemsize != 1) || (t.kind == 'f' && t.itemsize != 2 && t.itemsize != 4 && t.itemsize != 8) ||
        (integer && t.itemsize != 1 && t.itemsize != 2 && t.itemsize != 4 && t.itemsize != 8))
        throw std::invalid_argument("unsupported Zarr dtype '" + text + "'");
    if (t.order == '=' || (t.order == '|' && t.itemsize > 1))
        t.order = std::endian::native == std::endian::little ? '<' : '>';
    return t;
}

template <class T>
void put(std::vector<std::uint8_t>& out, T value)
{
    out.resize(sizeof(T));
    std::memcpy(out.data(), &value, sizeof(T));
}

// A value as IEEE binary16, rounded to nearest, ties to even.
std::uint16_t float16Bits(double v)
{
    const std::uint16_t sign = std::signbit(v) ? 0x8000 : 0;
    if (std::isnan(v))
        return sign | 0x7e00;
    const double magnitude = std::fabs(v);
    if (magnitude >= 65520.0)
        return sign | 0x7c00;
    if (magnitude < std::ldexp(1.0, -14))
        return sign | std::uint16_t(round_half_even(std::ldexp(magnitude, 24)));
    int exponent = 0;
    const double mantissa = std::frexp(magnitude, &exponent);  // in [0.5, 1)
    // A mantissa rounding up to 1024 carries into the exponent, as the format intends.
    return sign | std::uint16_t(((exponent + 14) << 10) + std::uint16_t(round_half_even((mantissa * 2.0 - 1.0) * 1024.0)));
}

// One fill element in native byte order.
std::vector<std::uint8_t> fillBytes(const Json& fill, const Typestr& t)
{
    std::vector<std::uint8_t> out(t.itemsize, 0);
    if (fill.is_null())
        return out;
    if (t.kind == 'f') {
        double v = 0;
        if (fill.is_string()) {
            const auto s = fill.get<std::string>();
            if (s == "NaN")
                v = std::numeric_limits<double>::quiet_NaN();
            else if (s == "Infinity")
                v = std::numeric_limits<double>::infinity();
            else if (s == "-Infinity")
                v = -std::numeric_limits<double>::infinity();
            else if (s.starts_with("0x")) {  // v3: the raw bits
                const auto bits = std::stoull(s.substr(2), nullptr, 16);
                if (t.itemsize == 2)
                    put(out, static_cast<std::uint16_t>(bits));
                else if (t.itemsize == 4)
                    put(out, static_cast<std::uint32_t>(bits));
                else
                    put(out, static_cast<std::uint64_t>(bits));
                return out;
            } else
                throw std::invalid_argument("invalid float fill_value '" + s + "'");
        } else if (fill.is_number() || fill.is_boolean()) {
            v = fill.is_boolean() ? double(fill.get<bool>()) : fill.get<double>();
        } else {
            throw std::invalid_argument("invalid fill_value " + fill.dump());
        }
        if (t.itemsize == 2)
            put(out, float16Bits(v));
        else if (t.itemsize == 4)
            put(out, static_cast<float>(v));
        else
            put(out, v);
        return out;
    }
    if (!fill.is_number_integer() && !fill.is_boolean() && !(fill.is_number_float() && std::floor(fill.get<double>()) == fill.get<double>()))
        throw std::invalid_argument("invalid integer fill_value " + fill.dump());
    const auto value = fill.is_boolean() ? std::int64_t(fill.get<bool>())
                       : fill.is_number_unsigned() ? static_cast<std::int64_t>(fill.get<std::uint64_t>())
                                                   : fill.is_number_integer() ? fill.get<std::int64_t>()
                                                                              : static_cast<std::int64_t>(fill.get<double>());
    switch (t.itemsize) {
    case 1: put(out, static_cast<std::uint8_t>(value)); break;
    case 2: put(out, static_cast<std::uint16_t>(value)); break;
    case 4: put(out, static_cast<std::uint32_t>(value)); break;
    default: put(out, static_cast<std::uint64_t>(value)); break;
    }
    return out;
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
    if (m.shape.empty() || m.shape.size() != m.chunks.size())
        throw std::invalid_argument("Zarr shape and chunks must have the same nonzero rank");
    for (std::size_t i = 0; i < m.shape.size(); ++i)
        if (m.shape[i] < 0 || m.chunks[i] <= 0)
            throw std::invalid_argument("invalid Zarr shape or chunks");
    if (!d.at("dtype").is_string())
        throw std::invalid_argument("structured Zarr dtypes are not supported");
    const auto t = parseTypestr(d.at("dtype").get<std::string>());
    const bool swap = (t.order == '>') != (std::endian::native == std::endian::big) && t.itemsize > 1;
    m.dtype = std::string(1, t.itemsize == 1 ? '|' : t.order) + t.kind + std::to_string(t.itemsize);
    m.type = ElementType{t.kind, t.itemsize, swap};
    m.fill = fillBytes(d.value("fill_value", Json()), t);
    if (swap)
        std::reverse(m.fill.begin(), m.fill.end());
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
