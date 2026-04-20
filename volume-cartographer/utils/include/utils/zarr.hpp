#pragma once
#include "utils/Json.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <memory>
#include <unordered_map>
#include <variant>

#if __has_include("compression.hpp")
#  include "compression.hpp"
#  define UTILS_HAS_COMPRESSION 1
#else
#  define UTILS_HAS_COMPRESSION 0
#endif

#if __has_include("http_fetch.hpp")
#  include "http_fetch.hpp"
#  ifndef UTILS_HAS_CURL
#    define UTILS_HAS_CURL 0
#  endif
#else
#  define UTILS_HAS_CURL 0
#endif

namespace utils {

// ---------------------------------------------------------------------------
// JSON type aliases (nlohmann/json)
// ---------------------------------------------------------------------------

using JsonValue  = utils::Json;
using JsonObject = utils::Json;
using JsonArray  = utils::Json;

/// Parse a JSON string. Replaces the former custom json_parse().
[[nodiscard]] JsonValue json_parse(std::string_view text);

/// Serialize a JSON value. indent=0 means compact, indent>0 means pretty.
[[nodiscard]] std::string json_serialize(const JsonValue& v, int indent = -1);

/// Build a JSON object from an initializer list.
[[nodiscard]] JsonValue json_object(
    std::initializer_list<std::pair<const std::string, JsonValue>> pairs);

/// Build a JSON array from an initializer list.
[[nodiscard]] JsonValue json_array(std::initializer_list<JsonValue> values);

/// Pointer-based find, matching the old custom JSON API.
/// Returns nullptr if key not found or value is not an object.
[[nodiscard]] const JsonValue* json_find(const JsonValue& obj, const std::string& key);

// ---------------------------------------------------------------------------
// ZarrVersion
// ---------------------------------------------------------------------------

enum class ZarrVersion : std::uint8_t { v2 = 2, v3 = 3 };

// ---------------------------------------------------------------------------
// ZarrDtype
// ---------------------------------------------------------------------------

enum class ZarrDtype : std::uint8_t {
    bool_,
    uint8, uint16, uint32, uint64,
    int8, int16, int32, int64,
    float16, float32, float64,
    complex64, complex128
};

[[nodiscard]] constexpr std::size_t dtype_size(ZarrDtype dt) noexcept {
    switch (dt) {
        case ZarrDtype::bool_:      return 1;
        case ZarrDtype::uint8:      return 1;
        case ZarrDtype::uint16:     return 2;
        case ZarrDtype::uint32:     return 4;
        case ZarrDtype::uint64:     return 8;
        case ZarrDtype::int8:       return 1;
        case ZarrDtype::int16:      return 2;
        case ZarrDtype::int32:      return 4;
        case ZarrDtype::int64:      return 8;
        case ZarrDtype::float16:    return 2;
        case ZarrDtype::float32:    return 4;
        case ZarrDtype::float64:    return 8;
        case ZarrDtype::complex64:  return 8;
        case ZarrDtype::complex128: return 16;
    }
    return 0;
}

/// Returns the zarr v2 dtype string WITHOUT the byte-order prefix (e.g. "u2").
[[nodiscard]] constexpr std::string_view dtype_string_v2(ZarrDtype dt) noexcept {
    switch (dt) {
        case ZarrDtype::bool_:      return "b1";
        case ZarrDtype::uint8:      return "u1";
        case ZarrDtype::uint16:     return "u2";
        case ZarrDtype::uint32:     return "u4";
        case ZarrDtype::uint64:     return "u8";
        case ZarrDtype::int8:       return "i1";
        case ZarrDtype::int16:      return "i2";
        case ZarrDtype::int32:      return "i4";
        case ZarrDtype::int64:      return "i8";
        case ZarrDtype::float16:    return "f2";
        case ZarrDtype::float32:    return "f4";
        case ZarrDtype::float64:    return "f8";
        case ZarrDtype::complex64:  return "c8";
        case ZarrDtype::complex128: return "c16";
    }
    return "";
}

/// Backward-compatible alias.
[[nodiscard]] constexpr std::string_view dtype_string(ZarrDtype dt) noexcept {
    return dtype_string_v2(dt);
}

/// Returns the zarr v3 data_type string (e.g. "uint16", "float32").
[[nodiscard]] constexpr std::string_view dtype_string_v3(ZarrDtype dt) noexcept {
    switch (dt) {
        case ZarrDtype::bool_:      return "bool";
        case ZarrDtype::uint8:      return "uint8";
        case ZarrDtype::uint16:     return "uint16";
        case ZarrDtype::uint32:     return "uint32";
        case ZarrDtype::uint64:     return "uint64";
        case ZarrDtype::int8:       return "int8";
        case ZarrDtype::int16:      return "int16";
        case ZarrDtype::int32:      return "int32";
        case ZarrDtype::int64:      return "int64";
        case ZarrDtype::float16:    return "float16";
        case ZarrDtype::float32:    return "float32";
        case ZarrDtype::float64:    return "float64";
        case ZarrDtype::complex64:  return "complex64";
        case ZarrDtype::complex128: return "complex128";
    }
    return "";
}

/// Parses a v2 dtype string such as "<u2", ">f4", "|u1". The byte-order
/// prefix is optional; if present it is ignored.
[[nodiscard]] constexpr std::optional<ZarrDtype> parse_dtype(std::string_view s) noexcept {
    // Strip optional byte-order prefix.
    if (!s.empty() && (s.front() == '<' || s.front() == '>' || s.front() == '|'))
        s.remove_prefix(1);
    if (s == "b1")  return ZarrDtype::bool_;
    if (s == "u1")  return ZarrDtype::uint8;
    if (s == "u2")  return ZarrDtype::uint16;
    if (s == "u4")  return ZarrDtype::uint32;
    if (s == "u8")  return ZarrDtype::uint64;
    if (s == "i1")  return ZarrDtype::int8;
    if (s == "i2")  return ZarrDtype::int16;
    if (s == "i4")  return ZarrDtype::int32;
    if (s == "i8")  return ZarrDtype::int64;
    if (s == "f2")  return ZarrDtype::float16;
    if (s == "f4")  return ZarrDtype::float32;
    if (s == "f8")  return ZarrDtype::float64;
    if (s == "c8")  return ZarrDtype::complex64;
    if (s == "c16") return ZarrDtype::complex128;
    return std::nullopt;
}

/// Parses a v3 data_type string (e.g. "uint16", "float64").
[[nodiscard]] constexpr std::optional<ZarrDtype> parse_dtype_v3(std::string_view s) noexcept {
    if (s == "bool")       return ZarrDtype::bool_;
    if (s == "uint8")      return ZarrDtype::uint8;
    if (s == "uint16")     return ZarrDtype::uint16;
    if (s == "uint32")     return ZarrDtype::uint32;
    if (s == "uint64")     return ZarrDtype::uint64;
    if (s == "int8")       return ZarrDtype::int8;
    if (s == "int16")      return ZarrDtype::int16;
    if (s == "int32")      return ZarrDtype::int32;
    if (s == "int64")      return ZarrDtype::int64;
    if (s == "float16")    return ZarrDtype::float16;
    if (s == "float32")    return ZarrDtype::float32;
    if (s == "float64")    return ZarrDtype::float64;
    if (s == "complex64")  return ZarrDtype::complex64;
    if (s == "complex128") return ZarrDtype::complex128;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// ZarrCodecConfig -- v3 codec pipeline entry
// ---------------------------------------------------------------------------

struct ZarrCodecConfig {
    std::string name;           // e.g. "blosc", "gzip", "zstd", "bytes", "transpose", "sharding_indexed"
    std::shared_ptr<JsonValue> configuration;  // codec-specific config as JSON (may be null)
};

// ---------------------------------------------------------------------------
// V2 Filter
// ---------------------------------------------------------------------------

enum class ZarrFilterId : std::uint8_t {
    delta,
    fixedscaleoffset,
    quantize
};

struct ZarrFilter {
    ZarrFilterId id = ZarrFilterId::delta;
    ZarrDtype dtype = ZarrDtype::int32;          // output dtype for delta
    ZarrDtype astype = ZarrDtype::int32;         // storage dtype for delta
    double offset = 0.0;                         // for fixedscaleoffset
    double scale = 1.0;                          // for fixedscaleoffset
    int digits = 10;                             // for quantize

    /// Apply filter forward (encode).
    [[nodiscard]] std::vector<std::byte> encode(std::span<const std::byte> input) const {
        switch (id) {
            case ZarrFilterId::delta:           return encode_delta(input);
            case ZarrFilterId::fixedscaleoffset: return encode_fixedscaleoffset(input);
            case ZarrFilterId::quantize:        return encode_quantize(input);
        }
        return {input.begin(), input.end()};
    }

    /// Apply filter inverse (decode).
    [[nodiscard]] std::vector<std::byte> decode(std::span<const std::byte> input) const {
        switch (id) {
            case ZarrFilterId::delta:           return decode_delta(input);
            case ZarrFilterId::fixedscaleoffset: return decode_fixedscaleoffset(input);
            case ZarrFilterId::quantize:        return {input.begin(), input.end()}; // quantize is lossy, decode is identity
        }
        return {input.begin(), input.end()};
    }

private:
    // -- Delta filter: stores differences between consecutive elements --------

    [[nodiscard]] std::vector<std::byte> encode_delta(std::span<const std::byte> input) const {
        const auto elem_sz = dtype_size(dtype);
        // Only uint8 (1 byte) and uint16 (2 bytes) are supported.
        if ((elem_sz != 1 && elem_sz != 2) || input.size() % elem_sz != 0)
            return {input.begin(), input.end()};

        const auto n = input.size() / elem_sz;
        std::vector<std::byte> out(input.size());
        // First element is stored as-is.
        std::memcpy(out.data(), input.data(), elem_sz);
        // Subsequent elements store the delta on whole elements.
        if (elem_sz == 1) {
            auto* src = reinterpret_cast<const std::uint8_t*>(input.data());
            auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
            for (std::size_t i = 1; i < n; ++i)
                dst[i] = static_cast<std::uint8_t>(src[i] - src[i - 1]);
        } else {
            auto* src = reinterpret_cast<const std::uint16_t*>(input.data());
            auto* dst = reinterpret_cast<std::uint16_t*>(out.data());
            for (std::size_t i = 1; i < n; ++i)
                dst[i] = static_cast<std::uint16_t>(src[i] - src[i - 1]);
        }
        return out;
    }

    [[nodiscard]] std::vector<std::byte> decode_delta(std::span<const std::byte> input) const {
        const auto elem_sz = dtype_size(dtype);
        // Only uint8 (1 byte) and uint16 (2 bytes) are supported.
        if ((elem_sz != 1 && elem_sz != 2) || input.size() % elem_sz != 0)
            return {input.begin(), input.end()};

        const auto n = input.size() / elem_sz;
        std::vector<std::byte> out(input.size());
        std::memcpy(out.data(), input.data(), elem_sz);
        if (elem_sz == 1) {
            auto* src = reinterpret_cast<const std::uint8_t*>(input.data());
            auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
            for (std::size_t i = 1; i < n; ++i)
                dst[i] = static_cast<std::uint8_t>(src[i] + dst[i - 1]);
        } else {
            auto* src = reinterpret_cast<const std::uint16_t*>(input.data());
            auto* dst = reinterpret_cast<std::uint16_t*>(out.data());
            for (std::size_t i = 1; i < n; ++i)
                dst[i] = static_cast<std::uint16_t>(src[i] + dst[i - 1]);
        }
        return out;
    }

    // -- FixedScaleOffset filter: linear transform ----------------------------

    [[nodiscard]] std::vector<std::byte> encode_fixedscaleoffset(std::span<const std::byte> input) const {
        // Simplified: applies (x - offset) * scale element-wise on float64 data.
        const auto elem_sz = dtype_size(dtype);
        if (elem_sz == 0 || input.size() % elem_sz != 0)
            return {input.begin(), input.end()};
        if (dtype != ZarrDtype::float64 && dtype != ZarrDtype::float32)
            return {input.begin(), input.end()};

        std::vector<std::byte> out(input.size());
        const auto n = input.size() / elem_sz;
        if (dtype == ZarrDtype::float64) {
            for (std::size_t i = 0; i < n; ++i) {
                double val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
                val = (val - offset) * scale;
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                float val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
                val = static_cast<float>((val - offset) * scale);
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
            }
        }
        return out;
    }

    [[nodiscard]] std::vector<std::byte> decode_fixedscaleoffset(std::span<const std::byte> input) const {
        const auto elem_sz = dtype_size(dtype);
        if (elem_sz == 0 || input.size() % elem_sz != 0)
            return {input.begin(), input.end()};
        if (dtype != ZarrDtype::float64 && dtype != ZarrDtype::float32)
            return {input.begin(), input.end()};

        std::vector<std::byte> out(input.size());
        const auto n = input.size() / elem_sz;
        if (dtype == ZarrDtype::float64) {
            for (std::size_t i = 0; i < n; ++i) {
                double val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
                val = val / scale + offset;
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                float val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
                val = static_cast<float>(val / scale + offset);
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
            }
        }
        return out;
    }

    // -- Quantize filter: reduce precision of floating-point data -------------

    [[nodiscard]] std::vector<std::byte> encode_quantize(std::span<const std::byte> input) const {
        const auto elem_sz = dtype_size(dtype);
        if (elem_sz == 0 || input.size() % elem_sz != 0)
            return {input.begin(), input.end()};

        std::vector<std::byte> out(input.size());
        const auto n = input.size() / elem_sz;
        const double factor = std::pow(10.0, digits);
        if (dtype == ZarrDtype::float64) {
            for (std::size_t i = 0; i < n; ++i) {
                double val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
                val = std::round(val * factor) / factor;
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
            }
        } else if (dtype == ZarrDtype::float32) {
            for (std::size_t i = 0; i < n; ++i) {
                float val;
                std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
                val = static_cast<float>(std::round(val * factor) / factor);
                std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
            }
        } else {
            return {input.begin(), input.end()};
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// ShardConfig -- zarr v3 sharding extension
// ---------------------------------------------------------------------------

struct ShardConfig {
    std::vector<std::size_t> sub_chunks;   // inner chunk shape
    std::vector<ZarrCodecConfig> index_codecs;  // codecs for the shard index
    std::vector<ZarrCodecConfig> sub_codecs;    // codecs for inner chunks
};

// ---------------------------------------------------------------------------
// ZarrMetadata
// ---------------------------------------------------------------------------

struct ZarrMetadata {
    ZarrVersion version = ZarrVersion::v2;

    // Common fields (v2 and v3).
    std::vector<std::size_t> shape;
    std::vector<std::size_t> chunks;
    ZarrDtype dtype = ZarrDtype::uint16;
    std::optional<double> fill_value;

    // V2-specific fields.
    char byte_order = '<';
    std::string compressor_id;           // "blosc", "zlib", "zstd", or "" for raw
    int compression_level = 5;
    std::string dimension_separator = ".";
    std::vector<ZarrFilter> filters;     // v2 filters applied before compression

    // V3-specific fields.
    std::vector<ZarrCodecConfig> codecs; // v3 codec pipeline
    std::string chunk_key_encoding = "default";  // "default" (separator "/") or "v2" (separator ".")
    std::optional<ShardConfig> shard_config;
    std::string node_type = "array";     // v3 node_type

    [[nodiscard]] std::size_t ndim() const noexcept { return shape.size(); }

    [[nodiscard]] std::size_t num_chunks_along(std::size_t dim) const noexcept {
        if (dim >= shape.size() || dim >= chunks.size() || chunks[dim] == 0) return 0;
        return (shape[dim] + chunks[dim] - 1) / chunks[dim];
    }

    [[nodiscard]] std::size_t chunk_byte_size() const noexcept {
        std::size_t n = dtype_size(dtype);
        for (auto c : chunks) n *= c;
        return n;
    }

    /// For sharded arrays, compute inner chunk byte size.
    [[nodiscard]] std::size_t sub_chunk_byte_size() const noexcept {
        if (!shard_config) return chunk_byte_size();
        std::size_t n = dtype_size(dtype);
        for (auto c : shard_config->sub_chunks) n *= c;
        return n;
    }

    /// Number of inner chunks per shard along a given dimension.
    [[nodiscard]] std::size_t sub_chunks_per_shard(std::size_t dim) const noexcept {
        if (!shard_config || dim >= chunks.size() ||
            dim >= shard_config->sub_chunks.size() ||
            shard_config->sub_chunks[dim] == 0)
            return 1;
        return chunks[dim] / shard_config->sub_chunks[dim];
    }

    /// Total number of inner chunks in one shard.
    [[nodiscard]] std::size_t total_sub_chunks_per_shard() const noexcept {
        if (!shard_config) return 1;
        std::size_t n = 1;
        for (std::size_t d = 0; d < chunks.size() && d < shard_config->sub_chunks.size(); ++d)
            n *= sub_chunks_per_shard(d);
        return n;
    }

    /// The v3 chunk key separator.
    [[nodiscard]] std::string v3_separator() const noexcept {
        return (chunk_key_encoding == "v2") ? "." : "/";
    }
};

/// Structural check: zarr v3, sharded, with 128^3 inner chunks. A true return
/// here means the layout matches our canonical disk-cache format, but does NOT
/// guarantee the inner bytes are VC3D/H.265 — only a runtime magic check on a
/// fetched inner chunk can confirm that.
[[nodiscard]] inline bool is_canonical_vc3d(const ZarrMetadata& m) noexcept {
    if (m.version != ZarrVersion::v3) return false;
    if (!m.shard_config) return false;
    const auto& sc = *m.shard_config;
    if (sc.sub_chunks.size() < 3) return false;
    if (sc.sub_chunks[0] != 128 || sc.sub_chunks[1] != 128 || sc.sub_chunks[2] != 128)
        return false;
    if (m.chunks.size() < 3) return false;
    for (int d = 0; d < 3; ++d) if (m.chunks[d] % 128 != 0) return false;
    if (m.dtype != ZarrDtype::uint8) return false;
    return true;
}

/// Structural check: zarr v3, sharded, with 256^3 inner chunks (c3d codec
/// atom). As with is_canonical_vc3d, this only checks shape; the inner
/// bytes still need a C3DC magic-check before decode.
[[nodiscard]] inline bool is_canonical_c3d(const ZarrMetadata& m) noexcept {
    if (m.version != ZarrVersion::v3) return false;
    if (!m.shard_config) return false;
    const auto& sc = *m.shard_config;
    if (sc.sub_chunks.size() < 3) return false;
    if (sc.sub_chunks[0] != 256 || sc.sub_chunks[1] != 256 || sc.sub_chunks[2] != 256)
        return false;
    if (m.chunks.size() < 3) return false;
    for (int d = 0; d < 3; ++d) if (m.chunks[d] % 256 != 0) return false;
    if (m.dtype != ZarrDtype::uint8) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Store abstraction
// ---------------------------------------------------------------------------

class Store {
public:
    virtual ~Store() = default;

    [[nodiscard]] virtual bool exists(const std::string& key) const = 0;
    [[nodiscard]] virtual std::vector<std::byte> get(const std::string& key) const = 0;
    [[nodiscard]] virtual std::optional<std::vector<std::byte>> get_if_exists(const std::string& key) const = 0;
    virtual void set(const std::string& key, std::span<const std::byte> value) = 0;
    virtual void erase(const std::string& key) = 0;

    /// Convenience: get as string.
    [[nodiscard]] std::string get_string(const std::string& key) const {
        auto data = get(key);
        return {reinterpret_cast<const char*>(data.data()), data.size()};
    }

    /// Convenience: set from string.
    void set_string(const std::string& key, std::string_view value) {
        set(key, {reinterpret_cast<const std::byte*>(value.data()), value.size()});
    }

    /// Get a byte range from a key [offset, offset+length).
    [[nodiscard]] virtual std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const {
        auto data = get_if_exists(key);
        if (!data) return std::nullopt;
        if (offset >= data->size()) return std::vector<std::byte>{};
        auto end = std::min(offset + length, data->size());
        return std::vector<std::byte>(data->begin() + static_cast<std::ptrdiff_t>(offset),
                                       data->begin() + static_cast<std::ptrdiff_t>(end));
    }
};

// ---------------------------------------------------------------------------
// FileSystemStore
// ---------------------------------------------------------------------------

class FileSystemStore final : public Store {
public:
    explicit FileSystemStore(std::filesystem::path root) : root_(std::move(root)) {}

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

    [[nodiscard]] bool exists(const std::string& key) const override {
        return std::filesystem::exists(safe_path(key));
    }

    [[nodiscard]] std::vector<std::byte> get(const std::string& key) const override {
        auto p = safe_path(key);
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("zarr store: cannot open: " + p.string());
        auto sz = f.tellg();
        f.seekg(0);
        std::vector<std::byte> buf(static_cast<std::size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        return buf;
    }

    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_if_exists(const std::string& key) const override {
        auto p = safe_path(key);
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) return std::nullopt;
        auto sz = f.tellg();
        f.seekg(0);
        std::vector<std::byte> buf(static_cast<std::size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        return buf;
    }

    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override {
        auto p = safe_path(key);
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) return std::nullopt;
        auto file_sz = static_cast<std::size_t>(f.tellg());
        if (offset >= file_sz) return std::vector<std::byte>{};
        auto end = std::min(offset + length, file_sz);
        auto actual_len = end - offset;
        f.seekg(static_cast<std::streamoff>(offset));
        std::vector<std::byte> buf(actual_len);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(actual_len));
        return buf;
    }

    void set(const std::string& key, std::span<const std::byte> value) override {
        auto p = safe_path(key);
        std::filesystem::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("zarr store: cannot write: " + p.string());
        f.write(reinterpret_cast<const char*>(value.data()),
                static_cast<std::streamsize>(value.size()));
    }

    void erase(const std::string& key) override {
        std::filesystem::remove(safe_path(key));
    }

private:
    [[nodiscard]] std::filesystem::path safe_path(const std::string& key) const {
        auto p = (root_ / key).lexically_normal();
        // Ensure the resolved path is within root_ (prevent ../ traversal).
        auto root_norm = root_.lexically_normal();
        auto [root_end, p_begin] = std::mismatch(
            root_norm.begin(), root_norm.end(), p.begin(), p.end());
        if (root_end != root_norm.end())
            throw std::runtime_error("zarr store: path traversal rejected: " + key);
        return p;
    }

    std::filesystem::path root_;
};

// ---------------------------------------------------------------------------
// HttpStore -- read-only store over HTTP/S3 (requires curl)
// ---------------------------------------------------------------------------
#if UTILS_HAS_CURL

class HttpStore final : public Store {
public:
    /// Construct from a base URL with default HttpClient settings.
    explicit HttpStore(std::string base_url)
        : base_url_(strip_trailing_slash(std::move(base_url)))
        , client_(std::make_shared<HttpClient>()) {}

    /// Construct from a base URL with a custom HttpClient config.
    HttpStore(std::string base_url, HttpClient::Config config)
        : base_url_(strip_trailing_slash(std::move(base_url)))
        , client_(std::make_shared<HttpClient>(std::move(config))) {}

    /// Construct from a base URL with AWS SigV4 authentication.
    HttpStore(std::string base_url, AwsAuth auth)
        : base_url_(strip_trailing_slash(std::move(base_url)))
    {
        HttpClient::Config cfg;
        cfg.aws_auth = std::move(auth);
        cfg.transfer_timeout = std::chrono::seconds{60};
        client_ = std::make_shared<HttpClient>(std::move(cfg));
    }

    [[nodiscard]] bool exists(const std::string& key) const override {
        return client_->head(make_url(key)).ok();
    }

    [[nodiscard]] std::vector<std::byte> get(const std::string& key) const override {
        auto data = get_if_exists(key);
        if (!data)
            throw std::runtime_error("HttpStore: key not found: " + key);
        return std::move(*data);
    }

    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_if_exists(const std::string& key) const override {
        auto resp = client_->get(make_url(key));
        if (!resp.ok()) return std::nullopt;
        return std::move(resp.body);
    }

    void set(const std::string& /*key*/, std::span<const std::byte> /*value*/) override {
        throw std::runtime_error("HttpStore is read-only");
    }

    void erase(const std::string& /*key*/) override {
        throw std::runtime_error("HttpStore is read-only");
    }

    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override {
        auto resp = client_->get_range(make_url(key), offset, length);
        if (!resp.ok()) return std::nullopt;
        return std::move(resp.body);
    }

private:
    [[nodiscard]] std::string make_url(const std::string& key) const {
        return base_url_ + "/" + key;
    }

    static std::string strip_trailing_slash(std::string s) {
        while (!s.empty() && s.back() == '/')
            s.pop_back();
        return s;
    }

    std::string base_url_;
    std::shared_ptr<HttpClient> client_;
};

#endif // UTILS_HAS_CURL

// ---------------------------------------------------------------------------
// detail -- I/O and metadata helpers
// ---------------------------------------------------------------------------

namespace detail {

// ----- File I/O helpers (kept for backward compatibility) -----

inline std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zarr: cannot open file: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::string buf(static_cast<std::size_t>(sz), '\0');
    f.read(buf.data(), sz);
    return buf;
}

inline std::vector<std::byte> read_file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zarr: cannot open file: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

inline void write_file(const std::filesystem::path& p, std::string_view data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("zarr: cannot write file: " + p.string());
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
}

inline void write_file_bytes(const std::filesystem::path& p, std::span<const std::byte> data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("zarr: cannot write file: " + p.string());
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
}

// ----- Endian helpers -----

inline bool is_little_endian() noexcept {
    const std::uint32_t one = 1;
    return *reinterpret_cast<const std::uint8_t*>(&one) == 1;
}

inline void byteswap_inplace(std::span<std::byte> data, std::size_t elem_size) {
    if (elem_size <= 1) return;
    for (std::size_t i = 0; i + elem_size <= data.size(); i += elem_size) {
        std::reverse(data.begin() + static_cast<std::ptrdiff_t>(i),
                     data.begin() + static_cast<std::ptrdiff_t>(i + elem_size));
    }
}

// ----- Little-endian integer encode/decode for shard index -----

inline void write_le64(std::byte* dst, std::uint64_t val) {
    for (int i = 0; i < 8; ++i)
        dst[i] = static_cast<std::byte>((val >> (8 * i)) & 0xFF);
}

inline std::uint64_t read_le64(const std::byte* src) {
    std::uint64_t val = 0;
    for (int i = 0; i < 8; ++i)
        val |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(src[i])) << (8 * i);
    return val;
}

// ----- Shard index -----

/// Shard index entry: offset and nbytes for one inner chunk.
/// If offset == 0xFFFF...F and nbytes == 0xFFFF...F, the chunk is missing.
struct ShardIndexEntry {
    std::uint64_t offset = ~std::uint64_t(0);
    std::uint64_t nbytes  = ~std::uint64_t(0);

    [[nodiscard]] bool is_missing() const noexcept {
        return offset == ~std::uint64_t(0) && nbytes == ~std::uint64_t(0);
    }
};

/// A shard index is an array of (offset, nbytes) pairs, one per inner chunk,
/// stored in C-order with respect to the inner chunk grid.
struct ShardIndex {
    std::vector<ShardIndexEntry> entries;

    /// Serialize to binary (little-endian).
    [[nodiscard]] std::vector<std::byte> serialize() const {
        std::vector<std::byte> buf(entries.size() * 16);
        for (std::size_t i = 0; i < entries.size(); ++i) {
            write_le64(buf.data() + i * 16,     entries[i].offset);
            write_le64(buf.data() + i * 16 + 8, entries[i].nbytes);
        }
        return buf;
    }

    /// Deserialize from binary.
    static ShardIndex deserialize(std::span<const std::byte> data, std::size_t num_chunks) {
        ShardIndex idx;
        idx.entries.resize(num_chunks);
        if (data.size() < num_chunks * 16)
            throw std::runtime_error("zarr: shard index too small");
        for (std::size_t i = 0; i < num_chunks; ++i) {
            idx.entries[i].offset = read_le64(data.data() + i * 16);
            idx.entries[i].nbytes  = read_le64(data.data() + i * 16 + 8);
        }
        return idx;
    }
};

// ----- .zarray (v2) parse / serialize -----

ZarrMetadata parse_zarray(std::string_view json_str);

inline std::string serialize_zarray(const ZarrMetadata& meta) {
    std::string s = "{\n";
    s += "  \"zarr_format\": 2,\n";

    // shape
    s += "  \"shape\": [";
    for (std::size_t i = 0; i < meta.shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(meta.shape[i]);
    }
    s += "],\n";

    // chunks
    s += "  \"chunks\": [";
    for (std::size_t i = 0; i < meta.chunks.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(meta.chunks[i]);
    }
    s += "],\n";

    // dtype
    s += "  \"dtype\": \"";
    s += meta.byte_order;
    s += dtype_string_v2(meta.dtype);
    s += "\",\n";

    // compressor
    if (meta.compressor_id.empty()) {
        s += "  \"compressor\": null,\n";
    } else {
        s += "  \"compressor\": {\"id\": \"" + meta.compressor_id + "\"";
        s += ", \"clevel\": " + std::to_string(meta.compression_level);
        s += "},\n";
    }

    // fill_value
    if (meta.fill_value.has_value()) {
        double fv = *meta.fill_value;
        if (fv == static_cast<double>(static_cast<std::int64_t>(fv)))
            s += "  \"fill_value\": " + std::to_string(static_cast<std::int64_t>(fv)) + ",\n";
        else
            s += "  \"fill_value\": " + std::to_string(fv) + ",\n";
    } else {
        s += "  \"fill_value\": null,\n";
    }

    s += "  \"order\": \"C\",\n";

    // filters
    if (meta.filters.empty()) {
        s += "  \"filters\": null,\n";
    } else {
        s += "  \"filters\": [";
        for (std::size_t i = 0; i < meta.filters.size(); ++i) {
            if (i) s += ", ";
            const auto& f = meta.filters[i];
            switch (f.id) {
                case ZarrFilterId::delta:
                    s += "{\"id\": \"delta\"";
                    s += ", \"dtype\": \"" + std::string(dtype_string_v2(f.dtype)) + "\"";
                    s += ", \"astype\": \"" + std::string(dtype_string_v2(f.astype)) + "\"";
                    s += "}";
                    break;
                case ZarrFilterId::fixedscaleoffset:
                    s += "{\"id\": \"fixedscaleoffset\"";
                    s += ", \"offset\": " + std::to_string(f.offset);
                    s += ", \"scale\": " + std::to_string(f.scale);
                    s += "}";
                    break;
                case ZarrFilterId::quantize:
                    s += "{\"id\": \"quantize\"";
                    s += ", \"digits\": " + std::to_string(f.digits);
                    s += "}";
                    break;
            }
        }
        s += "],\n";
    }

    s += "  \"dimension_separator\": \"" + meta.dimension_separator + "\"\n";
    s += "}\n";
    return s;
}

// ----- zarr.json (v3) parse / serialize -----

ZarrCodecConfig parse_codec_config(const JsonValue& jv);

ZarrMetadata parse_zarr_json(std::string_view json_str);

JsonValue codec_config_to_json(const ZarrCodecConfig& cc);

std::string serialize_zarr_json(const ZarrMetadata& meta);

// ----- Consolidated metadata (.zmetadata, v2) -----

struct ConsolidatedMetadata {
    std::unordered_map<std::string, ZarrMetadata> arrays;   // path -> metadata
    std::unordered_map<std::string, JsonValue> attrs;       // path -> attributes

    /// Parse a .zmetadata JSON file.
    static ConsolidatedMetadata parse(std::string_view json_str);
};

// ----- Auto-detect version -----

inline ZarrVersion detect_version(const std::filesystem::path& path) {
    if (std::filesystem::exists(path / "zarr.json"))
        return ZarrVersion::v3;
    if (std::filesystem::exists(path / ".zarray"))
        return ZarrVersion::v2;
    throw std::runtime_error("zarr: cannot detect version at: " + path.string() +
                             " (no .zarray or zarr.json found)");
}

} // namespace detail

// ---------------------------------------------------------------------------
// ZarrArray
// ---------------------------------------------------------------------------

class ZarrArray final {
public:
    /// Codec callback struct: compress and decompress functions.
    struct Codec {
        std::function<std::vector<std::byte>(std::span<const std::byte>)> compress;
        std::function<std::vector<std::byte>(std::span<const std::byte>, std::size_t)> decompress;
    };

    /// Named codec registry: maps codec name to Codec.
    using CodecRegistry = std::unordered_map<std::string, Codec>;

    // -- Open / Create -------------------------------------------------------

    /// Open an existing zarr array at `path`. Auto-detects v2 vs v3.
    static ZarrArray open(const std::filesystem::path& path, Codec codec = {}) {
        auto version = detail::detect_version(path);
        ZarrMetadata meta;
        if (version == ZarrVersion::v2) {
            auto json = detail::read_file(path / ".zarray");
            meta = detail::parse_zarray(json);
        } else {
            auto json = detail::read_file(path / "zarr.json");
            meta = detail::parse_zarr_json(json);
        }
        return ZarrArray(path, std::move(meta), std::move(codec));
    }

    /// Open with a codec registry (for v3 codec pipelines).
    static ZarrArray open(const std::filesystem::path& path, CodecRegistry registry) {
        auto version = detail::detect_version(path);
        ZarrMetadata meta;
        if (version == ZarrVersion::v2) {
            auto json = detail::read_file(path / ".zarray");
            meta = detail::parse_zarray(json);
        } else {
            auto json = detail::read_file(path / "zarr.json");
            meta = detail::parse_zarr_json(json);
        }

        // Try to find the appropriate codec from the registry.
        Codec codec;
        if (version == ZarrVersion::v2 && !meta.compressor_id.empty()) {
            auto it = registry.find(meta.compressor_id);
            if (it != registry.end()) codec = it->second;
        } else if (version == ZarrVersion::v3) {
            // Look for bytes-to-bytes codecs in the pipeline.
            for (const auto& cc : meta.codecs) {
                if (cc.name != "bytes" && cc.name != "transpose" && cc.name != "sharding_indexed") {
                    auto it = registry.find(cc.name);
                    if (it != registry.end()) { codec = it->second; break; }
                }
            }
        }

        return ZarrArray(path, std::move(meta), std::move(codec), std::move(registry));
    }

    /// Create a new zarr v2 array, writing .zarray metadata to disk.
    static ZarrArray create(const std::filesystem::path& path,
                            ZarrMetadata meta, Codec codec = {}) {
        std::filesystem::create_directories(path);
        if (meta.version == ZarrVersion::v3) {
            auto json = detail::serialize_zarr_json(meta);
            detail::write_file(path / "zarr.json", json);
        } else {
            auto json = detail::serialize_zarray(meta);
            detail::write_file(path / ".zarray", json);
        }
        return ZarrArray(path, std::move(meta), std::move(codec));
    }

    /// Create a new zarr v3 array with a codec registry.
    static ZarrArray create(const std::filesystem::path& path,
                            ZarrMetadata meta, CodecRegistry registry) {
        Codec codec;
        if (meta.version == ZarrVersion::v2 && !meta.compressor_id.empty()) {
            auto it = registry.find(meta.compressor_id);
            if (it != registry.end()) codec = it->second;
        } else if (meta.version == ZarrVersion::v3) {
            for (const auto& cc : meta.codecs) {
                if (cc.name != "bytes" && cc.name != "transpose" && cc.name != "sharding_indexed") {
                    auto it = registry.find(cc.name);
                    if (it != registry.end()) { codec = it->second; break; }
                }
            }
        }

        std::filesystem::create_directories(path);
        if (meta.version == ZarrVersion::v3) {
            auto json = detail::serialize_zarr_json(meta);
            detail::write_file(path / "zarr.json", json);
        } else {
            auto json = detail::serialize_zarray(meta);
            detail::write_file(path / ".zarray", json);
        }
        return ZarrArray(path, std::move(meta), std::move(codec), std::move(registry));
    }

    /// Open from a Store.
    static ZarrArray open(std::shared_ptr<Store> store, const std::string& array_key,
                          Codec codec = {}) {
        // Build store key without leading "/" when array_key is empty.
        auto store_key = [&](const std::string& name) -> std::string {
            return array_key.empty() ? name : array_key + "/" + name;
        };
        ZarrMetadata meta;
        if (store->exists(store_key("zarr.json"))) {
            auto data = store->get_string(store_key("zarr.json"));
            meta = detail::parse_zarr_json(data);
        } else if (store->exists(store_key(".zarray"))) {
            auto data = store->get_string(store_key(".zarray"));
            meta = detail::parse_zarray(data);
        } else {
            throw std::runtime_error("zarr: no metadata found at store key: " + array_key);
        }
        return ZarrArray(std::move(store), array_key, std::move(meta), std::move(codec));
    }

    /// Open with pre-parsed metadata (no file I/O). Used by open_from_consolidated.
    static ZarrArray open_with_metadata(const std::filesystem::path& path,
                                         ZarrMetadata meta, Codec codec = {}) {
        return ZarrArray(path, std::move(meta), std::move(codec));
    }

    // -- Accessors -----------------------------------------------------------

    [[nodiscard]] const ZarrMetadata& metadata() const noexcept { return meta_; }
    [[nodiscard]] ZarrVersion version() const noexcept { return meta_.version; }
    [[nodiscard]] bool is_sharded() const noexcept { return meta_.shard_config.has_value(); }

    // -- Chunk I/O (non-sharded) ---------------------------------------------

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk(std::span<const std::size_t> chunk_indices) const {
        if (is_sharded())
            return read_inner_chunk_from_shard(chunk_indices);

        auto raw = read_chunk_raw(chunk_indices);
        if (!raw) return std::nullopt;

        auto data = std::move(*raw);

        // Decompress.
        if (codec_.decompress && needs_decompression()) {
            data = codec_.decompress(data, meta_.chunk_byte_size());
        }

        // Apply v2 filter decoding (in reverse order).
        if (meta_.version == ZarrVersion::v2) {
            for (auto it = meta_.filters.rbegin(); it != meta_.filters.rend(); ++it)
                data = it->decode(data);
        }

        // Byte-swap if the stored byte order differs from native.
        if (needs_byteswap()) {
            detail::byteswap_inplace(data, dtype_size(meta_.dtype));
        }

        return data;
    }

    void write_chunk(std::span<const std::size_t> chunk_indices,
                     std::span<const std::byte> data) {
        // Apply v2 filters (forward order).
        std::vector<std::byte> buf;
        std::span<const std::byte> write_data = data;
        if (meta_.version == ZarrVersion::v2 && !meta_.filters.empty()) {
            buf.assign(data.begin(), data.end());
            for (const auto& f : meta_.filters) {
                buf = f.encode(buf);
            }
            write_data = buf;
        }

        // Compress.
        std::vector<std::byte> compressed;
        if (codec_.compress && needs_compression()) {
            compressed = codec_.compress(write_data);
            write_data = compressed;
        }

        write_chunk_raw(chunk_indices, write_data);
    }

    [[nodiscard]] bool chunk_exists(std::span<const std::size_t> chunk_indices) const {
        auto key = chunk_key(chunk_indices);
        if (store_)
            return store_->exists(key);
        return std::filesystem::exists(root_ / key);
    }

    [[nodiscard]] std::string
    chunk_key(std::span<const std::size_t> chunk_indices) const {
        if (meta_.version == ZarrVersion::v3)
            return chunk_key_v3(chunk_indices);
        return chunk_key_v2(chunk_indices);
    }

    /// Backward-compatible chunk_path (filesystem only).
    [[nodiscard]] std::filesystem::path
    chunk_path(std::span<const std::size_t> chunk_indices) const {
        return root_ / chunk_key(chunk_indices);
    }

    // -- Shard I/O -----------------------------------------------------------

    /// Read an inner chunk from a sharded array.
    /// `chunk_indices` are in terms of the inner chunk grid (not shard grid).
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_inner_chunk(std::span<const std::size_t> shard_indices,
                     std::span<const std::size_t> inner_indices) const {
        if (!is_sharded())
            throw std::runtime_error("zarr: not a sharded array");

        auto shard_key = chunk_key(shard_indices);
        auto shard_data = read_raw(shard_key);
        if (!shard_data) return std::nullopt;

        return extract_inner_chunk(*shard_data, inner_indices);
    }

    // 4k-align chunk payloads inside shards so decoders can mmap the shard
    // and hand direct pointers to the codec without buffering.  Index entries
    // use the aligned offset; padding bytes between chunks are ignored by
    // the reader (spec allows arbitrary gaps).
    static constexpr std::uint64_t kShardChunkAlign = 4096;

    /// Write a complete shard given a set of inner chunk data.
    /// `inner_chunks` is indexed by linear inner chunk index (C-order).
    /// Missing inner chunks should be std::nullopt.
    void write_shard(std::span<const std::size_t> shard_indices,
                     std::span<const std::optional<std::vector<std::byte>>> inner_chunks) {
        if (!is_sharded())
            throw std::runtime_error("zarr: not a sharded array");

        const auto& sc = *meta_.shard_config;
        const auto n_inner = meta_.total_sub_chunks_per_shard();

        if (inner_chunks.size() != n_inner)
            throw std::runtime_error("zarr: wrong number of inner chunks for shard");

        // Build shard data: [index][chunk0 padded to 4k][chunk1 padded to 4k]...
        std::vector<std::byte> shard_data;
        detail::ShardIndex index;
        index.entries.resize(n_inner);

        // Index always at start — reserve space for it.
        const std::size_t index_size = n_inner * 16;
        shard_data.resize(index_size);

        auto pad_to_align = [&]() {
            auto n = shard_data.size();
            auto aligned = (n + kShardChunkAlign - 1) & ~(kShardChunkAlign - 1);
            if (aligned > n) shard_data.resize(aligned, std::byte{0});
        };
        pad_to_align();   // ensures first chunk lands on 4k boundary

        for (std::size_t i = 0; i < n_inner; ++i) {
            if (!inner_chunks[i]) {
                index.entries[i].offset = ~std::uint64_t(0);
                index.entries[i].nbytes  = ~std::uint64_t(0);
                continue;
            }

            const auto& chunk_data = *inner_chunks[i];
            std::span<const std::byte> write_data = chunk_data;

            std::vector<std::byte> compressed;
            if (codec_.compress && needs_compression()) {
                compressed = codec_.compress(write_data);
                write_data = compressed;
            }

            index.entries[i].offset = shard_data.size();
            index.entries[i].nbytes  = write_data.size();
            shard_data.insert(shard_data.end(), write_data.begin(), write_data.end());
            pad_to_align();
        }

        // Write index at start.
        auto index_bytes = index.serialize();
        std::memcpy(shard_data.data(), index_bytes.data(), index_size);

        // Write shard file.
        write_chunk_raw(shard_indices, shard_data);
    }

    /// Append a single chunk to its shard file. Index at start (fixed 8KB header).
    /// Two tiny writes: (1) append chunk data at EOF, (2) update 16-byte index entry.
    /// Creates shard with empty index if it doesn't exist.
    void write_inner_chunk_to_shard(std::span<const std::size_t> chunk_indices,
                                    std::span<const std::byte> data) {
        if (!is_sharded())
            throw std::runtime_error("zarr: not a sharded array");

        const auto ndim = meta_.ndim();
        const auto n_inner = meta_.total_sub_chunks_per_shard();
        const std::size_t index_size = n_inner * 16;

        std::vector<std::size_t> shard_idx(ndim);
        std::vector<std::size_t> inner_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
            inner_idx[d] = chunk_indices[d] % ips;
        }

        std::size_t linear = 0;
        std::size_t stride = 1;
        for (std::size_t d = ndim; d-- > 0;) {
            linear += inner_idx[d] * stride;
            stride *= meta_.sub_chunks_per_shard(d);
        }

        auto key = chunk_key(shard_idx);
        auto p = root_ / key;
        std::filesystem::create_directories(p.parent_path());

        // Lock to prevent concurrent writes tearing this shard file.
        // Striped so writers to different shards run concurrently.
        //
        // Must cover the exists-check-and-create pair: without the lock
        // two writers racing on a fresh shard would both see !exists,
        // both truncate with an empty index, and the second writer would
        // overwrite the first's already-committed index entry.
        std::lock_guard lock(shard_mutex_for(p));

        if (!std::filesystem::exists(p)) {
            std::ofstream create(p, std::ios::binary);
            // Write empty index: all entries = (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
            std::vector<std::byte> empty_index(index_size);
            std::memset(empty_index.data(), 0xFF, index_size);
            create.write(reinterpret_cast<const char*>(empty_index.data()),
                         static_cast<std::streamsize>(index_size));
        }

        // Open for random read/write
        std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) return;

        // 1. Seek to EOF, round up to next 4k boundary, append chunk data.
        //    Padding bytes (if any) are left uninitialised — ext4 zero-fills
        //    them on sparse extension, and readers never look at them.
        f.seekp(0, std::ios::end);
        auto eof_offset = static_cast<std::uint64_t>(f.tellp());
        auto chunk_offset = (eof_offset + kShardChunkAlign - 1)
                          & ~(kShardChunkAlign - 1);
        if (chunk_offset != eof_offset) {
            f.seekp(static_cast<std::streamoff>(chunk_offset));
        }
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));

        // 2. Seek to index entry, write 16 bytes (offset + nbytes)
        auto nbytes = static_cast<std::uint64_t>(data.size());
        f.seekp(static_cast<std::streamoff>(linear * 16));
        f.write(reinterpret_cast<const char*>(&chunk_offset), 8);
        f.write(reinterpret_cast<const char*>(&nbytes), 8);
        f.flush();
    }

    /// Check if an inner chunk exists. Reads only the 16-byte index entry.
    [[nodiscard]] bool inner_chunk_exists(std::span<const std::size_t> chunk_indices) const {
        if (!is_sharded()) return false;
        const auto ndim = meta_.ndim();

        std::vector<std::size_t> shard_idx(ndim);
        std::vector<std::size_t> inner_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
            inner_idx[d] = chunk_indices[d] % ips;
        }

        std::size_t linear = 0;
        std::size_t stride = 1;
        for (std::size_t d = ndim; d-- > 0;) {
            linear += inner_idx[d] * stride;
            stride *= meta_.sub_chunks_per_shard(d);
        }

        auto p = root_ / chunk_key(shard_idx);
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;

        f.seekg(static_cast<std::streamoff>(linear * 16));
        std::uint64_t offset = 0, nbytes = 0;
        f.read(reinterpret_cast<char*>(&offset), 8);
        f.read(reinterpret_cast<char*>(&nbytes), 8);
        if (!f) return false;
        // Not present: (0xFF..FF, 0xFF..FF). Empty/zero: (0xFF..FE, 0).
        if (offset == ~std::uint64_t(0) && nbytes == ~std::uint64_t(0)) return false;
        if (offset == (~std::uint64_t(0) - 1) && nbytes == 0) return false;
        return true;
    }

    /// Check if an inner chunk is marked as known-empty (zero data).
    [[nodiscard]] bool inner_chunk_is_empty(std::span<const std::size_t> chunk_indices) const {
        if (!is_sharded()) return false;
        const auto ndim = meta_.ndim();

        std::vector<std::size_t> shard_idx(ndim);
        std::vector<std::size_t> inner_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
            inner_idx[d] = chunk_indices[d] % ips;
        }

        std::size_t linear = 0;
        std::size_t stride = 1;
        for (std::size_t d = ndim; d-- > 0;) {
            linear += inner_idx[d] * stride;
            stride *= meta_.sub_chunks_per_shard(d);
        }

        auto p = root_ / chunk_key(shard_idx);
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;

        f.seekg(static_cast<std::streamoff>(linear * 16));
        std::uint64_t offset = 0, nbytes = 0;
        f.read(reinterpret_cast<char*>(&offset), 8);
        f.read(reinterpret_cast<char*>(&nbytes), 8);
        if (!f) return false;
        return (offset == (~std::uint64_t(0) - 1) && nbytes == 0);
    }

    /// Mark an inner chunk as known-empty (zero data). Writes sentinel to index.
    void mark_inner_chunk_empty(std::span<const std::size_t> chunk_indices) {
        if (!is_sharded())
            throw std::runtime_error("zarr: not a sharded array");

        const auto ndim = meta_.ndim();
        const auto n_inner = meta_.total_sub_chunks_per_shard();
        const std::size_t index_size = n_inner * 16;

        std::vector<std::size_t> shard_idx(ndim);
        std::vector<std::size_t> inner_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
            inner_idx[d] = chunk_indices[d] % ips;
        }

        std::size_t linear = 0;
        std::size_t stride = 1;
        for (std::size_t d = ndim; d-- > 0;) {
            linear += inner_idx[d] * stride;
            stride *= meta_.sub_chunks_per_shard(d);
        }

        auto key = chunk_key(shard_idx);
        auto p = root_ / key;
        std::filesystem::create_directories(p.parent_path());

        if (!std::filesystem::exists(p)) {
            std::ofstream create(p, std::ios::binary);
            std::vector<std::byte> empty_index(index_size);
            std::memset(empty_index.data(), 0xFF, index_size);
            create.write(reinterpret_cast<const char*>(empty_index.data()),
                         static_cast<std::streamsize>(index_size));
        }

        std::lock_guard lock(shard_mutex_for(p));
        std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) return;

        // Write empty sentinel: (0xFF..FE, 0)
        std::uint64_t sentinel_offset = ~std::uint64_t(0) - 1;
        std::uint64_t sentinel_nbytes = 0;
        f.seekp(static_cast<std::streamoff>(linear * 16));
        f.write(reinterpret_cast<const char*>(&sentinel_offset), 8);
        f.write(reinterpret_cast<const char*>(&sentinel_nbytes), 8);
        f.flush();
    }

    /// Write a shard file whose index marks every inner chunk as empty.
    /// Used to record "this shard is known empty" without a remote round-trip.
    void write_empty_shard(std::span<const std::size_t> shard_indices) {
        if (!is_sharded())
            throw std::runtime_error("zarr: not a sharded array");
        const auto n_inner = meta_.total_sub_chunks_per_shard();

        auto key = chunk_key(std::vector<std::size_t>(shard_indices.begin(), shard_indices.end()));
        auto p = root_ / key;
        std::filesystem::create_directories(p.parent_path());

        std::vector<std::uint64_t> index(n_inner * 2);
        const std::uint64_t sentinel_offset = ~std::uint64_t(0) - 1;
        const std::uint64_t sentinel_nbytes = 0;
        for (std::size_t i = 0; i < n_inner; ++i) {
            index[i * 2]     = sentinel_offset;
            index[i * 2 + 1] = sentinel_nbytes;
        }

        std::lock_guard lock(shard_mutex_for(p));
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(reinterpret_cast<const char*>(index.data()),
                static_cast<std::streamsize>(index.size() * 8));
    }

    /// Root path of the zarr array on disk.
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return root_; }

    // -- Attributes ----------------------------------------------------------

    [[nodiscard]] std::string read_attrs() const;

    void write_attrs(std::string_view json);

private:
    // -- Construction --------------------------------------------------------

    ZarrArray(std::filesystem::path root, ZarrMetadata meta, Codec codec,
              CodecRegistry registry = {})
        : root_(std::move(root)), meta_(std::move(meta)),
          codec_(std::move(codec)), registry_(std::move(registry)) {}

    ZarrArray(std::shared_ptr<Store> store, std::string array_key,
              ZarrMetadata meta, Codec codec, CodecRegistry registry = {})
        : store_(std::move(store)), array_key_(std::move(array_key)),
          meta_(std::move(meta)), codec_(std::move(codec)),
          registry_(std::move(registry)) {}

    // -- Internal helpers ----------------------------------------------------

    [[nodiscard]] bool needs_compression() const noexcept {
        if (meta_.version == ZarrVersion::v2)
            return !meta_.compressor_id.empty();
        // v3: check for bytes-to-bytes codecs in pipeline.
        for (const auto& cc : meta_.codecs)
            if (cc.name != "bytes" && cc.name != "transpose" && cc.name != "sharding_indexed")
                return true;
        return false;
    }

    [[nodiscard]] bool needs_decompression() const noexcept {
        return needs_compression();
    }

    [[nodiscard]] bool needs_byteswap() const noexcept;

    // -- Chunk key generation ------------------------------------------------

    [[nodiscard]] std::string
    chunk_key_v2(std::span<const std::size_t> idx) const {
        std::string name;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (i) name += meta_.dimension_separator;
            name += std::to_string(idx[i]);
        }
        return name;
    }

    [[nodiscard]] std::string
    chunk_key_v3(std::span<const std::size_t> idx) const {
        std::string sep = meta_.v3_separator();
        std::string name = "c";
        for (std::size_t i = 0; i < idx.size(); ++i) {
            name += sep;
            name += std::to_string(idx[i]);
        }
        return name;
    }

    // -- Raw chunk I/O (no compression/filters) ------------------------------

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_raw(const std::string& key) const {
        if (store_) {
            auto full_key = array_key_.empty() ? key : array_key_ + "/" + key;
            return store_->get_if_exists(full_key);
        }
        auto p = root_ / key;
        if (!std::filesystem::exists(p)) return std::nullopt;
        return detail::read_file_bytes(p);
    }

public:
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk_raw(std::span<const std::size_t> idx) const {
        return read_raw(chunk_key(idx));
    }

    void write_chunk_raw(std::span<const std::size_t> idx,
                         std::span<const std::byte> data) {
        auto key = chunk_key(idx);
        if (store_) {
            auto full_key = array_key_.empty() ? key : array_key_ + "/" + key;
            store_->set(full_key, data);
            return;
        }
        auto p = root_ / key;
        std::filesystem::create_directories(p.parent_path());
        detail::write_file_bytes(p, data);
    }

    // -- Shard reading helpers -----------------------------------------------

    /// Read a single chunk from its shard. Reads only the 16-byte index entry
    /// + the chunk data. Does NOT read the whole shard.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_inner_chunk_from_shard(std::span<const std::size_t> chunk_indices) const {
        if (!is_sharded()) return std::nullopt;
        const auto ndim = meta_.ndim();

        std::vector<std::size_t> shard_idx(ndim);
        std::vector<std::size_t> inner_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
            inner_idx[d] = chunk_indices[d] % ips;
        }

        std::size_t linear = 0;
        std::size_t stride = 1;
        for (std::size_t d = ndim; d-- > 0;) {
            linear += inner_idx[d] * stride;
            stride *= meta_.sub_chunks_per_shard(d);
        }

        auto key = chunk_key(shard_idx);
        auto p = root_ / key;
        // Lock to prevent reading while another thread is writing the
        // same shard (striped — reads against other shards don't block).
        std::lock_guard lock(shard_mutex_for(p));
        std::ifstream f(p, std::ios::binary);
        if (!f) return std::nullopt;

        // Read 16-byte index entry at position linear*16
        f.seekg(static_cast<std::streamoff>(linear * 16));
        std::uint64_t offset = 0, nbytes = 0;
        f.read(reinterpret_cast<char*>(&offset), 8);
        f.read(reinterpret_cast<char*>(&nbytes), 8);
        if (!f) return std::nullopt;
        if (offset == ~std::uint64_t(0) && nbytes == ~std::uint64_t(0))
            return std::nullopt;

        // Read chunk data
        f.seekg(static_cast<std::streamoff>(offset));
        std::vector<std::byte> data(nbytes);
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
        if (!f) return std::nullopt;
        return data;
    }

    /// Read the whole shard file that contains the given inner-chunk indices.
    /// Returns the raw shard bytes (including the trailing index). Use with
    /// extract_inner_chunk() to pull out individual inner chunks — useful
    /// when a caller maintains an external shard-level cache and wants to
    /// serve many inner chunks from a single disk read.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_whole_shard(std::span<const std::size_t> chunk_indices) const {
        if (!is_sharded()) return std::nullopt;
        const auto ndim = meta_.ndim();
        std::vector<std::size_t> shard_idx(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            auto ips = meta_.sub_chunks_per_shard(d);
            shard_idx[d] = chunk_indices[d] / ips;
        }
        auto key = chunk_key(shard_idx);
        auto p = root_ / key;
        std::lock_guard lock(shard_mutex_for(p));
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) return std::nullopt;
        auto size = static_cast<std::streamsize>(f.tellg());
        if (size <= 0) return std::nullopt;
        std::vector<std::byte> data(static_cast<std::size_t>(size));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(data.data()), size);
        if (!f) return std::nullopt;
        return data;
    }

    /// Extract a single inner chunk from shard data.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    extract_inner_chunk(std::span<const std::byte> shard_data,
                        std::span<const std::size_t> inner_indices) const;

    // -- Members -------------------------------------------------------------

    std::filesystem::path root_;
    std::shared_ptr<Store> store_;
    std::string array_key_;
    ZarrMetadata meta_;
    Codec codec_;
    CodecRegistry registry_;

    // Striped mutexes for concurrent write safety. A single global mutex
    // serialized all writers across the entire array — with the parallel
    // pipeline writing to many shards at once this was the dominant
    // write-path bottleneck. 64 stripes gives us essentially free
    // concurrency across independent shards while still protecting a
    // single shard from being torn by two writers.
    static constexpr std::size_t kShardMutexStripes = 64;
    mutable std::shared_ptr<std::array<std::mutex, kShardMutexStripes>>
        shard_write_mutexes_ = std::make_shared<std::array<std::mutex, kShardMutexStripes>>();

    [[nodiscard]] std::mutex& shard_mutex_for(const std::filesystem::path& p) const {
        std::size_t h = std::hash<std::string>{}(p.native());
        return (*shard_write_mutexes_)[h % kShardMutexStripes];
    }
};

// ---------------------------------------------------------------------------
// open_remote -- open a ZarrArray from an HTTP/S3 URL (requires curl)
// ---------------------------------------------------------------------------
#if UTILS_HAS_CURL

/// Open a zarr array from a URL (HTTP or S3).
/// For S3 URLs (s3://bucket/key or s3+REGION://bucket/key), the URL is
/// auto-converted to HTTPS and AWS SigV4 auth is applied if provided.
/// If no auth is given for an S3 URL, credentials are read from the
/// environment (AWS_ACCESS_KEY_ID, etc.).
///
/// The url should point to the array root (the directory containing
/// .zarray or zarr.json). If array_key is empty, metadata is looked up
/// at the url root itself.
[[nodiscard]] inline ZarrArray open_remote(
    const std::string& url,
    std::optional<AwsAuth> auth = std::nullopt,
    ZarrArray::Codec codec = {},
    const std::string& array_key = {})
{
    std::string base_url = url;
    std::shared_ptr<Store> store;

    if (is_s3_url(url)) {
        auto parsed = parse_s3_url(url);
        if (!parsed)
            throw std::runtime_error("open_remote: invalid S3 URL: " + url);

        base_url = s3_to_https(*parsed);

        // Resolve auth: explicit > load (SSO/files/env)
        AwsAuth aws = auth.value_or(AwsAuth::load());

        // If the URL included a region, prefer that over env/explicit
        if (!parsed->region.empty())
            aws.region = parsed->region;

        store = std::make_shared<HttpStore>(base_url, std::move(aws));
    } else {
        if (auth) {
            store = std::make_shared<HttpStore>(base_url, std::move(*auth));
        } else {
            store = std::make_shared<HttpStore>(base_url);
        }
    }

    return ZarrArray::open(std::move(store), array_key, std::move(codec));
}

#endif // UTILS_HAS_CURL

// ---------------------------------------------------------------------------
// Consolidated metadata reader (v2)
// ---------------------------------------------------------------------------

/// Load consolidated metadata from .zmetadata file.
[[nodiscard]] detail::ConsolidatedMetadata
load_consolidated_metadata(const std::filesystem::path& root);

/// Open a zarr array using consolidated metadata (avoids per-array metadata reads).
/// Uses the pre-parsed metadata directly without writing files.
[[nodiscard]] inline ZarrArray open_from_consolidated(
    const std::filesystem::path& root,
    const std::string& array_path,
    const detail::ConsolidatedMetadata& cm,
    ZarrArray::Codec codec = {}) {
    auto it = cm.arrays.find(array_path);
    if (it == cm.arrays.end())
        throw std::runtime_error("zarr: array not found in consolidated metadata: " + array_path);
    return ZarrArray::open_with_metadata(root / array_path, it->second, std::move(codec));
}

// ---------------------------------------------------------------------------
// Pyramid helpers
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::size_t count_pyramid_levels(const std::filesystem::path& root) {
    std::size_t level = 0;
    while (std::filesystem::is_directory(root / std::to_string(level)))
        ++level;
    return level;
}

[[nodiscard]] inline std::vector<ZarrArray> open_pyramid(
    const std::filesystem::path& root, ZarrArray::Codec codec = {}) {
    std::vector<ZarrArray> levels;
    auto n = count_pyramid_levels(root);
    levels.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        levels.push_back(ZarrArray::open(root / std::to_string(i), codec));
    return levels;
}

// ===========================================================================
// OME-NGFF (OME-Zarr) metadata and reader/writer
// ===========================================================================

// ---------------------------------------------------------------------------
// OME-Zarr axis types
// ---------------------------------------------------------------------------

enum class AxisType : std::uint8_t {
    space,
    time,
    channel,
    custom
};

struct Axis {
    std::string name;
    AxisType type = AxisType::space;
    std::string unit;
};

// ---------------------------------------------------------------------------
// Coordinate transforms (OME-NGFF multiscale)
// ---------------------------------------------------------------------------

struct ScaleTransform {
    std::vector<double> scale;
};

struct TranslationTransform {
    std::vector<double> translation;
};

using CoordinateTransform = std::variant<ScaleTransform, TranslationTransform>;

// ---------------------------------------------------------------------------
// Multiscale metadata
// ---------------------------------------------------------------------------

struct MultiscaleDataset {
    std::string path;
    std::vector<CoordinateTransform> transforms;
};

struct MultiscaleMetadata {
    std::string version = "0.4";
    std::string name;
    std::string type;
    std::vector<Axis> axes;
    std::vector<MultiscaleDataset> datasets;

    [[nodiscard]] std::size_t num_levels() const noexcept {
        return datasets.size();
    }

    [[nodiscard]] std::size_t ndim() const noexcept {
        return axes.size();
    }

    [[nodiscard]] std::optional<std::size_t> axis_index(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (axes[i].name == name) return i;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::array<double, 3> voxel_size(std::size_t level) const {
        if (level >= datasets.size())
            throw std::runtime_error("zarr: OME level out of range");

        std::array<double, 3> vs{1.0, 1.0, 1.0};
        const auto& ds = datasets[level];

        const ScaleTransform* st = nullptr;
        for (const auto& t : ds.transforms) {
            if (auto* p = std::get_if<ScaleTransform>(&t)) {
                st = p;
                break;
            }
        }
        if (!st) return vs;

        auto xi = axis_index("x");
        auto yi = axis_index("y");
        auto zi = axis_index("z");
        if (xi && *xi < st->scale.size()) vs[0] = st->scale[*xi];
        if (yi && *yi < st->scale.size()) vs[1] = st->scale[*yi];
        if (zi && *zi < st->scale.size()) vs[2] = st->scale[*zi];
        return vs;
    }
};

// ---------------------------------------------------------------------------
// Label metadata (OME-NGFF labels)
// ---------------------------------------------------------------------------

struct LabelColor {
    std::uint32_t label_value;
    std::array<std::uint8_t, 4> rgba;
};

struct LabelMetadata {
    std::string version = "0.4";
    std::vector<LabelColor> colors;
    std::vector<std::string> properties;
};

// ---------------------------------------------------------------------------
// Plate/Well metadata (HCS - High Content Screening)
// ---------------------------------------------------------------------------

struct WellRef {
    std::string path;
    std::size_t row;
    std::size_t col;
};

struct PlateMetadata {
    std::string version = "0.4";
    std::string name;
    std::vector<std::string> columns;
    std::vector<std::string> rows;
    std::vector<WellRef> wells;
    std::size_t field_count = 1;
};

// ---------------------------------------------------------------------------
// OME metadata parsing helpers
// ---------------------------------------------------------------------------

namespace ome_detail {

[[nodiscard]] inline AxisType parse_axis_type(std::string_view s) noexcept {
    if (s == "space") return AxisType::space;
    if (s == "time")  return AxisType::time;
    if (s == "channel") return AxisType::channel;
    return AxisType::custom;
}

[[nodiscard]] inline std::string_view axis_type_string(AxisType t) noexcept {
    switch (t) {
        case AxisType::space:   return "space";
        case AxisType::time:    return "time";
        case AxisType::channel: return "channel";
        case AxisType::custom:  return "custom";
    }
    return "custom";
}

[[nodiscard]] std::vector<CoordinateTransform>
parse_transforms(const JsonValue& arr);

[[nodiscard]] JsonValue serialize_transforms(
    const std::vector<CoordinateTransform>& transforms);

} // namespace ome_detail

// ---------------------------------------------------------------------------
// parse_ome_metadata
// ---------------------------------------------------------------------------

[[nodiscard]] MultiscaleMetadata parse_ome_metadata(const JsonValue& attrs);

// ---------------------------------------------------------------------------
// serialize_ome_metadata
// ---------------------------------------------------------------------------

[[nodiscard]] JsonValue serialize_ome_metadata(const MultiscaleMetadata& meta);

// ---------------------------------------------------------------------------
// parse_label_metadata / parse_plate_metadata
// ---------------------------------------------------------------------------

[[nodiscard]] LabelMetadata parse_label_metadata(const JsonValue& attrs);

[[nodiscard]] PlateMetadata parse_plate_metadata(const JsonValue& attrs);

// ---------------------------------------------------------------------------
// Pyramid generation helpers
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::vector<double> compute_downsample_factors(
    const MultiscaleMetadata& meta, std::size_t from_level, std::size_t to_level) {
    if (from_level >= meta.datasets.size() || to_level >= meta.datasets.size())
        throw std::runtime_error("zarr: level index out of range");

    auto get_scale = [&](std::size_t level) -> const std::vector<double>& {
        for (const auto& t : meta.datasets[level].transforms) {
            if (auto* st = std::get_if<ScaleTransform>(&t))
                return st->scale;
        }
        throw std::runtime_error(
            "zarr: no scale transform at level " + std::to_string(level));
    };

    const auto& from_scale = get_scale(from_level);
    const auto& to_scale = get_scale(to_level);

    std::size_t ndim = std::min(from_scale.size(), to_scale.size());
    std::vector<double> factors(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        factors[i] = (from_scale[i] != 0.0) ? (to_scale[i] / from_scale[i]) : 1.0;
    }
    return factors;
}

[[nodiscard]] inline MultiscaleMetadata make_standard_multiscale(
    std::string_view name,
    std::vector<std::size_t> full_shape,
    std::size_t num_levels,
    std::vector<Axis> axes,
    std::array<double, 3> voxel_size = {1.0, 1.0, 1.0}) {

    MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.name = std::string(name);
    meta.type = "gaussian";
    meta.axes = std::move(axes);

    auto xi = meta.axis_index("x");
    auto yi = meta.axis_index("y");
    auto zi = meta.axis_index("z");

    for (std::size_t lvl = 0; lvl < num_levels; ++lvl) {
        MultiscaleDataset ds;
        ds.path = std::to_string(lvl);

        double factor = static_cast<double>(std::size_t{1} << lvl);
        std::vector<double> scale(meta.axes.size(), 1.0);
        if (xi && *xi < scale.size()) scale[*xi] = voxel_size[0] * factor;
        if (yi && *yi < scale.size()) scale[*yi] = voxel_size[1] * factor;
        if (zi && *zi < scale.size()) scale[*zi] = voxel_size[2] * factor;
        ds.transforms.emplace_back(ScaleTransform{std::move(scale)});

        meta.datasets.push_back(std::move(ds));
    }

    return meta;
}

// ---------------------------------------------------------------------------
// OmeZarrReader
// ---------------------------------------------------------------------------

class OmeZarrReader final {
public:
    explicit OmeZarrReader(const std::filesystem::path& root);

    [[nodiscard]] const MultiscaleMetadata& multiscale() const noexcept {
        return meta_;
    }

    [[nodiscard]] const std::vector<Axis>& axes() const noexcept {
        return meta_.axes;
    }

    [[nodiscard]] std::size_t num_levels() const noexcept {
        return meta_.num_levels();
    }

    [[nodiscard]] const ZarrArray& level(std::size_t idx) const {
        if (idx >= levels_.size())
            throw std::runtime_error("zarr: level index out of range");
        return levels_[idx];
    }

    [[nodiscard]] std::array<double, 3> voxel_size(std::size_t lvl = 0) const {
        return meta_.voxel_size(lvl);
    }

    [[nodiscard]] std::vector<std::size_t> shape(std::size_t lvl = 0) const {
        if (lvl >= levels_.size())
            throw std::runtime_error("zarr: level index out of range");
        return levels_[lvl].metadata().shape;
    }

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk(std::size_t lvl, std::span<const std::size_t> chunk_indices) const {
        return level(lvl).read_chunk(chunk_indices);
    }

    [[nodiscard]] bool has_labels() const noexcept {
        return std::filesystem::is_directory(root_ / "labels");
    }

    [[nodiscard]] std::vector<std::string> label_names() const;

    [[nodiscard]] OmeZarrReader open_label(std::string_view name) const {
        auto label_path = root_ / "labels" / std::string(name);
        return OmeZarrReader(label_path);
    }

    [[nodiscard]] bool is_plate() const noexcept;

    [[nodiscard]] std::optional<PlateMetadata> plate_metadata() const;

private:
    std::filesystem::path root_;
    std::shared_ptr<JsonValue> attrs_;
    MultiscaleMetadata meta_;
    std::vector<ZarrArray> levels_;
};

// ---------------------------------------------------------------------------
// OmeZarrWriter
// ---------------------------------------------------------------------------

class OmeZarrWriter final {
public:
    struct Config {
        std::filesystem::path root;
        MultiscaleMetadata multiscale;
        ZarrDtype dtype = ZarrDtype::uint16;
        std::vector<std::size_t> chunk_shape;
        ZarrArray::Codec codec = {};
    };

    explicit OmeZarrWriter(Config config)
        : config_(std::move(config)) {
        std::filesystem::create_directories(config_.root);
    }

    [[nodiscard]] ZarrArray& add_level(std::vector<std::size_t> shape) {
        std::size_t idx = levels_.size();
        if (idx >= config_.multiscale.datasets.size())
            throw std::runtime_error("zarr: too many levels for metadata");

        const auto& ds = config_.multiscale.datasets[idx];
        auto level_path = config_.root / ds.path;

        ZarrMetadata meta;
        meta.shape = std::move(shape);
        meta.chunks = config_.chunk_shape;
        meta.dtype = config_.dtype;
        meta.dimension_separator = "/";

        levels_.push_back(ZarrArray::create(level_path, std::move(meta), config_.codec));
        return levels_.back();
    }

    void write_chunk(std::size_t lvl,
                     std::span<const std::size_t> chunk_indices,
                     std::span<const std::byte> data) {
        if (lvl >= levels_.size())
            throw std::runtime_error("zarr: level index out of range");
        levels_[lvl].write_chunk(chunk_indices, data);
    }

    void finalize();

private:
    Config config_;
    std::vector<ZarrArray> levels_;
};

// ---------------------------------------------------------------------------
// Compression integration helpers
// ---------------------------------------------------------------------------

#if UTILS_HAS_COMPRESSION

/// Map a zarr compressor id string (e.g. "blosc", "zlib", "zstd", "gzip")
/// to a compression.hpp Codec enum value. Returns std::nullopt if unknown.
[[nodiscard]] inline std::optional<Codec> zarr_compressor_to_codec(const std::string& compressor_id) {
    if (compressor_id == "blosc")  return Codec::blosc_lz4;
    return parse_codec(compressor_id);
}

/// Build a ZarrArray::Codec from compression.hpp utilities for a given
/// compressor name (e.g. "blosc", "zlib", "zstd", "gzip") and compression level.
/// This is only available when compression.hpp is found.
[[nodiscard]] inline ZarrArray::Codec make_zarr_codec(const std::string& compressor_id,
                                                       int level = 5) {
    auto codec_enum = zarr_compressor_to_codec(compressor_id);
    if (!codec_enum)
        throw std::runtime_error("zarr: unknown compressor for compression.hpp: " + compressor_id);

    auto ce = *codec_enum;
    ZarrArray::Codec c;
    c.compress = [ce, level](std::span<const std::byte> data) -> std::vector<std::byte> {
        CompressParams params;
        params.codec = ce;
        params.level = level;
        return compress(data, params);
    };
    c.decompress = [ce](std::span<const std::byte> data, std::size_t out_size) -> std::vector<std::byte> {
        return decompress(data, ce, out_size);
    };
    return c;
}

#endif // UTILS_HAS_COMPRESSION

} // namespace utils
