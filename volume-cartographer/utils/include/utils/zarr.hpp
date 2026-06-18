#pragma once
#include "utils/Json.hpp"
#include "http_fetch.hpp"
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
#include <limits>

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
    [[nodiscard]] std::vector<std::byte> encode(std::span<const std::byte> input) const;

    /// Apply filter inverse (decode).
    [[nodiscard]] std::vector<std::byte> decode(std::span<const std::byte> input) const;

private:
    [[nodiscard]] std::vector<std::byte> encode_delta(std::span<const std::byte> input) const;
    [[nodiscard]] std::vector<std::byte> decode_delta(std::span<const std::byte> input) const;
    [[nodiscard]] std::vector<std::byte> encode_fixedscaleoffset(std::span<const std::byte> input) const;
    [[nodiscard]] std::vector<std::byte> decode_fixedscaleoffset(std::span<const std::byte> input) const;
    [[nodiscard]] std::vector<std::byte> encode_quantize(std::span<const std::byte> input) const;
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

/// Structural check: zarr v3, sharded, with 256^3 inner chunks (c3d codec
/// atom). This only checks shape; the inner bytes still need a C3DC
/// magic-check before decode.
[[nodiscard]] bool is_canonical_c3d(const ZarrMetadata& m) noexcept;

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
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const;
};

// ---------------------------------------------------------------------------
// FileSystemStore
// ---------------------------------------------------------------------------

class FileSystemStore final : public Store {
public:
    explicit FileSystemStore(std::filesystem::path root) : root_(std::move(root)) {}

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

    [[nodiscard]] bool exists(const std::string& key) const override;
    [[nodiscard]] std::vector<std::byte> get(const std::string& key) const override;
    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_if_exists(const std::string& key) const override;
    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override;
    void set(const std::string& key, std::span<const std::byte> value) override;
    void erase(const std::string& key) override;

private:
    [[nodiscard]] std::filesystem::path safe_path(const std::string& key) const;

    std::filesystem::path root_;
};

// ---------------------------------------------------------------------------
// HttpStore -- read-only store over HTTP/S3
// ---------------------------------------------------------------------------

class HttpStore final : public Store {
public:
    /// Construct from a base URL with default HttpClient settings.
    explicit HttpStore(std::string base_url);

    /// Construct from a base URL with a custom HttpClient config.
    HttpStore(std::string base_url, HttpClient::Config config);

    /// Construct from a base URL with AWS SigV4 authentication.
    HttpStore(std::string base_url, AwsAuth auth);

    [[nodiscard]] bool exists(const std::string& key) const override;
    [[nodiscard]] std::vector<std::byte> get(const std::string& key) const override;
    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_if_exists(const std::string& key) const override;
    void set(const std::string& key, std::span<const std::byte> value) override;
    void erase(const std::string& key) override;
    [[nodiscard]] std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override;

private:
    [[nodiscard]] std::string make_url(const std::string& key) const;
    static std::string strip_trailing_slash(std::string s);

    std::string base_url_;
    std::shared_ptr<HttpClient> client_;
};

// ---------------------------------------------------------------------------
// detail -- I/O and metadata helpers
// ---------------------------------------------------------------------------

namespace detail {

// ----- File I/O helpers -----

std::string read_file(const std::filesystem::path& p);
std::vector<std::byte> read_file_bytes(const std::filesystem::path& p);
void write_file(const std::filesystem::path& p, std::string_view data);
void write_file_bytes(const std::filesystem::path& p, std::span<const std::byte> data);

// ----- Endian helpers -----

bool is_little_endian() noexcept;
void byteswap_inplace(std::span<std::byte> data, std::size_t elem_size);

// ----- Little-endian integer encode/decode for shard index -----

void write_le64(std::byte* dst, std::uint64_t val);
std::uint64_t read_le64(const std::byte* src);

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
    [[nodiscard]] std::vector<std::byte> serialize() const;

    /// Deserialize from binary.
    static ShardIndex deserialize(std::span<const std::byte> data, std::size_t num_chunks);
};

// ----- .zarray (v2) parse / serialize -----

ZarrMetadata parse_zarray(std::string_view json_str);

std::string serialize_zarray(const ZarrMetadata& meta);

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

ZarrVersion detect_version(const std::filesystem::path& path);

} // namespace detail

// Owns a block of bytes that was either mmap'd read-only from a file
// (preferred on POSIX for shard files — kernel handles residency) or
// heap-allocated (fallback). Non-copyable, movable. Static factories
// make the ownership discriminator explicit at the call site.
class ShardBytes final {
public:
    ShardBytes() = default;
    ~ShardBytes() { release(); }
    ShardBytes(const ShardBytes&) = delete;
    ShardBytes& operator=(const ShardBytes&) = delete;
    ShardBytes(ShardBytes&& o) noexcept { take(std::move(o)); }
    ShardBytes& operator=(ShardBytes&& o) noexcept {
        if (this != &o) { release(); take(std::move(o)); }
        return *this;
    }

    static ShardBytes from_mmap(void* ptr, std::size_t n) noexcept;
    static ShardBytes from_vector(std::vector<std::byte> v) noexcept;

    const std::byte* data() const noexcept {
        return mapped_ ? static_cast<const std::byte*>(mapped_) : vec_.data();
    }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    std::span<const std::byte> span() const noexcept { return {data(), size_}; }
    // True when the backing is an mmap rather than heap (diagnostic).
    bool is_mmap() const noexcept { return mapped_ != nullptr; }

private:
    void release() noexcept;
    void take(ShardBytes&& o) noexcept;

    void* mapped_ = nullptr;      // mmap region if non-null
    std::vector<std::byte> vec_;  // otherwise heap-owned
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// ZarrArray
// ---------------------------------------------------------------------------

class ZarrArray final {
public:
    /// Codec callback struct: compress and decompress functions.
    /// `decompress_into` is optional; when provided, callers can avoid the
    /// per-call decompressed-buffer heap allocation by passing a reusable
    /// scratch span sized to the (sub-)chunk byte count.
    struct Codec {
        std::function<std::vector<std::byte>(std::span<const std::byte>)> compress;
        std::function<std::vector<std::byte>(std::span<const std::byte>, std::size_t)> decompress;
        std::function<void(std::span<const std::byte>, std::span<std::byte>)> decompress_into;
    };

    /// Named codec registry: maps codec name to Codec.
    using CodecRegistry = std::unordered_map<std::string, Codec>;

    // -- Open / Create -------------------------------------------------------

    /// Open an existing zarr array at `path`. Auto-detects v2 vs v3.
    static ZarrArray open(const std::filesystem::path& path, Codec codec = {});

    /// Open with a codec registry (for v3 codec pipelines).
    static ZarrArray open(const std::filesystem::path& path, CodecRegistry registry);

    /// Create a new zarr array, writing metadata to disk.
    static ZarrArray create(const std::filesystem::path& path,
                            ZarrMetadata meta, Codec codec = {});

    /// Create a new zarr array with a codec registry.
    static ZarrArray create(const std::filesystem::path& path,
                            ZarrMetadata meta, CodecRegistry registry);

    /// Open from a Store.
    static ZarrArray open(std::shared_ptr<Store> store, const std::string& array_key,
                          Codec codec = {});

    /// Open from a Store with a codec registry.
    static ZarrArray open(std::shared_ptr<Store> store, const std::string& array_key,
                          CodecRegistry registry);

    /// Open with pre-parsed metadata (no file I/O). Used by open_from_consolidated.
    static ZarrArray open_with_metadata(const std::filesystem::path& path,
                                         ZarrMetadata meta, Codec codec = {});

    // -- Accessors -----------------------------------------------------------

    [[nodiscard]] const ZarrMetadata& metadata() const noexcept { return meta_; }
    [[nodiscard]] ZarrVersion version() const noexcept { return meta_.version; }
    [[nodiscard]] bool is_sharded() const noexcept { return meta_.shard_config.has_value(); }

    // -- Chunk I/O (non-sharded) ---------------------------------------------

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk(std::span<const std::size_t> chunk_indices) const;

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk_encoded(std::span<const std::size_t> chunk_indices) const;

    [[nodiscard]] std::vector<std::byte>
    decode_chunk_payload(std::span<const std::byte> payload) const;

    [[nodiscard]] bool stores_chunks_with_codec(std::string_view codec_name) const noexcept;

    /// Read a chunk and decompress it directly into the caller-provided
    /// `output` buffer. `output` must be at least sub_chunk_byte_size().
    /// Returns false if the chunk is missing on disk; otherwise writes
    /// exactly sub_chunk_byte_size() bytes into `output` and returns true.
    ///
    /// When the codec exposes `decompress_into`, decompression skips the
    /// per-call heap allocation that `read_chunk` performs. v2 filters and
    /// host-byte-order mismatches force a fallback to the allocating path
    /// (and a final memcpy into `output`).
    [[nodiscard]] bool
    read_chunk_into(std::span<const std::size_t> chunk_indices,
                    std::span<std::byte> output) const;

    void write_chunk(std::span<const std::size_t> chunk_indices,
                     std::span<const std::byte> data);

    [[nodiscard]] bool chunk_exists(std::span<const std::size_t> chunk_indices) const;

    [[nodiscard]] std::string
    chunk_key(std::span<const std::size_t> chunk_indices) const;

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
                     std::span<const std::size_t> inner_indices) const;

    // 4k-align chunk payloads inside shards so decoders can mmap the shard
    // and hand direct pointers to the codec without buffering.  Index entries
    // use the aligned offset; padding bytes between chunks are ignored by
    // the reader (spec allows arbitrary gaps).
    static constexpr std::uint64_t kShardChunkAlign = 4096;

    /// Write a complete shard given a set of inner chunk data.
    /// `inner_chunks` is indexed by linear inner chunk index (C-order).
    /// Missing inner chunks should be std::nullopt.
    void write_shard(std::span<const std::size_t> shard_indices,
                     std::span<const std::optional<std::vector<std::byte>>> inner_chunks);

    /// Append a single chunk to its shard file. Index at start (fixed 8KB header).
    /// Two tiny writes: (1) append chunk data at EOF, (2) update 16-byte index entry.
    /// Creates shard with empty index if it doesn't exist.
    void write_inner_chunk_to_shard(std::span<const std::size_t> chunk_indices,
                                    std::span<const std::byte> data);

    /// Check if an inner chunk exists. Reads only the 16-byte index entry.
    [[nodiscard]] bool inner_chunk_exists(std::span<const std::size_t> chunk_indices) const;

    /// Check if an inner chunk is marked as known-empty (zero data).
    [[nodiscard]] bool inner_chunk_is_empty(std::span<const std::size_t> chunk_indices) const;

    /// Mark an inner chunk as known-empty (zero data). Writes sentinel to index.
    void mark_inner_chunk_empty(std::span<const std::size_t> chunk_indices);

    /// Write a shard file whose index marks every inner chunk as empty.
    /// Used to record "this shard is known empty" without a remote round-trip.
    void write_empty_shard(std::span<const std::size_t> shard_indices);

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

    [[nodiscard]] bool needs_compression() const noexcept;
    [[nodiscard]] bool needs_decompression() const noexcept;
    [[nodiscard]] bool needs_byteswap() const noexcept;

    // -- Chunk key generation ------------------------------------------------

    [[nodiscard]] std::string chunk_key_v2(std::span<const std::size_t> idx) const;
    [[nodiscard]] std::string chunk_key_v3(std::span<const std::size_t> idx) const;

    // -- Raw chunk I/O (no compression/filters) ------------------------------

    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_raw(const std::string& key) const;

public:
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_chunk_raw(std::span<const std::size_t> idx) const {
        return read_raw(chunk_key(idx));
    }

    void write_chunk_raw(std::span<const std::size_t> idx,
                         std::span<const std::byte> data);

    // -- Shard reading helpers -----------------------------------------------

    /// Read a single chunk from its shard. Reads only the 16-byte index entry
    /// + the chunk data. Does NOT read the whole shard.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    read_inner_chunk_from_shard(std::span<const std::size_t> chunk_indices) const;

    /// Read the whole shard file that contains the given inner-chunk indices.
    /// Returns the raw shard bytes (including the trailing index). Use with
    /// extract_inner_chunk() to pull out individual inner chunks — useful
    /// when a caller maintains an external shard-level cache and wants to
    /// serve many inner chunks from a single disk read.
    ///
    /// On POSIX the returned bytes are backed by a read-only mmap so the
    /// kernel's page cache owns residency — under memory pressure unused
    /// pages drop without touching our allocator. Inner chunks written
    /// later invalidate the view (caller must drop/refresh); shard_mutex
    /// serialises reads vs. write_inner_chunk_to_shard.
    [[nodiscard]] std::optional<ShardBytes>
    read_whole_shard(std::span<const std::size_t> chunk_indices) const;

    /// Extract a single inner chunk from shard data.
    [[nodiscard]] std::optional<std::vector<std::byte>>
    extract_inner_chunk(std::span<const std::byte> shard_data,
                        std::span<const std::size_t> inner_indices) const;

    // -- Members -------------------------------------------------------------
private:
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
// open_remote -- open a ZarrArray from an HTTP/S3 URL
// ---------------------------------------------------------------------------

/// Open a zarr array from a URL (HTTP or S3).
/// For S3 URLs (s3://bucket/key or s3+REGION://bucket/key), the URL is
/// auto-converted to HTTPS and AWS SigV4 auth is applied if provided.
/// If no auth is given for an S3 URL, credentials are read from the
/// environment (AWS_ACCESS_KEY_ID, etc.).
///
/// The url should point to the array root (the directory containing
/// .zarray or zarr.json). If array_key is empty, metadata is looked up
/// at the url root itself.
[[nodiscard]] ZarrArray open_remote(
    const std::string& url,
    std::optional<AwsAuth> auth = std::nullopt,
    ZarrArray::Codec codec = {},
    const std::string& array_key = {});

// ---------------------------------------------------------------------------
// Consolidated metadata reader (v2)
// ---------------------------------------------------------------------------

/// Load consolidated metadata from .zmetadata file.
[[nodiscard]] detail::ConsolidatedMetadata
load_consolidated_metadata(const std::filesystem::path& root);

/// Open a zarr array using consolidated metadata (avoids per-array metadata reads).
/// Uses the pre-parsed metadata directly without writing files.
[[nodiscard]] ZarrArray open_from_consolidated(
    const std::filesystem::path& root,
    const std::string& array_path,
    const detail::ConsolidatedMetadata& cm,
    ZarrArray::Codec codec = {});

// ---------------------------------------------------------------------------
// Pyramid helpers
// ---------------------------------------------------------------------------

[[nodiscard]] std::size_t count_pyramid_levels(const std::filesystem::path& root);

[[nodiscard]] std::vector<ZarrArray> open_pyramid(
    const std::filesystem::path& root, ZarrArray::Codec codec = {});

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

    [[nodiscard]] std::array<double, 3> voxel_size(std::size_t level) const;
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

[[nodiscard]] AxisType parse_axis_type(std::string_view s) noexcept;

[[nodiscard]] std::string_view axis_type_string(AxisType t) noexcept;

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

[[nodiscard]] std::vector<double> compute_downsample_factors(
    const MultiscaleMetadata& meta, std::size_t from_level, std::size_t to_level);

[[nodiscard]] MultiscaleMetadata make_standard_multiscale(
    std::string_view name,
    std::vector<std::size_t> full_shape,
    std::size_t num_levels,
    std::vector<Axis> axes,
    std::array<double, 3> voxel_size = {1.0, 1.0, 1.0});

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

    [[nodiscard]] ZarrArray& add_level(std::vector<std::size_t> shape);

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

} // namespace utils

namespace vc {
// Declared here (not VcDataset.hpp) so its CodecRegistry return type doesn't
// drag zarr.hpp into every VcDataset.hpp includer; defined in VcDataset.cpp.
utils::ZarrArray::CodecRegistry buildZarrCodecRegistry(int dtypeSize);
}
