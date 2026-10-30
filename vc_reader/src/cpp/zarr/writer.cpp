// Writing local Zarr arrays: metadata, encoded chunk files and regions.
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/native/codecs.hpp"
#include "format.hpp"
#include "files.hpp"

#include <cstring>
#include <fstream>

namespace fs = std::filesystem;

namespace vcr::zarr {
namespace {
// Advances a C-order index over the box [lo, hi]; false once past its end.
bool nextIndex(Index& idx, const Index& lo, const Index& hi)
{
    for (std::size_t i = idx.size(); i-- > 0;) {
        if (++idx[i] <= hi[i])
            return true;
        idx[i] = lo[i];
    }
    return false;
}

EncodeOptions encodeOptions(const Json& compressor)
{
    EncodeOptions o;
    if (compressor.is_null())
        return o;
    o.cname = compressor.value("cname", o.cname);
    o.level = compressor.value("clevel", compressor.value("level", compressor.value("acceleration", o.level)));
    o.shuffle = compressor.value("shuffle", o.shuffle);
    o.blocksize = compressor.value("blocksize", o.blocksize);
    o.q = compressor.value("q", o.q);
    o.checksum = compressor.value("checksum", false);
    return o;
}

Json compressorConfig(const CreateOptions& options)
{
    const auto& id = options.compressor;
    if (id.empty() || id == "none")
        return nullptr;
    const int level = options.level;
    if (id == "blosc")
        return {{"id", "blosc"}, {"cname", "lz4"}, {"clevel", level > 0 ? level : 3}, {"shuffle", 1}, {"blocksize", 0}};
    if (id == "zstd" || id == "zlib" || id == "gzip")
        return {{"id", id}, {"level", level > 0 ? level : (id == "zstd" ? 3 : 6)}};
    if (id == "lz4")
        return {{"id", "lz4"}, {"acceleration", level > 0 ? level : 1}};
    if (id == "vc-delta3d")
        return {{"id", "vc-delta3d"}, {"quant", 1}};
    if (id == "volcomp")
        return {{"id", "volcomp"}, {"q", level >= 0 ? double(level) : 8.0}};
    throw std::invalid_argument("cannot write Zarr chunks compressed with '" + id + "'");
}

Json v3Metadata(const ArrayMeta& meta, const CreateOptions& options)
{
    std::string dtype;
    for (const auto& [name, typestr] : detail::v3Types)
        if (typestr.substr(1) == meta.dtype.substr(1))
            dtype = name;
    if (dtype.empty())
        throw std::invalid_argument("unsupported Zarr v3 dtype " + meta.dtype);
    Json codecs = Json::array({{{"name", "bytes"},
                               {"configuration", {{"endian", meta.dtype[0] == '>' ? "big" : "little"}}}}});
    if (!meta.compressor.is_null()) {
        const auto id = meta.compressor.at("id").get<std::string>();
        auto config = meta.compressor;
        config.erase("id");
        if (id == "blosc") {
            config["typesize"] = meta.itemsize();
            config["shuffle"] = std::array{"noshuffle", "shuffle", "bitshuffle"}.at(config.at("shuffle").get<int>());
        }
        if (id == "zstd")
            config["checksum"] = false;
        std::string name;
        for (const auto& [candidate, v2] : detail::v3Compressors)
            if (id == v2)
                name = candidate;
        if (name.empty())
            throw std::invalid_argument("unsupported Zarr v3 compressor " + id);
        codecs.push_back({{"name", name}, {"configuration", config}});
    }
    if (!options.shardShape.empty()) {
        Json indexCodecs = Json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}});
        if (options.shardIndexCrc) indexCodecs.push_back({{"name", "crc32c"}});
        codecs = Json::array({{{"name", "sharding_indexed"},
                              {"configuration", {{"chunk_shape", meta.chunks}, {"codecs", codecs},
                                                  {"index_codecs", indexCodecs},
                                                  {"index_location", options.shardIndexEnd ? "end" : "start"}}}}});
    }
    return {{"zarr_format", 3}, {"node_type", "array"}, {"shape", meta.shape}, {"data_type", dtype},
            {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", options.shardShape.empty() ? meta.chunks : options.shardShape}}}}},
            {"chunk_key_encoding", {{"name", "default"}, {"configuration", {{"separator", meta.separator}}}}},
            {"fill_value", meta.raw.at("fill_value")}, {"codecs", codecs}, {"attributes", Json::object()}};
}

Json storedMetadata(const ArrayMeta& meta)
{
    return meta.v3 ? meta.v3->metadata : meta.raw;
}

fs::path metadataPath(const fs::path& path, const ArrayMeta& meta)
{
    return path / (meta.v3 ? "zarr.json" : ".zarray");
}

// Bounded lock table shared by all Writer instances. Canonical array paths make
// relative paths and symlinks to the same shard use the same lock.
std::mutex& shardMutex(const fs::path& path)
{
    static std::array<std::mutex, 256> locks;
    return locks[std::hash<std::string>{}(path.string()) % locks.size()];
}

void putLittle(std::uint8_t* out, std::uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; ++i) out[i] = std::uint8_t(value >> (8 * i));
}
}  // namespace

struct Writer::Impl {
    fs::path path;
    ArrayMeta meta;
    Codec codec = Codec::Raw;
    EncodeOptions encode;
    std::unique_ptr<Array> reader;
    std::unique_ptr<ShardLayout> shards;
    fs::path canonicalPath;

    // Keep encoded neighbors intact and bound scratch memory to one payload
    // and the index. A failed copy never replaces the existing shard.
    bool updateShard(const fs::path& file, std::size_t target, const std::vector<std::uint8_t>* replacement)
    {
        const auto& layout = *shards;
        std::lock_guard lock(shardMutex(canonicalPath / file.lexically_relative(path)));
        LocalShard previous(file, layout);
        bool existed = false;
        std::size_t remaining = 0;
        if (!previous.index.empty()) {
            for (std::size_t i = 0; i < layout.indexBytes / 16; ++i) {
                const bool present = layout.entry(previous.index, i, previous.total).has_value();
                if (i == target) existed = present;
                else remaining += present;
            }
        }
        if (!replacement && !existed) return false;
        if (!replacement && !remaining) {
            fs::remove(file);
            return true;
        }
        detail::atomicWriteFile(file, [&](std::ostream& out) {
            std::vector<std::uint8_t> index(layout.indexBytes, 255), buffer;
            if (!layout.indexEnd)
                out.write(reinterpret_cast<const char*>(index.data()), std::streamsize(index.size()));
            for (std::size_t i = 0; i < layout.indexBytes / 16; ++i) {
                const std::vector<std::uint8_t>* payload = replacement;
                if (i == target) {
                    if (!replacement) continue;
                } else {
                    if (!previous.readEntry(i, buffer)) continue;
                    payload = &buffer;
                }
                const auto offset = out.tellp();
                if (offset < 0) throw std::runtime_error("cannot seek shard output");
                putLittle(index.data() + 16 * i, std::uint64_t(offset), 8);
                putLittle(index.data() + 16 * i + 8, payload->size(), 8);
                out.write(reinterpret_cast<const char*>(payload->data()), std::streamsize(payload->size()));
                if (!out) throw std::runtime_error("cannot write shard payload");
            }
            if (layout.indexCrc)
                putLittle(index.data() + index.size() - 4, crc32c(index.data(), index.size() - 4), 4);
            if (!layout.indexEnd) out.seekp(0);
            out.write(reinterpret_cast<const char*>(index.data()), std::streamsize(index.size()));
        });
        return existed;
    }
};

Writer::Writer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Writer::Writer(Writer&&) noexcept = default;
Writer& Writer::operator=(Writer&&) noexcept = default;
Writer::~Writer() = default;

Writer Writer::create(const fs::path& path, const CreateOptions& options)
{
    if (options.version != 2 && options.version != 3)
        throw std::invalid_argument("Zarr version must be 2 or 3");
    if (options.version != 3 && !options.shardShape.empty())
        throw std::invalid_argument("indexed shards require Zarr v3");
    if (fs::exists(path / ".zgroup") ||
        (options.version == 3 && fs::exists(path / ".zarray")) ||
        (options.version == 2 && fs::exists(path / "zarr.json")))
        throw std::invalid_argument("cannot replace a group or change the Zarr format at " + path.string());
    if (options.version == 3 && fs::exists(path / "zarr.json")) {
        std::ifstream in(path / "zarr.json");
        if (Json::parse(in).value("node_type", "") != "array")
            throw std::invalid_argument("cannot replace a group with an array at " + path.string());
    }
    Json zarray{{"zarr_format", 2},
                {"shape", options.shape},
                {"chunks", options.chunks},
                {"dtype", options.dtype},
                {"compressor", compressorConfig(options)},
                {"fill_value", options.fillValue},
                {"filters", nullptr},
                {"order", "C"},
                {"dimension_separator", options.separator}};
    auto meta = parseArrayMeta(zarray);  // validates before anything is written
    const auto codec = parse_codec(meta.codec());
    if (codec == Codec::Volcomp && (meta.itemsize() != 1 || meta.chunks != Index{128, 128, 128}))
        throw std::invalid_argument("volcomp requires uint8 data and 128^3 chunks");
    if (codec == Codec::Delta3d && (meta.chunks.size() != 3 || meta.type.kind != 'u' || meta.itemsize() > 2))
        throw std::invalid_argument("vc-delta3d requires 3-D uint8 or uint16 chunks");
    if (options.version == 3) {
        zarray = v3Metadata(meta, options);
        meta = parseArrayMeta(zarray);
    }
    fs::create_directories(path);
    atomicWrite(metadataPath(path, meta), zarray.dump(2) + "\n");
    std::error_code ec;
    if (path.has_parent_path() && !fs::exists(path.parent_path() / ".zgroup", ec) &&
        !fs::exists(path.parent_path() / "zarr.json", ec))
        atomicWrite(path.parent_path() / (meta.v3 ? "zarr.json" : ".zgroup"),
                    meta.v3 ? "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {}}\n"
                            : "{\"zarr_format\": 2}\n");
    return open(path);
}

Writer Writer::open(const fs::path& path)
{
    std::ifstream in(path / ".zarray");
    if (!in)
        in.open(path / "zarr.json");
    if (!in)
        throw std::runtime_error("no Zarr array to write at " + path.string());
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    impl->meta = parseArrayMeta(Json::parse(in));
    if (impl->meta.v3 && !impl->meta.v3->shards.empty()) {
        const auto& v3 = *impl->meta.v3;
        impl->shards = std::make_unique<ShardLayout>(v3.shards, v3.indexEnd, v3.indexCrc, impl->meta.chunkBytes());
        impl->canonicalPath = fs::canonical(path);
    }
    impl->codec = parse_codec(impl->meta.codec());
    impl->encode = encodeOptions(impl->meta.compressor);
    OpenOptions options;
    options.threads = 4;
    // Mutable arrays must see atomic replacements by this or another writer.
    options.readMode = "slab";
    impl->reader = std::move(openArray(path.string(), options).array);
    return Writer(std::move(impl));
}

const ArrayMeta& Writer::meta() const { return impl_->meta; }
const fs::path& Writer::path() const { return impl_->path; }
Array& Writer::reader() { return *impl_->reader; }

void Writer::setFillValue(const Json& value)
{
    // Fetch the current v3 attributes as another writer may have updated them.
    auto raw = storedMetadata(impl_->meta);
    if (impl_->meta.v3) {
        std::ifstream in(metadataPath(impl_->path, impl_->meta));
        raw = Json::parse(in);
    }
    raw["fill_value"] = value;
    auto meta = parseArrayMeta(raw);
    OpenOptions options;
    options.threads = 4;
    options.readMode = "slab";
    ArrayHints hints;
    hints.metadata = raw;
    auto opened = openArray(impl_->path.string(), options, hints);
    atomicWrite(metadataPath(impl_->path, meta), raw.dump(2) + "\n");
    impl_->meta = std::move(meta);
    impl_->reader = std::move(opened.array);
}

fs::path Writer::chunkPath(const Index& chunk) const
{
    const auto& meta = impl_->meta;
    if (chunk.size() != meta.shape.size())
        throw std::invalid_argument("chunk index rank does not match the array");
    std::string key = meta.v3 ? meta.v3->prefix : "";
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        const auto grid = meta.shape[i] / meta.chunks[i] + (meta.shape[i] % meta.chunks[i] != 0);
        if (chunk[i] < 0 || chunk[i] >= std::max<std::int64_t>(grid, 1))
            throw std::out_of_range("chunk index outside the array");
        if (i)
            key += meta.separator;
        key += std::to_string(impl_->shards ? chunk[i] / impl_->shards->ratios[i] : chunk[i]);
    }
    return impl_->path / key;
}

bool Writer::chunkExists(const Index& chunk) const
{
    const auto file = chunkPath(chunk);
    if (impl_->shards) {
        LocalShard shard(file, *impl_->shards);
        return !shard.index.empty() &&
               impl_->shards->entry(shard.index, impl_->shards->locate(chunk).second, shard.total).has_value();
    }
    std::error_code ec;
    return fs::exists(file, ec);
}

void Writer::writeChunk(const Index& chunk, const std::uint8_t* data, std::size_t size)
{
    const auto& meta = impl_->meta;
    if (size != meta.chunkBytes())
        throw std::invalid_argument("chunk data has " + std::to_string(size) + " bytes, expected " +
                                    std::to_string(meta.chunkBytes()));
    const auto path = chunkPath(chunk);
    const auto bytes = encode(impl_->codec, impl_->encode, data, size, meta.itemsize(), meta.chunks);
    if (impl_->shards) {
        impl_->updateShard(path, impl_->shards->locate(chunk).second, &bytes);
        return;
    }
    atomicWrite(path, std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

bool Writer::writeChunkSkipEmpty(const Index& chunk, const std::uint8_t* data, std::size_t size)
{
    if (size != impl_->meta.chunkBytes())
        throw std::invalid_argument("wrong chunk data size");
    const auto& fill = impl_->meta.fill;
    bool allFill = size % fill.size() == 0;
    for (std::size_t i = 0; allFill && i < size; i += fill.size())
        allFill = std::memcmp(data + i, fill.data(), fill.size()) == 0;
    if (allFill) {
        removeChunk(chunk);
        return false;
    }
    writeChunk(chunk, data, size);
    return true;
}

bool Writer::removeChunk(const Index& chunk)
{
    const auto file = chunkPath(chunk);
    if (impl_->shards)
        return impl_->updateShard(file, impl_->shards->locate(chunk).second, nullptr);
    std::error_code ec;
    const bool removed = fs::remove(file, ec);
    if (ec)
        throw std::system_error(ec, "removing " + chunkPath(chunk).string());
    return removed;
}

void Writer::writeRegion(const Index& start, const Index& shape, const std::uint8_t* data)
{
    const auto& meta = impl_->meta;
    const std::size_t nd = meta.shape.size();
    if (start.size() != nd || shape.size() != nd)
        throw std::invalid_argument("region rank does not match the array");
    for (std::size_t i = 0; i < nd; ++i) {
        if (shape[i] == 0)
            return;
        if (start[i] < 0 || shape[i] < 0 || start[i] + shape[i] > meta.shape[i])
            throw std::invalid_argument("region outside the array");
    }
    const auto itemsize = meta.itemsize();
    const Index regionStrides = c_strides(shape, itemsize), chunkStrides = c_strides(meta.chunks, itemsize);
    Index c0(nd), c1(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        c0[i] = start[i] / meta.chunks[i];
        c1[i] = (start[i] + shape[i] - 1) / meta.chunks[i];
    }
    std::vector<std::uint8_t> buffer(meta.chunkBytes());
    Index idx = c0;
    do {
        Index base(nd), lo(nd), extent(nd);
        bool covered = true;
        for (std::size_t i = 0; i < nd; ++i) {
            base[i] = idx[i] * meta.chunks[i];
            const auto a = std::max(start[i], base[i]);
            const auto b = std::min(start[i] + shape[i], base[i] + meta.chunks[i]);
            lo[i] = a;
            extent[i] = b - a;
            // A chunk is covered when the region spans it inside the array (edge chunks
            // extend past the array; their outside part keeps the fill value).
            covered &= a == base[i] && b == std::min(base[i] + meta.chunks[i], meta.shape[i]);
        }
        if (covered) {
            for (std::size_t i = 0; i < buffer.size(); i += meta.fill.size())
                std::memcpy(buffer.data() + i, meta.fill.data(), meta.fill.size());
        } else {
            impl_->reader->read_many({base}, {meta.chunks}, {buffer.data()});
        }
        Index src(nd), dst(nd);
        for (std::size_t i = 0; i < nd; ++i) {
            src[i] = lo[i] - start[i];
            dst[i] = lo[i] - base[i];
        }
        copy_box(data + offset_of(src, regionStrides), regionStrides, buffer.data() + offset_of(dst, chunkStrides),
                 chunkStrides, extent, itemsize);
        writeChunk(idx, buffer.data(), buffer.size());
    } while (nextIndex(idx, c0, c1));
}

Json readAttributes(const fs::path& node)
{
    std::ifstream v3(node / "zarr.json");
    if (v3)
        return Json::parse(v3).value("attributes", Json::object());
    std::ifstream in(node / ".zattrs");
    return in ? Json::parse(in) : Json::object();
}

void writeAttributes(const fs::path& node, const Json& attributes)
{
    if (!attributes.is_object())
        throw std::invalid_argument("Zarr attributes must be an object");
    std::ifstream in(node / "zarr.json");
    if (in) {
        auto metadata = Json::parse(in);
        metadata["attributes"] = attributes;
        atomicWrite(node / "zarr.json", metadata.dump(2) + "\n");
        return;
    }
    atomicWrite(node / ".zattrs", attributes.dump(2) + "\n");
}

}  // namespace vcr::zarr
