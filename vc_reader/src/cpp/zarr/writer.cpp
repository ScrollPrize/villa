// Writing Zarr arrays, local or on S3: metadata, encoded chunks and regions.
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/native/codecs.hpp"
#include "format.hpp"
#include "files.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unistd.h>

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
        return {{"id", "blosc"}, {"cname", options.cname}, {"clevel", level > 0 ? level : 3}, {"shuffle", options.shuffle}, {"blocksize", 0}};
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
    if (options.packbits)
        codecs.insert(codecs.begin(), Json{{"name", "numcodecs.packbits"}, {"configuration", Json::object()}});
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

const char* metadataName(const ArrayMeta& meta)
{
    return meta.v3 ? "zarr.json" : ".zarray";
}

const char* groupMarker(int version)
{
    return version == 3 ? "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {}}\n"
                        : "{\"zarr_format\": 2}\n";
}

// A file named for this process and call, removed afterwards.
struct TemporaryFile {
    fs::path path;
    TemporaryFile()
    {
        static std::atomic<std::uint64_t> sequence{0};
        path = fs::temp_directory_path() / ("vc_reader-shard-" + std::to_string(::getpid()) + "-" + std::to_string(++sequence));
    }
    ~TemporaryFile()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

// The location's parent, when it has one to hold a group marker.
std::optional<fs::path> parentLocation(const Store& store)
{
    const fs::path path(store.location());
    if (store.remote() && store.location().find('/', store.location().find("://") + 3) == std::string::npos)
        return std::nullopt;  // a bucket has no parent
    return path.has_parent_path() ? std::optional<fs::path>(path.parent_path()) : std::nullopt;
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
    std::optional<Store> store;
    ArrayMeta meta;
    ChunkCodec codec;
    EncodeOptions encode;
    std::unique_ptr<Array> reader;
    std::unique_ptr<ShardLayout> shards;
    fs::path canonicalPath;

    // The object name of a chunk, or of the shard containing it.
    std::string chunkKey(const Index& chunk) const
    {
        if (chunk.size() != meta.shape.size())
            throw std::invalid_argument("chunk index rank does not match the array");
        std::string key = meta.v3 ? meta.v3->prefix : "";
        for (std::size_t i = 0; i < chunk.size(); ++i) {
            const auto grid = meta.shape[i] / meta.chunks[i] + (meta.shape[i] % meta.chunks[i] != 0);
            if (chunk[i] < 0 || chunk[i] >= std::max<std::int64_t>(grid, 1))
                throw std::out_of_range("chunk index outside the array");
            if (i)
                key += meta.separator;
            key += std::to_string(shards ? chunk[i] / shards->ratios[i] : chunk[i]);
        }
        return key;
    }

    // A remote shard edited as a local file: downloaded, patched and uploaded whole.
    bool updateShard(const std::string& key, std::size_t target, const std::vector<std::uint8_t>* replacement)
    {
        if (!store->remote()) {
            std::lock_guard lock(shardMutex((canonicalPath / key).string()));
            return patchShard(path / key, target, replacement);
        }
        std::lock_guard lock(shardMutex(store->location() + "/" + key));
        TemporaryFile local;
        if (const auto stored = store->read(key))
            atomicWrite(local.path, *stored);
        const bool result = patchShard(local.path, target, replacement);
        std::ifstream in(local.path, std::ios::binary);
        if (in) {
            std::ostringstream text;
            text << in.rdbuf();
            store->write(key, text.str());
        } else {
            store->discard(key);
        }
        return result;
    }

    // Keep encoded neighbors intact and bound scratch memory to one payload
    // and the index. A failed copy never replaces the existing shard.
    bool patchShard(const fs::path& file, std::size_t target, const std::vector<std::uint8_t>* replacement)
    {
        const auto& layout = *shards;
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

std::vector<std::uint8_t> compressBytes(const Json& compressor, std::span<const std::uint8_t> data, std::size_t itemsize, const Index& shape)
{
    const auto codec = parse_codec(compressor.is_null() ? std::string() : compressor.at("id").get<std::string>());
    return encode(codec, encodeOptions(compressor), data.data(), data.size(), itemsize,
                  shape.empty() ? Index{std::int64_t(data.size() / itemsize)} : shape);
}

void decompressBytes(const Json& compressor, std::span<const std::uint8_t> data, std::span<std::uint8_t> out)
{
    const auto codec = parse_codec(compressor.is_null() ? std::string() : compressor.at("id").get<std::string>());
    decode(codec, data.data(), data.size(), out.data(), out.size());
}

CreateOptions createOptionsLike(const ArrayMeta& meta)
{
    CreateOptions options;
    options.shape = meta.shape;
    options.chunks = meta.chunks;
    options.dtype = meta.dtype;
    options.separator = meta.separator;
    options.fillValue = meta.raw.value("fill_value", Json(0));
    options.compressor = meta.compressor.is_null() ? "none" : meta.compressor.at("id").get<std::string>();
    options.packbits = !meta.filters.is_null() && !meta.filters.empty();
    if (!meta.compressor.is_null()) {
        const auto encode = encodeOptions(meta.compressor);
        options.level = options.compressor == "volcomp" ? int(encode.q) : encode.level;
        options.cname = encode.cname;
        options.shuffle = encode.shuffle;
    }
    if (meta.v3) {
        options.version = 3;
        options.shardIndexEnd = meta.v3->indexEnd;
        options.shardIndexCrc = meta.v3->indexCrc;
        for (std::size_t axis = 0; axis < meta.v3->shards.size(); ++axis)
            options.shardShape.push_back(meta.v3->shards[axis] * meta.chunks[axis]);
    }
    return options;
}

Writer::Writer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Writer::Writer(Writer&&) noexcept = default;
Writer& Writer::operator=(Writer&&) noexcept = default;
Writer::~Writer() = default;

Writer Writer::create(const fs::path& path, const CreateOptions& options, const Access& access)
{
    if (options.version != 2 && options.version != 3)
        throw std::invalid_argument("Zarr version must be 2 or 3");
    if (options.version != 3 && !options.shardShape.empty())
        throw std::invalid_argument("indexed shards require Zarr v3");
    const Store store(path, access);
    if (options.overwrite)
        store.removeAll();
    if (store.exists(".zgroup") ||
        (options.version == 3 && store.exists(".zarray")) ||
        (options.version == 2 && store.exists("zarr.json")))
        throw std::invalid_argument("cannot replace a group or change the Zarr format at " + path.string());
    if (options.version == 3) {
        const auto existing = store.read("zarr.json");
        if (existing && Json::parse(*existing).value("node_type", "") != "array")
            throw std::invalid_argument("cannot replace a group with an array at " + path.string());
    }
    Json zarray{{"zarr_format", 2},
                {"shape", options.shape},
                {"chunks", options.chunks},
                {"dtype", options.dtype},
                {"compressor", compressorConfig(options)},
                {"fill_value", options.fillValue},
                {"filters", options.packbits ? Json::array({{{"id", "packbits"}}}) : Json(nullptr)},
                {"order", "C"},
                {"dimension_separator", options.separator}};
    auto meta = parseArrayMeta(zarray);  // validates before anything is written
    const auto codec = parse_chunk_codec(meta.codec());
    if (codec == Codec::Volcomp && (meta.itemsize() != 1 || meta.chunks != Index{128, 128, 128}))
        throw std::invalid_argument("volcomp requires uint8 data and 128^3 chunks");
    if (codec == Codec::Delta3d && (meta.chunks.size() != 3 || meta.type.kind != 'u' || meta.itemsize() > 2))
        throw std::invalid_argument("vc-delta3d requires 3-D uint8 or uint16 chunks");
    if (options.version == 3) {
        zarray = v3Metadata(meta, options);
        meta = parseArrayMeta(zarray);
    }
    if (!store.remote())
        fs::create_directories(path);
    store.write(metadataName(meta), zarray.dump(2) + "\n");
    if (const auto parent = parentLocation(store)) {
        const Store above(*parent, access, 1);
        if (!above.exists(".zgroup") && !above.exists("zarr.json"))
            above.write(meta.v3 ? "zarr.json" : ".zgroup", groupMarker(meta.v3 ? 3 : 2));
    }
    return open(path, access);
}

Writer Writer::open(const fs::path& path, const Access& access)
{
    Store store(path, access);
    auto stored = store.read(".zarray");
    if (!stored)
        stored = store.read("zarr.json");
    if (!stored)
        throw std::runtime_error("no Zarr array to write at " + path.string());
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    impl->store = std::move(store);
    impl->meta = parseArrayMeta(Json::parse(*stored));
    if (impl->meta.v3 && !impl->meta.v3->shards.empty()) {
        const auto& v3 = *impl->meta.v3;
        impl->shards = std::make_unique<ShardLayout>(v3.shards, v3.indexEnd, v3.indexCrc, impl->meta.chunkBytes());
        if (!impl->store->remote())
            impl->canonicalPath = fs::canonical(path);
    }
    impl->codec = parse_chunk_codec(impl->meta.codec());
    impl->encode = encodeOptions(impl->meta.compressor);
    OpenOptions options;
    options.access = access;
    options.threads = 4;
    // Mutable arrays must see atomic replacements by this or another writer.
    options.readMode = "slab";
    impl->reader = std::move(openArray(impl->store->location(), options).array);
    return Writer(std::move(impl));
}

const ArrayMeta& Writer::meta() const { return impl_->meta; }
const fs::path& Writer::path() const { return impl_->path; }
Array& Writer::reader() { return *impl_->reader; }

void Writer::setFillValue(const Json& value)
{
    // Fetch the current v3 attributes as another writer may have updated them.
    auto raw = storedMetadata(impl_->meta);
    if (impl_->meta.v3)
        raw = Json::parse(impl_->store->read(metadataName(impl_->meta)).value());
    raw["fill_value"] = value;
    auto meta = parseArrayMeta(raw);
    OpenOptions options;
    options.access = impl_->store->access();
    options.threads = 4;
    options.readMode = "slab";
    ArrayHints hints;
    hints.metadata = raw;
    auto opened = openArray(impl_->store->location(), options, hints);
    impl_->store->write(metadataName(meta), raw.dump(2) + "\n");
    impl_->meta = std::move(meta);
    impl_->reader = std::move(opened.array);
}

fs::path Writer::chunkPath(const Index& chunk) const
{
    return impl_->path / impl_->chunkKey(chunk);
}

bool Writer::chunkExists(const Index& chunk) const
{
    const auto key = impl_->chunkKey(chunk);
    if (impl_->shards) {
        TemporaryFile local;
        if (impl_->store->remote()) {
            const auto stored = impl_->store->read(key);
            if (!stored) return false;
            atomicWrite(local.path, *stored);
        }
        LocalShard shard(impl_->store->remote() ? local.path : impl_->path / key, *impl_->shards);
        return !shard.index.empty() &&
               impl_->shards->entry(shard.index, impl_->shards->locate(chunk).second, shard.total).has_value();
    }
    return impl_->store->exists(key);
}

std::vector<Index> Writer::storedChunks() const
{
    const auto& meta = impl_->meta;
    const std::string prefix = meta.v3 ? meta.v3->prefix : "";
    const auto rank = meta.shape.size();
    Index grid(rank);
    for (std::size_t i = 0; i < rank; ++i)
        grid[i] = std::max<std::int64_t>(meta.shape[i] / meta.chunks[i] + (meta.shape[i] % meta.chunks[i] != 0), 1);
    std::vector<Index> stored;
    for (const auto& object : impl_->store->list(prefix)) {
        Index index;
        std::string_view rest(object.key);
        rest.remove_prefix(prefix.size());
        for (;;) {
            const auto end = rest.find(meta.separator);
            const auto part = rest.substr(0, end);
            std::int64_t value = 0;
            const auto parsed = std::from_chars(part.data(), part.data() + part.size(), value);
            if (part.empty() || parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size()) {
                index.clear();
                break;
            }
            index.push_back(value);
            if (end == std::string_view::npos)
                break;
            rest.remove_prefix(end + meta.separator.size());
        }
        if (index.size() != rank)
            continue;
        if (!impl_->shards) {
            bool inside = true;
            for (std::size_t i = 0; i < rank; ++i)
                inside &= index[i] >= 0 && index[i] < grid[i];
            if (inside)
                stored.push_back(std::move(index));
            continue;
        }
        const auto& ratios = impl_->shards->ratios;
        std::size_t count = 1;
        for (std::size_t i = 0; i < rank; ++i)
            count *= std::size_t(ratios[i]);
        for (std::size_t n = 0; n < count; ++n) {
            Index chunk(rank);
            auto rem = n;
            bool inside = true;
            for (std::size_t i = rank; i-- > 0;) {
                chunk[i] = index[i] * ratios[i] + std::int64_t(rem % std::size_t(ratios[i]));
                rem /= std::size_t(ratios[i]);
                inside &= chunk[i] >= 0 && chunk[i] < grid[i];
            }
            if (inside && chunkExists(chunk))
                stored.push_back(std::move(chunk));
        }
    }
    std::sort(stored.begin(), stored.end());
    return stored;
}

void Writer::writeChunk(const Index& chunk, const std::uint8_t* data, std::size_t size)
{
    const auto& meta = impl_->meta;
    if (size != meta.chunkBytes())
        throw std::invalid_argument("chunk data has " + std::to_string(size) + " bytes, expected " +
                                    std::to_string(meta.chunkBytes()));
    const auto key = impl_->chunkKey(chunk);
    const auto bytes = encode(impl_->codec, impl_->encode, data, size, meta.itemsize(), meta.chunks);
    if (impl_->shards) {
        impl_->updateShard(key, impl_->shards->locate(chunk).second, &bytes);
        return;
    }
    impl_->store->write(key, std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
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
        if (impl_->shards)
            removeChunk(chunk);
        else
            impl_->store->discard(impl_->chunkKey(chunk));
        return false;
    }
    writeChunk(chunk, data, size);
    return true;
}

bool Writer::removeChunk(const Index& chunk)
{
    const auto key = impl_->chunkKey(chunk);
    if (impl_->shards)
        return impl_->updateShard(key, impl_->shards->locate(chunk).second, nullptr);
    return impl_->store->remove(key);
}

void Writer::writeRegion(const Index& start, const Index& shape, const std::uint8_t* data, bool skipEmpty)
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
        if (skipEmpty)
            writeChunkSkipEmpty(idx, buffer.data(), buffer.size());
        else
            writeChunk(idx, buffer.data(), buffer.size());
    } while (nextIndex(idx, c0, c1));
}

bool removeNode(const fs::path& path, const Access& access)
{
    return Store(path, access).removeAll();
}

std::vector<std::string> nodeChildren(const fs::path& path, const Access& access)
{
    std::vector<std::string> out;
    for (const auto& name : Store(path, access, 1).children())
        if (Store(path / name, access, 1).isNode())
            out.push_back(name);
    return out;
}

void createGroup(const fs::path& path, const std::optional<Json>& attributes, int version, const Access& access)
{
    if (version != 2 && version != 3)
        throw std::invalid_argument("Zarr version must be 2 or 3");
    const Store store(path, access, 1);
    if (!store.remote())
        fs::create_directories(path);
    if (store.exists(".zarray"))
        throw std::invalid_argument("an array is at " + path.string());
    if (!store.exists(".zgroup") && !store.exists("zarr.json"))
        store.write(version == 3 ? "zarr.json" : ".zgroup", groupMarker(version));
    if (attributes)
        writeAttributes(path, *attributes, access);
}

Json readAttributes(const fs::path& node, const Access& access)
{
    const Store store(node, access, 1);
    if (const auto v3 = store.read("zarr.json"))
        return Json::parse(*v3).value("attributes", Json::object());
    const auto attributes = store.read(".zattrs");
    return attributes ? Json::parse(*attributes) : Json::object();
}

void writeAttributes(const fs::path& node, const Json& attributes, const Access& access)
{
    if (!attributes.is_object())
        throw std::invalid_argument("Zarr attributes must be an object");
    const Store store(node, access, 1);
    if (const auto v3 = store.read("zarr.json")) {
        auto metadata = Json::parse(*v3);
        metadata["attributes"] = attributes;
        store.write("zarr.json", metadata.dump(2) + "\n");
        return;
    }
    store.write(".zattrs", attributes.dump(2) + "\n");
}

}  // namespace vcr::zarr
