// Writing local Zarr v2 arrays: metadata, encoded chunk files and regions.
#include "vc_reader/zarr.hpp"
#include "vc_reader/native/codecs.hpp"

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
}  // namespace

struct Writer::Impl {
    fs::path path;
    ArrayMeta meta;
    Codec codec = Codec::Raw;
    EncodeOptions encode;
    std::unique_ptr<Array> reader;
};

Writer::Writer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Writer::Writer(Writer&&) noexcept = default;
Writer& Writer::operator=(Writer&&) noexcept = default;
Writer::~Writer() = default;

Writer Writer::create(const fs::path& path, const CreateOptions& options)
{
    Json zarray{{"zarr_format", 2},
                {"shape", options.shape},
                {"chunks", options.chunks},
                {"dtype", options.dtype},
                {"compressor", compressorConfig(options)},
                {"fill_value", options.fillValue},
                {"filters", nullptr},
                {"order", "C"},
                {"dimension_separator", options.separator}};
    const auto meta = parseArrayMeta(zarray);  // validates before anything is written
    const auto codec = parse_codec(meta.codec());
    if (codec == Codec::Volcomp && (meta.itemsize() != 1 || meta.chunks != Index{128, 128, 128}))
        throw std::invalid_argument("volcomp requires uint8 data and 128^3 chunks");
    if (codec == Codec::Delta3d && (meta.chunks.size() != 3 || meta.type.kind != 'u' || meta.itemsize() > 2))
        throw std::invalid_argument("vc-delta3d requires 3-D uint8 or uint16 chunks");
    fs::create_directories(path);
    atomicWrite(path / ".zarray", zarray.dump(2) + "\n");
    std::error_code ec;
    if (path.has_parent_path() && !fs::exists(path.parent_path() / ".zgroup", ec) &&
        !fs::exists(path.parent_path() / "zarr.json", ec))
        atomicWrite(path.parent_path() / ".zgroup", "{\"zarr_format\": 2}\n");
    return open(path);
}

Writer Writer::open(const fs::path& path)
{
    std::ifstream in(path / ".zarray");
    if (!in)
        throw std::runtime_error("no Zarr v2 array to write at " + path.string());
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    impl->meta = parseArrayMeta(Json::parse(in));
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
    auto raw = impl_->meta.raw;
    raw["fill_value"] = value;
    auto meta = parseArrayMeta(raw);
    OpenOptions options;
    options.threads = 4;
    options.readMode = "slab";
    ArrayHints hints;
    hints.metadata = raw;
    auto opened = openArray(impl_->path.string(), options, hints);
    atomicWrite(impl_->path / ".zarray", raw.dump(2) + "\n");
    impl_->meta = std::move(meta);
    impl_->reader = std::move(opened.array);
}

fs::path Writer::chunkPath(const Index& chunk) const
{
    const auto& meta = impl_->meta;
    if (chunk.size() != meta.shape.size())
        throw std::invalid_argument("chunk index rank does not match the array");
    std::string key;
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        const auto grid = (meta.shape[i] + meta.chunks[i] - 1) / meta.chunks[i];
        if (chunk[i] < 0 || chunk[i] >= std::max<std::int64_t>(grid, 1))
            throw std::out_of_range("chunk index outside the array");
        if (i)
            key += meta.separator;
        key += std::to_string(chunk[i]);
    }
    return impl_->path / key;
}

bool Writer::chunkExists(const Index& chunk) const
{
    std::error_code ec;
    return fs::exists(chunkPath(chunk), ec);
}

void Writer::writeChunk(const Index& chunk, const std::uint8_t* data, std::size_t size)
{
    const auto& meta = impl_->meta;
    if (size != meta.chunkBytes())
        throw std::invalid_argument("chunk data has " + std::to_string(size) + " bytes, expected " +
                                    std::to_string(meta.chunkBytes()));
    const auto path = chunkPath(chunk);
    const auto bytes = encode(impl_->codec, impl_->encode, data, size, meta.itemsize(), meta.chunks);
    atomicWrite(path, std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

bool Writer::writeChunkSkipEmpty(const Index& chunk, const std::uint8_t* data, std::size_t size)
{
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
    std::error_code ec;
    const bool removed = fs::remove(chunkPath(chunk), ec);
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
    std::ifstream in(node / ".zattrs");
    return in ? Json::parse(in) : Json::object();
}

void writeAttributes(const fs::path& node, const Json& attributes)
{
    atomicWrite(node / ".zattrs", attributes.dump(2) + "\n");
}

}  // namespace vcr::zarr
