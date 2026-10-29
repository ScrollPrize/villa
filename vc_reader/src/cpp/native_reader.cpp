#include "vc_reader/reader.hpp"
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
        {"fill_value", d.at("fill_value")}, {"compressor", compressor}, {"filters", nullptr},
        {"order", "C"}, {"dimension_separator", sep},
        {"_vc_reader_v3", {{"metadata", d}, {"prefix", keyName == "default" ? "c" + sep : ""},
            {"shard_chunks", ratios}, {"index_end", indexEnd}, {"index_crc", indexCrc}}}}.dump();
}

struct CtReader::Impl {
    zarr::OpenedArray opened;
    std::array<int64_t, 3> shape;
};

CtReader::CtReader(const std::string& input, int level, size_t cacheBytes)
    : impl_(std::make_unique<Impl>())
{
    if (level < 0) throw std::invalid_argument("negative CT level");
    std::string location = input;
    while (location.size() > 1 && location.back() == '/') location.pop_back();
    if (location.empty()) throw std::invalid_argument("empty CT location");
    location = catalog::resolveOpenDataUrl(location);
    zarr::OpenOptions options;
    options.memoryCache = std::make_shared<MemoryCache>(cacheBytes);
    options.metadata = zarr::metadataReader(options.access);
    const bool array = options.metadata(location, ".zarray") ||
                       [&] {
                           const auto node = options.metadata(location, "zarr.json");
                           return node && Json::parse(*node).value("node_type", "") == "array";
                       }();
    if (array) {
        if (level != 0) throw std::invalid_argument("array source has only level zero");
        impl_->opened = zarr::openArray(location, options);
    } else {
        auto group = zarr::Group::open(location, options);
        std::string key = std::to_string(level);
        if (!group.multiscales().empty()) {
            const auto levels = group.levels();
            if (size_t(level) >= levels.size()) throw std::invalid_argument("CT level missing");
            key = levels[size_t(level)];
        }
        impl_->opened = group.array(key);
    }
    const auto& meta = impl_->opened.meta;
    if (meta.type.kind != 'u' || meta.itemsize() != 1)
        throw std::invalid_argument("CT normals require uint8 CT");
    if (meta.shape.size() != 3)
        throw std::invalid_argument("CT must be three dimensional");
    for (int axis = 0; axis < 3; ++axis) impl_->shape[axis] = meta.shape[axis];
}

CtReader::~CtReader() = default;
CtReader::CtReader(CtReader&&) noexcept = default;
CtReader& CtReader::operator=(CtReader&&) noexcept = default;

std::array<int64_t, 3> CtReader::shape() const { return impl_->shape; }
Array& CtReader::array() { return *impl_->opened.array; }

void CtReader::readRegion(std::array<int64_t, 3> start, std::array<int64_t, 3> shape,
                          std::span<uint8_t> out)
{
    size_t count = 1;
    for (int axis = 0; axis < 3; ++axis) {
        const auto size = shape[axis];
        if (size < 0 || (size && count > SIZE_MAX / uint64_t(size)) || start[axis] > INT64_MAX - size)
            throw std::invalid_argument("invalid CT region");
        count *= size;
    }
    if (out.size() != count) throw std::invalid_argument("CT buffer size mismatch");
    impl_->opened.array->read_many({Index(start.begin(), start.end())},
                           {Index(shape.begin(), shape.end())}, {out.data()});
}

template<class T>
void normals(std::span<const T> patches, std::span<const std::array<float, 4>> centers,
             std::span<std::array<float, 4>> out)
{
    if (centers.size() > SIZE_MAX / (109 * 109 * 109) ||
        patches.size() != centers.size() * 109 * 109 * 109 || out.size() != centers.size()) {
        throw std::invalid_argument("sheet normal buffer sizes differ");
    }
    for (const auto& center : centers) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(center[axis]) || center[axis] < 48 || center[axis] >= 51)
                throw std::invalid_argument("invalid sheet-normal center");
        }
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

std::vector<std::array<float, 4>> CtReader::sheetNormals(std::span<const std::array<double, 3>> xyz)
{
    std::vector<std::array<float, 4>> out(xyz.size());
    constexpr size_t patchSize = 109 * 109 * 109, batchSize = 8;
    for (size_t base = 0; base < xyz.size(); base += batchSize) {
        const size_t count = std::min(batchSize, xyz.size() - base);
        std::vector<uint8_t> patches(count * patchSize);
        std::vector<std::array<float, 4>> centers(count);
        std::vector<Index> starts, sizes;
        std::vector<uint8_t*> buffers;
        for (size_t i = 0; i < count; ++i) {
            std::array<int64_t, 3> start{};
            bool valid = true;
            for (int axis = 0; axis < 3; ++axis) {
                // Keep global coordinates precise until the lattice origin is removed.
                const double point = xyz[base + i][axis];
                if (!std::isfinite(point) || std::abs(point) >= 0x1p52)
                    throw std::invalid_argument("invalid sheet-normal point");
                const double lattice = std::floor(point / 3) * 3;
                start[2 - axis] = int64_t(lattice) - 54;
                centers[i][axis] = std::min(float(point - lattice) + 48.f, std::nextafter(51.f, 48.f));
                valid &= start[2 - axis] >= 0 && start[2 - axis] <= impl_->shape[2 - axis] - 109;
            }
            centers[i][3] = valid;
            if (valid) {
                starts.emplace_back(start.begin(), start.end());
                sizes.push_back({109, 109, 109});
                buffers.push_back(patches.data() + i * patchSize);
            }
        }
        if (!starts.empty()) impl_->opened.array->read_many(starts, sizes, buffers);
        sheetNormalsCpu(patches, centers, std::span(out).subspan(base, count));
    }
    return out;
}
} // namespace vcr
