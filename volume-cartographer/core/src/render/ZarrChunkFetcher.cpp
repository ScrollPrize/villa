#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/CacheCompression.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vc::render {

namespace {

class HttpStatusError final : public std::runtime_error {
public:
    HttpStatusError(long status, const std::string& key)
        : std::runtime_error("HTTP " + std::to_string(status) + " fetching " + key)
        , status_(status)
    {
    }

    long status() const noexcept { return status_; }

private:
    long status_ = 0;
};

bool hasSuffix(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

bool isOptionalMetadataProbe(const std::string& key)
{
    return key == "zarr.json" || key == ".zattrs" ||
           hasSuffix(key, "/zarr.json") || hasSuffix(key, "/.zattrs");
}

class ClassifyingHttpStore final : public utils::Store {
public:
    explicit ClassifyingHttpStore(std::string baseUrl, vc::HttpAuth auth = {})
        : baseUrl_(stripTrailingSlash(std::move(baseUrl)))
        , client_(makeClient(std::move(auth)))
    {
    }

    bool exists(const std::string& key) const override
    {
        auto response = client_.head(makeUrl(key));
        if (response.ok())
            return true;
        if (response.not_found())
            return false;
        if (response.status_code == 403 && isOptionalMetadataProbe(key))
            return false;
        throw HttpStatusError(response.status_code, key);
    }

    std::vector<std::byte> get(const std::string& key) const override
    {
        auto found = get_if_exists(key);
        if (!found)
            throw std::runtime_error("HTTP zarr key not found: " + key);
        return std::move(*found);
    }

    std::optional<std::vector<std::byte>> get_if_exists(const std::string& key) const override
    {
        auto response = client_.get(makeUrl(key));
        if (response.ok())
            return std::move(response.body);
        if (response.not_found())
            return std::nullopt;
        if (response.status_code == 403 && isOptionalMetadataProbe(key))
            return std::nullopt;
        throw HttpStatusError(response.status_code, key);
    }

    std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override
    {
        auto response = client_.get_range(makeUrl(key), offset, length);
        if (response.ok())
            return std::move(response.body);
        if (response.not_found())
            return std::nullopt;
        throw HttpStatusError(response.status_code, key);
    }

    void set(const std::string&, std::span<const std::byte>) override
    {
        throw std::runtime_error("HTTP zarr store is read-only");
    }

    void erase(const std::string&) override
    {
        throw std::runtime_error("HTTP zarr store is read-only");
    }

private:
    std::string makeUrl(const std::string& key) const
    {
        return vc::joinRemoteUrlPath(baseUrl_, key);
    }

    static std::string stripTrailingSlash(std::string value)
    {
        while (!value.empty() && value.back() == '/')
            value.pop_back();
        return value;
    }

    static utils::HttpClient makeClient(vc::HttpAuth auth)
    {
        utils::HttpClient::Config config;
        config.aws_auth = std::move(auth);
        config.transfer_timeout = std::chrono::seconds{60};
        return utils::HttpClient(std::move(config));
    }

    std::string baseUrl_;
    utils::HttpClient client_;
};

class ZarrChunkFetcher final : public IChunkFetcher {
public:
    explicit ZarrChunkFetcher(utils::ZarrArray array)
        : array_(std::make_unique<utils::ZarrArray>(std::move(array)))
    {
        // Source encodings already compact enough to persist verbatim,
        // avoiding a decode+re-encode round trip on the cache writer.
        if (array_->stores_chunks_with_codec("c3d"))
            persistEncodedExtension_ = ".c3d";
        else if (array_->stores_chunks_with_codec(vc::kVcz1CodecName))
            persistEncodedExtension_ = vc::kCompressedCacheExtension;
    }

    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ChunkFetchResult result;
        const std::array<std::size_t, 3> indices{
            static_cast<std::size_t>(key.iz),
            static_cast<std::size_t>(key.iy),
            static_cast<std::size_t>(key.ix)};

        try {
            if (!persistEncodedExtension_.empty()) {
                auto encoded = array_->read_chunk_encoded(indices);
                if (!encoded) {
                    result.status = ChunkFetchStatus::Missing;
                    return result;
                }
                result.status = ChunkFetchStatus::Found;
                result.persistentBytes = std::move(*encoded);
                result.hasPersistentBytes = true;
                result.bytes = array_->decode_chunk_payload(
                    std::span<const std::byte>(result.persistentBytes.data(),
                                               result.persistentBytes.size()));
                return result;
            }

            auto bytes = array_->read_chunk(indices);
            if (!bytes) {
                result.status = ChunkFetchStatus::Missing;
                return result;
            }
            result.status = ChunkFetchStatus::Found;
            result.bytes = std::move(*bytes);
            return result;
        } catch (const HttpStatusError& e) {
            result.status = ChunkFetchStatus::HttpError;
            result.httpStatus = static_cast<int>(e.status());
            result.message = e.what();
        } catch (const std::filesystem::filesystem_error& e) {
            result.status = ChunkFetchStatus::IoError;
            result.message = e.what();
        } catch (const std::exception& e) {
            result.status = ChunkFetchStatus::DecodeError;
            result.message = e.what();
        }
        return result;
    }

    std::string persistentCacheExtension(const ChunkKey&) const override
    {
        return persistEncodedExtension_.empty() ? ".bin" : persistEncodedExtension_;
    }

    ChunkFetchResult decodePersistentBytes(
        const ChunkKey&,
        std::vector<std::byte> bytes) const override
    {
        ChunkFetchResult result;
        try {
            result.status = ChunkFetchStatus::Found;
            if (!persistEncodedExtension_.empty()) {
                result.hasPersistentBytes = true;
                result.persistentBytes = std::move(bytes);
                result.bytes = array_->decode_chunk_payload(
                    std::span<const std::byte>(result.persistentBytes.data(),
                                               result.persistentBytes.size()));
            } else {
                result.bytes = std::move(bytes);
            }
        } catch (const std::exception& e) {
            result.status = ChunkFetchStatus::DecodeError;
            result.message = e.what();
        }
        return result;
    }

private:
    std::unique_ptr<utils::ZarrArray> array_;
    std::string persistEncodedExtension_;
};

std::array<int, 3> toArray3(const std::vector<std::size_t>& values, const char* name)
{
    if (values.size() != 3)
        throw std::runtime_error(std::string("zarr ") + name + " must be 3D");
    return {
        static_cast<int>(values[0]),
        static_cast<int>(values[1]),
        static_cast<int>(values[2])};
}

void addLevel(OpenedChunkedZarr& opened, utils::ZarrArray array)
{
    const auto& meta = array.metadata();
    ChunkDtype dtype = ChunkDtype::UInt8;
    if (meta.dtype == utils::ZarrDtype::uint16) {
        dtype = ChunkDtype::UInt16;
    } else if (meta.dtype != utils::ZarrDtype::uint8) {
        throw std::runtime_error("streaming zarr fetcher currently supports uint8 and uint16 only");
    }
    if (!opened.fetchers.empty() && opened.dtype != dtype)
        throw std::runtime_error("streaming zarr fetcher requires all levels to have the same dtype");

    std::vector<std::size_t> chunkShape = meta.chunks;
    if (meta.shard_config)
        chunkShape = meta.shard_config->sub_chunks;

    opened.shapes.push_back(toArray3(meta.shape, "shape"));
    opened.chunkShapes.push_back(toArray3(chunkShape, "chunk shape"));
    opened.storageChunkShapes.push_back(toArray3(meta.chunks, "storage chunk shape"));
    const int logicalLevel = static_cast<int>(opened.transforms.size());
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << logicalLevel);
    IChunkedArray::LevelTransform transform;
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    opened.transforms.push_back(transform);
    opened.fillValue = meta.fill_value.value_or(0.0);
    opened.dtype = dtype;
    opened.fetchers.push_back(std::make_shared<ZarrChunkFetcher>(std::move(array)));
    opened.fillValues.push_back(meta.fill_value.value_or(0.0));
}

void addPhysicalLevel(OpenedChunkedZarr& opened, int physicalLevel, utils::ZarrArray array)
{
    if (physicalLevel < 0)
        throw std::runtime_error("zarr physical level must be non-negative");

    const auto index = static_cast<std::size_t>(physicalLevel);
    if (opened.shapes.size() <= index) {
        opened.levelNumbers.resize(index + 1, -1);
        opened.transforms.resize(index + 1);
        opened.shapes.resize(index + 1, {0, 0, 0});
        opened.chunkShapes.resize(index + 1, {1, 1, 1});
        opened.storageChunkShapes.resize(index + 1, {1, 1, 1});
        opened.fetchers.resize(index + 1);
        opened.fillValues.resize(index + 1, 0.0);
    }
    if (opened.fetchers[index])
        throw std::runtime_error("duplicate zarr physical level " + std::to_string(physicalLevel));

    OpenedChunkedZarr single;
    addLevel(single, std::move(array));
    const bool hasExistingLevel = std::any_of(
        opened.fetchers.begin(),
        opened.fetchers.end(),
        [](const auto& fetcher) { return static_cast<bool>(fetcher); });
    if (hasExistingLevel && opened.dtype != single.dtype)
        throw std::runtime_error("streaming zarr fetcher requires all levels to have the same dtype");

    opened.levelNumbers[index] = physicalLevel;
    opened.shapes[index] = single.shapes[0];
    opened.chunkShapes[index] = single.chunkShapes[0];
    opened.storageChunkShapes[index] = single.storageChunkShapes[0];
    IChunkedArray::LevelTransform transform;
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << physicalLevel);
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    opened.transforms[index] = transform;
    opened.fillValue = single.fillValue;
    opened.dtype = single.dtype;
    opened.fetchers[index] = std::move(single.fetchers[0]);
    opened.fillValues[index] = single.fillValues[0];
}

bool paddedShapeOK(long long actual, long long expected, int padMultiple)
{
    return padMultiple > 0 && actual >= expected && actual - expected < padMultiple;
}

int ceilDivPow2(int value, int level)
{
    const auto divisor = std::uint64_t{1} << level;
    return static_cast<int>((static_cast<std::uint64_t>(value) + divisor - 1) / divisor);
}

bool finiteMetadataEqual(double a, double b)
{
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

void requireZyxAxes(const utils::JsonValue& axes, const std::string& context)
{
    if (!axes.is_array())
        throw std::runtime_error(context + " axes must be an array");
    const std::array<std::string, 3> expected{"z", "y", "x"};
    if (axes.size() != expected.size())
        throw std::runtime_error(context + " must declare exactly z, y, x axes");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& axis = axes[i];
        std::string name;
        if (axis.is_string()) {
            name = axis.get_string();
        } else if (axis.is_object() && axis.contains("name") && axis["name"].is_string()) {
            name = axis["name"].get_string();
        } else {
            throw std::runtime_error(context + " axis declaration is malformed");
        }
        if (name != expected[i])
            throw std::runtime_error(context + " axis order must be exactly z, y, x");
    }
}

void validateArrayAxes(const std::shared_ptr<utils::Store>& store,
                       const std::string& key)
{
    const auto validateAttrs = [&](const utils::JsonValue& attrs,
                                   const std::string& context) {
        for (const char* name : {"_ARRAY_DIMENSIONS", "dimension_names", "axes"}) {
            if (attrs.is_object() && attrs.contains(name))
                requireZyxAxes(attrs[name], context + " " + name);
        }
    };

    if (auto data = store->get_if_exists(key + "/.zattrs")) {
        const std::string json(reinterpret_cast<const char*>(data->data()), data->size());
        validateAttrs(utils::json_parse(json), "zarr group /" + key);
    }
    if (auto data = store->get_if_exists(key + "/zarr.json")) {
        const std::string json(reinterpret_cast<const char*>(data->data()), data->size());
        const auto root = utils::json_parse(json);
        if (root.is_object() && root.contains("dimension_names"))
            requireZyxAxes(root["dimension_names"], "zarr group /" + key + " dimension_names");
        if (root.is_object() && root.contains("attributes"))
            validateAttrs(root["attributes"], "zarr group /" + key + " attributes");
    }
}

struct StrictMultiscaleDescriptor {
    bool advertised = false;
    std::vector<std::string> keys;
    bool levelZeroTransformIsIdentity = true;
};

StrictMultiscaleDescriptor strictRemoteLevelsFromZattrs(
    const std::shared_ptr<utils::Store>& store)
{
    StrictMultiscaleDescriptor result;
    auto data = store->get_if_exists(".zattrs");
    if (!data)
        return result;
    const std::string json(reinterpret_cast<const char*>(data->data()), data->size());
    const auto attrs = utils::json_parse(json);
    if (!attrs.is_object() || !attrs.contains("multiscales"))
        return result;
    result.advertised = true;
    if (!attrs["multiscales"].is_array() || attrs["multiscales"].empty() ||
        !attrs["multiscales"][0].is_object()) {
        throw std::runtime_error("OME multiscales metadata must contain one descriptor");
    }
    const auto& ms = attrs["multiscales"][0];
    if (ms.contains("axes"))
        requireZyxAxes(ms["axes"], "OME multiscales");
    if (!ms.contains("datasets") || !ms["datasets"].is_array() || ms["datasets"].empty())
        throw std::runtime_error("OME multiscales datasets must be a nonempty array");

    std::vector<std::optional<std::array<double, 3>>> scales;
    std::vector<std::optional<std::array<double, 3>>> translations;
    int maximum = -1;
    for (const auto& dataset : ms["datasets"]) {
        if (!dataset.is_object() || !dataset.contains("path") || !dataset["path"].is_string())
            throw std::runtime_error("OME multiscales dataset path must be a numeric string");
        const std::string path = dataset["path"].get_string();
        if (path.empty() || (path.size() > 1 && path.front() == '0') ||
            !std::all_of(path.begin(), path.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            throw std::runtime_error("OME multiscales dataset paths must be canonical numeric group names");
        }
        int level = 0;
        const auto parsed = std::from_chars(path.data(), path.data() + path.size(), level);
        if (parsed.ec != std::errc{} || parsed.ptr != path.data() + path.size() || level < 0 || level >= 32)
            throw std::runtime_error("OME multiscales dataset path is outside the supported 0..31 range");
        maximum = std::max(maximum, level);
        if (result.keys.size() <= static_cast<std::size_t>(level)) {
            result.keys.resize(level + 1);
            scales.resize(level + 1);
            translations.resize(level + 1);
        }
        if (!result.keys[level].empty())
            throw std::runtime_error("duplicate OME multiscales numeric dataset path /" + path);
        result.keys[level] = path;

        if (!dataset.contains("coordinateTransformations"))
            continue;
        const auto& transforms = dataset["coordinateTransformations"];
        if (!transforms.is_array())
            throw std::runtime_error("OME coordinateTransformations must be an array");
        std::array<double, 3> scale{1.0, 1.0, 1.0};
        std::array<double, 3> translation{0.0, 0.0, 0.0};
        for (const auto& transform : transforms) {
            if (!transform.is_object() || !transform.contains("type") ||
                !transform["type"].is_string())
                throw std::runtime_error("OME coordinate transformation is malformed");
            const auto type = transform["type"].get_string();
            const char* valuesKey = type == "scale" ? "scale" : type == "translation" ? "translation" : nullptr;
            if (!valuesKey)
                throw std::runtime_error("unsupported OME coordinate transformation type '" + type + "'");
            if (!transform.contains(valuesKey) || !transform[valuesKey].is_array() ||
                transform[valuesKey].size() != 3)
                throw std::runtime_error("OME " + type + " transformation must have three values");
            auto& target = type == "scale" ? scale : translation;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!transform[valuesKey][axis].is_number())
                    throw std::runtime_error("OME " + type + " values must be numeric");
                target[axis] = transform[valuesKey][axis].get_double();
            }
        }
        scales[level] = scale;
        translations[level] = translation;
    }
    for (int level = 0; level <= maximum; ++level) {
        if (result.keys[level].empty())
            throw std::runtime_error("OME multiscales numeric dataset paths contain a gap at /" +
                                     std::to_string(level));
    }

    const std::array<double, 3> baseScale =
        (!scales.empty() && scales[0])
            ? *scales[0]
            : std::array<double, 3>{1.0, 1.0, 1.0};
    if (!scales.empty() && scales[0]) {
        for (double value : *scales[0]) {
            if (!finiteMetadataEqual(value, 1.0))
                result.levelZeroTransformIsIdentity = false;
        }
    }
    for (int level = 0; level <= maximum; ++level) {
        if (translations[level]) {
            for (double value : *translations[level]) {
                if (!finiteMetadataEqual(value, 0.0))
                    throw std::runtime_error("OME coordinate translations must be finite and zero");
            }
        }
        if (!scales[level])
            continue;
        const auto& scale = *scales[level];
        for (double value : scale) {
            if (!std::isfinite(value) || value <= 0.0)
                throw std::runtime_error("OME coordinate scales must be finite and positive");
        }
        if (!finiteMetadataEqual(scale[0], scale[1]) || !finiteMetadataEqual(scale[0], scale[2]))
            throw std::runtime_error("OME coordinate scales must be isotropic");
        const double expected = baseScale[0] *
            static_cast<double>(std::uint64_t{1} << level);
        if (!finiteMetadataEqual(scale[0], expected))
            throw std::runtime_error("OME coordinate scale for /" + std::to_string(level) +
                                     " is not dyadic relative to /0");
    }
    return result;
}

std::vector<int> localLevelNumbers(const std::filesystem::path& root)
{
    std::vector<int> levels;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory())
            continue;
        const auto name = entry.path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            }))
            continue;
        if (std::filesystem::exists(entry.path() / ".zarray") ||
            std::filesystem::exists(entry.path() / "zarr.json")) {
            levels.push_back(std::stoi(name));
        }
    }
    std::sort(levels.begin(), levels.end());
    return levels;
}

std::vector<std::pair<int, std::string>> remoteLevelKeysFromZattrs(
    const std::shared_ptr<utils::Store>& store,
    int firstLevel)
{
    auto data = store->get_if_exists(".zattrs");
    if (!data)
        return {};

    const std::string json(reinterpret_cast<const char*>(data->data()), data->size());
    auto attrs = utils::json_parse(json);
    if (!attrs.contains("multiscales") || !attrs["multiscales"].is_array() ||
        attrs["multiscales"].empty()) {
        return {};
    }

    const auto& ms0 = attrs["multiscales"][0];
    if (!ms0.contains("datasets") || !ms0["datasets"].is_array())
        return {};

    std::vector<std::pair<int, std::string>> keys;
    int datasetIndex = 0;
    for (const auto& dataset : ms0["datasets"]) {
        if (!dataset.contains("path") || !dataset["path"].is_string()) {
            ++datasetIndex;
            continue;
        }
        std::string path = dataset["path"].get_string();
        while (!path.empty() && path.front() == '/')
            path.erase(path.begin());
        while (!path.empty() && path.back() == '/')
            path.pop_back();
        if (!path.empty() && datasetIndex >= firstLevel)
            keys.emplace_back(datasetIndex, std::move(path));
        ++datasetIndex;
    }
    return keys;
}

void addRemoteLevelFromKey(
    OpenedChunkedZarr& opened,
    const std::shared_ptr<utils::Store>& store,
    const std::string& key,
    int physicalLevel)
{
    auto array = utils::ZarrArray::open(store, key, vc::buildZarrCodecRegistry(1));
    if (array.metadata().dtype == utils::ZarrDtype::uint16)
        array = utils::ZarrArray::open(store, key, vc::buildZarrCodecRegistry(2));
    addPhysicalLevel(opened, physicalLevel, std::move(array));
}

} // namespace

OpenedChunkedZarr validateAndRebaseVcPyramid(
    OpenedChunkedZarr opened,
    int baseScaleLevel)
{
    if (baseScaleLevel < 0 || baseScaleLevel > vc::kMaxRemoteVolumeBaseScale) {
        throw std::invalid_argument("VC pyramid base scale must be from 0 through " +
                                    std::to_string(vc::kMaxRemoteVolumeBaseScale));
    }
    int lastLevel = -1;
    for (std::size_t level = 0; level < opened.fetchers.size(); ++level) {
        if (opened.fetchers[level])
            lastLevel = static_cast<int>(level);
    }
    if (lastLevel < 0)
        throw std::runtime_error("VC pyramid contains no numeric groups");
    if (baseScaleLevel > lastLevel)
        throw std::runtime_error("requested VC pyramid base group /" +
                                 std::to_string(baseScaleLevel) + " is unavailable");
    for (int level = 0; level <= lastLevel; ++level) {
        const auto index = static_cast<std::size_t>(level);
        if (index >= opened.fetchers.size() || !opened.fetchers[index])
            throw std::runtime_error("VC pyramid numeric groups contain a gap at /" +
                                     std::to_string(level));
        if (index >= opened.levelNumbers.size() || opened.levelNumbers[index] != level)
            throw std::runtime_error("VC pyramid physical group numbering is inconsistent at /" +
                                     std::to_string(level));
    }

    const auto baseShape = opened.shapes[0];
    for (int level = 0; level <= lastLevel; ++level) {
        const auto index = static_cast<std::size_t>(level);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const int expected = ceilDivPow2(baseShape[axis], level);
            if (!paddedShapeOK(opened.shapes[index][axis], expected,
                               opened.storageChunkShapes[index][axis])) {
                throw std::runtime_error(
                    "VC pyramid group /" + std::to_string(level) +
                    " shape is not a dyadic downscale of /0 within storage-chunk padding tolerance");
            }
        }
    }

    const double retainedFill = opened.fillValues[static_cast<std::size_t>(baseScaleLevel)];
    for (int level = baseScaleLevel; level <= lastLevel; ++level) {
        if (opened.fillValues[static_cast<std::size_t>(level)] != retainedFill) {
            throw std::runtime_error("VC pyramid retained groups must have one consistent fill value");
        }
    }

    const auto erasePrefix = [baseScaleLevel](auto& values) {
        values.erase(values.begin(), values.begin() + baseScaleLevel);
    };
    erasePrefix(opened.levelNumbers);
    erasePrefix(opened.transforms);
    erasePrefix(opened.shapes);
    erasePrefix(opened.chunkShapes);
    erasePrefix(opened.storageChunkShapes);
    erasePrefix(opened.fetchers);
    erasePrefix(opened.fillValues);
    for (std::size_t logicalLevel = 0; logicalLevel < opened.fetchers.size(); ++logicalLevel) {
        opened.levelNumbers[logicalLevel] = static_cast<int>(logicalLevel);
        const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << logicalLevel);
        opened.transforms[logicalLevel].scaleFromLevel0 = {invScale, invScale, invScale};
    }
    opened.fillValue = retainedFill;
    return opened;
}

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root)
{
    OpenedChunkedZarr opened;
    for (int level : localLevelNumbers(root)) {
        auto array = utils::ZarrArray::open(root / std::to_string(level),
                                            vc::buildZarrCodecRegistry(1));
        if (array.metadata().dtype == utils::ZarrDtype::uint16) {
            array = utils::ZarrArray::open(root / std::to_string(level),
                                           vc::buildZarrCodecRegistry(2));
        }
        addPhysicalLevel(opened, level, std::move(array));
    }
    if (opened.fetchers.empty()) {
        auto array = utils::ZarrArray::open(root, vc::buildZarrCodecRegistry(1));
        if (array.metadata().dtype == utils::ZarrDtype::uint16)
            array = utils::ZarrArray::open(root, vc::buildZarrCodecRegistry(2));
        addPhysicalLevel(opened, 0, std::move(array));
    }
    return opened;
}

OpenedChunkedZarr openHttpZarrPyramid(
    const std::string& url,
    const vc::HttpAuth& auth,
    std::optional<int> explicitBaseScaleLevel)
{
    const auto spec = vc::parseRemoteVolumeSpec(url);
    if (explicitBaseScaleLevel && spec.hasBaseScaleSelector &&
        *explicitBaseScaleLevel != spec.baseScaleLevel) {
        throw std::invalid_argument(
            "explicit base scale conflicts with the remote volume locator selector");
    }
    const int baseScaleLevel = explicitBaseScaleLevel.value_or(spec.baseScaleLevel);
    auto store = std::make_shared<ClassifyingHttpStore>(spec.sourceUrl, auth);
    OpenedChunkedZarr opened;
    const bool strictRebasedOpen = baseScaleLevel > 0 || explicitBaseScaleLevel.has_value();

    if (strictRebasedOpen) {
        const auto descriptor = strictRemoteLevelsFromZattrs(store);
        opened.physicalLevelZeroTransformIsIdentity =
            descriptor.levelZeroTransformIsIdentity;
        if (descriptor.advertised) {
            for (std::size_t level = 0; level < descriptor.keys.size(); ++level) {
                validateArrayAxes(store, descriptor.keys[level]);
                addRemoteLevelFromKey(opened, store, descriptor.keys[level],
                                      static_cast<int>(level));
            }
        } else {
            bool sawGap = false;
            for (int physicalLevel = 0; physicalLevel < 32; ++physicalLevel) {
                const auto key = std::to_string(physicalLevel);
                const bool present = store->exists(key + "/.zarray") ||
                                     store->exists(key + "/zarr.json");
                if (!present) {
                    sawGap = true;
                    continue;
                }
                if (sawGap) {
                    throw std::runtime_error(
                        "VC pyramid contains a numeric group above a missing intermediate group");
                }
                validateArrayAxes(store, key);
                addRemoteLevelFromKey(opened, store, key, physicalLevel);
            }
        }
        return validateAndRebaseVcPyramid(std::move(opened), baseScaleLevel);
    }

    const int firstPhysicalLevel = 0;

    const auto zattrsLevelKeys = remoteLevelKeysFromZattrs(store, firstPhysicalLevel);
    if (!zattrsLevelKeys.empty()) {
        for (const auto& [physicalLevel, key] : zattrsLevelKeys) {
            addRemoteLevelFromKey(opened, store, key, physicalLevel);
        }
        return opened;
    }

    for (int physicalLevel = firstPhysicalLevel; physicalLevel < 32; ++physicalLevel) {
        const auto key = std::to_string(physicalLevel);
        try {
            addRemoteLevelFromKey(opened, store, key, physicalLevel);
        } catch (const HttpStatusError& e) {
            if (e.status() == 404 ||
                (e.status() == 403 && (!opened.fetchers.empty() || firstPhysicalLevel == 0)))
                break;
            throw;
        } catch (const std::exception&) {
            if (physicalLevel == firstPhysicalLevel)
                throw;
            break;
        }
    }
    if (opened.fetchers.empty() && firstPhysicalLevel == 0) {
        auto array = utils::ZarrArray::open(store, "", vc::buildZarrCodecRegistry(1));
        if (array.metadata().dtype == utils::ZarrDtype::uint16)
            array = utils::ZarrArray::open(store, "", vc::buildZarrCodecRegistry(2));
        addPhysicalLevel(opened, 0, std::move(array));
    }
    return opened;
}

OpenedChunkedZarr openHttpZarrPyramid(const std::string& url)
{
    return openHttpZarrPyramid(url, vc::HttpAuth{}, std::nullopt);
}

std::unique_ptr<ChunkCache> createChunkCache(
    OpenedChunkedZarr opened,
    std::size_t decodedByteCapacity,
    std::size_t maxConcurrentReads)
{
    std::vector<ChunkCache::LevelInfo> levels;
    levels.reserve(opened.shapes.size());
    for (std::size_t i = 0; i < opened.shapes.size(); ++i) {
        levels.push_back({opened.shapes[i], opened.chunkShapes[i], opened.transforms[i]});
    }

    ChunkCache::Options options;
    options.decodedByteCapacity = decodedByteCapacity;
    options.maxConcurrentReads = maxConcurrentReads;
    return std::make_unique<ChunkCache>(
        std::move(levels),
        std::move(opened.fetchers),
        opened.fillValue,
        opened.dtype,
        std::move(options));
}

} // namespace vc::render
