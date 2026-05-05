#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <filesystem>
#include <memory>
#include <optional>
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
        return baseUrl_ + "/" + key;
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
    }

    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ChunkFetchResult result;
        const std::array<std::size_t, 3> indices{
            static_cast<std::size_t>(key.iz),
            static_cast<std::size_t>(key.iy),
            static_cast<std::size_t>(key.ix)};

        try {
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

private:
    std::unique_ptr<utils::ZarrArray> array_;
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
    const int logicalLevel = static_cast<int>(opened.transforms.size());
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << logicalLevel);
    IChunkedArray::LevelTransform transform;
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    opened.transforms.push_back(transform);
    opened.fillValue = meta.fill_value.value_or(0.0);
    opened.dtype = dtype;
    opened.fetchers.push_back(std::make_shared<ZarrChunkFetcher>(std::move(array)));
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
        opened.fetchers.resize(index + 1);
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
    IChunkedArray::LevelTransform transform;
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << physicalLevel);
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    opened.transforms[index] = transform;
    opened.fillValue = single.fillValue;
    opened.dtype = single.dtype;
    opened.fetchers[index] = std::move(single.fetchers[0]);
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
    int baseScaleLevel)
{
    auto store = std::make_shared<ClassifyingHttpStore>(url, auth);
    OpenedChunkedZarr opened;
    const int firstPhysicalLevel = std::max(0, baseScaleLevel);

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
    return openHttpZarrPyramid(url, vc::HttpAuth{}, 0);
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
