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
    if (meta.dtype != utils::ZarrDtype::uint8)
        throw std::runtime_error("streaming zarr fetcher currently supports uint8 only");

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
    opened.fetchers.push_back(std::make_shared<ZarrChunkFetcher>(std::move(array)));
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

} // namespace

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root)
{
    OpenedChunkedZarr opened;
    for (int level : localLevelNumbers(root)) {
        addLevel(opened, utils::ZarrArray::open(root / std::to_string(level),
                                                vc::buildZarrCodecRegistry(1)));
    }
    if (opened.fetchers.empty()) {
        addLevel(opened, utils::ZarrArray::open(root, vc::buildZarrCodecRegistry(1)));
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
    for (int physicalLevel = firstPhysicalLevel; physicalLevel < 32; ++physicalLevel) {
        const auto key = std::to_string(physicalLevel);
        try {
            addLevel(opened, utils::ZarrArray::open(store, key, vc::buildZarrCodecRegistry(1)));
        } catch (const HttpStatusError& e) {
            if (e.status() == 404)
                break;
            throw;
        } catch (const std::exception&) {
            if (physicalLevel == firstPhysicalLevel)
                throw;
            break;
        }
    }
    if (opened.fetchers.empty() && firstPhysicalLevel == 0)
        addLevel(opened, utils::ZarrArray::open(store, "", vc::buildZarrCodecRegistry(1)));
    return opened;
}

OpenedChunkedZarr openHttpZarrPyramid(const std::string& url)
{
    return openHttpZarrPyramid(url, vc::HttpAuth{}, 0);
}

} // namespace vc::render
