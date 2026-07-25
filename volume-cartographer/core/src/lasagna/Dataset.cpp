#include "vc/lasagna/Dataset.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <span>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vc::lasagna {
namespace {

// Lasagna values drive geometry and must never pass through the configurable
// lossy volume-cache encoder. This store is deliberately an object-for-object
// read-through cache: publish() persists the exact bytes returned by the Zarr
// origin, whose own compressor remains responsible for lossless decoding.
class PersistentHttpStore final : public utils::Store {
public:
    PersistentHttpStore(std::string baseUrl, std::filesystem::path cacheRoot)
        : remote_(baseUrl), baseUrl_(std::move(baseUrl)), cacheRoot_(std::move(cacheRoot)),
          budget_(vc::render::PersistentZarrCacheBudget::findForPath(cacheRoot_)) {}

    bool exists(const std::string& key) const override
    {
        const auto relative = checkedRelativePath(key);
        const auto filename = relative.filename().string();
        if (filename == "zarr.json" &&
            metadataExists(relative.parent_path() / ".zarray")) {
            return false;
        }
        if (filename == ".zarray" &&
            metadataExists(relative.parent_path() / "zarr.json")) {
            return false;
        }
        const auto path = cachePath(relative);
        if (std::filesystem::is_regular_file(path))
            return true;
        if (isMetadataPath(relative) &&
            std::filesystem::is_regular_file(cacheRoot_ / relative)) {
            return true;
        }
        return remote_.exists(key);
    }

    std::vector<std::byte> get(const std::string& key) const override
    {
        auto bytes = get_if_exists(key);
        if (!bytes)
            throw std::runtime_error("Remote Lasagna Zarr key not found: " + key);
        return std::move(*bytes);
    }

    std::optional<std::vector<std::byte>> get_if_exists(const std::string& key) const override
    {
        const auto relative = checkedRelativePath(key);
        const auto path = cachePath(relative);
        if (auto bytes = readIfExists(path, !isMetadataPath(relative)))
            return bytes;
        if (isMetadataPath(relative)) {
            const auto legacyPath = cacheRoot_ / relative;
            if (auto bytes = readIfExists(legacyPath, false)) {
                publish(path, *bytes, false);
                return bytes;
            }
        }

        const std::string artifactKey =
            baseUrl_ + '\n' + cacheRoot_.lexically_normal().string();
        const std::string requestKey = artifactKey + '\n' + key;
        std::shared_ptr<InFlightRequest> request;
        bool ownsRequest = false;
        bool announceStreaming = false;
        {
            std::lock_guard<std::mutex> lock(inFlightMutex_);
            if (auto bytes = readIfExists(path, !isMetadataPath(relative)))
                return bytes;
            if (auto it = inFlight_.find(requestKey); it != inFlight_.end()) {
                request = it->second;
            } else {
                request = std::make_shared<InFlightRequest>();
                inFlight_.emplace(requestKey, request);
                ownsRequest = true;
                announceStreaming = announcedArtifacts_.insert(artifactKey).second;
            }
        }

        if (announceStreaming) {
            std::clog << "[lasagna] streaming uncached data into "
                      << cacheRoot_.string() << std::endl;
        }

        if (!ownsRequest) {
            std::unique_lock<std::mutex> lock(inFlightMutex_);
            request->finished.wait(lock, [&]() { return request->done; });
            const auto error = request->error;
            const bool found = request->found;
            auto sharedBytes = request->bytes;
            lock.unlock();
            if (error)
                std::rethrow_exception(error);
            return found ? std::move(sharedBytes) : std::nullopt;
        }

        std::optional<std::vector<std::byte>> bytes;
        std::exception_ptr error;
        try {
            bytes = remote_.get_if_exists(key);
            if (bytes)
                // Preserve the source object byte-for-byte. Do not decode and
                // re-encode it through the remote volume cache.
                publish(path, *bytes, !isMetadataPath(relative));
        } catch (...) {
            error = std::current_exception();
        }

        size_t cachedCount = 0;
        {
            std::lock_guard<std::mutex> lock(inFlightMutex_);
            request->found = bytes.has_value();
            request->bytes = bytes;
            request->error = error;
            request->done = true;
            inFlight_.erase(requestKey);
            if (bytes)
                cachedCount = ++cachedObjectCounts_[artifactKey];
        }
        request->finished.notify_all();
        if (cachedCount != 0 && cachedCount % 64 == 0) {
            std::clog << "[lasagna] cached " << cachedCount
                      << " remote objects in " << cacheRoot_.string() << std::endl;
        }
        if (error)
            std::rethrow_exception(error);
        return bytes;
    }

    std::optional<std::vector<std::byte>> get_partial(
        const std::string& key, std::size_t offset, std::size_t length) const override
    {
        auto bytes = get_if_exists(key);
        if (!bytes || offset > bytes->size())
            return std::nullopt;
        const auto count = std::min(length, bytes->size() - offset);
        return std::vector<std::byte>(bytes->begin() + static_cast<std::ptrdiff_t>(offset),
                                      bytes->begin() + static_cast<std::ptrdiff_t>(offset + count));
    }

    void set(const std::string&, std::span<const std::byte>) override
    {
        throw std::runtime_error("Remote Lasagna cache store is read-only");
    }
    void erase(const std::string&) override
    {
        throw std::runtime_error("Remote Lasagna cache store is read-only");
    }

private:
    struct InFlightRequest {
        std::condition_variable finished;
        bool done = false;
        bool found = false;
        std::optional<std::vector<std::byte>> bytes;
        std::exception_ptr error;
    };

    [[nodiscard]] static bool isMetadataPath(const std::filesystem::path& relative)
    {
        const auto filename = relative.filename().string();
        return filename == ".zarray" || filename == ".zattrs" ||
               filename == ".zgroup" || filename == "zarr.json";
    }

    [[nodiscard]] std::filesystem::path checkedRelativePath(const std::string& key) const
    {
        const std::filesystem::path relative(key);
        if (relative.empty() || relative.is_absolute())
            throw std::runtime_error("Invalid remote Lasagna Zarr key: " + key);
        const auto normalized = relative.lexically_normal();
        if (normalized.empty() || *normalized.begin() == "..")
            throw std::runtime_error("Remote Lasagna Zarr key escapes cache root: " + key);
        return normalized;
    }

    [[nodiscard]] std::filesystem::path metadataPath(
        const std::filesystem::path& relative) const
    {
        return cacheRoot_ / ".lasagna-zarr-metadata" / relative;
    }

    [[nodiscard]] std::filesystem::path cachePath(
        const std::filesystem::path& relative) const
    {
        return isMetadataPath(relative) ? metadataPath(relative)
                                        : cacheRoot_ / relative;
    }

    [[nodiscard]] bool metadataExists(const std::filesystem::path& relative) const
    {
        return std::filesystem::is_regular_file(metadataPath(relative)) ||
               std::filesystem::is_regular_file(cacheRoot_ / relative);
    }

    std::optional<std::vector<std::byte>> readIfExists(
        const std::filesystem::path& path, bool managed) const
    {
        auto pin = managed && budget_
            ? budget_->pinRead(path)
            : vc::render::PersistentZarrCacheBudget::ReadPin{};
        if (!std::filesystem::is_regular_file(path))
            return std::nullopt;
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to read cached Lasagna object: " + path.string());
        in.seekg(0, std::ios::end);
        const auto size = in.tellg();
        in.seekg(0);
        std::vector<std::byte> out(static_cast<std::size_t>(size));
        if (!out.empty())
            in.read(reinterpret_cast<char*>(out.data()),
                    static_cast<std::streamsize>(size));
        pin.complete(true);
        return out;
    }

    bool publish(const std::filesystem::path& path,
                 std::span<const std::byte> bytes,
                 bool managed) const
    {
        auto reservation = managed && budget_
            ? budget_->reserveWrite(path, bytes.size())
            : vc::render::PersistentZarrCacheBudget::WriteReservation{};
        if (managed && budget_ && !reservation)
            return false;
        static std::atomic<std::uint64_t> serial{0};
        std::filesystem::create_directories(path.parent_path());
        const auto tmp = std::filesystem::path(
            path.string() + ".tmp-" + std::to_string(serial.fetch_add(1)));
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error("Failed to create Lasagna cache file: " + tmp.string());
            if (!bytes.empty())
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
            out.close();
            if (!out)
                throw std::runtime_error("Failed to write Lasagna cache file: " + tmp.string());
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec && !std::filesystem::is_regular_file(path)) {
            std::filesystem::remove(tmp);
            throw std::runtime_error("Failed to publish Lasagna cache file: " + ec.message());
        }
        std::filesystem::remove(tmp, ec);
        reservation.commit();
        return true;
    }

    utils::HttpStore remote_;
    std::string baseUrl_;
    std::filesystem::path cacheRoot_;
    std::shared_ptr<vc::render::PersistentZarrCacheBudget> budget_;
    static std::mutex inFlightMutex_;
    static std::unordered_map<std::string, std::shared_ptr<InFlightRequest>> inFlight_;
    static std::unordered_set<std::string> announcedArtifacts_;
    static std::unordered_map<std::string, size_t> cachedObjectCounts_;
};

std::mutex PersistentHttpStore::inFlightMutex_;
std::unordered_map<std::string, std::shared_ptr<PersistentHttpStore::InFlightRequest>>
    PersistentHttpStore::inFlight_;
std::unordered_set<std::string> PersistentHttpStore::announcedArtifacts_;
std::unordered_map<std::string, size_t> PersistentHttpStore::cachedObjectCounts_;

void loadRemoteMarker(LasagnaDatasetManifest& manifest)
{
    const auto markerPath = manifest.baseDirectory / kLasagnaRemoteMarker;
    if (!std::filesystem::is_regular_file(markerPath))
        return;
    const auto marker = nlohmann::json::parse(std::ifstream(markerPath));
    const auto url = marker.value("artifact_url", std::string{});
    if (url.empty())
        throw std::runtime_error("Lasagna remote marker has no artifact_url");
    const auto manifestFile = marker.value("manifest_file", std::string{});
    if (manifestFile.empty() ||
        std::filesystem::absolute(manifest.baseDirectory / manifestFile).lexically_normal() !=
            std::filesystem::absolute(manifest.manifestPath).lexically_normal()) {
        throw std::runtime_error(
            "Lasagna remote marker does not identify the opened manifest");
    }
    manifest.remoteBaseUrl = url;
    while (!manifest.remoteBaseUrl.empty() && manifest.remoteBaseUrl.back() == '/')
        manifest.remoteBaseUrl.pop_back();
    manifest.remoteCacheRoot = manifest.baseDirectory;
    for (const auto& group : manifest.groups) {
        const std::filesystem::path key(group.relativeZarrKey);
        const auto normalized = key.lexically_normal();
        if (key.empty() || key.is_absolute() || normalized.empty() ||
            *normalized.begin() == "..") {
            throw std::runtime_error(
                "Remote Lasagna group path must remain inside the artifact: " +
                group.relativeZarrKey);
        }
    }
}

} // namespace

LasagnaDataset::LasagnaDataset(LasagnaDatasetManifest manifest)
    : manifest_(std::move(manifest))
{
}

LasagnaDataset LasagnaDataset::open(const std::filesystem::path& manifestPath,
                                    LasagnaDatasetOpenOptions options)
{
    if (!(options.workingToBaseScale > 0.0) ||
        !std::isfinite(options.workingToBaseScale)) {
        throw std::runtime_error("Lasagna working-to-base scale must be positive");
    }
    auto manifest = LasagnaDatasetManifest::parseFile(manifestPath);
    manifest.workingToBaseScale = options.workingToBaseScale;
    loadRemoteMarker(manifest);
    return LasagnaDataset(std::move(manifest));
}

const LasagnaDatasetManifest& LasagnaDataset::manifest() const noexcept
{
    return manifest_;
}

bool LasagnaDataset::hasNormalSource() const noexcept
{
    return manifest_.hasNormalSource();
}

const std::filesystem::path& LasagnaDataset::normalSourcePath() const
{
    if (!manifest_.normalPath.has_value()) {
        throw std::runtime_error("Lasagna dataset manifest has no normal source path");
    }
    return *manifest_.normalPath;
}

utils::ZarrArray openLasagnaChannelArray(
    const LasagnaDatasetManifest& manifest,
    const LasagnaChannelGroup& group,
    std::size_t dtypeSize)
{
    auto registry = vc::buildZarrCodecRegistry(dtypeSize);
    if (manifest.remoteBaseUrl.empty())
        return utils::ZarrArray::open(group.zarrPath, std::move(registry));
    auto store = std::make_shared<PersistentHttpStore>(
        manifest.remoteBaseUrl, manifest.remoteCacheRoot);
    return utils::ZarrArray::open(
        std::move(store), group.relativeZarrKey, std::move(registry));
}

} // namespace vc::lasagna
