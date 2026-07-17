#include "vc/lasagna/Dataset.hpp"

#include "vc/core/types/Volume.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <span>

#include <stdexcept>

namespace vc::lasagna {
namespace {

class PersistentHttpStore final : public utils::Store {
public:
    PersistentHttpStore(std::string baseUrl, std::filesystem::path cacheRoot)
        : remote_(std::move(baseUrl)), cacheRoot_(std::move(cacheRoot)) {}

    bool exists(const std::string& key) const override
    {
        return get_if_exists(key).has_value();
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
        const auto path = safePath(key);
        {
            std::lock_guard<std::mutex> lock(ioMutex_);
            if (std::filesystem::is_regular_file(path))
                return read(path);
        }
        auto bytes = remote_.get_if_exists(key);
        if (!bytes) return std::nullopt;
        {
            std::lock_guard<std::mutex> lock(ioMutex_);
            if (std::filesystem::is_regular_file(path))
                return read(path);
            publish(path, *bytes);
        }
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
    std::filesystem::path safePath(const std::string& key) const
    {
        const std::filesystem::path relative(key);
        if (relative.empty() || relative.is_absolute())
            throw std::runtime_error("Invalid remote Lasagna Zarr key: " + key);
        const auto normalized = relative.lexically_normal();
        if (normalized.empty() || *normalized.begin() == "..")
            throw std::runtime_error("Remote Lasagna Zarr key escapes cache root: " + key);
        return cacheRoot_ / normalized;
    }

    static std::vector<std::byte> read(const std::filesystem::path& path)
    {
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
        return out;
    }

    static void publish(const std::filesystem::path& path,
                        std::span<const std::byte> bytes)
    {
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
    }

    utils::HttpStore remote_;
    std::filesystem::path cacheRoot_;
    static std::mutex ioMutex_;
};

std::mutex PersistentHttpStore::ioMutex_;

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
