#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/PredictionPooling.hpp"

#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <span>

#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "vc_reader/sftp.hpp"
#include "vc_reader/zarr.hpp"

namespace vc::lasagna {
namespace {

[[nodiscard]] bool isAbsoluteLocalPathString(const std::string& value)
{
    return std::filesystem::path(value).is_absolute();
}

// A location's persistent, credential-free name: no query or fragment, no slash ending it.
[[nodiscard]] std::string sourceName(std::string_view location)
{
    auto source = std::string(location.substr(0, location.find_first_of("?#")));
    if (vcr::sftp::isSftpUrl(source))
        source = vcr::sftp::parseUrl(source).str();
    while (!source.empty() && source.back() == '/')
        source.pop_back();
    return source;
}

[[nodiscard]] std::string redacted(std::string_view location)
{
    const auto query = location.find('?');
    return query == std::string_view::npos ? std::string(location) : std::string(location.substr(0, query)) + "?<redacted>";
}

// Keys sign s3:// requests; otherwise public buckets are read anonymously and others
// with discovered AWS credentials when allowed.
[[nodiscard]] vcr::zarr::Access lasagnaAccess(const std::string& location, const vcr::Auth& keys, bool discover)
{
    if (!keys.kind.empty())
        if (const auto s3 = vc::s3Location(location); s3 && !vcr::zarr::isPublicLocation(*s3))
            return vcr::zarr::s3Access(*s3, keys);
    return vcr::zarr::resolveAccess(location, discover ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous);
}

void fetchManifest(const std::string& location, const std::filesystem::path& path,
                   const std::function<std::optional<std::string>(const std::string&, const std::string&)>& read)
{
    const auto source = sourceName(location);
    const auto slash = source.rfind('/');
    if (slash == std::string::npos)
        throw std::runtime_error("Remote Lasagna manifest has no object path: " + redacted(location));
    const auto text = read(source.substr(0, slash), source.substr(slash + 1));
    if (!text)
        throw std::runtime_error("Remote Lasagna manifest does not exist: " + redacted(location));
    std::filesystem::create_directories(path.parent_path());
    vcr::zarr::atomicWrite(path, *text);
}

[[nodiscard]] std::string remoteParentUrl(const std::string& normalizedRemoteUrl)
{
    const auto query = normalizedRemoteUrl.find('?');
    const std::string path = query == std::string::npos
        ? normalizedRemoteUrl
        : normalizedRemoteUrl.substr(0, query);
    const std::string suffix = query == std::string::npos
        ? std::string{}
        : normalizedRemoteUrl.substr(query);

    const auto schemeEnd = path.find("://");
    const auto authorityStart = schemeEnd == std::string::npos
        ? std::string::npos
        : schemeEnd + 3;
    const auto slash = path.rfind('/');
    if (slash == std::string::npos ||
        (authorityStart != std::string::npos && slash < authorityStart)) {
        return path + suffix;
    }
    return path.substr(0, slash) + suffix;
}

[[nodiscard]] std::string normalizedRelativeRemoteKey(const std::string& rawPath)
{
    const std::filesystem::path path(rawPath);
    const auto normalized = path.lexically_normal();
    if (rawPath.empty() || path.is_absolute() || normalized.empty() ||
        *normalized.begin() == "..") {
        throw std::runtime_error(
            "Remote Lasagna group path must remain inside the artifact: " +
            rawPath);
    }
    return normalized.generic_string();
}

void applyRemoteGroup(LasagnaChannelGroup& group,
                      std::filesystem::path remoteCacheRoot,
                      const vcr::Auth& remoteAuth,
                      bool discoverAwsCredentials)
{
    if (remoteCacheRoot.empty()) {
        throw std::runtime_error("Remote Lasagna group requires a remote cache directory: " + group.relativeZarrKey);
    }
    group.remoteCacheRoot = std::move(remoteCacheRoot);
    group.remoteAuth = remoteAuth;
    group.discoverAwsCredentials = discoverAwsCredentials;
}

void resolveGroupLocations(
    LasagnaDatasetManifest& manifest,
    const LasagnaDatasetOpenOptions& options)
{
    const std::filesystem::path optionRemoteCacheRoot =
        options.remoteCacheRoot.empty()
            ? std::filesystem::path{}
            : std::filesystem::absolute(options.remoteCacheRoot).lexically_normal();
    // vc_reader keys its cache by URL (<root>/<scheme>/<bucket>/<path>), so
    // the caller's cache root is shared by every dataset and tool.
    const std::filesystem::path defaultRemoteCacheRoot =
        optionRemoteCacheRoot.empty()
            ? manifest.remoteCacheRoot
            : optionRemoteCacheRoot;
    const bool discoverAwsCredentials =
        manifest.discoverAwsCredentials && options.discoverAwsCredentials;
    const vcr::Auth remoteAuth = discoverAwsCredentials
        ? options.remoteAuth
        : vcr::Auth{};

    for (auto& group : manifest.groups) {
        group.sourceLocation.clear();
        group.zarrPath.clear();
        group.remoteCacheRoot.clear();
        group.discoverAwsCredentials = true;

        if (vcr::zarr::isRemote(group.relativeZarrKey)) {
            group.sourceLocation = sourceName(group.relativeZarrKey);
            applyRemoteGroup(group, defaultRemoteCacheRoot,
                             remoteAuth, discoverAwsCredentials);
            continue;
        }

        if (isAbsoluteLocalPathString(group.relativeZarrKey)) {
            group.zarrPath = std::filesystem::absolute(
                std::filesystem::path(group.relativeZarrKey)).lexically_normal();
            group.sourceLocation = group.zarrPath.string();
            continue;
        }

        if (!manifest.remoteBaseUrl.empty()) {
            const auto key = normalizedRelativeRemoteKey(group.relativeZarrKey);
            group.sourceLocation = vcr::joinLocation(
                manifest.remoteSourceBaseLocation.empty()
                    ? manifest.remoteBaseUrl
                    : manifest.remoteSourceBaseLocation,
                key);
            applyRemoteGroup(group, defaultRemoteCacheRoot, remoteAuth,
                             discoverAwsCredentials);
            continue;
        }

        group.zarrPath = std::filesystem::absolute(
            manifest.baseDirectory / group.relativeZarrKey).lexically_normal();
        group.sourceLocation = group.zarrPath.string();
    }
}

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
    manifest.remoteBaseUrl = vc::normalizedRemoteUrl(url);
    manifest.remoteSourceBaseLocation = sourceName(url);
    // Its arrays cache under the shared root, not beside the manifest.
    manifest.remoteCacheRoot = vc::settings::remoteCachePath();
    manifest.discoverAwsCredentials =
        !marker.value("anonymous", false);
}

} // namespace

LasagnaDataset::LasagnaDataset(LasagnaDatasetManifest manifest)
    : manifest_(std::move(manifest))
{
}

void downloadLasagnaManifest(const std::string& location, const std::filesystem::path& path)
{
    fetchManifest(location, path, vcr::zarr::metadataReader(vcr::zarr::resolveAccess(location, vcr::zarr::AuthMode::Anonymous)));
}

MaterializedLasagnaManifest materializeLasagnaManifest(const std::string& manifestLocation, const LasagnaDatasetOpenOptions& options)
{
    if (!vcr::zarr::isRemote(manifestLocation)) {
        return {
            std::filesystem::absolute(std::filesystem::path(manifestLocation)).lexically_normal(),
            manifestLocation,
            true,
        };
    }
    if (options.remoteCacheRoot.empty()) {
        throw std::runtime_error("Remote Lasagna manifest requires --remote-cache-dir: " + redacted(manifestLocation));
    }

    const auto root = std::filesystem::absolute(options.remoteCacheRoot).lexically_normal();
    const auto manifestPath = vcr::zarr::decodedCachePath(root, sourceName(manifestLocation));
    const auto normalized = vc::normalizedRemoteUrl(manifestLocation);
    const bool hit = !options.refreshManifest && std::filesystem::is_regular_file(manifestPath);
    if (!hit) {
        const auto keys = options.discoverAwsCredentials ? options.remoteAuth : vcr::Auth{};
        fetchManifest(manifestLocation, manifestPath, options.manifestReader ? options.manifestReader
            : vcr::zarr::metadataReader(lasagnaAccess(sourceName(manifestLocation), keys, options.discoverAwsCredentials)));
    }
    std::clog << "Lasagna manifest cache " << (hit ? "hit: " : "download: ") << redacted(normalized) << '\n';
    return {manifestPath, normalized, hit};
}

namespace {

void validateOpenOptions(const LasagnaDatasetOpenOptions& options)
{
    if (!(options.workingToBaseScale > 0.0) ||
        !std::isfinite(options.workingToBaseScale)) {
        throw std::runtime_error("Lasagna working-to-base scale must be positive");
    }
}

} // namespace

LasagnaDataset LasagnaDataset::open(const std::filesystem::path& manifestPath,
                                    LasagnaDatasetOpenOptions options)
{
    validateOpenOptions(options);
    auto manifest = LasagnaDatasetManifest::parseFile(manifestPath);
    manifest.manifestLocation = manifest.manifestPath.string();
    manifest.manifestIsRemote = false;
    manifest.workingToBaseScale = options.workingToBaseScale;
    loadRemoteMarker(manifest);
    resolveGroupLocations(manifest, options);
    return LasagnaDataset(std::move(manifest));
}

LasagnaDataset LasagnaDataset::openLocation(
    const std::string& manifestLocation,
    LasagnaDatasetOpenOptions options)
{
    validateOpenOptions(options);
    if (!vcr::zarr::isRemote(manifestLocation)) {
        return open(std::filesystem::path(manifestLocation), std::move(options));
    }
    auto materialized = materializeLasagnaManifest(manifestLocation, options);
    auto openMaterialized = [&](const MaterializedLasagnaManifest& source,
                                const LasagnaDatasetOpenOptions& openOptions) {
        auto manifest = LasagnaDatasetManifest::parseFile(source.path);
        manifest.manifestLocation = sourceName(manifestLocation);
        manifest.manifestIsRemote = true;
        manifest.workingToBaseScale = openOptions.workingToBaseScale;
        manifest.remoteBaseUrl = remoteParentUrl(source.normalizedLocation);
        manifest.remoteSourceBaseLocation = remoteParentUrl(sourceName(manifestLocation));
        manifest.remoteCacheRoot = source.path.parent_path();
        resolveGroupLocations(manifest, openOptions);
        return LasagnaDataset(std::move(manifest));
    };
    try {
        return openMaterialized(materialized, options);
    } catch (...) {
        if (!materialized.cacheHit)
            throw;
    }

    // A cached object may still contain invalid JSON. Download it again, once;
    // explicit sidecar-backed local manifests are never touched.
    auto refreshOptions = options;
    refreshOptions.refreshManifest = true;
    materialized = materializeLasagnaManifest(manifestLocation, refreshOptions);
    return openMaterialized(materialized, refreshOptions);
}

std::string lasagnaGroupSourceLocation(const LasagnaChannelGroup& group)
{
    if (!group.sourceLocation.empty())
        return group.sourceLocation;
    if (!group.zarrPath.empty())
        return std::filesystem::absolute(group.zarrPath).lexically_normal().string();
    throw std::runtime_error(
        "Lasagna group has no resolved source location: " + group.name);
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

vcr::zarr::OpenedArray openLasagnaGroupArray(
    const LasagnaChannelGroup& group, std::shared_ptr<vcr::MemoryCache> cache)
{
    if (!group.isRemote() && std::filesystem::exists(group.zarrPath / ".native_pool.json"))
        return openPooledPredictionArray(group.zarrPath, std::move(cache));
    vcr::zarr::OpenOptions options;
    options.memoryCache = std::move(cache);
    if (!group.isRemote())
        return vcr::zarr::openArray(lasagnaGroupSourceLocation(group), options);
    const auto& location = group.sourceLocation;
    options.access = lasagnaAccess(location, group.remoteAuth, group.discoverAwsCredentials);
    options.cacheDirs = {group.remoteCacheRoot.string()};
    return vcr::zarr::openArray(location, options);
}

} // namespace vc::lasagna
