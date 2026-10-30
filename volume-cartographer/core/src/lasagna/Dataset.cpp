#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/PredictionPooling.hpp"

#include "vc/core/util/RemoteFileCache.hpp"
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

namespace vc::lasagna {
namespace {

[[nodiscard]] bool startsWithNoCase(std::string_view value, std::string_view prefix)
{
    if (value.size() < prefix.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(value[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool isRemoteLocation(std::string_view value)
{
    return startsWithNoCase(value, "http://") ||
           startsWithNoCase(value, "sftp://") ||
           startsWithNoCase(value, "ssh://") ||
           startsWithNoCase(value, "https://") ||
           startsWithNoCase(value, "s3://") ||
           (startsWithNoCase(value, "s3+") &&
            value.find("://") != std::string_view::npos);
}

[[nodiscard]] bool isAbsoluteLocalPathString(const std::string& value)
{
    return std::filesystem::path(value).is_absolute();
}

[[nodiscard]] vc::ResolvedUrl resolveRemoteEndpoint(const std::string& location)
{
    auto resolved = vc::resolveRemoteUrl(location);
    resolved.httpsUrl = vc::core::util::normalizeRemoteFileLocation(location);
    return resolved;
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
                      const vc::HttpAuth& remoteAuth,
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
    const vc::HttpAuth remoteAuth = discoverAwsCredentials
        ? options.remoteAuth
        : vc::HttpAuth{};

    for (auto& group : manifest.groups) {
        group.sourceLocation.clear();
        group.zarrPath.clear();
        group.remoteCacheRoot.clear();
        group.discoverAwsCredentials = true;

        if (isRemoteLocation(group.relativeZarrKey)) {
            group.sourceLocation = vc::core::util::remoteFileCacheSource(
                group.relativeZarrKey);
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
            group.sourceLocation = vc::joinRemoteUrlPath(
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
    manifest.remoteBaseUrl = resolveRemoteEndpoint(url).httpsUrl;
    manifest.remoteSourceBaseLocation =
        vc::core::util::remoteFileCacheSource(url);
    manifest.remoteCacheRoot = manifest.baseDirectory;
    manifest.discoverAwsCredentials =
        !marker.value("anonymous", false);
}

} // namespace

LasagnaDataset::LasagnaDataset(LasagnaDatasetManifest manifest)
    : manifest_(std::move(manifest))
{
}

bool isRemoteLasagnaLocation(std::string_view location)
{
    return isRemoteLocation(location);
}

MaterializedLasagnaManifest materializeLasagnaManifest(const std::string& manifestLocation, const LasagnaDatasetOpenOptions& options)
{
    if (!isRemoteLocation(manifestLocation)) {
        return {
            std::filesystem::absolute(std::filesystem::path(manifestLocation)).lexically_normal(),
            manifestLocation,
            true,
        };
    }
    if (options.remoteCacheRoot.empty()) {
        throw std::runtime_error("Remote Lasagna manifest requires --remote-cache-dir: " + vc::core::util::redactedRemoteLocation(manifestLocation));
    }

    const auto root = std::filesystem::absolute(options.remoteCacheRoot).lexically_normal();
    const auto manifestPath = root / vc::core::util::remoteFileCachePath(
        manifestLocation);

    vc::core::util::RemoteFileCacheOptions cacheOptions;
    cacheOptions.cacheRoot = root;
    cacheOptions.destination = manifestPath.lexically_relative(root);
    cacheOptions.policy = options.cachePolicy;
    cacheOptions.auth = options.discoverAwsCredentials
        ? options.remoteAuth
        : vc::HttpAuth{};
    cacheOptions.discoverAwsCredentials = options.discoverAwsCredentials;
    cacheOptions.fetcher = options.remoteFileFetcher;
    const auto cached = vc::core::util::cacheRemoteFile(manifestLocation, cacheOptions);
    std::clog << "Lasagna manifest cache "
              << (cached.cacheHit ? "hit: " : "download: ")
              << vc::core::util::redactedRemoteLocation(cached.normalizedEndpoint)
              << '\n';
    return {cached.path, cached.normalizedEndpoint, cached.cacheHit};
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
    if (!isRemoteLocation(manifestLocation)) {
        return open(std::filesystem::path(manifestLocation), std::move(options));
    }
    auto materialized = materializeLasagnaManifest(manifestLocation, options);
    auto openMaterialized = [&](const MaterializedLasagnaManifest& source,
                                const LasagnaDatasetOpenOptions& openOptions) {
        auto manifest = LasagnaDatasetManifest::parseFile(source.path);
        manifest.manifestLocation =
            vc::core::util::remoteFileCacheSource(manifestLocation);
        manifest.manifestIsRemote = true;
        manifest.workingToBaseScale = openOptions.workingToBaseScale;
        manifest.remoteBaseUrl = remoteParentUrl(source.normalizedLocation);
        manifest.remoteSourceBaseLocation = remoteParentUrl(
            vc::core::util::remoteFileCacheSource(manifestLocation));
        manifest.remoteCacheRoot = source.path.parent_path();
        resolveGroupLocations(manifest, openOptions);
        return LasagnaDataset(std::move(manifest));
    };
    try {
        return openMaterialized(materialized, options);
    } catch (...) {
        if (!materialized.cacheHit ||
            options.cachePolicy == vc::core::util::RemoteFileCachePolicy::Refresh) {
            throw;
        }
    }

    // A size-valid cached object may still contain invalid JSON. Invalidate and
    // refetch once; explicit sidecar-backed local manifests are never touched.
    const auto root = std::filesystem::absolute(options.remoteCacheRoot)
                          .lexically_normal();
    vc::core::util::RemoteFileCacheOptions cacheOptions;
    cacheOptions.cacheRoot = root;
    cacheOptions.destination = materialized.path.lexically_relative(root);
    vc::core::util::invalidateRemoteFileCacheEntry(cacheOptions);
    auto refreshOptions = options;
    refreshOptions.cachePolicy = vc::core::util::RemoteFileCachePolicy::Refresh;
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
    const auto& keys = group.remoteAuth;
    // Explicit keys sign s3:// requests; otherwise public buckets are read
    // anonymously and others with discovered AWS credentials when allowed.
    options.access = keys.access_key.empty() || !location.starts_with("s3")
        ? vcr::zarr::resolveAccess(location, group.discoverAwsCredentials
              ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous)
        : vcr::zarr::s3Access(location,
              vcr::Auth{"s3", keys.access_key, keys.secret_key, keys.session_token, keys.region});
    options.cacheDirs = {group.remoteCacheRoot.string()};
    return vcr::zarr::openArray(location, options);
}

} // namespace vc::lasagna
