#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/PredictionPooling.hpp"

#include "vc/core/util/RemoteCacheSettings.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string_view>
#include <utility>
#include "vc_reader/sftp.hpp"
#include "vc_reader/source_reader.hpp"
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

// Public buckets are read anonymously, others with the environment's AWS credentials
// when allowed (vcr::zarr::resolveAccess).
[[nodiscard]] vcr::zarr::Access lasagnaAccess(const std::string& location, bool discover)
{
    return vcr::zarr::resolveAccess(location, discover ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous);
}

// A remote location with no slash ending its path; its query (a signed URL's token) kept.
[[nodiscard]] std::string runtimeLocation(std::string_view location)
{
    const auto query = location.find_first_of("?#");
    auto path = std::string(location.substr(0, query));
    while (path.size() > 1 && path.back() == '/')
        path.pop_back();
    return path + (query == std::string_view::npos ? std::string() : std::string(location.substr(query)));
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
                      bool discoverAwsCredentials)
{
    if (remoteCacheRoot.empty()) {
        throw std::runtime_error("Remote Lasagna group requires a remote cache directory: " + group.relativeZarrKey);
    }
    group.remoteCacheRoot = std::move(remoteCacheRoot);
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

    for (auto& group : manifest.groups) {
        group.sourceLocation.clear();
        group.zarrPath.clear();
        group.remoteCacheRoot.clear();
        group.discoverAwsCredentials = true;

        if (vcr::zarr::isRemote(group.relativeZarrKey)) {
            group.sourceLocation = runtimeLocation(group.relativeZarrKey);
            applyRemoteGroup(group, defaultRemoteCacheRoot, discoverAwsCredentials);
            continue;
        }

        if (isAbsoluteLocalPathString(group.relativeZarrKey)) {
            group.zarrPath = std::filesystem::absolute(
                std::filesystem::path(group.relativeZarrKey)).lexically_normal();
            group.sourceLocation = group.zarrPath.string();
            continue;
        }

        if (!manifest.remoteSourceBaseLocation.empty()) {
            const auto key = normalizedRelativeRemoteKey(group.relativeZarrKey);
            group.sourceLocation = vcr::joinLocation(manifest.remoteSourceBaseLocation, key);
            applyRemoteGroup(group, defaultRemoteCacheRoot, discoverAwsCredentials);
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
    manifest.remoteSourceBaseLocation = runtimeLocation(url);
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
    const auto identity = sourceName(manifestLocation);
    const auto slash = identity.rfind('/');
    if (slash == std::string::npos || slash + 1 == identity.size() || identity.find("://") + 3 > slash)
        throw std::runtime_error("Remote Lasagna manifest has no object path: " + redacted(manifestLocation));

    // vc_reader's file cache keeps the manifest at the location's place under the root, under
    // the root's byte limit and free-space floor (VC's settings). A cached copy is trusted;
    // refreshManifest reads it again. The request goes to the location as given, query included.
    const auto root = std::filesystem::absolute(options.remoteCacheRoot).lexically_normal();
    const auto location = runtimeLocation(manifestLocation);
    const auto manifestPath = vcr::zarr::decodedCachePath(root, location);
    if (options.refreshManifest) {
        std::error_code ignored;
        std::filesystem::remove(manifestPath, ignored);
    }
    const bool hit = std::filesystem::is_regular_file(manifestPath);
    const auto access = lasagnaAccess(identity, options.discoverAwsCredentials);
    vcr::sources::EncodedFiles files([root, access](const std::string&) {
        auto open = vc::settings::remoteOpenOptions(root);
        open.access = access;
        return open;
    });
    try {
        (void)files.open(location);
    } catch (const vcr::zarr::NotFound&) {
        throw std::runtime_error("Remote Lasagna manifest does not exist: " + redacted(location));
    }
    std::clog << "Lasagna manifest cache " << (hit ? "hit: " : "download: ") << redacted(location) << '\n';
    return {manifestPath, location, hit};
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
    const auto materialized = materializeLasagnaManifest(manifestLocation, options);
    auto openMaterialized = [&](const MaterializedLasagnaManifest& source,
                                const LasagnaDatasetOpenOptions& openOptions) {
        auto manifest = LasagnaDatasetManifest::parseFile(source.path);
        manifest.manifestLocation = sourceName(manifestLocation);
        manifest.manifestIsRemote = true;
        manifest.workingToBaseScale = openOptions.workingToBaseScale;
        manifest.remoteSourceBaseLocation = vcr::parentLocation(source.location);
        manifest.remoteCacheRoot = source.path.parent_path();
        resolveGroupLocations(manifest, openOptions);
        return LasagnaDataset(std::move(manifest));
    };
    return openMaterialized(materialized, options);
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

vcr::zarr::OpenedArray openLasagnaGroupArray(
    const LasagnaChannelGroup& group, std::shared_ptr<vcr::MemoryCache> cache)
{
    if (!group.isRemote() && std::filesystem::exists(group.zarrPath / ".native_pool.json"))
        return openPooledPredictionArray(group.zarrPath, std::move(cache));
    if (!group.isRemote()) {
        vcr::zarr::OpenOptions options;
        options.memoryCache = std::move(cache);
        return vcr::zarr::openArray(lasagnaGroupSourceLocation(group), options);
    }
    // Chunks cache under the group's root with VC's byte limit, floor and parallelism.
    const auto& location = group.sourceLocation;
    auto options = vc::settings::remoteOpenOptions(group.remoteCacheRoot);
    options.memoryCache = std::move(cache);
    options.access = lasagnaAccess(location, group.discoverAwsCredentials);
    return vcr::zarr::openArray(location, options);
}

} // namespace vc::lasagna
