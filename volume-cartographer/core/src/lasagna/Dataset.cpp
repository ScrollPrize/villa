#include "vc/lasagna/Dataset.hpp"

#include "vc/core/util/RemoteCacheSettings.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include <stdexcept>
#include <string_view>
#include <utility>
#include "vc_reader/native/http.hpp"
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

namespace {

[[nodiscard]] vcr::zarr::Access remoteAccess(const std::string& location, bool discoverAwsCredentials)
{
    return vcr::zarr::resolveAccess(
        location, discoverAwsCredentials ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous);
}

// GET a remote object whole into `path` (written beside it, then renamed);
// throws when it does not exist or cannot be fetched.
void downloadObject(vcr::HttpClient& client, const vcr::zarr::Access& access,
                    const std::string& location, const std::filesystem::path& path)
{
    std::vector<std::uint8_t> body;
    if (!client.get(access.url(location), body))
        throw std::runtime_error("Remote Lasagna object does not exist: " + redacted(location));
    std::filesystem::create_directories(path.parent_path());
    vcr::zarr::atomicWrite(path, body.data(), body.size());
}

} // namespace

void cacheLasagnaProvenance(const LasagnaDatasetManifest& manifest,
                            const LasagnaFileFetcher& fetcher)
{
    if (manifest.manifestIsRemote || manifest.remoteSourceBaseLocation.empty()) return;
    const auto it = manifest.raw.find("provenance");
    if (it == manifest.raw.end() || !it->is_string() || it->get<std::string>().empty()) return;
    const std::filesystem::path name(it->get<std::string>());
    if (name.is_absolute() || name.lexically_normal() != name || *name.begin() == "..")
        throw std::runtime_error("Lasagna manifest names an unsafe provenance record: " + name.string());
    const auto destination = manifest.baseDirectory / name;
    if (std::filesystem::is_regular_file(destination)) return;
    const auto location = vcr::joinLocation(manifest.remoteSourceBaseLocation, name.generic_string());
    if (!fetcher) {
        const auto access = remoteAccess(location, manifest.discoverAwsCredentials);
        vcr::HttpOptions http;
        http.connections = 1;
        http.auth = access.auth;
        vcr::HttpClient client(http);
        downloadObject(client, access, location, destination);
        return;
    }
    std::filesystem::create_directories(destination.parent_path());
    const auto temporary = std::filesystem::path(destination.string() + ".partial");
    fetcher(location, temporary);
    std::filesystem::rename(temporary, destination);
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
    const auto access = vcr::zarr::resolveAccess(identity, options.discoverAwsCredentials ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous);
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
    if (!group.isRemote()) {
        vcr::zarr::OpenOptions options;
        options.memoryCache = std::move(cache);
        return vcr::zarr::openArray(lasagnaGroupSourceLocation(group), options);
    }
    // Chunks cache under the group's root with VC's byte limit, floor and parallelism.
    const auto& location = group.sourceLocation;
    auto options = vc::settings::remoteOpenOptions(group.remoteCacheRoot);
    options.memoryCache = std::move(cache);
    options.access = vcr::zarr::resolveAccess(location, group.discoverAwsCredentials ? vcr::zarr::AuthMode::Auto : vcr::zarr::AuthMode::Anonymous);
    return vcr::zarr::openArray(location, options);
}

namespace {
// Byte-for-byte copies of remote arrays, under the cache root at their URL's
// place, apart from the tree vc_reader caches the same arrays in.
constexpr const char* kLasagnaArrayCopies = "lasagna_arrays";
} // namespace

std::optional<std::filesystem::path> lasagnaChannelArrayCacheDirectory(
    const LasagnaChannelGroup& group)
{
    if (!group.isRemote()) return std::nullopt;
    return vcr::zarr::decodedCachePath(group.remoteCacheRoot / kLasagnaArrayCopies, group.sourceLocation);
}

namespace {

constexpr const char* kCachedArrayMarker = ".vc-cached-array.json";
constexpr int kCachedArrayWorkers = 64;

struct ListedObject {
    std::string relative;  // under the array's location
    std::uint64_t size = 0;
};

// The objects a marker records, empty when there is none (or an unreadable one).
std::vector<ListedObject> markedObjects(const std::filesystem::path& marker)
{
    std::vector<ListedObject> objects;
    try {
        std::ifstream in(marker);
        if (!in) return {};
        const auto document = nlohmann::json::parse(in);
        if (document.value("version", 0) != 1) return {};
        for (const auto& entry : document.at("objects"))
            objects.push_back({entry.at(0).get<std::string>(), entry.at(1).get<std::uint64_t>()});
    } catch (const std::exception&) {
        return {};
    }
    return objects;
}

bool cachedWhole(const std::filesystem::path& directory, const ListedObject& object)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(directory / object.relative, ec);
    return !ec && size == object.size;
}

std::vector<ListedObject> listArrayObjects(const LasagnaChannelGroup& group, const vcr::zarr::Access& access)
{
    const auto& location = group.sourceLocation;
    std::vector<ListedObject> objects;
    for (const auto& object : vcr::zarr::listObjects(access, location)) {
        const std::filesystem::path relative(object.key);
        if (object.key.empty() || object.key.back() == '/') continue;
        if (relative.is_absolute() || relative.lexically_normal() != relative || *relative.begin() == "..")
            throw std::runtime_error("Lasagna array " + redacted(location) + " lists an unsafe key: " + object.key);
        objects.push_back({object.key, object.size});
    }
    if (objects.empty()) throw std::runtime_error("Lasagna array " + redacted(location) + " lists no objects");
    return objects;
}

} // namespace

bool cacheLasagnaChannelArray(
    const LasagnaChannelGroup& group,
    const std::function<void(const LasagnaArrayCacheProgress&)>& progress,
    const std::function<bool()>& cancelled)
{
    const auto directory = lasagnaChannelArrayCacheDirectory(group);
    if (!directory) throw std::runtime_error("Lasagna group " + group.name + " is not remote");
    const auto marker = *directory / kCachedArrayMarker;
    auto objects = markedObjects(marker);
    const bool marked = !objects.empty();
    if (cancelled && cancelled()) return false;
    const auto access = remoteAccess(group.sourceLocation, group.discoverAwsCredentials);
    if (!marked) objects = listArrayObjects(group, access);

    std::vector<const ListedObject*> missing;
    for (const auto& object : objects)
        if (!cachedWhole(*directory, object)) missing.push_back(&object);
    if (marked && missing.empty()) return true;

    // Written whole, beside vc_reader's budgeted cache: eviction must not take
    // the array before it is read whole.
    vcr::HttpOptions http;
    http.connections = kCachedArrayWorkers;
    http.auth = access.auth;
    vcr::HttpClient client(http);
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<bool> stop{false};
    std::mutex reportMutex;
    std::exception_ptr failure;
    auto work = [&]() {
        for (std::size_t i = next++; i < missing.size() && !stop; i = next++) {
            if (cancelled && cancelled()) {
                stop = true;
                return;
            }
            try {
                const auto& object = *missing[i];
                downloadObject(client, access, vcr::joinLocation(group.sourceLocation, object.relative),
                               *directory / object.relative);
                const LasagnaArrayCacheProgress snapshot{++done, missing.size()};
                if (progress) {
                    std::lock_guard lock(reportMutex);
                    progress(snapshot);
                }
            } catch (...) {
                std::lock_guard lock(reportMutex);
                if (!failure) failure = std::current_exception();
                stop = true;
            }
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < kCachedArrayWorkers - 1; ++i) workers.emplace_back(work);
    work();
    for (auto& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    if (stop) return false;

    std::size_t refused = 0;
    for (const auto* object : missing)
        if (!cachedWhole(*directory, *object)) ++refused;
    if (refused)
        throw std::runtime_error(
            "Kept " + std::to_string(missing.size() - refused) + " of " +
            std::to_string(missing.size()) + " objects of " + directory->string() +
            " (an object changed size since it was listed?)");
    if (!marked) {
        nlohmann::json listing = nlohmann::json::array();
        for (const auto& object : objects) listing.push_back({object.relative, object.size});
        vcr::zarr::atomicWrite(marker, nlohmann::json{{"version", 1}, {"objects", listing}}.dump());
    }
    return true;
}

} // namespace vc::lasagna
