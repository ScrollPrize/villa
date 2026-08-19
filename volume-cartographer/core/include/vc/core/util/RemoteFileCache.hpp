#pragma once

#include "vc/core/util/RemoteAuth.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace vc::core::util
{

enum class RemoteFileCachePolicy {
    CacheFirst,
    Refresh,
};

enum class RemoteFileCacheAccounting {
    Unmanaged,
    Managed,
};

using RemoteFileFetcher = std::function<void(const std::string& sourceLocation, const std::filesystem::path& temporaryPath)>;

struct RemoteFileCacheOptions {
    std::filesystem::path cacheRoot;
    std::filesystem::path destination;
    RemoteFileCachePolicy policy = RemoteFileCachePolicy::CacheFirst;
    RemoteFileCacheAccounting accounting = RemoteFileCacheAccounting::Unmanaged;
    vc::HttpAuth auth;
    // When false and auth is empty, S3 requests are always anonymous and the
    // machine's ambient AWS configuration is not consulted.
    bool discoverAwsCredentials = true;
    RemoteFileFetcher fetcher;
};

struct RemoteFileCacheResult {
    std::filesystem::path path;
    std::string normalizedEndpoint;
    bool cacheHit = false;
    std::optional<vc::render::PersistentZarrCacheBudget::ReadPin> readPin;
};

[[nodiscard]] std::string normalizeRemoteFileLocation(const std::string& sourceLocation);
[[nodiscard]] std::string redactedRemoteLocation(std::string_view location);
// Return the persistent, credential-free source name used by cache sidecars.
[[nodiscard]] std::string remoteFileCacheSource(std::string_view sourceLocation);
// Mirror a remote source below remote_sources/<scheme>/<authority>/<path>.
// Components that are unsafe or non-portable as filesystem names are rejected.
[[nodiscard]] std::filesystem::path remoteFileCachePath(std::string_view sourceLocation);

[[nodiscard]] RemoteFileCacheResult cacheRemoteFile(const std::string& sourceLocation, const RemoteFileCacheOptions& options);

void invalidateRemoteFileCacheEntry(const RemoteFileCacheOptions& options);

}  // namespace vc::core::util
