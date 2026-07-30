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
[[nodiscard]] std::string remoteFileIdentityHex(std::string_view normalizedLocation);

// Append a collision-free segmented URL identity below base. The identity is
// hex rather than a digest so collisions cannot alias unrelated remote files.
[[nodiscard]] std::filesystem::path remoteFileIdentityPath(const std::filesystem::path& base, std::string_view normalizedLocation);

[[nodiscard]] RemoteFileCacheResult cacheRemoteFile(const std::string& sourceLocation, const RemoteFileCacheOptions& options);

void invalidateRemoteFileCacheEntry(const RemoteFileCacheOptions& options);

}  // namespace vc::core::util
