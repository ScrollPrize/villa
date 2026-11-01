#pragma once

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vc::core::util
{

// Definite answers for small remote metadata objects (Zarr .zgroup/.zarray/
// .zattrs/.zmetadata/zarr.json and volume meta.json) published under the Open
// Data public channels. Those objects are treated as immutable: once a GET
// answered 200 or 404, later opens reuse the answer from memory or disk
// instead of asking the network again. Callers record nothing else (403s,
// server errors), so those keep their uncached classification. invalidate()
// forgets everything under a URL prefix; the next open refetches it.
class RemoteMetadataCache {
public:
    // The object's bytes, or nullopt when the object does not exist.
    using Answer = std::optional<std::vector<std::byte>>;

    explicit RemoteMetadataCache(std::filesystem::path root);

    // Whether answers for this URL may be cached: an https or s3 URL, without
    // a query, under an Open Data public channel.
    [[nodiscard]] static bool cacheable(std::string_view url);

    [[nodiscard]] std::optional<Answer> lookup(std::string_view url);
    void record(std::string_view url, const Answer& answer);
    void invalidate(std::string_view urlPrefix);

private:
    std::filesystem::path diskPath(const std::string& url) const;

    std::filesystem::path root_;
    std::mutex mutex_;
    std::unordered_map<std::string, Answer> answers_;
};

// The process-wide cache below remoteCachePath()/remote_metadata.
[[nodiscard]] RemoteMetadataCache& remoteMetadataCache();

}  // namespace vc::core::util
