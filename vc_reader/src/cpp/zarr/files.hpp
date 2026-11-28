#pragma once
#include <filesystem>
#include <functional>
#include <ostream>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vcr::zarr {
struct Access;
}

namespace vcr::zarr::detail {
void releaseDiskBudget(const std::string& root);

// Writes beside the destination and publishes only after the complete file closes.
void atomicWriteFile(const std::filesystem::path& path, const std::function<void(std::ostream&)>& write);

// Where a cache metadata file goes under the disk policy of its root: the cache directory
// (of `cacheDirs`, or `vc3dVolumeDir`'s parent) that `path` lies under. Written when the root
// keeps its free-space floor (`maxBytes` replaces the root's stored limit, as an array open
// does); a path under no known root is written as it is. Whether written.
bool writeCacheMetadata(const std::filesystem::path& path, const std::string& data, const std::vector<std::string>& cacheDirs,
                        const std::optional<std::filesystem::path>& vc3dVolumeDir, std::optional<std::int64_t> maxBytes,
                        std::optional<std::int64_t> minFreeBytes);

// Drops the array documents this process knows (see readArrayDocument) at a remote
// location and below it, so their next open reads them as published.
void forgetKnownArrays(const std::string& location, const Access& access);
}
