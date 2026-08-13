#pragma once

#include <filesystem>
#include <memory>
#include <string>

class CState;
class Volume;

namespace vc3d {

// Stable identity string for a volume's persistent chunk cache. Remote
// volumes key on URL/base-level/id so the cache survives re-attachment;
// local volumes key on the canonical path.
std::string normalizedVolumeCacheIdentity(const std::shared_ptr<Volume>& volume);

// FNV-1a hex digest used to turn a cache identity into a directory name.
std::string stableHexHash(const std::string& value);

// Root directory for remote chunk caches, resolving the per-volpkg setting,
// the user's persisted setting, and the host-mount overrides in
// vc3d::remoteCachePath() (in that priority order).
std::filesystem::path remoteCacheRootForState(const CState* state);

// Directory holding the persistent chunk cache for this volume, mirroring
// how Volume::createChunkCache resolves it: a volume with an explicit
// remote cache root uses <root>/<id>, otherwise the viewer-managed
// <cache_root>/<identity_hash> location. Empty for non-remote volumes.
std::filesystem::path persistentCacheDirForVolume(
    const std::shared_ptr<Volume>& volume,
    const CState* state);

} // namespace vc3d
