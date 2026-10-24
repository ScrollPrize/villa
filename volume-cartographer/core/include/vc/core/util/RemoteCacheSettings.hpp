#pragma once

#include <cstdint>
#include <filesystem>

namespace vc::settings {

inline constexpr auto kRemoteCacheDirectory = "viewer/remote_cache_dir";
inline constexpr auto kRemoteCacheDelta3d = "perf/remote_cache_delta3d";
inline constexpr bool kRemoteCacheDelta3dDefault = false;
// Disk budget for every managed remote Zarr chunk beneath remoteCachePath().
// Zero maximum means unlimited.
inline constexpr auto kRemoteCacheMaxGiB = "perf/remote_cache_max_gib";
inline constexpr std::uint64_t kRemoteCacheMaxGiBDefault = 0;
inline constexpr auto kRemoteCacheMinFreeGiB = "perf/remote_cache_min_free_gib";
inline constexpr std::uint64_t kRemoteCacheMinFreeGiBDefault = 20;

[[nodiscard]] std::filesystem::path settingsFilePath();
// Resolved once on first use. Changes written to VC3D.ini take effect after
// the process restarts.
[[nodiscard]] std::filesystem::path remoteCachePath();
[[nodiscard]] bool remoteCacheDelta3dEnabled();
// Applies the VC3D.ini disk budget above to remoteCachePath(), as the VC3D app
// does at startup. Command-line tools call it before opening a remote volume;
// without a configured budget the persistent cache grows until the disk fills.
// Reuses a budget that is already configured for that root.
void configureRemoteCacheBudget();

} // namespace vc::settings
