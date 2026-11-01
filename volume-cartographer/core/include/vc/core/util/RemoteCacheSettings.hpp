#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include <vc_reader/zarr.hpp>

namespace vc::settings {

inline constexpr auto kRemoteCacheDirectory = "viewer/remote_cache_dir";
inline constexpr auto kRemoteCacheDelta3d = "perf/remote_cache_delta3d";
inline constexpr bool kRemoteCacheDelta3dDefault = false;
inline constexpr auto kRemoteCacheMaxGib = "perf/remote_cache_max_gib";
inline constexpr unsigned long long kRemoteCacheMaxGibDefault = 0;  // unlimited
inline constexpr auto kRemoteCacheMinFreeGib = "perf/remote_cache_min_free_gib";
inline constexpr unsigned long long kRemoteCacheMinFreeGibDefault = 20;
inline constexpr auto kRemoteDownloadAutomatic = "perf/remote_download_automatic";
inline constexpr bool kRemoteDownloadAutomaticDefault = true;
inline constexpr auto kRemoteDownloadParallelism = "perf/remote_download_parallelism";
inline constexpr int kRemoteDownloadParallelismDefault = 16;

[[nodiscard]] std::filesystem::path settingsFilePath();
// Resolved once on first use. Changes written to VC3D.ini take effect after
// the process restarts.
[[nodiscard]] std::filesystem::path remoteCachePath();
[[nodiscard]] bool remoteCacheDelta3dEnabled();
// The byte limit of the remote cache root; nullopt means unlimited.
[[nodiscard]] std::optional<std::int64_t> remoteCacheMaxBytes();
[[nodiscard]] std::int64_t remoteCacheMinFreeBytes();
// Simultaneous downloads; nullopt leaves the choice to vc_reader.
[[nodiscard]] std::optional<int> remoteDownloadParallelism();
// These settings as vc_reader options for the cache root `root`: the root as the cache
// directory with its byte limit (unlimited lifts a limit stored there) and free-space
// floor, and the download parallelism. An empty root sets the parallelism only. Access
// is the caller's.
[[nodiscard]] vcr::zarr::OpenOptions remoteOpenOptions(const std::filesystem::path& root);

} // namespace vc::settings
