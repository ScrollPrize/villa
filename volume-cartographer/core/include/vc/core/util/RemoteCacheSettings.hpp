#pragma once

#include <filesystem>

namespace vc::settings {

inline constexpr auto kRemoteCacheDirectory = "viewer/remote_cache_dir";
inline constexpr auto kRemoteCacheDelta3d = "perf/remote_cache_delta3d";
inline constexpr bool kRemoteCacheDelta3dDefault = false;

[[nodiscard]] std::filesystem::path settingsFilePath();
// Resolved once on first use. Changes written to VC3D.ini take effect after
// the process restarts.
[[nodiscard]] std::filesystem::path remoteCachePath();
[[nodiscard]] bool remoteCacheDelta3dEnabled();

} // namespace vc::settings
