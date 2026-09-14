#pragma once

#include <filesystem>

namespace vc::settings {

inline constexpr auto kRemoteCacheDirectory = "viewer/remote_cache_dir";

[[nodiscard]] std::filesystem::path settingsFilePath();
// Resolved once on first use. Changes written to VC3D.ini take effect after
// the process restarts.
[[nodiscard]] std::filesystem::path remoteCachePath();

} // namespace vc::settings
