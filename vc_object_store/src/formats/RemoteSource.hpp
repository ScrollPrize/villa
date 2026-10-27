#pragma once
#include <filesystem>
#include <string>

namespace vc::objects::formats {
// Registered sources addressed by http(s):// or s3:// URL (public buckets).
bool isRemoteSource(const std::string& location);
// Download a remote tifxyz directory (meta.json, x/y/z.tif, and mask.tif when
// present) into an existing local directory.
void fetchRemoteSurface(const std::string& location, const std::filesystem::path& directory);
}
