#pragma once
#include <filesystem>
#include <string>

namespace vc::objects::formats {
// Download a remote tifxyz directory at an http(s):// or s3:// (public bucket)
// URL (meta.json, x/y/z.tif, and mask.tif when present) into an existing local
// directory.
void fetchRemoteSurface(const std::string& location, const std::filesystem::path& directory);
}
