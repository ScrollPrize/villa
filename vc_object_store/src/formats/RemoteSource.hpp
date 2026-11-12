#pragma once
#include <filesystem>
#include <string>

namespace vc::objects { struct GeometrySource; }
namespace vc::objects::formats {
// Registered sources addressed by http(s):// or s3:// URL (public buckets).
bool isRemoteSource(const std::string& location);
GeometrySource readRemoteSurfaceGeometry(const std::string& location);
}
