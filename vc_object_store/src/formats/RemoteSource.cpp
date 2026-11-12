#include "formats/RemoteSource.hpp"
#include "formats/GeometryInput.hpp"
#include <vc_reader/native/platform_io.hpp>
#include <vc_reader/catalog.hpp>
#include <vc_reader/native/http.hpp>
#include <vc_reader/native/atomic_file.hpp>

namespace vc::objects::formats {
bool isRemoteSource(const std::string& location)
{
    return location.starts_with("http://") || location.starts_with("https://") || location.starts_with("s3://");
}

static void fetchRemoteSurface(const std::string& location, const std::filesystem::path& directory)
{
    auto base = vcr::catalog::resolveOpenDataUrl(location);
    while (base.size() > 1 && base.back() == '/') base.pop_back();
    vcr::HttpClient client(vcr::HttpOptions{});
    for (const auto& [name, required] : std::initializer_list<std::pair<const char*, bool>>{
             {"meta.json", true}, {"x.tif", true}, {"y.tif", true}, {"z.tif", true},
             {"mask.tif", false}}) {
        std::vector<std::uint8_t> body;
        if (!client.get(base + "/" + name, body)) {
            if (required) throw std::runtime_error("remote surface is missing " + std::string(name) + ": " + base);
            continue;
        }
        vcr::atomic_write((directory / name).string(), body.data(), body.size());
    }
}

Prepared readRemoteSurfaceGeometry(const std::string& location)
{
    namespace fs = std::filesystem;
    struct Staging {
        fs::path path;
        ~Staging() { std::error_code ignored; fs::remove_all(path, ignored); }
    } staging{vcr::platform::make_temp_directory((fs::temp_directory_path() / "vc-remote-surface-XXXXXX").string())};
    fetchRemoteSurface(location, staging.path);
    return surfaceGeometry(staging.path);
}
}
