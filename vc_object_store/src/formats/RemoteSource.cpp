#include "formats/RemoteSource.hpp"
#include <vc_reader/catalog.hpp>
#include <vc_reader/native/http.hpp>
#include <fstream>

namespace vc::objects::formats {
void fetchRemoteSurface(const std::string& location, const std::filesystem::path& directory)
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
        std::ofstream output(directory / name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(body.data()), std::streamsize(body.size()));
        if (!output) throw std::runtime_error("cannot write fetched " + std::string(name));
    }
}
}
