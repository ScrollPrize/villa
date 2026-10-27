#include "formats/RemoteSource.hpp"
#include <vc_reader/native/http.hpp>
#include <fstream>

namespace vc::objects::formats {
namespace {
std::string httpLocation(std::string location)
{
    while (location.size() > 1 && location.back() == '/') location.pop_back();
    if (location.starts_with("s3://")) {
        const auto rest = location.substr(5);
        const auto slash = rest.find('/');
        if (slash == std::string::npos) throw std::invalid_argument("missing S3 key: " + location);
        location = "https://" + rest.substr(0, slash) + ".s3.amazonaws.com/" + rest.substr(slash + 1);
    }
    return location;
}
}

void fetchRemoteSurface(const std::string& location, const std::filesystem::path& directory)
{
    const auto base = httpLocation(location);
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
