#include "formats/GeometryInput.hpp"
#include "vc_object_store/Uuid.hpp"
#include <openssl/evp.h>
#include <fstream>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Json readJson(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return Json::parse(input);
}
std::string uuid(const Json& data)
{
    return util::canonicalUuid(data.at("uuid").get<std::string>());
}
namespace {
// RFC 9562 name-based (SHA-1) UUID, matching Python's uuid.uuid5.
std::string uuid5(const std::string& space, const std::string& name)
{
    const auto canonical = util::canonicalUuid(space);
    std::string input;
    for (size_t i = 0; i < canonical.size(); ++i) {
        if (canonical[i] == '-') continue;
        input.push_back(char(std::stoi(canonical.substr(i, 2), nullptr, 16)));
        ++i;
    }
    input += name;
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned size = 0;
    if (EVP_Digest(input.data(), input.size(), digest, &size, EVP_sha1(), nullptr) != 1 || size < 16)
        throw std::runtime_error("SHA-1 digest failed");
    digest[6] = (digest[6] & 0x0f) | 0x50;
    digest[8] = (digest[8] & 0x3f) | 0x80;
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result.push_back('-');
        result.push_back(hex[digest[i] >> 4]);
        result.push_back(hex[digest[i] & 0x0f]);
    }
    return result;
}
}
void deriveSourceUuid(Json& root, const std::string& kind, const std::string& key,
    const std::optional<std::string>& uuidNamespace)
{
    if (!uuidNamespace) return;
    const auto existing = root.find("uuid");
    if (existing != root.end() && existing->is_string() && util::isUuid(existing->get<std::string>())) return;
    if (key.empty()) throw std::runtime_error("cannot derive a UUID from an empty source name");
    const auto name = kind + ":" + key;
    root["uuid"] = uuid5(*uuidNamespace, name);
    root["uuid_derivation"] = {{"namespace", util::canonicalUuid(*uuidNamespace)}, {"name", name}};
}
ObjectRecord sourceObject(const Json& root, const std::string& kind, const fs::path& source)
{
    ObjectRecord result;
    result.uuid = uuid(root); result.kind = kind; result.name = source.stem().string();
    result.storageKind = "file"; result.storageRef = fs::absolute(source).lexically_normal().string();
    result.formatVersion = root.value("version", int64_t(1));
    result.name = root.value("name", result.name);
    if (root.contains("tags") && root.at("tags").is_array())
        for (const auto& tag : root.at("tags")) result.tags.push_back(tag.get<std::string>());
    if (kind == "point_collection" && root.contains("tags") && root.at("tags").is_object())
        for (const auto& [name, value] : root.at("tags").items()) result.tags.push_back(name);
    return result;
}
}
