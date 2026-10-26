#include "vc_object_store/Uuid.hpp"
#include <openssl/evp.h>

namespace vc::objects::util {

std::string nameUuid(std::string_view name)
{
    return nameUuid("e14f6968-9145-494d-8b06-ab2f2fac7f3b", name);
}

std::string nameUuid(std::string_view uuidNamespace, std::string_view name)
{
    const auto canonical = canonicalUuid(std::string(uuidNamespace));
    std::string input;
    for (std::size_t i = 0; i < canonical.size(); ++i) {
        if (canonical[i] == '-') continue;
        input.push_back(char(std::stoi(canonical.substr(i, 2), nullptr, 16)));
        ++i;
    }
    input.append(name);
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(input.data(), input.size(), digest, &size, EVP_sha1(), nullptr) != 1 || size < 16)
        throw std::runtime_error("UUID: SHA-1 failed");
    digest[6] = static_cast<unsigned char>((digest[6] & 0x0f) | 0x50);
    digest[8] = static_cast<unsigned char>((digest[8] & 0x3f) | 0x80);
    constexpr char hex[] = "0123456789abcdef";
    std::string uuid;
    uuid.reserve(36);
    for (std::size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid.push_back('-');
        uuid.push_back(hex[digest[i] >> 4]);
        uuid.push_back(hex[digest[i] & 0x0f]);
    }
    return uuid;
}

} // namespace vc::objects::util
