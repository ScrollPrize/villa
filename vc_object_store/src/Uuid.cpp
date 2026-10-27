#include "vc_object_store/Uuid.hpp"
#include <openssl/evp.h>

namespace vc::objects::util {

namespace {
std::string v5(const unsigned char (&space)[16], std::string_view name)
{
    std::string input(reinterpret_cast<const char*>(space), sizeof(space));
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
}

std::string nameUuid(std::string_view name)
{
    // Fixed namespace for names minted by vc_object_store; never change it.
    static constexpr unsigned char kNamespace[16] = {0xe1, 0x4f, 0x69, 0x68, 0x91, 0x45, 0x49, 0x4d,
                                                     0x8b, 0x06, 0xab, 0x2f, 0x2f, 0xac, 0x7f, 0x3b};
    return v5(kNamespace, name);
}

std::string nameUuid(const std::string& namespaceUuid, std::string_view name)
{
    const auto canonical = canonicalUuid(namespaceUuid);
    unsigned char space[16];
    for (std::size_t i = 0, j = 0; i < 16; ++i, j += 2) {
        if (canonical[j] == '-') ++j;
        space[i] = static_cast<unsigned char>(std::stoi(canonical.substr(j, 2), nullptr, 16));
    }
    return v5(space, name);
}

} // namespace vc::objects::util
