#include "vc_object_store/Uuid.hpp"
#include <openssl/evp.h>
#include "internal/Sqlite.hpp"
#include <memory>

namespace vc::objects::util {

namespace {
std::string v5(const std::array<unsigned char,16>& space, std::string_view name)
{
    // Reuse a thread-local context and explicitly fetched algorithm: EVP_Digest
    // otherwise repeatedly allocates a context and resolves providers globally.
    thread_local const auto context=std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(),EVP_MD_CTX_free);
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    static const auto algorithm=std::unique_ptr<EVP_MD,decltype(&EVP_MD_free)>(EVP_MD_fetch(nullptr,"SHA1",nullptr),EVP_MD_free);
    const auto* sha1=algorithm.get();
#else
    const auto* sha1=EVP_sha1();
#endif
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    // After the first digest, retain this thread's selected implementation.
    // Passing it again needlessly changes shared provider reference counts.
    thread_local bool initialized=false;
    if (!context || !sha1 || EVP_DigestInit_ex(context.get(),initialized?nullptr:sha1,nullptr)!=1 ||
        EVP_DigestUpdate(context.get(),space.data(),space.size())!=1 ||
        EVP_DigestUpdate(context.get(),name.data(),name.size())!=1 ||
        EVP_DigestFinal_ex(context.get(),digest,&size)!=1 || size<16)
        throw std::runtime_error("UUID: SHA-1 failed");
    initialized=true;
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
    static constexpr std::array<unsigned char,16> kNamespace = {0xe1, 0x4f, 0x69, 0x68, 0x91, 0x45, 0x49, 0x4d,
                                                     0x8b, 0x06, 0xab, 0x2f, 0x2f, 0xac, 0x7f, 0x3b};
    return v5(kNamespace, name);
}

std::string nameUuid(const std::string& namespaceUuid, std::string_view name)
{
    return v5(detail::parseUuid(namespaceUuid), name);
}
std::string nameUuid(const std::array<unsigned char,16>& namespaceUuid,std::string_view name)
{return v5(namespaceUuid,name);}

} // namespace vc::objects::util
