#pragma once
#include "vc_object_store/migration/ConvertStore.hpp"
struct sqlite3;
namespace vc::objects::migration {
// Explicit converter only: ordinary runtime opens never understand schema 21.
bool isSchema21Store(const std::filesystem::path&);
ConversionResult convertSchema21(const std::filesystem::path&,const std::filesystem::path&,sqlite3* lockedSource=nullptr,
                                const std::filesystem::path& memberDirectory={});
}
