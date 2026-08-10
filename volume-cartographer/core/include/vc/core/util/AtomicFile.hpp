#pragma once

#include <filesystem>
#include <string_view>

namespace vc::core::util {

void replaceFileAtomically(
    const std::filesystem::path& source,
    const std::filesystem::path& destination);

void atomicWriteString(
    const std::filesystem::path& target,
    std::string_view text);

} // namespace vc::core::util
