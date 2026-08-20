#pragma once

#include <filesystem>
#include <span>
#include <cstddef>
#include <string_view>

namespace vc::core::util {

void replaceFileAtomically(
    const std::filesystem::path& source,
    const std::filesystem::path& destination);

void atomicWriteString(
    const std::filesystem::path& target,
    std::string_view text);

void atomicWriteBytes(
    const std::filesystem::path& target,
    std::span<const std::byte> bytes);

} // namespace vc::core::util
