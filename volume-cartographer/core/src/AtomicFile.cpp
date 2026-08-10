#include "vc/core/util/AtomicFile.hpp"

#include <fstream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vc::core::util {

void replaceFileAtomically(
    const std::filesystem::path& source,
    const std::filesystem::path& destination)
{
#if defined(_WIN32)
    if (!::MoveFileExW(source.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code error(
            static_cast<int>(::GetLastError()), std::system_category());
        throw std::filesystem::filesystem_error(
            "cannot replace file", source, destination, error);
    }
#else
    std::filesystem::rename(source, destination);
#endif
}

void atomicWriteString(
    const std::filesystem::path& target,
    std::string_view text)
{
    if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
    auto temporary = target;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error(
                "cannot open " + temporary.string() + " for write");
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!output)
            throw std::runtime_error("write failed for " + temporary.string());
    }
    replaceFileAtomically(temporary, target);
}

} // namespace vc::core::util
