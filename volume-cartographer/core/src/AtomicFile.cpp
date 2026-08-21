#include "vc/core/util/AtomicFile.hpp"

#include <fstream>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <random>
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
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vc::core::util {

namespace {

std::string uniqueTemporarySuffix()
{
    // Isolated/containerized processes can reuse the same PID while sharing a
    // cache directory. Include a per-process random tag so an interrupted run
    // cannot collide with a later writer's temporary name.
    static const auto processTag = static_cast<std::uint64_t>(
        std::random_device{}());
    static std::atomic<std::uint64_t> counter{0};
    return ".tmp." + std::to_string(processTag) + "." +
        std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

} // namespace

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
    atomicWriteBytes(
        target,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(text.data()), text.size()));
}

void atomicWriteBytes(
    const std::filesystem::path& target,
    std::span<const std::byte> bytes)
{
    if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
    auto temporary = target;
    temporary += uniqueTemporarySuffix();
    try {
        {
            errno = 0;
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                const std::error_code error(errno, std::generic_category());
                throw std::filesystem::filesystem_error(
                    "cannot open temporary file for write", temporary, error);
            }
            output.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
            if (!output)
                throw std::runtime_error("write failed for " + temporary.string());
            output.flush();
            if (!output)
                throw std::runtime_error("flush failed for " + temporary.string());
        }
#if defined(_WIN32)
        HANDLE file = ::CreateFileW(
            temporary.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE || !::FlushFileBuffers(file)) {
            const auto error = static_cast<int>(::GetLastError());
            if (file != INVALID_HANDLE_VALUE)
                ::CloseHandle(file);
            throw std::filesystem::filesystem_error(
                "cannot flush temporary file", temporary,
                std::error_code(error, std::system_category()));
        }
        ::CloseHandle(file);
#else
        const int file = ::open(temporary.c_str(), O_RDONLY);
        if (file < 0 || ::fsync(file) != 0) {
            const std::error_code error(errno, std::generic_category());
            if (file >= 0)
                ::close(file);
            throw std::filesystem::filesystem_error(
                "cannot fsync temporary file", temporary, error);
        }
        ::close(file);
#endif
        replaceFileAtomically(temporary, target);
#if !defined(_WIN32)
        const auto parent = target.parent_path().empty()
            ? std::filesystem::path{"."}
            : target.parent_path();
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (directory < 0 || ::fsync(directory) != 0) {
            const std::error_code error(errno, std::generic_category());
            if (directory >= 0)
                ::close(directory);
            throw std::filesystem::filesystem_error(
                "cannot fsync parent directory", parent, error);
        }
        ::close(directory);
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace vc::core::util
