#pragma once

#include <filesystem>
#include <cstddef>
#include <memory>
#include <span>
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

// Callers must hold exclusive ownership of the tree while removing abandoned
// writes; otherwise another process's live temporary file is indistinguishable
// from one left by a crash.
std::size_t cleanupAtomicWriteTemporaryFiles(
    const std::filesystem::path& root);

class ExclusiveDirectoryLock
{
public:
    explicit ExclusiveDirectoryLock(const std::filesystem::path& directory);
    ~ExclusiveDirectoryLock();

    ExclusiveDirectoryLock(const ExclusiveDirectoryLock&) = delete;
    ExclusiveDirectoryLock& operator=(const ExclusiveDirectoryLock&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::core::util
