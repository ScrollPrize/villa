#pragma once

#include <filesystem>
#include <cstddef>
#include <memory>

namespace vc::core::util {

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
