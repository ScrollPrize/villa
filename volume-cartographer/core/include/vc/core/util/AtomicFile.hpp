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

// Atomic rename without file or directory fsync.  Intended only for
// regenerable trial artifacts; callers that publish durable results must use
// atomicWriteString instead.
void ephemeralWriteString(
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

// A new directory written under a hidden sibling name and renamed to
// `target` by commit(), replacing a directory already there; removed if never
// committed. Readers never see it half written.
class StagedDirectory
{
public:
    explicit StagedDirectory(std::filesystem::path target);
    ~StagedDirectory();
    StagedDirectory(const StagedDirectory&) = delete;
    StagedDirectory& operator=(const StagedDirectory&) = delete;

    // Where to write until commit().
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return staging_; }
    [[nodiscard]] const std::filesystem::path& target() const noexcept { return target_; }
    void commit();

private:
    std::filesystem::path target_;
    std::filesystem::path staging_;
    bool committed_ = false;
};

} // namespace vc::core::util
