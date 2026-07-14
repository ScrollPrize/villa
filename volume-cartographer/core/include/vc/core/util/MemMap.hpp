#pragma once

// Portable memory-mapping helpers: POSIX mmap on *nix, file mappings /
// VirtualAlloc on Windows. Implementation lives in MemMap.cpp so <windows.h>
// never leaks into headers.

#include <cstddef>
#include <filesystem>

namespace vc::memmap {

// Current process id (getpid / _getpid).
int pid();

// Whole-file read-only mapping. The underlying file handle is closed once the
// view is established; the view stays valid until close()/destruction.
class MappedFileRO {
public:
    MappedFileRO() = default;
    explicit MappedFileRO(const std::filesystem::path& path) { open(path); }
    MappedFileRO(const MappedFileRO&) = delete;
    MappedFileRO& operator=(const MappedFileRO&) = delete;
    MappedFileRO(MappedFileRO&& other) noexcept { moveFrom(other); }
    MappedFileRO& operator=(MappedFileRO&& other) noexcept
    {
        if (this != &other) {
            close();
            moveFrom(other);
        }
        return *this;
    }
    ~MappedFileRO() { close(); }

    // Maps the entire file; throws std::runtime_error on failure (including
    // empty files — check size beforehand if empty is a valid state).
    void open(const std::filesystem::path& path);
    void close() noexcept;

    const void* data() const { return data_; }
    std::size_t size() const { return size_; }
    explicit operator bool() const { return data_ != nullptr; }

private:
    void moveFrom(MappedFileRO& other) noexcept
    {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    void* data_ = nullptr;
    std::size_t size_ = 0;
};

// Shared read-write mapping of `bytes` bytes backed by `path`. Creates the
// file if missing and extends it to `bytes` if shorter. The mapping is not
// tracked — release it with unmapRW(). Throws std::runtime_error on failure.
void* mapFileRW(const std::filesystem::path& path, std::size_t bytes);
void unmapRW(void* ptr, std::size_t bytes) noexcept;

// Zero-initialised anonymous pages (MAP_ANONYMOUS / VirtualAlloc). Pages only
// become resident on first touch. Throws std::bad_alloc on failure.
void* anonAlloc(std::size_t bytes);
void anonFree(void* ptr, std::size_t bytes) noexcept;

}  // namespace vc::memmap
