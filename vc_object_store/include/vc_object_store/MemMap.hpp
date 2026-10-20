#pragma once

// Portable file mappings: POSIX mmap on *nix, file mappings on Windows.
// Implementation lives in MemMap.cpp so <windows.h> never leaks into headers.

#include <cstddef>
#include <filesystem>

namespace vc::objects::memmap {

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

}  // namespace vc::objects::memmap
