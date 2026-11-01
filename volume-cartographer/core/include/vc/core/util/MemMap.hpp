#pragma once

// Portable memory-mapping helpers: POSIX mmap on *nix, file mappings /
// VirtualAlloc on Windows. Implementation lives in MemMap.cpp so <windows.h>
// never leaks into headers.

#include <cstddef>
#include <filesystem>

namespace vc::memmap {

// Current process id (getpid / _getpid).
int pid();

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
