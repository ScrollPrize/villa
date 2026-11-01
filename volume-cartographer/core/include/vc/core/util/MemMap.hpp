#pragma once

// Portable memory-mapping helpers: POSIX mmap on *nix, file mappings /
// VirtualAlloc on Windows. Implementation lives in MemMap.cpp so <windows.h>
// never leaks into headers.

#include <cstddef>
#include <filesystem>

namespace vc::memmap {

// Current process id (getpid / _getpid).
int pid();

// Zero-initialised anonymous pages (MAP_ANONYMOUS / VirtualAlloc). Pages only
// become resident on first touch. Throws std::bad_alloc on failure.
void* anonAlloc(std::size_t bytes);
void anonFree(void* ptr, std::size_t bytes) noexcept;

}  // namespace vc::memmap
