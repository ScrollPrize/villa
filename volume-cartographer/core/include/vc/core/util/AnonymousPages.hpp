#pragma once

// Anonymous, lazily committed pages on POSIX and Windows.

#include <cstddef>

namespace vc::pages {

// Current process id (getpid / _getpid).
int pid();

// Zero-initialised anonymous pages (MAP_ANONYMOUS / VirtualAlloc). Pages only
// become resident on first touch. Throws std::bad_alloc on failure.
void* anonAlloc(std::size_t bytes);
void anonFree(void* ptr, std::size_t bytes) noexcept;

}  // namespace vc::pages
