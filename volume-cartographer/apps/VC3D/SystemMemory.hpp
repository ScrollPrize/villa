#pragma once

#include <cstdint>

namespace vc3d {

// Installed physical memory in bytes, or 0 if it can't be determined.
std::uint64_t physicalMemoryBytes();

// Default RAM chunk-cache size: a third of installed memory, between 2 GB and
// the historical 10 GB default. The fixed 10 GB used to push 8-16 GB machines
// into swap.
int defaultRamCacheSizeGB();

}  // namespace vc3d
