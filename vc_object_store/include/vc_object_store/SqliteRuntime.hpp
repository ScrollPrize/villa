#pragma once
#include <cstddef>
#include <optional>

namespace vc::objects {
// Explicit application startup configuration, before SQLite initialization.
// Does not open/close connections, change stored data, or alter read checks.
// A missing memoryStatistics preserves SQLite's ordinary default.
void configureSqliteRuntime(std::optional<bool> memoryStatistics = {},
                            std::size_t pageCacheBulkBytes = 0);
}
