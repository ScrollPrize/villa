#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vc::objects {
struct ImportReport {
    int64_t objectsImported = 0;
    int64_t edgesImported = 0;
    std::vector<std::string> problems;  // unresolved/one-sided entries, skipped files
};

} // namespace vc::objects
