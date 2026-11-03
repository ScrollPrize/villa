#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace vc3d::line_annotation {

struct FiberDeleteTarget {
    uint64_t requestedId = 0;
    std::string objectUuid;
};

struct FiberDeleted {
    uint64_t id = 0;
    std::string objectUuid;
};

struct FiberDeleteOutcome {
    bool aborted = false;
    std::string error;
    std::vector<FiberDeleteTarget> requested;
    std::vector<FiberDeleted> deleted;
    std::vector<uint64_t> deletedIds;

    [[nodiscard]] bool deletedRequested(uint64_t requestedId) const
    {
        for (const auto& target : requested) {
            if (target.requestedId != requestedId) continue;
            return std::any_of(deleted.begin(), deleted.end(), [&](const auto& entry) {
                return !target.objectUuid.empty() && target.objectUuid == entry.objectUuid;
            });
        }
        return false;
    }
};

} // namespace vc3d::line_annotation
