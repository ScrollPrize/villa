#pragma once

#include "FiberSourceDedupe.hpp"
#include <cstdint>

namespace vc3d {

// Keep identities even while a source is unregistered. IDs are never reused
// during the controller lifetime, including those held by unsaved sessions.
class FiberRuntimeIds {
public:
    void remember(const std::filesystem::path& source, const std::string& file,
                  uint64_t id)
    {
        if (id == 0) return;
        _next = std::max(_next, id + 1);
        if (!file.empty()) _ids[fiberSourceFileKey(source, file)] = id;
    }

    uint64_t forFile(const std::filesystem::path& source, const std::string& file)
    {
        auto [it, inserted] = _ids.try_emplace(fiberSourceFileKey(source, file), 0);
        if (inserted) it->second = allocate();
        return it->second;
    }

    uint64_t allocate() { return _next++; }

private:
    uint64_t _next = 1;
    std::unordered_map<std::string, uint64_t> _ids;
};

} // namespace vc3d
