#pragma once

#include "FiberSourceDedupe.hpp"
#include <cstdint>

namespace vc3d {

// Keep identities even while a source is unregistered. IDs are never reused
// during the controller lifetime, including those held by unsaved sessions.
//
// A binding is (source, file name) -> id. It follows the fiber, not the name:
// a rename moves it, a delete releases it, so a file that later appears under
// a name a different fiber once had is a different fiber and gets a fresh id.
class FiberRuntimeIds {
public:
    void remember(const std::filesystem::path& source, const std::string& file,
                  uint64_t id)
    {
        if (id == 0) return;
        _next = std::max(_next, id + 1);
        if (!file.empty()) _ids[fiberSourceFileKey(source, file)] = id;
    }

    // The fiber was renamed within its source: its id moves to the new name
    // and the old name is free.
    void rename(const std::filesystem::path& source, const std::string& oldFile,
                const std::string& newFile, uint64_t id)
    {
        if (!oldFile.empty() && oldFile != newFile) {
            _ids.erase(fiberSourceFileKey(source, oldFile));
        }
        remember(source, newFile, id);
    }

    // The fiber's file is gone: its name is free. The id it held is never
    // handed out again.
    void forget(const std::filesystem::path& source, const std::string& file)
    {
        if (!file.empty()) _ids.erase(fiberSourceFileKey(source, file));
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
