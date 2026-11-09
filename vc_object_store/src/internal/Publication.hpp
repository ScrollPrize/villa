#pragma once
#include <vc_reader/native/platform_io.hpp>
#include "vc_object_store/Uuid.hpp"
#include <filesystem>

namespace vc::objects::detail {
// The caller must close/checkpoint and sync the staging file first. Atomic
// no-replace publication avoids an exists()+rename race. A directory-sync error
// after linking means the complete destination may exist but durability is not
// confirmed; never treat that as permission to overwrite an existing file.
inline void publishNewFile(const std::filesystem::path& staging,
                           const std::filesystem::path& destination)
{
    const auto from = std::filesystem::absolute(staging).lexically_normal();
    const auto to = std::filesystem::absolute(destination).lexically_normal();
    std::filesystem::create_hard_link(from, to);
    vcr::platform::sync_directory(to.parent_path(), "publish vcobj store");
    std::filesystem::remove(from);
    vcr::platform::sync_directory(from.parent_path(), "remove vcobj staging name");
}

// SQLite sibling staging, including any sidecars left after an interrupted
// initialization. Packed builder locators already live in a private container.
struct StagedDatabase {
    std::filesystem::path destination, path;
    bool active, cleanup;
    StagedDatabase(const std::filesystem::path& target, bool privatePacked)
        : destination(target), path(privatePacked ? target : std::filesystem::path(target.string() + ".building-" + util::newUuid())), active(!privatePacked), cleanup(!privatePacked) {}
    StagedDatabase(const StagedDatabase&) = delete;
    ~StagedDatabase()
    {
        if (cleanup) for (const auto* suffix : {"", "-wal", "-shm", "-journal", "-owner"}) {
            std::error_code error; std::filesystem::remove(path.string() + suffix, error);
        }
    }
    void publish() { if (active) { publishNewFile(path, destination); active = false; } }
};
} // namespace vc::objects::detail
