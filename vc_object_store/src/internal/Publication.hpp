#pragma once
#include <vc_reader/native/platform_io.hpp>
#include "vc_object_store/Uuid.hpp"
#include <filesystem>

namespace vc::objects::detail {
// The caller must close/checkpoint and sync the staging file first. Atomic
// no-replace publication avoids an exists()+rename race. Keep the staging link
// as an ownership witness until publication is durable, so an error can remove
// only this attempt's destination (never another publisher's file).
inline void publishNewFile(const std::filesystem::path& staging,
                           const std::filesystem::path& destination)
{
    const auto from = std::filesystem::absolute(staging).lexically_normal();
    const auto to = std::filesystem::absolute(destination).lexically_normal();
    std::filesystem::create_hard_link(from, to);
    try {
        vcr::platform::sync_directory(to.parent_path(), "publish vcobj store");
        std::filesystem::remove(from);
        // No fallible work after unlinking the ownership witness. The output
        // is durable; a crash may retain its harmless private staging name.
    } catch (...) {
        std::error_code ignored;
        if (std::filesystem::equivalent(from, to, ignored)) std::filesystem::remove(to, ignored);
        throw;
    }
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
