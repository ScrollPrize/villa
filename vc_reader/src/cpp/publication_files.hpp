#pragma once
#include <filesystem>
#include <functional>
#include <system_error>

namespace vcr::files::detail {
// True for the errors a rename reports when the filesystem refuses the no-replace
// flag itself (NFS on Linux answers EINVAL, exFAT on macOS ENOTSUP).
bool renameFlagRefused(const std::error_code& error);

using Move = std::function<void(const std::filesystem::path&, const std::filesystem::path&)>;

// Publishes a staged directory of regular files at a new destination without the
// no-replace rename: mkdir reserves the destination, the files move in with
// meta.json last, and a failure moves them back and removes the reservation.
// Returns false and changes nothing when the destination already exists.
// `move` replaces the rename, for tests.
bool publishNewByFiles(const std::filesystem::path& source, const std::filesystem::path& destination,
                       const Move& move = {});
}
