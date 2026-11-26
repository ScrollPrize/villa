#pragma once
#include <filesystem>
#include <functional>
#include <system_error>

namespace vcr::files::detail {
// True for the errors a rename reports when the filesystem refuses the no-replace
// flag itself (NFS on Linux answers EINVAL, exFAT on macOS ENOTSUP).
bool renameFlagRefused(const std::error_code& error);

using Rename = std::function<void(const std::filesystem::path&, const std::filesystem::path&)>;

// Publishes a staged directory where the filesystem refuses the no-replace rename. A
// plain rename still publishes the whole directory at once and fails with EEXIST on a
// destination that holds anything, though one created empty in the meantime would be
// replaced. An error after the staged directory is gone means the rename took effect
// (NFS can report it failed), so that counts as published. `rename` is for tests.
void renameNewDirectory(const std::filesystem::path& source, const std::filesystem::path& destination,
                        const Rename& rename = {});

// publishPath without overwrite for a staged directory, falling back to
// renameNewDirectory where the no-replace rename is refused (NFS, exFAT).
void publishNewDirectory(const std::filesystem::path& source, const std::filesystem::path& destination);

// How a save publishes its staged directory. With overwrite, a destination that existed
// when the save began goes through publishDirectory's journal, and one that did not is
// published new, so a segment another save published meanwhile makes this one fail.
void publishStaged(const std::filesystem::path& source, const std::filesystem::path& destination,
                   bool overwrite, bool existed);
}
