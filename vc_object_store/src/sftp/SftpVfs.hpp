#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>

namespace vc::objects::detail {

// SQLite VFS for hub files addressed as sftp:// or ssh:// URLs. It locks with
// the same `<file>.lock` directory as SQLite's unix-dotfile VFS, so clients on
// network mounts and SFTP clients share one hub safely. One ssh connection per
// destination serves every file and helper below.
const char* sftpVfs();
bool isRemotePath(const std::filesystem::path& path);
std::string canonicalRemotePath(const std::filesystem::path& path);
// The reason the last SFTP open on this thread failed, for error messages.
std::string sftpLastError();

struct RemoteStat {
    bool exists = false, directory = false;
    uint64_t size = 0;
    std::optional<uint32_t> modified;  // seconds since the Unix epoch
};
RemoteStat remoteStat(const std::string& url);
// Reads exactly out.size() bytes or throws.
void remoteRead(const std::string& url, uint64_t offset, std::span<std::byte> out);
// Streams the whole file to sink; returns the bytes delivered.
uint64_t remoteCopy(const std::string& url, const std::function<void(std::span<const std::byte>)>& sink);
void remoteRmdir(const std::string& url);

}  // namespace vc::objects::detail
