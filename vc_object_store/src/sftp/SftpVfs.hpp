#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <memory>
#include <span>
#include <string>

namespace vc::objects::detail {

// A nonempty legacy lock directory prevents old clients from accessing hubs
// that now use process locks. Never remove/recreate its lock file: flock is
// attached to its inode, not its pathname.
inline constexpr const char* kProcessLockName = "process-v1";
class RemoteHubScope {
public:
    explicit RemoteHubScope(const std::filesystem::path& hub);
    ~RemoteHubScope();
    RemoteHubScope(const RemoteHubScope&) = delete;
    RemoteHubScope& operator=(const RemoteHubScope&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Stable across healthy scopes on this thread; changes after any failed scope.
// Retaining it retains no server process or lock between operations.
std::shared_ptr<const void> remoteHubCacheIdentity();
// Remote SQLite access must be inside RemoteHubScope. One locked SSH process
// serves all files of that operation, without reconnecting after any failure.
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
void remoteEnableProcessLock(const std::string& hub);

}  // namespace vc::objects::detail
