#pragma once

// SFTP v3 client over the system OpenSSH `ssh` program (`ssh -s HOST sftp`).
// OpenSSH alone handles configuration, keys, the agent, ProxyJump and host
// verification; no SSH library is linked and no remote shell runs. Without an
// authenticator ssh runs in batch mode and never prompts. With one, every ssh
// prompt (password, key passphrase, one-time code, new host key) is answered by
// the application through SSH_ASKPASS (Linux and macOS).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vcr::sftp {

// sftp://[user@]host[:port]/absolute/path; ssh:// is accepted as an alias.
// Passwords, queries and fragments are rejected; the path is percent-decoded.
struct Url {
    std::string user, host;  // host without IPv6 brackets
    std::optional<uint16_t> port;
    std::string path = "/";
    // [user@]host[:port], decoded, as given to ssh.
    std::string authority() const;
    // Canonical sftp:// form with a percent-encoded path.
    std::string str() const;
    Url withPath(std::string path) const;
};
bool isSftpUrl(std::string_view text) noexcept;
Url parseUrl(std::string_view text);  // throws std::invalid_argument
// RFC 3986 path encoding: unreserved, sub-delims, ':', '@' and '/' stay literal.
std::string encodePath(std::string_view path);

// One question from ssh. `confirmation` questions (accepting a new host key)
// expect "yes" or "no"; the others expect secret text.
struct Prompt {
    std::string destination;  // [user@]host[:port]
    std::string text;         // ssh's own wording
    bool confirmation = false;
};
// Returns the answer, or nullopt to cancel the connection attempt. It may be
// called from any thread while a connection is being established.
using Authenticator = std::function<std::optional<std::string>(const Prompt&)>;
// Process-wide. Sessions started afterwards use it; empty restores batch mode.
void setAuthenticator(Authenticator authenticator);
// Asks on the controlling terminal, without echo for secrets. Empty when the
// process has no terminal or the platform has no askpass support.
Authenticator terminalAuthenticator();

struct Options {
    std::chrono::seconds connectTimeout{30};
    // A request fails when the server sends nothing for this long. Time spent
    // waiting for the user to answer a prompt does not count.
    std::chrono::milliseconds idleTimeout{30000};
    std::function<bool()> cancelled;  // polled while waiting
};

enum Status : uint32_t {
    Eof = 1, NoSuchFile = 2, PermissionDenied = 3, Failure = 4, BadMessage = 5,
    NoConnection = 6, ConnectionLost = 7, OpUnsupported = 8,
};
// The server answered with an error status; the session remains usable.
struct StatusError : std::runtime_error {
    uint32_t status;
    StatusError(uint32_t status, const std::string& message);
};

struct Attributes {
    std::optional<uint64_t> size;
    std::optional<uint32_t> permissions;  // POSIX mode including type bits
    std::optional<uint32_t> modified;     // seconds since the Unix epoch
    bool directory() const { return permissions && (*permissions & 0170000) == 0040000; }
    bool symlink() const { return permissions && (*permissions & 0170000) == 0120000; }
};
struct DirectoryEntry {
    std::string name;
    Attributes attributes;  // as listed; symlinks are not followed
};

enum OpenFlags : uint32_t { Read = 1, Write = 2, Append = 4, Create = 8, Truncate = 0x10, Exclusive = 0x20 };

// One ssh process. Not thread-safe: callers serialize use of a session.
// Transport failures throw std::runtime_error and leave the session unusable;
// StatusError leaves it usable.
class Session {
public:
    explicit Session(const Url& destination, Options options = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void setOptions(Options options);
    const Url& destination() const;
    bool supports(std::string_view extension) const;  // e.g. "fsync@openssh.com"

    Attributes stat(const std::string& path);
    Attributes lstat(const std::string& path);
    std::vector<DirectoryEntry> list(const std::string& path);
    std::string realpath(const std::string& path);
    void mkdir(const std::string& path, uint32_t mode = 0777);
    void rmdir(const std::string& path);
    void remove(const std::string& path);
    void rename(const std::string& from, const std::string& to);

    using Handle = std::string;
    Handle open(const std::string& path, uint32_t flags, uint32_t mode = 0644);
    void close(const Handle& handle);
    Attributes fstat(const Handle& handle);
    void truncate(const Handle& handle, uint64_t size);
    void setPermissions(const Handle& handle, uint32_t mode);  // not subject to the server's umask
    void fsync(const Handle& handle);  // requires fsync@openssh.com
    // Reads up to out.size() bytes with several requests in flight; returns
    // fewer only at end of file.
    size_t read(const Handle& handle, uint64_t offset, std::span<std::byte> out);
    // Streams [offset, offset + length) to sink in order; stops early at end of
    // file and returns the number of bytes delivered.
    uint64_t read(const Handle& handle, uint64_t offset, uint64_t length,
                  const std::function<void(std::span<const std::byte>)>& sink);
    // Queues the write without waiting for its reply. Every other request on
    // this session first waits for queued writes, so reads observe them;
    // flush() waits explicitly and reports the first failure.
    void write(const Handle& handle, uint64_t offset, std::span<const std::byte> data);
    void flush();
    bool writesPending() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vcr::sftp
