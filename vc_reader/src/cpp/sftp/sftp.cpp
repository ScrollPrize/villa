#include "vc_reader/sftp.hpp"
#include "ssh_process.hpp"

#include <algorithm>
#include <cctype>
#include <deque>
#include <map>
#include <mutex>

namespace vcr::sftp {
namespace {
constexpr uint32_t kMaxPacket = 16 * 1024 * 1024;
constexpr uint32_t kChunk = 32768;          // per READ/WRITE request; every server accepts it
constexpr size_t kReadsInFlight = 16;
constexpr size_t kWritesInFlight = 64;

enum Type : uint8_t {
    Init = 1, Version = 2, Open = 3, Close = 4, ReadRequest = 5, WriteRequest = 6, Lstat = 7, Fstat = 8,
    Fsetstat = 10, Opendir = 11, Readdir = 12, Remove = 13, Mkdir = 14, Rmdir = 15, Realpath = 16, Stat = 17,
    Rename = 18, StatusReply = 101, HandleReply = 102, DataReply = 103, NameReply = 104, AttrsReply = 105,
    Extended = 200,
};
enum AttributeFlags : uint32_t { SizeFlag = 1, OwnerFlag = 2, PermissionsFlag = 4, TimesFlag = 8, ExtendedFlag = 0x80000000u };

std::mutex authenticatorMutex;
Authenticator processAuthenticator;

void u32(std::string& b, uint32_t v)
{
    for (int n = 24; n >= 0; n -= 8) b += char(v >> n);
}
void u64(std::string& b, uint64_t v)
{
    u32(b, uint32_t(v >> 32));
    u32(b, uint32_t(v));
}
void str(std::string& b, std::string_view v)
{
    u32(b, uint32_t(v.size()));
    b.append(v);
}

struct Packet {
    std::string bytes;
    size_t pos = 0;
    uint8_t byte()
    {
        if (pos >= bytes.size()) throw std::runtime_error("truncated SFTP packet");
        return uint8_t(bytes[pos++]);
    }
    uint32_t integer()
    {
        if (bytes.size() - pos < 4) throw std::runtime_error("truncated SFTP packet");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v = (v << 8) | uint8_t(bytes[pos++]);
        return v;
    }
    uint64_t large()
    {
        const uint64_t hi = integer();
        return (hi << 32) | integer();
    }
    std::string string()
    {
        const auto n = integer();
        if (n > bytes.size() - pos) throw std::runtime_error("invalid SFTP string length");
        auto out = bytes.substr(pos, n);
        pos += n;
        return out;
    }
    bool done() const { return pos >= bytes.size(); }
};

Attributes attributes(Packet& p)
{
    Attributes a;
    const auto flags = p.integer();
    if (flags & SizeFlag) a.size = p.large();
    if (flags & OwnerFlag) { p.integer(); p.integer(); }
    if (flags & PermissionsFlag) a.permissions = p.integer();
    if (flags & TimesFlag) { p.integer(); a.modified = p.integer(); }
    if (flags & ExtendedFlag)
        for (auto n = p.integer(); n--;) { p.string(); p.string(); }
    return a;
}

int hexValue(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
std::string decode(std::string_view text)
{
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '%') { out += text[i]; continue; }
        if (i + 2 >= text.size() || hexValue(text[i + 1]) < 0 || hexValue(text[i + 2]) < 0)
            throw std::invalid_argument("Invalid percent encoding in SFTP URL");
        out += char(hexValue(text[i + 1]) * 16 + hexValue(text[i + 2]));
        i += 2;
    }
    return out;
}
std::string encode(std::string_view text, std::string_view literal)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out;
    for (const char c : text) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || std::string_view("-._~").find(c) != std::string_view::npos ||
            literal.find(c) != std::string_view::npos)
            out += c;
        else { out += '%'; out += digits[u >> 4]; out += digits[u & 15]; }
    }
    return out;
}
}  // namespace

StatusError::StatusError(uint32_t code, const std::string& message)
    : std::runtime_error("SFTP status " + std::to_string(code) + ": " + message), status(code)
{}

bool isSftpUrl(std::string_view text) noexcept
{
    auto starts = [&](std::string_view prefix) {
        if (text.size() < prefix.size()) return false;
        for (size_t i = 0; i < prefix.size(); ++i)
            if (std::tolower(static_cast<unsigned char>(text[i])) != prefix[i]) return false;
        return true;
    };
    return starts("sftp://") || starts("ssh://");
}

std::string encodePath(std::string_view path) { return encode(path, "!$&'()*+,;=:@/"); }

Url parseUrl(std::string_view text)
{
    const auto invalid = [] {
        return std::invalid_argument("Expected sftp://[user@]host[:port]/absolute/path (no password, query or fragment)");
    };
    if (!isSftpUrl(text)) throw invalid();
    for (const char c : text)
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) throw invalid();
    const auto rest = text.substr(text.find("://") + 3);
    if (rest.find_first_of("?#") != std::string_view::npos) throw invalid();
    const auto slash = rest.find('/');
    const auto authority = rest.substr(0, slash);
    Url url;
    auto hostPort = authority;
    if (const auto at = authority.rfind('@'); at != std::string_view::npos) {
        const auto user = authority.substr(0, at);
        if (user.empty() || user.find(':') != std::string_view::npos) throw invalid();
        url.user = decode(user);
        hostPort = authority.substr(at + 1);
    }
    std::string_view port;
    if (hostPort.starts_with('[')) {
        const auto close = hostPort.find(']');
        if (close == std::string_view::npos) throw invalid();
        url.host = hostPort.substr(1, close - 1);
        const auto after = hostPort.substr(close + 1);
        if (!after.empty()) {
            if (after[0] != ':') throw invalid();
            port = after.substr(1);
            if (port.empty()) throw invalid();
        }
        if (url.host.empty() || url.host.find_first_not_of("0123456789abcdefABCDEF:.") != std::string::npos) throw invalid();
    } else {
        const auto colon = hostPort.rfind(':');
        url.host = hostPort.substr(0, colon);
        if (colon != std::string_view::npos) {
            port = hostPort.substr(colon + 1);
            if (port.empty()) throw invalid();
        }
        for (const char c : url.host)
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '-' && c != '_') throw invalid();
    }
    if (url.host.empty() || url.host[0] == '-' || (!url.user.empty() && url.user[0] == '-')) throw invalid();
    std::transform(url.host.begin(), url.host.end(), url.host.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (!port.empty()) {
        if (port.size() > 5 || port.find_first_not_of("0123456789") != std::string_view::npos) throw invalid();
        const auto value = std::stoul(std::string(port));
        if (value == 0 || value > 65535) throw invalid();
        url.port = uint16_t(value);
    }
    url.path = slash == std::string_view::npos ? "/" : decode(rest.substr(slash));
    if (url.path.find_first_of(std::string_view("\0\n\r", 3)) != std::string::npos)
        throw std::invalid_argument("Invalid control character in SFTP path");
    return url;
}

std::string Url::authority() const
{
    auto out = user.empty() ? std::string() : user + "@";
    out += host.find(':') != std::string::npos ? "[" + host + "]" : host;
    if (port) out += ":" + std::to_string(*port);
    return out;
}

std::string Url::str() const
{
    auto out = std::string("sftp://") + (user.empty() ? "" : encode(user, "!$&'()*+,;=") + "@");
    out += host.find(':') != std::string::npos ? "[" + host + "]" : host;
    if (port) out += ":" + std::to_string(*port);
    return out + encodePath(path);
}

Url Url::withPath(std::string newPath) const
{
    auto out = *this;
    out.path = std::move(newPath);
    return out;
}

void setAuthenticator(Authenticator authenticator)
{
    std::lock_guard lock(authenticatorMutex);
    processAuthenticator = std::move(authenticator);
}

struct Session::Impl {
    Url url;
    Options options;
    std::unique_ptr<detail::SshProcess> process;
    uint32_t serial = 0;
    std::map<uint32_t, Packet> replies;
    std::map<std::string, std::string, std::less<>> extensions;
    std::deque<uint32_t> writes;  // unacknowledged WRITE requests, oldest first
    std::chrono::steady_clock::time_point lastActivity = std::chrono::steady_clock::now();
    bool waitingForUser = false;

    void check()
    {
        if (options.cancelled && options.cancelled()) throw std::runtime_error("SFTP transfer cancelled");
        const auto now = std::chrono::steady_clock::now();
        if (process->prompting()) { waitingForUser = true; return; }
        if (waitingForUser) { waitingForUser = false; lastActivity = now; }
        if (now - lastActivity > options.idleTimeout)
            throw std::runtime_error("SSH connection failed or stalled: " + process->diagnostics() +
                                     " Verify login and host trust using ssh from a terminal.");
    }
    std::string readExact(size_t n)
    {
        std::string out(n, '\0');
        for (size_t have = 0; have < n;) {
            check();
            const auto got = process->read(std::as_writable_bytes(std::span(out).subspan(have)), std::chrono::milliseconds(100));
            if (got) { have += got; lastActivity = std::chrono::steady_clock::now(); }
        }
        return out;
    }
    Packet packet()
    {
        Packet header{readExact(4)};
        const auto size = header.integer();
        if (!size || size > kMaxPacket) throw std::runtime_error("invalid SFTP packet size");
        return {readExact(size)};
    }
    void send(const std::string& body)
    {
        std::string framed;
        u32(framed, uint32_t(body.size()));
        framed += body;
        lastActivity = std::chrono::steady_clock::now();
        process->write(std::as_bytes(std::span(framed)), [this] { check(); });
    }
    uint32_t request(uint8_t type, const std::string& payload)
    {
        const auto id = ++serial;
        if (id == 0) throw std::runtime_error("SFTP request IDs exhausted; reconnecting");
        std::string b(1, char(type));
        u32(b, id);
        send(b + payload);
        return id;
    }
    // Replies to pipelined requests may arrive in any order.
    Packet response(uint32_t id, uint8_t expected)
    {
        while (!replies.contains(id)) {
            auto incoming = packet();
            incoming.pos = 1;
            const auto responseId = incoming.integer();
            incoming.pos = 0;
            if (responseId == 0 || responseId > serial || replies.size() > kReadsInFlight + kWritesInFlight ||
                !replies.emplace(responseId, std::move(incoming)).second)
                throw std::runtime_error("Invalid SFTP response ID");
        }
        auto p = std::move(replies.at(id));
        replies.erase(id);
        const auto type = p.byte();
        if (p.integer() != id) throw std::runtime_error("SFTP response ID mismatch");
        if (type == StatusReply) {
            const auto status = p.integer();
            const auto message = p.string();
            if (status || expected != StatusReply) throw StatusError(status, message);
        } else if (type != expected)
            throw std::runtime_error("Unexpected SFTP response type");
        return p;
    }
    // Every request other than WRITE first collects queued write replies, so a
    // later read or sync cannot overtake them.
    void flush()
    {
        std::optional<StatusError> failure;
        while (!writes.empty()) {
            const auto id = writes.front();
            writes.pop_front();
            try { response(id, StatusReply); }
            catch (const StatusError& error) { if (!failure) failure = error; }
        }
        if (failure) throw *failure;
    }
    Packet call(uint8_t type, const std::string& payload, uint8_t expected)
    {
        flush();
        return response(request(type, payload), expected);
    }
    std::string pathPayload(const std::string& path)
    {
        std::string b;
        str(b, path);
        return b;
    }
};

Session::Session(const Url& destination, Options options) : impl_(std::make_unique<Impl>())
{
    impl_->url = destination;
    impl_->options = std::move(options);
    Authenticator authenticator;
    {
        std::lock_guard lock(authenticatorMutex);
        authenticator = processAuthenticator;
    }
    impl_->process = std::make_unique<detail::SshProcess>(destination, impl_->options, std::move(authenticator));
    std::string init(1, char(Init));
    u32(init, 3);
    impl_->send(init);
    auto version = impl_->packet();
    if (version.byte() != Version || version.integer() != 3) throw std::runtime_error("Server does not support SFTP v3");
    while (!version.done()) {
        auto name = version.string();
        impl_->extensions[std::move(name)] = version.string();
    }
    impl_->process->authenticated();
}

Session::~Session() = default;

void Session::setOptions(Options options)
{
    impl_->options = std::move(options);
    impl_->lastActivity = std::chrono::steady_clock::now();
}
const Url& Session::destination() const { return impl_->url; }
bool Session::supports(std::string_view extension) const { return impl_->extensions.find(extension) != impl_->extensions.end(); }

Attributes Session::stat(const std::string& path)
{
    auto p = impl_->call(Stat, impl_->pathPayload(path), AttrsReply);
    return attributes(p);
}
Attributes Session::lstat(const std::string& path)
{
    auto p = impl_->call(Lstat, impl_->pathPayload(path), AttrsReply);
    return attributes(p);
}

std::vector<DirectoryEntry> Session::list(const std::string& path)
{
    const auto handle = impl_->call(Opendir, impl_->pathPayload(path), HandleReply).string();
    std::vector<DirectoryEntry> result;
    for (;;) {
        Packet p;
        try {
            p = impl_->call(Readdir, impl_->pathPayload(handle), NameReply);
        } catch (const StatusError& error) {
            if (error.status == Eof) break;
            close(handle);
            throw;
        }
        for (auto count = p.integer(); count--;) {
            auto name = p.string();
            p.string();  // long name
            auto a = attributes(p);
            if (name == "." || name == "..") continue;
            if (name.find('/') != std::string::npos || name.find('\0') != std::string::npos)
                throw std::runtime_error("Invalid SFTP directory entry");
            result.push_back({std::move(name), a});
        }
    }
    close(handle);
    return result;
}

std::string Session::realpath(const std::string& path)
{
    auto p = impl_->call(Realpath, impl_->pathPayload(path), NameReply);
    if (p.integer() != 1) throw std::runtime_error("Invalid SFTP realpath reply");
    return p.string();
}

void Session::mkdir(const std::string& path, uint32_t mode)
{
    auto b = impl_->pathPayload(path);
    u32(b, PermissionsFlag);
    u32(b, mode);
    impl_->call(Mkdir, b, StatusReply);
}
void Session::rmdir(const std::string& path) { impl_->call(Rmdir, impl_->pathPayload(path), StatusReply); }
void Session::remove(const std::string& path) { impl_->call(Remove, impl_->pathPayload(path), StatusReply); }
void Session::rename(const std::string& from, const std::string& to)
{
    // Plain SFTP rename refuses an existing target; OpenSSH's extension replaces it.
    const bool posix = supports("posix-rename@openssh.com");
    std::string b;
    if (posix) str(b, "posix-rename@openssh.com");
    str(b, from);
    str(b, to);
    impl_->call(posix ? Extended : Rename, b, StatusReply);
}

Session::Handle Session::open(const std::string& path, uint32_t flags, uint32_t mode)
{
    auto b = impl_->pathPayload(path);
    u32(b, flags);
    if (flags & Create) { u32(b, PermissionsFlag); u32(b, mode); }
    else u32(b, 0);
    return impl_->call(Open, b, HandleReply).string();
}
void Session::close(const Handle& handle) { impl_->call(Close, impl_->pathPayload(handle), StatusReply); }
Attributes Session::fstat(const Handle& handle)
{
    auto p = impl_->call(Fstat, impl_->pathPayload(handle), AttrsReply);
    return attributes(p);
}
void Session::truncate(const Handle& handle, uint64_t size)
{
    auto b = impl_->pathPayload(handle);
    u32(b, SizeFlag);
    u64(b, size);
    impl_->call(Fsetstat, b, StatusReply);
}
void Session::setPermissions(const Handle& handle, uint32_t mode)
{
    auto b = impl_->pathPayload(handle);
    u32(b, PermissionsFlag);
    u32(b, mode);
    impl_->call(Fsetstat, b, StatusReply);
}
void Session::fsync(const Handle& handle)
{
    if (!supports("fsync@openssh.com")) throw StatusError(OpUnsupported, "server lacks fsync@openssh.com");
    std::string b;
    str(b, "fsync@openssh.com");
    str(b, handle);
    impl_->call(Extended, b, StatusReply);
}

size_t Session::read(const Handle& handle, uint64_t offset, std::span<std::byte> out)
{
    impl_->flush();
    struct Pending { uint32_t id, size; uint64_t offset; };
    std::deque<Pending> pending;
    uint64_t submitted = 0, end = out.size();
    auto submit = [&](uint64_t from, uint32_t size) {
        auto b = impl_->pathPayload(handle);
        u64(b, offset + from);
        u32(b, size);
        pending.push_back({impl_->request(ReadRequest, b), size, from});
    };
    while (submitted < end || !pending.empty()) {
        while (submitted < end && pending.size() < kReadsInFlight) {
            const auto n = uint32_t(std::min<uint64_t>(kChunk, end - submitted));
            submit(submitted, n);
            submitted += n;
        }
        const auto r = pending.front();
        pending.pop_front();
        Packet p;
        try {
            p = impl_->response(r.id, DataReply);
        } catch (const StatusError& error) {
            if (error.status != Eof) {
                for (const auto& rest : pending) try { impl_->response(rest.id, DataReply); } catch (const StatusError&) {}
                throw;
            }
            end = std::min(end, r.offset);  // the file ends here
            continue;
        }
        const auto data = p.string();
        if (data.empty() || data.size() > r.size) throw std::runtime_error("Invalid SFTP read length");
        std::copy_n(reinterpret_cast<const std::byte*>(data.data()), data.size(), out.begin() + ptrdiff_t(r.offset));
        // Servers may return short reads: request the remainder separately.
        if (data.size() < r.size && r.offset + data.size() < end)
            submit(r.offset + data.size(), uint32_t(std::min<uint64_t>(r.size - data.size(), end - r.offset - data.size())));
    }
    return size_t(end);
}

uint64_t Session::read(const Handle& handle, uint64_t offset, uint64_t length,
                       const std::function<void(std::span<const std::byte>)>& sink)
{
    std::vector<std::byte> window(4 << 20);
    uint64_t delivered = 0;
    while (delivered < length) {
        const auto want = size_t(std::min<uint64_t>(window.size(), length - delivered));
        const auto got = read(handle, offset + delivered, std::span(window).first(want));
        if (got) sink(std::span(window).first(got));
        delivered += got;
        if (got < want) break;
    }
    return delivered;
}

void Session::write(const Handle& handle, uint64_t offset, std::span<const std::byte> data)
{
    for (size_t done = 0; done < data.size();) {
        const auto n = std::min<size_t>(kChunk, data.size() - done);
        auto b = impl_->pathPayload(handle);
        u64(b, offset + done);
        str(b, std::string_view(reinterpret_cast<const char*>(data.data() + done), n));
        impl_->writes.push_back(impl_->request(WriteRequest, b));
        done += n;
        if (impl_->writes.size() > kWritesInFlight) {
            const auto id = impl_->writes.front();
            impl_->writes.pop_front();
            impl_->response(id, StatusReply);
        }
    }
}
void Session::flush() { impl_->flush(); }
bool Session::writesPending() const { return !impl_->writes.empty(); }

}  // namespace vcr::sftp
