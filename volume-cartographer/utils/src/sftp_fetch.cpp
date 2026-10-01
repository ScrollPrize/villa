#include "utils/sftp_fetch.hpp"

#include <QProcess>
#include <QUrl>
#include <QElapsedTimer>
#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <map>
#include <cctype>
#include <stdexcept>

namespace utils
{
namespace
{
using Bytes = QByteArray;
constexpr uint32_t maxPacket = 16 * 1024 * 1024;

void u32(Bytes& b, uint32_t v)
{
    for (int n = 24; n >= 0; n -= 8)
        b.append(char(v >> n));
}
void u64(Bytes& b, uint64_t v)
{
    u32(b, uint32_t(v >> 32));
    u32(b, uint32_t(v));
}
void str(Bytes& b, const Bytes& v)
{
    u32(b, uint32_t(v.size()));
    b.append(v);
}

struct Packet {
    Bytes bytes;
    qsizetype pos = 0;
    uint32_t integer()
    {
        if (bytes.size() - pos < 4)
            throw std::runtime_error("truncated SFTP packet");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v = (v << 8) | uint8_t(bytes[pos++]);
        return v;
    }
    uint64_t large()
    {
        const auto hi = integer();
        return (uint64_t(hi) << 32) | integer();
    }
    Bytes string()
    {
        const auto n = integer();
        if (n > uint64_t(bytes.size() - pos))
            throw std::runtime_error("invalid SFTP string length");
        auto out = bytes.mid(pos, n);
        pos += n;
        return out;
    }
};

struct Attributes {
    uint64_t size = 0;
    uint32_t permissions = 0;
    bool hasSize = false;
    bool hasPermissions = false;
};
Attributes attributes(Packet& p)
{
    Attributes a;
    const auto flags = p.integer();
    if (flags & 1) {
        a.size = p.large();
        a.hasSize = true;
    }
    if (flags & 2) {
        p.integer();
        p.integer();
    }
    if (flags & 4) {
        a.permissions = p.integer();
        a.hasPermissions = true;
    }
    if (flags & 8) {
        p.integer();
        p.integer();
    }
    if (flags & 0x80000000u) {
        auto n = p.integer();
        while (n--) {
            p.string();
            p.string();
        }
    }
    return a;
}

struct StatusError : std::runtime_error {
    uint32_t status;
    StatusError(uint32_t code, const std::string& message)
        : std::runtime_error("SFTP status " + std::to_string(code) + ": " + message), status(code)
    {
    }
};

QUrl parse(std::string_view input)
{
    const auto text = QString::fromUtf8(input.data(), qsizetype(input.size()));
    QUrl url(text, QUrl::StrictMode);
    if (!is_sftp_url(input) || !url.isValid() || url.host().isEmpty() || !url.password().isEmpty() || url.userInfo().contains(':') ||
        url.hasQuery() || url.hasFragment() || url.port() == 0 || url.host().startsWith('-') || url.userName().startsWith('-'))
        throw std::invalid_argument("Expected sftp://[user@]host[:port]/absolute/path (no password, query or fragment)");
    url.setScheme("sftp");
    const auto path = url.path();
    if (path.contains(QChar(0)) || path.contains('\n') || path.contains('\r'))
        throw std::invalid_argument("Invalid control character in SFTP path");
    if (path.isEmpty())
        url.setPath("/");
    return url;
}

// Binary SFTP v3 over OpenSSH stdin/stdout. OpenSSH alone handles config,
// agent/key authentication, ProxyJump and host verification; no remote shell.
class Session
{
    QProcess process;
    uint32_t serial = 0;
    int timeoutMs = 30000;
    QElapsedTimer idle;
    std::map<uint32_t, Packet> replies;
    std::string diagnostic() { return process.readAllStandardError().right(8192).toStdString(); }
    void check()
    {
        if (HttpClient::isAborted())
            throw std::runtime_error("SFTP transfer cancelled");
        if (process.state() == QProcess::NotRunning || idle.elapsed() > timeoutMs)
            throw std::runtime_error(
                "SSH connection failed or stalled: " + diagnostic() + " Verify login and host trust using ssh from a terminal.");
    }
    Bytes read(qsizetype n)
    {
        Bytes b;
        while (b.size() < n) {
            check();
            const auto part = process.read(n - b.size());
            if (!part.isEmpty()) {
                b += part;
                idle.restart();
            } else
                process.waitForReadyRead(100);
        }
        return b;
    }
    Packet packet()
    {
        Packet header{read(4)};
        const auto size = header.integer();
        if (!size || size > maxPacket)
            throw std::runtime_error("invalid SFTP packet size");
        return {read(size)};
    }
    void send(const Bytes& body)
    {
        Bytes framed;
        u32(framed, uint32_t(body.size()));
        framed += body;
        if (process.write(framed) != framed.size())
            throw std::runtime_error("SSH write failed");
        while (process.bytesToWrite()) {
            check();
            process.waitForBytesWritten(100);
        }
    }

public:
    Session(const QUrl& url, const HttpClient::Config& config)
    {
        QStringList args{
            "-T",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=yes",
            "-oClearAllForwardings=yes",
            "-oConnectTimeout=" + QString::number(config.connect_timeout.count())};
        if (!url.userName().isEmpty())
            args << "-l" << url.userName();
        if (url.port() != -1)
            args << "-p" << QString::number(url.port());
        args << "-s" << "--" << url.host() << "sftp";
        process.setProgram("ssh");
        process.setArguments(args);
        process.start();
        if (!process.waitForStarted(int(config.connect_timeout.count() * 1000)))
            throw std::runtime_error("Cannot start OpenSSH ssh: " + process.errorString().toStdString());
        idle.start();
        Bytes init(1, char(1));
        u32(init, 3);
        send(init);
        auto version = packet();
        if (uint8_t(version.bytes[version.pos++]) != 2 || version.integer() != 3)
            throw std::runtime_error("Server does not support SFTP v3");
    }
    ~Session()
    {
        process.kill();
        process.waitForFinished(1000);
    }
    void begin(const HttpClient::Config& config)
    {
        timeoutMs = int(std::clamp<int64_t>(
            (config.low_speed_time.count() > 0     ? config.low_speed_time.count()
             : config.transfer_timeout.count() > 0 ? config.transfer_timeout.count()
                                                   : 30) *
                1000,
            100,
            std::numeric_limits<int>::max()));
        idle.restart();
    }
    uint32_t request(uint8_t type, const Bytes& payload)
    {
        Bytes b(1, char(type));
        const auto id = ++serial;
        if (id == 0)
            throw std::runtime_error("SFTP request IDs exhausted; reconnecting");
        u32(b, id);
        b += payload;
        send(b);
        return id;
    }
    Packet response(uint32_t id, uint8_t expected)
    {
        // SFTP permits replies to pipelined reads in a different order.
        while (!replies.contains(id)) {
            auto incoming = packet();
            incoming.pos = 1;
            const auto responseId = incoming.integer();
            incoming.pos = 0;
            if (responseId == 0 || responseId > serial || replies.size() >= 16 || !replies.emplace(responseId, std::move(incoming)).second)
                throw std::runtime_error("Invalid SFTP response ID");
        }
        auto p = std::move(replies.at(id));
        replies.erase(id);
        const auto type = uint8_t(p.bytes[p.pos++]);
        if (p.integer() != id)
            throw std::runtime_error("SFTP response ID mismatch");
        if (type == 101) {
            const auto status = p.integer();
            const auto message = p.string().toStdString();
            if (status || expected != 101)
                throw StatusError(status, message);
        } else if (type != expected)
            throw std::runtime_error("Unexpected SFTP response type");
        return p;
    }
    Attributes stat(const Bytes& path)
    {
        Bytes b;
        str(b, path);
        auto p = response(request(17, b), 105);
        return attributes(p);
    }
    Bytes open(const Bytes& path, bool directory)
    {
        Bytes b;
        str(b, path);
        if (!directory) {
            u32(b, 1);
            u32(b, 0);
        }
        return response(request(directory ? 11 : 3, b), 102).string();
    }
    void close(const Bytes& handle)
    {
        Bytes b;
        str(b, handle);
        response(request(4, b), 101);
    }
    HttpResponse fetch(const QUrl& url, bool head, std::optional<std::pair<std::size_t, std::size_t>> range, const HttpClient::DownloadObserver& observer)
    {
        const auto path = url.path().toUtf8();
        Attributes a;
        try {
            a = stat(path);
        } catch (const StatusError& e) {
            // A missing sparse chunk is a normal response, not a broken
            // connection. Keep the session for subsequent chunk requests.
            if (e.status != 2 && e.status != 3)
                throw;
            HttpResponse out;
            out.status_code = e.status == 2 ? 404 : 403;
            out.error_message = e.what();
            return out;
        }
        if (!a.hasSize)
            throw std::runtime_error("SFTP file has no size");
        if ((a.permissions & 0170000) == 0040000)
            throw std::runtime_error("SFTP path is a directory, not a file");
        HttpResponse out;
        out.status_code = range ? 206 : 200;
        if (a.size > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("SFTP file too large");
        out.content_length = std::size_t(a.size);
        if (head)
            return out;
        const uint64_t start = range ? range->first : 0;
        if (start > a.size) {
            out.status_code = 416;
            return out;
        }
        const uint64_t size = range ? std::min<uint64_t>(range->second, a.size - start) : a.size;
        out.content_length = size;
        out.body.reserve(size);
        const auto handle = open(path, false);
        // Keep several reads in flight per file to avoid one RTT per 32 KiB.
        struct Read {
            uint32_t id;
            uint32_t size;
            uint64_t offset;
        };
        std::deque<Read> pending;
        uint64_t submitted = 0;
        while (submitted < size || !pending.empty()) {
            while (submitted < size && pending.size() < 16) {
                const auto n = uint32_t(std::min<uint64_t>(32768, size - submitted));
                Bytes b;
                str(b, handle);
                u64(b, start + submitted);
                u32(b, n);
                pending.push_back({request(5, b), n, start + submitted});
                submitted += n;
            }
            auto r = pending.front();
            pending.pop_front();
            auto data = response(r.id, 103).string();
            if (data.isEmpty() || data.size() > r.size)
                throw std::runtime_error("Invalid SFTP read length");
            // Servers may return short reads. Queue the remainder behind
            // existing requests and assemble by offset, not arrival order.
            const auto pos = r.offset - start;
            out.body.resize(std::max<uint64_t>(out.body.size(), pos + data.size()));
            std::copy_n(reinterpret_cast<const std::byte*>(data.constData()), data.size(), out.body.begin() + pos);
            if (observer)
                observer(data.size());
            if (data.size() < r.size) {
                Bytes b;
                str(b, handle);
                u64(b, r.offset + data.size());
                u32(b, r.size - data.size());
                pending.push_back({request(5, b), uint32_t(r.size - data.size()), r.offset + uint64_t(data.size())});
            }
        }
        close(handle);
        return out;
    }
    std::vector<SftpDirectoryEntry> list(const QUrl& url)
    {
        const auto handle = open(url.path().toUtf8(), true);
        std::vector<SftpDirectoryEntry> result;
        for (;;) {
            Bytes b;
            str(b, handle);
            Packet p;
            try {
                p = response(request(12, b), 104);
            } catch (const StatusError& e) {
                if (e.status == 1)
                    break;
                throw;
            }
            auto count = p.integer();
            while (count--) {
                const auto name = QString::fromUtf8(p.string());
                p.string();
                auto a = attributes(p);
                if (name == "." || name == "..")
                    continue;
                if (name.contains('/') || name.contains(QChar(0)))
                    throw std::runtime_error("Invalid SFTP directory entry");
                auto child = url;
                auto path = child.path();
                if (!path.endsWith('/'))
                    path += '/';
                child.setPath(path + name);
                const bool unknownType = !a.hasPermissions || (a.permissions & 0170000) == 0;
                if (unknownType || (a.permissions & 0170000) == 0120000) {
                    try {
                        a = stat(child.path().toUtf8());
                    } catch (const StatusError& e) {
                        if (unknownType || (e.status != 2 && e.status != 3))
                            throw;
                    }
                }
                if (!a.hasPermissions || (a.permissions & 0170000) == 0)
                    throw std::runtime_error("SFTP server did not supply an entry type for " + child.path().toStdString());
                const bool directory = (a.permissions & 0170000) == 0040000;
                if (directory)
                    child.setPath(child.path() + '/');
                result.push_back({name.toStdString(), child.toEncoded().toStdString(), directory});
            }
        }
        close(handle);
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.directory != b.directory ? a.directory > b.directory : a.name < b.name;
        });
        return result;
    }
};

// A bounded per-worker connection cache. No QProcess crosses a thread boundary.
struct CachedSession {
    QString authority;
    std::unique_ptr<Session> session;
};
thread_local std::deque<CachedSession> sessions;
Session& session(const QUrl& url, const HttpClient::Config& config)
{
    const auto key = url.authority();
    auto it = std::find_if(sessions.begin(), sessions.end(), [&](const auto& s) { return s.authority == key; });
    if (it == sessions.end()) {
        if (sessions.size() == 4)
            sessions.pop_front();
        sessions.push_back({key, std::make_unique<Session>(url, config)});
        it = std::prev(sessions.end());
    }
    it->session->begin(config);
    return *it->session;
}
void discard(const QUrl& url)
{
    std::erase_if(sessions, [&](const auto& s) { return s.authority == url.authority(); });
}
}  // namespace

bool is_sftp_url(std::string_view url) noexcept
{
    auto starts = [&](std::string_view prefix) {
        if (url.size() < prefix.size())
            return false;
        for (size_t i = 0; i < prefix.size(); ++i)
            if (std::tolower(static_cast<unsigned char>(url[i])) != prefix[i])
                return false;
        return true;
    };
    return starts("sftp://") || starts("ssh://");
}
std::string canonical_sftp_url(std::string_view url)
{
    return parse(url).toEncoded().toStdString();
}
std::string sftp_cache_authority(std::string_view url)
{
    return QUrl::toPercentEncoding(parse(url).authority(), "@.-_").toStdString();
}
HttpResponse fetch_sftp(
    const HttpClient::Config& config, std::string_view input, bool head, std::optional<std::pair<std::size_t, std::size_t>> range, const HttpClient::DownloadObserver& observer)
{
    const auto url = parse(input);
    for (size_t attempt = 0;; ++attempt) {
        try {
            if (HttpClient::isAborted())
                throw std::runtime_error("SFTP transfer cancelled");
            return session(url, config).fetch(url, head, range, observer);
        } catch (const StatusError& e) {
            discard(url);
            HttpResponse out;
            out.status_code = e.status == 2 ? 404 : e.status == 3 ? 403 : 500;
            out.error_message = e.what();
            return out;
        } catch (const std::exception& e) {
            discard(url);
            if (attempt < config.max_retries && !HttpClient::isAborted())
                continue;
            HttpResponse out;
            out.error_message = e.what();
            return out;
        }
    }
}
std::vector<SftpDirectoryEntry> list_sftp_directory(std::string_view input)
{
    const auto url = parse(input);
    try {
        return session(url, {}).list(url);
    } catch (...) {
        discard(url);
        throw;
    }
}
}  // namespace utils
