#include "sftp/SftpVfs.hpp"
#include "vc_object_store/sftp/Sftp.hpp"

#include <sqlite3.h>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <charconv>
#include <cstdlib>
#include <exception>

namespace vc::objects::detail {
namespace {
using vc::sftp::Session;
using vc::sftp::StatusError;
constexpr const char* kVfsName = "vc-sftp";

thread_local std::string lastError;

// Hub operations own a dedicated connection that cannot reconnect. Standalone
// inspection helpers retain their ordinary SFTP connection per destination.
struct Connection {
    std::mutex mutex;
    vc::sftp::Url destination;
    std::unique_ptr<Session> session;
    uint64_t generation = 0;
    std::string hubPath;
    bool failed = false;
    bool active = false;
    unsigned lockedFiles = 0;
    std::optional<uint32_t> permissions;
    Session& get()
    {
        if (failed) throw std::runtime_error("hub: locked SSH session lost; retry the whole hub operation");
        if (!hubPath.empty() && !active)
            throw std::runtime_error("hub: remote I/O outside a locked operation");
        // Reestablish an idle transport before launching another command on
        // it. Locked sessions must never take this reconnect path.
        if (hubPath.empty() && session && !session->writesPending() && session->transportClosed())
            session.reset();
        if (!session) {
            vc::sftp::Options options;
            options.shareConnection = true;
            if (!hubPath.empty()) {
                options.processLock = hubPath + ".lock/" + kProcessLockName;
                if (const char* configured = std::getenv("VC_HUB_SESSION_SECONDS")) {
                    const std::string_view value(configured);
                    int seconds = 0;
                    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), seconds);
                    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || seconds <= 0)
                        throw std::invalid_argument("VC_HUB_SESSION_SECONDS must be a positive integer");
                    options.processLifetime = std::chrono::seconds(seconds);
                }
            }
            // Other clients hold the hub lock for whole syncs; a stalled server
            // should still fail a request eventually.
            options.idleTimeout = std::chrono::seconds(120);
            session = std::make_unique<Session>(destination, options);
            ++generation;
        }
        return *session;
    }
};

thread_local std::shared_ptr<Connection> activeHub;

bool belongsToHub(const vc::sftp::Url& url, const Connection& hub)
{
    return url.authority() == hub.destination.authority() &&
        (url.path == hub.hubPath || url.path == hub.hubPath + "-journal");
}

std::shared_ptr<Connection> standaloneConnection(const vc::sftp::Url& url)
{
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<Connection>> connections;
    std::lock_guard lock(mutex);
    auto& found = connections[url.authority()];
    if (!found) {
        found = std::make_shared<Connection>();
        found->destination = url.withPath("/");
    }
    return found;
}

std::shared_ptr<Connection> connection(const vc::sftp::Url& url)
{
    if (activeHub && belongsToHub(url, *activeHub)) return activeHub;
    return standaloneConnection(url);
}

// Runs work on the destination's session. A transport failure is retried once
// on a new connection, unless writes were still unacknowledged: they may be
// lost, so the caller must see the failure.
template<class F> auto withSession(Connection& c, F&& work)
{
    std::lock_guard lock(c.mutex);
    for (int attempt = 0;; ++attempt) {
        auto& session = c.get();
        try {
            return work(session);
        } catch (const StatusError&) {
            throw;
        } catch (const std::exception&) {
            const bool lostWrites = c.session && c.session->writesPending();
            c.session.reset();
            if (!c.hubPath.empty()) {
                c.failed = true;
                throw;  // Never replay I/O after the process owning the lock died.
            }
            if (attempt > 0 || lostWrites) throw;
        }
    }
}

struct State {
    std::shared_ptr<Connection> connection;
    std::string path;
    uint32_t flags = 0;  // SFTP open flags without creation
    Session::Handle handle;
    uint64_t generation = 0;
    int lock = SQLITE_LOCK_NONE;
    bool deleteOnClose = false;
    // A handle from an earlier connection is invalid on the current one.
    Session::Handle& open(Session& session)
    {
        if (generation != connection->generation) {
            handle = session.open(path, flags);
            generation = connection->generation;
        }
        return handle;
    }
};
struct File {
    sqlite3_file base;
    State* state;
};
State& state(sqlite3_file* file) { return *reinterpret_cast<File*>(file)->state; }

int fileClose(sqlite3_file* file)
{
    auto* s = reinterpret_cast<File*>(file)->state;
    int rc = SQLITE_OK;
    try {
        // Cached SQLite handles can be destroyed between operations. Their
        // old server already closed every descriptor; never reconnect to close.
        if (s->connection->session && !s->connection->failed) {
            withSession(*s->connection, [&](Session& session) {
                session.flush();
                if (s->generation == s->connection->generation) session.close(s->handle);
                if (s->deleteOnClose) session.remove(s->path);
            });
        }
    } catch (const std::exception& error) {
        lastError = error.what();
        rc = SQLITE_IOERR_CLOSE;
    }
    if (s->lock != SQLITE_LOCK_NONE) --s->connection->lockedFiles;
    delete s;
    reinterpret_cast<File*>(file)->state = nullptr;
    return rc;
}

int fileRead(sqlite3_file* file, void* buffer, int amount, sqlite3_int64 offset)
{
    auto& s = state(file);
    try {
        const auto got = withSession(*s.connection, [&](Session& session) {
            return session.read(s.open(session), uint64_t(offset), std::span(static_cast<std::byte*>(buffer), size_t(amount)));
        });
        if (got < size_t(amount)) {
            std::memset(static_cast<char*>(buffer) + got, 0, size_t(amount) - got);
            return SQLITE_IOERR_SHORT_READ;
        }
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_READ;
    }
}

int fileWrite(sqlite3_file* file, const void* buffer, int amount, sqlite3_int64 offset)
{
    auto& s = state(file);
    try {
        withSession(*s.connection, [&](Session& session) {
            session.write(s.open(session), uint64_t(offset), std::span(static_cast<const std::byte*>(buffer), size_t(amount)));
        });
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_WRITE;
    }
}

int fileTruncate(sqlite3_file* file, sqlite3_int64 size)
{
    auto& s = state(file);
    try {
        withSession(*s.connection, [&](Session& session) { session.truncate(s.open(session), uint64_t(size)); });
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_TRUNCATE;
    }
}

int fileSync(sqlite3_file* file, int)
{
    auto& s = state(file);
    try {
        withSession(*s.connection, [&](Session& session) {
            auto& handle = s.open(session);
            session.flush();
            session.fsync(handle);
        });
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_FSYNC;
    }
}

int fileSize(sqlite3_file* file, sqlite3_int64* size)
{
    auto& s = state(file);
    try {
        const auto attributes = withSession(*s.connection, [&](Session& session) { return session.fstat(s.open(session)); });
        if (s.path == s.connection->hubPath) s.connection->permissions = attributes.permissions;
        if (!attributes.size) throw std::runtime_error("SFTP server reported no file size");
        *size = sqlite3_int64(*attributes.size);
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_FSTAT;
    }
}

// The SSH process owns an exclusive kernel lock for the whole operation.
// SQLite levels are bookkeeping only; a lost process invalidates all handles.
int fileLock(sqlite3_file* file, int level)
{
    auto& s = state(file);
    if (s.connection->failed || !s.connection->active) return SQLITE_IOERR_LOCK;
    if (s.lock == SQLITE_LOCK_NONE && level != SQLITE_LOCK_NONE) ++s.connection->lockedFiles;
    s.lock = level;
    return SQLITE_OK;
}

int fileUnlock(sqlite3_file* file, int level)
{
    auto& s = state(file);
    try {
        withSession(*s.connection, [&](Session& session) { session.flush(); });
        if (s.lock != SQLITE_LOCK_NONE && level == SQLITE_LOCK_NONE) --s.connection->lockedFiles;
        s.lock = level;
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_UNLOCK;
    }
}

int fileCheckReservedLock(sqlite3_file* file, int* reserved)
{
    auto& s = state(file);
    if (s.connection->failed || !s.connection->active) return SQLITE_IOERR_CHECKRESERVEDLOCK;
    // No other process can be writing. A journal from the previous owner is
    // hot and must be recovered before this operation reads the database.
    *reserved = s.lock > SQLITE_LOCK_SHARED;
    return SQLITE_OK;
}

int fileControl(sqlite3_file*, int, void*) { return SQLITE_NOTFOUND; }
int fileSectorSize(sqlite3_file*) { return 4096; }
int fileDeviceCharacteristics(sqlite3_file*) { return 0; }

const sqlite3_io_methods kMethods = {
    1, fileClose, fileRead, fileWrite, fileTruncate, fileSync, fileSize, fileLock, fileUnlock,
    fileCheckReservedLock, fileControl, fileSectorSize, fileDeviceCharacteristics,
};

sqlite3_vfs* defaultVfs() { return sqlite3_vfs_find(nullptr); }

int vfsOpen(sqlite3_vfs*, const char* name, sqlite3_file* file, int flags, int* outFlags)
{
    file->pMethods = nullptr;
    if (!name) {
        // temp_store=MEMORY keeps temporary tables and statement journals in memory.
        lastError = "the SFTP VFS has no local temporary files";
        return SQLITE_CANTOPEN;
    }
    try {
        const auto url = vc::sftp::parseUrl(name);
        if (!activeHub || !belongsToHub(url, *activeHub))
            throw std::runtime_error("hub: remote databases must be accessed through the hub operations");
        auto s = std::make_unique<State>();
        s->connection = connection(url);
        s->path = url.path;
        s->deleteOnClose = flags & SQLITE_OPEN_DELETEONCLOSE;
        s->flags = vc::sftp::Read | (flags & SQLITE_OPEN_READWRITE ? vc::sftp::Write : 0);
        auto create = s->flags | (flags & SQLITE_OPEN_CREATE ? vc::sftp::Create : 0) |
                      (flags & SQLITE_OPEN_EXCLUSIVE ? vc::sftp::Exclusive : 0);
        withSession(*s->connection, [&](Session& session) {
            if ((flags & SQLITE_OPEN_READWRITE) && !session.supports("fsync@openssh.com"))
                throw std::runtime_error("the SFTP server lacks fsync@openssh.com, so writes cannot be made durable");
            s->handle = session.open(s->path, create);
            s->generation = s->connection->generation;
            // Like SQLite's unix VFS, a journal takes its database's permission
            // bits, so every user who may write the hub can roll it back.
            constexpr std::string_view suffix = "-journal";
            if ((flags & SQLITE_OPEN_MAIN_JOURNAL) && (flags & SQLITE_OPEN_CREATE) && s->path.ends_with(suffix)) {
                // SQLite normally already fstat'ed the database under this
                // operation's lock. Reuse those permissions, not another RTT.
                auto& permissions = s->connection->permissions;
                if (!permissions) permissions = session.stat(s->connection->hubPath).permissions;
                if (permissions) session.setPermissions(s->handle, *permissions & 07777);
            }
        });
        reinterpret_cast<File*>(file)->state = s.release();
        file->pMethods = &kMethods;
        if (outFlags) *outFlags = flags;
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_CANTOPEN;
    }
}

int vfsDelete(sqlite3_vfs*, const char* name, int)
{
    try {
        const auto url = vc::sftp::parseUrl(name);
        if (!activeHub || !belongsToHub(url, *activeHub))
            throw std::runtime_error("hub: journal removal requires the operation's locked SSH session");
        withSession(*connection(url), [&](Session& session) { session.remove(url.path); });
        return SQLITE_OK;
    } catch (const StatusError& error) {
        if (error.status == vc::sftp::NoSuchFile) return SQLITE_IOERR_DELETE_NOENT;
        lastError = error.what();
        return SQLITE_IOERR_DELETE;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_DELETE;
    }
}

int vfsAccess(sqlite3_vfs*, const char* name, int, int* result)
{
    try {
        const auto stat = remoteStat(name);
        *result = stat.exists;
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_ACCESS;
    }
}

int vfsFullPathname(sqlite3_vfs*, const char* name, int size, char* out)
{
    try {
        const auto canonical = vc::sftp::parseUrl(name).str();
        if (canonical.size() + 1 > size_t(size)) return SQLITE_CANTOPEN;
        std::memcpy(out, canonical.c_str(), canonical.size() + 1);
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_CANTOPEN;
    }
}

void* vfsDlOpen(sqlite3_vfs*, const char*) { return nullptr; }
void vfsDlError(sqlite3_vfs*, int size, char* out) { sqlite3_snprintf(size, out, "loadable extensions are unsupported"); }
void (*vfsDlSym(sqlite3_vfs*, void*, const char*))(void) { return nullptr; }
void vfsDlClose(sqlite3_vfs*, void*) {}
int vfsRandomness(sqlite3_vfs*, int size, char* out) { return defaultVfs()->xRandomness(defaultVfs(), size, out); }
int vfsSleep(sqlite3_vfs*, int microseconds) { return defaultVfs()->xSleep(defaultVfs(), microseconds); }
int vfsCurrentTime(sqlite3_vfs*, double* now) { return defaultVfs()->xCurrentTime(defaultVfs(), now); }
int vfsGetLastError(sqlite3_vfs*, int size, char* out)
{
    sqlite3_snprintf(size, out, "%s", lastError.c_str());
    return 0;
}
int vfsCurrentTimeInt64(sqlite3_vfs*, sqlite3_int64* now)
{
    auto* base = defaultVfs();
    if (base->iVersion >= 2 && base->xCurrentTimeInt64) return base->xCurrentTimeInt64(base, now);
    double days = 0;
    const int rc = base->xCurrentTime(base, &days);
    *now = sqlite3_int64(days * 86400000.0);
    return rc;
}

sqlite3_vfs vfs = {
    2, int(sizeof(File)), 4096, nullptr, kVfsName, nullptr,
    vfsOpen, vfsDelete, vfsAccess, vfsFullPathname, vfsDlOpen, vfsDlError, vfsDlSym, vfsDlClose,
    vfsRandomness, vfsSleep, vfsCurrentTime, vfsGetLastError, vfsCurrentTimeInt64,
};
}  // namespace

struct RemoteHubScope::Impl {
    std::shared_ptr<Connection> connection;
    int exceptions = std::uncaught_exceptions();
};

RemoteHubScope::RemoteHubScope(const std::filesystem::path& hub)
{
    if (!isRemotePath(hub)) return;
    if (activeHub) throw std::logic_error("hub: nested remote hub operation");
    auto state = std::make_unique<Impl>();
    const auto url = vc::sftp::parseUrl(canonicalRemotePath(hub));
#ifndef _WIN32
    // This ordinary SFTP session owns the shared transport, never a hub lock.
    // Its completed handshake makes multiplexing available before the locked
    // command starts. No background SSH daemon or user config change is needed.
    withSession(*standaloneConnection(url), [](Session&) {});
#endif
    // Each thread owns its SQLite handles. Healthy, unlocked handles can keep
    // their page caches while their SFTP handles reopen in the next generation.
    static thread_local std::map<std::string, std::weak_ptr<Connection>> connections;
    auto& cached = connections[url.str()];
    state->connection = cached.lock();
    if (!state->connection || state->connection->failed)
        state->connection = std::make_shared<Connection>();
    cached = state->connection;
    state->connection->destination = url.withPath("/");
    state->connection->hubPath = url.path;
    state->connection->permissions.reset();
    state->connection->active = true;
    // Acquire before opening SQLite, including its initial header read.
    try { state->connection->get(); }
    catch (...) {
        state->connection->active = false;
        state->connection->failed = true;
        throw;
    }
    activeHub = state->connection;
    impl_ = std::move(state);
}

RemoteHubScope::~RemoteHubScope()
{
    if (!impl_) return;
    // No transaction may survive the server lock. On failure the next scope
    // gets a different identity, so callers discard its SQLite cache as well.
    if (impl_->connection->lockedFiles || std::uncaught_exceptions() > impl_->exceptions)
        impl_->connection->failed = true;
    impl_->connection->active = false;
    impl_->connection->session.reset();
    activeHub.reset();
}

std::shared_ptr<const void> remoteHubCacheIdentity() { return activeHub; }

const char* sftpVfs()
{
    static std::once_flag registered;
    std::call_once(registered, [] {
        if (sqlite3_vfs_register(&vfs, 0) != SQLITE_OK) throw std::runtime_error("object store: cannot register the SFTP VFS");
    });
    return kVfsName;
}

// Native text: generic_string() would collapse the URL's "//".
std::string pathText(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
}
bool isRemotePath(const std::filesystem::path& path) { return vc::sftp::isSftpUrl(pathText(path)); }
std::string canonicalRemotePath(const std::filesystem::path& path) { return vc::sftp::parseUrl(pathText(path)).str(); }
std::string sftpLastError() { return lastError; }

RemoteStat remoteStat(const std::string& text)
{
    const auto url = vc::sftp::parseUrl(text);
    return withSession(*connection(url), [&](Session& session) {
        RemoteStat result;
        try {
            const auto attributes = session.stat(url.path);
            result.exists = true;
            result.directory = attributes.directory();
            result.size = attributes.size.value_or(0);
            result.modified = attributes.modified;
        } catch (const StatusError& error) {
            if (error.status != vc::sftp::NoSuchFile) throw;
        }
        return result;
    });
}

void remoteRead(const std::string& text, uint64_t offset, std::span<std::byte> out)
{
    const auto url = vc::sftp::parseUrl(text);
    withSession(*connection(url), [&](Session& session) {
        const auto handle = session.open(url.path, vc::sftp::Read);
        size_t got = 0;
        try { got = session.read(handle, offset, out); } catch (...) { session.close(handle); throw; }
        session.close(handle);
        if (got != out.size()) throw std::runtime_error("SFTP read ended early: " + text);
    });
}

uint64_t remoteCopy(const std::string& text, const std::function<void(std::span<const std::byte>)>& sink)
{
    const auto url = vc::sftp::parseUrl(text);
    // Not retried: a partial copy has already reached the sink.
    auto c = connection(url);
    std::lock_guard lock(c->mutex);
    auto& session = c->get();
    try {
        const auto handle = session.open(url.path, vc::sftp::Read);
        const auto copied = session.read(handle, 0, std::numeric_limits<uint64_t>::max(), sink);
        session.close(handle);
        return copied;
    } catch (const StatusError&) {
        throw;
    } catch (...) {
        c->session.reset();
        if (!c->hubPath.empty()) c->failed = true;
        throw;
    }
}

void remoteEnableProcessLock(const std::string& text)
{
    const auto url = vc::sftp::parseUrl(text);
    // Migration is deliberately not passed through the reconnect/retry helper.
    // An uncertain mkdir is left for the operator, never guessed to be ours.
    Session session(url);
    if (!session.supports("fsync@openssh.com"))
        throw std::runtime_error("hub: server lacks fsync@openssh.com");
    const auto database = session.stat(url.path);
    if (database.directory()) throw std::runtime_error("hub: expected a database file");
    const auto directory = url.path + ".lock";
    const auto marker = directory + "/" + kProcessLockName;
    try {
        const auto existing = session.lstat(marker);
        if (!existing.permissions || (*existing.permissions & 0170000) != 0100000)
            throw std::runtime_error("hub: invalid process lock file");
        return;
    } catch (const StatusError& error) {
        if (error.status != vc::sftp::NoSuchFile) throw;
    }
    try { session.mkdir(directory); }
    catch (const StatusError& error) {
        throw std::runtime_error("hub: cannot create the permanent lock guard; stop all clients and resolve any legacy lock first: " +
                                 std::string(error.what()));
    }
    if (database.permissions) {
        const auto mode = *database.permissions & 0666;
        session.setPathPermissions(directory, mode | ((mode & 0444) >> 2) | ((mode & 0222) >> 1));
    }
    const auto handle = session.open(marker, vc::sftp::Write | vc::sftp::Create | vc::sftp::Exclusive);
    // The lock file needs the same sharing permissions as the hub.
    if (database.permissions) session.setPermissions(handle, *database.permissions & 0666);
    session.fsync(handle);
    session.close(handle);
}

void remoteRmdir(const std::string& text)
{
    const auto url = vc::sftp::parseUrl(text);
    withSession(*connection(url), [&](Session& session) { session.rmdir(url.path); });
}

}  // namespace vc::objects::detail
