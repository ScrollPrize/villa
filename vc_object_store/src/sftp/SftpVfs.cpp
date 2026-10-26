#include "sftp/SftpVfs.hpp"
#include "vc_object_store/sftp/Sftp.hpp"

#include <sqlite3.h>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

namespace vc::objects::detail {
namespace {
using vc::sftp::Session;
using vc::sftp::StatusError;
constexpr const char* kVfsName = "vc-sftp";

thread_local std::string lastError;

// One ssh connection per destination. A broken connection reconnects on next
// use; open files reopen their handles when the generation changes.
struct Connection {
    std::mutex mutex;
    vc::sftp::Url destination;
    std::unique_ptr<Session> session;
    uint64_t generation = 0;
    Session& get()
    {
        if (!session) {
            vc::sftp::Options options;
            // Other clients hold the hub lock for whole syncs; a stalled server
            // should still fail a request eventually.
            options.idleTimeout = std::chrono::seconds(120);
            session = std::make_unique<Session>(destination, options);
            ++generation;
        }
        return *session;
    }
};

std::shared_ptr<Connection> connection(const vc::sftp::Url& url)
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
            if (attempt > 0 || lostWrites) throw;
        }
    }
}

struct State {
    std::shared_ptr<Connection> connection;
    std::string path, lockPath;
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
        withSession(*s->connection, [&](Session& session) {
            session.flush();
            if (s->generation == s->connection->generation) session.close(s->handle);
            if (s->deleteOnClose) session.remove(s->path);
        });
    } catch (const std::exception& error) {
        lastError = error.what();
        rc = SQLITE_IOERR_CLOSE;
    }
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
        if (!attributes.size) throw std::runtime_error("SFTP server reported no file size");
        *size = sqlite3_int64(*attributes.size);
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_FSTAT;
    }
}

// The unix-dotfile protocol: any lock is the directory `<file>.lock`, created
// atomically by mkdir; levels above the first change only local state.
int fileLock(sqlite3_file* file, int level)
{
    auto& s = state(file);
    if (s.lock > SQLITE_LOCK_NONE) {
        s.lock = level;
        return SQLITE_OK;
    }
    try {
        return withSession(*s.connection, [&](Session& session) {
            for (int attempt = 0;; ++attempt) {
                try {
                    session.mkdir(s.lockPath);
                    break;
                } catch (const StatusError& failed) {
                    try {
                        session.stat(s.lockPath);
                        return SQLITE_BUSY;  // another client holds the lock
                    } catch (const StatusError& missing) {
                        // A lock released since mkdir is retried; mkdir failing
                        // with no lock present is the real error.
                        if (missing.status != vc::sftp::NoSuchFile || attempt == 2) throw failed;
                    }
                }
            }
            s.lock = level;
            return SQLITE_OK;
        });
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_LOCK;
    }
}

int fileUnlock(sqlite3_file* file, int level)
{
    auto& s = state(file);
    if (s.lock == level) return SQLITE_OK;
    if (level == SQLITE_LOCK_SHARED) {
        s.lock = level;
        return SQLITE_OK;
    }
    int rc = SQLITE_OK;
    try {
        // Every write reaches the server before others may take the lock.
        withSession(*s.connection, [&](Session& session) { session.flush(); });
    } catch (const std::exception& error) {
        lastError = error.what();
        rc = SQLITE_IOERR_UNLOCK;
    }
    try {
        withSession(*s.connection, [&](Session& session) {
            try { session.rmdir(s.lockPath); }
            catch (const StatusError& error) { if (error.status != vc::sftp::NoSuchFile) throw; }
        });
    } catch (const std::exception& error) {
        lastError = error.what();
        rc = SQLITE_IOERR_UNLOCK;
    }
    s.lock = SQLITE_LOCK_NONE;
    return rc;
}

int fileCheckReservedLock(sqlite3_file* file, int* reserved)
{
    auto& s = state(file);
    if (s.lock > SQLITE_LOCK_SHARED) { *reserved = 1; return SQLITE_OK; }
    try {
        *reserved = withSession(*s.connection, [&](Session& session) {
            try { session.stat(s.lockPath); return 1; }
            catch (const StatusError& error) { if (error.status == vc::sftp::NoSuchFile) return 0; throw; }
        });
        return SQLITE_OK;
    } catch (const std::exception& error) {
        lastError = error.what();
        return SQLITE_IOERR_CHECKRESERVEDLOCK;
    }
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
        auto s = std::make_unique<State>();
        s->connection = connection(url);
        s->path = url.path;
        s->lockPath = url.path + ".lock";
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
                const auto database = session.stat(s->path.substr(0, s->path.size() - suffix.size()));
                if (database.permissions) session.setPermissions(s->handle, *database.permissions & 07777);
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
        throw;
    }
}

void remoteRmdir(const std::string& text)
{
    const auto url = vc::sftp::parseUrl(text);
    withSession(*connection(url), [&](Session& session) { session.rmdir(url.path); });
}

}  // namespace vc::objects::detail
