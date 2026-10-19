// Cross-process FIFO budget for decoded chunk caches. SQLite serializes the
// rename, accounting and eviction together; reads never wait on this index.
#pragma once

#include "chunk_io.hpp"
#include "http.hpp"

#include <sqlite3.h>
#include <algorithm>
#include <mutex>
#include <tuple>
#include <sys/file.h>

namespace vcr {

class DiskCache {
public:
    DiskCache(const std::string& root, std::int64_t max_bytes)
        : root_(std::filesystem::weakly_canonical(root))
    {
        if (max_bytes < -1)
            throw std::invalid_argument("invalid cache_max_bytes");
        std::filesystem::create_directories(root_);
        const auto path = (root_ / ".vc_reader_cache.sqlite3").string();
        if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                            SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK) {
            const std::string error = db_ ? sqlite3_errmsg(db_) : "cannot open disk cache index";
            sqlite3_close(db_);
            db_ = nullptr;
            throw std::runtime_error(error);
        }
        sqlite3_busy_timeout(db_, 60000);
        try {
            exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
                 "CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY, max_bytes INTEGER,"
                 " total INTEGER NOT NULL DEFAULT 0, clock INTEGER NOT NULL DEFAULT 0);"
                 "INSERT OR IGNORE INTO budget(id,max_bytes) VALUES(1,1073741824);"
                 "CREATE TABLE IF NOT EXISTS chunks(path TEXT PRIMARY KEY, size INTEGER, age INTEGER);"
                 "CREATE INDEX IF NOT EXISTS oldest_chunk ON chunks(age,path);"
                 "CREATE TRIGGER IF NOT EXISTS add_chunk AFTER INSERT ON chunks BEGIN "
                 "UPDATE budget SET total=total+new.size WHERE id=1; END;"
                 "CREATE TRIGGER IF NOT EXISTS remove_chunk AFTER DELETE ON chunks BEGIN "
                 "UPDATE budget SET total=total-old.size WHERE id=1; END;"
                 "CREATE TRIGGER IF NOT EXISTS resize_chunk AFTER UPDATE OF size ON chunks BEGIN "
                 "UPDATE budget SET total=total+new.size-old.size WHERE id=1; END;");
            exec("CREATE TABLE IF NOT EXISTS file_headers(path TEXT PRIMARY KEY, etag TEXT, modified TEXT);");
            exec("CREATE TABLE IF NOT EXISTS downloads(path TEXT PRIMARY KEY, size INTEGER NOT NULL);");
            transaction([&] {
                // Reconcile after a crash or writes by an unbounded reader.
                // Only numeric chunk files in the decoded-cache namespaces
                // count. Metadata, markers, VC3D caches and other files stay.
                exec("DELETE FROM chunks; UPDATE budget SET total=0,clock=0 WHERE id=1;");
                using Entry = std::tuple<std::filesystem::file_time_type, std::string, std::int64_t>;
                std::vector<Entry> files;
                for (const auto* scheme : {"http", "https", "s3"}) {
                    const auto base = root_ / scheme;
                    if (!std::filesystem::is_directory(base) || std::filesystem::is_symlink(base))
                        continue;
                    scan(base, files);
                }
                const auto encoded = root_ / "encoded";
                if (std::filesystem::is_directory(encoded)) {
                    for (const auto& file : std::filesystem::directory_iterator(encoded))
                        if (file.is_regular_file() && !file.is_symlink() && file.path().extension() == ".blob")
                            files.emplace_back(file.last_write_time(), file.path().string(), file.file_size());
                }
                if (std::filesystem::is_directory(encoded)) {
                    for (const auto& file : std::filesystem::directory_iterator(encoded))
                        if (file.is_regular_file() && file.path().filename().string().find(".blob.tmp.") != std::string::npos)
                            remove_abandoned(file.path().string());
                }
                clean_downloads();
                std::sort(files.begin(), files.end());
                for (const auto& [time, file, size] : files)
                    record(file, size);
                exec("DELETE FROM file_headers WHERE path NOT IN (SELECT path FROM chunks);");
                if (max_bytes >= 0) set_limit(max_bytes);
                evict();
            });
        } catch (...) {
            sqlite3_close(db_);
            db_ = nullptr;
            throw;
        }
    }

    ~DiskCache() { sqlite3_close(db_); }
    DiskCache(const DiskCache&) = delete;
    DiskCache& operator=(const DiskCache&) = delete;

    void set_max_bytes(std::int64_t bytes)
    {
        if (bytes < 0)
            throw std::invalid_argument("cache_max_bytes must be nonnegative");
        std::lock_guard lock(mutex_);
        transaction([&] { set_limit(bytes); evict(); });
    }

    void write(const std::string& path, const std::uint8_t* data, std::size_t size)
    {
        check_path(path);
        std::lock_guard lock(mutex_);
        transaction([&] {
            const auto available = number("SELECT max_bytes FROM budget WHERE id=1") - reserved_bytes();
            if (size > static_cast<std::uint64_t>(available))
                return;  // A chunk larger than the budget is served without caching.
            evict_to(available - static_cast<std::int64_t>(size));
            atomic_write(path, data, size);
            record(path, static_cast<std::int64_t>(size));
            evict();
        });
    }

    std::int64_t max_bytes()
    {
        std::lock_guard lock(mutex_);
        return number("SELECT max_bytes FROM budget WHERE id=1");
    }

    // Pin the inode before releasing the cross-process publication lock. A
    // later eviction unlinks it without invalidating an active decoder's FD.
    std::tuple<int, std::string, std::string, std::size_t> open_file(
        const std::string& path, const std::string& url, const Auth& auth, bool offline, bool refresh)
    {
        check_path(path);
        if (std::filesystem::path(path).parent_path() != root_ / "encoded")
            throw std::invalid_argument("encoded files must use the encoded cache namespace");
        // Only callers for the same object wait through its network request.
        // Keep the lock file: unlinking it could split concurrent lock owners.
        std::filesystem::create_directories(root_ / "encoded");
        File object_lock(::open((path + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644));
        if (!object_lock) throw std::system_error(errno, std::generic_category(), "open cache file lock");
        lock_file(object_lock.fd());
        int fd = -1;
        std::string etag, modified;
        std::size_t bytes = 0;
        std::unique_ptr<AtomicFile> output;
        std::string temporary;
        const auto cleanup = [&] {
            std::lock_guard lock(mutex_);
            transaction([&] {
                // Delete/truncate payload before making its reservation reusable.
                output.reset();
                forget_download(temporary);
            });
        };
        try {
            {
                std::lock_guard lock(mutex_);
                transaction([&] {
                    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd < 0 && errno != ENOENT)
                        throw std::system_error(errno, std::generic_category(), "open " + path);
                    if (fd >= 0) {
                        Statement h(db_, "SELECT etag,modified FROM file_headers WHERE path=?");
                        h.text(1, path);
                        if (h.step()) {
                            etag = reinterpret_cast<const char*>(sqlite3_column_text(h.get(), 0));
                            modified = reinterpret_cast<const char*>(sqlite3_column_text(h.get(), 1));
                        }
                    }
                    if (!offline && (fd < 0 || refresh)) {
                        output = std::make_unique<AtomicFile>(path);
                        temporary = output->temporary_path();
                        lock_file(output->fd());
                        record_download(temporary, 0);
                    }
                });
            }
            if (output) {
                HttpOptions options;
                options.connections = 1;
                options.auth = auth;
                HttpClient client(options);
                std::size_t received = 0, reserved = 0;
                std::vector<std::pair<std::string, std::string>> headers;
                if (fd >= 0 && !etag.empty()) headers.emplace_back("If-None-Match", etag);
                else if (fd >= 0 && !modified.empty()) headers.emplace_back("If-Modified-Since", modified);
                auto response = client.get_stream(url, [&] {
                    std::lock_guard lock(mutex_);
                    transaction([&] { output->reset(); record_download(temporary, 0); });
                    received = reserved = 0;
                }, [&](const char* data, std::size_t n) {
                    if (n > reserved - received)
                        throw std::runtime_error("encoded file exceeds cache_max_bytes or declared Content-Length");
                    output->write(reinterpret_cast<const std::uint8_t*>(data), n);
                    received += n;
                }, headers, 200, [&](const HttpResponse& header) {
                    // Content-Length describes wire bytes. Compressed or
                    // unsized responses reserve the whole budget before writing.
                    const auto size = header.content_encoding.empty() || header.content_encoding == "identity"
                        ? header.content_length : std::nullopt;
                    reserved = reserve_download(temporary, size);
                });
                bytes = response.bytes;
                {
                    std::lock_guard lock(mutex_);
                    transaction([&] {
                        if (response.status == 200) {
                            output->publish();
                            forget_download(temporary);
                            record(path, static_cast<std::int64_t>(received));
                            etag = response.etag;
                            modified = response.modified;
                            Statement h(db_, "INSERT INTO file_headers(path,etag,modified) VALUES(?,?,?) "
                                             "ON CONFLICT(path) DO UPDATE SET etag=excluded.etag,modified=excluded.modified");
                            h.text(1, path); h.text(2, etag); h.text(3, modified); h.step();
                            if (fd >= 0) ::close(fd);
                            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                            if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path);
                        } else if (response.status == 404) {
                            if (fd >= 0) ::close(fd);
                            fd = -1;
                        }
                        // A 304 uses the inode pinned before the request, even if
                        // another download evicted its pathname in the meantime.
                        output.reset();
                        forget_download(temporary);
                    });
                }
            }
        } catch (...) {
            if (fd >= 0) ::close(fd);
            try { cleanup(); } catch (...) {} // Preserve the original I/O error; startup reclaims abandoned reservations.
            throw;
        }
        return {fd, etag, modified, bytes};
    }

private:
    class Statement {
    public:
        Statement(sqlite3* db, const char* sql) : db_(db)
        {
            if (sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db_));
        }
        ~Statement() { sqlite3_finalize(stmt_); }
        void text(int index, const std::string& value)
        {
            if (sqlite3_bind_text(stmt_, index, value.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db_));
        }
        void integer(int index, std::int64_t value) { sqlite3_bind_int64(stmt_, index, value); }
        bool step()
        {
            const int rc = sqlite3_step(stmt_);
            if (rc != SQLITE_ROW && rc != SQLITE_DONE)
                throw std::runtime_error(sqlite3_errmsg(db_));
            return rc == SQLITE_ROW;
        }
        sqlite3_stmt* get() { return stmt_; }
    private:
        sqlite3* db_;
        sqlite3_stmt* stmt_ = nullptr;
    };

    void exec(const char* sql)
    {
        if (sqlite3_exec(db_, sql, nullptr, nullptr, nullptr) != SQLITE_OK)
            throw std::runtime_error(sqlite3_errmsg(db_));
    }

    template<class F> void transaction(F work)
    {
        exec("BEGIN IMMEDIATE");
        try {
            work();
            exec("COMMIT");
        } catch (...) {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
    }

    std::int64_t number(const char* sql)
    {
        Statement s(db_, sql);
        s.step();
        return sqlite3_column_int64(s.get(), 0);
    }

    void set_limit(std::int64_t bytes)
    {
        clean_downloads();
        if (bytes < reserved_bytes())
            throw std::runtime_error("cache_max_bytes is below active download reservations");
        Statement s(db_, "UPDATE budget SET max_bytes=? WHERE id=1");
        s.integer(1, bytes);
        s.step();
    }

    void record(const std::string& path, std::int64_t size)
    {
        exec("UPDATE budget SET clock=clock+1 WHERE id=1");
        Statement s(db_, "INSERT INTO chunks(path,size,age) VALUES(?,?,(SELECT clock FROM budget WHERE id=1)) "
                         "ON CONFLICT(path) DO UPDATE SET size=excluded.size,age=excluded.age");
        s.text(1, path);
        s.integer(2, size);
        s.step();
    }

    void evict() { evict_to(number("SELECT max_bytes FROM budget WHERE id=1") - reserved_bytes()); }

    static void lock_file(int fd)
    {
        while (::flock(fd, LOCK_EX) != 0)
            if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "lock cache file");
    }

    // Called under the publication transaction, so no live writer can register
    // a new temporary file between opening it and checking its lifetime lock.
    static bool remove_abandoned(const std::string& path)
    {
        auto file = File::open_read(path);
        if (!file) return true;
        if (::flock(file.fd(), LOCK_EX | LOCK_NB) != 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) return false;
            throw std::system_error(errno, std::generic_category(), "check download lock");
        }
        std::filesystem::remove(path);
        return true;
    }

    void forget_download(const std::string& path)
    {
        Statement s(db_, "DELETE FROM downloads WHERE path=?");
        s.text(1, path); s.step();
    }

    void record_download(const std::string& path, std::size_t size)
    {
        Statement s(db_, "INSERT INTO downloads(path,size) VALUES(?,?) "
                         "ON CONFLICT(path) DO UPDATE SET size=excluded.size");
        s.text(1, path); s.integer(2, static_cast<std::int64_t>(size)); s.step();
    }

    std::int64_t reserved_bytes() { return number("SELECT COALESCE(SUM(size),0) FROM downloads"); }

    void clean_downloads()
    {
        std::vector<std::string> paths;
        {
            Statement s(db_, "SELECT path FROM downloads");
            while (s.step()) paths.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(s.get(), 0)));
        }
        for (const auto& path : paths) {
            check_path(path);
            if (remove_abandoned(path)) forget_download(path);
        }
    }

    std::size_t reserve_download(const std::string& path, std::optional<std::size_t> size)
    {
        for (;;) {
            {
                std::lock_guard lock(mutex_);
                bool admitted = false;
                std::size_t needed = 0;
                transaction([&] {
                    const auto limit = number("SELECT max_bytes FROM budget WHERE id=1");
                    needed = size.value_or(static_cast<std::size_t>(limit));
                    if (needed > static_cast<std::uint64_t>(limit))
                        throw std::runtime_error("encoded file exceeds cache_max_bytes; increase the disk limit");
                    auto reserved = reserved_bytes();
                    if (needed > static_cast<std::uint64_t>(limit - reserved)) {
                        clean_downloads();
                        reserved = reserved_bytes();
                    }
                    if (needed <= static_cast<std::uint64_t>(limit - reserved)) {
                        evict_to(limit - reserved - static_cast<std::int64_t>(needed));
                        record_download(path, needed);
                        admitted = true;
                    }
                });
                if (admitted) return needed;
            }
            // Cross-process capacity wait; neither the mutex nor SQLite writer
            // lock is held while active transfers finish and publish their files.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void evict_to(std::int64_t limit)
    {
        while (number("SELECT total FROM budget WHERE id=1") > limit) {
            std::string path;
            {
                Statement s(db_, "SELECT path FROM chunks ORDER BY age,path LIMIT 1");
                if (!s.step())
                    throw std::runtime_error("disk cache accounting is inconsistent");
                path = reinterpret_cast<const char*>(sqlite3_column_text(s.get(), 0));
            }
            check_path(path);
            std::filesystem::remove(path);  // unlink, never truncate a mapped file
            Statement s(db_, "DELETE FROM chunks WHERE path=?");
            s.text(1, path);
            s.step();
            Statement h(db_, "DELETE FROM file_headers WHERE path=?");
            h.text(1, path); h.step();
        }
    }

    void check_path(const std::string& path)
    {
        const auto rel = std::filesystem::path(path).lexically_relative(root_);
        if (rel.empty() || rel.is_absolute() || *rel.begin() == "..")
            throw std::runtime_error("disk cache entry outside its root: " + path);
    }

    template<class Entries> void scan(const std::filesystem::path& base, Entries& files)
    {
        namespace fs = std::filesystem;
        for (auto it = fs::recursive_directory_iterator(base); it != fs::recursive_directory_iterator(); ++it) {
            if (it->is_symlink()) {
                it.disable_recursion_pending();
                continue;
            }
            if (it->path().filename() != ".zarray" || !it->is_regular_file())
                continue;
            const auto array = it->path().parent_path();
            for (auto chunk = fs::recursive_directory_iterator(array); chunk != fs::recursive_directory_iterator(); ++chunk) {
                if (chunk->is_symlink()) {
                    chunk.disable_recursion_pending();
                    continue;
                }
                if (!chunk->is_regular_file())
                    continue;
                const auto key = chunk->path().lexically_relative(array).generic_string();
                if (key.empty() || key.front() < '0' || key.front() > '9' ||
                    key.find_first_not_of("0123456789./") != std::string::npos)
                    continue;
                files.emplace_back(chunk->last_write_time(), chunk->path().string(), chunk->file_size());
            }
        }
    }

    std::filesystem::path root_;
    sqlite3* db_ = nullptr;
    std::mutex mutex_;
};

}  // namespace vcr
