// Cross-process FIFO budget for decoded chunk caches. SQLite serializes the
// rename, accounting and eviction together; reads never wait on this index.
#pragma once

#include "chunk_io.hpp"
#include "http.hpp"

#include <atomic>
#include <iostream>
#include <sqlite3.h>
#include <algorithm>
#include <limits>
#include <mutex>
#include <tuple>
#include <sys/file.h>

namespace vcr {

class DiskCache {
public:
    // max_bytes -1 keeps the limit stored in the root, unlimited for a new root;
    // anything else replaces it. Only a limit ever evicts.
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
                 "INSERT OR IGNORE INTO budget(id,max_bytes) VALUES(1,9223372036854775807);"
                 "CREATE TABLE IF NOT EXISTS chunks(path TEXT PRIMARY KEY, size INTEGER, age INTEGER);"
                 "CREATE INDEX IF NOT EXISTS oldest_chunk ON chunks(age,path);"
                 "CREATE TRIGGER IF NOT EXISTS add_chunk AFTER INSERT ON chunks BEGIN "
                 "UPDATE budget SET total=total+new.size WHERE id=1; END;"
                 "CREATE TRIGGER IF NOT EXISTS remove_chunk AFTER DELETE ON chunks BEGIN "
                 "UPDATE budget SET total=total-old.size WHERE id=1; END;"
                 "CREATE TRIGGER IF NOT EXISTS resize_chunk AFTER UPDATE OF size ON chunks BEGIN "
                 "UPDATE budget SET total=total+new.size-old.size WHERE id=1; END;");
            // Remote files this budget downloaded (open_file); files found in place are not owned.
            exec("DROP TABLE IF EXISTS file_headers;"
                 "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY);");
            exec("CREATE TABLE IF NOT EXISTS adopted_namespaces(path TEXT PRIMARY KEY);");
            exec("CREATE TABLE IF NOT EXISTS downloads(path TEXT PRIMARY KEY, size INTEGER NOT NULL);");
            transaction([&] {
                clean_downloads();
                // Budgets used to be seeded at 1 GiB, which nobody chose and which evicted the
                // cache of whoever opened the root without a limit. Once, per index: lift it.
                if (number("PRAGMA user_version") == 0)
                    exec("UPDATE budget SET max_bytes=9223372036854775807 WHERE id=1 AND max_bytes=1073741824;"
                         "PRAGMA user_version=1");
                if (max_bytes >= 0) set_limit(max_bytes);
                // Without a byte limit nothing reads the index: a walk of a large root on
                // every open would only keep an unused total right.
                if (number("SELECT max_bytes FROM budget WHERE id=1") != std::numeric_limits<std::int64_t>::max())
                    reconcile();
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

    // Adopt existing array chunks under a cache namespace once, preserving metadata.
    // Persist its namespace so crash reconciliation also counts these chunks.
    void adopt_chunks(const std::string& path)
    {
        const auto canonical=std::filesystem::weakly_canonical(path).string();
        check_path(canonical);
        std::lock_guard lock(mutex_);
        transaction([&] {
            Statement known(db_, "SELECT 1 FROM adopted_namespaces WHERE path=?");
            known.text(1,canonical);
            if (known.step()) return;
            Statement adopted(db_, "INSERT INTO adopted_namespaces(path) VALUES(?)");
            adopted.text(1,canonical); adopted.step();
            if (!reconciled_) return;  // no limit: the walk waits until one needs the index
            using Entry=std::tuple<std::filesystem::file_time_type,std::string,std::int64_t>;
            std::vector<Entry> files;
            scan(canonical,files);
            std::sort(files.begin(),files.end());
            for (const auto& [time,file,size] : files) record(file,size);
            evict();
        });
    }

    void set_max_bytes(std::int64_t bytes)
    {
        if (bytes < 0)
            throw std::invalid_argument("cache_max_bytes must be nonnegative");
        std::lock_guard lock(mutex_);
        transaction([&] {
            set_limit(bytes);
            if (!reconciled_ && bytes != std::numeric_limits<std::int64_t>::max()) reconcile();
            evict();
        });
    }

    // Keep this much of the disk free: a write that would leave less is skipped
    // (served without caching) and nothing cached is removed. The first skip warns
    // on stderr; low_space() reports the current state.
    void set_min_free_bytes(std::int64_t bytes)
    {
        if (bytes < 0)
            throw std::invalid_argument("cache_min_free_bytes must be nonnegative");
        min_free_bytes_.store(bytes, std::memory_order_relaxed);
    }

    void write(const std::string& path, const std::uint8_t* data, std::size_t size)
    {
        check_path(path);
        std::lock_guard lock(mutex_);
        transaction([&] {
            const auto available = number("SELECT max_bytes FROM budget WHERE id=1") - reserved_bytes();
            if (size > static_cast<std::uint64_t>(available))
                return;  // A chunk larger than the budget is served without caching.
            if (!leaves_free_space(size))
                return;
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

    // Whether the last write was skipped to keep the disk's free space.
    bool low_space() const noexcept { return low_space_.load(std::memory_order_relaxed); }

    // The bytes this budget accounts for.
    std::int64_t total_bytes()
    {
        std::lock_guard lock(mutex_);
        if (!reconciled_) transaction([&] { reconcile(); });
        return number("SELECT total FROM budget WHERE id=1");
    }

    // A remote file cached at `path` (its URL's decoded-cache path, so mirrors
    // laid out by bucket path are found in place). A cached copy is trusted like
    // a chunk: it is never revalidated. Missing files are downloaded into the
    // budget; files found in place are used, not owned, so never evicted.
    // Returns the open descriptor (-1 if the remote file does not exist) and the
    // bytes received. Eviction unlinks without invalidating the descriptor.
    std::pair<int, std::size_t> open_file(const std::string& path, const std::string& url, const Auth& auth)
    {
        check_path(path);
        if (int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC); fd >= 0 || errno != ENOENT) {
            if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path);
            struct stat st;
            if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
                ::close(fd);
                throw std::runtime_error("cached file is not a regular file: " + path);
            }
            return {fd, 0};
        }
        // Only callers for the same object wait through its network request.
        // Keep the lock file: unlinking it could split concurrent lock owners.
        const auto locks = root_ / ".vc_reader_locks";
        std::filesystem::create_directories(locks);
        const auto lock_path = locks / (std::to_string(std::hash<std::string>{}(path)) + ".lock");
        File object_lock(::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644));
        if (!object_lock) throw std::system_error(errno, std::generic_category(), "open cache file lock");
        lock_file(object_lock.fd());
        if (int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC); fd >= 0)
            return {fd, 0};  // another owner published it while we waited
        int fd = -1;
        std::unique_ptr<AtomicFile> output;
        std::string temporary;
        try {
            {
                std::lock_guard lock(mutex_);
                transaction([&] {
                    output = std::make_unique<AtomicFile>(path);
                    temporary = output->temporary_path();
                    lock_file(output->fd());
                    record_download(temporary, 0);
                });
            }
            HttpOptions options;
            options.connections = 1;
            options.auth = auth;
            HttpClient client(options);
            std::size_t received = 0, reserved = 0;
            bool grows = false;
            File unsized_turn;
            auto response = client.get_stream(url, [&] {
                std::lock_guard lock(mutex_);
                transaction([&] { output->reset(); record_download(temporary, 0); });
                received = reserved = 0;
            }, [&](const char* data, std::size_t n) {
                if (n > reserved - received) {
                    if (!grows)
                        throw std::runtime_error("encoded file exceeds cache_max_bytes or declared Content-Length");
                    reserved = grow_download(temporary, received + n);
                }
                output->write(reinterpret_cast<const std::uint8_t*>(data), n);
                received += n;
            }, {}, 200, [&](const HttpResponse& header) {
                // Content-Length describes wire bytes. Compressed or unsized responses
                // reserve as their bytes arrive, one at a time on a limited budget: two
                // each holding part of it could otherwise wait for each other forever.
                // Sized downloads never grow, so they always finish and free what they hold.
                const auto size = header.content_encoding.empty() || header.content_encoding == "identity"
                    ? header.content_length : std::nullopt;
                grows = !size;
                if (grows && unsized_turn.fd() < 0 && max_bytes() != std::numeric_limits<std::int64_t>::max()) {
                    unsized_turn = File(::open((locks / "unsized.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644));
                    if (unsized_turn.fd() < 0) throw std::system_error(errno, std::generic_category(), "open cache file lock");
                    lock_file(unsized_turn.fd());
                }
                reserved = size ? reserve_download(temporary, *size) : 0;
            });
            std::lock_guard lock(mutex_);
            transaction([&] {
                if (response.status == 200) {
                    output->publish();
                    forget_download(temporary);
                    record(path, static_cast<std::int64_t>(received));
                    Statement f(db_, "INSERT OR IGNORE INTO files(path) VALUES(?)");
                    f.text(1, path); f.step();
                    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path);
                }
                output.reset();
                forget_download(temporary);
            });
            return {fd, response.bytes};
        } catch (...) {
            if (fd >= 0) ::close(fd);
            try {
                std::lock_guard lock(mutex_);
                // Delete/truncate payload before making its reservation reusable.
                transaction([&] { output.reset(); forget_download(temporary); });
            } catch (...) {} // Preserve the original I/O error; startup reclaims abandoned reservations.
            throw;
        }
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

    std::size_t reserve_download(const std::string& path, std::size_t needed)
    {
        for (;;) {
            {
                std::lock_guard lock(mutex_);
                bool admitted = false;
                transaction([&] {
                    const auto limit = number("SELECT max_bytes FROM budget WHERE id=1");
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

    // Raises a growing download's reservation to at least `needed` bytes, evicting only what
    // that takes. Spare room is reserved too, so a free budget is not asked on every write.
    std::size_t grow_download(const std::string& path, std::size_t needed)
    {
        constexpr std::int64_t step = 16 << 20;
        for (;;) {
            {
                std::lock_guard lock(mutex_);
                std::size_t granted = 0;
                transaction([&] {
                    const auto limit = number("SELECT max_bytes FROM budget WHERE id=1");
                    if (needed > static_cast<std::uint64_t>(limit))
                        throw std::runtime_error("encoded file exceeds cache_max_bytes; increase the disk limit");
                    std::int64_t own = 0;
                    {
                        Statement s(db_, "SELECT COALESCE(SUM(size),0) FROM downloads WHERE path=?");
                        s.text(1, path);
                        s.step();
                        own = sqlite3_column_int64(s.get(), 0);
                    }
                    auto others = reserved_bytes() - own;
                    const auto wanted = static_cast<std::int64_t>(needed);
                    if (wanted > limit - others) {
                        clean_downloads();
                        others = reserved_bytes() - own;
                        if (wanted > limit - others) return;  // sized downloads hold it; they finish
                    }
                    const auto spare = limit - others - number("SELECT total FROM budget WHERE id=1") - wanted;
                    const auto target = wanted + std::clamp<std::int64_t>(spare, 0, step);
                    evict_to(limit - others - target);
                    record_download(path, static_cast<std::size_t>(target));
                    granted = static_cast<std::size_t>(target);
                });
                if (granted) return granted;
            }
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
            Statement f(db_, "DELETE FROM files WHERE path=?");
            f.text(1, path); f.step();
        }
    }

    bool leaves_free_space(std::size_t size)
    {
        const auto wanted = static_cast<std::uintmax_t>(min_free_bytes_.load(std::memory_order_relaxed));
        if (wanted == 0)
            return true;
        const bool enough = std::filesystem::space(root_).available >= wanted + size;
        if (!enough && !low_space_warned_.exchange(true, std::memory_order_relaxed))
            std::cerr << "vc_reader: under " << wanted << " bytes free on the disk of " << root_.string()
                      << "; no longer caching chunks there (reads continue from the network)\n";
        low_space_.store(!enough, std::memory_order_relaxed);
        return enough;
    }

    void check_path(const std::string& path)
    {
        const auto rel = std::filesystem::path(path).lexically_relative(root_);
        if (rel.empty() || rel.is_absolute() || *rel.begin() == "..")
            throw std::runtime_error("disk cache entry outside its root: " + path);
    }

    // Rebuild the index from the files, after a crash or writes by an unbounded reader.
    // Only numeric chunk files in the decoded-cache namespaces count, plus explicitly
    // adopted VC3D arrays. Metadata and markers stay.
    void reconcile()
    {
        exec("DELETE FROM chunks; UPDATE budget SET total=0,clock=0 WHERE id=1;");
        using Entry = std::tuple<std::filesystem::file_time_type, std::string, std::int64_t>;
        std::vector<Entry> files;
        for (const auto* scheme : {"http", "https", "s3"}) {
            const auto base = root_ / scheme;
            if (!std::filesystem::is_directory(base) || std::filesystem::is_symlink(base))
                continue;
            scan(base, files);
        }
        {
            Statement adopted(db_, "SELECT path FROM adopted_namespaces");
            while (adopted.step()) {
                const std::string path=reinterpret_cast<const char*>(sqlite3_column_text(adopted.get(),0));
                check_path(path);
                if (std::filesystem::is_directory(path) && !std::filesystem::is_symlink(path))
                    scan(path,files);
            }
        }
        {
            Statement owned(db_, "SELECT path FROM files");
            while (owned.step()) {
                const std::filesystem::path path = reinterpret_cast<const char*>(sqlite3_column_text(owned.get(), 0));
                check_path(path.string());
                std::error_code ec;
                if (std::filesystem::is_regular_file(std::filesystem::symlink_status(path, ec)))
                    files.emplace_back(std::filesystem::last_write_time(path), path.string(),
                                       std::filesystem::file_size(path));
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& [time, file, size] : files)
            record(file, size);
        exec("DELETE FROM files WHERE path NOT IN (SELECT path FROM chunks);");
        evict();
        reconciled_ = true;
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
    bool reconciled_ = false;
    std::atomic<std::int64_t> min_free_bytes_{0};
    std::atomic<bool> low_space_{false}, low_space_warned_{false};
};

}  // namespace vcr
