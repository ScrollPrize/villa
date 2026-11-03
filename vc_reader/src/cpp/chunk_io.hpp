// Box geometry, copies, file reads (pread slabs, cached mmaps) and atomic writes.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vcr {

using Index = std::vector<std::int64_t>;

// Byte strides of a C-order array.
inline Index c_strides(const Index& shape, std::size_t itemsize)
{
    Index s(shape.size());
    std::int64_t acc = static_cast<std::int64_t>(itemsize);
    for (std::size_t d = shape.size(); d-- > 0;) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

// Byte offset of element `lo` in an array with byte strides `strides`.
inline std::int64_t offset_of(const Index& lo, const Index& strides)
{
    std::int64_t off = 0;
    for (std::size_t i = 0; i < lo.size(); ++i)
        off += lo[i] * strides[i];
    return off;
}

// Copies a box of `extent` elements. `s` and `d` point at the box's first
// element in the source and destination, which have byte strides `ss` and `ds`.
// Each innermost row is one memcpy. Every extent must be positive.
inline void copy_box(const std::uint8_t* s, const Index& ss, std::uint8_t* d, const Index& ds, const Index& extent,
                     std::size_t itemsize)
{
    std::size_t nd = extent.size();
    std::size_t run = static_cast<std::size_t>(extent[nd - 1]) * itemsize;
    // Fold axes only when both arrays are contiguous across their rows.
    // Whole chunks then use one memcpy instead of thousands of short copies.
    while (nd > 1 && ss[nd - 2] == static_cast<std::int64_t>(run) &&
                     ds[nd - 2] == static_cast<std::int64_t>(run)) {
        --nd;
        run *= static_cast<std::size_t>(extent[nd - 1]);
    }
    if (nd == 1) {
        std::memcpy(d, s, run);
        return;
    }
    if (nd <= 3) {
        const std::int64_t planes = nd == 3 ? extent[0] : 1;
        const auto rows = [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
            for (std::int64_t z = 0; z < planes; ++z) {
                const auto* src = s + (nd == 3 ? z * ss[0] : 0);
                auto* dst = d + (nd == 3 ? z * ds[0] : 0);
                for (std::int64_t y = 0; y < extent[nd - 2]; ++y)
                    std::memcpy(dst + y * ds[nd - 2], src + y * ss[nd - 2], N ? N : run);
            }
        };
        // Short rows (sub-chunk blocks) are inlined fixed-size copies, not library calls.
        switch (run) {
        case 16: rows(std::integral_constant<std::size_t, 16>{}); break;
        case 32: rows(std::integral_constant<std::size_t, 32>{}); break;
        case 64: rows(std::integral_constant<std::size_t, 64>{}); break;
        case 128: rows(std::integral_constant<std::size_t, 128>{}); break;
        default: rows(std::integral_constant<std::size_t, 0>{});
        }
        return;
    }
    Index pos(nd - 1, 0);  // position over every axis but the last
    for (;;) {
        std::memcpy(d, s, run);
        std::size_t ax = nd - 1;
        for (;;) {
            if (ax-- == 0)
                return;
            if (++pos[ax] < extent[ax]) {
                s += ss[ax];
                d += ds[ax];
                break;
            }
            s -= (extent[ax] - 1) * ss[ax];
            d -= (extent[ax] - 1) * ds[ax];
            pos[ax] = 0;
        }
    }
}

// Writes `count` copies of the element `fill` from d on.
inline void fill_elements(std::uint8_t* d, std::size_t count, const std::vector<std::uint8_t>& fill)
{
    const std::size_t total = count * fill.size();
    if (total == 0)
        return;
    std::memcpy(d, fill.data(), fill.size());
    for (std::size_t done = fill.size(); done < total;) {  // double the filled prefix
        const std::size_t n = std::min(done, total - done);
        std::memcpy(d + done, d, n);
        done += n;
    }
}

// Fills a box of `extent` elements starting at `d` (strides `ds`) with the fill pattern.
inline void fill_box(std::uint8_t* d, const Index& ds, const Index& extent, const std::vector<std::uint8_t>& fill)
{
    const std::size_t nd = extent.size();
    const std::size_t itemsize = fill.size();
    std::vector<std::uint8_t> row(static_cast<std::size_t>(extent[nd - 1]) * itemsize);
    fill_elements(row.data(), static_cast<std::size_t>(extent[nd - 1]), fill);
    Index ss(nd, 0);  // every row of the "source" is the same row
    ss[nd - 1] = static_cast<std::int64_t>(itemsize);
    copy_box(row.data(), ss, d, ds, extent, itemsize);
}

class File {
public:
    File() = default;
    explicit File(int fd) : fd_(fd) {}
    File(File&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    File& operator=(File&& o) noexcept { std::swap(fd_, o.fd_); return *this; }
    ~File() { if (fd_ >= 0) ::close(fd_); }
    int fd() const noexcept { return fd_; }
    explicit operator bool() const noexcept { return fd_ >= 0; }

    // Opens for reading; an empty File if it does not exist.
    static File open_read(const std::string& path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0 && errno != ENOENT && errno != ENOTDIR)
            throw std::system_error(errno, std::generic_category(), "open " + path);
        return File(fd);
    }

private:
    int fd_ = -1;
};

inline bool exists(const std::string& path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

// Reads exactly n bytes at offset; throws on a short read.
inline void pread_exact(int fd, std::uint8_t* dst, std::size_t n, std::int64_t offset, const std::string& what)
{
    std::size_t got = 0;
    while (got < n) {
        const ssize_t k = ::pread(fd, dst + got, n - got, static_cast<off_t>(offset + got));
        if (k < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), "read " + what);
        }
        if (k == 0)
            throw std::runtime_error(what + ": file is shorter than a chunk");
        got += static_cast<std::size_t>(k);
    }
}

inline std::vector<std::uint8_t> read_whole(int fd, const std::string& what)
{
    struct stat st;
    if (::fstat(fd, &st) != 0)
        throw std::system_error(errno, std::generic_category(), "stat " + what);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(st.st_size));
    if (!data.empty())
        pread_exact(fd, data.data(), data.size(), 0, what);
    return data;
}

inline void hint_willneed(int fd, std::int64_t offset, std::int64_t length)
{
#if defined(POSIX_FADV_WILLNEED)
    ::posix_fadvise(fd, offset, length, POSIX_FADV_WILLNEED);
#else
    (void)fd; (void)offset; (void)length;
#endif
}

// Writes `data` to `path` atomically (temporary file + rename), creating parent directories.
// Shared atomic publication for chunk buffers and streamed image files.
class AtomicFile {
public:
    explicit AtomicFile(const std::string& path) : path_(path)
    {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        tmp_ = path + ".tmp." + std::to_string(::getpid()) + "." +
               std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        fd_ = ::open(tmp_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "create " + tmp_);
    }
    ~AtomicFile() { if (fd_ >= 0) ::close(fd_); if (!tmp_.empty()) ::unlink(tmp_.c_str()); }
    AtomicFile(const AtomicFile&) = delete;
    AtomicFile& operator=(const AtomicFile&) = delete;
    int fd() const noexcept { return fd_; }
    const std::string& temporary_path() const noexcept { return tmp_; }
    void reset()
    {
        if (::ftruncate(fd_, 0) || ::lseek(fd_, 0, SEEK_SET) < 0)
            throw std::system_error(errno, std::generic_category(), "truncate " + tmp_);
    }
    void write(const std::uint8_t* data, std::size_t n)
    {
        for (std::size_t done = 0; done < n;) {
            const ssize_t k = ::write(fd_, data+done, n-done);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) throw std::system_error(k < 0 ? errno : EIO, std::generic_category(), "write " + tmp_);
            done += static_cast<std::size_t>(k);
        }
    }
    void publish()
    {
        const int fd = fd_;
        fd_ = -1;
        if (::close(fd)) throw std::system_error(errno, std::generic_category(), "close " + tmp_);
        if (::rename(tmp_.c_str(), path_.c_str()))
            throw std::system_error(errno, std::generic_category(), "rename " + tmp_);
        tmp_.clear();
    }
private:
    std::string path_, tmp_;
    int fd_ = -1;
};

inline void atomic_write(const std::string& path, const std::uint8_t* data, std::size_t n)
{
    AtomicFile file(path);
    file.write(data, n);
    file.publish();
}

// LRU of read-only file mappings, so repeated reads of a chunk need no mmap/munmap.
class MapCache {
public:
    struct Mapping {
        const std::uint8_t* data = nullptr;
        std::size_t size = 0;
        ~Mapping() { if (data) ::munmap(const_cast<std::uint8_t*>(data), size); }
    };
    using MappingPtr = std::shared_ptr<const Mapping>;

    explicit MapCache(std::size_t capacity) : capacity_(capacity) {}

    // The mapping of `path` (nullptr if the file does not exist).
    MappingPtr get(const std::string& path, bool& fresh)
    {
        fresh = false;
        {
            std::lock_guard lock(mutex_);
            if (auto it = index_.find(path); it != index_.end()) {
                order_.splice(order_.begin(), order_, it->second);
                return it->second->second;
            }
        }
        File f = File::open_read(path);
        if (!f)
            return nullptr;
        struct stat st;
        if (::fstat(f.fd(), &st) != 0)
            throw std::system_error(errno, std::generic_category(), "stat " + path);
        auto m = std::make_shared<Mapping>();
        m->size = static_cast<std::size_t>(st.st_size);
        if (m->size) {
            void* p = ::mmap(nullptr, m->size, PROT_READ, MAP_SHARED, f.fd(), 0);
            if (p == MAP_FAILED)
                throw std::system_error(errno, std::generic_category(), "mmap " + path);
            m->data = static_cast<const std::uint8_t*>(p);
        }
        std::lock_guard lock(mutex_);
        if (auto it = index_.find(path); it != index_.end())
            return it->second->second;
        order_.emplace_front(path, m);
        index_.emplace(path, order_.begin());
        fresh = true;
        while (order_.size() > capacity_) {
            index_.erase(order_.back().first);
            order_.pop_back();  // unmapped once no reader holds it
        }
        return m;
    }

private:
    std::size_t capacity_;
    std::list<std::pair<std::string, MappingPtr>> order_;
    std::unordered_map<std::string, std::list<std::pair<std::string, MappingPtr>>::iterator> index_;
    std::mutex mutex_;
};

}  // namespace vcr
