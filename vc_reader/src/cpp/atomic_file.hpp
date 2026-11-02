// Atomic file publication shared by catalog, image, and chunk writers.
#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <system_error>
#include <thread>

namespace vcr {
// The temporary name a write of `path` is published from: <path>.tmp.<process>.<thread>, two
// decimal numbers, so an abandoned one is recognisable by name. The process part is a random
// tag rather than the pid: containers reuse pids while sharing a cache directory, and an
// interrupted run must not collide with a later writer's temporary.
inline std::string temporary_path(const std::string& path)
{
    static const std::uint64_t process = std::random_device{}() | (std::uint64_t(std::random_device{}()) << 32);
    return path + ".tmp." + std::to_string(process) + "." +
           std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

// Writes `data` to `path` atomically (temporary file + rename), creating parent directories.
// Shared atomic publication for chunk buffers, documents and streamed image files.
class AtomicFile {
public:
    explicit AtomicFile(const std::string& path) : path_(path)
    {
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        tmp_ = vcr::temporary_path(path);
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
    // Renames the temporary over `path`. `durable` first syncs the file to disk and, after
    // the rename, its directory, so the published file survives a crash; otherwise the OS
    // writes back in its own time (chunk and cache writes).
    void publish(bool durable = false)
    {
        if (durable && ::fsync(fd_)) throw std::system_error(errno, std::generic_category(), "fsync " + tmp_);
        const int fd = fd_;
        fd_ = -1;
        if (::close(fd)) throw std::system_error(errno, std::generic_category(), "close " + tmp_);
        if (::rename(tmp_.c_str(), path_.c_str()))
            throw std::system_error(errno, std::generic_category(), "rename " + tmp_);
        tmp_.clear();
        if (durable) {
            const auto parent = std::filesystem::path(path_).parent_path();
            const int dir = ::open((parent.empty() ? std::filesystem::path(".") : parent).c_str(), O_RDONLY | O_DIRECTORY);
            if (dir < 0 || ::fsync(dir)) {
                const int error = errno;
                if (dir >= 0) ::close(dir);
                throw std::system_error(error, std::generic_category(), "fsync directory of " + path_);
            }
            ::close(dir);
        }
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

} // namespace vcr
