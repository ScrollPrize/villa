// File primitives the reader's caches, chunk reads and writers share. POSIX
// definitions are inline here; the Windows (MinGW/UCRT) ones are in platform_io.cpp,
// part of vc_reader_catalog, so public headers never pull in <windows.h>. Each
// mirrors the POSIX call it replaces: -1 (or nullptr) with errno set on failure,
// unless noted.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#ifndef _WIN32
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vcr::platform {

struct FileTime {
    std::int64_t seconds = 0, nanoseconds = 0;
};

// What fstat(2) reports, with 64-bit sizes on every platform (MinGW's struct stat
// holds a 32-bit size). Windows reports no inode and whole-second times.
struct FileStat {
    std::int64_t size = 0;
    bool regular = false, directory = false;
    std::uint64_t device = 0, inode = 0;
    FileTime modified, changed;
};

#ifdef _WIN32

// open(2). The descriptor is binary and not inherited, and the file is shared for
// read, write and delete, so it can be renamed over or removed while open, as on POSIX.
int open_file(const std::filesystem::path& path, int flags, int mode = 0644);
// pread(2), reading at most 1 GiB. Moves the descriptor's file position: readers
// that depend on it get their own descriptor (reopen_read).
std::int64_t pread(int fd, void* buffer, std::size_t n, std::int64_t offset);
// A descriptor for reading the same file from its start, with its own position.
int reopen_read(int fd);
int fsync(int fd);
// No-op: NTFS journals the rename itself.
void sync_directory(const std::filesystem::path& directory, const std::string& what);
// MoveFileEx(MOVEFILE_REPLACE_EXISTING); `durable` also waits for the move to reach the disk.
void replace_file(const std::filesystem::path& from, const std::filesystem::path& to, bool durable);
int fstat(int fd, FileStat& out);
const std::uint8_t* map_read(int fd, std::size_t size);
void unmap(const std::uint8_t* data, std::size_t size);
// No-op on Windows.
inline void advise_willneed(const std::uint8_t*, std::int64_t, std::int64_t) {}
// LockFileEx over the whole file. Windows locks are mandatory: lock only dedicated
// lock files or files only the holder reads and writes.
int lock_exclusive(int fd, bool wait);
std::string make_temp_directory(std::string pattern);
long process_id();

#else

// open(2) with close-on-exec.
inline int open_file(const std::filesystem::path& path, int flags, int mode = 0644)
{
    return ::open(path.c_str(), flags | O_CLOEXEC, mode);
}

// pread(2): up to n bytes at offset, 0 at end of file.
inline std::int64_t pread(int fd, void* buffer, std::size_t n, std::int64_t offset)
{
    return ::pread(fd, buffer, n, static_cast<off_t>(offset));
}

// A descriptor for reading the same file from its start. dup(2) shares the file
// offset with `fd`.
inline int reopen_read(int fd)
{
    const int copy = ::dup(fd);
    if (copy >= 0) ::lseek(copy, 0, SEEK_SET);
    return copy;
}

inline int fsync(int fd) { return ::fsync(fd); }

// Makes a rename in `directory` durable; throws std::system_error.
inline void sync_directory(const std::filesystem::path& directory, const std::string& what)
{
    const int dir = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0 || ::fsync(dir)) {
        const int error = errno;
        if (dir >= 0) ::close(dir);
        throw std::system_error(error, std::generic_category(), "fsync directory of " + what);
    }
    ::close(dir);
}

// Renames `from` over `to`, replacing it atomically if it exists; throws
// std::filesystem::filesystem_error. Callers wanting durability sync the directory.
inline void replace_file(const std::filesystem::path& from, const std::filesystem::path& to, bool /*durable*/)
{
    std::filesystem::rename(from, to);
}

inline int fstat(int fd, FileStat& out)
{
    struct stat st{};
    if (::fstat(fd, &st) != 0) return -1;
#if defined(__APPLE__)
    out.modified = {static_cast<std::int64_t>(st.st_mtimespec.tv_sec), static_cast<std::int64_t>(st.st_mtimespec.tv_nsec)};
    out.changed = {static_cast<std::int64_t>(st.st_ctimespec.tv_sec), static_cast<std::int64_t>(st.st_ctimespec.tv_nsec)};
#else
    out.modified = {static_cast<std::int64_t>(st.st_mtim.tv_sec), static_cast<std::int64_t>(st.st_mtim.tv_nsec)};
    out.changed = {static_cast<std::int64_t>(st.st_ctim.tv_sec), static_cast<std::int64_t>(st.st_ctim.tv_nsec)};
#endif
    out.size = static_cast<std::int64_t>(st.st_size);
    out.regular = S_ISREG(st.st_mode);
    out.directory = S_ISDIR(st.st_mode);
    out.device = static_cast<std::uint64_t>(st.st_dev);
    out.inode = static_cast<std::uint64_t>(st.st_ino);
    return 0;
}

// A read-only shared mapping of the first `size` bytes of `fd`; nullptr on failure.
inline const std::uint8_t* map_read(int fd, std::size_t size)
{
    void* view = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    return view == MAP_FAILED ? nullptr : static_cast<const std::uint8_t*>(view);
}

inline void unmap(const std::uint8_t* data, std::size_t size) { ::munmap(const_cast<std::uint8_t*>(data), size); }

// Hints that bytes [first, first + length) of a mapping are read soon.
inline void advise_willneed(const std::uint8_t* mapping, std::int64_t first, std::int64_t length)
{
    const std::int64_t page = ::sysconf(_SC_PAGESIZE);
    const std::int64_t aligned = first / page * page;
    ::madvise(const_cast<std::uint8_t*>(mapping) + aligned, static_cast<std::size_t>(first + length - aligned),
              MADV_WILLNEED);
}

// An exclusive whole-file lock, released when the descriptor closes (flock(2)).
// Without `wait`, fails with EWOULDBLOCK while another descriptor holds it.
inline int lock_exclusive(int fd, bool wait) { return ::flock(fd, LOCK_EX | (wait ? 0 : LOCK_NB)); }

// mkdtemp(3): creates a new directory named `pattern` with its trailing XXXXXX
// replaced, and returns its name. Throws std::system_error.
inline std::string make_temp_directory(std::string pattern)
{
    if (!::mkdtemp(pattern.data())) throw std::system_error(errno, std::generic_category(), "mkdtemp");
    return pattern;
}

inline long process_id() { return static_cast<long>(::getpid()); }

#endif

}  // namespace vcr::platform
