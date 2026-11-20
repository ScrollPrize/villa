// Windows definitions of the file primitives in platform_io.hpp. POSIX ones are
// inline in the header, so this file is empty there.
#ifdef _WIN32

#include "platform_io.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>

#include <algorithm>
#include <random>

#ifndef O_ACCMODE  // not in the Microsoft <fcntl.h>
#define O_ACCMODE (O_RDONLY | O_WRONLY | O_RDWR)
#endif

namespace vcr::platform {
namespace {
int errno_from(DWORD error)
{
    switch (error) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_NAME:
    case ERROR_BAD_NETPATH:
        return ENOENT;
    case ERROR_ACCESS_DENIED:
        return EACCES;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
        return EEXIST;
    case ERROR_SHARING_VIOLATION:
        return EBUSY;
    case ERROR_LOCK_VIOLATION:
    case ERROR_IO_PENDING:
        return EWOULDBLOCK;
    case ERROR_TOO_MANY_OPEN_FILES:
        return EMFILE;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
        return ENOSPC;
    case ERROR_DIRECTORY:
        return ENOTDIR;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
        return ENOMEM;
    case ERROR_INVALID_HANDLE:
        return EBADF;
    default:
        return EIO;
    }
}

HANDLE os_handle(int fd) { return reinterpret_cast<HANDLE>(::_get_osfhandle(fd)); }

constexpr DWORD kShareAll = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

// A CRT descriptor owning `handle`; closes the handle if none is available.
int adopt(HANDLE handle, int flags)
{
    const int fd = ::_open_osfhandle(reinterpret_cast<intptr_t>(handle), flags | _O_NOINHERIT);
    if (fd < 0) {
        ::CloseHandle(handle);
        errno = EMFILE;
    }
    return fd;
}
}  // namespace

int open_file(const std::filesystem::path& path, int flags, int /*mode*/)
{
    DWORD access = GENERIC_READ;
    if ((flags & O_ACCMODE) == O_WRONLY) access = GENERIC_WRITE;
    else if ((flags & O_ACCMODE) == O_RDWR) access = GENERIC_READ | GENERIC_WRITE;
    DWORD disposition = OPEN_EXISTING;
    if ((flags & O_CREAT) && (flags & O_EXCL)) disposition = CREATE_NEW;
    else if ((flags & O_CREAT) && (flags & O_TRUNC)) disposition = CREATE_ALWAYS;
    else if (flags & O_CREAT) disposition = OPEN_ALWAYS;
    else if (flags & O_TRUNC) disposition = TRUNCATE_EXISTING;
    const HANDLE handle = ::CreateFileW(path.c_str(), access, kShareAll, nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = errno_from(::GetLastError());
        return -1;
    }
    return adopt(handle, flags & (O_ACCMODE | O_APPEND));
}

std::int64_t pread(int fd, void* buffer, std::size_t n, std::int64_t offset)
{
    OVERLAPPED at{};
    at.Offset = static_cast<DWORD>(static_cast<std::uint64_t>(offset));
    at.OffsetHigh = static_cast<DWORD>(static_cast<std::uint64_t>(offset) >> 32);
    DWORD got = 0;
    const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n, std::size_t(1) << 30));
    if (!::ReadFile(os_handle(fd), buffer, want, &got, &at)) {
        const DWORD error = ::GetLastError();
        if (error == ERROR_HANDLE_EOF) return 0;
        errno = errno_from(error);
        return -1;
    }
    return got;
}

int reopen_read(int fd)
{
    const HANDLE handle = ::ReOpenFile(os_handle(fd), GENERIC_READ, kShareAll, 0);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = errno_from(::GetLastError());
        return -1;
    }
    return adopt(handle, _O_RDONLY);
}

int fsync(int fd) { return ::_commit(fd); }

void sync_directory(const std::filesystem::path&, const std::string&) {}

void replace_file(const std::filesystem::path& from, const std::filesystem::path& to, bool durable)
{
    const DWORD flags = MOVEFILE_REPLACE_EXISTING | (durable ? MOVEFILE_WRITE_THROUGH : 0);
    if (!::MoveFileExW(from.c_str(), to.c_str(), flags)) {
        const std::error_code error(static_cast<int>(::GetLastError()), std::system_category());
        throw std::filesystem::filesystem_error("cannot replace file", from, to, error);
    }
}

int fstat(int fd, FileStat& out)
{
    struct _stat64 st{};
    if (::_fstat64(fd, &st) != 0) return -1;
    out.size = static_cast<std::int64_t>(st.st_size);
    out.regular = (st.st_mode & _S_IFMT) == _S_IFREG;
    out.directory = (st.st_mode & _S_IFMT) == _S_IFDIR;
    out.device = static_cast<std::uint64_t>(st.st_dev);
    out.inode = static_cast<std::uint64_t>(st.st_ino);
    out.modified = {static_cast<std::int64_t>(st.st_mtime), 0};
    out.changed = {static_cast<std::int64_t>(st.st_ctime), 0};
    return 0;
}

const std::uint8_t* map_read(int fd, std::size_t size)
{
    const HANDLE mapping = ::CreateFileMappingW(os_handle(fd), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        errno = errno_from(::GetLastError());
        return nullptr;
    }
    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size);
    const DWORD error = ::GetLastError();
    ::CloseHandle(mapping);  // the view keeps the mapping alive
    if (!view) errno = errno_from(error);
    return static_cast<const std::uint8_t*>(view);
}

void unmap(const std::uint8_t* data, std::size_t) { ::UnmapViewOfFile(data); }

int lock_exclusive(int fd, bool wait)
{
    OVERLAPPED whole{};
    const DWORD flags = LOCKFILE_EXCLUSIVE_LOCK | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY);
    if (::LockFileEx(os_handle(fd), flags, 0, MAXDWORD, MAXDWORD, &whole)) return 0;
    errno = errno_from(::GetLastError());
    return -1;
}

std::string make_temp_directory(std::string pattern)
{
    static constexpr char letters[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    if (pattern.size() < 6 || pattern.compare(pattern.size() - 6, 6, "XXXXXX") != 0)
        throw std::system_error(EINVAL, std::generic_category(), "mkdtemp " + pattern);
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
        for (std::size_t i = pattern.size() - 6; i < pattern.size(); ++i)
            pattern[i] = letters[random() % (sizeof letters - 1)];
        std::error_code error;
        if (std::filesystem::create_directory(pattern, error)) return pattern;
        if (error) throw std::system_error(error, "mkdtemp " + pattern);
    }
    throw std::system_error(EEXIST, std::generic_category(), "mkdtemp " + pattern);
}

long process_id() { return static_cast<long>(::_getpid()); }

}  // namespace vcr::platform

#endif
