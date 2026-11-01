#include "vc/core/util/MemMap.hpp"

#include <new>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace vc::memmap {

int pid()
{
#if defined(_WIN32)
    return ::_getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

void* mapFileRW(const std::filesystem::path& path, std::size_t bytes)
{
#if defined(_WIN32)
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("mapFileRW open failed: " + path.string());
    }
    // CreateFileMapping extends the file to `bytes` when it is shorter.
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READWRITE,
                                          static_cast<DWORD>(static_cast<unsigned long long>(bytes) >> 32),
                                          static_cast<DWORD>(bytes & 0xffffffffu), nullptr);
    ::CloseHandle(file);
    if (!mapping) {
        throw std::runtime_error("mapFileRW mapping failed: " + path.string());
    }
    void* ptr = ::MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, bytes);
    ::CloseHandle(mapping);
    if (!ptr) {
        throw std::runtime_error("mapFileRW map view failed: " + path.string());
    }
    return ptr;
#else
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, static_cast<mode_t>(0600));
    if (fd < 0) {
        throw std::runtime_error("mapFileRW open failed: " + path.string());
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("mapFileRW stat failed: " + path.string());
    }
    if (static_cast<std::size_t>(st.st_size) < bytes && ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        ::close(fd);
        throw std::runtime_error("mapFileRW ftruncate failed: " + path.string());
    }
    void* ptr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mapFileRW mmap failed: " + path.string());
    }
    return ptr;
#endif
}

void unmapRW(void* ptr, std::size_t bytes) noexcept
{
    if (!ptr) {
        return;
    }
#if defined(_WIN32)
    (void)bytes;
    ::UnmapViewOfFile(ptr);
#else
    ::munmap(ptr, bytes);
#endif
}

void* anonAlloc(std::size_t bytes)
{
#if defined(_WIN32)
    void* ptr = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
#else
    void* ptr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    return ptr;
#endif
}

void anonFree(void* ptr, std::size_t bytes) noexcept
{
    if (!ptr) {
        return;
    }
#if defined(_WIN32)
    (void)bytes;
    ::VirtualFree(ptr, 0, MEM_RELEASE);
#else
    ::munmap(ptr, bytes);
#endif
}

}  // namespace vc::memmap
