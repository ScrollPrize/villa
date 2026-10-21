#include "vc_object_store/MemMap.hpp"

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
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace vc::objects::memmap {

void MappedFileRO::open(const std::filesystem::path& path)
{
    close();
#if defined(_WIN32)
    // FILE_SHARE_DELETE so other handles can still rename/replace the file
    // while we hold a view (matches POSIX unlink-while-mapped semantics as
    // closely as Windows allows).
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("mmap open failed: " + path.string());
    }
    LARGE_INTEGER sz{};
    if (!::GetFileSizeEx(file, &sz) || sz.QuadPart <= 0) {
        ::CloseHandle(file);
        throw std::runtime_error("mmap stat failed: " + path.string());
    }
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    ::CloseHandle(file);  // the section holds its own reference
    if (!mapping) {
        throw std::runtime_error("mmap failed: " + path.string());
    }
    void* ptr = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    ::CloseHandle(mapping);  // the view holds its own reference
    if (!ptr) {
        throw std::runtime_error("mmap failed: " + path.string());
    }
    data_ = ptr;
    size_ = static_cast<std::size_t>(sz.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("mmap open failed: " + path.string());
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("mmap stat failed: " + path.string());
    }
    const auto sz = static_cast<std::size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping owns the pages; the descriptor is only needed for mmap().
    ::close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + path.string());
    }
    data_ = ptr;
    size_ = sz;
#endif
}

void MappedFileRO::close() noexcept
{
    if (data_) {
#if defined(_WIN32)
        ::UnmapViewOfFile(data_);
#else
        ::munmap(data_, size_);
#endif
    }
    data_ = nullptr;
    size_ = 0;
}

}  // namespace vc::objects::memmap
