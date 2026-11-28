#pragma once

// One owner at a time, in any process, of a file a reader writes in place under
// a cache root (a normal-grid range pack). The owner holds the returned token for
// as long as it uses the file; the disk cache does not evict an owned file.
//
// No descriptor per file: each directory has one lock file, `.owners.lock`,
// opened once per process and kept open, in which an owned file is one locked
// byte (at a hash of its name). fcntl locks do not conflict within a process, so
// a registry of the files this process owns answers for it. (A lock file closed
// would drop all of a process's locks on it, which is why it stays open.) Two
// names hashing to one byte cost only an owner that keeps its file in memory, or
// a file not evicted while the other is owned.

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#endif

namespace vcr {

namespace file_ownership_detail {

struct Registry {
    std::mutex mutex;
    std::set<std::string> owned;           // by this process
    std::map<std::string, int> lock_files;  // directory -> its lock file, kept open
};

}  // namespace file_ownership_detail
}  // namespace vcr

// This process's registry, in whichever library holds this copy. Every library
// that includes this header holds a copy. On Linux the dynamic linker binds them
// all to one, but a macOS library keeps its own, so the libraries would disagree
// about which files this process owns; there, registry() asks for the first copy
// loaded (first_loaded_registry), so all of them use one. Each library must emit
// it as a symbol for that search to find (used, noinline: inlined into its
// callers, a library would not export it).
extern "C" inline
#if !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
__attribute__((visibility("default"), used, noinline))
#endif
void* vcr_file_ownership_registry_v1()
{
    static auto* r = new vcr::file_ownership_detail::Registry;  // never destroyed: its lock files stay open
    return r;
}

namespace vcr {
namespace file_ownership_detail {

#ifdef __APPLE__
// The registry of the first image, in load order, that exports one. Each copy
// walks the same list, so each finds the same image (a library loaded with
// RTLD_LOCAL included, which a search of the global scope would skip).
inline void* first_loaded_registry()
{
    using Get = void* (*)();
    const std::uint32_t images = _dyld_image_count();
    for (std::uint32_t i = 0; i < images; ++i) {
        const char* name = _dyld_get_image_name(i);
        if (!name) continue;
        void* image = ::dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
        if (!image) continue;
        auto get = reinterpret_cast<Get>(::dlsym(image, "vcr_file_ownership_registry_v1"));
        ::dlclose(image);
        if (get) return get();
    }
    return nullptr;
}
#endif

inline Registry& registry()
{
#ifdef __APPLE__
    static Registry* r = [] {
        void* found = first_loaded_registry();
        return static_cast<Registry*>(found ? found : vcr_file_ownership_registry_v1());
    }();
    return *r;
#else
    return *static_cast<Registry*>(vcr_file_ownership_registry_v1());
#endif
}

// One spelling per file, as the disk cache spells it (symlinks resolved: on
// macOS /var is /private/var), so an owner and the cache agree.
inline std::string key(const std::string& path)
{
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) canonical = std::filesystem::absolute(path, ec);
    return (ec ? std::filesystem::path(path) : canonical).lexically_normal().string();
}

#ifndef _WIN32
inline off_t byte_of(const std::string& key)
{
    const auto name = std::filesystem::path(key).filename().string();
    std::uint64_t h = 1469598103934665603ull;  // FNV-1a
    for (const unsigned char c : name) h = (h ^ c) * 1099511628211ull;
    return static_cast<off_t>(h & ((std::uint64_t(1) << 40) - 1));
}

// Caller holds the registry's mutex.
inline int lock_file(Registry& r, const std::string& key)
{
    const auto dir = std::filesystem::path(key).parent_path().string();
    const auto found = r.lock_files.find(dir);
    if (found != r.lock_files.end()) return found->second;
    const int fd = ::open((dir + "/.owners.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) r.lock_files.emplace(dir, fd);
    return fd;
}

inline bool set_lock(int fd, off_t byte, short type)
{
    struct flock lock{};
    lock.l_type = type;
    lock.l_whence = SEEK_SET;
    lock.l_start = byte;
    lock.l_len = 1;
    while (::fcntl(fd, F_SETLK, &lock) != 0)
        if (errno != EINTR) return false;
    return true;
}
#endif

}  // namespace file_ownership_detail

// The file `path` owned by the caller while the token lives; nullptr when it is
// owned already (here or in another process).
inline std::shared_ptr<void> ownFile(const std::string& path)
{
    using namespace file_ownership_detail;
    auto& r = registry();
    const auto k = key(path);
    std::lock_guard lock(r.mutex);
    if (r.owned.count(k)) return nullptr;
#ifndef _WIN32
    const int fd = lock_file(r, k);
    if (fd < 0 || !set_lock(fd, byte_of(k), F_WRLCK)) return nullptr;
#endif
    r.owned.insert(k);
    return std::shared_ptr<void>(new int(0), [k](void* p) {
        delete static_cast<int*>(p);
        auto& r = registry();
        std::lock_guard lock(r.mutex);
        r.owned.erase(k);
#ifndef _WIN32
        const auto found = r.lock_files.find(std::filesystem::path(k).parent_path().string());
        if (found != r.lock_files.end()) (void)set_lock(found->second, byte_of(k), F_UNLCK);
#endif
    });
}

// Whether a reader, in this process or another, owns `path`.
inline bool fileOwned(const std::string& path)
{
    using namespace file_ownership_detail;
    auto& r = registry();
    const auto k = key(path);
    std::lock_guard lock(r.mutex);
    if (r.owned.count(k)) return true;
#ifndef _WIN32
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(k).parent_path() / ".owners.lock", ec)) return false;
    const int fd = lock_file(r, k);
    if (fd < 0) return false;
    struct flock probe{};
    probe.l_type = F_WRLCK;
    probe.l_whence = SEEK_SET;
    probe.l_start = byte_of(k);
    probe.l_len = 1;
    if (::fcntl(fd, F_GETLK, &probe) != 0) return false;
    return probe.l_type != F_UNLCK;  // held by another process
#else
    return false;
#endif
}

}  // namespace vcr
