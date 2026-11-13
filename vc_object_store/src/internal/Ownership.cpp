#include "internal/Ownership.hpp"

#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace vc::objects::detail {
namespace {
std::filesystem::path guardPath(const std::filesystem::path& store)
{
    // weakly_canonical resolves symlinks of the existing prefix (the store
    // exists for every writable open except create, where its directory does).
    auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(store));
    // SQLite journals and the ownership guard are both pathname-based.
    // Distinct hard links would give the same database independent sidecars.
    if (std::filesystem::is_regular_file(path) && std::filesystem::hard_link_count(path) > 1)
        throw std::runtime_error("object store: writable open of a hard-linked file is not supported: " + path.string());
    path += "-owner";
    return path;
}

[[noreturn]] void busy(const std::filesystem::path& store, OwnershipGuard::Mode mode)
{
    if (mode == OwnershipGuard::Mode::Service)
        throw std::runtime_error("object store: " + store.string() +
            " is open for writing by another process or handle; close standalone writers before hosting it");
    throw std::runtime_error("object store: " + store.string() +
        " is owned by its hosting service; connect to the service, or stop it before opening the store directly");
}
}

#ifdef _WIN32
struct OwnershipGuard::State { HANDLE file = INVALID_HANDLE_VALUE; };

OwnershipGuard::OwnershipGuard(const std::filesystem::path& store, Mode mode) : state_(std::make_unique<State>())
{
    const auto path = guardPath(store);
    state_->file = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (state_->file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("object store: cannot open ownership guard " + path.string());
    OVERLAPPED overlapped{};
    const DWORD flags = LOCKFILE_FAIL_IMMEDIATELY | (mode == Mode::Service ? LOCKFILE_EXCLUSIVE_LOCK : 0);
    if (!LockFileEx(state_->file, flags, 0, 1, 0, &overlapped)) {
        CloseHandle(state_->file);
        busy(store, mode);
    }
}

OwnershipGuard::~OwnershipGuard()
{
    if (state_ && state_->file != INVALID_HANDLE_VALUE) CloseHandle(state_->file);
}
#else
struct OwnershipGuard::State { int fd = -1; };

OwnershipGuard::OwnershipGuard(const std::filesystem::path& store, Mode mode) : state_(std::make_unique<State>())
{
    const auto path = guardPath(store);
    state_->fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0664);
    if (state_->fd < 0)
        throw std::system_error(errno, std::generic_category(), "object store: cannot open ownership guard " + path.string());
    // flock locks belong to the open file description: two handles in one
    // process conflict exactly like two processes.
    int rc;
    do rc = ::flock(state_->fd, (mode == Mode::Service ? LOCK_EX : LOCK_SH) | LOCK_NB);
    while (rc != 0 && errno == EINTR);
    if (rc != 0) {
        const int error = errno;
        ::close(state_->fd);
        state_->fd = -1;
        if (error == EWOULDBLOCK) busy(store, mode);
        throw std::system_error(error, std::generic_category(), "object store: cannot lock ownership guard " + path.string());
    }
}

OwnershipGuard::~OwnershipGuard()
{
    if (state_ && state_->fd >= 0) ::close(state_->fd);
}
#endif
}
