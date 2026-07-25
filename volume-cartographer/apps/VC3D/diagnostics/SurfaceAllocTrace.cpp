// Opt-in allocation stack tracer for diagnosing VC3D's surface-cache memory
// pipeline. Build target: vc_surface_alloc_trace. Usage:
//
//   VC_SURFACE_ALLOC_TRACE=1 \
//   VC_SURFACE_ALLOC_TRACE_MIN_KB=256 \
//   VC_SURFACE_ALLOC_TRACE_FILE=/tmp/vc3d-surface-alloc.log \
//   LD_PRELOAD="$PWD/build/lib/libvc_surface_alloc_trace.so" \
//   build/bin/VC3D --load-first traces
//
// This file deliberately uses glibc's internal allocation entry points. It is
// a Linux-only diagnostic preload library and is never linked into VC3D.

#include <features.h>

#if !defined(__GLIBC__)
#error "vc_surface_alloc_trace requires glibc"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <string_view>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern "C" {
void* __libc_malloc(std::size_t);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
void __libc_free(void*);
void* __libc_memalign(std::size_t, std::size_t);
}

namespace {

constexpr std::size_t kDefaultThreshold = 256ULL << 10;
constexpr int kMaxFrames = 48;
constexpr std::size_t kLogBufferSize = 32768;

thread_local bool g_inHook = false;
std::atomic<int> g_enabled{-1};
std::atomic<std::size_t> g_threshold{0};
std::atomic<int> g_logFd{-2};
std::atomic<std::uint64_t> g_sequence{0};
std::atomic<std::size_t> g_liveLargeBytes{0};
std::atomic<std::size_t> g_peakLargeBytes{0};
std::atomic_flag g_logLock = ATOMIC_FLAG_INIT;

bool truthy(const char* value) noexcept
{
    if (!value || !*value)
        return false;
    const std::string_view text(value);
    return text != "0" && text != "false" && text != "FALSE" &&
           text != "off" && text != "OFF";
}

bool traceEnabled() noexcept
{
    int enabled = g_enabled.load(std::memory_order_acquire);
    if (enabled >= 0)
        return enabled != 0;
    enabled = truthy(std::getenv("VC_SURFACE_ALLOC_TRACE")) ? 1 : 0;
    g_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

std::size_t traceThreshold() noexcept
{
    std::size_t threshold = g_threshold.load(std::memory_order_acquire);
    if (threshold != 0)
        return threshold;

    const auto parseThreshold = [](const char* name,
                                   unsigned long long multiplier,
                                   std::size_t& result) noexcept {
        const char* value = std::getenv(name);
        if (!value || !*value)
            return false;
        unsigned long long megabytes = 0;
        const char* end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, megabytes);
        if (parsed.ec != std::errc{} || parsed.ptr != end || megabytes == 0 ||
            megabytes > ULLONG_MAX / multiplier)
            return false;
        const unsigned long long bytes = megabytes * multiplier;
        if (bytes > SIZE_MAX)
            return false;
        result = static_cast<std::size_t>(bytes);
        return true;
    };

    threshold = kDefaultThreshold;
    // The byte and KiB forms make it possible to catch tile-sized allocations.
    // Preserve MIN_MB as a backwards-compatible fallback.
    if (!parseThreshold("VC_SURFACE_ALLOC_TRACE_MIN_BYTES", 1, threshold) &&
        !parseThreshold("VC_SURFACE_ALLOC_TRACE_MIN_KB", 1ULL << 10,
                        threshold)) {
        parseThreshold("VC_SURFACE_ALLOC_TRACE_MIN_MB", 1ULL << 20, threshold);
    }
    g_threshold.store(threshold, std::memory_order_release);
    return threshold;
}

int logFd() noexcept
{
    int fd = g_logFd.load(std::memory_order_acquire);
    if (fd != -2)
        return fd;

    const char* path = std::getenv("VC_SURFACE_ALLOC_TRACE_FILE");
    if (!path || !*path)
        path = "/tmp/vc3d-surface-alloc.log";
    fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    int expected = -2;
    if (!g_logFd.compare_exchange_strong(expected, fd,
                                         std::memory_order_acq_rel)) {
        if (fd >= 0)
            ::close(fd);
        fd = expected;
    }
    return fd;
}

void append(char* buffer, std::size_t& used, const char* format, ...) noexcept
{
    if (used >= kLogBufferSize)
        return;
    va_list args;
    va_start(args, format);
    const int count =
        std::vsnprintf(buffer + used, kLogBufferSize - used, format, args);
    va_end(args);
    if (count <= 0)
        return;
    used += std::min<std::size_t>(static_cast<std::size_t>(count),
                                  kLogBufferSize - used);
}

std::size_t addLive(std::size_t bytes) noexcept
{
    const std::size_t live =
        g_liveLargeBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::size_t peak = g_peakLargeBytes.load(std::memory_order_relaxed);
    while (peak < live &&
           !g_peakLargeBytes.compare_exchange_weak(
               peak, live, std::memory_order_relaxed)) {
    }
    return live;
}

std::size_t subtractLive(std::size_t bytes) noexcept
{
    std::size_t live = g_liveLargeBytes.load(std::memory_order_relaxed);
    for (;;) {
        const std::size_t next = live > bytes ? live - bytes : 0;
        if (g_liveLargeBytes.compare_exchange_weak(
                live, next, std::memory_order_relaxed)) {
            return next;
        }
    }
}

void writeRecord(const char* operation,
                 void* pointer,
                 std::size_t requested,
                 std::size_t usable,
                 std::size_t live,
                 bool stack) noexcept
{
    const int fd = logFd();
    if (fd < 0)
        return;

    char buffer[kLogBufferSize];
    std::size_t used = 0;
    timespec now{};
    ::clock_gettime(CLOCK_MONOTONIC, &now);
    append(buffer, used,
           "[surface-alloc] seq=%llu ns=%llu tid=%ld op=%s ptr=%p "
           "pid=%ld requested=%zu usable=%zu live_large=%zu "
           "peak_large=%zu\n",
           static_cast<unsigned long long>(
               g_sequence.fetch_add(1, std::memory_order_relaxed)),
           static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL +
               static_cast<unsigned long long>(now.tv_nsec),
           static_cast<long>(::syscall(SYS_gettid)), operation, pointer,
           static_cast<long>(::getpid()), requested, usable, live,
           g_peakLargeBytes.load(std::memory_order_relaxed));

    if (stack) {
        void* frames[kMaxFrames];
        const int frameCount = ::backtrace(frames, kMaxFrames);
        for (int i = 2; i < frameCount; ++i) {
            Dl_info info{};
            if (::dladdr(frames[i], &info) && info.dli_fname &&
                info.dli_fbase) {
                const auto offset =
                    static_cast<unsigned long long>(
                        reinterpret_cast<std::uintptr_t>(frames[i]) -
                        reinterpret_cast<std::uintptr_t>(info.dli_fbase));
                append(buffer, used, "  #%d %s+0x%llx %s\n", i - 2,
                       info.dli_fname, offset,
                       info.dli_sname ? info.dli_sname : "?");
            } else {
                append(buffer, used, "  #%d %p\n", i - 2, frames[i]);
            }
        }
    }

    while (g_logLock.test_and_set(std::memory_order_acquire)) {
    }
    const char* cursor = buffer;
    std::size_t remaining = used;
    while (remaining > 0) {
        const ssize_t written =
            static_cast<ssize_t>(::syscall(SYS_write, fd, cursor, remaining));
        if (written > 0) {
            cursor += written;
            remaining -= static_cast<std::size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    g_logLock.clear(std::memory_order_release);
}

std::size_t usableSize(void* pointer) noexcept
{
    return pointer ? ::malloc_usable_size(pointer) : 0;
}

void traceAllocation(const char* operation, void* pointer,
                     std::size_t requested) noexcept
{
    if (!pointer || !traceEnabled())
        return;
    const std::size_t usable = usableSize(pointer);
    if (usable < traceThreshold())
        return;
    const std::size_t live = addLive(usable);
    writeRecord(operation, pointer, requested, usable, live, true);
}

void traceFree(void* pointer) noexcept
{
    if (!pointer || !traceEnabled())
        return;
    const std::size_t usable = usableSize(pointer);
    if (usable < traceThreshold())
        return;
    const std::size_t live = subtractLive(usable);
    writeRecord("free", pointer, 0, usable, live, false);
}

struct HookGuard {
    HookGuard() noexcept { g_inHook = true; }
    ~HookGuard() { g_inHook = false; }
};

void disableTracingInForkChild() noexcept
{
    g_enabled.store(0, std::memory_order_relaxed);
    g_logLock.clear(std::memory_order_relaxed);
}

__attribute__((constructor)) void initializeTracer() noexcept
{
    if (!traceEnabled())
        return;
    HookGuard guard;
    const std::size_t threshold = traceThreshold();
    // Open the file and emit a marker before main(). This makes a successful
    // preload distinguishable from a run that merely had no qualifying
    // allocations.
    writeRecord("tracer_start", nullptr, threshold, 0, 0, false);

    // Keep the diagnostic interposer confined to VC3D. QProcess and similar
    // launchers inherit the environment, so leaving LD_PRELOAD set would also
    // inject this library into helpers such as ssh.
    ::pthread_atfork(nullptr, nullptr, disableTracingInForkChild);
    ::unsetenv("LD_PRELOAD");
}

} // namespace

extern "C" void* malloc(std::size_t size) noexcept
{
    void* pointer = __libc_malloc(size);
    if (!g_inHook) {
        HookGuard guard;
        traceAllocation("malloc", pointer, size);
    }
    return pointer;
}

extern "C" void* calloc(std::size_t count, std::size_t size) noexcept
{
    void* pointer = __libc_calloc(count, size);
    if (!g_inHook) {
        HookGuard guard;
        const std::size_t requested =
            size != 0 && count > SIZE_MAX / size ? SIZE_MAX : count * size;
        traceAllocation("calloc", pointer, requested);
    }
    return pointer;
}

extern "C" void* realloc(void* oldPointer, std::size_t size) noexcept
{
    if (g_inHook)
        return __libc_realloc(oldPointer, size);
    HookGuard guard;
    const std::size_t oldUsable =
        oldPointer && traceEnabled() ? usableSize(oldPointer) : 0;
    void* pointer = __libc_realloc(oldPointer, size);
    // A failed non-zero realloc leaves the old allocation live.
    if (oldUsable >= traceThreshold() && (pointer || size == 0)) {
        const std::size_t live = subtractLive(oldUsable);
        writeRecord("realloc_old", oldPointer, 0, oldUsable, live, false);
    }
    traceAllocation("realloc", pointer, size);
    return pointer;
}

extern "C" void free(void* pointer) noexcept
{
    if (!g_inHook) {
        HookGuard guard;
        traceFree(pointer);
    }
    __libc_free(pointer);
}

extern "C" void* memalign(std::size_t alignment, std::size_t size) noexcept
{
    void* pointer = __libc_memalign(alignment, size);
    if (!g_inHook) {
        HookGuard guard;
        traceAllocation("memalign", pointer, size);
    }
    return pointer;
}

extern "C" void* aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    void* pointer = __libc_memalign(alignment, size);
    if (!g_inHook) {
        HookGuard guard;
        traceAllocation("aligned_alloc", pointer, size);
    }
    return pointer;
}

extern "C" int posix_memalign(void** result, std::size_t alignment,
                              std::size_t size) noexcept
{
    if (!result || alignment < sizeof(void*) ||
        (alignment & (alignment - 1)) != 0) {
        return EINVAL;
    }
    void* pointer = __libc_memalign(alignment, size);
    if (!pointer)
        return ENOMEM;
    *result = pointer;
    if (!g_inHook) {
        HookGuard guard;
        traceAllocation("posix_memalign", pointer, size);
    }
    return 0;
}
