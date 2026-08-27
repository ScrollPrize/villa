#include "GuiStallSampler.hpp"

#include <QtGlobal>

#ifdef Q_OS_LINUX

#include <QTimer>
#include <QCoreApplication>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

#include <execinfo.h>
#include <pthread.h>
#include <unistd.h>

namespace vc3d::diag {
namespace {

std::atomic<std::int64_t> lastHeartbeatMs{0};
std::atomic<std::int64_t> stallSinceMs{0};
pthread_t guiThread;

std::int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Async-signal handler: only write()/backtrace are used. backtrace_symbols_fd
// is not formally async-signal-safe but is reliable in practice for
// diagnostics; this code exists only to name a frozen frame.
void stallSignalHandler(int)
{
    const std::int64_t stalledFor =
        nowMs() - stallSinceMs.load(std::memory_order_relaxed);
    char header[128];
    const int len = std::snprintf(
        header, sizeof(header),
        "\n=== GUI STALL SAMPLE (stalled ~%" PRId64 " ms) ===\n", stalledFor);
    if (len > 0) {
        (void)!write(STDERR_FILENO, header, static_cast<size_t>(len));
    }
    void* frames[64];
    const int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    static constexpr char footer[] = "=== END GUI STALL SAMPLE ===\n";
    (void)!write(STDERR_FILENO, footer, sizeof(footer) - 1);
}

void monitorLoop()
{
    constexpr std::int64_t kStallThresholdMs = 600;
    constexpr std::int64_t kResampleIntervalMs = 1000;
    std::int64_t lastSampleMs = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const std::int64_t now = nowMs();
        const std::int64_t beat = lastHeartbeatMs.load(std::memory_order_relaxed);
        if (beat == 0) {
            continue;
        }
        if (now - beat <= kStallThresholdMs) {
            stallSinceMs.store(0, std::memory_order_relaxed);
            continue;
        }
        if (stallSinceMs.load(std::memory_order_relaxed) == 0) {
            stallSinceMs.store(beat, std::memory_order_relaxed);
            lastSampleMs = 0;
        }
        if (now - lastSampleMs >= kResampleIntervalMs) {
            lastSampleMs = now;
            pthread_kill(guiThread, SIGUSR2);
        }
    }
}

}  // namespace

void installGuiStallSampler()
{
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;

    guiThread = pthread_self();
    lastHeartbeatMs.store(nowMs(), std::memory_order_relaxed);

    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = stallSignalHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    sigaction(SIGUSR2, &action, nullptr);

    auto* heartbeat = new QTimer(QCoreApplication::instance());
    heartbeat->setInterval(100);
    heartbeat->setTimerType(Qt::CoarseTimer);
    QObject::connect(heartbeat, &QTimer::timeout, []() {
        lastHeartbeatMs.store(nowMs(), std::memory_order_relaxed);
    });
    heartbeat->start();

    std::thread(monitorLoop).detach();
}

}  // namespace vc3d::diag

#else

namespace vc3d::diag {
void installGuiStallSampler() {}
}  // namespace vc3d::diag

#endif
