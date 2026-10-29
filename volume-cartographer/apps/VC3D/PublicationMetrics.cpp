#include "PublicationMetrics.hpp"

#include <QObject>
#include <QTimer>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vc3d::metrics {
namespace {

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_nextId{1};
std::atomic<int> g_nextThread{0};
std::atomic<uint64_t> g_failures{0};

thread_local std::vector<uint64_t> t_stack;
thread_local std::string t_thread;

int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

const std::string& threadName()
{
    if (t_thread.empty()) t_thread = "t" + std::to_string(g_nextThread.fetch_add(1));
    return t_thread;
}

// Records go through a bounded queue to a writer thread, so neither the GUI
// thread nor the hub worker (which traces while holding its sync mutex) ever
// waits for the filesystem. Dropped records are counted and announced in the
// log, so the analysis can reject the run.
class Writer {
public:
    ~Writer() { shutdown(); }

    bool open(const std::filesystem::path& path)
    {
        std::lock_guard lock(mutex_);
        if (thread_.joinable()) return false;
        log_.open(path, std::ios::out | std::ios::trunc);
        if (!log_.is_open()) { g_failures.fetch_add(1); return false; }
        stop_ = false;
        thread_ = std::thread([this] { run(); });
        running_ = true;
        return true;
    }

    void push(std::string line)
    {
        std::lock_guard lock(mutex_);
        // Lifecycle state lives under the mutex: the thread object itself is
        // joined outside it by shutdown().
        if (!running_ || stop_ || failed_) return;
        // Room for the record and, when records were dropped, the marker
        // announcing them first.
        if (queue_.size() + (dropped_ ? 2 : 1) > kMaxQueued) { ++dropped_; return; }
        if (dropped_) {
            queue_.push_back("event 0 " + threadName() + " " + std::to_string(nowNs()) + " metrics_dropped count=" +
                             std::to_string(dropped_));
            dropped_ = 0;
        }
        queue_.push_back(std::move(line));
        wake_.notify_one();
    }

    bool quitMarked()
    {
        std::lock_guard lock(mutex_);
        return quitMarked_;
    }

    void markQuit()
    {
        std::lock_guard lock(mutex_);
        quitMarked_ = true;
    }

    void shutdown()
    {
        {
            std::lock_guard lock(mutex_);
            if (!running_) return;
            running_ = false;
            stop_ = true;
            wake_.notify_all();
        }
        thread_.join();
        std::lock_guard lock(mutex_);
        // A stream that failed earlier (disk full) gets one more chance to
        // record how the run ended.
        log_.clear();
        drain();
        if (dropped_) {
            log_ << "event 0 gui " << nowNs() << " metrics_dropped count=" << dropped_ << '\n';
            dropped_ = 0;
        }
        // The end marker: a log without it is an incomplete run.
        log_ << "header " << nowNs() << " end=1 failures=" << g_failures.load() << '\n';
        log_.flush();
        if (!log_) g_failures.fetch_add(1);
        log_.close();
        if (log_.fail()) g_failures.fetch_add(1);
    }

private:
    static constexpr size_t kMaxQueued = 200000;

    void run() noexcept
    {
        // The batch is swapped in under the lock and written, then emptied,
        // outside it: no allocation and no I/O while producers wait. A
        // failure that is not a stream error (allocation, anywhere in this
        // thread) ends recording for good rather than retrying with the lock
        // held; nothing leaves the thread entry.
        try {
            std::deque<std::string> batch;
            std::unique_lock lock(mutex_);
            try {
                while (true) {
                    wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                    if (stop_ && queue_.empty()) return;
                    batch.swap(queue_);
                    lock.unlock();
                    for (const auto& line : batch) log_ << line << '\n';
                    log_.flush();
                    if (!log_) g_failures.fetch_add(1);
                    batch.clear();
                    lock.lock();
                }
            } catch (...) {
                g_failures.fetch_add(1);
                if (!lock.owns_lock()) lock.lock();
                failed_ = true;
                queue_.clear();
            }
        } catch (...) {
            g_failures.fetch_add(1);
            failed_ = true;
        }
    }

    // Caller holds mutex_; the writer thread has stopped.
    void drain()
    {
        for (const auto& line : queue_) log_ << line << '\n';
        queue_.clear();
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::string> queue_;
    std::thread thread_;
    bool stop_ = false;
    bool running_ = false;  // under mutex_: producers may submit
    std::atomic<bool> failed_{false};  // set from the writer's failure path without the mutex
    bool quitMarked_ = false;
    std::ofstream log_;
    uint64_t dropped_ = 0;
};

Writer& writer()
{
    static Writer instance;
    return instance;
}

void write(std::string line)
{
    try {
        writer().push(std::move(line));
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

void spanRecord(char kind, uint64_t id, uint64_t parent, int64_t ns, std::string_view name, const std::string& attrs)
{
    std::string line = "span ";
    line += kind; line += ' ';
    line += std::to_string(id); line += ' ';
    line += std::to_string(parent); line += ' ';
    line += threadName(); line += ' ';
    line += std::to_string(ns); line += ' ';
    line += name;
    if (!attrs.empty()) { line += ' '; line += attrs; }
    write(std::move(line));
}

}  // namespace

bool enabled() { return g_enabled.load(std::memory_order_relaxed); }

void enable(const std::filesystem::path& logPath)
{
    try {
        if (writer().open(logPath)) g_enabled.store(true, std::memory_order_relaxed);
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

void shutdown()
{
    // One caller finalizes; a concurrent or repeated call is a no-op.
    if (!g_enabled.exchange(false, std::memory_order_acq_rel)) return;
    // A run that ends without a quit request (the window was closed) ends
    // its measurement here: what follows is nothing.
    try {
        if (!writer().quitMarked()) writer().push("header " + std::to_string(nowNs()) + " quit=1");
    } catch (...) { g_failures.fetch_add(1); }
    // Stopped and joined whatever happened to the marker.
    try { writer().shutdown(); } catch (...) { g_failures.fetch_add(1); }
}

uint64_t failures() { return g_failures.load(std::memory_order_relaxed); }

void recordFailure() { g_failures.fetch_add(1); }

void nameThread(std::string_view name)
{
    try { t_thread = std::string(name); } catch (...) { g_failures.fetch_add(1); }
}

void header(const std::string& attrs)
{
    if (!enabled()) return;
    try {
        if (attrs == "quit=1") writer().markQuit();
        write("header " + std::to_string(nowNs()) + " " + attrs);
    } catch (...) { g_failures.fetch_add(1); }
}

void startHeartbeat(QObject* parent, int intervalMs)
{
    if (!enabled()) return;
    try {
        nameThread("gui");
        auto* timer = new QTimer(parent);
        timer->setTimerType(Qt::PreciseTimer);
        timer->setInterval(intervalMs);
        auto last = std::make_shared<int64_t>(nowNs());
        QObject::connect(timer, &QTimer::timeout, parent, [last] {
            if (!enabled()) return;
            try {
                const auto now = nowNs();
                write("hb " + std::to_string(now) + " " + std::to_string(now - *last));
                *last = now;
            } catch (...) {
                g_failures.fetch_add(1);
            }
        });
        timer->start();
        header("heartbeat_ms=" + std::to_string(intervalMs));
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

uint64_t begin(std::string_view name, const std::string& attrs)
{
    if (!enabled()) return 0;
    try {
        const uint64_t id = g_nextId.fetch_add(1);
        const uint64_t parent = t_stack.empty() ? 0 : t_stack.back();
        // The record is built before the span is opened: a failure leaves
        // the stack as it was and the caller holds no span.
        std::string line = "span B ";
        line += std::to_string(id); line += ' ';
        line += std::to_string(parent); line += ' ';
        line += threadName(); line += ' ';
        line += std::to_string(nowNs()); line += ' ';
        line += name;
        if (!attrs.empty()) { line += ' '; line += attrs; }
        t_stack.push_back(id);
        write(std::move(line));
        return id;
    } catch (...) {
        g_failures.fetch_add(1);
        return 0;
    }
}

void end(const std::string& attrs)
{
    // The stack is maintained whether or not recording is still on, so an
    // end after shutdown (or a loop below) always makes progress.
    if (t_stack.empty()) return;
    const uint64_t id = t_stack.back();
    t_stack.pop_back();
    if (!enabled()) return;
    try {
        const uint64_t parent = t_stack.empty() ? 0 : t_stack.back();
        spanRecord('E', id, parent, nowNs(), "-", attrs);
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

void end(uint64_t id, const std::string& attrs)
{
    if (!id || std::find(t_stack.begin(), t_stack.end(), id) == t_stack.end()) return;
    while (!t_stack.empty() && t_stack.back() != id) end("unbalanced=1");
    end(attrs);
}

std::string word(std::string_view value)
{
    // One record per line, one word per attribute: every control character
    // and space becomes '_' (lossy, like the rest of the format; attribute
    // values are identifiers, counts and paths).
    std::string result(value);
    for (auto& c : result) if (static_cast<unsigned char>(c) <= ' ' || c == 0x7f) c = '_';
    return result.empty() ? "-" : result;
}

uint64_t current() { return t_stack.empty() ? 0 : t_stack.back(); }

void event(std::string_view name, const std::string& attrs)
{
    if (!enabled()) return;
    try {
        std::string line = "event ";
        line += std::to_string(current()); line += ' ';
        line += threadName(); line += ' ';
        line += std::to_string(nowNs()); line += ' ';
        line += name;
        if (!attrs.empty()) { line += ' '; line += attrs; }
        write(std::move(line));
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

Span::Span(std::string_view name, const std::string& attrs) : id_(begin(name, attrs)) {}

Span::Span(std::string_view name, const char* attrs) : id_(0)
{
    if (!enabled()) return;
    try { id_ = begin(name, std::string(attrs)); } catch (...) { g_failures.fetch_add(1); }
}

Span::~Span()
{
    if (id_) end(id_, endAttrs_);
}

void Span::attr(const char* keyValue)
{
    if (!id_) return;
    try { attr(std::string(keyValue)); } catch (...) { g_failures.fetch_add(1); }
}

void Span::attr(const std::string& keyValue)
{
    if (!id_ || keyValue.empty()) return;
    try {
        if (!endAttrs_.empty()) endAttrs_ += ' ';
        endAttrs_ += keyValue;
    } catch (...) {
        g_failures.fetch_add(1);
    }
}

}  // namespace vc3d::metrics
