#pragma once

// Span log for the line-annotation publication path (clicks, landings,
// commits, the Committed handler and its consumers, hub sync and lease
// waits). Enabled by `VC3D --publication-metrics <file>`; every call is a
// cheap no-op otherwise. `scripts/publication_metrics.py` reads the file.
//
// Line format (space separated, one record per line, times are
// steady_clock nanoseconds):
//   header <key=value ...>
//   span B <id> <parent> <thread> <ns> <name> [key=value ...]
//   span E <id> <parent> <thread> <ns> <name> [key=value ...]
//   event <parent> <thread> <ns> <name> [key=value ...]
//   hb <ns> <gap_ns>
// Spans nest per thread: a span's parent is the innermost open span on the
// same thread when it begins. Every heartbeat sample is recorded, so the
// script's gap distribution has the full denominator.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

class QObject;

namespace vc3d::metrics {

bool enabled();
void enable(const std::filesystem::path& logPath);
// Stops the writer thread and flushes; later calls are no-ops. Records are
// queued (bounded) to a writer thread, never written on the calling thread;
// dropped records are announced in the log as `event metrics_dropped`.
void shutdown();
// Recording failures so far (allocation or write errors, never thrown).
uint64_t failures();
// Counts a recording failure the caller contained itself.
void recordFailure();
// Attribute text built only while recording, inside a failure-counting
// boundary: call sites never format (or allocate) for a disabled recorder,
// and a failing builder cannot reach application code.
template <class F>
std::string attrs(F&& build)
{
    if (!enabled()) return {};
    try { return std::string(build()); } catch (...) { recordFailure(); return {}; }
}
// Names the calling thread in the log (the GUI thread calls this once).
void nameThread(std::string_view name);
// Starts the GUI-thread heartbeat; `parent` owns the timer.
void startHeartbeat(QObject* parent, int intervalMs = 20);
void header(const std::string& attrs);

uint64_t begin(std::string_view name, const std::string& attrs = {});
// Ends the innermost open span on this thread.
void end(const std::string& attrs = {});
// Ends `id`; spans left open above it (an exception skipped their end) are
// closed too, marked unbalanced=1.
void end(uint64_t id, const std::string& attrs);
// Attribute values are single words: whitespace becomes '_'.
std::string word(std::string_view value);
// Innermost open span on this thread, 0 when none.
uint64_t current();
void event(std::string_view name, const std::string& attrs = {});

class Span {
public:
    // Attributes: a literal, or a builder evaluated only while recording.
    explicit Span(std::string_view name, const char* attrs = "");
    Span(std::string_view name, const std::string& attrs);
    template <class F>
        requires std::is_invocable_v<F&>
    Span(std::string_view name, F&& build) : Span(name, metrics::attrs(std::forward<F>(build))) {}
    ~Span();
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
    // Appended to the end record: a literal, or a builder evaluated only
    // while this span records.
    void attr(const char* keyValue);
    void attr(const std::string& keyValue);
    template <class F>
        requires std::is_invocable_v<F&>
    void attr(F&& build) { if (id_) attr(metrics::attrs(std::forward<F>(build))); }
    uint64_t id() const { return id_; }

private:
    uint64_t id_ = 0;
    std::string endAttrs_;
};

}  // namespace vc3d::metrics
