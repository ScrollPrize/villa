#pragma once
#include "vc/core/util/Progress.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace vc::util {
// Extracted from vc_zarr_download_region's terminal progress renderer.
inline std::string progressBar(std::uint64_t done, std::uint64_t total)
{
    constexpr int width = 24;
    const double fraction = total ? std::clamp(double(done) / double(total), 0.0, 1.0) : 0;
    const auto filled = static_cast<size_t>(std::round(fraction * width));
    return "[" + std::string(filled, '#') + std::string(width - filled, '-') + "]";
}

inline bool stderrIsTerminal()
{
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

// One instance per operation. Owns a heartbeat thread so slow I/O, index builds,
// and individual work items still display activity even without new callbacks.
// Output stays on stderr; redirected output is newline-delimited and throttled.
class TerminalProgress {
public:
    explicit TerminalProgress(std::string title, bool enabled = true,
        std::ostream& output = std::cerr, bool terminal = stderrIsTerminal(),
        std::chrono::milliseconds interval = std::chrono::milliseconds(0))
        : title_(std::move(title)), output_(output), enabled_(enabled), terminal_(terminal),
          interval_(interval.count() > 0 ? interval : terminal ? std::chrono::milliseconds(200)
                                                             : std::chrono::milliseconds(5000))
    {
        if (!enabled_) return;
        render();
        worker_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!wake_.wait_for(lock, interval_, [this] { return stopped_; })) render();
        });
    }
    TerminalProgress(const TerminalProgress&) = delete;
    TerminalProgress& operator=(const TerminalProgress&) = delete;
    ~TerminalProgress() { stop("interrupted"); }

    ProgressCallback callback()
    {
        if (!enabled_) return {};
        return [this](const ProgressUpdate& event) { update(event); };
    }
    void update(const ProgressUpdate& event)
    {
        if (!enabled_) return;
        std::lock_guard lock(mutex_);
        if (stopped_) return;
        const bool changed = stage_ != event.stage || total_ != event.total || event.completed < completed_;
        if (changed) stageStart_ = Clock::now();
        stage_ = clean(event.stage);
        detail_ = clean(event.detail);
        completed_ = event.completed;
        total_ = event.total;
        // Coalesce rapid transitions too. Rendering every stage change or item
        // completion makes nested work flash and floods redirected logs.
        if (Clock::now() - lastPrint_ >= interval_) render();
    }
    // Call after the operation returns, before printing its result summary.
    // "finished" means the call returned; the caller's report describes problems.
    void finish() { stop("finished"); }

private:
    using Clock = std::chrono::steady_clock;
    static std::string clean(std::string_view value)
    {
        std::string result(value.substr(0, 100));
        for (auto& c : result) if (static_cast<unsigned char>(c) < 32) c = ' ';
        return result;
    }
    void render(std::string_view status = {})
    {
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - started_).count();
        std::ostringstream line;
        line << title_ << ": " << stage_ << ' ';
        if (total_) {
            line << progressBar(completed_, total_) << ' ' << std::fixed << std::setprecision(1)
                 << std::min(100.0, 100.0 * double(completed_) / double(total_)) << "% "
                 << completed_ << '/' << total_;
            const double seconds = std::chrono::duration<double>(now - stageStart_).count();
            if (completed_ && completed_ < total_ && seconds >= 1)
                line << " ETA " << std::setprecision(0) << seconds * double(total_ - completed_) / double(completed_) << 's';
        } else line << "|/-\\"[ticks_++ % 4];
        line << " elapsed " << elapsed << 's';
        if (!detail_.empty()) line << " — " << detail_;
        if (!status.empty()) line << " (" << status << ')';
        auto text = line.str();
        if (terminal_) {
            size_t columns = 120;
#ifndef _WIN32
            winsize size{};
            if (ioctl(fileno(stderr), TIOCGWINSZ, &size) == 0 && size.ws_col > 4) columns = size.ws_col;
#endif
            // Leave the final column unused to avoid wrapping carriage-return output.
            if (text.size() >= columns) text = text.substr(0, columns - 4) + "...";
            output_ << '\r' << text << std::string(previousWidth_ > text.size() ? previousWidth_ - text.size() : 0, ' ');
            previousWidth_ = text.size();
            if (!status.empty()) output_ << '\n';
        } else output_ << text << '\n';
        output_.flush();
        lastPrint_ = now;
    }
    void stop(std::string_view status)
    {
        if (!enabled_) return;
        {
            std::lock_guard lock(mutex_);
            if (stopped_) return;
            stopped_ = true;
            render(status);
        }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    std::string title_, stage_ = "starting", detail_;
    std::ostream& output_;
    bool enabled_, terminal_, stopped_ = false;
    std::chrono::milliseconds interval_;
    Clock::time_point started_ = Clock::now(), stageStart_ = started_, lastPrint_ = started_;
    std::uint64_t completed_ = 0, total_ = 0, ticks_ = 0;
    size_t previousWidth_ = 0;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
};
} // namespace vc::util
