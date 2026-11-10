#include "helirec/progress.hpp"

#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace helirec {

namespace {

using Clock = std::chrono::steady_clock;

// MM:SS, or H:MM:SS from an hour on
std::string interval(double seconds) {
    const long long s = std::max(0LL, std::llround(seconds));
    char buf[32];
    if (s >= 3600) std::snprintf(buf, sizeof buf, "%lld:%02lld:%02lld", s / 3600, s / 60 % 60, s % 60);
    else std::snprintf(buf, sizeof buf, "%02lld:%02lld", s / 60, s % 60);
    return buf;
}

// columns of a UTF-8 string (code points)
int columns(const std::string& s) {
    int n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

bool utf8_locale() {
    for (const char* name : {"LC_ALL", "LC_CTYPE", "LANG"}) {
        const char* v = std::getenv(name);
        if (!v || !*v) continue;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s.find("utf-8") != std::string::npos || s.find("utf8") != std::string::npos;
    }
    return false;
}

void write_all(int fd, const std::string& s) {
    const char* p = s.data();
    std::size_t left = s.size();
    while (left) {
        const ssize_t k = ::write(fd, p, left);
        if (k <= 0) return;   // a closed or broken stream: progress output is best effort
        p += k;
        left -= static_cast<std::size_t>(k);
    }
}

}  // namespace

std::string ProgressBar::format(std::uint64_t n, std::uint64_t total, double elapsed, double rate, const std::string& desc,
                                const std::string& unit, const std::string& postfix, int width, bool unicode) {
    char buf[96];
    std::string speed = "?";
    if (rate > 0) {
        if (rate >= 100) std::snprintf(buf, sizeof buf, "%.0f %s/s", rate, unit.c_str());
        else if (rate >= 1) std::snprintf(buf, sizeof buf, "%.1f %s/s", rate, unit.c_str());
        else std::snprintf(buf, sizeof buf, "%.1f s/%s", 1.0 / rate, unit.c_str());
        speed = buf;
    }
    const std::string head = desc.empty() ? std::string() : desc + ": ";
    const std::string tail = postfix.empty() ? std::string() : ", " + postfix;
    if (!total) return head + std::to_string(n) + " " + unit + " [" + interval(elapsed) + ", " + speed + tail + "]";
    const std::uint64_t done = std::min(n, total);
    const double frac = static_cast<double>(done) / static_cast<double>(total);
    const std::string remaining = done == total ? "00:00" : rate > 0 ? interval(static_cast<double>(total - done) / rate) : "?";
    std::snprintf(buf, sizeof buf, "%3d%%", static_cast<int>(frac * 100));
    const std::string left = head + buf;
    const std::string right = " " + std::to_string(n) + "/" + std::to_string(total) + " [" + interval(elapsed) + "<" +
                              remaining + ", " + speed + tail + "]";
    const int room = width - columns(left) - columns(right) - 2;   // the two '|'
    if (room < 1) return left + right;
    // the bar in eighths of a column (unicode) or tenths (ascii, as a digit)
    static const char* const eighths[] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    const int steps = unicode ? 8 : 10;
    const long long filled = static_cast<long long>(frac * room * steps);
    const int full = static_cast<int>(filled / steps), part = static_cast<int>(filled % steps);
    std::string bar;
    for (int i = 0; i < full; ++i) bar += unicode ? "█" : "#";
    int used = full;
    if (used < room && part) {
        bar += unicode ? std::string(eighths[part]) : std::string(1, static_cast<char>('0' + part));
        ++used;
    }
    bar.append(static_cast<std::size_t>(room - used), ' ');
    return left + "|" + bar + "|" + right;
}

struct ProgressBar::Impl {
    Options o;
    std::uint64_t total = 0;
    bool terminal = false, unicode = false;
    Clock::time_point t0 = Clock::now();
    std::atomic<std::uint64_t> n{0};
    std::atomic<double> next_draw{0.0};   // update() draws from this elapsed time on
    std::mutex m;                         // everything below, and the stream
    std::string postfix;
    std::deque<std::pair<double, std::uint64_t>> samples{{0.0, 0}};   // (elapsed, count), about a second apart
    bool closed = false;
    std::condition_variable cv;
    std::thread ticker;

    double elapsed() const { return std::chrono::duration<double>(Clock::now() - t0).count(); }

    int width() const {
        if (o.width > 0) return o.width;
        if (terminal) {
            winsize ws{};
            if (::ioctl(o.fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
        }
        return 80;
    }

    std::string text(double t) {
        const std::uint64_t k = n.load();
        if (t - samples.back().first >= 1.0) samples.push_back({t, k});
        while (samples.size() > 1 && samples[1].first <= t - o.rate_window) samples.pop_front();
        const double dt = t - samples.front().first;
        const double rate = dt > 0 && k > samples.front().second ? static_cast<double>(k - samples.front().second) / dt : 0.0;
        return format(k, total, t, rate, o.desc, o.unit, postfix, width(), unicode);
    }

    // (m held)
    void draw() {
        const double t = elapsed();
        next_draw.store(t + (terminal ? o.min_interval : o.log_interval));
        write_all(o.fd, terminal ? "\r" + text(t) + "\x1b[K" : text(t) + "\n");
    }
};

ProgressBar::ProgressBar(std::uint64_t total, Options options) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.o = std::move(options);
    I.total = total;
    I.terminal = I.o.terminal < 0 ? ::isatty(I.o.fd) == 1 : I.o.terminal != 0;
    I.unicode = utf8_locale();
    {
        std::lock_guard<std::mutex> lock(I.m);
        I.draw();
    }
    if (I.terminal)
        I.ticker = std::thread([&I] {
            std::unique_lock<std::mutex> lock(I.m);
            while (!I.cv.wait_for(lock, std::chrono::seconds(1), [&I] { return I.closed; })) I.draw();
        });
}

ProgressBar::ProgressBar(std::uint64_t total, std::string desc, std::string unit)
    : ProgressBar(total, [&] {
          Options o;
          o.desc = std::move(desc);
          o.unit = std::move(unit);
          return o;
      }()) {}

ProgressBar::~ProgressBar() { close(); }

void ProgressBar::update(std::uint64_t n) {
    auto& I = *impl_;
    I.n.fetch_add(n);
    if (I.elapsed() < I.next_draw.load()) return;
    std::lock_guard<std::mutex> lock(I.m);
    if (!I.closed) I.draw();
}

void ProgressBar::set_postfix(const std::string& text) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed || I.postfix == text) return;
    I.postfix = text;
    if (I.terminal) I.draw();
}

void ProgressBar::set_description(const std::string& desc) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed) return;
    I.o.desc = desc;
    if (I.terminal) I.draw();
}

void ProgressBar::write(const std::string& line) {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (I.closed || !I.terminal) {
        write_all(I.o.fd, line + "\n");
        return;
    }
    write_all(I.o.fd, "\r\x1b[K" + line + "\n");
    I.draw();
}

void ProgressBar::refresh() {
    auto& I = *impl_;
    std::lock_guard<std::mutex> lock(I.m);
    if (!I.closed) I.draw();
}

void ProgressBar::close() {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> lock(I.m);
        if (I.closed) return;
        I.closed = true;
        if (!I.terminal) write_all(I.o.fd, I.text(I.elapsed()) + "\n");
        else if (I.o.leave) write_all(I.o.fd, "\r" + I.text(I.elapsed()) + "\x1b[K\n");
        else write_all(I.o.fd, "\r\x1b[K");
    }
    I.cv.notify_all();
    if (I.ticker.joinable()) I.ticker.join();
}

std::uint64_t ProgressBar::count() const { return impl_->n.load(); }
std::uint64_t ProgressBar::total() const { return impl_->total; }

}  // namespace helirec
