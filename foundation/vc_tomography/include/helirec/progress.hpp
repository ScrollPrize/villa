#pragma once
// Terminal progress bar in the style of tqdm, for C++ loops and (bound as helirec.ProgressBar) Python ones:
//
//   reconstruct:  42%|████████▍           | 2272/5398 [03:12<04:25, 11.8 slice/s, backproject]
//
// On a terminal the bar is redrawn in place: at most every `min_interval` seconds as it advances, and once a second
// while it does not, so the elapsed time keeps running through a long step. When the stream is not a terminal (a log
// file, a pipe) it prints one line every `log_interval` seconds instead, and the final one. All methods are
// thread-safe: update() can be called from the workers of a parallel loop.
#include <cstdint>
#include <memory>
#include <string>

namespace helirec {

class ProgressBar {
  public:
    struct Options {
        std::string desc;            // text before the bar
        std::string unit = "it";     // what is counted
        int fd = 2;                  // file descriptor written to (stderr)
        double min_interval = 0.1;   // seconds between redraws on a terminal
        double log_interval = 10.0;  // seconds between lines when the stream is not a terminal
        double rate_window = 60.0;   // the rate (and the remaining time) is the average over this many seconds
        int width = 0;               // columns; 0: the terminal's (80 when it cannot be asked)
        int terminal = -1;           // 1 / 0: redraw in place / print lines; -1: isatty(fd)
        bool leave = true;           // keep the finished bar on screen
    };
    ProgressBar(std::uint64_t total, Options options);
    explicit ProgressBar(std::uint64_t total, std::string desc = {}, std::string unit = "it");
    ~ProgressBar();   // close()
    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    void update(std::uint64_t n = 1);               // n more are done
    void set_postfix(const std::string& text);      // text after the rate (e.g. the current step); redraws
    void set_description(const std::string& desc);
    void write(const std::string& line);            // a line of text above the bar
    void refresh();                                 // redraw now
    void close();                                   // final state; further calls do nothing
    std::uint64_t count() const;
    std::uint64_t total() const;

    // The meter text in `width` columns: `rate` in units per second (<= 0: unknown). total = 0: no bar, a count only.
    static std::string format(std::uint64_t n, std::uint64_t total, double elapsed, double rate, const std::string& desc,
                              const std::string& unit, const std::string& postfix, int width, bool unicode);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
