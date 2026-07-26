#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <format>

// Forward declaration
class MinimalLogger;

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
void SetDebugLoggingEnabled(bool enabled);
bool DebugLoggingEnabled();
void SetProfileLoggingEnabled(bool enabled);
bool ProfileLoggingEnabled();
auto Logger() -> std::shared_ptr<MinimalLogger>;

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Off = 4
};

class MinimalLogger {
public:
    MinimalLogger();
    ~MinimalLogger();

    // Variadic template logging methods
    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Info, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Warn, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Error, fmt, std::forward<Args>(args)...);
    }

    // Simple string overloads
    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);

    void set_level(LogLevel level);
    void add_file(const std::filesystem::path& path);
    void clear_files();

private:
    // The template stays trivial — make_format_args is cheap. The heavy
    // std::vformat instantiation (the bulk of <format>'s compile cost) lives
    // once in vlog() in Logging.cpp instead of in every TU that logs.
    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
        if (level < current_level_) return;
        vlog(level, fmt.get(), std::make_format_args(args...));
    }

    void vlog(LogLevel level, std::string_view fmt, std::format_args args);
    void write_log(LogLevel level, const std::string& msg);
    const char* level_string(LogLevel level);

    LogLevel current_level_;
    std::unique_ptr<class LoggerImpl> impl_;
};
