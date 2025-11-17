#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <format>

// Forward declaration
class MinimalLogger;

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
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

private:
    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
        if (level < current_level_) return;
        std::string msg = std::format(fmt, std::forward<Args>(args)...);
        write_log(level, msg);
    }

    void write_log(LogLevel level, const std::string& msg);
    const char* level_string(LogLevel level);

    LogLevel current_level_;
    std::unique_ptr<class LoggerImpl> impl_;
};
