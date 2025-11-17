#include "vc/core/util/Logging.hpp"

#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>

// Internal implementation class
class LoggerImpl {
public:
    std::mutex mutex;
    std::vector<std::shared_ptr<std::ofstream>> file_sinks;

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm tm;
        #ifdef _WIN32
        localtime_s(&tm, &time_t);
        #else
        localtime_r(&time_t, &tm);
        #endif

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }
};

MinimalLogger::MinimalLogger()
    : current_level_(LogLevel::Info)
    , impl_(std::make_unique<LoggerImpl>())
{
}

MinimalLogger::~MinimalLogger() = default;

const char* MinimalLogger::level_string(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        default:              return "?????";
    }
}

void MinimalLogger::write_log(LogLevel level, const std::string& msg) {
    if (level < current_level_) return;

    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::string timestamp = impl_->get_timestamp();
    std::string formatted = std::format("[{}] [{}] {}", timestamp, level_string(level), msg);

    // Always write to stdout
    std::cout << formatted << std::endl;

    // Write to file sinks
    for (auto& file : impl_->file_sinks) {
        if (file && file->is_open()) {
            (*file) << formatted << std::endl;
            file->flush();
        }
    }
}

void MinimalLogger::debug(const std::string& msg) {
    write_log(LogLevel::Debug, msg);
}

void MinimalLogger::info(const std::string& msg) {
    write_log(LogLevel::Info, msg);
}

void MinimalLogger::warn(const std::string& msg) {
    write_log(LogLevel::Warn, msg);
}

void MinimalLogger::error(const std::string& msg) {
    write_log(LogLevel::Error, msg);
}

void MinimalLogger::set_level(LogLevel level) {
    current_level_ = level;
}

void MinimalLogger::add_file(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto file = std::make_shared<std::ofstream>(path, std::ios::app);
    if (file->is_open()) {
        impl_->file_sinks.push_back(file);
    }
}

// Global functions
auto Logger() -> std::shared_ptr<MinimalLogger> {
    static auto logger = std::make_shared<MinimalLogger>();
    return logger;
}

void AddLogFile(const std::filesystem::path& path) {
    Logger()->add_file(path);
}

void SetLogLevel(const std::string& s) {
    LogLevel level = LogLevel::Info;

    if (s == "debug" || s == "DEBUG") {
        level = LogLevel::Debug;
    } else if (s == "info" || s == "INFO") {
        level = LogLevel::Info;
    } else if (s == "warn" || s == "WARN" || s == "warning" || s == "WARNING") {
        level = LogLevel::Warn;
    } else if (s == "error" || s == "ERROR") {
        level = LogLevel::Error;
    } else if (s == "off" || s == "OFF") {
        level = LogLevel::Off;
    }

    Logger()->set_level(level);
}
