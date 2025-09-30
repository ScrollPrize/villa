#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vc::util {

// Internal: fill `out` with UTC broken-down time in a thread-safe way.
inline bool tm_from_utc(std::time_t tt, std::tm& out) noexcept {
#if defined(_WIN32)
    // Windows secure variant
    return ::gmtime_s(&out, &tt) == 0;
#elif defined(__unix__) || defined(__APPLE__)
    // POSIX reentrant variant
    return ::gmtime_r(&tt, &out) != nullptr;
#else
#  error "No thread-safe gmtime available on this platform. Provide an implementation or switch to C++20 chrono calendar formatting."
#endif
}

// Format a system_clock::time_point as UTC using strftime-style `fmt`
// Example fmt: "%Y-%m-%dT%H:%M:%SZ"
inline std::string format_utc(std::chrono::system_clock::time_point tp,
                              const char* fmt) {
    const std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    if (!tm_from_utc(tt, tm)) {
        throw std::runtime_error("UTC conversion failed");
    }
    std::ostringstream oss;
    oss << std::put_time(&tm, fmt);
    return oss.str();
}

// Convenience: ISO‑8601 UTC timestamp for "now"
inline std::string iso8601_utc_now() {
    return format_utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
}

} // namespace vc::util
