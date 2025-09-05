#pragma once

/**
 * @file DateTime.hpp
 *
 * @ingroup Core
 */

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>


/**
 * @brief Returns a string representation of the current date and time
 *
 * Format: YYYYMMDDHMS
 *
 * Code from <a href="http://stackoverflow.com/a/16358264/1917043">here</a>.
 */
inline auto DateTime() -> std::string
{
    const auto now = std::time(nullptr);
    const auto* time = std::localtime(&now);

    std::stringstream ss;
    ss << std::put_time(time, "%Y%m%d%H%M%S");
    return ss.str();
}

/**
 * @brief Returns a string representation of the provided time duration
 *
 * Format: dd:hh:mm:ss
 *
 * Code from <a href="https://stackoverflow.com/a/46134506">here</a>.
 */
template <typename Rep, typename Period>
auto DurationToString(std::chrono::duration<Rep, Period> input) -> std::string
{
    using namespace std::chrono;
    using days = duration<int, std::ratio<86400>>;
    const auto d = duration_cast<days>(input);
    input -= d;
    const auto h = duration_cast<hours>(input);
    input -= h;
    const auto m = duration_cast<minutes>(input);
    input -= m;
    const auto s = duration_cast<seconds>(input);

    const auto dc = d.count();
    const auto hc = h.count();
    const auto mc = m.count();

    std::stringstream ss;
    ss.fill('0');
    if (dc) {
        ss << d.count() << "d";
    }
    if (dc || hc) {
        if (dc) {
            ss << std::setw(2);
        }  // pad if second set of numbers
        ss << h.count() << "h";
    }
    if (dc || hc || mc) {
        if (dc || hc) {
            ss << std::setw(2);
        }
        ss << m.count() << "m";
    }

    // Always display seconds
    if (dc || hc || mc) {
        ss << std::setw(2);
    }
    ss << s.count() << 's';

    return ss.str();
}

/**
 * @brief Convert a duration string to a a std::chrono duration.
 *
 * Accepts strings with ordered combinations of `#h`, `#m`, `#s`, and `#ms`.
 * Examples: `10s`, `500ms`, `1m30s`.
 *
 * @tparam Duration Output duration type
 * @param str Input string
 */
template <typename Duration = std::chrono::milliseconds>
auto DurationFromString(const std::string& str) -> Duration
{
    using namespace std::chrono;
    const std::regex dReg{
        R"((\d+h){0,1}(\d+m(?!s)){0,1}(\d+s){0,1}(\d+ms){0,1})"};
    std::smatch match;
    Duration d(0);
    if (std::regex_search(str, match, dReg)) {
        for (std::size_t i = 1; i < 5; i++) {
            if (match[i].length() == 0) {
                continue;
            }
            auto part = match[i].str();
            auto end = (i == 4) ? part.size() - 2 : part.size() - 1;
            auto val = std::stoi(part.substr(0, end));
            if (i == 1) {
                d += std::chrono::hours(val);
            } else if (i == 2) {
                d += std::chrono::minutes(val);
            } else if (i == 3) {
                d += std::chrono::seconds(val);
            } else if (i == 4) {
                d += std::chrono::milliseconds(val);
            }
        }
    } else {
        throw std::invalid_argument("Cannot parse as duration: " + str);
    }
    return d;
}



static inline std::string get_surface_time_str()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y%m%d%H%M%S"); // HH:MM:SS
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}
