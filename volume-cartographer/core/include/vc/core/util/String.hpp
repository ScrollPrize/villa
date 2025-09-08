#pragma once

/** @file */

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>



/** @brief Convert string characters to upper case (in place) */
static inline void to_upper(std::string& s)
{
    const auto& f = std::use_facet<std::ctype<char>>(std::locale());
    f.toupper(s.data(), s.data() + s.size());
}

/** @brief Convert string characters to upper case (r-value) */
static inline std::string to_upper(std::string&& s)
{
    to_upper(s);
    return s;
}

/** @brief Convert string characters to upper case (copy) */
static inline std::string to_upper_copy(std::string s)
{
    to_upper(s);
    return s;
}

/** @brief Convert string characters to lower case (in place) */
static inline void to_lower(std::string& s)
{
    const auto& f = std::use_facet<std::ctype<char>>(std::locale());
    f.tolower(s.data(), s.data() + s.size());
}

/** @brief Convert string characters to lower case (r-value) */
static inline std::string to_lower(std::string&& s)
{
    to_lower(s);
    return s;
}

/** @brief Convert string characters to lower case (copy) */
static inline std::string to_lower_copy(std::string s)
{
    to_lower(s);
    return s;
}

/** @brief Convert a string to a bool */
static inline auto to_bool(const std::string& s) -> bool
{
    bool b{false};
    if (s.size() == 1) {
        std::stringstream(s) >> b;
    } else if (s.size() > 1) {
        std::stringstream(s) >> std::boolalpha >> b;
    }
    return b;
}

/**
 * @brief Left trim (in place)
 *
 * https://stackoverflow.com/a/217605
 */
static inline void trim_left(std::string& s)
{
    const auto& loc = std::locale();
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), [&loc](auto ch) {
                return std::isspace(ch, loc);
            }));
}

/**
 * @brief Left trim (r-value)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim_left(std::string&& s)
{
    trim_left(s);
    return s;
}

/**
 * @brief Left trim (copy)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim_left_copy(std::string s)
{
    trim_left(s);
    return s;
}

/**
 * @brief Right trim (in place)
 *
 * https://stackoverflow.com/a/217605
 */
static inline void trim_right(std::string& s)
{
    const auto& loc = std::locale();
    s.erase(
        std::find_if_not(
            s.rbegin(), s.rend(),
            [&loc](auto ch) { return std::isspace(ch, loc); })
            .base(),
        s.end());
}

/**
 * @brief Right trim (r-value)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim_right(std::string&& s)
{
    trim_right(s);
    return s;
}

/**
 * @brief Right trim (copy)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim_right_copy(std::string s)
{
    trim_right(s);
    return s;
}

/**
 * @brief Trim from both ends (in place)
 *
 * https://stackoverflow.com/a/217605
 */
static inline void trim(std::string& s)
{
    trim_left(s);
    trim_right(s);
}

/**
 * @brief Trim from both ends (r-value)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim(std::string&& s)
{
    trim(s);
    return s;
}

/**
 * @brief Right trim (copy)
 *
 * https://stackoverflow.com/a/217605
 */
static inline std::string trim_copy(std::string s)
{
    trim(s);
    return s;
}

/** @brief Split a string by a delimiter */
template <typename... Ds>
static inline std::vector<std::string> split(
    const std::string& s, const Ds&... ds)
{
    // Build delimiters list
    std::vector<char> delims;
    if (sizeof...(ds)) {
        delims = {ds...};
    } else {
        delims.emplace_back(' ');
    }

    // Get a list of all delimiter start positions
    std::vector<std::string::size_type> delimPos;
    for (const auto& delim : delims) {
        auto b = s.find(delim, 0);
        while (b != std::string::npos) {
            delimPos.emplace_back(b);
            b = s.find(delim, b + 1);
        }
    }

    // Sort the delimiter start positions
    std::sort(delimPos.begin(), delimPos.end());

    // Split string
    std::vector<std::string> tokens;
    std::string::size_type begin{0};
    for (std::size_t it = 0; it < delimPos.size(); it++) {
        auto end = delimPos[it];
        auto t = s.substr(begin, end - begin);
        if (not t.empty()) {
            tokens.emplace_back(t);
        }
        begin = end + 1;
    }
    auto t = s.substr(begin);
    if (not t.empty()) {
        tokens.emplace_back(t);
    }

    return tokens;
}

/** @brief Partition a string by a separator substring */
static inline auto partition(std::string_view s, std::string_view sep)
    -> std::tuple<std::string, std::string, std::string>
{
    // Find the starting position
    const auto startPos = s.find(sep);

    // Didn't find the delimiter
    if (startPos == std::string::npos) {
        return {std::string(s), "", ""};
    }

    // Split into parts
    const auto endPos = startPos + sep.size();
    auto pre = std::string(s.substr(0, startPos));
    auto mid = std::string(s.substr(startPos, endPos));
    auto post = std::string(s.substr(endPos));

    // Return the parts
    return {pre, mid, post};
}

/** @brief Convert an Integer to a padded string */
template <
    typename Integer,
    std::enable_if_t<std::is_integral<Integer>::value, bool> = true>
auto to_padded_string(Integer val, int padding, char fill = '0') -> std::string
{
    std::stringstream stream;
    stream << std::setw(padding) << std::setfill(fill) << val;
    return stream.str();
}

