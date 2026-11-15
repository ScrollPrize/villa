#pragma once
// ISO 8601 time stamps of the acquisition records as milliseconds since the epoch.
#include <cctype>
#include <cstdint>
#include <string>

#include "helirec/common.hpp"

namespace helirec::calibration {

inline std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

// datetime.fromisoformat(s), in UTC when it carries an offset, as numpy datetime64[ms]: milliseconds since the
// epoch (the microseconds rounded down)
inline std::int64_t iso_ms(const std::string& s) {
    const auto bad = [&]() { return Error("invalid isoformat string: '" + s + "'"); };
    std::size_t i = 0;
    const auto digits = [&](int n) {
        if (i + n > s.size()) throw bad();
        int v = 0;
        for (int k = 0; k < n; ++k) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) throw bad();
            v = v * 10 + (s[i++] - '0');
        }
        return v;
    };
    const auto expect = [&](char c) {
        if (i >= s.size() || s[i] != c) throw bad();
        ++i;
    };
    const int year = digits(4);
    expect('-');
    const int month = digits(2);
    expect('-');
    const int day = digits(2);
    int hour = 0, minute = 0, second = 0;
    std::int64_t micro = 0, offset_us = 0;
    if (i < s.size()) {
        ++i;  // the separator (any one character)
        hour = digits(2);
        if (i < s.size() && s[i] == ':') {
            ++i;
            minute = digits(2);
            if (i < s.size() && s[i] == ':') {
                ++i;
                second = digits(2);
                if (i < s.size() && (s[i] == '.' || s[i] == ',')) {
                    ++i;
                    int n = 0;
                    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                        if (n < 6) micro = micro * 10 + (s[i] - '0');
                        ++n, ++i;
                    }
                    if (n == 0) throw bad();
                    for (; n < 6; ++n) micro *= 10;
                }
            }
        }
        if (i < s.size()) {
            const char c = s[i++];
            if (c == 'Z' && i == s.size()) {
            } else if (c == '+' || c == '-') {
                const int oh = digits(2);
                int om = 0, os = 0;
                std::int64_t ou = 0;
                if (i < s.size() && s[i] == ':') ++i;
                if (i < s.size()) om = digits(2);
                if (i < s.size() && s[i] == ':') ++i;
                if (i < s.size()) os = digits(2);
                if (i < s.size() && s[i] == '.') {
                    ++i;
                    ou = digits(6);
                }
                if (i != s.size()) throw bad();
                offset_us = (((oh * 60LL + om) * 60 + os) * 1000000 + ou) * (c == '-' ? -1 : 1);
            } else {
                throw bad();
            }
        }
    }
    if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59) throw bad();
    const std::int64_t us = ((days_from_civil(year, month, day) * 24 + hour) * 60 + minute) * 60000000LL +
                            second * 1000000LL + micro - offset_us;
    return us >= 0 ? us / 1000 : -((-us + 999) / 1000);
}

}  // namespace helirec::calibration
