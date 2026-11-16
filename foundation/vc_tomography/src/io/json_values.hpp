#pragma once
// Text and JSON values read as Python reads them: the parameters of a reconstruction come from Python dicts, command
// lines and JSON files, and mean what Python's float(), bool() and str.strip() make of them.
#include <cctype>
#include <cstdlib>
#include <string>

#include "io/json.hpp"

namespace vc_tomography::io {

// str.strip(): without leading and trailing whitespace
inline std::string strip(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\n\r\f\v");
    if (a == std::string::npos) return std::string();
    return s.substr(a, s.find_last_not_of(" \t\n\r\f\v") - a + 1);
}

inline std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// float(text): the whole text (whitespace around it aside) a real number, also inf and nan; not hexadecimal
inline bool real_of(const std::string& t, double& v) {
    const std::string s = strip(t);
    if (s.empty() || s.find_first_of("xX") != std::string::npos) return false;
    char* end = nullptr;
    v = std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}

// float(value): numbers, booleans, and text that reads as one
inline bool number_of(const Json& v, double& out) {
    if (v.is(Json::Number)) return out = v.number, true;
    if (v.is(Json::Bool)) return out = v.boolean ? 1.0 : 0.0, true;
    if (v.is(Json::String)) return real_of(v.string, out);
    return false;
}

// bool(value)
inline bool truth(const Json& v) {
    switch (v.type) {
        case Json::Bool: return v.boolean;
        case Json::Number: return v.number != 0;
        case Json::String: return !v.string.empty();
        case Json::Array: return !v.array.empty();
        case Json::Object: return !v.object.empty();
        default: return false;
    }
}

}  // namespace vc_tomography::io
