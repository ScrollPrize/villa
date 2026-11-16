#pragma once
// Paths as Python's os.path treats them, lexically (no file system access but abspath's working directory):
// normpath, join, abspath, basename ("" after a trailing /), dirname ("" without a /).
#include <filesystem>
#include <string>
#include <vector>

namespace vc_tomography::io {

inline std::string normpath(const std::string& p) {
    if (p.empty()) return ".";
    const bool absolute = p[0] == '/';
    int initial = absolute ? 1 : 0;
    if (absolute && p.size() > 1 && p[1] == '/' && (p.size() < 3 || p[2] != '/')) initial = 2;  // POSIX "//"
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i <= p.size()) {
        std::size_t j = p.find('/', i);
        if (j == std::string::npos) j = p.size();
        const std::string c = p.substr(i, j - i);
        if (c.empty() || c == ".") {
        } else if (c != "..") {
            parts.push_back(c);
        } else if ((!absolute && parts.empty()) || (!parts.empty() && parts.back() == "..")) {
            parts.push_back(c);
        } else if (!parts.empty()) {
            parts.pop_back();
        }
        i = j + 1;
    }
    std::string out(static_cast<std::size_t>(initial), '/');
    for (std::size_t k = 0; k < parts.size(); ++k) out += (k ? "/" : "") + parts[k];
    return out.empty() ? "." : out;
}

inline std::string join(const std::string& a, const std::string& b) {
    if (!b.empty() && b[0] == '/') return b;
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

inline std::string abspath(const std::string& p) {
    if (!p.empty() && p[0] == '/') return normpath(p);
    return normpath(join(std::filesystem::current_path().string(), p));
}

inline std::string basename(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}

inline std::string dirname(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    if (i == std::string::npos) return std::string();
    std::string head = p.substr(0, i + 1);
    if (head.find_first_not_of('/') != std::string::npos)
        while (!head.empty() && head.back() == '/') head.pop_back();
    return head;
}

}  // namespace vc_tomography::io
