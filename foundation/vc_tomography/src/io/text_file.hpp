#pragma once
// Whole files of text, and files written whole or not at all (a temporary beside the file, then renamed).
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "vc_tomography/common.hpp"

namespace vc_tomography::io {

inline bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}
inline bool is_file(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
inline bool is_dir(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw IoError("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// text, then a newline unless `newline` is false
inline void write_text(const std::string& path, const std::string& text, bool newline = true) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw IoError("cannot write " + path);
    o << text;
    if (newline) o << "\n";
    if (!o) throw IoError("cannot write " + path);
}

// write(std::ofstream&) fills a temporary beside `path` (opened binary), which then replaces `path`: a reader sees the
// old file or the whole new one, never part of it
template <class Write>
void write_atomic(const std::string& path, Write&& write) {
    const std::string tmp = path + ".tmp" + std::to_string(static_cast<long>(::getpid()));
    {
        std::ofstream o(tmp, std::ios::binary);
        if (!o) throw IoError("cannot write " + tmp);
        write(o);
        if (!o) {
            o.close();
            std::remove(tmp.c_str());
            throw IoError("cannot write " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw IoError("cannot write " + path);
    }
}

// text and a newline, written whole or not at all
inline void write_text_atomic(const std::string& path, const std::string& text) {
    write_atomic(path, [&](std::ofstream& o) { o << text << "\n"; });
}

}  // namespace vc_tomography::io
