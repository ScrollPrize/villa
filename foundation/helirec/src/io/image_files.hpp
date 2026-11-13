#pragma once
// Single images as files: baseline TIFF (one uncompressed strip of float32 or uint8 grey values, as tifffile.imwrite
// writes a 2-D array) and 8-bit grey PNG (zlib). What helirec-reconstruct writes for slices and previews.
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

namespace image_files_detail {
inline void put16(std::vector<unsigned char>& b, std::uint16_t v) {
    b.push_back(static_cast<unsigned char>(v)), b.push_back(static_cast<unsigned char>(v >> 8));
}
inline void put32(std::vector<unsigned char>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<unsigned char>(v >> (8 * i)));
}
inline void put32be(std::string& b, std::uint32_t v) {
    for (int i = 3; i >= 0; --i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}
inline void write_all(const std::string& path, const void* a, std::size_t na, const void* b, std::size_t nb) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw Error("cannot write " + path);
    const bool ok = std::fwrite(a, 1, na, f) == na && (nb == 0 || std::fwrite(b, 1, nb, f) == nb);
    if (std::fclose(f) != 0 || !ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw Error("cannot write " + path);
    }
}
}  // namespace image_files_detail

// (ny, nx) values, little-endian TIFF; float32 (sample format IEEE float) or uint8
template <class T>
void write_tiff(const std::string& path, const T* data, int ny, int nx) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 1, "float32 or uint8");
    using namespace image_files_detail;
    const std::uint32_t bytes = static_cast<std::uint32_t>(sizeof(T)) * static_cast<std::uint32_t>(nx) * static_cast<std::uint32_t>(ny);
    std::vector<unsigned char> h = {'I', 'I', 42, 0};
    put32(h, 8);  // the directory follows the header; the values follow it
    const std::uint16_t n_tags = 10;
    const std::uint32_t data_at = 8 + 2 + n_tags * 12 + 4;
    put16(h, n_tags);
    const auto tag = [&](std::uint16_t id, std::uint16_t type, std::uint32_t value) {
        put16(h, id), put16(h, type), put32(h, 1);
        if (type == 3) put16(h, static_cast<std::uint16_t>(value)), put16(h, 0);
        else put32(h, value);
    };
    tag(256, 4, static_cast<std::uint32_t>(nx));            // width
    tag(257, 4, static_cast<std::uint32_t>(ny));            // length
    tag(258, 3, 8 * sizeof(T));                              // bits per sample
    tag(259, 3, 1);                                          // no compression
    tag(262, 3, 1);                                          // black is zero
    tag(273, 4, data_at);                                    // strip offset
    tag(277, 3, 1);                                          // samples per pixel
    tag(278, 4, static_cast<std::uint32_t>(ny));            // rows per strip
    tag(279, 4, bytes);                                      // strip bytes
    tag(339, 3, sizeof(T) == 4 ? 3 : 1);                     // sample format: IEEE float or unsigned
    put32(h, 0);                                             // no further directory
    write_all(path, h.data(), h.size(), data, bytes);
}

// (ny, nx) uint8 as an 8-bit grey PNG
inline void write_png(const std::string& path, const std::uint8_t* img, int ny, int nx) {
    using namespace image_files_detail;
    std::vector<unsigned char> raw;
    raw.reserve(static_cast<std::size_t>(ny) * (static_cast<std::size_t>(nx) + 1));
    for (int y = 0; y < ny; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), img + static_cast<std::size_t>(y) * nx, img + static_cast<std::size_t>(y + 1) * nx);
    }
    uLongf len = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> z(len);
    if (compress2(z.data(), &len, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) throw Error("png: zlib failed");
    std::string png = "\x89PNG\r\n\x1a\n";
    const auto chunk = [&](const char* type, const std::string& body) {
        put32be(png, static_cast<std::uint32_t>(body.size()));
        const std::string t = std::string(type, 4) + body;
        png += t;
        put32be(png, static_cast<std::uint32_t>(crc32(0, reinterpret_cast<const Bytef*>(t.data()), static_cast<uInt>(t.size()))));
    };
    std::string ihdr;
    put32be(ihdr, static_cast<std::uint32_t>(nx)), put32be(ihdr, static_cast<std::uint32_t>(ny));
    ihdr += std::string("\x08\x00\x00\x00\x00", 5);  // 8 bits, grey, deflate, adaptive filtering, no interlace
    chunk("IHDR", ihdr);
    chunk("IDAT", std::string(reinterpret_cast<const char*>(z.data()), len));
    chunk("IEND", "");
    write_all(path, png.data(), png.size(), nullptr, 0);
}

}  // namespace helirec
