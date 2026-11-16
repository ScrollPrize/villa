#pragma once
// Slices and previews as image files (vc_reader's writers), and a slice windowed to uint8 on the CPU.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vc_reader/image_write.hpp"
#include "vc_tomography/kernels.hpp"

namespace vc_tomography::io {

// a single-page TIFF of one plane (dtype "|u1" or "<f4"), deflate-compressed or not
inline void write_tiff(const std::string& path, const void* data, const char* dtype, int ny, int nx, bool deflate = false) {
    vcr::sources::TiffOptions o;
    o.compression = deflate ? "deflate" : "none";
    vcr::sources::writeTiff(path, data, dtype, ny, nx, 1, o);
}
inline void write_png(const std::string& path, const std::uint8_t* data, int ny, int nx) {
    vcr::sources::writeImage(path, vcr::sources::ImageView{data, "|u1", ny, nx});
}

// rint(clip((img - lo) / (hi - lo), 0, 1) 255) (cpu::to_uint8; NaN gives 0)
inline std::vector<std::uint8_t> to_u8(const float* img, int ny, int nx, const std::array<double, 2>& window) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(ny) * nx);
    Uint8Window w;
    w.lo = window[0], w.hi = window[1];
    cpu::to_uint8(img, static_cast<std::size_t>(nx), out.data(), ny, nx, w);
    return out;
}

}  // namespace vc_tomography::io
