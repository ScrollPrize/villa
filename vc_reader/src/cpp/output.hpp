// Optional floating output conversion. Intensity normalization lives in augment/.
#pragma once
#include "chunk_io.hpp"
#include "element_type.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace vcr {

// Rows of a strided box, with independent input/output element widths.
template <class Row>
void transform_box(const std::uint8_t* src, const Index& ss, std::uint8_t* dst,
                   const Index& ds, const Index& extent, Row&& row)
{
    const std::size_t nd = extent.size();
    Index pos(nd, 0);
    for (;;) {
        row(src, dst, static_cast<std::size_t>(extent.back()));
        std::size_t axis = nd - 1;
        for (;;) {
            if (axis-- == 0) return;
            if (++pos[axis] < extent[axis]) {
                src += ss[axis]; dst += ds[axis];
                break;
            }
            src -= (extent[axis] - 1) * ss[axis];
            dst -= (extent[axis] - 1) * ds[axis];
            pos[axis] = 0;
        }
    }
}

struct BoxOutput {
    std::size_t itemsize;
    std::vector<std::uint8_t> fill;
    std::function<void(const std::uint8_t*, const Index&, std::uint8_t*, const Index&, const Index&)> copy;

    template <class T>
    static BoxOutput make(const ElementType& source, const std::vector<std::uint8_t>& source_fill)
    {
        BoxOutput output;
        output.itemsize = sizeof(T);
        output.fill.resize(sizeof(T));
        source.visit_load([&](auto load) {
            const T value = static_cast<T>(load(source_fill.data()));
            std::memcpy(output.fill.data(), &value, sizeof(T));
            output.copy = [load, stride = source.itemsize](const std::uint8_t* s, const Index& ss,
                    std::uint8_t* d, const Index& ds, const Index& extent) {
                transform_box(s, ss, d, ds, extent, [&](const std::uint8_t* src, std::uint8_t* dst, std::size_t n) {
                    auto* out = reinterpret_cast<T*>(dst);
                    for (std::size_t i = 0; i < n; ++i)
                        out[i] = static_cast<T>(load(src + i * stride));
                });
            };
        });
        return output;
    }


};


}  // namespace vcr
