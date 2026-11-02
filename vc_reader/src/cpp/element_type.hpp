#pragma once
#include <algorithm>
#include <bit>
#include <string_view>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace vcr {
// How sample() reads stored elements as numbers.
struct ElementType {
    char kind = 'u';  // numpy kind: 'u', 'i', 'f' or 'b'
    std::size_t itemsize = 1;
    bool swap = false;  // stored in the other byte order

    double load(const std::uint8_t* p) const
    {
        std::uint8_t b[8];
        std::memcpy(b, p, itemsize);
        if (swap)
            std::reverse(b, b + itemsize);
        switch (kind) {
        case 'b':
            return b[0] != 0;
        case 'f':
            if (itemsize == 2) {
                std::uint16_t bits;
                std::memcpy(&bits, b, 2);
                const int exponent = (bits >> 10) & 0x1f, fraction = bits & 0x3ff;
                const double magnitude = exponent == 0x1f ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                                                       : std::numeric_limits<double>::infinity())
                                         : exponent == 0   ? std::ldexp(double(fraction), -24)
                                                           : std::ldexp(double(fraction + 1024), exponent - 25);
                return bits & 0x8000 ? -magnitude : magnitude;
            }
            if (itemsize == 4) {
                float v;
                std::memcpy(&v, b, 4);
                return v;
            } else {
                double v;
                std::memcpy(&v, b, 8);
                return v;
            }
        case 'i':
            switch (itemsize) {
            case 1: return static_cast<double>(static_cast<std::int8_t>(b[0]));
            case 2: { std::int16_t v; std::memcpy(&v, b, 2); return v; }
            case 4: { std::int32_t v; std::memcpy(&v, b, 4); return v; }
            default: { std::int64_t v; std::memcpy(&v, b, 8); return static_cast<double>(v); }
            }
        default:
            switch (itemsize) {
            case 1: return b[0];
            case 2: { std::uint16_t v; std::memcpy(&v, b, 2); return v; }
            case 4: { std::uint32_t v; std::memcpy(&v, b, 4); return v; }
            default: { std::uint64_t v; std::memcpy(&v, b, 8); return static_cast<double>(v); }
            }
        }
    }

    template <class F>
    void visit_load(F&& dispatch) const
    {
        const auto with = [&]<class T>() {
            dispatch([swap = swap](const std::uint8_t* p) {
                T v;
                if (swap && sizeof(T) > 1) {
                    std::uint8_t bytes[sizeof(T)];
                    std::reverse_copy(p, p + sizeof(T), bytes);
                    std::memcpy(&v, bytes, sizeof(T));
                } else {
                    std::memcpy(&v, p, sizeof(T));
                }
                return v;
            });
        };
        if (kind == 'f' && itemsize == 2)
            return dispatch([this](const std::uint8_t* p) { return load(p); });
        if (kind == 'b')
            return dispatch([](const std::uint8_t* p) { return *p != 0; });
        switch (kind == 'f' ? 100 + itemsize : kind == 'i' ? 10 + itemsize : itemsize) {
        case 1: return with.template operator()<std::uint8_t>();
        case 2: return with.template operator()<std::uint16_t>();
        case 4: return with.template operator()<std::uint32_t>();
        case 8: return with.template operator()<std::uint64_t>();
        case 11: return with.template operator()<std::int8_t>();
        case 12: return with.template operator()<std::int16_t>();
        case 14: return with.template operator()<std::int32_t>();
        case 18: return with.template operator()<std::int64_t>();
        case 104: return with.template operator()<float>();
        case 108: return with.template operator()<double>();
        default: throw std::invalid_argument("unsupported numeric source dtype");
        }
    }
};


// Validated storage dtype. Native aliases are resolved once at the boundary.
struct Dtype {
    char order = '|', kind = 'u';
    std::size_t itemsize = 1;
    ElementType element() const {
        return {kind, itemsize, itemsize > 1 && ((order == '>') == (std::endian::native == std::endian::little))};
    }
    std::string str() const { return std::string(1, itemsize == 1 ? '|' : order) + kind + std::to_string(itemsize); }
};
inline Dtype parseDtype(std::string_view text)
{
    const auto invalid = [&]() -> Dtype { throw std::invalid_argument("unsupported dtype '" + std::string(text) + "'"); };
    Dtype t;
    std::size_t i = 0;
    if (!text.empty() && std::string_view("<>|=").find(text[0]) != std::string_view::npos) t.order = text[i++];
    if (i >= text.size() || std::string_view("buif").find(text[i]) == std::string_view::npos) return invalid();
    t.kind = text[i++];
    if (i + 1 != text.size() || std::string_view("1248").find(text[i]) == std::string_view::npos) return invalid();
    t.itemsize = std::size_t(text[i] - '0');
    if ((t.kind == 'b' && t.itemsize != 1) || (t.kind == 'f' && t.itemsize == 1)) return invalid();
    if (t.order == '=' || (t.order == '|' && t.itemsize > 1)) t.order = std::endian::native == std::endian::little ? '<' : '>';
    return t;
}

inline ElementType elementOf(const std::string& typestr) { return parseDtype(typestr).element(); }

}  // namespace vcr
