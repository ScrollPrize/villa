#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

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

}  // namespace vcr
