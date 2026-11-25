#pragma once
#include "element_type.hpp"
#include "rounding.hpp"
#include <vector>
#include <nlohmann/json.hpp>

namespace vcr {
// Zarr allows null, special floating-point strings and narrowing integer fills.
// Source declarations require a representable numeric value.
enum class ValueEncoding { Exact, ZarrFill };

inline std::uint16_t float16Bits(double v)
{
    const std::uint16_t sign = std::signbit(v) ? 0x8000 : 0;
    if (std::isnan(v)) return sign | 0x7e00;
    const double magnitude = std::fabs(v);
    if (magnitude >= 65520.0) return sign | 0x7c00;
    if (magnitude < std::ldexp(1.0, -14))
        return sign | std::uint16_t(round_half_even(std::ldexp(magnitude, 24)));
    int exponent = 0;
    const double mantissa = std::frexp(magnitude, &exponent);
    return sign | std::uint16_t(((exponent + 14) << 10) + std::uint16_t(round_half_even((mantissa * 2.0 - 1.0) * 1024.0)));
}

inline std::vector<std::uint8_t> encodeElement(const nlohmann::json& value, ElementType type, ValueEncoding policy)
{
    const bool exact = policy == ValueEncoding::Exact;
    const auto bad = [&]() { throw std::invalid_argument("fill value " + value.dump() + " is not representable in " +
                                                       std::string(1, type.kind) + std::to_string(type.itemsize)); };
    std::vector<std::uint8_t> out(type.itemsize, 0);
    if (!exact && value.is_null()) return out;
    const auto put = [&](auto v) { std::memcpy(out.data(), &v, sizeof(v)); };
    const auto bits = [&](std::uint64_t v) {
        switch (type.itemsize) {
        case 1: put(std::uint8_t(v)); break;
        case 2: put(std::uint16_t(v)); break;
        case 4: put(std::uint32_t(v)); break;
        case 8: put(v); break;
        }
    };
    if (type.kind == 'f') {
        double v = 0;
        bool raw = false;
        if (!exact && value.is_string()) {
            const auto s = value.get<std::string>();
            if (s == "NaN") v = std::numeric_limits<double>::quiet_NaN();
            else if (s == "Infinity") v = std::numeric_limits<double>::infinity();
            else if (s == "-Infinity") v = -std::numeric_limits<double>::infinity();
            else if (s.starts_with("0x")) { bits(std::stoull(s.substr(2), nullptr, 16)); raw = true; }
            else bad();
        } else if (value.is_number()) v = value.get<double>();
        else if (!exact && value.is_boolean()) v = double(value.get<bool>());
        else bad();
        if (!raw) {
            if (type.itemsize == 2) {
                const auto half = float16Bits(v);
                if (exact && std::isfinite(v) && (half & 0x7c00) == 0x7c00) bad();
                put(half);
            } else if (type.itemsize == 4) {
                const float f = float(v);
                if (exact && std::isfinite(v) && !std::isfinite(f)) bad();
                put(f);
            } else put(v);
        }
    } else {
        if (!value.is_number() && !value.is_boolean()) bad();
        std::uint64_t magnitude = 0;
        bool negative = false;
        if (value.is_number_float()) {
            const double v = value.get<double>();
            if (!std::isfinite(v) || v != std::floor(v) || std::abs(v) >= 0x1p64) bad();
            negative = v < 0;
            magnitude = std::uint64_t(std::abs(v));
        } else if (value.is_number_unsigned()) magnitude = value.get<std::uint64_t>();
        else if (value.is_boolean()) magnitude = value.get<bool>();
        else {
            const auto v = value.get<std::int64_t>();
            negative = v < 0;
            magnitude = negative ? ~std::uint64_t(v) + 1 : std::uint64_t(v);
        }
        if (exact) {
            const bool isSigned = type.kind == 'i';
            const int width = int(type.itemsize) * 8;
            const std::uint64_t maximum = type.kind == 'b' ? 1 : isSigned ? (std::uint64_t(1) << (width - 1)) - 1
                : width == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << width) - 1;
            if (negative && (!isSigned || magnitude > maximum + 1)) bad();
            if (!negative && magnitude > maximum) bad();
        }
        bits(negative ? ~magnitude + 1 : magnitude);
    }
    if (type.swap) std::reverse(out.begin(), out.end());
    return out;
}

// Preserve integer values exactly; a uint64 label never passes through double.
inline nlohmann::json decodeElement(const std::uint8_t* data, const ElementType& type)
{
    nlohmann::json out;
    type.visit_load([&](auto load) {
        if (type.kind == 'b') out = int(load(data) != 0);
        else out = load(data);
    });
    return out;
}
} // namespace vcr
