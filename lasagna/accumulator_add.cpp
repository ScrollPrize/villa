#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LASAGNA_X86_TARGETS 1
#else
#define LASAGNA_X86_TARGETS 0
#endif

namespace py = pybind11;

namespace {

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 | mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
        if (mantissa != 0) bits |= 0x00400000u;
    } else {
        bits = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t float_to_half_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        return static_cast<uint16_t>(sign | 0x7e00u | (mantissa >> 13));
    }
    const int new_exp = static_cast<int>(exponent) - 127 + 15;
    if (new_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (new_exp <= 0) {
        if (new_exp < -10) return sign;
        mantissa |= 0x800000u;
        const int shift = 14 - new_exp;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x400u) {
            rounded = 0;
            if (new_exp + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | ((new_exp + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (new_exp << 10) | rounded);
}

void add_half_scalar(uint16_t* dst, const float* src, ssize_t count) {
    for (ssize_t i = 0; i < count; ++i) {
        dst[i] = float_to_half_rne(half_to_float(dst[i]) + src[i]);
    }
}

void add_float_scalar(float* dst, const float* src, ssize_t count) {
    for (ssize_t i = 0; i < count; ++i) dst[i] += src[i];
}

#if LASAGNA_X86_TARGETS
__attribute__((target("avx512f,f16c")))
void add_half_avx512(uint16_t* dst, const float* src, ssize_t count) {
    ssize_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i half = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m512 current = _mm512_cvtph_ps(half);
        const __m512 incoming = _mm512_loadu_ps(src + i);
        const __m512 sum = _mm512_add_ps(current, incoming);
        const __m256i packed = _mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), packed);
    }
    add_half_scalar(dst + i, src + i, count - i);
}

__attribute__((target("avx512f")))
void add_float_avx512(float* dst, const float* src, ssize_t count) {
    ssize_t i = 0;
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_add_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(src + i)));
    }
    add_float_scalar(dst + i, src + i, count - i);
}

bool avx512_available() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("f16c");
}
#else
bool avx512_available() { return false; }
#endif

std::string selected_backend(const std::string& requested) {
    if (requested != "auto" && requested != "scalar" && requested != "avx512") {
        throw std::invalid_argument("backend must be auto, scalar, or avx512");
    }
    if (requested == "scalar") return "scalar";
    if (avx512_available()) return "avx512";
    if (requested == "avx512") throw std::runtime_error("AVX-512F+F16C is unavailable on this CPU/build");
    return "scalar";
}

void add_inplace(py::array dst, py::array src, const std::string& requested) {
    if (!dst.writeable()) throw std::invalid_argument("destination must be writable");
    const py::buffer_info d = dst.request();
    const py::buffer_info s = src.request();
    if (d.ndim != 3 || s.ndim != 3) throw std::invalid_argument("destination and source must be 3D");
    for (int axis = 0; axis < 3; ++axis) {
        if (d.shape[axis] != s.shape[axis]) throw std::invalid_argument("destination/source shape mismatch");
    }
    // PEP 3118 uses ``e`` for IEEE binary16.  Avoid pybind11's optional
    // half type so this also builds with older distro pybind11 releases.
    const bool dst_half = d.itemsize == 2 && d.format == "e";
    const bool dst_float = d.itemsize == 4 && d.format == py::format_descriptor<float>::format();
    if (!dst_half && !dst_float) throw std::invalid_argument("destination dtype must be float16 or float32");
    if (s.itemsize != 4 || s.format != py::format_descriptor<float>::format()) {
        throw std::invalid_argument("source dtype must be float32");
    }
    if (d.strides[2] != d.itemsize || s.strides[2] != s.itemsize) {
        throw std::invalid_argument("X rows must be contiguous");
    }
	for (int axis = 0; axis < 3; ++axis) {
		if (d.strides[axis] < 0 || s.strides[axis] < 0) {
			throw std::invalid_argument("negative strides are unsupported");
		}
	}
	const std::string backend = selected_backend(requested);
	if (d.shape[0] == 0 || d.shape[1] == 0 || d.shape[2] == 0) return;
    const auto d_begin = reinterpret_cast<uintptr_t>(d.ptr);
    const auto s_begin = reinterpret_cast<uintptr_t>(s.ptr);
    const size_t d_span = static_cast<size_t>((d.shape[0]-1)*d.strides[0] + (d.shape[1]-1)*d.strides[1] + d.shape[2]*d.itemsize);
    const size_t s_span = static_cast<size_t>((s.shape[0]-1)*s.strides[0] + (s.shape[1]-1)*s.strides[1] + s.shape[2]*s.itemsize);
    if (d_begin < s_begin + s_span && s_begin < d_begin + d_span) {
        throw std::invalid_argument("destination and source must not overlap");
    }
    py::gil_scoped_release release;
    for (ssize_t z = 0; z < d.shape[0]; ++z) {
        for (ssize_t y = 0; y < d.shape[1]; ++y) {
            char* dp = static_cast<char*>(d.ptr) + z*d.strides[0] + y*d.strides[1];
            const char* sp = static_cast<const char*>(s.ptr) + z*s.strides[0] + y*s.strides[1];
            if (dst_half) {
#if LASAGNA_X86_TARGETS
                if (backend == "avx512") add_half_avx512(reinterpret_cast<uint16_t*>(dp), reinterpret_cast<const float*>(sp), d.shape[2]);
                else
#endif
                add_half_scalar(reinterpret_cast<uint16_t*>(dp), reinterpret_cast<const float*>(sp), d.shape[2]);
            } else {
#if LASAGNA_X86_TARGETS
                if (backend == "avx512") add_float_avx512(reinterpret_cast<float*>(dp), reinterpret_cast<const float*>(sp), d.shape[2]);
                else
#endif
                add_float_scalar(reinterpret_cast<float*>(dp), reinterpret_cast<const float*>(sp), d.shape[2]);
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(accumulator_add, module) {
    module.doc() = "Portable and AVX-512 accelerated strided accumulator updates";
    module.def("add_inplace", &add_inplace, py::arg("destination"), py::arg("source"), py::arg("backend") = "auto");
    module.def("backend", []() { return selected_backend("auto"); });
    module.def("has_avx512", []() { return avx512_available(); });
}
