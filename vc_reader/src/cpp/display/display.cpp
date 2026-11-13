#include "pixels.hpp"
#include <algorithm>
namespace vcr::display {
namespace {
// detail::converted over an array, in a form compilers vectorize: no branches, no libm
// calls, and the value-to-integer conversion in a loop of its own (it can trap, so a
// compiler will not vectorize it behind the selects that clamp the value).
// Real is the type the value is held in: double as in detail::converted, or float when
// the scale is 1 (double(v) * 1 is v, and clamping, truncating and v - low are exact in
// float). A clamped value is in [0, maximum], so its floor is the truncating integer
// conversion and fmod(low, 2) != 0 is that integer's low bit.
template<class T, class Real, Rounding R>
void convertAll(const float* in, std::size_t n, T* out, Real scale, bool saturateInfinity) {
    constexpr Real maximum = sizeof(T) == 1 ? 255 : 65535;
    constexpr std::size_t kBlock = 1024;
    const std::int32_t dropInfinity = !saturateInfinity;
    Real clamped[kBlock];
    for (std::size_t at = 0; at < n; at += kBlock) {
        const std::size_t m = std::min(kBlock, n - at);
        for (std::size_t i = 0; i < m; ++i) {
            const float v = in[at + i];
            Real value = Real(v) * scale;
            const std::int32_t nonfinite = !(std::fabs(v) <= 3.402823466e38f);
            const std::int32_t zero = (v != v) | (nonfinite & dropInfinity) | (value != value);
            value = std::isless(value, Real(0)) ? Real(0) : value;
            value = std::isgreater(value, maximum) ? maximum : value;
            clamped[i] = zero ? Real(0) : value;
        }
        for (std::size_t i = 0; i < m; ++i) {
            std::int32_t rounded = std::int32_t(clamped[i]);
            const Real fraction = clamped[i] - Real(rounded);
            if constexpr (R == Rounding::NearestEven) {
                const std::int32_t above = fraction > Real(.5), tie = fraction == Real(.5);
                rounded += above | (tie & rounded);
            } else if constexpr (R == Rounding::HalfUp) {
                rounded += std::int32_t(fraction >= Real(.5));
            }
            out[at + i] = T(rounded);
        }
    }
}
template<class T, class Real>
void convertAll(const float* in, std::size_t n, T* out, Real scale, Conversion conversion) {
    if (conversion.rounding == Rounding::NearestEven) convertAll<T, Real, Rounding::NearestEven>(in, n, out, scale, conversion.saturateInfinity);
    else if (conversion.rounding == Rounding::HalfUp) convertAll<T, Real, Rounding::HalfUp>(in, n, out, scale, conversion.saturateInfinity);
    else convertAll<T, Real, Rounding::Truncate>(in, n, out, scale, conversion.saturateInfinity);
}
template<class T>
void convertAll(const float* in, std::size_t n, T* out, Conversion conversion) {
    if (conversion.scale == 1) convertAll<T, float>(in, n, out, 1.0f, conversion);
    else convertAll<T, double>(in, n, out, conversion.scale, conversion);
}
}
void toUint8(const float* in,std::size_t n,uint8_t* out,uint8_t* coverage, Conversion conversion) {
    convertAll(in, n, out, conversion);
    if(coverage) for(std::size_t i=0;i<n;++i) coverage[i]=detail::finite(in[i]);
}
void toUint16(const float* in,std::size_t n,uint16_t* out,Conversion conversion) {
    convertAll(in, n, out, conversion);
}
void windowLevelLut(std::array<uint32_t,256>& out,float low,float high) noexcept {
    for(int i=0;i<256;++i) out[i]=detail::window(i,low,high);
}
void paletteLut(std::array<uint32_t,256>& out,const std::array<uint32_t,256>& palette,float low,float high,bool blackZero) noexcept {
    for(int i=0;i<256;++i) out[i]=detail::palette(i,palette.data(),low,high,blackZero);
}
void overlayLut(std::array<uint32_t,256>& out,const std::array<uint32_t,256>& color,float low,float high,bool intensity,bool premultiplied) noexcept {
    for(int i=0;i<256;++i) out[i]=detail::overlay(i,color[i],low,high,intensity,premultiplied);
}
uint32_t blendArgb(uint32_t base,uint32_t overlay,float opacity) noexcept {return detail::mix(base,overlay,opacity);}
void applyLut(const uint8_t* in,std::size_t rows,std::size_t cols,std::size_t stride,const uint32_t* lut,uint32_t* out,std::size_t outStride) {
    for(std::size_t r=0;r<rows;++r) for(std::size_t c=0;c<cols;++c) out[r*outStride+c]=lut[in[r*stride+c]];
}
void blend(const uint32_t* base,const uint32_t* overlay,std::size_t n,float opacity,uint32_t* out) {
    for(std::size_t i=0;i<n;++i) out[i]=detail::mix(base[i],overlay[i],opacity);
}
}
