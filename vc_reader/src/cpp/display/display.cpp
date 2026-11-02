#include "pixels.hpp"
namespace vcr::display {
void toUint8(const float* in,std::size_t n,uint8_t* out,uint8_t* coverage) {
    for(std::size_t i=0;i<n;++i) { out[i]=detail::byte(in[i]); if(coverage) coverage[i]=detail::finite(in[i]); }
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
