#pragma once
#include "vc_reader/display.hpp"
#include <cmath>
#ifdef __CUDACC__
#define VCR_PIXEL __host__ __device__
#else
#define VCR_PIXEL
#endif
namespace vcr::display::detail {
VCR_PIXEL inline float clamp(float x,float a,float b) { return x<a?a:x>b?b:x; }
VCR_PIXEL inline bool finite(float x) { return x<=3.402823466e38f && x>=-3.402823466e38f; }
VCR_PIXEL inline uint8_t byte(float v) { return finite(v) ? uint8_t(clamp(v,0,255)) : 0; }
VCR_PIXEL inline uint32_t gray(int i,float low,float high) {
    const int lo=int(clamp(finite(low)?low:0,0,255));
    // A top endpoint of 255 still yields a well-defined one-unit span.
    const int hi=int(clamp(finite(high)?high:255,float(lo<255?lo+1:255),255));
    const float span=hi>lo?float(hi-lo):1.f;
    return byte((float(i)-lo)/span*255.f);
}
VCR_PIXEL inline uint32_t window(int i,float low,float high) {
    const auto v=gray(i,low,high);return 0xff000000u|(v<<16)|(v<<8)|v;
}
VCR_PIXEL inline uint32_t palette(int i,const uint32_t* colors,float low,float high,bool blackZero) {
    return i==0 && blackZero ? 0xff000000u : colors[gray(i,low,high)];
}
VCR_PIXEL inline uint32_t overlay(int i,uint32_t color,float low,float high,bool intensity,bool premultiplied) {
    if (!intensity) return float(i)>=low && float(i)<=high ? color : 0;
    const auto t=gray(i,low,high);
    if (!t) return 0;
    const auto r=(color>>16)&255,g=(color>>8)&255,b=color&255;
    if (premultiplied) {
        const auto a=r>g?(r>b?r:b):(g>b?g:b);
        return a ? (a<<24)|(color&0xffffffu):0;
    }
    return (t<<24)|(((r*t+127)/255)<<16)|(((g*t+127)/255)<<8)|((b*t+127)/255);
}
VCR_PIXEL inline uint32_t mix(uint32_t base,uint32_t over,float opacity) {
    const auto alpha=over>>24;
    if (!alpha) return base;
    const float a=clamp(finite(opacity)?opacity:0,0,1),keep=1-a*(float(alpha)/255.f);
    uint32_t out=0xff000000u;
    for(int shift=0;shift<24;shift+=8) {
        const float v=keep*float((base>>shift)&255)+a*float((over>>shift)&255);
        out|=uint32_t(clamp(floorf(v+0.5f),0,255))<<shift;
    }
    return out;
}
}
#undef VCR_PIXEL
