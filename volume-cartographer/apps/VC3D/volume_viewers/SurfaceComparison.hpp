#pragma once
#include <QImage>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vc3d {
inline uint32_t alphaBlendArgb(uint32_t base, uint32_t overlay, float alpha)
{
    const float a = std::clamp(alpha, 0.0f, 1.0f);
    const auto mix = [a](uint32_t b, uint32_t o) -> uint32_t {
        return static_cast<uint32_t>(
            std::clamp(std::lround((1.0f - a) * float(b) + a * float(o)), 0L, 255L));
    };
    const uint32_t br = (base >> 16) & 0xFFu;
    const uint32_t bg = (base >> 8) & 0xFFu;
    const uint32_t bb = base & 0xFFu;
    const uint32_t or_ = (overlay >> 16) & 0xFFu;
    const uint32_t og = (overlay >> 8) & 0xFFu;
    const uint32_t ob = overlay & 0xFFu;
    return 0xFF000000u | (mix(br, or_) << 16) | (mix(bg, og) << 8) | mix(bb, ob);
}

// Only covered comparison pixels contribute. Holes keep the base image;
// opacity zero restores it exactly, including genuinely black CT pixels.
inline QImage blendSurfaceImages(const QImage& base,const QImage& overlay,
                                 const cv::Mat_<uint8_t>& coverage,float opacity) {
    if(base.size()!=overlay.size()||coverage.cols!=base.width()||coverage.rows!=base.height())return base;
    auto result=base.convertToFormat(QImage::Format_RGB32);
    const auto source=overlay.convertToFormat(QImage::Format_RGB32);
    for(int y=0;y<result.height();++y) {
        auto* dst=reinterpret_cast<uint32_t*>(result.scanLine(y));
        const auto* src=reinterpret_cast<const uint32_t*>(source.constScanLine(y));
        for(int x=0;x<result.width();++x)if(coverage(y,x))dst[x]=alphaBlendArgb(dst[x],src[x],opacity);
    }
    return result;
}
} // namespace vc3d
