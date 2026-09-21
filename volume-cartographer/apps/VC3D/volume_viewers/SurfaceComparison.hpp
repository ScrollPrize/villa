#pragma once
#include <QColor>
#include <QImage>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

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
inline void blendSurfaceImageInPlace(QImage& result,const QImage& overlay,
                                    const cv::Mat_<uint8_t>& coverage,float opacity,const QColor& tint = {}) {
    if(result.size()!=overlay.size()||coverage.cols!=result.width()||coverage.rows!=result.height())return;
    result=result.convertToFormat(QImage::Format_RGB32);
    if(opacity<=0)return;
    const auto source=overlay.convertToFormat(QImage::Format_RGB32);
    const float tintOpacity=tint.isValid()?tint.alphaF():0.f;
    const auto tintRgb=tintOpacity>0?tint.rgb():0;
    for(int y=0;y<result.height();++y) {
        auto* dst=reinterpret_cast<uint32_t*>(result.scanLine(y));
        const auto* src=reinterpret_cast<const uint32_t*>(source.constScanLine(y));
        for(int x=0;x<result.width();++x)if(coverage(y,x)) {
            const auto pixel=tintOpacity>0?alphaBlendArgb(src[x],tintRgb,tintOpacity):src[x];
            dst[x]=alphaBlendArgb(dst[x],pixel,opacity);
        }
    }
}
inline QImage blendSurfaceImages(const QImage& base,const QImage& overlay,
                                 const cv::Mat_<uint8_t>& coverage,float opacity,const QColor& tint = {}) {
    auto result=base;
    blendSurfaceImageInPlace(result,overlay,coverage,opacity,tint);
    return result;
}
struct SurfaceImageLayer {
    const QImage& image;
    const cv::Mat_<uint8_t>& coverage;
    float opacity;
    QColor tint;
};
inline QImage blendSurfaceLayers(const QImage& base,std::span<const SurfaceImageLayer> layers) {
    auto result=base;
    for(const auto& layer:layers)blendSurfaceImageInPlace(result,layer.image,layer.coverage,layer.opacity,layer.tint);
    return result;
}
} // namespace vc3d
