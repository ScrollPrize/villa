#pragma once
#include "pixels.hpp"
#include "vc_reader/slab.hpp"
#ifdef __CUDACC__
#define VCR_SLAB __host__ __device__
#else
#define VCR_SLAB
#endif
namespace vcr::display::detail {
VCR_SLAB inline float opacityValue(int v,float low,float high,float opacity,float gamma,uint8_t cutoff,float segmentLength) {
    if(v<cutoff)return 0;
    const float lo=clamp(low,0,1)*255,hi=clamp(high,0,1)*255;
    const float rho=clamp((float(v)-lo)/fmaxf(hi-lo,1e-3f),0,1);
    return fminf(fmaxf(opacity,0)*fmaxf(segmentLength,1)*powf(rho,fmaxf(gamma,1e-3f)),1);
}
VCR_SLAB inline void slabPixel(const Slab& slab,int x,int y,uint8_t* rgb,uint8_t* coverage) {
    rgb[0]=rgb[1]=rgb[2]=0;*coverage=0;
    const auto& proj=slab.projection;
    const bool perspective=proj.perspective;
    const int numLayers=slab.count,rows=slab.rows,cols=slab.cols;
    const float px=float(x+slab.left),py=float(y+slab.top),lightingStrength=slab.lighting;
    const float* opacityLut=slab.opacity;
    constexpr float kEarlyOutT=0.004f;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            float T = 1.0f;
            bool anyValid = false;
            // Orthographic: the linear part is shared by all layers; only the
            // offset varies. Perspective: the pixel's ray direction and the
            // 1/r_w division are per-pixel; per layer it's one multiply-add.
            float baseU = 0.0f, baseV = 0.0f;
            float rayU = 0.0f, rayV = 0.0f, invRayW = 0.0f;
            if (perspective) {
                const float s1 = px - proj.centerU;
                const float s2 = py - proj.centerV;
                rayU = proj.rayBase[0] + s1 * proj.e1OverF[0] + s2 * proj.e2OverF[0];
                rayV = proj.rayBase[1] + s1 * proj.e1OverF[1] + s2 * proj.e2OverF[1];
                const float rayW =
                    proj.rayBase[2] + s1 * proj.e1OverF[2] + s2 * proj.e2OverF[2];
                if (rayW >= -1e-4f)
                    return;  // ray points away from the slab
                invRayW = 1.0f / rayW;
            } else {
                baseU = proj.m00 * px + proj.m01 * py;
                baseV = proj.m10 * px + proj.m11 * py;
            }
            // Near-to-far: the camera sits on the +w side, so highest w first.
            for (int i = numLayers - 1; i >= 0; --i) {
                float qu, qv;
                if (perspective) {
                    const float t = proj.layerNum[i] * invRayW;
                    qu = proj.pos[0] + t * rayU;
                    qv = proj.pos[1] + t * rayV;
                } else {
                    qu = baseU + proj.layerOffsets[2*i];
                    qv = baseV + proj.layerOffsets[2*i+1];
                }
                if (!finite(qu) || !finite(qv) || qu < 0 || qv < 0 || qu >= float(cols-1) || qv >= float(rows-1)) continue;
                const int x0 = int(floorf(qu));
                const int y0 = int(floorf(qv));
                if (x0 < 0 || y0 < 0 || x0 + 1 >= cols || y0 + 1 >= rows)
                    continue;
                const float fx = qu - float(x0);
                const float fy = qv - float(y0);
                const auto& layer = slab.layers[i];
                const uint8_t* v0 = layer.values + y0*layer.valueStride + x0;
                const uint8_t* v1 = layer.values + (y0+1)*layer.valueStride + x0;
                const uint8_t* c0 = layer.coverage + y0*layer.coverageStride + x0;
                const uint8_t* c1 = layer.coverage + (y0+1)*layer.coverageStride + x0;
                // Coverage-weighted bilinear: uncovered texels are fully
                // transparent, not value 0 (avoids dark halos at patch
                // borders under tilt).
                const float w00 = c0[0] ? (1.0f - fx) * (1.0f - fy) : 0.0f;
                const float w10 = c0[1] ? fx * (1.0f - fy) : 0.0f;
                const float w01 = c1[0] ? (1.0f - fx) * fy : 0.0f;
                const float w11 = c1[1] ? fx * fy : 0.0f;
                const float wsum = w00 + w10 + w01 + w11;
                if (wsum <= 1e-6f)
                    continue;
                const float value = (w00 * float(v0[0]) + w10 * float(v0[1]) +
                                     w01 * float(v1[0]) + w11 * float(v1[1])) / wsum;
                anyValid = true;
                const int idx = int(clamp(float(int(value + 0.5f)),0,255));
                const float alpha = opacityLut[idx];
                if (alpha <= 0.0f)
                    continue;
                const uint32_t color = slab.colors[idx];
                const float e[3]{alpha * float((color >> 16) & 255),alpha * float((color >> 8) & 255),alpha * float(color & 255)};
                float shade = 1.0f;
                const float lightAmount = clamp(lightingStrength,0,1);
                if (lightAmount > 0.0f) {
                    // Central differences in the already-extracted voxel
                    // stack. Missing neighbours fall back to the center, so
                    // coverage boundaries do not create artificial normals.
                    auto sample = [&](int layer, int sx, int sy) {
                        if (layer < 0 || layer >= numLayers)
                            return value;
                        const int xx = int(clamp(float(x0 + sx),0,float(cols-1)));
                        const int yy = int(clamp(float(y0 + sy),0,float(rows-1)));
                        if (!slab.layers[layer].coverage[yy*slab.layers[layer].coverageStride+xx])
                            return value;
                        return float(slab.layers[layer].values[yy*slab.layers[layer].valueStride+xx]);
                    };
                    const float gx = sample(i, 1, 0) - sample(i, -1, 0);
                    const float gy = sample(i, 0, 1) - sample(i, 0, -1);
                    const float gz = sample(i + 1, 0, 0) - sample(i - 1, 0, 0);
                    const float len2 = gx * gx + gy * gy + gz * gz;
                    if (len2 > 1e-6f) {
                        const float invLen = 1.0f / sqrtf(len2);
                        // Fixed upper-left raking light, 35 degrees above the
                        // slab. At maximum strength, 10% ambient preserves
                        // some detail on back-facing sides.
                        constexpr float lx = -0.579228f;
                        constexpr float ly = -0.579228f;
                        constexpr float lz =  0.573576f;
                        const float diffuse = fmaxf(
                            0.0f, (gx * lx + gy * ly + gz * lz) * invLen);
                        const float fullyLit = 0.10f + 0.90f * diffuse;
                        shade = 1.0f + lightAmount * (fullyLit - 1.0f);
                    }
                }
                r += T * e[0] * shade;
                g += T * e[1] * shade;
                b += T * e[2] * shade;
                T *= 1.0f - alpha;
                if (T < kEarlyOutT)
                    break;
            }
            if (anyValid) {
                rgb[0] = uint8_t(fminf(r,255.f));
                rgb[1] = uint8_t(fminf(g,255.f));
                rgb[2] = uint8_t(fminf(b,255.f));
                *coverage = 1;
            }
}
}
#undef VCR_SLAB
