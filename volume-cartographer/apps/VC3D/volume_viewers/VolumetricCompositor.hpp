#pragma once

// True-orthographic volumetric compositing for the flattened-segment view.
//
// Works in slab space (u, v, w): u,v are output-raster pixels, w is the layer
// offset along the surface normal (in layer units). The slab is rendered as a
// rigid stack rotated by yaw/pitch and projected orthographically onto the
// screen (parallel rays along the view direction, orthonormal screen basis),
// rotating about the view center on the w = 0 plane. At yaw = pitch = 0 this
// reproduces the pixel-aligned layer stack exactly. Rays are straight in slab
// space, i.e. they bend with the surface in world space — the render "follows
// the page".
//
// Qt-free; unit-testable.

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc3d::volumetric {

// Turntable camera: azimuth spins the patch about the surface normal (a pure
// in-plane rotation of the image, even at zero tilt); tilt then tips the view
// about the screen-horizontal axis, from straight down (0) toward flat — on
// screen, tilt is always a vertical foreshortening, regardless of azimuth.
struct CameraParams {
    float azimuthDeg = 0.0f; // patch spin about the normal (degrees)
    float tiltDeg = 0.0f;    // angle away from the +W axis (0 = straight down)
    // 0 = orthographic; (0..1] = true pinhole perspective, mapping to a
    // half-field-of-view of perspective * 45 degrees. The focal length equals
    // the camera distance, so magnification is exactly 1 on the plane through
    // the view center perpendicular to the view axis — switching between
    // orthographic and perspective preserves the central ray and the
    // coverage at that anchor depth.
    float perspective = 0.0f;
};

// Screen -> slab mapping. Per layer: q = M * p + layerOffsets[i], with the
// 2x2 matrix M shared by all layers (an orthographic view of a rotated stack
// is one affine per layer with a common linear part). M encodes the
// cos(tilt) foreshortening; the per-layer offset encodes the w * tan(tilt)
// shift plus the rotation-center terms.
struct SlabProjection {
    float m00 = 1.0f, m01 = 0.0f;
    float m10 = 0.0f, m11 = 1.0f;
    std::vector<std::array<float, 2>> layerOffsets;
};

// Unit view direction (du, dv, dw) for the given tilt, dw > 0.
// Tilt is clamped to [0, 85] degrees.
std::array<float, 3> viewDirection(const CameraParams& cam);

// Screen->slab projection for the rotated stack (orthographic).
//  numLayers  layer i sits at w = zStart + i (layer units)
//  outputScale  output pixels per voxel (layer offsets are in voxels)
//  centerU/V  view center in output pixels (the rotation stays centered there)
SlabProjection slabProjection(const CameraParams& cam,
                              int numLayers,
                              int zStart,
                              float outputScale,
                              float centerU,
                              float centerV);

// Pinhole camera for the rotated stack (cam.perspective > 0). A screen pixel
// p (raw coords) maps onto layer i by intersecting the ray
//   X(t) = pos + t * r,   r = rayBase + s1*e1OverF + s2*e2OverF,
//   s = p - (centerU, centerV)
// with the layer plane: t = layerNum[i] / r_w. The division by r_w is
// per-pixel, not per-layer. Rays with r_w >= 0 point away from the slab and
// hit nothing.
struct PerspectiveCamera {
    std::array<float, 3> pos{};              // C, in slab px (u, v, w)
    std::array<float, 3> rayBase{};          // -d (view axis, toward the slab)
    std::array<float, 3> e1OverF{};          // screen basis / focal length
    std::array<float, 3> e2OverF{};
    float centerU = 0.0f, centerV = 0.0f;
    std::vector<float> layerNum;             // wPx_i - C_w
};

PerspectiveCamera perspectiveCamera(const CameraParams& cam,
                                    int numLayers,
                                    int zStart,
                                    float outputScale,
                                    float centerU,
                                    float centerV,
                                    float screenW,
                                    float screenH);

// Point mapping between screen and the slab plane at height wPx, for overlay
// positioning (crosshair, focus marker). All coordinates are relative to the
// view center; in-plane slab coords and wPx are in output pixels
// (wPx = w * outputScale, matching SlabProjection/PerspectiveCamera).
// halfSpan sets the pinhole camera distance exactly as perspectiveCamera():
// 0.5 * max(screenW, screenH); ignored for orthographic (cam.perspective==0).
// Exact for both render modes; the two functions are inverses. Degenerate
// denominators (past the horizon / behind the camera) clamp to far-but-finite
// values, matching the render's convention.
std::array<float, 2> slabPointToScreen(const CameraParams& cam,
                                       float halfSpan,
                                       const std::array<float, 2>& slabUV,
                                       float wPx);
std::array<float, 2> screenToSlabPoint(const CameraParams& cam,
                                       float halfSpan,
                                       const std::array<float, 2>& screenUV,
                                       float wPx);

// Extra slab-space sampling border (in output pixels) around the on-screen
// window. Tilted/perspective rays read the layer stack outside the screen
// rect near the edges; sampling only the screen-sized window silently clips
// the visible region there. Layer buffers must be sized
// (outH + top + bottom) x (outW + left + right), with the screen window at
// offset (left, top).
struct SlabMargins {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
    bool any() const { return left > 0 || top > 0 || right > 0 || bottom > 0; }
};

// Tight per-side margins for the given camera and slab: the bounding box of
// the screen-corner rays intersected with the outermost layers (exact for the
// rigid-slab projection the compositor uses; the mapping is affine per layer
// in the orthographic case and a homography per layer in the perspective
// case, so corners suffice). Zero at untilted orthographic. Each side is
// clamped to max(outW, outH) / 2 to bound cost at extreme tilt/perspective.
SlabMargins computeSlabMargins(const CameraParams& cam,
                               int numLayers,
                               int zStart,
                               float outputScale,
                               int outW,
                               int outH);

// Opacity transfer function: 256-entry LUT from raw value.
// alphaMin/alphaMax window the raw value to rho in [0,1], then
// alpha = opacity * rho^gamma, scaled by segmentLength (the per-frame
// ray-segment length correction 1/|d_w|) and clamped to 1. Values below
// isoCutoff get alpha 0 (hard highpass).
std::array<float, 256> buildOpacityLut(float alphaMin,
                                       float alphaMax,
                                       float opacity,
                                       float gamma,
                                       uint8_t isoCutoff,
                                       float segmentLength);

// Front-to-back emission/absorption compositing through the layer stack.
//  layerValues/layerCoverage  one buffer per layer, sized to the output
//                             raster plus `margins`; layer i sits at
//                             w = zStart + i
//  colorLut   ARGB32 emission per raw value (window/level + colormap)
//  opacityLut alpha per raw value (buildOpacityLut)
//  margins    sampling border around the screen window (computeSlabMargins);
//             the view center and camera stay anchored to the screen window
// Layers are walked near-to-far from the camera (highest w first — the camera
// is always on the +w side). Out-of-buffer or coverage-0 samples are fully
// transparent. colorOut gets the accumulated RGB over black; coverageOut is 1
// where any sample was valid; both are created at the layer size minus the
// margins.
void compositeVolumetric(const std::vector<cv::Mat_<uint8_t>>& layerValues,
                         const std::vector<cv::Mat_<uint8_t>>& layerCoverage,
                         const CameraParams& cam,
                         int zStart,
                         float outputScale,
                         const std::array<uint32_t, 256>& colorLut,
                         const std::array<float, 256>& opacityLut,
                         cv::Mat_<cv::Vec3b>& colorOut,
                         cv::Mat_<uint8_t>& coverageOut,
                         const SlabMargins& margins = {},
                         float lightingStrength = 0.0f);

} // namespace vc3d::volumetric
