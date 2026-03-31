#pragma once

#include <cstdint>
#include <memory>
#include <vector>

class Volume;

namespace vc::render {

// Platform-agnostic rendering core. Zero Qt dependencies.
// Owns a framebuffer (ARGB32) and camera state. The display layer
// (Qt/WASM/SDL) calls render() then blits pixels() to screen.
class RenderCore {
public:
    RenderCore(int width, int height);
    ~RenderCore();

    // Framebuffer
    void resize(int width, int height);
    const uint32_t* pixels() const { return pixels_.get(); }
    int width() const { return width_; }
    int height() const { return height_; }

    // Volume
    void setVolume(Volume* vol);

    // Camera: define the cutting plane
    // origin: center of the plane in world (level-0 voxel) coords
    // normal: plane normal (will be normalized)
    void setPlane(float ox, float oy, float oz,
                  float nx, float ny, float nz);

    // Camera: pan in the plane's local XY (pixel deltas at current zoom)
    void pan(float dx, float dy);

    // Camera: multiplicative zoom (>1 = zoom in, <1 = zoom out)
    void zoom(float factor);

    // Camera: scroll along the plane normal
    void scrollSlice(float dz);

    // Window/level: maps [level-window/2, level+window/2] -> [0, 255]
    void setWindowLevel(float window, float level);

    // Render the current view into the framebuffer.
    // Returns number of tiles rendered (0 if no volume).
    int render();

    // Render settings
    void setTileSize(int px);
    void setThreadCount(int n);

    // Read-only camera state (for display layers that need it)
    float scale() const { return scale_; }
    float zOffset() const { return zOff_; }

private:
    void rebuildLut();
    void recalcBasis();   // derive vx_, vy_ from normal_
    void recalcPyramidLevel();

    // Framebuffer
    std::unique_ptr<uint32_t[]> pixels_;
    int width_ = 0;
    int height_ = 0;

    // Volume (not owned)
    Volume* volume_ = nullptr;

    // Camera: plane definition
    float origin_[3] = {0, 0, 0};
    float normal_[3] = {0, 0, 1};
    float vx_[3] = {1, 0, 0};  // in-plane basis X (derived from normal)
    float vy_[3] = {0, 1, 0};  // in-plane basis Y (derived from normal)
    float scale_ = 0.5f;       // zoom: pixels per voxel
    float zOff_ = 0.0f;        // slice offset along normal

    // Pyramid level (derived from scale_)
    int dsScaleIdx_ = 1;

    // Window/level
    float windowLow_ = 0.0f;
    float windowHigh_ = 255.0f;
    uint32_t lut_[256];  // fused window/level -> ARGB32

    // Tiling
    int tilePx_ = 512;
    int threadCount_ = 0;  // 0 = hardware_concurrency
};

}  // namespace vc::render
