#pragma once

#include "chunk_cache.hpp"
#include <vector>

// Simplified volume slice renderer for WASM.
// Samples an axis-aligned or arbitrary plane through the volume,
// writing RGBA pixels to an output buffer for WebGL display.

class Renderer {
public:
    void init(int width, int height);
    void resize(int width, int height);

    // Set the slice plane: origin point + two basis vectors defining
    // the pixel-to-voxel mapping.
    // origin: (x, y, z) in voxel coordinates
    // vx_step: voxel offset per output pixel in x direction
    // vy_step: voxel offset per output pixel in y direction
    void setPlane(float ox, float oy, float oz,
                  float vx_x, float vx_y, float vx_z,
                  float vy_x, float vy_y, float vy_z);

    // Simple axis-aligned slice: show XY plane at given Z depth.
    // scale = pixels per voxel (zoom), panX/panY = pixel offset.
    void setAxisAlignedSlice(float z, float scale, float panX, float panY);

    void setZoom(float scale);
    void setLevel(int level);

    // Render current view into RGBA buffer. Returns pointer to pixel data.
    // Buffer is width*height*4 bytes (RGBA8).
    const uint8_t* render(ChunkCache& cache);

    int width() const { return width_; }
    int height() const { return height_; }
    const uint8_t* pixels() const { return pixels_.data(); }

private:
    // Trilinear sampling from cache at current level
    float sampleTrilinear(ChunkCache& cache, float vz, float vy, float vx);
    uint8_t sampleNearest(ChunkCache& cache, float vz, float vy, float vx);

    int width_ = 0;
    int height_ = 0;
    int level_ = 0;
    float zoom_ = 1.0f;

    // Plane definition
    float ox_ = 0, oy_ = 0, oz_ = 0;     // origin in voxel coords
    float vx_x_ = 1, vx_y_ = 0, vx_z_ = 0; // voxel step per output pixel X
    float vy_x_ = 0, vy_y_ = 1, vy_z_ = 0; // voxel step per output pixel Y

    std::vector<uint8_t> pixels_; // RGBA output buffer
};
