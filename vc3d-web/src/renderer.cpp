#include "renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

void Renderer::init(int width, int height) {
    width_ = width;
    height_ = height;
    pixels_.resize(width * height * 4, 0);
}

void Renderer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    pixels_.resize(width * height * 4, 0);
}

void Renderer::setPlane(float ox, float oy, float oz,
                        float vx_x, float vx_y, float vx_z,
                        float vy_x, float vy_y, float vy_z) {
    ox_ = ox; oy_ = oy; oz_ = oz;
    vx_x_ = vx_x; vx_y_ = vx_y; vx_z_ = vx_z;
    vy_x_ = vy_x; vy_y_ = vy_y; vy_z_ = vy_z;
}

void Renderer::setAxisAlignedSlice(float z, float scale, float panX, float panY) {
    // Map output pixel (px, py) to voxel (px/scale - panX, py/scale - panY, z)
    float invScale = 1.0f / scale;
    ox_ = -panX * invScale;
    oy_ = -panY * invScale;
    oz_ = z;
    vx_x_ = invScale; vx_y_ = 0; vx_z_ = 0;
    vy_x_ = 0; vy_y_ = invScale; vy_z_ = 0;
}

void Renderer::setZoom(float scale) {
    zoom_ = scale;
}

void Renderer::setLevel(int level) {
    level_ = level;
}

uint8_t Renderer::sampleNearest(ChunkCache& cache, float vz, float vy, float vx) {
    auto shape = cache.levelShape(level_);
    int sz = shape[0], sy = shape[1], sx = shape[2];
    if (vx < 0 || vy < 0 || vz < 0 || vx >= sx || vy >= sy || vz >= sz)
        return 0;

    int ix = static_cast<int>(vx + 0.5f);
    int iy = static_cast<int>(vy + 0.5f);
    int iz = static_cast<int>(vz + 0.5f);
    if (ix >= sx) ix = sx - 1;
    if (iy >= sy) iy = sy - 1;
    if (iz >= sz) iz = sz - 1;

    auto cs = cache.chunkShape(level_);
    int cz = cs[0], cy = cs[1], cx = cs[2];
    int ciz = iz / cz, ciy = iy / cy, cix = ix / cx;
    int lz = iz % cz, ly = iy % cy, lx = ix % cx;

    auto chunk = cache.get(ChunkKey{level_, ciz, ciy, cix});
    if (!chunk) return 0;

    size_t offset = static_cast<size_t>(lz) * cy * cx +
                    static_cast<size_t>(ly) * cx + lx;
    if (offset >= chunk->bytes.size()) return 0;
    return chunk->data()[offset];
}

float Renderer::sampleTrilinear(ChunkCache& cache, float vz, float vy, float vx) {
    auto shape = cache.levelShape(level_);
    int sz = shape[0], sy = shape[1], sx = shape[2];
    if (vx < 0 || vy < 0 || vz < 0 || vx >= sx - 1 || vy >= sy - 1 || vz >= sz - 1)
        return sampleNearest(cache, vz, vy, vx);

    int ix = static_cast<int>(vx);
    int iy = static_cast<int>(vy);
    int iz = static_cast<int>(vz);
    float fx = vx - ix, fy = vy - iy, fz = vz - iz;

    auto cs = cache.chunkShape(level_);
    int cz = cs[0], cy = cs[1], cx = cs[2];

    // Sample 8 corners -- for simplicity, use nearest for each corner
    auto sample = [&](int z, int y, int x) -> float {
        int ciz = z / cz, ciy = y / cy, cix = x / cx;
        int lz = z % cz, ly = y % cy, lx = x % cx;
        auto chunk = cache.get(ChunkKey{level_, ciz, ciy, cix});
        if (!chunk) return 0;
        size_t off = static_cast<size_t>(lz) * cy * cx +
                     static_cast<size_t>(ly) * cx + lx;
        if (off >= chunk->bytes.size()) return 0;
        return chunk->data()[off];
    };

    float c000 = sample(iz, iy, ix);
    float c001 = sample(iz, iy, ix + 1);
    float c010 = sample(iz, iy + 1, ix);
    float c011 = sample(iz, iy + 1, ix + 1);
    float c100 = sample(iz + 1, iy, ix);
    float c101 = sample(iz + 1, iy, ix + 1);
    float c110 = sample(iz + 1, iy + 1, ix);
    float c111 = sample(iz + 1, iy + 1, ix + 1);

    float c00 = c000 + fx * (c001 - c000);
    float c01 = c010 + fx * (c011 - c010);
    float c10 = c100 + fx * (c101 - c100);
    float c11 = c110 + fx * (c111 - c110);

    float c0 = c00 + fy * (c01 - c00);
    float c1 = c10 + fy * (c11 - c10);

    return c0 + fz * (c1 - c0);
}

const uint8_t* Renderer::render(ChunkCache& cache) {
    if (width_ <= 0 || height_ <= 0) return pixels_.data();

    std::memset(pixels_.data(), 0, pixels_.size());

    for (int py = 0; py < height_; py++) {
        for (int px = 0; px < width_; px++) {
            float vx = ox_ + vx_x_ * px + vy_x_ * py;
            float vy = oy_ + vx_y_ * px + vy_y_ * py;
            float vz = oz_ + vx_z_ * px + vy_z_ * py;

            float val = sampleTrilinear(cache, vz, vy, vx);
            uint8_t v = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));

            size_t idx = (py * width_ + px) * 4;
            pixels_[idx + 0] = v; // R
            pixels_[idx + 1] = v; // G
            pixels_[idx + 2] = v; // B
            pixels_[idx + 3] = 255; // A
        }
    }

    return pixels_.data();
}
