#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace vc {

// -- Vectors ----------------------------------------------------------------
// Minimal value types with named fields. Arithmetic is free functions so the
// structs stay trivial aggregates.

struct Vec2i { int x, y; };
struct Vec3i { int x, y, z; };
struct Vec2f { float x, y; };
struct Vec3f { float x, y, z; };
struct Vec3d { double x, y, z; };

// Arithmetic
constexpr Vec3f operator+(Vec3f a, Vec3f b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
constexpr Vec3f operator-(Vec3f a, Vec3f b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
constexpr Vec3f operator*(Vec3f v, float s) { return {v.x*s, v.y*s, v.z*s}; }
constexpr Vec3f operator*(float s, Vec3f v) { return v * s; }
constexpr float dot(Vec3f a, Vec3f b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
constexpr Vec3f cross(Vec3f a, Vec3f b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float length(Vec3f v) { return std::sqrt(dot(v, v)); }
inline Vec3f normalize(Vec3f v) { float l = length(v); return v * (1.0f / l); }

constexpr Vec2f operator+(Vec2f a, Vec2f b) { return {a.x+b.x, a.y+b.y}; }
constexpr Vec2f operator-(Vec2f a, Vec2f b) { return {a.x-b.x, a.y-b.y}; }
constexpr Vec2f operator*(Vec2f v, float s) { return {v.x*s, v.y*s}; }

constexpr Vec3i operator+(Vec3i a, Vec3i b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
constexpr Vec3i operator-(Vec3i a, Vec3i b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }

constexpr bool operator==(Vec2i a, Vec2i b) { return a.x==b.x && a.y==b.y; }
constexpr bool operator==(Vec3i a, Vec3i b) { return a.x==b.x && a.y==b.y && a.z==b.z; }
constexpr bool operator==(Vec2f a, Vec2f b) { return a.x==b.x && a.y==b.y; }
constexpr bool operator==(Vec3f a, Vec3f b) { return a.x==b.x && a.y==b.y && a.z==b.z; }

// -- Geometry ---------------------------------------------------------------

struct Rect { int x, y, w, h; };
struct Box3f { Vec3f min, max; };

constexpr bool contains(Rect r, Vec2i p) {
    return p.x >= r.x && p.x < r.x+r.w && p.y >= r.y && p.y < r.y+r.h;
}
constexpr bool contains(Box3f b, Vec3f p) {
    return p.x >= b.min.x && p.x <= b.max.x
        && p.y >= b.min.y && p.y <= b.max.y
        && p.z >= b.min.z && p.z <= b.max.z;
}

// -- Matrix<T> --------------------------------------------------------------
// Owning row-major 2D buffer. Use mdspan() for a non-owning view.

template<typename T>
struct Matrix {
    std::vector<T> storage;
    int rows = 0, cols = 0;

    Matrix() = default;
    Matrix(int r, int c) : storage(r * c), rows(r), cols(c) {}
    Matrix(int r, int c, T fill) : storage(r * c, fill), rows(r), cols(c) {}

    T& operator()(int r, int c) { return storage[r * cols + c]; }
    const T& operator()(int r, int c) const { return storage[r * cols + c]; }

    T* data() { return storage.data(); }
    const T* data() const { return storage.data(); }
    T* row(int r) { return storage.data() + r * cols; }
    const T* row(int r) const { return storage.data() + r * cols; }

    std::size_t size_bytes() const { return storage.size() * sizeof(T); }
    bool empty() const { return storage.empty(); }

    std::span<T> flat() { return storage; }
    std::span<const T> flat() const { return storage; }
};

// -- Framebuffer ------------------------------------------------------------
// ARGB32 pixel buffer for rendering output.

struct Framebuffer {
    std::vector<uint32_t> pixels;
    int width = 0, height = 0;

    Framebuffer() = default;
    Framebuffer(int w, int h) : pixels(w * h), width(w), height(h) {}

    uint32_t* data() { return pixels.data(); }
    int stride() const { return width; }
};

// -- Volume hierarchy constants ---------------------------------------------

inline constexpr int BLOCK_DIM = 16;  // smallest addressable unit

} // namespace vc
