#pragma once
// Shared by the CPU and CUDA view implementations.
#include "vc_reader/views.hpp"
#include "../splitmix.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#define VCR_VIEWS_HD __host__ __device__ inline
#else
#define VCR_VIEWS_HD inline
#endif

namespace vcr::views::detail {
inline void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

// The stream documented in views.hpp.
struct Stream {
    std::uint64_t state;
    explicit Stream(std::uint64_t seed) : state(seed) {}
    std::uint64_t bits() { return splitmix::next(state); }
    double uniform() { return double(bits() >> 11) * 0x1.0p-53; }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }
    std::uint64_t below(std::uint64_t n)
    {
        const std::uint64_t threshold = (std::uint64_t(0) - n) % n;
        std::uint64_t value;
        do { value = bits(); } while (value < threshold);
        return value % n;
    }
};

inline constexpr int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

inline std::int64_t product(const Index3& n) { return n[0] * n[1] * n[2]; }

// Token cells: shape / patch per axis, validated.
inline Index3 cells(const Index3& shape, const Index3& patch)
{
    Index3 out{};
    for (int a = 0; a < 3; ++a) {
        require(shape[a] > 0 && patch[a] > 0, "shape and patch must be positive");
        require(shape[a] % patch[a] == 0, "shape must be divisible by patch on every axis");
        out[a] = shape[a] / patch[a];
    }
    return out;
}

// A pair with orientations resolved, as kernels read it.
struct PairPlan {
    std::int64_t grid;
    std::int32_t studentOrigin[3], teacherOrigin[3];
    std::int8_t studentPerm[3], teacherPerm[3];
    std::int8_t studentFlip[3], teacherFlip[3];
};
struct PairGeometry {
    std::int32_t student[3], teacher[3];               // unoriented shapes
    std::int32_t tokens[3];                             // support grid shape
    std::int64_t gridSize, studentTokens;
};

inline PairGeometry pairGeometry(const Supports& support, const Pairs& pairs)
{
    const auto limit = std::int64_t(std::numeric_limits<std::int32_t>::max());
    require(support.grids > 0 || pairs.count == 0, "support needs at least one grid");
    PairGeometry g{};
    for (int a = 0; a < 3; ++a) {
        require(support.shape[a] > 0 && support.shape[a] <= limit, "support grid shape must be positive and fit int32");
        require(pairs.studentShape[a] > 0 && pairs.teacherShape[a] > 0, "crop shapes must be positive");
        require(pairs.studentShape[a] <= limit && pairs.teacherShape[a] <= limit, "crop shapes must fit int32");
        g.student[a] = std::int32_t(pairs.studentShape[a]);
        g.teacher[a] = std::int32_t(pairs.teacherShape[a]);
        g.tokens[a] = std::int32_t(support.shape[a]);
    }
    require(product(pairs.teacherShape) <= limit, "teacher token count must fit int32");
    require(product(pairs.studentShape) <= limit, "student token count must fit int32");
    g.gridSize = product(support.shape);
    g.studentTokens = product(pairs.studentShape);
    return g;
}

inline Orientation resolve(std::int64_t id)
{
    require(id >= 0 && id < ORIENTATION_COUNT, "orientation id out of range");
    return orientation(int(id));
}

inline std::vector<PairPlan> pairPlans(const Supports& support, const Pairs& pairs, const PairGeometry& g)
{
    std::vector<PairPlan> plans(pairs.count);
    if (pairs.count)
        require(pairs.supportIndex && pairs.studentOrigins && pairs.studentOrientations && pairs.teacherOrigins &&
                pairs.teacherOrientations, "pair arrays are required");
    // Lattice coordinates of every crop token must fit int32.
    const auto lo = std::int64_t(std::numeric_limits<std::int32_t>::min());
    const auto hi = std::int64_t(std::numeric_limits<std::int32_t>::max());
    for (std::size_t p = 0; p < pairs.count; ++p) {
        auto& plan = plans[p];
        require(pairs.supportIndex[p] >= 0 && std::uint64_t(pairs.supportIndex[p]) < support.grids,
                "support index out of range");
        plan.grid = pairs.supportIndex[p];
        const auto s = resolve(pairs.studentOrientations[p]), t = resolve(pairs.teacherOrientations[p]);
        for (int a = 0; a < 3; ++a) {
            const auto so = pairs.studentOrigins[3 * p + a], to = pairs.teacherOrigins[3 * p + a];
            require(so >= lo && so <= hi - g.student[a] + 1, "student origin out of int32 range");
            require(to >= lo && to <= hi - g.teacher[a] + 1, "teacher origin out of int32 range");
            plan.studentOrigin[a] = std::int32_t(so);
            plan.teacherOrigin[a] = std::int32_t(to);
            plan.studentPerm[a] = std::int8_t(s.perm[a]);
            plan.teacherPerm[a] = std::int8_t(t.perm[a]);
            plan.studentFlip[a] = s.flips[a];
            plan.teacherFlip[a] = t.flips[a];
        }
    }
    return plans;
}

// One student token of one pair.
VCR_VIEWS_HD void correspond(const PairGeometry& g, const PairPlan& p, const std::uint8_t* support,
                             std::int64_t i, std::int32_t* teacherIndex, std::uint8_t* studentValid,
                             std::int32_t* lattice)
{
    std::int32_t os[3], ot[3];
    for (int k = 0; k < 3; ++k) {
        os[k] = g.student[p.studentPerm[k]];
        ot[k] = g.teacher[p.teacherPerm[k]];
    }
    std::int64_t at[3];
    at[2] = i % os[2];
    at[1] = (i / os[2]) % os[1];
    at[0] = i / (std::int64_t(os[1]) * os[2]);
    std::int64_t lat[3];
    for (int k = 0; k < 3; ++k) {
        const int axis = p.studentPerm[k];
        const std::int64_t u = p.studentFlip[k] ? g.student[axis] - 1 - at[k] : at[k];
        lat[axis] = p.studentOrigin[axis] + u;
    }
    bool inGrid = true;
    for (int a = 0; a < 3; ++a) inGrid = inGrid && lat[a] >= 0 && lat[a] < g.tokens[a];
    const bool valid = inGrid &&
        support[p.grid * g.gridSize + (lat[0] * g.tokens[1] + lat[1]) * g.tokens[2] + lat[2]] != 0;
    std::int64_t ut[3];
    bool inside = valid;
    for (int a = 0; a < 3; ++a) {
        ut[a] = lat[a] - p.teacherOrigin[a];
        inside = inside && ut[a] >= 0 && ut[a] < g.teacher[a];
    }
    std::int32_t index = -1;
    if (inside) {
        std::int64_t j[3];
        for (int k = 0; k < 3; ++k) {
            const int axis = p.teacherPerm[k];
            j[k] = p.teacherFlip[k] ? g.teacher[axis] - 1 - ut[axis] : ut[axis];
        }
        index = std::int32_t((j[0] * ot[1] + j[1]) * ot[2] + j[2]);
    }
    teacherIndex[i] = index;
    studentValid[i] = valid ? 1 : 0;
    if (lattice)
        for (int a = 0; a < 3; ++a) lattice[3 * i + a] = std::int32_t(lat[a]);
}

// Token support of one cell.
VCR_VIEWS_HD std::uint8_t cellSupported(const std::uint8_t* valid, const std::int64_t* shape,
                                        const std::int64_t* patch, std::int64_t z, std::int64_t y, std::int64_t x)
{
    for (std::int64_t dz = 0; dz < patch[0]; ++dz)
        for (std::int64_t dy = 0; dy < patch[1]; ++dy) {
            const auto* row = valid + ((z * patch[0] + dz) * shape[1] + y * patch[1] + dy) * shape[2] + x * patch[2];
            for (std::int64_t dx = 0; dx < patch[2]; ++dx)
                if (!row[dx]) return 0;
        }
    return 1;
}

// float to IEEE binary16, round to nearest even (F. Giesen's float_to_half_fast3_rtne).
inline std::uint16_t toHalf(float value)
{
    std::uint32_t f;
    std::memcpy(&f, &value, 4);
    const std::uint32_t sign = f & 0x80000000u;
    f ^= sign;
    std::uint16_t o;
    if (f >= (127u + 16u) << 23) {
        o = f > 0x7f800000u ? 0x7e00 : 0x7c00;
    } else if (f < 113u << 23) {
        const std::uint32_t magicBits = ((127u - 15u) + (23u - 10u) + 1u) << 23;
        float magic, x;
        std::memcpy(&magic, &magicBits, 4);
        std::memcpy(&x, &f, 4);
        x += magic;
        std::memcpy(&f, &x, 4);
        o = std::uint16_t(f - magicBits);
    } else {
        const std::uint32_t odd = (f >> 13) & 1u;
        f += (std::uint32_t(15 - 127) << 23) + 0xfffu;
        f += odd;
        o = std::uint16_t(f >> 13);
    }
    return std::uint16_t(o | (sign >> 16));
}

inline std::uint16_t toBFloat16(float value)
{
    std::uint32_t f;
    std::memcpy(&f, &value, 4);
    if ((f & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((f >> 16) | 0x40u);
    return std::uint16_t((f + 0x7fffu + ((f >> 16) & 1u)) >> 16);
}

// The fill's bit pattern and element size.
inline std::uint32_t fillBits(Real type, float fill)
{
    if (type == Real::Float16) return toHalf(fill);
    if (type == Real::BFloat16) return toBFloat16(fill);
    std::uint32_t bits;
    std::memcpy(&bits, &fill, 4);
    return bits;
}
inline std::size_t elementBytes(Real type) { return type == Real::Float32 ? 4 : 2; }

inline Index3 validateCells(const CellImages& images)
{
    const auto grid = cells(images.shape, images.patch);
    require(images.batch == 0 || images.channels == 0 || (images.images && images.out), "image pointers are required");
    require(images.type == Real::Float32 || images.type == Real::Float16 || images.type == Real::BFloat16,
            "unsupported image type");
    return grid;
}
}  // namespace vcr::views::detail
