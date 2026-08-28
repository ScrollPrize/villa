// Shared R-tree point-to-quad-surface core used by spiral_graph, the
// vc_spiral.surface_index binding, and the standalone patch merger tools.
// Query results are reported per input point in ascending surface-index
// order, keeping the closest hit per surface within tolerance.  Unlike the
// historical implementation, quads are evaluated as bilinear patches rather
// than as two triangles, so fractional grid coordinates refer to the same
// interpolation used by the surface itself.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>

namespace surfcore {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& a, float scalar)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

inline float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float distance_squared(const Vec3& a, const Vec3& b)
{
    const Vec3 d = a - b;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

inline bool finite(const Vec3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

// QuadSurface uses x == -1 as the inexpensive validity sentinel. Tifxyz's
// invalid vertices are [-1, -1, -1], so retain the same test here.
inline bool valid_surface_point(const Vec3& point)
{
    return point.x != -1.0f && finite(point);
}

struct TriangleHit {
    Vec3 bary{};
    float distance_squared = std::numeric_limits<float>::max();
};

inline TriangleHit closest_point_on_triangle(
    const Vec3& point, const Vec3& a, const Vec3& b, const Vec3& c)
{
    TriangleHit hit;
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        hit.bary = {1.0f, 0.0f, 0.0f};
        hit.distance_squared = distance_squared(point, a);
        return hit;
    }

    const Vec3 bp = point - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        hit.bary = {0.0f, 1.0f, 0.0f};
        hit.distance_squared = distance_squared(point, b);
        return hit;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        hit.bary = {1.0f - v, v, 0.0f};
        hit.distance_squared = distance_squared(point, a + ab * v);
        return hit;
    }

    const Vec3 cp = point - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        hit.bary = {0.0f, 0.0f, 1.0f};
        hit.distance_squared = distance_squared(point, c);
        return hit;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        hit.bary = {1.0f - w, 0.0f, w};
        hit.distance_squared = distance_squared(point, a + ac * w);
        return hit;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.bary = {0.0f, 1.0f - w, w};
        hit.distance_squared = distance_squared(point, b + (c - b) * w);
        return hit;
    }

    const float inverse_denom = 1.0f / (va + vb + vc);
    const float v = vb * inverse_denom;
    const float w = vc * inverse_denom;
    hit.bary = {1.0f - v - w, v, w};
    hit.distance_squared = distance_squared(point, a + ab * v + ac * w);
    return hit;
}

inline float clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

struct BilinearHit {
    bool valid = false;
    double u = 0.0;
    double v = 0.0;
    double distance_squared = std::numeric_limits<double>::infinity();
};

struct PolynomialRoots {
    std::array<double, 5> values{};
    std::size_t count = 0;

    void add(double value)
    {
        if (!std::isfinite(value) || value < -1e-12 || value > 1.0 + 1e-12) {
            return;
        }
        value = std::max(0.0, std::min(1.0, value));
        for (std::size_t index = 0; index < count; ++index) {
            if (std::abs(values[index] - value) <= 2e-11) {
                return;
            }
        }
        if (count < values.size()) {
            values[count++] = value;
        }
    }
};

inline double polynomial_value(
    const std::array<double, 6>& coefficients, int degree, double value)
{
    double result = coefficients[static_cast<std::size_t>(degree)];
    for (int power = degree - 1; power >= 0; --power) {
        result = result * value + coefficients[static_cast<std::size_t>(power)];
    }
    return result;
}

// Isolate every real polynomial root in [0, 1].  Roots of the derivative
// partition the domain into monotone intervals, so recursive isolation plus
// bisection is complete for the degree-five polynomial used below.  Testing
// the partition points also retains even-multiplicity roots.
inline PolynomialRoots polynomial_roots_on_unit_interval(
    std::array<double, 6> coefficients, int degree)
{
    double coefficient_scale = 0.0;
    for (int power = 0; power <= degree; ++power) {
        coefficient_scale = std::max(
            coefficient_scale, std::abs(coefficients[static_cast<std::size_t>(power)]));
    }
    if (coefficient_scale == 0.0) {
        return {};
    }
    const double trim_tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * coefficient_scale;
    while (degree > 0
           && std::abs(coefficients[static_cast<std::size_t>(degree)])
               <= trim_tolerance) {
        --degree;
    }

    PolynomialRoots roots;
    if (degree == 0) {
        return roots;
    }
    if (degree == 1) {
        roots.add(-coefficients[0] / coefficients[1]);
        return roots;
    }

    std::array<double, 6> derivative{};
    for (int power = 1; power <= degree; ++power) {
        derivative[static_cast<std::size_t>(power - 1)]
            = static_cast<double>(power)
            * coefficients[static_cast<std::size_t>(power)];
    }
    PolynomialRoots critical = polynomial_roots_on_unit_interval(
        derivative, degree - 1);
    std::sort(critical.values.begin(), critical.values.begin() + critical.count);

    std::array<double, 7> partition{};
    std::size_t partition_count = 0;
    partition[partition_count++] = 0.0;
    for (std::size_t index = 0; index < critical.count; ++index) {
        if (critical.values[index] > 0.0 && critical.values[index] < 1.0) {
            partition[partition_count++] = critical.values[index];
        }
    }
    partition[partition_count++] = 1.0;

    double value_scale = 0.0;
    for (int power = 0; power <= degree; ++power) {
        value_scale += std::abs(coefficients[static_cast<std::size_t>(power)]);
    }
    const double zero_tolerance = 512.0 * std::numeric_limits<double>::epsilon()
        * value_scale;
    for (std::size_t index = 0; index < partition_count; ++index) {
        const double value = polynomial_value(
            coefficients, degree, partition[index]);
        if (std::abs(value) <= zero_tolerance) {
            roots.add(partition[index]);
        }
    }

    for (std::size_t index = 0; index + 1 < partition_count; ++index) {
        double low = partition[index];
        double high = partition[index + 1];
        double low_value = polynomial_value(coefficients, degree, low);
        double high_value = polynomial_value(coefficients, degree, high);
        if (std::abs(low_value) <= zero_tolerance
            || std::abs(high_value) <= zero_tolerance
            || std::signbit(low_value) == std::signbit(high_value)) {
            continue;
        }
        for (int iteration = 0; iteration < 64; ++iteration) {
            const double middle = 0.5 * (low + high);
            const double middle_value = polynomial_value(coefficients, degree, middle);
            if (std::signbit(middle_value) == std::signbit(low_value)) {
                low = middle;
                low_value = middle_value;
            } else {
                high = middle;
                high_value = middle_value;
            }
        }
        roots.add(0.5 * (low + high));
    }
    std::sort(roots.values.begin(), roots.values.begin() + roots.count);
    return roots;
}

// Return the closest point on the actual bilinear quad.  The former
// two-triangle implementation gave coordinates on a piecewise-planar
// surrogate, which is measurably wrong for a twisted quad even when the query
// lies exactly on the bilinear surface.
//
// Boundary minima are line-segment projections.  Interior minima are all
// roots of the exact degree-five stationarity polynomial, so a sufficiently
// warped quad with several local minima is still evaluated globally without
// introducing a subdivision approximation.
inline BilinearHit closest_point_on_bilinear_quad(
    const Vec3& point,
    const Vec3& p00,
    const Vec3& p10,
    const Vec3& p01,
    const Vec3& p11)
{
    struct DVec3 {
        double x;
        double y;
        double z;
    };
    const auto convert = [](const Vec3& value) {
        return DVec3{value.x, value.y, value.z};
    };
    const auto add = [](const DVec3& a, const DVec3& b) {
        return DVec3{a.x + b.x, a.y + b.y, a.z + b.z};
    };
    const auto subtract = [](const DVec3& a, const DVec3& b) {
        return DVec3{a.x - b.x, a.y - b.y, a.z - b.z};
    };
    const auto multiply = [](const DVec3& value, double scalar) {
        return DVec3{value.x * scalar, value.y * scalar, value.z * scalar};
    };
    const auto ddot = [](const DVec3& a, const DVec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    const auto clamp = [](double value) {
        return std::max(0.0, std::min(1.0, value));
    };

    const DVec3 q = convert(point);
    const DVec3 a = convert(p00);
    const DVec3 b = subtract(convert(p10), a);
    const DVec3 c = subtract(convert(p01), a);
    const DVec3 d = add(
        subtract(convert(p11), convert(p10)),
        subtract(convert(p00), convert(p01)));

    const auto surface_point = [&](double u, double v) {
        return add(add(a, multiply(b, u)),
                   add(multiply(c, v), multiply(d, u * v)));
    };
    const auto objective = [&](double u, double v) {
        const DVec3 residual = subtract(surface_point(u, v), q);
        return ddot(residual, residual);
    };

    BilinearHit best;
    const auto record = [&](double u, double v) {
        u = clamp(u);
        v = clamp(v);
        const double squared = objective(u, v);
        if (!std::isfinite(squared) || squared >= best.distance_squared) {
            return;
        }
        best = {true, u, v, squared};
    };

    // Every constrained minimum on an edge is an ordinary segment
    // projection.  Recording all four also covers corners and all degenerate
    // quads before the interior solve begins.
    const auto record_segment = [&](const DVec3& start, const DVec3& end,
                                    bool varying_u, double fixed) {
        const DVec3 direction = subtract(end, start);
        const double length_squared = ddot(direction, direction);
        double parameter = 0.0;
        if (length_squared > 0.0) {
            parameter = clamp(ddot(subtract(q, start), direction) / length_squared);
        }
        if (varying_u) {
            record(parameter, fixed);
        } else {
            record(fixed, parameter);
        }
    };
    record_segment(a, add(a, b), true, 0.0);
    record_segment(add(a, c), add(add(a, c), add(b, d)), true, 1.0);
    record_segment(a, add(a, c), false, 0.0);
    record_segment(add(a, b), add(add(a, b), add(c, d)), false, 1.0);

    // For a fixed v, the unconstrained minimizing u is -P(v)/Q(v), where
    // P=(A+vC).(B+vD) and Q=|B+vD|^2.  Substitution into the remaining
    // stationarity equation produces a degree-five polynomial:
    //
    //   R Q^2 - P S Q + P^2 T = 0.
    //
    // Isolating all its roots gives every interior stationary point; combined
    // with the four exact edge minima above, this evaluates the global
    // constrained minimum rather than relying on a local iterative fit.
    const DVec3 residual_base = subtract(a, q);
    std::array<double, 6> p{};
    p[0] = ddot(residual_base, b);
    p[1] = ddot(residual_base, d) + ddot(c, b);
    p[2] = ddot(c, d);
    std::array<double, 6> q_polynomial{};
    q_polynomial[0] = ddot(b, b);
    q_polynomial[1] = 2.0 * ddot(b, d);
    q_polynomial[2] = ddot(d, d);
    std::array<double, 6> r{};
    r[0] = ddot(residual_base, c);
    r[1] = ddot(c, c);
    std::array<double, 6> s{};
    s[0] = ddot(residual_base, d) + ddot(b, c);
    s[1] = 2.0 * ddot(c, d);
    std::array<double, 6> t{};
    t[0] = ddot(b, d);
    t[1] = ddot(d, d);

    const auto polynomial_product = [](
        const std::array<double, 6>& first,
        const std::array<double, 6>& second) {
        std::array<double, 6> result{};
        for (std::size_t i = 0; i < first.size(); ++i) {
            for (std::size_t j = 0; i + j < result.size(); ++j) {
                result[i + j] += first[i] * second[j];
            }
        }
        return result;
    };
    const auto rq2 = polynomial_product(
        r, polynomial_product(q_polynomial, q_polynomial));
    const auto psq = polynomial_product(
        polynomial_product(p, s), q_polynomial);
    const auto p2t = polynomial_product(polynomial_product(p, p), t);
    std::array<double, 6> stationary{};
    for (std::size_t power = 0; power < stationary.size(); ++power) {
        stationary[power] = rq2[power] - psq[power] + p2t[power];
    }

    const PolynomialRoots roots = polynomial_roots_on_unit_interval(stationary, 5);
    const double q_scale = std::max({
        std::abs(q_polynomial[0]),
        std::abs(q_polynomial[1]), std::abs(q_polynomial[2])});
    for (std::size_t index = 0; index < roots.count; ++index) {
        const double v = roots.values[index];
        const double denominator = polynomial_value(q_polynomial, 2, v);
        if (q_scale == 0.0
            || denominator <= 64.0 * std::numeric_limits<double>::epsilon() * q_scale) {
            continue;
        }
        const double u = -polynomial_value(p, 2, v) / denominator;
        if (u >= -1e-11 && u <= 1.0 + 1e-11) {
            record(u, v);
        }
    }

    // Polish the best algebraic candidate in parameter space.  Besides
    // removing root-isolation roundoff, this handles nearly multiple roots
    // where the degree-five polynomial is poorly conditioned.  The bounded
    // Gauss-Newton step is deterministic and only refines an already-global
    // candidate; exact edge minima remain part of the comparison above.
    if (best.valid) {
        double u = best.u;
        double v = best.v;
        for (int iteration = 0; iteration < 24; ++iteration) {
            const DVec3 residual = subtract(surface_point(u, v), q);
            const DVec3 tangent_u = add(b, multiply(d, v));
            const DVec3 tangent_v = add(c, multiply(d, u));
            const double h00 = ddot(tangent_u, tangent_u);
            const double h01 = ddot(tangent_u, tangent_v);
            const double h11 = ddot(tangent_v, tangent_v);
            const double g0 = ddot(tangent_u, residual);
            const double g1 = ddot(tangent_v, residual);
            const double determinant = h00 * h11 - h01 * h01;
            const double scale = std::max({1.0, h00, h11});
            if (determinant <= 128.0 * std::numeric_limits<double>::epsilon()
                    * scale * scale) break;
            const double du = (-h11 * g0 + h01 * g1) / determinant;
            const double dv = (h01 * g0 - h00 * g1) / determinant;
            if (std::max(std::abs(du), std::abs(dv)) <= 2e-14) break;
            const double before = objective(u, v);
            bool accepted = false;
            double step = 1.0;
            for (int line_search = 0; line_search < 16; ++line_search) {
                const double next_u = clamp(u + step * du);
                const double next_v = clamp(v + step * dv);
                if (objective(next_u, next_v) < before) {
                    u = next_u;
                    v = next_v;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) break;
        }
        record(u, v);
    }

    // Remove insignificant root-isolation residue for exact-on-surface queries,
    // preserving the public zero-tolerance behavior.  The threshold is far
    // below float input resolution, including for large world coordinates.
    const double coordinate_scale = std::max({
        1.0, std::abs(static_cast<double>(point.x)),
        std::abs(static_cast<double>(point.y)),
        std::abs(static_cast<double>(point.z)),
        std::abs(static_cast<double>(p00.x)), std::abs(static_cast<double>(p00.y)),
        std::abs(static_cast<double>(p00.z)), std::abs(static_cast<double>(p10.x)),
        std::abs(static_cast<double>(p10.y)), std::abs(static_cast<double>(p10.z)),
        std::abs(static_cast<double>(p01.x)), std::abs(static_cast<double>(p01.y)),
        std::abs(static_cast<double>(p01.z)), std::abs(static_cast<double>(p11.x)),
        std::abs(static_cast<double>(p11.y)), std::abs(static_cast<double>(p11.z))});
    const double zero_distance = 64.0 * std::numeric_limits<double>::epsilon()
        * coordinate_scale;
    if (best.valid && best.distance_squared <= zero_distance * zero_distance) {
        best.distance_squared = 0.0;
    }
    return best;
}

struct Aabb {
    Vec3 low{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Vec3 high{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };

    void extend(const Vec3& point)
    {
        low.x = std::min(low.x, point.x);
        low.y = std::min(low.y, point.y);
        low.z = std::min(low.z, point.z);
        high.x = std::max(high.x, point.x);
        high.y = std::max(high.y, point.y);
        high.z = std::max(high.z, point.z);
    }

    void extend(const Aabb& other)
    {
        extend(other.low);
        extend(other.high);
    }

    void pad(float padding)
    {
        if (padding <= 0.0f) {
            return;
        }
        low.x -= padding;
        low.y -= padding;
        low.z -= padding;
        high.x += padding;
        high.y += padding;
        high.z += padding;
    }

    bool intersects(const Aabb& other) const
    {
        return low.x <= other.high.x && high.x >= other.low.x
            && low.y <= other.high.y && high.y >= other.low.y
            && low.z <= other.high.z && high.z >= other.low.z;
    }

    float centroid(int axis) const
    {
        if (axis == 0) {
            return (low.x + high.x) * 0.5f;
        }
        if (axis == 1) {
            return (low.y + high.y) * 0.5f;
        }
        return (low.z + high.z) * 0.5f;
    }

    bool valid() const
    {
        return finite(low) && finite(high)
            && low.x <= high.x && low.y <= high.y && low.z <= high.z;
    }
};

class SurfacePointSource {
public:
    virtual ~SurfacePointSource() = default;
    virtual Vec3 at(size_t row, size_t col) const = 0;
    virtual bool valid(size_t row, size_t col) const
    {
        return valid_surface_point(at(row, col));
    }
};

// Opens the three float32 coordinate bands without copying their pixels.
// Mappings close their file descriptors after construction and support
// concurrent point reads. Unsupported TIFF layouts throw so callers can
// explicitly fall back to owned geometry.
std::shared_ptr<const SurfacePointSource> open_mapped_tifxyz_point_source(
    const std::filesystem::path& directory, size_t rows, size_t cols);

// Decode a TIFXYZ vertex mask while leaving the much larger coordinate planes
// memory-mapped. Nonzero mask values are valid.
std::vector<std::uint8_t> read_tifxyz_mask(
    const std::filesystem::path& path, size_t rows, size_t cols);

struct SurfaceData {
    std::string id;
    size_t rows = 0;
    size_t cols = 0;
    float scale_i = 1.0f;
    float scale_j = 1.0f;
    std::vector<Vec3> xyz;
    std::shared_ptr<const SurfacePointSource> point_source;
    // Optional declared TIFXYZ metadata bbox used for the first, one-box-per-
    // surface broad phase. Geometry-derived tile bounds remain authoritative
    // in the next phase.
    Aabb metadata_bounds;
    bool has_metadata_bounds = false;
    // Optional row-major (rows - 1) x (cols - 1) mask. An empty mask keeps
    // the historical vertex-sentinel behavior used by the public index.
    std::vector<uint8_t> valid_quads;

    Vec3 at(size_t row, size_t col) const
    {
        return point_source ? point_source->at(row, col) : xyz[row * cols + col];
    }

    bool valid_point(size_t row, size_t col) const
    {
        return point_source ? point_source->valid(row, col)
                            : valid_surface_point(xyz[row * cols + col]);
    }

    bool valid_quad(size_t row, size_t col) const
    {
        if (!valid_quads.empty()) {
            return valid_quads[row * (cols - 1) + col] != 0;
        }
        return valid_point(row, col) && valid_point(row, col + 1)
            && valid_point(row + 1, col) && valid_point(row + 1, col + 1);
    }
};

struct Tile {
    Aabb bounds;
    int32_t surface = -1;
    int row = 0;
    int col = 0;
};

// Broad-phase evidence that one tile of a seed surface has a padded AABB
// intersection with at least one tile of another surface. The narrow phase
// still validates the continuous bilinear geometry before accepting a hit.
struct SurfaceTileCandidate {
    int32_t surface = -1;
    int seed_row = 0;
    int seed_col = 0;
    int target_row = 0;
    int target_col = 0;
    uint32_t seed_tile = 0;
    uint32_t target_tile = 0;
};

// Closest-point hit on one surface. ``i`` runs along columns and ``j`` along
// rows, matching the original module's convention (the Python wrapper emits
// (j, i) pairs, i.e. (row, col)).
struct BestHit {
    bool valid = false;
    float distance_squared = std::numeric_limits<float>::infinity();
    float i = 0.0f;
    float j = 0.0f;
    uint32_t tile = std::numeric_limits<uint32_t>::max();
};

// One query result: the closest point on ``surface`` within tolerance.
struct SurfaceHit {
    int32_t surface = -1;
    float distance = 0.0f;
    float i = 0.0f;  // column coordinate
    float j = 0.0f;  // row coordinate
    uint32_t tile = std::numeric_limits<uint32_t>::max();
};

// Reusable per-thread buffers so hot query loops perform no allocations.
struct QueryScratch {
    std::vector<uint32_t> tiles;
    std::vector<int32_t> surfaces;
    std::vector<BestHit> best_hits;
};

class SurfacePatchIndex {
    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box3 = bg::model::box<Point3>;
    using TreeEntry = std::pair<Box3, uint32_t>;
    using TileTree = bgi::rtree<TreeEntry, bgi::quadratic<32>>;

    static Box3 tree_box(const Aabb& bounds)
    {
        return Box3(
            Point3(bounds.low.x, bounds.low.y, bounds.low.z),
            Point3(bounds.high.x, bounds.high.y, bounds.high.z));
    }

public:
    void rebuild(
        std::vector<std::shared_ptr<SurfaceData>> new_surfaces,
        float bbox_padding = 0.0f,
        int sampling_stride = 1)
    {
        if (sampling_stride < 1) {
            throw std::runtime_error("sampling_stride must be >= 1");
        }

        const int new_tile_stride = compute_tile_stride(sampling_stride);
        std::vector<std::vector<Tile>> per_surface_tiles(new_surfaces.size());
#pragma omp parallel for schedule(dynamic, 16)
        for (std::int64_t signed_surface_index = 0;
             signed_surface_index < static_cast<std::int64_t>(new_surfaces.size());
             ++signed_surface_index) {
            const size_t surface_index = static_cast<size_t>(signed_surface_index);
            const SurfaceData& surface = *new_surfaces[surface_index];
            if (surface.rows < 2 || surface.cols < 2) {
                continue;
            }
            std::vector<Tile>& surface_tiles = per_surface_tiles[surface_index];
            surface_tiles.reserve(
                ((surface.rows - 1 + new_tile_stride - 1) / new_tile_stride)
                * ((surface.cols - 1 + new_tile_stride - 1) / new_tile_stride));
            for (size_t row = 0; row + 1 < surface.rows; row += new_tile_stride) {
                for (size_t col = 0; col + 1 < surface.cols; col += new_tile_stride) {
                    const size_t row_end = std::min(
                        surface.rows - 1, row + static_cast<size_t>(new_tile_stride));
                    const size_t col_end = std::min(
                        surface.cols - 1, col + static_cast<size_t>(new_tile_stride));
                    Aabb bounds;
                    bool any_valid = false;
                    for (size_t r = row; r <= row_end; ++r) {
                        for (size_t c = col; c <= col_end; ++c) {
                            const Vec3 point = surface.at(r, c);
                            if (surface.valid_point(r, c)) {
                                bounds.extend(point);
                                any_valid = true;
                            }
                        }
                    }
                    if (any_valid) {
                        bounds.pad(bbox_padding);
                        surface_tiles.push_back({
                            bounds,
                            static_cast<int32_t>(surface_index),
                            static_cast<int>(row),
                            static_cast<int>(col),
                        });
                    }
                }
            }
        }
        size_t tile_count = 0;
        for (const auto& tiles : per_surface_tiles) tile_count += tiles.size();
        std::vector<Tile> new_tiles;
        std::vector<size_t> new_surface_tile_offsets(new_surfaces.size() + 1, 0);
        size_t merged_tile_count = 0;
        new_tiles.clear();
        new_tiles.reserve(tile_count);
        for (size_t surface_index = 0;
             surface_index < per_surface_tiles.size(); ++surface_index) {
            new_surface_tile_offsets[surface_index] = merged_tile_count;
            auto& surface_tiles = per_surface_tiles[surface_index];
            merged_tile_count += surface_tiles.size();
            new_tiles.insert(new_tiles.end(),
                std::make_move_iterator(surface_tiles.begin()),
                std::make_move_iterator(surface_tiles.end()));
        }
        new_surface_tile_offsets.back() = merged_tile_count;

        std::vector<TreeEntry> entries;
        entries.reserve(new_tiles.size());
        for (uint32_t tile_index = 0;
             tile_index < static_cast<uint32_t>(new_tiles.size()); ++tile_index) {
            entries.emplace_back(tree_box(new_tiles[tile_index].bounds), tile_index);
        }
        std::unique_ptr<TileTree> new_tree;
        if (!new_tiles.empty()) {
            // The range constructor bulk-packs the tree. This is the same
            // construction path used by Volume Cartographer and avoids the
            // recursive O(N log^2 N) stable sorts of the previous BVH.
            new_tree = std::make_unique<TileTree>(entries.begin(), entries.end());
        }

        std::vector<Aabb> new_surface_bounds(new_surfaces.size());
        std::vector<TreeEntry> surface_entries;
        surface_entries.reserve(new_surfaces.size());
        for (uint32_t surface_index = 0;
             surface_index < static_cast<uint32_t>(new_surfaces.size()); ++surface_index) {
            const size_t first = new_surface_tile_offsets[surface_index];
            const size_t last = new_surface_tile_offsets[surface_index + 1];
            if (first == last) continue;
            Aabb bounds = new_surfaces[surface_index]->has_metadata_bounds
                ? new_surfaces[surface_index]->metadata_bounds : Aabb{};
            if (!new_surfaces[surface_index]->has_metadata_bounds) {
                for (size_t tile_index = first; tile_index < last; ++tile_index) {
                    bounds.extend(new_tiles[tile_index].bounds);
                }
            }
            new_surface_bounds[surface_index] = bounds;
            surface_entries.emplace_back(tree_box(bounds), surface_index);
        }
        std::unique_ptr<TileTree> new_surface_tree;
        if (!surface_entries.empty()) {
            new_surface_tree = std::make_unique<TileTree>(
                surface_entries.begin(), surface_entries.end());
        }

        surfaces_ = std::move(new_surfaces);
        tiles_ = std::move(new_tiles);
        surface_tile_offsets_ = std::move(new_surface_tile_offsets);
        surface_bounds_ = std::move(new_surface_bounds);
        tree_ = std::move(new_tree);
        surface_tree_ = std::move(new_surface_tree);
        sampling_stride_ = sampling_stride;
        tile_stride_ = new_tile_stride;
    }

    const std::vector<std::shared_ptr<SurfaceData>>& surfaces() const
    {
        return surfaces_;
    }

    size_t tile_count() const noexcept { return tiles_.size(); }

    // Cheap first broad phase: one packed-tree lookup per surface rather than
    // one lookup per tile. Tile-level intersections are deferred until a pair
    // actually needs dense rescue sampling.
    void query_surface_candidates(
        int32_t seed_surface,
        float tolerance,
        int32_t first_surface,
        std::vector<int32_t>& out) const
    {
        out.clear();
        if (!surface_tree_ || tolerance < 0.0f || seed_surface < 0
            || static_cast<size_t>(seed_surface) >= surface_bounds_.size()
            || static_cast<size_t>(seed_surface + 1) >= surface_tile_offsets_.size()
            || surface_tile_offsets_[static_cast<size_t>(seed_surface)]
                == surface_tile_offsets_[static_cast<size_t>(seed_surface) + 1]) {
            return;
        }
        Aabb padded = surface_bounds_[static_cast<size_t>(seed_surface)];
        padded.pad(tolerance);
        surface_tree_->query(
            bgi::intersects(tree_box(padded)),
            boost::make_function_output_iterator([&](const TreeEntry& entry) {
                const int32_t candidate = static_cast<int32_t>(entry.second);
                if (candidate >= first_surface && candidate != seed_surface) {
                    out.push_back(candidate);
                }
            }));
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    // After metadata bboxes nominate a surface pair, compare only those two
    // surfaces' tile arrays directly. For the small candidate degree typical
    // here this is cheaper than millions of queries against the global tile
    // tree and produces the exact tile-pair work list needed by the merger.
    void query_surface_pair_tiles(
        int32_t seed_surface,
        int32_t target_surface,
        float tolerance,
        std::vector<SurfaceTileCandidate>& out) const
    {
        out.clear();
        if (tolerance < 0.0f || seed_surface < 0 || target_surface < 0
            || static_cast<size_t>(seed_surface + 1) >= surface_tile_offsets_.size()
            || static_cast<size_t>(target_surface + 1) >= surface_tile_offsets_.size()) {
            return;
        }
        const size_t seed_first = surface_tile_offsets_[static_cast<size_t>(seed_surface)];
        const size_t seed_last = surface_tile_offsets_[static_cast<size_t>(seed_surface) + 1];
        const size_t target_first = surface_tile_offsets_[static_cast<size_t>(target_surface)];
        const size_t target_last = surface_tile_offsets_[static_cast<size_t>(target_surface) + 1];
        for (size_t seed_index = seed_first; seed_index < seed_last; ++seed_index) {
            const Tile& seed_tile = tiles_[seed_index];
            Aabb padded = seed_tile.bounds;
            padded.pad(tolerance);
            for (size_t target_index = target_first;
                 target_index < target_last; ++target_index) {
                const Tile& target_tile = tiles_[target_index];
                if (!padded.intersects(target_tile.bounds)) continue;
                out.push_back({
                    target_surface,
                    seed_tile.row,
                    seed_tile.col,
                    target_tile.row,
                    target_tile.col,
                    static_cast<uint32_t>(seed_index),
                    static_cast<uint32_t>(target_index),
                });
            }
        }
    }

    // Append the closest hit for every surface within ``tolerance`` of
    // ``point`` to ``out``, in ascending surface-index order. ``included``
    // (parallel to surfaces) optionally restricts the surface set. The
    // scratch avoids per-call allocations on hot paths.
    void query_point(
        const Vec3& point,
        float tolerance,
        std::vector<SurfaceHit>& out,
        QueryScratch& scratch,
        const std::vector<uint8_t>* included = nullptr,
        int32_t first_surface = 0,
        int32_t only_surface = -1,
        bool exact_bilinear = true) const
    {
        if (tolerance < 0.0f || !finite(point)) {
            return;
        }
        const float tolerance_squared = tolerance * tolerance;
        // Seeding each tile evaluation with the (just above tolerance)
        // bound prunes quads that could never produce a reported hit; a hit
        // at exactly the tolerance still records, so results are unchanged.
        const float tile_bound = std::nextafter(
            tolerance_squared, std::numeric_limits<float>::infinity());

        scratch.tiles.clear();
        find_candidate_tiles(point, tolerance, scratch.tiles);
        // Keep per-point storage proportional to BVH candidates while
        // retaining the API's ascending surface-index order.
        scratch.surfaces.clear();
        scratch.surfaces.reserve(scratch.tiles.size());
        for (const uint32_t tile_index : scratch.tiles) {
            const int32_t surface_index = tiles_[tile_index].surface;
            if (surface_index >= first_surface
                && (only_surface < 0 || surface_index == only_surface)
                && (included == nullptr
                    || (*included)[static_cast<size_t>(surface_index)])) {
                scratch.surfaces.push_back(surface_index);
            }
        }
        std::sort(scratch.surfaces.begin(), scratch.surfaces.end());
        scratch.surfaces.erase(
            std::unique(scratch.surfaces.begin(), scratch.surfaces.end()),
            scratch.surfaces.end());

        scratch.best_hits.assign(scratch.surfaces.size(), BestHit{});
        for (const uint32_t tile_index : scratch.tiles) {
            const Tile& tile = tiles_[tile_index];
            const size_t surface_index = static_cast<size_t>(tile.surface);
            if (tile.surface < first_surface
                || (only_surface >= 0 && tile.surface != only_surface)
                || (included != nullptr && !(*included)[surface_index])) {
                continue;
            }
            const auto surface = std::lower_bound(
                scratch.surfaces.begin(), scratch.surfaces.end(), tile.surface);
            BestHit& best = scratch.best_hits[static_cast<size_t>(
                surface - scratch.surfaces.begin())];
            // Also bound by this surface's current best: a farther hit
            // would be discarded by the comparison below anyway.
            const BestHit tile_hit = evaluate_tile(
                tile, point, std::min(tile_bound, best.distance_squared),
                exact_bilinear);
            if (tile_hit.valid && tile_hit.distance_squared < best.distance_squared) {
                best = tile_hit;
                best.tile = tile_index;
            }
        }

        for (size_t candidate_index = 0;
             candidate_index < scratch.surfaces.size();
             ++candidate_index) {
            const BestHit& hit = scratch.best_hits[candidate_index];
            if (!hit.valid || hit.distance_squared > tolerance_squared) {
                continue;
            }
            out.push_back({
                scratch.surfaces[candidate_index],
                std::sqrt(hit.distance_squared),
                hit.i,
                hit.j,
                hit.tile,
            });
        }
    }

    // Return conservative broad-phase candidates for one seed surface. Each
    // result identifies a seed tile whose tolerance-padded bounds intersect
    // at least one indexed tile on the candidate surface. Results are sorted
    // and deduplicated so callers remain deterministic across tree layouts.
    void query_surface_tile_candidates(
        int32_t seed_surface,
        float tolerance,
        int32_t first_surface,
        std::vector<SurfaceTileCandidate>& out,
        QueryScratch& scratch,
        int32_t only_surface = -1,
        const std::vector<uint8_t>* included = nullptr) const
    {
        out.clear();
        if (!tree_ || tolerance < 0.0f || seed_surface < 0
            || static_cast<size_t>(seed_surface + 1) >= surface_tile_offsets_.size()) {
            return;
        }
        const size_t first = surface_tile_offsets_[static_cast<size_t>(seed_surface)];
        const size_t last = surface_tile_offsets_[static_cast<size_t>(seed_surface) + 1];
        for (size_t seed_tile_index = first; seed_tile_index < last; ++seed_tile_index) {
            const Tile& seed_tile = tiles_[seed_tile_index];
            Aabb padded = seed_tile.bounds;
            padded.pad(tolerance);
            scratch.tiles.clear();
            tree_->query(
                bgi::intersects(tree_box(padded)),
                boost::make_function_output_iterator(
                    [&](const TreeEntry& entry) { scratch.tiles.push_back(entry.second); }));
            for (const uint32_t candidate_tile_index : scratch.tiles) {
                const int32_t candidate_surface = tiles_[candidate_tile_index].surface;
                if (candidate_surface < first_surface
                    || (only_surface >= 0 && candidate_surface != only_surface)
                    || (included != nullptr
                        && !(*included)[static_cast<size_t>(candidate_surface)])
                    || candidate_surface == seed_surface) {
                    continue;
                }
                const Tile& candidate_tile = tiles_[candidate_tile_index];
                out.push_back({
                    candidate_surface,
                    seed_tile.row,
                    seed_tile.col,
                    candidate_tile.row,
                    candidate_tile.col,
                    static_cast<uint32_t>(seed_tile_index),
                    candidate_tile_index,
                });
            }
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
            if (a.surface != b.surface) return a.surface < b.surface;
            if (a.seed_row != b.seed_row) return a.seed_row < b.seed_row;
            if (a.seed_col != b.seed_col) return a.seed_col < b.seed_col;
            if (a.target_row != b.target_row) return a.target_row < b.target_row;
            return a.target_col < b.target_col;
        });
        out.erase(std::unique(out.begin(), out.end(), [](const auto& a, const auto& b) {
            return a.surface == b.surface && a.seed_row == b.seed_row
                && a.seed_col == b.seed_col && a.target_row == b.target_row
                && a.target_col == b.target_col;
        }), out.end());
    }

    // Narrow-phase one point against one known target tile, bypassing another
    // global tree lookup. This is the hot path for merger harvesting after the
    // tile-pair broad phase has already established spatial locality.
    bool query_tile_point(
        const SurfaceTileCandidate& candidate,
        const Vec3& point,
        float tolerance,
        bool exact_bilinear,
        SurfaceHit& out) const
    {
        if (tolerance < 0.0f || !finite(point)
            || candidate.target_tile >= tiles_.size()) {
            return false;
        }
        const Tile& target = tiles_[candidate.target_tile];
        if (target.surface != candidate.surface) return false;
        Aabb query;
        query.low = {point.x - tolerance, point.y - tolerance, point.z - tolerance};
        query.high = {point.x + tolerance, point.y + tolerance, point.z + tolerance};
        if (!target.bounds.intersects(query)) return false;
        const float tolerance_squared = tolerance * tolerance;
        const BestHit hit = evaluate_tile(
            target, point,
            std::nextafter(tolerance_squared, std::numeric_limits<float>::infinity()),
            exact_bilinear);
        if (!hit.valid || hit.distance_squared > tolerance_squared) return false;
        out = {
            target.surface,
            std::sqrt(hit.distance_squared),
            hit.i,
            hit.j,
            candidate.target_tile,
        };
        return true;
    }

    int tile_stride() const noexcept { return tile_stride_; }

private:
    static int compute_tile_stride(int sampling_stride)
    {
        if (sampling_stride >= 8) {
            return sampling_stride;
        }
        return ((8 + sampling_stride - 1) / sampling_stride) * sampling_stride;
    }

    void find_candidate_tiles(
        const Vec3& point,
        float tolerance,
        std::vector<uint32_t>& out) const
    {
        if (!tree_) {
            return;
        }
        Aabb query;
        query.low = {point.x - tolerance, point.y - tolerance, point.z - tolerance};
        query.high = {point.x + tolerance, point.y + tolerance, point.z + tolerance};
        tree_->query(
            bgi::intersects(tree_box(query)),
            boost::make_function_output_iterator(
                [&](const TreeEntry& entry) { out.push_back(entry.second); }));
        std::sort(out.begin(), out.end());
    }

    // Closest point on the tile's quads, reported only when strictly below
    // ``bound`` (squared). The bound both seeds the running minimum and
    // enables the per-quad box rejection below; passing +inf reproduces the
    // original unbounded search.
    BestHit evaluate_tile(
        const Tile& tile,
        const Vec3& point,
        float bound = std::numeric_limits<float>::infinity(),
        bool exact_bilinear = true) const
    {
        BestHit best;
        best.distance_squared = bound;
        const SurfaceData& surface = *surfaces_[static_cast<size_t>(tile.surface)];
        const int row_limit = std::min(
            tile.row + tile_stride_, static_cast<int>(surface.rows) - 1);
        const int col_limit = std::min(
            tile.col + tile_stride_, static_cast<int>(surface.cols) - 1);

        for (int row = tile.row; row < row_limit; row += sampling_stride_) {
            for (int col = tile.col; col < col_limit; col += sampling_stride_) {
                const int row_step = std::min(
                    sampling_stride_, static_cast<int>(surface.rows) - 1 - row);
                const int col_step = std::min(
                    sampling_stride_, static_cast<int>(surface.cols) - 1 - col);
                if (row_step <= 0 || col_step <= 0) {
                    continue;
                }

                const Vec3 p00 = surface.at(row, col);
                const Vec3 p10 = surface.at(row, col + col_step);
                const Vec3 p01 = surface.at(row + row_step, col);
                const Vec3 p11 = surface.at(row + row_step, col + col_step);
                if (!surface.valid_point(row, col)
                    || !surface.valid_point(row, col + col_step)
                    || !surface.valid_point(row + row_step, col)
                    || !surface.valid_point(row + row_step, col + col_step)) {
                    continue;
                }
                if (sampling_stride_ == 1
                    && !surface.valid_quad(static_cast<size_t>(row),
                                           static_cast<size_t>(col))) {
                    continue;
                }

                // A bilinear quad is a convex combination of its four
                // corners, so its corner AABB is also an exact lower bound.
                auto axis_gap = [](float value, float a, float b, float c, float d) {
                    const float low = std::min(std::min(a, b), std::min(c, d));
                    const float high = std::max(std::max(a, b), std::max(c, d));
                    return value < low ? low - value : (value > high ? value - high : 0.0f);
                };
                const float gx = axis_gap(point.x, p00.x, p10.x, p01.x, p11.x);
                const float gy = axis_gap(point.y, p00.y, p10.y, p01.y, p11.y);
                const float gz = axis_gap(point.z, p00.z, p10.z, p01.z, p11.z);
                if (gx * gx + gy * gy + gz * gz >= best.distance_squared) {
                    continue;
                }

                if (exact_bilinear) {
                    const BilinearHit hit = closest_point_on_bilinear_quad(
                        point, p00, p10, p01, p11);
                    if (hit.valid && hit.distance_squared < best.distance_squared) {
                        best.valid = true;
                        best.distance_squared = static_cast<float>(hit.distance_squared);
                        best.i = static_cast<float>(static_cast<double>(col)
                            + hit.u * static_cast<double>(col_step));
                        best.j = static_cast<float>(static_cast<double>(row)
                            + hit.v * static_cast<double>(row_step));
                    }
                } else {
                    const auto record_triangle = [&](double u, double v,
                                                     float distance_squared) {
                        if (distance_squared >= best.distance_squared) return;
                        best.valid = true;
                        best.distance_squared = distance_squared;
                        best.i = static_cast<float>(static_cast<double>(col)
                            + u * static_cast<double>(col_step));
                        best.j = static_cast<float>(static_cast<double>(row)
                            + v * static_cast<double>(row_step));
                    };
                    const TriangleHit first = closest_point_on_triangle(
                        point, p00, p10, p01);
                    record_triangle(first.bary.y, first.bary.z,
                                    first.distance_squared);
                    const TriangleHit second = closest_point_on_triangle(
                        point, p10, p11, p01);
                    record_triangle(second.bary.x + second.bary.y,
                                    second.bary.y + second.bary.z,
                                    second.distance_squared);
                }
            }
        }
        return best;
    }

    std::vector<std::shared_ptr<SurfaceData>> surfaces_;
    std::vector<Tile> tiles_;
    std::vector<size_t> surface_tile_offsets_;
    std::vector<Aabb> surface_bounds_;
    std::unique_ptr<TileTree> tree_;
    std::unique_ptr<TileTree> surface_tree_;
    int sampling_stride_ = 1;
    int tile_stride_ = 8;
};

}  // namespace surfcore
