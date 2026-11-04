#pragma once
#include <array>
#include <type_traits>
namespace vc::objects::detail {
// Closest point of triangle abc to p (Ericson, Real-Time Collision Detection
// 5.1.5) with barycentrics. Generic over the vector type so each caller keeps
// its own scalar arithmetic: relationship discovery runs it on cv::Vec3f,
// store proximity queries on doubles.
template<class V> struct TriangleHit {
    using Scalar = std::remove_cvref_t<decltype(std::declval<V>()[0])>;
    V closest;
    std::array<Scalar, 3> bary;
    Scalar distSq;
};
template<class V> auto dot(const V& a, const V& b) { return a.dot(b); }
template<class V> TriangleHit<V> closestPointOnTriangle(const V& p, const V& a, const V& b, const V& c)
{
    TriangleHit<V> hit;
    using S = typename TriangleHit<V>::Scalar;

    const V ab = b - a;
    const V ac = c - a;
    const V ap = p - a;

    S d1 = dot(ab, ap);
    S d2 = dot(ac, ap);
    if (d1 <= S(0) && d2 <= S(0)) {
        hit.closest = a;
        hit.bary = {S(1), S(0), S(0)};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    const V bp = p - b;
    S d3 = dot(ab, bp);
    S d4 = dot(ac, bp);
    if (d3 >= S(0) && d4 <= d3) {
        hit.closest = b;
        hit.bary = {S(0), S(1), S(0)};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    S vc = d1 * d4 - d3 * d2;
    if (vc <= S(0) && d1 >= S(0) && d3 <= S(0)) {
        S v = d1 / (d1 - d3);
        hit.closest = a + v * ab;
        hit.bary = {S(1) - v, v, S(0)};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    const V cp = p - c;
    S d5 = dot(ab, cp);
    S d6 = dot(ac, cp);
    if (d6 >= S(0) && d5 <= d6) {
        hit.closest = c;
        hit.bary = {S(0), S(0), S(1)};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    S vb = d5 * d2 - d1 * d6;
    if (vb <= S(0) && d2 >= S(0) && d6 <= S(0)) {
        S w = d2 / (d2 - d6);
        hit.closest = a + w * ac;
        hit.bary = {S(1) - w, S(0), w};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    S va = d3 * d6 - d5 * d4;
    if (va <= S(0) && (d4 - d3) >= S(0) && (d5 - d6) >= S(0)) {
        S w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.closest = b + w * (c - b);
        hit.bary = {S(0), S(1) - w, w};
        const V d = p - hit.closest;
        hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        return hit;
    }

    S denom = S(1) / (va + vb + vc);
    S v = vb * denom;
    S w = vc * denom;
    S u = S(1) - v - w;
    hit.closest = a + ab * v + ac * w;
    hit.bary = {u, v, w};
    const V d = p - hit.closest;
    hit.distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    return hit;
}
} // namespace vc::objects::detail
