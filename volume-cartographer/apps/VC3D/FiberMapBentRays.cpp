#include "FiberMapBentRays.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace vc3d::fiber_map::bent
{

namespace
{

constexpr double kEps = 1e-12;
constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

cv::Vec3d unit(const cv::Vec3d& v)
{
    const double l = length(v);
    return l > kEps ? v * (1.0 / l) : cv::Vec3d(0.0, 0.0, 0.0);
}

// Shortest signed angular step from a to b.
double wrappedDelta(double a, double b)
{
    double d = b - a;
    while (d > kPi) {
        d -= 2.0 * kPi;
    }
    while (d < -kPi) {
        d += 2.0 * kPi;
    }
    return d;
}

bool lexLess(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
}

// Segment p0->p1 against triangle (a, b, c): parameter and barycentric
// weights of b and c at the hit (Moller-Trumbore), inclusive edges;
// `coplanar` when the segment lies in the triangle's plane.
struct TriangleHit {
    bool hit = false;
    bool coplanar = false;
    double t = 0.0;
    double u = 0.0;
    double v = 0.0;
};

TriangleHit segmentTriangle(const cv::Vec3d& p0, const cv::Vec3d& p1,
                            const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c)
{
    TriangleHit out;
    const cv::Vec3d d = p1 - p0;
    const cv::Vec3d e1 = b - a;
    const cv::Vec3d e2 = c - a;
    const cv::Vec3d n = e1.cross(e2);
    const double nLen = length(n);
    if (nLen < kEps) {
        return out;
    }
    const cv::Vec3d h = d.cross(e2);
    const double det = e1.dot(h);
    const double scale = length(d) * nLen;
    if (std::abs(det) <= 1e-9 * scale) {
        const cv::Vec3d nu = n * (1.0 / nLen);
        const double d0 = std::abs(nu.dot(p0 - a));
        const double d1 = std::abs(nu.dot(p1 - a));
        const double planeEps = 1e-7 * std::max({length(e1), length(e2), length(d), 1.0});
        out.coplanar = d0 <= planeEps && d1 <= planeEps;
        return out;
    }
    const double f = 1.0 / det;
    const cv::Vec3d s = p0 - a;
    const double u = f * s.dot(h);
    if (u < -1e-9 || u > 1.0 + 1e-9) {
        return out;
    }
    const cv::Vec3d q = s.cross(e1);
    const double v = f * d.dot(q);
    if (v < -1e-9 || u + v > 1.0 + 1e-9) {
        return out;
    }
    const double t = f * e2.dot(q);
    if (t < -1e-9 || t > 1.0 + 1e-9) {
        return out;
    }
    out.hit = true;
    out.t = std::clamp(t, 0.0, 1.0);
    out.u = std::clamp(u, 0.0, 1.0);
    out.v = std::clamp(v, 0.0, 1.0 - out.u);
    return out;
}

// Barycentric coordinates (weights of a, b, c) of a point in the plane of
// the triangle, by projection onto its plane.
// The point is first projected ORTHOGONALLY onto the triangle's plane (a
// probe off the surface must be judged by its foot, not by where it lands
// along a dropped axis), then the weights are computed in the dominant
// projection (the axis of the normal's largest component dropped): the
// Gram-determinant form cancels on slivers that the face-validity rule
// (the doubled area against the edge scale) still accepts, and every
// accepted face must blend.
cv::Vec3d barycentric(const cv::Vec3d& p, const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c)
{
    const cv::Vec3d v0 = b - a;
    const cv::Vec3d v1 = c - a;
    const cv::Vec3d n = v0.cross(v1);
    const double area2 = length(n);
    const double scale = std::max({length(v0), length(v1), length(c - b)});
    if (!(area2 > 1e-12 * std::max(1.0, scale * scale)) || !std::isfinite(area2)) {
        return cv::Vec3d(kNaN, kNaN, kNaN);
    }
    const cv::Vec3d unitNormal = n * (1.0 / area2);
    const cv::Vec3d v2 = (p - a) - unitNormal * unitNormal.dot(p - a);
    int k = 0;
    for (int axis = 1; axis < 3; ++axis) {
        if (std::abs(n[axis]) > std::abs(n[k])) {
            k = axis;
        }
    }
    const int i = (k + 1) % 3;
    const int j = (k + 2) % 3;
    const double denom = v0[i] * v1[j] - v0[j] * v1[i];
    const double v = (v2[i] * v1[j] - v2[j] * v1[i]) / denom;
    const double w = (v0[i] * v2[j] - v0[j] * v2[i]) / denom;
    return cv::Vec3d(1.0 - v - w, v, w);
}

struct Box {
    cv::Vec3d lo{kInf, kInf, kInf};
    cv::Vec3d hi{-kInf, -kInf, -kInf};
    void add(const cv::Vec3d& p)
    {
        for (int i = 0; i < 3; ++i) {
            lo[i] = std::min(lo[i], p[i]);
            hi[i] = std::max(hi[i], p[i]);
        }
    }
    [[nodiscard]] bool overlaps(const Box& other) const
    {
        for (int i = 0; i < 3; ++i) {
            if (hi[i] < other.lo[i] || lo[i] > other.hi[i]) {
                return false;
            }
        }
        return true;
    }
};

} // namespace

void SheetNormalField::axes(const std::vector<cv::Vec3d>& points,
                            std::vector<std::optional<cv::Vec3d>>& out) const
{
    out.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        out[i] = axis(points[i]);
    }
}

ConditioningProfile conditioningProfile(const SheetNormalField& field,
                                        const std::vector<cv::Vec3d>& points,
                                        const UmbilicusFrame& frame)
{
    ConditioningProfile profile;
    profile.value.assign(points.size(), kNaN);
    profile.axis.assign(points.size(), std::nullopt);
    std::vector<std::optional<cv::Vec3d>> axes;
    field.axes(points, axes);
    for (std::size_t i = 0; i < points.size() && i < axes.size(); ++i) {
        if (!axes[i] || length(*axes[i]) <= kEps) {
            continue;
        }
        const cv::Vec3d n = unit(*axes[i]);
        profile.axis[i] = n;
        profile.value[i] = std::abs(n.dot(frame.radialUnit(points[i])));
    }
    return profile;
}

std::vector<unsigned char> illConditionedSamples(const ConditioningProfile& profile, double gate)
{
    std::vector<unsigned char> out(profile.value.size(), 0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = profile.value[i] < gate ? 1 : 0;
    }
    return out;
}

std::vector<std::pair<std::size_t, std::size_t>> illConditionedRuns(
    const std::vector<unsigned char>& illConditioned)
{
    std::vector<std::pair<std::size_t, std::size_t>> runs;
    std::size_t i = 0;
    while (i < illConditioned.size()) {
        if (!illConditioned[i]) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j + 1 < illConditioned.size() && illConditioned[j + 1]) {
            ++j;
        }
        runs.emplace_back(i, j);
        i = j + 1;
    }
    return runs;
}

bool canonicalForward(const std::vector<cv::Vec3d>& points, std::size_t first, std::size_t last)
{
    // From the lexicographically smaller end. Equal end points: the first
    // pair of samples that differ, from the two ends inward, decides.
    for (std::size_t k = 0; first + k < last - k; ++k) {
        const cv::Vec3d& a = points[first + k];
        const cv::Vec3d& b = points[last - k];
        if (lexLess(a, b)) {
            return true;
        }
        if (lexLess(b, a)) {
            return false;
        }
    }
    return true;
}

std::vector<std::size_t> symmetricStarts(const std::vector<cv::Vec3d>& points,
                                         std::size_t first, std::size_t last, double strideVx)
{
    std::vector<std::size_t> out;
    if (points.empty() || first > last || last >= points.size()) {
        return out;
    }
    out.push_back(first);
    out.push_back(last);
    if (strideVx > 0.0 && last > first) {
        // Canonical traversal: from the lexicographically smaller end, so
        // the arithmetic (and its rounding) is the same whichever way the
        // samples are stored.
        const bool forward = canonicalForward(points, first, last);
        const std::size_t count = last - first + 1;
        const auto at = [&](std::size_t k) { return forward ? first + k : last - k; };
        double arclength = 0.0;
        double bucket = 0.0;
        for (std::size_t k = 1; k + 1 < count; ++k) {
            arclength += length(points[at(k)] - points[at(k - 1)]);
            const double next = std::floor(arclength / strideVx);
            if (next > bucket) {
                out.push_back(at(k));
                bucket = next;
            }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

FiberOrientation orientFiber(const std::vector<cv::Vec3d>& points,
                             const ConditioningProfile& profile,
                             const UmbilicusFrame& frame,
                             const BentRayParams& params)
{
    FiberOrientation out;
    const std::size_t n = points.size();
    out.normal.assign(n, std::nullopt);
    if (n == 0 || profile.value.size() != n || profile.axis.size() != n) {
        return out;
    }
    // Transport by continuity within every stretch where the field has a
    // value, then anchor the stretch as a whole.
    std::vector<cv::Vec3d> transported(n, cv::Vec3d(0.0, 0.0, 0.0));
    std::size_t i = 0;
    while (i < n) {
        if (!profile.axis[i]) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j + 1 < n && profile.axis[j + 1]) {
            ++j;
        }
        // Transport from the stretch's canonical end (the lexicographically
        // smaller end point, as symmetricStarts), so the basis of a stretch
        // the vote leaves unoriented is the same whichever way the samples
        // are stored.
        const bool forward = canonicalForward(points, i, j);
        const auto at = [&](std::size_t k) { return forward ? i + k : j - k; };
        cv::Vec3d current = *profile.axis[at(0)];
        transported[at(0)] = current;
        for (std::size_t k = 1; k <= j - i; ++k) {
            cv::Vec3d m = *profile.axis[at(k)];
            if (m.dot(current) < 0.0) {
                m = -m;
            }
            current = m;
            transported[at(k)] = current;
        }
        // The vote, sample by sample in the same canonical traversal: each
        // well-conditioned sample says whether the transported axis points
        // along +e_r (+1) or against it (-1). The transported basis is one
        // sign away from the sheet's outward normal over any stretch where
        // the transport stayed consistent, so a CHANGE of that sign between
        // two voters means the basis flipped between them (or the sheet
        // returned on itself): one sign cannot serve both sides. The stretch
        // is therefore cut into sections of one vote sign, each anchored by
        // its own unanimous vote, and the voteless samples between two
        // sections of different signs form a contested section of their own
        // (nothing anchors them; the assembly withholds their readings
        // unless a witness lies on them). Voteless samples between two
        // sections of one sign, and at either end, join that section.
        struct Voter {
            std::size_t sample;
            int sign;
            double weight;
        };
        std::vector<Voter> voters;
        for (std::size_t kk = 0; kk <= j - i; ++kk) {
            const std::size_t k = at(kk);
            const double value = profile.value[k];
            if (!(value >= params.conditioningGate)) {
                continue;
            }
            const double along = transported[k].dot(frame.radialUnit(points[k]));
            if (along > 0.0) {
                voters.push_back({k, 1, value});
            } else if (along < 0.0) {
                voters.push_back({k, -1, value});
            }
        }
        // Sections are delimited in storage order (sample bounds); their
        // weights are summed in the canonical traversal, so the sums are
        // the same whichever way the samples are stored.
        std::vector<Voter> canonicalVoters = voters;
        std::sort(voters.begin(), voters.end(),
                  [](const Voter& a, const Voter& b) { return a.sample < b.sample; });
        // The section (by its first voter's sample) each voter belongs to.
        std::vector<std::size_t> sectionOfSample(n, 0);
        {
            std::size_t sectionStart = 0;
            for (std::size_t v = 0; v < voters.size(); ++v) {
                if (v == 0 || voters[v].sign != voters[v - 1].sign) {
                    sectionStart = voters[v].sample;
                }
                sectionOfSample[voters[v].sample] = sectionStart;
            }
        }
        std::vector<double> weightOfSection(n, 0.0);
        for (const Voter& voter : canonicalVoters) {
            weightOfSection[sectionOfSample[voter.sample]] += voter.weight;
        }
        const auto emitSection = [&](std::size_t first, std::size_t last, RunStatus status, int sign,
                              double agree, double disagree) {
            OrientedStretch section;
            section.firstSample = first;
            section.lastSample = last;
            section.status = status;
            section.provisionalSign = sign;
            section.agreeWeight = agree;
            section.disagreeWeight = disagree;
            for (std::size_t k = first; k <= last; ++k) {
                out.normal[k] = transported[k] * static_cast<double>(sign);
            }
            out.stretches.push_back(section);
        };
        if (voters.empty()) {
            emitSection(i, j, RunStatus::Unsupported, 1, 0.0, 0.0);
        } else {
            // Sections of one sign over the voters; the samples between the
            // last voter of a section and the first of the next are the
            // section's when the signs agree, contested otherwise.
            std::size_t sectionFirst = i;
            std::size_t v = 0;
            while (v < voters.size()) {
                const int sign = voters[v].sign;
                const double weight = weightOfSection[voters[v].sample];
                std::size_t lastVoter = voters[v].sample;
                while (v < voters.size() && voters[v].sign == sign) {
                    lastVoter = voters[v].sample;
                    ++v;
                }
                if (v == voters.size()) {
                    // The last section runs to the stretch's end.
                    emitSection(sectionFirst, j, RunStatus::Oriented, sign,
                         sign > 0 ? weight : 0.0, sign > 0 ? 0.0 : weight);
                    break;
                }
                // The next voter disagrees: the section ends at its last
                // voter, the gap to the next voter is contested.
                emitSection(sectionFirst, lastVoter, RunStatus::Oriented, sign,
                     sign > 0 ? weight : 0.0, sign > 0 ? 0.0 : weight);
                const std::size_t nextVoter = voters[v].sample;
                if (nextVoter > lastVoter + 1) {
                    emitSection(lastVoter + 1, nextVoter - 1, RunStatus::Unoriented, 1, 0.0, 0.0);
                }
                sectionFirst = nextVoter;
            }
        }
        i = j + 1;
    }
    for (const auto& [first, last] : illConditionedRuns(illConditionedSamples(profile, params.conditioningGate))) {
        OrientedRun run;
        run.firstSample = first;
        run.lastSample = last;
        run.status = RunStatus::Unsupported;
        for (const OrientedStretch& stretch : out.stretches) {
            if (first >= stretch.firstSample && last <= stretch.lastSample) {
                run.status = stretch.status;
                break;
            }
        }
        out.runs.push_back(run);
    }
    return out;
}

BentCurtain traceCurtain(const SheetNormalField& field,
                         const std::vector<cv::Vec3d>& points,
                         const FiberOrientation& orientation,
                         const UmbilicusFrame& frame,
                         const BentRayParams& params,
                         double minRadiusVx)
{
    BentCurtain curtain;
    if (points.empty() || orientation.normal.size() != points.size()) {
        return curtain;
    }
    const double step = params.stepVx > 0.0 ? params.stepVx : 8.0;
    const auto trace = [&](std::size_t startSample, int side) {
        BentRay ray;
        ray.startSample = startSample;
        ray.side = side;
        const cv::Vec3d start = points[startSample];
        // A start inside the umbilicus cutoff is angularly ill-conditioned
        // for everything the solver does with it: no ray (a dead ray, which
        // breaks the curtain there).
        if (frame.radius(start) < minRadiusVx) {
            return ray;
        }
        cv::Vec3d ref = unit(*orientation.normal[startSample]) * static_cast<double>(side);
        if (length(ref) == 0.0) {
            return ray;
        }
        ray.points.push_back(start);
        ray.s.push_back(0.0);
        ray.theta.push_back(0.0);
        {
            const std::optional<cv::Vec3d> axisStart = field.axis(start);
            ray.conditioning.push_back(axisStart && length(*axisStart) > kEps
                                           ? std::abs(unit(*axisStart).dot(frame.radialUnit(start)))
                                           : 1.0);
        }
        cv::Vec3d cur = start;
        std::optional<cv::Vec3d> axisHere = field.axis(cur);
        double s = 0.0;
        double theta = 0.0;
        while (s + step <= params.maxLengthVx + 1e-9) {
            if (!axisHere || length(*axisHere) <= kEps) {
                break;
            }
            cv::Vec3d m = unit(*axisHere);
            if (m.dot(ref) < 0.0) {
                m = -m;
            }
            const cv::Vec3d next = cur + m * step;
            // The whole step stays outside the cutoff (a chord whose ends
            // clear the axis can dip inside between them).
            double stepMinRadius = frame.radius(next);
            if (frame.minRadiusAlong) {
                stepMinRadius = std::min(stepMinRadius, frame.minRadiusAlong(cur, next));
            } else {
                for (int k = 1; k < 16; ++k) {
                    stepMinRadius = std::min(stepMinRadius,
                                             frame.radius(cur + m * (step * static_cast<double>(k) / 16.0)));
                }
            }
            if (stepMinRadius < minRadiusVx) {
                break;
            }
            // The step is kept only when its end point is supported.
            const std::optional<cv::Vec3d> axisNext = field.axis(next);
            if (!axisNext || length(*axisNext) <= kEps) {
                break;
            }
            theta += wrappedDelta(frame.theta(cur), frame.theta(next));
            ref = m;
            cur = next;
            axisHere = axisNext;
            s += step;
            ray.points.push_back(cur);
            ray.s.push_back(s);
            ray.theta.push_back(theta);
            ray.conditioning.push_back(std::abs(unit(*axisNext).dot(frame.radialUnit(cur))));
        }
        return ray;
    };
    for (const OrientedRun& run : orientation.runs) {
        if (run.status == RunStatus::Unoriented) {
            ++curtain.unorientedRunCount;
        } else if (run.status == RunStatus::Unsupported) {
            ++curtain.unsupportedRunCount;
        }
        // Every run with a transported basis is traced, whatever its vote:
        // the assembly decides, from link witnesses, whether and with which
        // sign its readings constrain (see BentStretch::status).
        BentStretch stretch;
        stretch.status = run.status;
        stretch.provisionalSign = 1;
        for (const OrientedStretch& os : orientation.stretches) {
            if (run.firstSample >= os.firstSample && run.lastSample <= os.lastSample) {
                stretch.provisionalSign = os.provisionalSign;
            }
        }
        stretch.firstSample = run.firstSample;
        stretch.lastSample = run.lastSample;
        stretch.starts = symmetricStarts(points, run.firstSample, run.lastSample, params.spacingVx);
        for (const std::size_t start : stretch.starts) {
            if (!orientation.normal[start]) {
                continue;
            }
            stretch.rays.push_back(trace(start, 1));
            stretch.rays.push_back(trace(start, -1));
        }
        curtain.stretches.push_back(std::move(stretch));
    }
    return curtain;
}

std::vector<CurtainHit> intersectCurtain(const BentCurtain& curtain,
                                         const std::vector<cv::Vec3d>& polyline)
{
    std::vector<CurtainHit> hits;
    if (polyline.size() < 2) {
        return hits;
    }
    const std::size_t segmentCount = polyline.size() - 1;
    std::vector<Box> segmentBoxes(segmentCount);
    std::vector<unsigned char> degenerate(segmentCount, 0);
    Box polyBox;
    for (std::size_t j = 0; j < segmentCount; ++j) {
        segmentBoxes[j].add(polyline[j]);
        segmentBoxes[j].add(polyline[j + 1]);
        polyBox.add(polyline[j]);
        polyBox.add(polyline[j + 1]);
        degenerate[j] = length(polyline[j + 1] - polyline[j]) <= 1e-9 ? 1 : 0;
    }
    const auto prevDistinct = [&](std::size_t vertex) -> std::optional<std::size_t> {
        for (std::size_t k = vertex; k-- > 0;) {
            if (length(polyline[k] - polyline[vertex]) > 1e-9) {
                return k;
            }
        }
        return std::nullopt;
    };
    const auto nextDistinct = [&](std::size_t vertex) -> std::optional<std::size_t> {
        for (std::size_t k = vertex + 1; k < polyline.size(); ++k) {
            if (length(polyline[k] - polyline[vertex]) > 1e-9) {
                return k;
            }
        }
        return std::nullopt;
    };
    // A face of the mesh: its three points (ordered so that the normal is
    // consistent over the strip) and its unit normal. A face of zero area
    // (two rays from one point - a repeated owner sample seeds two
    // identical rays - or a dead row) is no surface: it is not crossed,
    // not incident, and does not bound the probes.
    struct Face {
        cv::Vec3d p[3];
        cv::Vec3d normal;
        bool valid = false;
    };
    const auto faceOf = [](const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c) {
        Face f;
        f.p[0] = a;
        f.p[1] = b;
        f.p[2] = c;
        const cv::Vec3d cross = (b - a).cross(c - a);
        const double area2 = length(cross);
        const double scale = std::max({length(b - a), length(c - a), length(c - b)});
        f.valid = area2 > 1e-12 * std::max(1.0, scale * scale) && std::isfinite(area2);
        f.normal = f.valid ? cross * (1.0 / area2) : cv::Vec3d(0.0, 0.0, 0.0);
        return f;
    };
    // Unsigned distance from a point to a triangle.
    const auto pointTriangleDistance = [](const cv::Vec3d& q, const Face& f) {
        const cv::Vec3d w = barycentric(q, f.p[0], f.p[1], f.p[2]);
        if (!std::isnan(w[0]) && w[0] >= 0.0 && w[1] >= 0.0 && w[2] >= 0.0) {
            return std::abs(f.normal.dot(q - f.p[0]));
        }
        double best = kInf;
        for (int e = 0; e < 3; ++e) {
            const cv::Vec3d& a = f.p[e];
            const cv::Vec3d& b = f.p[(e + 1) % 3];
            const cv::Vec3d d = b - a;
            const double l2 = d.dot(d);
            const double t = l2 > 0.0 ? std::clamp((q - a).dot(d) / l2, 0.0, 1.0) : 0.0;
            best = std::min(best, length(q - (a + d * t)));
        }
        return best;
    };
    // The side (+1/-1/0) of a point against the surface formed by the
    // incident faces: the sign at the face nearest to it.
    const auto sideOf = [&](const cv::Vec3d& q, const std::vector<Face>& faces) {
        double bestDistance = kInf;
        int side = 0;
        for (const Face& f : faces) {
            const double distance = pointTriangleDistance(q, f);
            if (distance < bestDistance) {
                bestDistance = distance;
                const double signedDistance = f.normal.dot(q - f.p[0]);
                side = signedDistance > 1e-12 ? 1 : (signedDistance < -1e-12 ? -1 : 0);
            }
        }
        return side;
    };
    struct Vertex {
        cv::Vec3d p;
        double s;
        double theta;
        double across;
        double along;
    };
    // The two triangles of quad k of the strip between rays A and B, in a
    // vertex order whose normals agree over the strip.
    const auto quadFaces = [&](const BentRay& A, const BentRay& B, std::size_t k, Face out[2]) {
        const cv::Vec3d& a0 = A.points[k];
        const cv::Vec3d& a1 = A.points[k + 1];
        const cv::Vec3d& b0 = B.points[k];
        const cv::Vec3d& b1 = B.points[k + 1];
        const double diagA = length(b1 - a0);
        const double diagB = length(b0 - a1);
        bool useA0B1 = diagA < diagB;
        if (diagA == diagB) {
            useA0B1 = !lexLess(0.5 * (a1 + b0), 0.5 * (a0 + b1));
        }
        if (useA0B1) {
            out[0] = faceOf(a0, a1, b1);
            out[1] = faceOf(a0, b1, b0);
        } else {
            out[0] = faceOf(a0, a1, b0);
            out[1] = faceOf(a1, b1, b0);
        }
        return useA0B1;
    };
    // The faces of quad k of the strip between rays A and B as stored,
    // computed on the rays in canonical order (the lexicographically smaller
    // start first) so the arithmetic - and its rounding - is the same
    // whichever way the owner is stored and whichever way it passes
    // through the same two points; the normals are then turned to the
    // stored orientation (consistent over the strip and its neighbours).
    // Returns the split of the canonical pair; `swapped` says the canonical
    // first ray is B.
    const auto canonicalFaces = [&](const BentRay& A, const BentRay& B, std::size_t k, Face out[2],
                                    bool& swapped) {
        swapped = lexLess(B.points.front(), A.points.front());
        const bool split = swapped ? quadFaces(B, A, k, out) : quadFaces(A, B, k, out);
        if (swapped) {
            out[0].normal = -out[0].normal;
            out[1].normal = -out[1].normal;
        }
        return split;
    };
    for (std::size_t st = 0; st < curtain.stretches.size(); ++st) {
        const BentStretch& stretch = curtain.stretches[st];
        for (const int side : {1, -1}) {
            std::vector<std::size_t> rays;
            for (std::size_t r = 0; r < stretch.rays.size(); ++r) {
                if (stretch.rays[r].side == side) {
                    rays.push_back(r);
                }
            }
            // The faces of quad k of strip q (empty when the strip or quad
            // does not exist), for the incident-surface test at creases.
            const auto facesAt = [&](std::size_t q, std::size_t k, std::vector<Face>& out) {
                if (q + 1 >= rays.size()) {
                    return;
                }
                const BentRay& A = stretch.rays[rays[q]];
                const BentRay& B = stretch.rays[rays[q + 1]];
                const std::size_t n = std::min(A.points.size(), B.points.size());
                if (n < 2 || k + 1 >= n) {
                    return;
                }
                Face f[2];
                bool swapped = false;
                canonicalFaces(A, B, k, f, swapped);
                for (const Face& face : f) {
                    if (face.valid) {
                        out.push_back(face);
                    }
                }
            };
            std::vector<CurtainHit> sideHits;
            for (std::size_t q = 0; q + 1 < rays.size(); ++q) {
                const BentRay& A = stretch.rays[rays[q]];
                const BentRay& B = stretch.rays[rays[q + 1]];
                const std::size_t n = std::min(A.points.size(), B.points.size());
                if (n < 2) {
                    continue;
                }
                Box stripBox;
                for (std::size_t k = 0; k < n; ++k) {
                    stripBox.add(A.points[k]);
                    stripBox.add(B.points[k]);
                }
                if (!stripBox.overlaps(polyBox)) {
                    continue;
                }
                std::vector<std::size_t> candidates;
                for (std::size_t j = 0; j < segmentCount; ++j) {
                    if (!degenerate[j] && segmentBoxes[j].overlaps(stripBox)) {
                        candidates.push_back(j);
                    }
                }
                if (candidates.empty()) {
                    continue;
                }
                for (std::size_t k = 0; k + 1 < n; ++k) {
                    Box quadBox;
                    quadBox.add(A.points[k]);
                    quadBox.add(A.points[k + 1]);
                    quadBox.add(B.points[k]);
                    quadBox.add(B.points[k + 1]);
                    // The quad's vertices in canonical ray order (P the
                    // canonical first ray), carrying the stored-frame
                    // across (0 on A, 1 on B).
                    Face faces[2];
                    bool swapped = false;
                    const bool useA0B1 = canonicalFaces(A, B, k, faces, swapped);
                    const BentRay& P = swapped ? B : A;
                    const BentRay& Q = swapped ? A : B;
                    const double acrossP = swapped ? 1.0 : 0.0;
                    const double acrossQ = swapped ? 0.0 : 1.0;
                    const Vertex a0{P.points[k], P.s[k], P.theta[k], acrossP, 0.0};
                    const Vertex a1{P.points[k + 1], P.s[k + 1], P.theta[k + 1], acrossP, 1.0};
                    const Vertex b0{Q.points[k], Q.s[k], Q.theta[k], acrossQ, 0.0};
                    const Vertex b1{Q.points[k + 1], Q.s[k + 1], Q.theta[k + 1], acrossQ, 1.0};
                    const Vertex* tris[2][3] = {{&a0, &a1, &b1}, {&a0, &b1, &b0}};
                    const Vertex* trisAlt[2][3] = {{&a0, &a1, &b0}, {&a1, &b1, &b0}};
                    const auto& split = useA0B1 ? tris : trisAlt;
                    for (const std::size_t j : candidates) {
                        if (!segmentBoxes[j].overlaps(quadBox)) {
                            continue;
                        }
                        const cv::Vec3d& p0 = polyline[j];
                        const cv::Vec3d& p1 = polyline[j + 1];
                        // The intersection arithmetic runs on the segment in
                        // canonical direction (lexicographically smaller end
                        // first), so a hit is bit for bit the same whichever
                        // way the polyline is stored and however many times
                        // it passes through the same two points; the
                        // parameter is then mapped back onto the stored
                        // direction.
                        const bool flipped = lexLess(p1, p0);
                        const cv::Vec3d& pa = flipped ? p1 : p0;
                        const cv::Vec3d& pb = flipped ? p0 : p1;
                        for (int ti = 0; ti < 2; ++ti) {
                            const auto& tri = split[ti];
                            const Face& face = faces[ti];
                            if (!face.valid) {
                                continue;
                            }
                            TriangleHit th = segmentTriangle(pa, pb, tri[0]->p, tri[1]->p, tri[2]->p);
                            const double tCanonical = th.t;
                            if (th.hit && flipped) {
                                th.t = 1.0 - th.t;
                            }
                            CurtainHit hit;
                            hit.stretch = st;
                            hit.rayA = rays[q];
                            hit.rayB = rays[q + 1];
                            hit.seedA = A.startSample;
                            hit.seedB = B.startSample;
                            hit.side = side;
                            hit.quad = k;
                            hit.triangle = ti;
                            // Both rays support this strip: use the larger
                            // turn and the smaller conditioning. Choosing a
                            // canonical ray alone would change eligibility
                            // when a rotation swaps the lexicographic order.
                            for (const BentRay* ray : {&P, &Q}) {
                                if (ray->points.size() > 1 && k + 1 < ray->points.size()) {
                                    const cv::Vec3d first = unit(ray->points[1] - ray->points[0]);
                                    const cv::Vec3d here = unit(ray->points[k + 1] - ray->points[k]);
                                    hit.turnDeg = std::max(hit.turnDeg,
                                        std::acos(std::clamp(first.dot(here), -1.0, 1.0)) * 180.0 / kPi);
                                }
                                for (std::size_t c = 0; c <= k + 1 && c < ray->conditioning.size(); ++c) {
                                    hit.minConditioning = std::min(hit.minConditioning, ray->conditioning[c]);
                                }
                            }
                            hit.segment = j;
                            if (th.coplanar) {
                                const cv::Vec3d w0 = barycentric(pa, tri[0]->p, tri[1]->p, tri[2]->p);
                                const cv::Vec3d w1 = barycentric(pb, tri[0]->p, tri[1]->p, tri[2]->p);
                                if (std::isnan(w0[0]) || std::isnan(w1[0])) {
                                    continue;
                                }
                                double t0 = 0.0;
                                double t1 = 1.0;
                                for (int c = 0; c < 3 && t0 <= t1; ++c) {
                                    const double dw = w1[c] - w0[c];
                                    if (std::abs(dw) < kEps) {
                                        if (w0[c] < -1e-9) {
                                            t0 = 1.0;
                                            t1 = 0.0;
                                        }
                                    } else {
                                        const double tc = -w0[c] / dw;
                                        if (dw > 0.0) {
                                            t0 = std::max(t0, tc);
                                        } else {
                                            t1 = std::min(t1, tc);
                                        }
                                    }
                                }
                                if (t0 > t1 + 1e-9) {
                                    continue;
                                }
                                const double tm = 0.5 * (t0 + t1);
                                const cv::Vec3d wm = w0 + (w1 - w0) * tm;
                                hit.across = wm[0] * tri[0]->across + wm[1] * tri[1]->across + wm[2] * tri[2]->across;
                                hit.along = wm[0] * tri[0]->along + wm[1] * tri[1]->along + wm[2] * tri[2]->along;
                                // An overlap lying on the seed edge (s = 0)
                                // is the fiber itself: one record, owned by
                                // the outward side like a seed-edge crossing.
                                {
                                    const cv::Vec3d p0w = w0;
                                    const cv::Vec3d p1w = w1;
                                    const double along0 = p0w[0] * tri[0]->along + p0w[1] * tri[1]->along + p0w[2] * tri[2]->along;
                                    const double along1 = p1w[0] * tri[0]->along + p1w[1] * tri[1]->along + p1w[2] * tri[2]->along;
                                    const double alongLo = along0 + (along1 - along0) * t0;
                                    const double alongHi = along0 + (along1 - along0) * t1;
                                    if (k == 0 && std::abs(alongLo) <= 1e-9 && std::abs(alongHi) <= 1e-9 &&
                                        side != 1) {
                                        continue;
                                    }
                                }
                                hit.s = wm[0] * tri[0]->s + wm[1] * tri[1]->s + wm[2] * tri[2]->s;
                                hit.theta = wm[0] * tri[0]->theta + wm[1] * tri[1]->theta + wm[2] * tri[2]->theta;
                                hit.point = pa + (pb - pa) * tm;
                                hit.t = flipped ? 1.0 - tm : tm;
                                hit.transversality = 0.0;
                                hit.tangential = true;
                                hit.overlapT0 = flipped ? 1.0 - t1 : t0;
                                hit.overlapT1 = flipped ? 1.0 - t0 : t1;
                                sideHits.push_back(hit);
                                continue;
                            }
                            if (!th.hit) {
                                continue;
                            }
                            const double w0 = 1.0 - th.u - th.v;
                            const auto blend = [&](auto member) {
                                return w0 * (tri[0]->*member) + th.u * (tri[1]->*member) +
                                       th.v * (tri[2]->*member);
                            };
                            hit.across = blend(&Vertex::across);
                            hit.along = blend(&Vertex::along);
                            hit.s = blend(&Vertex::s);
                            hit.theta = blend(&Vertex::theta);
                            hit.point = pa + (pb - pa) * tCanonical;
                            hit.t = th.t;
                            // The seed edge (s = 0) is shared by both sides:
                            // the polyline passes through the fiber itself
                            // there, with no side to read. One touch, owned
                            // by the outward side.
                            if (k == 0 && hit.along <= 1e-9) {
                                if (side != 1) {
                                    continue;
                                }
                                hit.touch = true;
                                hit.transversality = 0.0;
                                sideHits.push_back(hit);
                                continue;
                            }
                            // The polyline's directions into and out of the
                            // passage (the nearest distinct samples at a
                            // vertex, the segment otherwise).
                            cv::Vec3d before = p0;
                            cv::Vec3d after = p1;
                            // Endpoint membership is decided on the canonical
                            // parameter (1 - t rounds differently), then
                            // mapped onto the stored direction.
                            const bool atCanonicalStart = tCanonical <= 1e-9;
                            const bool atCanonicalEnd = tCanonical >= 1.0 - 1e-9;
                            const bool atStart = flipped ? atCanonicalEnd : atCanonicalStart;
                            const bool atEnd = flipped ? atCanonicalStart : atCanonicalEnd;
                            bool polylineEnd = false;
                            std::optional<std::size_t> vertexBefore;
                            std::optional<std::size_t> vertexAfter;
                            if (atStart || atEnd) {
                                const std::size_t vertex = atStart ? j : j + 1;
                                hit.point = polyline[vertex];
                                hit.t = atStart ? 0.0 : 1.0;
                                // A vertex hit is found from either adjacent
                                // segment: its blends are recomputed from the
                                // vertex itself so the record does not carry
                                // which segment found it.
                                {
                                    const cv::Vec3d w = barycentric(hit.point, tri[0]->p, tri[1]->p, tri[2]->p);
                                    if (!std::isnan(w[0])) {
                                        const auto blendAt = [&](auto member) {
                                            return w[0] * (tri[0]->*member) + w[1] * (tri[1]->*member) +
                                                   w[2] * (tri[2]->*member);
                                        };
                                        hit.across = blendAt(&Vertex::across);
                                        hit.along = blendAt(&Vertex::along);
                                        hit.s = blendAt(&Vertex::s);
                                        hit.theta = blendAt(&Vertex::theta);
                                    }
                                }
                                if (vertex == 0) {
                                    hit.segment = 0;
                                    hit.t = 0.0;
                                } else {
                                    hit.segment = vertex - 1;
                                    hit.t = 1.0;
                                }
                                vertexBefore = prevDistinct(vertex);
                                vertexAfter = nextDistinct(vertex);
                                polylineEnd = !vertexBefore || !vertexAfter;
                                if (!polylineEnd) {
                                    before = polyline[*vertexBefore];
                                    after = polyline[*vertexAfter];
                                }
                            }
                            // The incident surface: this face, and at a crease
                            // the faces meeting there. Only the quad's
                            // diagonal joins the two triangles: the vertex
                            // of this triangle that is not on the diagonal
                            // has zero weight there (an outer edge has not).
                            std::vector<Face> incident{face};
                            const double offDiagonalWeight =
                                useA0B1 ? (ti == 0 ? th.u : th.v) : (ti == 0 ? w0 : th.u);
                            if (offDiagonalWeight <= 1e-9 && faces[1 - ti].valid) {
                                incident.push_back(faces[1 - ti]);
                            }
                            // On a row edge the quad before/after along the
                            // rays; on a shared ray the neighbouring strip;
                            // at a vertex every quad meeting it, the
                            // diagonal neighbour included.
                            const int dqLo = (hit.across <= 1e-9 && q > 0) ? -1 : 0;
                            const int dqHi = hit.across >= 1.0 - 1e-9 ? 1 : 0;
                            const int dkLo = (hit.along <= 1e-9 && k > 0) ? -1 : 0;
                            const int dkHi = hit.along >= 1.0 - 1e-9 ? 1 : 0;
                            for (int dq = dqLo; dq <= dqHi; ++dq) {
                                for (int dk = dkLo; dk <= dkHi; ++dk) {
                                    if (dq == 0 && dk == 0) {
                                        continue;
                                    }
                                    facesAt(static_cast<std::size_t>(static_cast<long>(q) + dq),
                                            static_cast<std::size_t>(static_cast<long>(k) + dk), incident);
                                }
                            }
                            // Of the neighbouring faces only those that
                            // actually contain the point are incident (a
                            // neighbouring quad's other triangle is not,
                            // unless the point is on its diagonal too).
                            // Containment is judged against the face's own
                            // scale (its longest edge) plus the rounding of
                            // the coordinates, never against the distance
                            // from the origin: a neighbouring face that
                            // merely passes close is not incident.
                            const double coordinateRounding =
                                1e-14 * std::max({std::abs(hit.point[0]), std::abs(hit.point[1]), std::abs(hit.point[2])});
                            incident.erase(std::remove_if(incident.begin() + 1, incident.end(),
                                                          [&](const Face& f) {
                                                              double edge = 0.0;
                                                              for (int e = 0; e < 3; ++e) {
                                                                  edge = std::max(edge, length(f.p[(e + 1) % 3] - f.p[e]));
                                                              }
                                                              const double tolerance = 1e-7 * edge + coordinateRounding;
                                                              return pointTriangleDistance(hit.point, f) > tolerance;
                                                          }),
                                           incident.end());
                            if (polylineEnd) {
                                hit.touch = true;
                                hit.transversality = 0.0;
                            } else {
                                // Sides are read just off the passage along
                                // the polyline, where the incident faces are
                                // the surface: a thousandth of the way to the
                                // neighbouring sample, and no further than a
                                // thousandth of the shortest incident edge,
                                // so the probe is local to the mesh however
                                // long the polyline's segments are.
                                const cv::Vec3d stepIn = hit.point - before;
                                const cv::Vec3d stepOut = after - hit.point;
                                double reach = 1e-3 * std::min(length(stepIn), length(stepOut));
                                for (const Face& f : incident) {
                                    for (int e = 0; e < 3; ++e) {
                                        reach = std::min(reach, 1e-3 * length(f.p[(e + 1) % 3] - f.p[e]));
                                    }
                                }
                                const int sideBefore = sideOf(hit.point - unit(stepIn) * reach, incident);
                                const int sideAfter = sideOf(hit.point + unit(stepOut) * reach, incident);
                                hit.touch = !(sideBefore != 0 && sideAfter != 0 && sideBefore != sideAfter);
                                // Transversality: the smallest over the faces
                                // containing the point and the incident
                                // segments.
                                double tr = kInf;
                                for (const Face& f : incident) {
                                    if (vertexBefore && vertexAfter) {
                                        const std::size_t v = atStart ? j : j + 1;
                                        tr = std::min(tr, std::abs(f.normal.dot(unit(polyline[v] - polyline[*vertexBefore]))));
                                        tr = std::min(tr, std::abs(f.normal.dot(unit(polyline[*vertexAfter] - polyline[v]))));
                                    } else {
                                        tr = std::min(tr, std::abs(f.normal.dot(unit(p1 - p0))));
                                    }
                                }
                                hit.transversality = hit.touch ? 0.0 : (tr == kInf ? 0.0 : tr);
                            }
                            sideHits.push_back(hit);
                        }
                    }
                }
            }
            // The coincident-sample run a vertex hit belongs to.
            const auto vertexRunOf = [&](const CurtainHit& hit) {
                std::size_t vertex = hit.t >= 1.0 ? hit.segment + 1 : hit.segment;
                while (vertex > 0 && length(polyline[vertex - 1] - polyline[vertex]) <= 1e-9) {
                    --vertex;
                }
                return vertex;
            };
            const auto samePosition = [&](const CurtainHit& a, const CurtainHit& b) {
                if (a.tangential != b.tangential) {
                    return false;
                }
                if (length(a.point - b.point) >= 1e-6) {
                    return false;
                }
                const bool aVertex = a.t <= 0.0 || a.t >= 1.0;
                const bool bVertex = b.t <= 0.0 || b.t >= 1.0;
                if (aVertex && bVertex) {
                    return vertexRunOf(a) == vertexRunOf(b);
                }
                if (aVertex != bVertex) {
                    return false;
                }
                return a.segment == b.segment && std::abs(a.t - b.t) < 1e-9;
            };
            // The boundary two faces share, when they share one: the
            // diagonal of one quad, the row between two quads of one strip,
            // or the shared ray between two strips at one row; true when the
            // point lies on it.
            const auto onSharedBoundary = [&](const CurtainHit& a, const CurtainHit& b) {
                const auto onSegment = [&](const cv::Vec3d& p, const cv::Vec3d& q) {
                    const cv::Vec3d d = q - p;
                    const double l2 = d.dot(d);
                    const double t = l2 > 0.0 ? std::clamp((a.point - p).dot(d) / l2, 0.0, 1.0) : 0.0;
                    return length(a.point - (p + d * t)) < 1e-6;
                };
                const BentRay& A = stretch.rays[a.rayA];
                const BentRay& B = stretch.rays[a.rayB];
                if (a.rayA == b.rayA) {
                    if (a.quad == b.quad) {
                        if (a.triangle == b.triangle) {
                            return true;
                        }
                        // The diagonal of the canonical split (the faces
                        // were built on the canonical pair).
                        Face f[2];
                        bool swapped = false;
                        const bool useP0Q1 = canonicalFaces(A, B, a.quad, f, swapped);
                        const BentRay& P = swapped ? B : A;
                        const BentRay& Q = swapped ? A : B;
                        return useP0Q1 ? onSegment(P.points[a.quad], Q.points[a.quad + 1])
                                       : onSegment(P.points[a.quad + 1], Q.points[a.quad]);
                    }
                    const std::size_t hi = std::max(a.quad, b.quad);
                    if (hi != std::min(a.quad, b.quad) + 1) {
                        return false;
                    }
                    return onSegment(A.points[hi], B.points[hi]);
                }
                // Different strips: they share a ray when one's B is the
                // other's A. The shared boundary is that ray's segment at a
                // common row, or its vertex between consecutive rows: the
                // point must lie on the ray's segment at BOTH quads.
                const CurtainHit* lower = a.rayB == b.rayA ? &a : (b.rayB == a.rayA ? &b : nullptr);
                if (lower == nullptr || std::max(a.quad, b.quad) > std::min(a.quad, b.quad) + 1) {
                    return false;
                }
                const BentRay& R = stretch.rays[lower->rayB];
                for (const std::size_t k : {a.quad, b.quad}) {
                    if (k + 1 >= R.points.size() || !onSegment(R.points[k], R.points[k + 1])) {
                        return false;
                    }
                }
                return true;
            };
            // A zero-length overlap (a segment through a corner of a face it
            // otherwise lies beside) already inside a positive-length overlap
            // of the same segment is that overlap's contact, not a second
            // one.
            {
                std::vector<CurtainHit> kept;
                kept.reserve(sideHits.size());
                for (const CurtainHit& hit : sideHits) {
                    bool absorbed = false;
                    if (hit.tangential && hit.overlapT1 - hit.overlapT0 <= 1e-9) {
                        for (const CurtainHit& other : sideHits) {
                            if (!other.tangential || other.segment != hit.segment ||
                                other.overlapT1 - other.overlapT0 <= 1e-9) {
                                continue;
                            }
                            if (hit.overlapT0 >= other.overlapT0 - 1e-9 && hit.overlapT0 <= other.overlapT1 + 1e-9) {
                                absorbed = true;
                                break;
                            }
                        }
                    }
                    if (!absorbed) {
                        kept.push_back(hit);
                    }
                }
                sideHits = std::move(kept);
            }
            // A segment lying in a face does not cross the curtain where it
            // leaves that face across a crease: its crossing at an END of
            // the overlap, on a face sharing the boundary the end lies on,
            // is a contact. A crossing of an unrelated face at that point
            // stands.
            for (CurtainHit& hit : sideHits) {
                if (hit.tangential || hit.touch) {
                    continue;
                }
                for (const CurtainHit& flat : sideHits) {
                    if (!flat.tangential || flat.segment != hit.segment) {
                        continue;
                    }
                    const double distance = std::min(std::abs(hit.t - flat.overlapT0), std::abs(hit.t - flat.overlapT1));
                    if (distance > 1e-6) {
                        continue;
                    }
                    const cv::Vec3d end0 = polyline[flat.segment] + (polyline[flat.segment + 1] - polyline[flat.segment]) * flat.overlapT0;
                    const cv::Vec3d end1 = polyline[flat.segment] + (polyline[flat.segment + 1] - polyline[flat.segment]) * flat.overlapT1;
                    if (length(hit.point - end0) > 1e-6 && length(hit.point - end1) > 1e-6) {
                        continue;
                    }
                    CurtainHit flatAtEnd = flat;
                    flatAtEnd.point = hit.point;
                    if (!onSharedBoundary(hit, flatAtEnd)) {
                        continue;
                    }
                    hit.touch = true;
                    hit.transversality = 0.0;
                    break;
                }
            }
            // The start of the ray of `hit`'s strip that is NOT shared with
            // `other`'s strip (by ray provenance, never by distance); for
            // one strip, its A start.
            const auto otherStart = [&](const CurtainHit& hit, const CurtainHit& other) {
                if (hit.rayA != other.rayA && (hit.rayA == other.rayB)) {
                    return stretch.rays[hit.rayB].points.front();
                }
                return stretch.rays[hit.rayA].points.front();
            };
            std::vector<CurtainHit> merged;
            for (CurtainHit& hit : sideHits) {
                hit.contributingStrips.emplace_back(std::min(hit.rayA, hit.rayB), std::max(hit.rayA, hit.rayB));
                bool absorbed = false;
                for (CurtainHit& kept : merged) {
                    if (!samePosition(kept, hit) || !onSharedBoundary(kept, hit)) {
                        continue;
                    }
                    const bool creaseTouch = kept.touch != hit.touch;
                    const double tr = creaseTouch ? 0.0 : std::min(kept.transversality, hit.transversality);
                    // At a shared ray/row the incident strips all support
                    // this crossing. Preserve their worst telemetry before
                    // choosing its storage-independent representative.
                    const double turn = std::max(kept.turnDeg, hit.turnDeg);
                    const double conditioning = std::min(kept.minConditioning, hit.minConditioning);
                    auto strips = kept.contributingStrips;
                    strips.insert(strips.end(), hit.contributingStrips.begin(), hit.contributingStrips.end());
                    std::sort(strips.begin(), strips.end());
                    strips.erase(std::unique(strips.begin(), strips.end()), strips.end());
                    if (lexLess(otherStart(hit, kept), otherStart(kept, hit))) {
                        kept = hit;
                    }
                    kept.contributingStrips = std::move(strips);
                    kept.transversality = tr;
                    kept.turnDeg = turn;
                    kept.minConditioning = conditioning;
                    kept.touch = kept.touch || creaseTouch;
                    absorbed = true;
                    break;
                }
                if (!absorbed) {
                    merged.push_back(hit);
                }
            }
            // One canonical order, by one lexicographic key: the strip (its
            // two ray starts as an unordered pair, the smaller first, so the
            // key does not depend on which ray is A, i.e. on the owner's
            // storage order), then the hit (s, the greater transversality
            // first, the hit point), then storage-order tiebreakers that
            // only separate geometrically identical records.
            const auto compareVec = [](const cv::Vec3d& a, const cv::Vec3d& b) {
                for (int i = 0; i < 3; ++i) {
                    if (a[i] < b[i]) {
                        return -1;
                    }
                    if (a[i] > b[i]) {
                        return 1;
                    }
                }
                return 0;
            };
            const auto stripKey = [&](const CurtainHit& h) {
                const cv::Vec3d& pa = stretch.rays[h.rayA].points.front();
                const cv::Vec3d& pb = stretch.rays[h.rayB].points.front();
                return compareVec(pa, pb) <= 0 ? std::pair<cv::Vec3d, cv::Vec3d>{pa, pb}
                                               : std::pair<cv::Vec3d, cv::Vec3d>{pb, pa};
            };
            std::stable_sort(merged.begin(), merged.end(), [&](const CurtainHit& a, const CurtainHit& b) {
                const auto ka = stripKey(a);
                const auto kb = stripKey(b);
                if (const int c = compareVec(ka.first, kb.first)) {
                    return c < 0;
                }
                if (const int c = compareVec(ka.second, kb.second)) {
                    return c < 0;
                }
                if (a.s != b.s) {
                    return a.s < b.s;
                }
                if (a.transversality != b.transversality) {
                    return a.transversality > b.transversality;
                }
                if (const int c = compareVec(a.point, b.point)) {
                    return c < 0;
                }
                return std::tie(a.segment, a.t, a.rayA) < std::tie(b.segment, b.t, b.rayA);
            });
            hits.insert(hits.end(), merged.begin(), merged.end());
        }
    }
    // One canonical order over ALL runs and sides, by the same geometric
    // key (strip as an unordered pair of ray starts, side, s, greater
    // transversality first, hit point); run indices follow the owner's
    // storage order and only break ties between identical records.
    const auto compareVec = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        for (int i = 0; i < 3; ++i) {
            if (a[i] < b[i]) {
                return -1;
            }
            if (a[i] > b[i]) {
                return 1;
            }
        }
        return 0;
    };
    const auto stripKey = [&](const CurtainHit& h) {
        const BentStretch& st = curtain.stretches[h.stretch];
        const cv::Vec3d& pa = st.rays[h.rayA].points.front();
        const cv::Vec3d& pb = st.rays[h.rayB].points.front();
        return compareVec(pa, pb) <= 0 ? std::pair<cv::Vec3d, cv::Vec3d>{pa, pb}
                                       : std::pair<cv::Vec3d, cv::Vec3d>{pb, pa};
    };
    std::stable_sort(hits.begin(), hits.end(), [&](const CurtainHit& a, const CurtainHit& b) {
        const auto ka = stripKey(a);
        const auto kb = stripKey(b);
        if (const int c = compareVec(ka.first, kb.first)) {
            return c < 0;
        }
        if (const int c = compareVec(ka.second, kb.second)) {
            return c < 0;
        }
        if (a.side != b.side) {
            return a.side > b.side;
        }
        if (a.s != b.s) {
            return a.s < b.s;
        }
        if (a.transversality != b.transversality) {
            return a.transversality > b.transversality;
        }
        if (const int c = compareVec(a.point, b.point)) {
            return c < 0;
        }
        return std::tie(a.stretch, a.segment, a.t, a.rayA) < std::tie(b.stretch, b.segment, b.t, b.rayA);
    });
    return hits;
}

CurtainSelfCrossings curtainSelfCrossings(const BentCurtain& curtain,
                                        const std::vector<cv::Vec3d>& points,
                                        const std::vector<double>& thetaLine)
{
    CurtainSelfCrossings crossings;
    // The fiber's own polyline against its curtain: the first CROSSING of
    // each strip side at positive arclength ON THE SAME WINDING (the
    // crossing's unwrapped angle, the ray's accumulated angle taken out,
    // within half a turn of the seeds': the sheet folding back through
    // its curtain, not the fiber's next winding passing through it) bounds
    // the readings beyond which a pair's hit looks through the fiber's
    // own fold. Touches and tangential overlaps are not contacts here: a
    // fiber grazes its own curtain all along (neighbouring runs, the
    // samples past a run's ends), and counting them withholds most
    // readings for nothing. A fold whose returning limb passes beside the
    // curtain without crossing it (its limbs offset in height) is not
    // seen; such readings are left to the solver's repair and to the user.
    for (const CurtainHit& hit : intersectCurtain(curtain, points)) {
        if (hit.touch || hit.tangential || !(hit.s > 0.0) || hit.segment + 1 >= thetaLine.size() ||
            hit.seedA >= thetaLine.size() || hit.seedB >= thetaLine.size()) {
            continue;
        }
        const double thetaSeed =
            thetaLine[hit.seedA] + hit.across * (thetaLine[hit.seedB] - thetaLine[hit.seedA]);
        const double thetaHit =
            thetaLine[hit.segment] + hit.t * (thetaLine[hit.segment + 1] - thetaLine[hit.segment]);
        const double turns = (thetaHit - thetaSeed - hit.theta) / (2.0 * kPi);
        if (std::llround(turns) != 0) {
            continue;
        }
        // A self-crossing on a shared ray bounds both incident strips,
        // whichever one represents the merged hit.
        for (const auto& [rayA, rayB] : hit.contributingStrips) {
            const auto key = std::make_tuple(hit.stretch, rayA, rayB, hit.side);
            const auto found = crossings.find(key);
            if (found == crossings.end() || hit.s < found->second) {
                crossings[key] = hit.s;
            }
        }
    }
    return crossings;
}

double firstSelfCrossing(const CurtainHit& hit, const CurtainSelfCrossings& crossings)
{
    double first = kInf;
    for (const auto& [rayA, rayB] : hit.contributingStrips) {
        const auto found = crossings.find({hit.stretch, rayA, rayB, hit.side});
        if (found != crossings.end()) {
            first = std::min(first, found->second);
        }
    }
    return first;
}

} // namespace vc3d::fiber_map::bent
