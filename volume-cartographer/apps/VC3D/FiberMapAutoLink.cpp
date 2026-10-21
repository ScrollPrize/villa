#include "FiberMapAutoLink.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace vc3d::fiber_map::autolink
{
namespace
{

struct DrawnRun {
    std::vector<QPointF> points;
    std::vector<double> arclengths;
};

// The drawn geometry of every run (displayRunPoints) with the matching
// arclength per drawn sample: where the overlap sample was replaced by a
// control's drawn position, the arclength is that control's.
std::vector<DrawnRun> drawnRuns(const PlacedFiber& fiber)
{
    std::vector<DrawnRun> runs;
    runs.reserve(fiber.runs.size());
    for (std::size_t runIndex = 0; runIndex < fiber.runs.size(); ++runIndex) {
        const Run& run = fiber.runs[runIndex];
        DrawnRun drawn;
        drawn.points = displayRunPoints(fiber, runIndex);
        drawn.arclengths = run.arclengths;
        if (drawn.arclengths.size() != drawn.points.size()) {
            // A run without arclengths cannot be inverted; it is not searched.
            drawn.points.clear();
            drawn.arclengths.clear();
            runs.push_back(std::move(drawn));
            continue;
        }
        if (drawn.points.size() >= 2) {
            // The same conditions displayRunPoints replaces the overlap
            // samples under (a styled run, or a neighbour of one), decided
            // from the run styles - never from comparing points, whose
            // fuzzy equality would miss a replacement by a near-coincident
            // control.
            const auto styled = [](const Run& other) { return other.gap || other.damaged; };
            const bool previousStyled = runIndex > 0 && styled(fiber.runs[runIndex - 1]);
            const bool nextStyled =
                runIndex + 1 < fiber.runs.size() && styled(fiber.runs[runIndex + 1]);
            const auto controlArc = [&fiber](int index) -> std::optional<double> {
                if (index < 0 || static_cast<std::size_t>(index) >= fiber.controlArclength.size() ||
                    static_cast<std::size_t>(index) >= fiber.controlPoints.size()) {
                    return std::nullopt;
                }
                return fiber.controlArclength[static_cast<std::size_t>(index)];
            };
            if (styled(run) || previousStyled) {
                if (const auto s = controlArc(run.firstControl)) {
                    drawn.arclengths.front() = *s;
                }
            }
            if (styled(run) || nextStyled) {
                if (const auto s = controlArc(run.lastControl)) {
                    drawn.arclengths.back() = *s;
                }
            }
        }
        runs.push_back(std::move(drawn));
    }
    return runs;
}

struct Box {
    double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
};

Box boundsOf(const std::vector<QPointF>& points)
{
    Box box{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    for (const QPointF& p : points) {
        box.x0 = std::min(box.x0, p.x());
        box.y0 = std::min(box.y0, p.y());
        box.x1 = std::max(box.x1, p.x());
        box.y1 = std::max(box.y1, p.y());
    }
    return box;
}

bool boxesOverlap(const Box& a, const Box& b, double margin)
{
    return a.x0 <= b.x1 + margin && b.x0 <= a.x1 + margin &&
           a.y0 <= b.y1 + margin && b.y0 <= a.y1 + margin;
}

// a*b - c*d to within ~1.5 ulp whatever the cancellation (Kahan), through
// the fused multiply-add: the cross products below keep their accuracy for
// near-parallel directions, fused or unfused arithmetic alike.
double differenceOfProducts(double a, double b, double c, double d)
{
    const double cd = c * d;
    const double err = std::fma(-c, d, cd);
    const double dop = std::fma(a, b, -cd);
    return dop + err;
}

double cross(const QPointF& a, const QPointF& b)
{
    return differenceOfProducts(a.x(), b.y(), a.y(), b.x());
}

// Inclusive segment-segment intersection (a touch at a vertex or endpoint
// counts). A collinear overlap reports the point of the overlap nearest
// `preferNear` (the click), so the meeting on coincident stretches follows
// the user, not the segment's start.
std::optional<std::pair<double, double>> segmentIntersection(const QPointF& p0, const QPointF& p1,
                                                             const QPointF& q0, const QPointF& q1,
                                                             const QPointF& preferNear)
{
    const QPointF r = p1 - p0;
    const QPointF s = q1 - q0;
    // Every cross product below is Kahan-accurate (differenceOfProducts),
    // so the determinant is zero exactly when the directions are parallel
    // in double precision and otherwise correct relative to its size: two
    // segments with ANY nonzero angle meet at one point, however shallow
    // (a 2e-9 tilt over 10000 voxels included), and the formula finds it.
    // Only an exactly parallel pair is tested for a collinear overlap.
    const double denom = cross(r, s);
    const QPointF qp = q0 - p0;
    if (denom == 0.0) {
        const double rr = QPointF::dotProduct(r, r);
        if (cross(qp, r) != 0.0 || cross(q1 - p0, r) != 0.0) {
            return std::nullopt;  // parallel, not collinear
        }
        if (rr <= 0.0) {
            return std::nullopt;
        }
        double t0 = QPointF::dotProduct(qp, r) / rr;
        double t1 = QPointF::dotProduct(qp + s, r) / rr;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        const double lo = std::max(0.0, t0);
        const double hi = std::min(1.0, t1);
        if (lo > hi) {
            return std::nullopt;
        }
        const double t = std::clamp(QPointF::dotProduct(preferNear - p0, r) / rr, lo, hi);
        const double ss = QPointF::dotProduct(s, s);
        const double u = ss > 0.0 ? std::clamp(QPointF::dotProduct(p0 + t * r - q0, s) / ss, 0.0, 1.0)
                                  : 0.0;
        return std::make_pair(t, u);
    }
    const double t = cross(qp, s) / denom;
    const double u = cross(qp, r) / denom;
    const double tol = 1e-9;
    if (t < -tol || t > 1.0 + tol || u < -tol || u > 1.0 + tol) {
        return std::nullopt;
    }
    return std::make_pair(std::clamp(t, 0.0, 1.0), std::clamp(u, 0.0, 1.0));
}

double distance2(const QPointF& a, const QPointF& b)
{
    const QPointF d = a - b;
    return QPointF::dotProduct(d, d);
}

// Closest point of segment [a, b] to p: returns the parameter.
double projectParameter(const QPointF& p, const QPointF& a, const QPointF& b)
{
    const QPointF ab = b - a;
    const double len2 = QPointF::dotProduct(ab, ab);
    if (len2 <= 0.0) {
        return 0.0;
    }
    return std::clamp(QPointF::dotProduct(p - a, ab) / len2, 0.0, 1.0);
}

double lerp(double a, double b, double t)
{
    return a + (b - a) * t;
}

MeetingSide sideAt(const std::vector<DrawnRun>& runs, std::size_t runIndex, std::size_t sample, double t)
{
    MeetingSide side;
    side.runIndex = runIndex;
    side.sample = sample;
    side.t = t;
    const DrawnRun& run = runs[runIndex];
    side.arclength = lerp(run.arclengths[sample], run.arclengths[sample + 1], t);
    return side;
}

}  // namespace

std::optional<MapMeeting> findMapMeeting(const PlacedFiber& a,
                                         const PlacedFiber& b,
                                         QPointF preferNear,
                                         double maxApproachVx)
{
    const std::vector<DrawnRun> runsA = drawnRuns(a);
    const std::vector<DrawnRun> runsB = drawnRuns(b);
    std::vector<Box> boxesB;
    boxesB.reserve(runsB.size());
    for (const DrawnRun& run : runsB) {
        boxesB.push_back(boundsOf(run.points));
    }

    std::optional<MapMeeting> best;
    double bestScore = std::numeric_limits<double>::infinity();
    // Pass 1: crossings; the one nearest preferNear wins.
    for (std::size_t ia = 0; ia < runsA.size(); ++ia) {
        const DrawnRun& ra = runsA[ia];
        if (ra.points.size() < 2) {
            continue;
        }
        const Box boxA = boundsOf(ra.points);
        for (std::size_t ib = 0; ib < runsB.size(); ++ib) {
            const DrawnRun& rb = runsB[ib];
            if (rb.points.size() < 2 || !boxesOverlap(boxA, boxesB[ib], 0.0)) {
                continue;
            }
            for (std::size_t i = 0; i + 1 < ra.points.size(); ++i) {
                for (std::size_t j = 0; j + 1 < rb.points.size(); ++j) {
                    const auto hit = segmentIntersection(ra.points[i], ra.points[i + 1],
                                                         rb.points[j], rb.points[j + 1],
                                                         preferNear);
                    if (!hit) {
                        continue;
                    }
                    const QPointF point = ra.points[i] + hit->first * (ra.points[i + 1] - ra.points[i]);
                    const double score = distance2(point, preferNear);
                    if (score < bestScore) {
                        bestScore = score;
                        MapMeeting meeting;
                        meeting.point = point;
                        meeting.crossing = true;
                        meeting.distance = 0.0;
                        meeting.a = sideAt(runsA, ia, i, hit->first);
                        meeting.b = sideAt(runsB, ib, j, hit->second);
                        best = meeting;
                    }
                }
            }
        }
    }
    if (best) {
        return best;
    }

    // Pass 2: closest approach of the two polylines (point-to-segment both
    // ways), accepted within maxApproachVx.
    const double limit2 = maxApproachVx * maxApproachVx;
    double bestDistance2 = std::numeric_limits<double>::infinity();
    const auto consider = [&](const std::vector<DrawnRun>& runsP, const std::vector<DrawnRun>& runsQ,
                              bool pIsA) {
        for (std::size_t ip = 0; ip < runsP.size(); ++ip) {
            const DrawnRun& rp = runsP[ip];
            if (rp.points.size() < 2) {
                continue;
            }
            const Box boxP = boundsOf(rp.points);
            for (std::size_t iq = 0; iq < runsQ.size(); ++iq) {
                const DrawnRun& rq = runsQ[iq];
                if (rq.points.size() < 2 || !boxesOverlap(boxP, boundsOf(rq.points), maxApproachVx)) {
                    continue;
                }
                for (std::size_t i = 0; i < rp.points.size(); ++i) {
                    const QPointF p = rp.points[i];
                    for (std::size_t j = 0; j + 1 < rq.points.size(); ++j) {
                        const double u = projectParameter(p, rq.points[j], rq.points[j + 1]);
                        const QPointF q = rq.points[j] + u * (rq.points[j + 1] - rq.points[j]);
                        const double d2 = distance2(p, q);
                        // Inclusive limit; among equals the first found wins.
                        if (d2 > limit2 || d2 >= bestDistance2) {
                            continue;
                        }
                        bestDistance2 = d2;
                        MapMeeting meeting;
                        meeting.point = 0.5 * (p + q);
                        meeting.crossing = false;
                        meeting.distance = std::sqrt(d2);
                        // The vertex side: the vertex is sample i of its run
                        // (parameter 0 on segment [i, i+1], or 1 on the last).
                        const std::size_t vs = i + 1 < rp.points.size() ? i : i - 1;
                        const double vt = i + 1 < rp.points.size() ? 0.0 : 1.0;
                        const MeetingSide vertexSide = sideAt(runsP, ip, vs, vt);
                        const MeetingSide segmentSide = sideAt(runsQ, iq, j, u);
                        meeting.a = pIsA ? vertexSide : segmentSide;
                        meeting.b = pIsA ? segmentSide : vertexSide;
                        best = meeting;
                    }
                }
            }
        }
    };
    consider(runsA, runsB, true);
    consider(runsB, runsA, false);
    return best;
}

double sourcePositionAtArclength(const std::vector<double>& raw, double s)
{
    if (raw.empty()) {
        return 0.0;
    }
    const std::size_t n = raw.size();
    if (!(s > raw.front())) {
        return 0.0;
    }
    const auto it = std::lower_bound(raw.begin(), raw.end(), s);
    if (it == raw.end()) {
        return static_cast<double>(n - 1);
    }
    const std::size_t j = static_cast<std::size_t>(it - raw.begin());
    if (raw[j] == s) {
        return static_cast<double>(j);
    }
    if (j == 0) {
        return 0.0;
    }
    const double span = raw[j] - raw[j - 1];
    return static_cast<double>(j - 1) + (s - raw[j - 1]) / span;
}

cv::Vec3d pointAtLinePosition(const std::vector<cv::Vec3d>& linePoints, double linePosition)
{
    if (linePoints.empty()) {
        return {0.0, 0.0, 0.0};
    }
    const double maxPosition = static_cast<double>(linePoints.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    const std::size_t lower = static_cast<std::size_t>(std::floor(linePosition));
    const std::size_t upper = std::min(lower + 1, linePoints.size() - 1);
    const double t = linePosition - static_cast<double>(lower);
    return linePoints[lower] * (1.0 - t) + linePoints[upper] * t;
}

std::vector<double> cumulativeArclengths(const std::vector<cv::Vec3d>& linePoints)
{
    std::vector<double> s(linePoints.size(), 0.0);
    for (std::size_t i = 1; i < linePoints.size(); ++i) {
        s[i] = s[i - 1] + cv::norm(linePoints[i] - linePoints[i - 1]);
    }
    return s;
}

namespace
{

double arclengthAt(const std::vector<double>& cumulative, double position)
{
    if (cumulative.empty()) {
        return 0.0;
    }
    position = std::clamp(position, 0.0, static_cast<double>(cumulative.size() - 1));
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, cumulative.size() - 1);
    const double t = position - static_cast<double>(lower);
    return cumulative[lower] * (1.0 - t) + cumulative[upper] * t;
}

// The fractional line position at a cumulative arclength (first match on a
// plateau), clamped to the line.
double positionAtArclength(const std::vector<double>& cumulative, double s)
{
    return sourcePositionAtArclength(cumulative, s);
}

// The exact window of one side: fractional line positions [lo, hi] whose
// arclength is within +-window of the arclength at `position` (clipped to
// the line), never empty.
std::pair<double, double> windowPositions(const std::vector<double>& cumulative,
                                          double position, double window)
{
    const std::size_t n = cumulative.size();
    if (n == 0) {
        return {0.0, 0.0};
    }
    const double s = arclengthAt(cumulative, position);
    double lo = positionAtArclength(cumulative, s - window);
    double hi = positionAtArclength(cumulative, s + window);
    const double maxPosition = static_cast<double>(n - 1);
    lo = std::clamp(std::min(lo, position), 0.0, maxPosition);
    hi = std::clamp(std::max(hi, position), 0.0, maxPosition);
    return {lo, hi};
}

cv::Vec3d accurateCross(const cv::Vec3d& u, const cv::Vec3d& v)
{
    return {differenceOfProducts(u[1], v[2], u[2], v[1]),
            differenceOfProducts(u[2], v[0], u[0], v[2]),
            differenceOfProducts(u[0], v[1], u[1], v[0])};
}

// Closest points of two 3D segments. The unconstrained minimum of the
// line-line problem is `s = ((p2 - p1) x d2) . n / |n|^2`, `t = ((p2 - p1)
// x d1) . n / |n|^2` with `n = d1 x d2`; every cross product is computed
// with differenceOfProducts, so the tiny `n` of near-parallel segments is
// accurate relative to its own size and the ratio is right (no `a*e - b*b`
// cancellation). The boundary cases clamp one parameter and re-solve the
// other; the lower of the two candidate pairs wins.
void closestSegmentPoints(const cv::Vec3d& p1, const cv::Vec3d& q1,
                          const cv::Vec3d& p2, const cv::Vec3d& q2,
                          double& s, double& t)
{
    const cv::Vec3d d1 = q1 - p1;
    const cv::Vec3d d2 = q2 - p2;
    const cv::Vec3d w = p2 - p1;
    const double a = d1.dot(d1);
    const double e = d2.dot(d2);
    constexpr double kEps = 1e-300;
    const auto tFor = [&](double sValue) {
        return e > kEps ? std::clamp((sValue * d1 - w).dot(d2) / e, 0.0, 1.0) : 0.0;
    };
    const auto sFor = [&](double tValue) {
        return a > kEps ? std::clamp((w + tValue * d2).dot(d1) / a, 0.0, 1.0) : 0.0;
    };
    if (a <= kEps || e <= kEps) {
        s = a <= kEps ? 0.0 : sFor(0.0);
        t = e <= kEps ? 0.0 : tFor(s);
        if (a > kEps) {
            s = sFor(t);
        }
        return;
    }
    const cv::Vec3d n = accurateCross(d1, d2);
    const double nn = n.dot(n);
    double sStar = 0.0;
    double tStar = 0.0;
    if (nn <= 1e-40 * a * e) {
        // Parallel within rounding: the distance is flat along the overlap;
        // segment 2's start decides.
        tStar = 0.0;
        sStar = sFor(tStar);
        tStar = tFor(sStar);
    } else {
        sStar = accurateCross(w, d2).dot(n) / nn;
        tStar = accurateCross(w, d1).dot(n) / nn;
    }
    s = sStar;
    t = tStar;
    if (s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0) {
        const double sA = std::clamp(sStar, 0.0, 1.0);
        const double tA = tFor(sA);
        const double tB = std::clamp(tStar, 0.0, 1.0);
        const double sB = sFor(tB);
        const auto dist2 = [&](double sv, double tv) {
            const cv::Vec3d d = (p1 + sv * d1) - (p2 + tv * d2);
            return d.dot(d);
        };
        if (dist2(sA, tA) <= dist2(sB, tB)) {
            s = sA;
            t = tA;
        } else {
            s = sB;
            t = tB;
        }
    }
}

}  // namespace

std::optional<ClosestPass> findClosestPass(const FiberSide& a, const FiberSide& b)
{
    if (!a.linePoints || !b.linePoints || !a.cumulative || !b.cumulative ||
        a.linePoints->empty() || b.linePoints->empty() ||
        a.cumulative->size() != a.linePoints->size() ||
        b.cumulative->size() != b.linePoints->size()) {
        return std::nullopt;
    }
    const auto [loA, hiA] = windowPositions(*a.cumulative, a.position, a.windowVx);
    const auto [loB, hiB] = windowPositions(*b.cumulative, b.position, b.windowVx);
    const std::vector<cv::Vec3d>& la = *a.linePoints;
    const std::vector<cv::Vec3d>& lb = *b.linePoints;

    ClosestPass best;
    best.windowLoA = loA;
    best.windowHiA = hiA;
    best.windowLoB = loB;
    best.windowHiB = hiB;
    best.firstA = static_cast<std::size_t>(std::floor(loA));
    best.lastA = static_cast<std::size_t>(std::ceil(hiA));
    best.firstB = static_cast<std::size_t>(std::floor(loB));
    best.lastB = static_cast<std::size_t>(std::ceil(hiB));
    double bestDistance2 = std::numeric_limits<double>::infinity();
    const auto consider = [&](double posA, double posB) {
        const cv::Vec3d pa = pointAtLinePosition(la, posA);
        const cv::Vec3d pb = pointAtLinePosition(lb, posB);
        const cv::Vec3d d = pa - pb;
        const double d2 = d.dot(d);
        if (d2 < bestDistance2) {
            bestDistance2 = d2;
            best.positionA = posA;
            best.positionB = posB;
            best.pointA = pa;
            best.pointB = pb;
            best.distance = std::sqrt(d2);
        }
    };
    // The window's sub-segments: segment i clipped to [lo, hi] exactly, so
    // nothing outside +-window is ever searched and the boundary segments are
    // neither widened nor dropped.
    struct SubSegment {
        double from = 0.0;  // fractional line position of its start
        double to = 0.0;    // ... and end (to >= from)
    };
    const auto subSegments = [](double lo, double hi) {
        std::vector<SubSegment> segments;
        if (hi <= lo) {
            segments.push_back({lo, lo});
            return segments;
        }
        for (double i = std::floor(lo); i < hi; i += 1.0) {
            const double from = std::max(lo, i);
            const double to = std::min(hi, i + 1.0);
            if (to > from) {
                segments.push_back({from, to});
            }
        }
        if (segments.empty()) {
            segments.push_back({lo, hi});
        }
        return segments;
    };
    const std::vector<SubSegment> segmentsA = subSegments(loA, hiA);
    const std::vector<SubSegment> segmentsB = subSegments(loB, hiB);
    for (const SubSegment& sa : segmentsA) {
        const cv::Vec3d p1 = pointAtLinePosition(la, sa.from);
        const cv::Vec3d q1 = pointAtLinePosition(la, sa.to);
        for (const SubSegment& sb : segmentsB) {
            const cv::Vec3d p2 = pointAtLinePosition(lb, sb.from);
            const cv::Vec3d q2 = pointAtLinePosition(lb, sb.to);
            double s = 0.0;
            double t = 0.0;
            closestSegmentPoints(p1, q1, p2, q2, s, t);
            consider(sa.from + s * (sa.to - sa.from), sb.from + t * (sb.to - sb.from));
        }
    }
    return best;
}

namespace
{

bool insideGap(const ControlChoiceInput& input, double position)
{
    for (const auto& [lower, upper] : input.gapSpans) {
        if (lower >= input.controlLineIndex.size() || upper >= input.controlLineIndex.size()) {
            continue;
        }
        const double lo = static_cast<double>(input.controlLineIndex[lower]);
        const double hi = static_cast<double>(input.controlLineIndex[upper]);
        if (position > std::min(lo, hi) && position < std::max(lo, hi)) {
            return true;
        }
    }
    return false;
}

bool beyondTaggedEnd(const ControlChoiceInput& input, double position)
{
    const std::size_t n = input.controlLineIndex.size();
    if (n == 0) {
        return false;
    }
    const auto tagged = [&input](std::size_t index) {
        return index < input.kollesisTerminations.size() && input.kollesisTerminations[index];
    };
    if (tagged(0) && position < static_cast<double>(input.controlLineIndex.front())) {
        return true;
    }
    if (tagged(n - 1) && position > static_cast<double>(input.controlLineIndex.back())) {
        return true;
    }
    return false;
}

}  // namespace

ControlChoice chooseControlPoint(const ControlChoiceInput& input)
{
    ControlChoice choice;
    if (!input.linePoints || !input.cumulative || input.linePoints->empty() ||
        input.cumulative->size() != input.linePoints->size()) {
        choice.reason = "no line geometry";
        return choice;
    }
    const std::vector<cv::Vec3d>& line = *input.linePoints;
    const std::vector<double>& cumulative = *input.cumulative;
    const double passArc = arclengthAt(cumulative, input.passPosition);
    const cv::Vec3d passPoint = pointAtLinePosition(line, input.passPosition);
    const auto controlArc = [&](std::size_t control) {
        return cumulative[std::min(input.controlLineIndex[control], cumulative.size() - 1)];
    };
    const auto controlPoint = [&](std::size_t control) {
        return line[std::min(input.controlLineIndex[control], line.size() - 1)];
    };
    const auto within = [&](const cv::Vec3d& p, const cv::Vec3d& q) {
        const cv::Vec3d d = p - q;
        return d.dot(d) <= input.ambiguityTolerance * input.ambiguityTolerance;
    };

    // D4: the nearest UNLINKED control within reuseVx.
    std::optional<std::size_t> reuse;
    double reuseDistance = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < input.controlLineIndex.size(); ++c) {
        const bool linked = c < input.linked.size() && input.linked[c];
        const double d = std::abs(controlArc(c) - passArc);
        if (!linked && d <= input.reuseVx && d < reuseDistance) {
            reuseDistance = d;
            reuse = c;
        }
    }
    if (reuse) {
        const cv::Vec3d p = controlPoint(*reuse);
        for (std::size_t c = 0; c < input.controlLineIndex.size(); ++c) {
            if (c != *reuse && within(p, controlPoint(c))) {
                choice.reason = "two control points share one position";
                return choice;
            }
        }
        choice.kind = ControlChoice::Kind::Reuse;
        choice.controlIndex = static_cast<int>(*reuse);
        choice.linePosition = static_cast<double>(input.controlLineIndex[*reuse]);
        choice.point = p;
        return choice;
    }

    // Place: the nearest position to the pass (either side) inside the window
    // that passes every gate at once.
    const auto gatesAt = [&](double position, std::string* why) -> bool {
        const cv::Vec3d p = pointAtLinePosition(line, position);
        const double arc = arclengthAt(cumulative, position);
        for (std::size_t c = 0; c < input.controlLineIndex.size(); ++c) {
            // Strict: never a span shorter than the minimum spacing. A
            // boundary candidate whose reconstructed arclength comes back a
            // few ulps short is covered by its representable neighbours
            // (see the candidate list), not by a tolerance here.
            if (std::abs(controlArc(c) - arc) < input.minSpacingVx) {
                if (why) *why = "too close to an existing control point";
                return false;
            }
            if (within(p, controlPoint(c))) {
                if (why) *why = "coincides with an existing control point";
                return false;
            }
        }
        if (beyondTaggedEnd(input, position)) {
            if (why) *why = "beyond a kollesis termination";
            return false;
        }
        if (insideGap(input, position)) {
            if (why) *why = "inside a gap span";
            return false;
        }
        if (input.inFocusBounds && !input.inFocusBounds(p)) {
            if (why) *why = "outside the active focus bounds";
            return false;
        }
        return true;
    };
    std::string lastReason;
    if (gatesAt(input.passPosition, &lastReason)) {
        choice.kind = ControlChoice::Kind::Place;
        choice.linePosition = input.passPosition;
        choice.point = passPoint;
        return choice;
    }
    // The admissible set along the line is the window minus the analytic
    // exclusions (the minimum-spacing band around every control, the span
    // beyond a tagged end, the gap spans); the focus bounds are a point
    // predicate. Candidates are the pass, every boundary of those exclusions
    // and every line vertex inside the window, each checked against all the
    // gates; the nearest admissible candidate to the pass wins. A boundary
    // of an exclusion band is the nearest admissible position whenever the
    // pass itself lies inside the band, so a valid position between two
    // vertices is never missed.
    const double lo = std::min(input.windowLo, input.windowHi);
    const double hi = std::max(input.windowLo, input.windowHi);
    std::vector<double> candidates;
    const auto addCandidate = [&](double position) {
        if (std::isfinite(position) && position >= lo && position <= hi) {
            candidates.push_back(position);
        }
    };
    // A boundary (of a spacing band, of the focus box) reconstructed
    // through an arclength or a slab parameter can land a few ulps on the
    // wrong side of the exact edge: the representable positions next to it
    // are candidates too, so the first admissible one is found without any
    // gate being relaxed.
    const auto addBoundaryCandidate = [&](double position) {
        addCandidate(position);
        // Representable neighbours either way, in two resolutions: ulps of
        // the POSITION (1, 2, 4, ... 4096), and steps that move the POINT by
        // 1, 2, 4, ... 4096 ulps of its own coordinates along its segment
        // (a position near 0 has ulps far smaller than the coordinate's;
        // the gates read the point, so the reconstruction error lives in
        // the coordinate's ulps). The first admissible position is among
        // them whatever the magnitudes.
        double up = position;
        double down = position;
        for (int k = 0; k < 13; ++k) {
            const int steps = 1 << k;
            for (int i = 0; i < steps; ++i) {
                up = std::nextafter(up, std::numeric_limits<double>::infinity());
                down = std::nextafter(down, -std::numeric_limits<double>::infinity());
            }
            addCandidate(up);
            addCandidate(down);
        }
        const std::size_t i = static_cast<std::size_t>(
            std::clamp(std::floor(position), 0.0, static_cast<double>(line.size() - 2)));
        const cv::Vec3d p = pointAtLinePosition(line, position);
        const cv::Vec3d d = line[i + 1] - line[i];
        const double segmentLength = std::sqrt(d.dot(d));
        if (segmentLength <= 0.0) {
            return;
        }
        const double magnitude = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2]), 1.0});
        const double coordinateUlp = std::nextafter(magnitude, std::numeric_limits<double>::infinity()) - magnitude;
        const double stepPosition = coordinateUlp / segmentLength;
        for (int k = 0; k < 13; ++k) {
            const double delta = static_cast<double>(1 << k) * stepPosition;
            addCandidate(position + delta);
            addCandidate(position - delta);
        }
    };
    const auto addArcCandidate = [&](double arc) {
        addBoundaryCandidate(positionAtArclength(cumulative, arc));
    };
    // Exact spacing-band boundaries (the gate is strict, so the boundary
    // itself is admissible: controls at 0 and 60 with spacing 30 admit 30).
    for (std::size_t c = 0; c < input.controlLineIndex.size(); ++c) {
        addArcCandidate(controlArc(c) + input.minSpacingVx);
        addArcCandidate(controlArc(c) - input.minSpacingVx);
    }
    // Every line vertex inside the window, and the window's ends.
    for (long index = static_cast<long>(std::ceil(lo)); index <= static_cast<long>(std::floor(hi));
         ++index) {
        candidates.push_back(static_cast<double>(index));
    }
    candidates.push_back(lo);
    candidates.push_back(hi);
    // Where a window segment leaves the ambiguity sphere (radius
    // ambiguityTolerance) around a control: the pass itself can coincide
    // with a distant control of a line that returns near it, and the
    // admissible stretch then begins a hair along the line.
    for (std::size_t c = 0; c < input.controlLineIndex.size(); ++c) {
        const cv::Vec3d center = controlPoint(c);
        const double radius2 = input.ambiguityTolerance * input.ambiguityTolerance;
        (void)radius2;
        for (long index = static_cast<long>(std::floor(lo)); index < static_cast<long>(std::ceil(hi));
             ++index) {
            const std::size_t i = static_cast<std::size_t>(std::max<long>(index, 0));
            if (i + 1 >= line.size()) {
                break;
            }
            const cv::Vec3d a = line[i];
            const cv::Vec3d d = line[i + 1] - a;
            const double qa = d.dot(d);
            if (qa <= 0.0) {
                continue;
            }
            // Geometrically, not through |m|^2 - r^2 (a tiny radius vanishes
            // against a large |m|^2): the foot of the control on the
            // segment's line, the distance from it as a vector norm, then
            // the half chord.
            const double t0 = (center - a).dot(d) / qa;
            const cv::Vec3d foot = a + t0 * d;
            const cv::Vec3d offset = foot - center;
            const double h2 = offset.dot(offset);
            if (h2 > radius2) {
                continue;
            }
            const double halfChord = std::sqrt(std::max(0.0, radius2 - h2) / qa);
            for (const double tt : {t0 - halfChord, t0 + halfChord}) {
                if (tt >= 0.0 && tt <= 1.0) {
                    addBoundaryCandidate(static_cast<double>(i) + tt);
                }
            }
        }
    }
    // The exact entry and exit of every window segment through the focus
    // box (slab clipping), so an admissible stretch strictly inside a
    // segment is found at its boundary.
    if (input.focusBox) {
        const cv::Vec3d boxLo = input.focusBox->first;
        const cv::Vec3d boxHi = input.focusBox->second;
        for (long index = static_cast<long>(std::floor(lo)); index < static_cast<long>(std::ceil(hi));
             ++index) {
            const std::size_t i = static_cast<std::size_t>(std::max<long>(index, 0));
            if (i + 1 >= line.size()) {
                break;
            }
            const cv::Vec3d a = line[i];
            const cv::Vec3d d = line[i + 1] - a;
            double t0 = 0.0;
            double t1 = 1.0;
            bool empty = false;
            for (int axis = 0; axis < 3 && !empty; ++axis) {
                if (std::abs(d[axis]) < 1e-15) {
                    if (a[axis] < boxLo[axis] || a[axis] > boxHi[axis]) {
                        empty = true;
                    }
                    continue;
                }
                double ta = (boxLo[axis] - a[axis]) / d[axis];
                double tb = (boxHi[axis] - a[axis]) / d[axis];
                if (ta > tb) {
                    std::swap(ta, tb);
                }
                t0 = std::max(t0, ta);
                t1 = std::min(t1, tb);
                if (t0 > t1) {
                    empty = true;
                }
            }
            if (!empty) {
                addBoundaryCandidate(static_cast<double>(i) + t0);
                addBoundaryCandidate(static_cast<double>(i) + t1);
            }
        }
    }
    // Nearest to the pass first; equidistant candidates by line position
    // (lower first), so a symmetric choice is the same on every platform.
    std::sort(candidates.begin(), candidates.end(),
              [&](double x, double y) {
                  const double dx = std::abs(arclengthAt(cumulative, x) - passArc);
                  const double dy = std::abs(arclengthAt(cumulative, y) - passArc);
                  if (dx != dy) {
                      return dx < dy;
                  }
                  return x < y;
              });
    std::optional<double> bestPosition;
    for (const double position : candidates) {
        if (gatesAt(position, &lastReason)) {
            bestPosition = position;
            break;
        }
    }
    if (!bestPosition) {
        choice.reason = lastReason.empty() ? "no admissible position in the search window" : lastReason;
        return choice;
    }
    choice.kind = ControlChoice::Kind::Place;
    choice.linePosition = *bestPosition;
    choice.point = pointAtLinePosition(line, *bestPosition);
    return choice;
}

}  // namespace vc3d::fiber_map::autolink
