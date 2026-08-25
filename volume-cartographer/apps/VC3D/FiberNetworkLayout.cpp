#include "FiberNetworkLayout.hpp"

#include <QDebug>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace vc3d::fiber_map
{

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
// Fraction of a network's own extent taken as padding; the LayoutParams pads are
// the floor under it. Dimensionless, so no unit to get wrong.
constexpr double kPadFraction = 0.05;

// Whole-turn rounding with the ties-to-even behaviour of Python's round().
double roundTurns(double turns)
{
    return std::nearbyint(turns);
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

// np.interp: linear interpolation with clamped ends.
double interpolate(double query, const std::vector<double>& xs,
                   const std::vector<double>& ys)
{
    if (xs.empty()) {
        return 0.0;
    }
    if (query <= xs.front()) {
        return ys.front();
    }
    if (query >= xs.back()) {
        return ys.back();
    }
    const auto upper = std::upper_bound(xs.begin(), xs.end(), query);
    const std::size_t hi = static_cast<std::size_t>(upper - xs.begin());
    const std::size_t lo = hi - 1;
    const double span = xs[hi] - xs[lo];
    if (span <= 0.0) {
        return ys[lo];
    }
    return ys[lo] + (query - xs[lo]) / span * (ys[hi] - ys[lo]);
}

// np.unwrap: shift each successive delta into (-pi, pi].
std::vector<double> unwrapAngles(const std::vector<double>& raw)
{
    std::vector<double> out(raw.size());
    if (raw.empty()) {
        return out;
    }
    out[0] = raw[0];
    for (std::size_t i = 1; i < raw.size(); ++i) {
        const double delta = raw[i] - raw[i - 1];
        double wrapped = std::fmod(delta + M_PI, kTwoPi);
        if (wrapped < 0.0) {
            wrapped += kTwoPi;
        }
        wrapped -= M_PI;
        if (wrapped == -M_PI && delta > 0.0) {
            wrapped = M_PI;
        }
        out[i] = out[i - 1] + wrapped;
    }
    return out;
}

std::vector<double> arclengths(const std::vector<QPointF>& points)
{
    std::vector<double> s(points.size(), 0.0);
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double dx = points[i].x() - points[i - 1].x();
        const double dy = points[i].y() - points[i - 1].y();
        s[i] = s[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    return s;
}

// Resample a polyline at uniform arclength and Gaussian-smooth it. The line
// points wander around the fiber's true run (placement noise plus interpolated
// stretches); smoothing in arclength keeps the low-frequency shape and is
// independent of the very uneven raw point spacing. Returns the arclength grid
// alongside the smoothed points so positions at any raw arclength can be read
// back and land exactly on the drawn curve.
void smoothPolyline(const std::vector<QPointF>& points, double sigma, double step,
                    std::vector<double>& sOut, std::vector<QPointF>& qOut)
{
    const std::vector<double> sRaw = arclengths(points);
    const double total = sRaw.empty() ? 0.0 : sRaw.back();
    // The unsmoothed fallback also covers parameters that cannot be resampled
    // sanely: non-finite lengths, and steps so small the sample count would
    // not fit an int (a public parameter must not be able to reach the
    // undefined float-to-int conversion or an absurd allocation).
    constexpr double kMaxSamples = 16.0 * 1024.0 * 1024.0;
    if (!(total >= 2.0 * step) || points.size() < 3 ||
        !std::isfinite(total) || !std::isfinite(sigma) ||
        total / step > kMaxSamples) {
        sOut = sRaw;
        qOut = points;
        return;
    }

    const int count = static_cast<int>(std::ceil(total / step + 0.5));
    sOut.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        sOut[static_cast<std::size_t>(i)] = static_cast<double>(i) * step;
    }
    sOut.back() = total;

    std::vector<double> xs(points.size());
    std::vector<double> ys(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        xs[i] = points[i].x();
        ys[i] = points[i].y();
    }
    qOut.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double s = sOut[static_cast<std::size_t>(i)];
        qOut[static_cast<std::size_t>(i)] =
            QPointF(interpolate(s, sRaw, xs), interpolate(s, sRaw, ys));
    }
    if (sigma <= 0.0) {
        return;
    }

    const int radius = std::clamp(
        static_cast<int>(std::nearbyint(std::min(3.0 * sigma / step, 4096.0))),
        1, 4096);
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double kernelSum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double t = static_cast<double>(i) * step / sigma;
        const double weight = std::exp(-0.5 * t * t);
        kernel[static_cast<std::size_t>(i + radius)] = weight;
        kernelSum += weight;
    }
    for (double& weight : kernel) {
        weight /= kernelSum;
    }

    // Reflect-pad so the ends do not shrink toward the interior. The mirror
    // index is clamped for curves shorter than the kernel radius.
    const int last = count - 1;
    const auto padded = [&](int index) {
        if (index < 0) {
            const int mirror = std::min(-index, last);
            return QPointF(2.0 * qOut[0].x() - qOut[static_cast<std::size_t>(mirror)].x(),
                           2.0 * qOut[0].y() - qOut[static_cast<std::size_t>(mirror)].y());
        }
        if (index > last) {
            const int mirror = std::max(last - (index - last), 0);
            return QPointF(2.0 * qOut[static_cast<std::size_t>(last)].x() -
                               qOut[static_cast<std::size_t>(mirror)].x(),
                           2.0 * qOut[static_cast<std::size_t>(last)].y() -
                               qOut[static_cast<std::size_t>(mirror)].y());
        }
        return qOut[static_cast<std::size_t>(index)];
    };
    std::vector<QPointF> smoothed(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        double x = 0.0;
        double y = 0.0;
        for (int d = -radius; d <= radius; ++d) {
            const QPointF point = padded(i + d);
            const double weight = kernel[static_cast<std::size_t>(d + radius)];
            x += weight * point.x();
            y += weight * point.y();
        }
        smoothed[static_cast<std::size_t>(i)] = QPointF(x, y);
    }
    qOut = std::move(smoothed);
}

std::size_t searchSortedLeft(const std::vector<double>& values, double query)
{
    return static_cast<std::size_t>(
        std::lower_bound(values.begin(), values.end(), query) - values.begin());
}

std::size_t searchSortedRight(const std::vector<double>& values, double query)
{
    return static_cast<std::size_t>(
        std::upper_bound(values.begin(), values.end(), query) - values.begin());
}

// The umbilicus as three z-sorted interpolation tables.
struct UmbilicusInterp {
    std::vector<double> z;
    std::vector<double> x;
    std::vector<double> y;
};

UmbilicusInterp interpolateUmbilicus(const std::vector<cv::Vec3f>& umbilicusCenters)
{
    std::vector<cv::Vec3f> centers = umbilicusCenters;
    std::stable_sort(centers.begin(), centers.end(),
                     [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });
    UmbilicusInterp interp;
    interp.z.resize(centers.size());
    interp.x.resize(centers.size());
    interp.y.resize(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i) {
        interp.z[i] = centers[i][2];
        interp.x[i] = centers[i][0];
        interp.y[i] = centers[i][1];
    }
    return interp;
}

// A fiber unrolled about the umbilicus; angles are fiber-local until the
// whole-turn offset is applied.
struct PreparedFiber {
    const InputFiber* input = nullptr;
    std::vector<double> thetaLine;
    std::vector<double> radius;
    std::vector<std::size_t> controlLineIndex;
    double offset = 0.0;

    double thetaAt(int controlIndex) const
    {
        return thetaLine[controlLineIndex[static_cast<std::size_t>(controlIndex)]];
    }
    double placedThetaAt(int controlIndex) const
    {
        return thetaAt(controlIndex) + offset;
    }
};

PreparedFiber prepareFiber(const InputFiber& fiber, const UmbilicusInterp& umbilicus)
{
    PreparedFiber entry;
    entry.input = &fiber;
    std::vector<double> raw(fiber.linePoints.size());
    entry.radius.resize(fiber.linePoints.size());
    for (std::size_t i = 0; i < fiber.linePoints.size(); ++i) {
        const cv::Vec3d& point = fiber.linePoints[i];
        const double dx = point[0] - interpolate(point[2], umbilicus.z, umbilicus.x);
        const double dy = point[1] - interpolate(point[2], umbilicus.z, umbilicus.y);
        raw[i] = std::atan2(dy, dx);
        entry.radius[i] = std::sqrt(dx * dx + dy * dy);
    }
    entry.thetaLine = unwrapAngles(raw);
    entry.controlLineIndex.resize(fiber.controlPoints.size());
    for (std::size_t i = 0; i < fiber.controlPoints.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t bestIndex = 0;
        for (std::size_t j = 0; j < fiber.linePoints.size(); ++j) {
            const cv::Vec3d delta = fiber.linePoints[j] - fiber.controlPoints[i];
            const double distance = delta.dot(delta);
            if (distance < best) {
                best = distance;
                bestIndex = j;
            }
        }
        entry.controlLineIndex[i] = bestIndex;
    }
    return entry;
}

struct LinkRecord {
    std::size_t a = 0;
    int ia = -1;
    std::size_t b = 0;
    int ib = -1;
    double turnErr = 0.0;
    bool pending = false;
};

struct HeapEntry {
    double frac = 0.0;
    std::size_t link = 0;
    std::size_t from = 0;
    std::size_t to = 0;
    double offset = 0.0;

    bool operator>(const HeapEntry& other) const
    {
        if (frac != other.frac) {
            return frac > other.frac;
        }
        return link > other.link;
    }
};

struct FiberGeometry {
    std::vector<double> sampleArclength;
    std::vector<QPointF> samples;
    std::vector<double> controlArclength;
    std::vector<QPointF> controlPoints;
};

struct NetworkDraft {
    int networkIndex = 0;
    double rRefVx = 0.0;
    std::vector<PlacedFiber> fibers;
    std::vector<PlacedLink> links;
    double loXVx = 0.0;
    double hiXVx = 0.0;
    double loYVx = 0.0;
    double hiYVx = 0.0;
};

// Fibers without geometry cannot be unrolled, so they take no part in the
// link graph either; both entry points share one notion of "placeable" and
// one deterministic order.
std::vector<const InputFiber*> orderPlaceableFibers(const std::vector<InputFiber>& fibers)
{
    std::vector<const InputFiber*> ordered;
    ordered.reserve(fibers.size());
    for (const InputFiber& fiber : fibers) {
        if (!fiber.controlPoints.empty() && !fiber.linePoints.empty()) {
            ordered.push_back(&fiber);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const InputFiber* a, const InputFiber* b) {
                  if (a->label != b->label) {
                      return a->label < b->label;
                  }
                  return a->id < b->id;
              });
    return ordered;
}

// Links validated and deduped once, over every placeable fiber. Deduped by
// their sorted endpoint pair: the reciprocal ref of an already-seen crossing
// is the same physical link, and a half-updated reciprocal pair still reads
// as pending. Validation matters beyond placement: a link naming a control
// point that does not exist cannot position anything, but joining on it
// anyway pulled an otherwise unconnected fiber into the component - it then
// counted towards minFibers, skewed the network's median radius, and was
// drawn at its default zero turn offset as though it were linked to
// something.
std::vector<LinkRecord> collectValidLinks(
    const std::vector<const InputFiber*>& ordered,
    const std::unordered_map<uint64_t, std::size_t>& indexById)
{
    std::vector<LinkRecord> links;
    std::map<std::pair<std::pair<std::size_t, int>, std::pair<std::size_t, int>>,
             std::size_t>
        seen;
    for (std::size_t member = 0; member < ordered.size(); ++member) {
        const InputFiber& fiber = *ordered[member];
        const int controlCount = static_cast<int>(fiber.controlPoints.size());
        for (const InputLink& link : fiber.links) {
            const auto target = indexById.find(link.branchFiberId);
            if (target == indexById.end()) {
                continue;
            }
            const std::size_t other = target->second;
            const int otherCount =
                static_cast<int>(ordered[other]->controlPoints.size());
            const int ia = link.controlPointIndex;
            const int ib = link.branchControlPointIndex;
            if (ia < 0 || ia >= controlCount || ib < 0 || ib >= otherCount) {
                qWarning() << "fiber map: link" << fiber.label << ia << "->"
                           << ordered[other]->label << ib << "out of range; skipped";
                continue;
            }
            const std::pair<std::size_t, int> here{member, ia};
            const std::pair<std::size_t, int> there{other, ib};
            const auto inserted =
                seen.emplace(here < there ? std::make_pair(here, there)
                                          : std::make_pair(there, here),
                             links.size());
            if (!inserted.second) {
                links[inserted.first->second].pending |= link.pending;
                continue;
            }
            links.push_back(LinkRecord{member, ia, other, ib, 0.0, link.pending});
        }
    }
    std::sort(links.begin(), links.end(),
              [](const LinkRecord& a, const LinkRecord& b) {
                  return std::tie(a.a, a.ia, a.b, a.ib) <
                         std::tie(b.a, b.ia, b.b, b.ib);
              });
    return links;
}

// Snap each fiber's whole-turn offset to its neighbours, growing the link
// tree Prim-style from the best-agreeing link so one wrong-winding link
// cannot decide a fiber's offset when a clean link to the same fiber exists.
// Writes the offsets into `prepared` and each link's residual turn error into
// `links`.
void snapComponentOffsets(const std::vector<std::size_t>& component,
                          std::vector<LinkRecord>& links,
                          std::unordered_map<std::size_t, PreparedFiber>& prepared)
{
    std::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> adjacency;
    adjacency.reserve(component.size());
    for (std::size_t li = 0; li < links.size(); ++li) {
        adjacency[links[li].a].emplace_back(links[li].b, li);
        adjacency[links[li].b].emplace_back(links[li].a, li);
    }
    std::size_t root = component.front();
    std::size_t rootDegree = 0;
    for (const std::size_t member : component) {
        const auto entry = adjacency.find(member);
        const std::size_t degree = entry == adjacency.end() ? 0 : entry->second.size();
        if (degree >= rootDegree) {
            rootDegree = degree;
            root = member;
        }
    }
    std::set<std::size_t> placed{root};
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    const auto pushEdges = [&](std::size_t from) {
        const auto entry = adjacency.find(from);
        if (entry == adjacency.end()) {
            return;
        }
        for (const auto& [to, li] : entry->second) {
            if (placed.count(to) != 0) {
                continue;
            }
            const LinkRecord& link = links[li];
            const int here = link.a == from ? link.ia : link.ib;
            const int there = link.a == from ? link.ib : link.ia;
            const double thetaHere = prepared.at(from).placedThetaAt(here);
            const double thetaThere = prepared.at(to).thetaAt(there);
            const double offset =
                roundTurns((thetaHere - thetaThere) / kTwoPi) * kTwoPi;
            const double frac =
                std::fabs(thetaHere - (thetaThere + offset)) / kTwoPi;
            heap.push(HeapEntry{frac, li, from, to, offset});
        }
    };
    pushEdges(root);
    while (!heap.empty()) {
        const HeapEntry entry = heap.top();
        heap.pop();
        if (placed.count(entry.to) != 0) {
            continue;
        }
        prepared.at(entry.to).offset = entry.offset;
        placed.insert(entry.to);
        pushEdges(entry.to);
    }

    for (LinkRecord& link : links) {
        const double thetaA = prepared.at(link.a).placedThetaAt(link.ia);
        const double thetaB = prepared.at(link.b).placedThetaAt(link.ib);
        link.turnErr = std::fabs(thetaA - thetaB) / kTwoPi;
    }
}

// Unroll one fiber at x = (thetaScale * theta + offsetRad) * rRef, y = z,
// smooth and resample it, read the control points off the smoothed curve, and
// clip to the control span. line_points overshoot the outermost control
// points by over a cm on many fibers; those tails carry no segment metadata
// and are not drawn, so they are clipped out of the geometry entirely --
// otherwise label anchors and the extents would be computed from invisible
// curve.
FiberGeometry buildFiberGeometry(const PreparedFiber& entry, double thetaScale,
                                 double offsetRad, double rRefVx, double sigmaVx,
                                 double resampleStepVx)
{
    const InputFiber& fiber = *entry.input;
    std::vector<QPointF> unrolled(fiber.linePoints.size());
    for (std::size_t i = 0; i < fiber.linePoints.size(); ++i) {
        unrolled[i] = QPointF((thetaScale * entry.thetaLine[i] + offsetRad) * rRefVx,
                              fiber.linePoints[i][2]);
    }

    FiberGeometry geo;
    const std::vector<double> rawArclength = arclengths(unrolled);
    smoothPolyline(unrolled, sigmaVx, resampleStepVx, geo.sampleArclength,
                   geo.samples);
    std::vector<double> sampleX(geo.samples.size());
    std::vector<double> sampleY(geo.samples.size());
    for (std::size_t i = 0; i < geo.samples.size(); ++i) {
        sampleX[i] = geo.samples[i].x();
        sampleY[i] = geo.samples[i].y();
    }
    geo.controlArclength.resize(entry.controlLineIndex.size());
    geo.controlPoints.resize(entry.controlLineIndex.size());
    for (std::size_t i = 0; i < entry.controlLineIndex.size(); ++i) {
        const double s = rawArclength[entry.controlLineIndex[i]];
        geo.controlArclength[i] = s;
        geo.controlPoints[i] =
            QPointF(interpolate(s, geo.sampleArclength, sampleX),
                    interpolate(s, geo.sampleArclength, sampleY));
    }

    // Extremes, not front/back: a malformed fiber whose control points map
    // backwards along the line points must clip to a valid (possibly whole)
    // range rather than erase past the end of a shortened vector.
    const auto [minArc, maxArc] = std::minmax_element(
        geo.controlArclength.begin(), geo.controlArclength.end());
    const std::size_t begin = searchSortedLeft(geo.sampleArclength, *minArc);
    const std::size_t end = searchSortedRight(geo.sampleArclength, *maxArc);
    const std::size_t clipBegin = begin > 0 ? begin - 1 : 0;
    const std::size_t clipEnd = std::min(geo.samples.size(), end + 1);
    geo.sampleArclength.erase(geo.sampleArclength.begin() +
                                  static_cast<std::ptrdiff_t>(clipEnd),
                              geo.sampleArclength.end());
    geo.sampleArclength.erase(geo.sampleArclength.begin(),
                              geo.sampleArclength.begin() +
                                  static_cast<std::ptrdiff_t>(clipBegin));
    geo.samples.erase(geo.samples.begin() +
                          static_cast<std::ptrdiff_t>(clipEnd),
                      geo.samples.end());
    geo.samples.erase(geo.samples.begin(),
                      geo.samples.begin() +
                          static_cast<std::ptrdiff_t>(clipBegin));
    return geo;
}

// Traced runs draw solid, thick and vivid; segments that are only
// interpolations draw thin, dashed and faded -- "dashed = not real trace
// data" at a glance.
PlacedFiber makePlacedFiber(const InputFiber& fiber, const FiberGeometry& geo)
{
    PlacedFiber placedFiber;
    placedFiber.id = fiber.id;
    placedFiber.fileName = fiber.fileName;
    placedFiber.label = fiber.label;
    placedFiber.hvTag = fiber.hvTag;
    placedFiber.controlPoints = geo.controlPoints;

    const std::size_t spanCount =
        fiber.controlPoints.empty() ? 0 : fiber.controlPoints.size() - 1;
    const bool haveFlags = spanCount > 0 &&
                           fiber.tracedSegments.size() == spanCount;
    if (!haveFlags) {
        if (geo.samples.size() > 1) {
            placedFiber.runs.push_back(Run{true, geo.samples});
        }
    } else {
        std::size_t k = 0;
        while (k < spanCount) {
            std::size_t j = k;
            while (j + 1 < spanCount &&
                   fiber.tracedSegments[j + 1] == fiber.tracedSegments[k]) {
                ++j;
            }
            const std::size_t begin = searchSortedLeft(
                geo.sampleArclength, geo.controlArclength[k]);
            const std::size_t end = searchSortedRight(
                geo.sampleArclength, geo.controlArclength[j + 1]);
            const std::size_t from = begin > 0 ? begin - 1 : 0;
            const std::size_t to = std::min(geo.samples.size(), end + 1);
            if (to > from + 1) {
                placedFiber.runs.push_back(Run{
                    fiber.tracedSegments[k],
                    std::vector<QPointF>(
                        geo.samples.begin() + static_cast<std::ptrdiff_t>(from),
                        geo.samples.begin() + static_cast<std::ptrdiff_t>(to))});
            }
            k = j + 1;
        }
    }
    return placedFiber;
}

} // namespace

Result buildLayout(const std::vector<InputFiber>& fibers,
                   const std::vector<cv::Vec3f>& umbilicusCenters,
                   const LayoutParams& params)
{
    Result result;
    if (umbilicusCenters.empty()) {
        return result;
    }
    const UmbilicusInterp umbilicus = interpolateUmbilicus(umbilicusCenters);

    const std::vector<const InputFiber*> ordered = orderPlaceableFibers(fibers);
    const std::size_t fiberCount = ordered.size();
    if (fiberCount == 0) {
        return result;
    }

    std::unordered_map<uint64_t, std::size_t> indexById;
    indexById.reserve(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        indexById.emplace(ordered[i]->id, i);
    }

    const std::vector<LinkRecord> allLinks = collectValidLinks(ordered, indexById);

    std::vector<std::size_t> parent(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        parent[i] = i;
    }
    const auto findRoot = [&parent](std::size_t index) {
        while (parent[index] != index) {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    };
    for (const LinkRecord& link : allLinks) {
        const std::size_t a = findRoot(link.a);
        const std::size_t b = findRoot(link.b);
        if (a != b) {
            parent[a] = b;
        }
    }
    std::unordered_map<std::size_t, std::vector<std::size_t>> componentsByRoot;
    for (std::size_t i = 0; i < fiberCount; ++i) {
        componentsByRoot[findRoot(i)].push_back(i);
    }
    std::vector<std::vector<std::size_t>> components;
    components.reserve(componentsByRoot.size());
    for (auto& entry : componentsByRoot) {
        components.push_back(std::move(entry.second));
    }
    for (auto& component : components) {
        std::sort(component.begin(), component.end());
    }
    std::sort(components.begin(), components.end(),
              [](const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
                  if (a.size() != b.size()) {
                      return a.size() > b.size();
                  }
                  return a.front() < b.front();
              });

    // Voxels throughout: line points, radii and every tuning length arrive in
    // the same unit, so there is nothing to convert.
    const double sigmaVx = std::max(0.0, params.smoothVx);
    // A step of zero would resample forever, so it falls back to the documented
    // default rather than to a literal repeated from the header.
    const double resampleStepVx = params.resampleStepVx > 0.0
        ? params.resampleStepVx
        : LayoutParams{}.resampleStepVx;
    const int minFibers = std::max(1, params.minFibers);

    std::vector<NetworkDraft> drafts;
    int networkIndex = -1;
    for (const std::vector<std::size_t>& component : components) {
        if (component.size() < static_cast<std::size_t>(minFibers)) {
            continue;
        }
        ++networkIndex;
        ++result.qualifyingNetworkCount;
        if (static_cast<int>(drafts.size()) >= std::max(0, params.maxNetworks)) {
            continue;
        }

        std::unordered_map<std::size_t, PreparedFiber> prepared;
        prepared.reserve(component.size());
        for (const std::size_t member : component) {
            prepared.emplace(member, prepareFiber(*ordered[member], umbilicus));
        }

        // This component's slice of the validated links; both endpoints are
        // members by construction (links are what defined the components).
        const std::size_t componentRoot = findRoot(component.front());
        std::vector<LinkRecord> links;
        for (const LinkRecord& link : allLinks) {
            if (findRoot(link.a) == componentRoot) {
                links.push_back(link);
            }
        }
        if (links.empty()) {
            continue;
        }

        snapComponentOffsets(component, links, prepared);

        // Centre the component on a whole turn so the winding numbers stay
        // small, and take the reference radius from the crossings.
        std::vector<double> controlThetas;
        std::vector<double> controlRadii;
        for (const std::size_t member : component) {
            const PreparedFiber& entry = prepared.at(member);
            for (std::size_t i = 0; i < entry.controlLineIndex.size(); ++i) {
                controlThetas.push_back(entry.thetaLine[entry.controlLineIndex[i]] +
                                        entry.offset);
                controlRadii.push_back(entry.radius[entry.controlLineIndex[i]]);
            }
        }
        const double shiftTurns = roundTurns(median(controlThetas) / kTwoPi) * kTwoPi;
        for (const std::size_t member : component) {
            prepared.at(member).offset -= shiftTurns;
        }
        const double rRefVx = median(controlRadii);

        NetworkDraft draft;
        draft.networkIndex = networkIndex;
        draft.rRefVx = rRefVx;
        draft.loXVx = std::numeric_limits<double>::infinity();
        draft.hiXVx = -std::numeric_limits<double>::infinity();
        draft.loYVx = std::numeric_limits<double>::infinity();
        draft.hiYVx = -std::numeric_limits<double>::infinity();

        std::unordered_map<std::size_t, FiberGeometry> geometry;
        geometry.reserve(component.size());
        for (const std::size_t member : component) {
            const PreparedFiber& entry = prepared.at(member);
            FiberGeometry geo = buildFiberGeometry(entry, 1.0, entry.offset,
                                                   rRefVx, sigmaVx, resampleStepVx);
            for (const QPointF& point : geo.samples) {
                draft.loXVx = std::min(draft.loXVx, point.x());
                draft.hiXVx = std::max(draft.hiXVx, point.x());
                draft.loYVx = std::min(draft.loYVx, point.y());
                draft.hiYVx = std::max(draft.hiYVx, point.y());
            }
            geometry.emplace(member, std::move(geo));
        }
        if (!(draft.loXVx <= draft.hiXVx) || !(draft.loYVx <= draft.hiYVx)) {
            continue;
        }

        for (const std::size_t member : component) {
            draft.fibers.push_back(makePlacedFiber(*prepared.at(member).input,
                                                   geometry.at(member)));
        }

        for (const LinkRecord& link : links) {
            PlacedLink placedLink;
            placedLink.fiberA = ordered[link.a]->id;
            placedLink.cpA = link.ia;
            placedLink.fiberB = ordered[link.b]->id;
            placedLink.cpB = link.ib;
            placedLink.a = geometry.at(link.a)
                               .controlPoints[static_cast<std::size_t>(link.ia)];
            placedLink.b = geometry.at(link.b)
                               .controlPoints[static_cast<std::size_t>(link.ib)];
            placedLink.turnErr = link.turnErr;
            placedLink.pending = link.pending;
            placedLink.suspect = link.turnErr > params.suspectTurns;
            if (placedLink.suspect) {
                ++result.suspectLinkCount;
            }
            draft.links.push_back(std::move(placedLink));
        }

        const double padX =
            std::max(kPadFraction * (draft.hiXVx - draft.loXVx), params.minPadXVx);
        const double padY =
            std::max(kPadFraction * (draft.hiYVx - draft.loYVx), params.minPadYVx);
        draft.loXVx -= padX;
        draft.hiXVx += padX;
        draft.loYVx -= padY;
        draft.hiYVx += padY;
        drafts.push_back(std::move(draft));
    }
    if (drafts.empty()) {
        return result;
    }

    // Panels ordered inner -> outer by median distance from the umbilicus.
    // Unrolled length starts at 0 on the left and runs continuously through
    // every panel: each next panel starts on the global tick grid, so every
    // panel shows the same labeling interval, and the gap between networks is
    // whatever that snap requires (at least minGapVx).
    std::stable_sort(drafts.begin(), drafts.end(),
                     [](const NetworkDraft& a, const NetworkDraft& b) {
                         return a.rRefVx < b.rRefVx;
                     });

    const double tickVx = params.panelTickVx > 0.0 ? params.panelTickVx
                                                  : LayoutParams{}.panelTickVx;
    double panelStart = 0.0;
    int windingNumber = 0;
    result.yMinVx = std::numeric_limits<double>::infinity();
    result.yMaxVx = -std::numeric_limits<double>::infinity();
    result.networks.reserve(drafts.size());
    for (NetworkDraft& draft : drafts) {
        const double width = draft.hiXVx - draft.loXVx;
        const double shift = panelStart - draft.loXVx;

        PlacedNetwork network;
        network.networkIndex = draft.networkIndex;
        network.rRefVx = draft.rRefVx;
        network.x0Vx = panelStart;
        network.x1Vx = panelStart + width;
        network.fibers = std::move(draft.fibers);
        network.links = std::move(draft.links);
        for (PlacedFiber& fiber : network.fibers) {
            for (Run& run : fiber.runs) {
                for (QPointF& point : run.points) {
                    point.setX(point.x() + shift);
                }
            }
            for (QPointF& point : fiber.controlPoints) {
                point.setX(point.x() + shift);
            }
        }
        for (PlacedLink& link : network.links) {
            link.a.setX(link.a.x() + shift);
            link.b.setX(link.b.x() + shift);
        }

        const double circumference = kTwoPi * draft.rRefVx;
        if (circumference > 0.0) {
            const long long first =
                static_cast<long long>(std::ceil(draft.loXVx / circumference));
            const long long lastMark =
                static_cast<long long>(std::floor(draft.hiXVx / circumference));
            for (long long mark = first; mark <= lastMark; ++mark) {
                network.windings.push_back(WindingMark{
                    static_cast<double>(mark) * circumference + shift, windingNumber});
                ++windingNumber;
            }
        }

        result.yMinVx = std::min(result.yMinVx, draft.loYVx);
        result.yMaxVx = std::max(result.yMaxVx, draft.hiYVx);
        result.widthVx = panelStart + width;
        result.networks.push_back(std::move(network));
        panelStart = tickVx * std::ceil((panelStart + width + params.minGapVx) / tickVx);
    }
    return result;
}

GlobalResult buildGlobalLayout(const std::vector<InputFiber>& fibers,
                               const std::vector<cv::Vec3f>& umbilicusCenters,
                               const GlobalLayoutParams& params)
{
    GlobalResult result;
    const auto sortUnplaced = [&result]() {
        std::sort(result.unplaced.begin(), result.unplaced.end(),
                  [](const UnplacedFiber& a, const UnplacedFiber& b) {
                      if (a.label != b.label) {
                          return a.label < b.label;
                      }
                      return a.id < b.id;
                  });
    };
    if (umbilicusCenters.empty()) {
        // Nothing can be unrolled, and "every fiber" still has to hold: the
        // whole input is unplaceable, not silently absent.
        for (const InputFiber& fiber : fibers) {
            result.unplaced.push_back(
                UnplacedFiber{fiber.id, fiber.fileName, fiber.label, fiber.hvTag});
        }
        sortUnplaced();
        return result;
    }
    for (const InputFiber& fiber : fibers) {
        if (fiber.controlPoints.empty() || fiber.linePoints.empty()) {
            result.unplaced.push_back(
                UnplacedFiber{fiber.id, fiber.fileName, fiber.label, fiber.hvTag});
        }
    }
    const UmbilicusInterp umbilicus = interpolateUmbilicus(umbilicusCenters);

    const std::vector<const InputFiber*> ordered = orderPlaceableFibers(fibers);
    const std::size_t fiberCount = ordered.size();
    if (fiberCount == 0) {
        sortUnplaced();
        return result;
    }
    std::unordered_map<uint64_t, std::size_t> indexById;
    indexById.reserve(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        indexById.emplace(ordered[i]->id, i);
    }

    std::vector<PreparedFiber> prepared;
    prepared.reserve(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        prepared.push_back(prepareFiber(*ordered[i], umbilicus));
    }

    // The solver sees only the control-point-bounded domain of every fiber:
    // the undrawn line-point tails must not constrain the solve any more than
    // they may set the drawn extents.
    std::vector<std::size_t> domainBegin(fiberCount, 0);
    std::vector<winding::FiberTrace> traces(fiberCount);
    std::vector<double> allRadii;
    for (std::size_t i = 0; i < fiberCount; ++i) {
        const PreparedFiber& entry = prepared[i];
        const auto [minIt, maxIt] = std::minmax_element(
            entry.controlLineIndex.begin(), entry.controlLineIndex.end());
        // One sample beyond each outer control, matching the drawn clip -
        // and keeping linked crossings, which sit exactly on the outermost
        // controls, interior to the trace instead of on an fp-fragile edge.
        const std::size_t begin = *minIt > 0 ? *minIt - 1 : 0;
        const std::size_t end =
            std::min(*maxIt + 1, entry.thetaLine.size() - 1);
        domainBegin[i] = begin;
        winding::FiberTrace& trace = traces[i];
        trace.hvTag = ordered[i]->hvTag;
        // One model-traced span trusts the whole fiber; a fiber with none is
        // control-point interpolation and must never be declared a winding
        // error. Empty flags get the benefit of the doubt, exactly as the
        // drawing renders them (a single traced run).
        const std::vector<bool>& tracedFlags = ordered[i]->tracedSegments;
        trace.trusted = tracedFlags.empty() ||
                        std::any_of(tracedFlags.begin(), tracedFlags.end(),
                                    [](bool traced) { return traced; });
        trace.theta.assign(entry.thetaLine.begin() + static_cast<std::ptrdiff_t>(begin),
                           entry.thetaLine.begin() + static_cast<std::ptrdiff_t>(end) + 1);
        trace.radius.assign(entry.radius.begin() + static_cast<std::ptrdiff_t>(begin),
                            entry.radius.begin() + static_cast<std::ptrdiff_t>(end) + 1);
        trace.z.reserve(end - begin + 1);
        for (std::size_t j = begin; j <= end; ++j) {
            trace.z.push_back(ordered[i]->linePoints[j][2]);
        }
        allRadii.insert(allRadii.end(), trace.radius.begin(), trace.radius.end());
    }

    const std::vector<LinkRecord> allLinks = collectValidLinks(ordered, indexById);

    // Linked-network membership, for the dock's grouping and the map's
    // network co-highlight: components of the manual link graph, numbered by
    // size descending. Singletons keep -1.
    std::vector<int> networkIdOf(fiberCount, -1);
    std::vector<int> networkSizeOf(fiberCount, 1);
    {
        std::vector<std::size_t> parent(fiberCount);
        for (std::size_t i = 0; i < fiberCount; ++i) {
            parent[i] = i;
        }
        const auto findRoot = [&parent](std::size_t index) {
            while (parent[index] != index) {
                parent[index] = parent[parent[index]];
                index = parent[index];
            }
            return index;
        };
        for (const LinkRecord& link : allLinks) {
            const std::size_t a = findRoot(link.a);
            const std::size_t b = findRoot(link.b);
            if (a != b) {
                parent[a] = b;
            }
        }
        std::map<std::size_t, std::vector<std::size_t>> byRoot;
        for (std::size_t i = 0; i < fiberCount; ++i) {
            byRoot[findRoot(i)].push_back(i);
        }
        std::vector<std::vector<std::size_t>> networks;
        for (auto& entry : byRoot) {
            if (entry.second.size() > 1) {
                networks.push_back(std::move(entry.second));
            }
        }
        // ordered[] is already (label, id)-sorted, so front() is each
        // network's first fiber by label.
        std::sort(networks.begin(), networks.end(),
                  [](const std::vector<std::size_t>& a,
                     const std::vector<std::size_t>& b) {
                      if (a.size() != b.size()) {
                          return a.size() > b.size();
                      }
                      return a.front() < b.front();
                  });
        for (std::size_t n = 0; n < networks.size(); ++n) {
            for (const std::size_t member : networks[n]) {
                networkIdOf[member] = static_cast<int>(n);
                networkSizeOf[member] = static_cast<int>(networks[n].size());
            }
        }
    }

    std::vector<winding::LinkInput> linkInputs;
    linkInputs.reserve(allLinks.size());
    for (const LinkRecord& link : allLinks) {
        linkInputs.push_back(winding::LinkInput{
            link.a,
            prepared[link.a].controlLineIndex[static_cast<std::size_t>(link.ia)] -
                domainBegin[link.a],
            link.b,
            prepared[link.b].controlLineIndex[static_cast<std::size_t>(link.ib)] -
                domainBegin[link.b]});
    }

    winding::SolverParams solverParams = params.solver;
    // One suspicion threshold: the confidence a link solves with and the
    // suspicion it is reported with must never disagree.
    solverParams.linkSuspectTurns = params.suspectTurns;
    const winding::SolveResult solve =
        winding::solveWindings(traces, linkInputs, solverParams);
    result.chirality = solve.chirality;
    result.islandCount = solve.islandCount;
    result.unresolvedCount = solve.unresolvedCount;
    result.tieCount = solve.tieCount;
    result.gatedSegmentCount = solve.gatedSegmentCount;
    result.tangentialCount = solve.tangentialCount;

    // One reference radius for the whole map. It is a display scale, never
    // evidence, so the median over everything is enough.
    double rRefVx = median(std::move(allRadii));
    if (!(rRefVx > 0.0)) {
        rRefVx = 1.0;
    }
    result.rRefVx = rRefVx;

    const double sigmaVx = std::max(0.0, params.smoothVx);
    const double resampleStepVx = params.resampleStepVx > 0.0
        ? params.resampleStepVx
        : GlobalLayoutParams{}.resampleStepVx;
    const double thetaScale = static_cast<double>(solve.chirality);

    double loX = std::numeric_limits<double>::infinity();
    double hiX = -std::numeric_limits<double>::infinity();
    double loY = std::numeric_limits<double>::infinity();
    double hiY = -std::numeric_limits<double>::infinity();
    std::vector<FiberGeometry> geometry;
    std::vector<char> drawable(fiberCount, 0);
    geometry.reserve(fiberCount);
    result.fibers.reserve(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        const winding::Placement& placement = solve.placements[i];
        const double offsetRad = kTwoPi * placement.turns;
        FiberGeometry geo = buildFiberGeometry(prepared[i], thetaScale, offsetRad,
                                               rRefVx, sigmaVx, resampleStepVx);
        GlobalPlacedFiber placed;
        placed.fiber = makePlacedFiber(*ordered[i], geo);
        // Geometry too degenerate to draw a single run (a one-point trace,
        // say) or containing non-finite coordinates is unplaceable, honestly,
        // rather than a placed fiber the map never shows. Note the fiber's
        // trace has already informed the winding solve by this point; that is
        // deliberate - the annotation geometry is real even when it cannot be
        // drawn.
        double fiberLoX = std::numeric_limits<double>::infinity();
        double fiberHiX = -std::numeric_limits<double>::infinity();
        double fiberLoY = std::numeric_limits<double>::infinity();
        double fiberHiY = -std::numeric_limits<double>::infinity();
        for (const QPointF& point : geo.samples) {
            fiberLoX = std::min(fiberLoX, point.x());
            fiberHiX = std::max(fiberHiX, point.x());
            fiberLoY = std::min(fiberLoY, point.y());
            fiberHiY = std::max(fiberHiY, point.y());
        }
        if (placed.fiber.runs.empty() || !std::isfinite(fiberLoX) ||
            !std::isfinite(fiberHiX) || !std::isfinite(fiberLoY) ||
            !std::isfinite(fiberHiY)) {
            result.unplaced.push_back(UnplacedFiber{ordered[i]->id,
                                                    ordered[i]->fileName,
                                                    ordered[i]->label,
                                                    ordered[i]->hvTag});
            geometry.push_back(std::move(geo));
            continue;
        }
        drawable[i] = 1;
        loX = std::min(loX, fiberLoX);
        hiX = std::max(hiX, fiberHiX);
        loY = std::min(loY, fiberLoY);
        hiY = std::max(hiY, fiberHiY);
        placed.meta.linked = placement.linked;
        placed.meta.networkId = networkIdOf[i];
        placed.meta.networkSize = networkSizeOf[i];
        placed.meta.sheetDriftSuspect = placement.sheetDriftSuspect;
        placed.meta.windingLo = placement.windingLo;
        placed.meta.windingHi = placement.windingHi;
        switch (placement.anchor) {
        case winding::ComponentAnchor::Primary:
            placed.meta.anchor = GlobalAnchor::Primary;
            break;
        case winding::ComponentAnchor::Radius:
            placed.meta.anchor = GlobalAnchor::Radius;
            break;
        case winding::ComponentAnchor::AmbiguousRadius:
            placed.meta.anchor = GlobalAnchor::AmbiguousRadius;
            break;
        case winding::ComponentAnchor::Unresolved:
            placed.meta.anchor = GlobalAnchor::Unresolved;
            break;
        }
        result.fibers.push_back(std::move(placed));
        geometry.push_back(std::move(geo));
    }
    if (!(loX <= hiX) || !(loY <= hiY)) {
        // Nothing drew at all; the accounting still owes the caller every
        // fiber it was about to place.
        for (const GlobalPlacedFiber& placed : result.fibers) {
            result.unplaced.push_back(UnplacedFiber{placed.fiber.id,
                                                    placed.fiber.fileName,
                                                    placed.fiber.label,
                                                    placed.fiber.hvTag});
        }
        result.fibers.clear();
        sortUnplaced();
        return result;
    }

    std::set<std::size_t> droppedLinks(solve.droppedLinks.begin(),
                                       solve.droppedLinks.end());
    result.links.reserve(allLinks.size());
    for (std::size_t l = 0; l < allLinks.size(); ++l) {
        const LinkRecord& link = allLinks[l];
        if (!drawable[link.a] || !drawable[link.b]) {
            continue;
        }
        PlacedLink placedLink;
        placedLink.fiberA = ordered[link.a]->id;
        placedLink.cpA = link.ia;
        placedLink.fiberB = ordered[link.b]->id;
        placedLink.cpB = link.ib;
        placedLink.a =
            geometry[link.a].controlPoints[static_cast<std::size_t>(link.ia)];
        placedLink.b =
            geometry[link.b].controlPoints[static_cast<std::size_t>(link.ib)];
        placedLink.turnErr = solve.linkTurnErrors[l];
        placedLink.pending = link.pending;
        // A link the repair had to drop is winding-suspect whatever its
        // residual now reads: the map placed its endpoints against it. The
        // boundary is inclusive because a residual AT the threshold already
        // solves with zero confidence. Suspicion requires model-traced
        // geometry on both ends - the residual of a link into an
        // interpolated fiber is as suspect as the unwrap it rode on.
        placedLink.suspect = (traces[link.a].trusted && traces[link.b].trusted) &&
                             (placedLink.turnErr >= params.suspectTurns ||
                              droppedLinks.count(l) != 0);
        if (placedLink.suspect) {
            ++result.suspectLinkCount;
        }
        result.links.push_back(std::move(placedLink));
    }

    for (const winding::Crossing& crossing : solve.crossings) {
        // Declared errors only: a drop involving an untrusted fiber is the
        // interpolation's fault, and a drop the final map SATISFIES anyway is
        // greedy-repair debris (the real culprit fell in a later cycle) -
        // neither is an annotation mistake to ring in red.
        if (crossing.status != winding::CrossingStatus::Dropped ||
            !crossing.declarable ||
            crossing.violationTurns < solverParams.declarationViolationTurns) {
            continue;
        }
        ++result.droppedCrossingCount;
        if (!drawable[crossing.hFiber]) {
            continue;
        }
        const double x =
            (crossing.psiH + kTwoPi * solve.placements[crossing.hFiber].turns) *
            rRefVx;
        result.suspectCrossings.push_back(CrossingMark{QPointF(x, crossing.zVx)});
    }

    const double padX = std::max(kPadFraction * (hiX - loX), params.minPadXVx);
    const double padY = std::max(kPadFraction * (hiY - loY), params.minPadYVx);
    result.x0Vx = loX - padX;
    result.x1Vx = hiX + padX;
    result.yMinVx = loY - padY;
    result.yMaxVx = hiY + padY;

    // One gridline per integer winding across the padded extent; the mark
    // number IS the winding coordinate (innermost anchored winding = 0).
    const double circumference = kTwoPi * rRefVx;
    const long long first =
        static_cast<long long>(std::ceil(result.x0Vx / circumference));
    const long long last =
        static_cast<long long>(std::floor(result.x1Vx / circumference));
    for (long long mark = first; mark <= last; ++mark) {
        result.windings.push_back(WindingMark{
            static_cast<double>(mark) * circumference, static_cast<int>(mark)});
    }
    sortUnplaced();
    return result;
}

} // namespace vc3d::fiber_map
