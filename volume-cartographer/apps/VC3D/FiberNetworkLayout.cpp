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
// Uniform arclength resampling step of the drawn geometry, in cm.
constexpr double kResampleStepCm = 0.025;
// Minimum padding around a network, in cm: room for a few rows of label chips.
constexpr double kMinPadXCm = 2.2;
constexpr double kMinPadYCm = 1.6;
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
    if (total < 2.0 * step || points.size() < 3) {
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

    const int radius =
        std::max(1, static_cast<int>(std::nearbyint(3.0 * sigma / step)));
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
    double rRefCm = 0.0;
    std::vector<PlacedFiber> fibers;
    std::vector<PlacedLink> links;
    double loXCm = 0.0;
    double hiXCm = 0.0;
    double loYCm = 0.0;
    double hiYCm = 0.0;
};

} // namespace

Result buildLayout(const std::vector<InputFiber>& fibers,
                   const std::vector<cv::Vec3f>& umbilicusCenters,
                   const LayoutParams& params)
{
    Result result;
    if (umbilicusCenters.empty()) {
        return result;
    }

    std::vector<cv::Vec3f> centers = umbilicusCenters;
    std::stable_sort(centers.begin(), centers.end(),
                     [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });
    std::vector<double> centerZ(centers.size());
    std::vector<double> centerX(centers.size());
    std::vector<double> centerY(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i) {
        centerZ[i] = centers[i][2];
        centerX[i] = centers[i][0];
        centerY[i] = centers[i][1];
    }

    // Fibers without geometry cannot be unrolled, so they take no part in the
    // link graph either.
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
    const std::size_t fiberCount = ordered.size();
    if (fiberCount == 0) {
        return result;
    }

    std::unordered_map<uint64_t, std::size_t> indexById;
    indexById.reserve(fiberCount);
    for (std::size_t i = 0; i < fiberCount; ++i) {
        indexById.emplace(ordered[i]->id, i);
    }

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
    for (std::size_t i = 0; i < fiberCount; ++i) {
        for (const InputLink& link : ordered[i]->links) {
            const auto target = indexById.find(link.branchFiberId);
            if (target == indexById.end()) {
                continue;
            }
            const std::size_t a = findRoot(i);
            const std::size_t b = findRoot(target->second);
            if (a != b) {
                parent[a] = b;
            }
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

    const double toCm = params.voxelSizeUm / 10000.0;
    const double sigmaCm = params.smoothMm / 10.0;
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
            const InputFiber& fiber = *ordered[member];
            PreparedFiber entry;
            entry.input = &fiber;
            std::vector<double> raw(fiber.linePoints.size());
            entry.radius.resize(fiber.linePoints.size());
            for (std::size_t i = 0; i < fiber.linePoints.size(); ++i) {
                const cv::Vec3d& point = fiber.linePoints[i];
                const double dx = point[0] - interpolate(point[2], centerZ, centerX);
                const double dy = point[1] - interpolate(point[2], centerZ, centerY);
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
            prepared.emplace(member, std::move(entry));
        }

        // Links, deduped by their sorted endpoint pair: the reciprocal ref of
        // an already-seen crossing is the same physical link. The map values
        // are indices into links, so the pending flags of both refs can be
        // merged onto the record that survived.
        std::vector<LinkRecord> links;
        std::map<std::pair<std::pair<std::size_t, int>, std::pair<std::size_t, int>>,
                 std::size_t>
            seen;
        for (const std::size_t member : component) {
            const InputFiber& fiber = *ordered[member];
            const int controlCount = static_cast<int>(fiber.controlPoints.size());
            for (const InputLink& link : fiber.links) {
                const auto target = indexById.find(link.branchFiberId);
                if (target == indexById.end() ||
                    prepared.find(target->second) == prepared.end()) {
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
                    // A half-updated reciprocal pair still reads as pending.
                    links[inserted.first->second].pending |= link.pending;
                    continue;
                }
                links.push_back(LinkRecord{member, ia, other, ib, 0.0, link.pending});
            }
        }
        if (links.empty()) {
            continue;
        }
        std::sort(links.begin(), links.end(),
                  [](const LinkRecord& a, const LinkRecord& b) {
                      return std::tie(a.a, a.ia, a.b, a.ib) <
                             std::tie(b.a, b.ia, b.b, b.ib);
                  });

        // Snap each fiber's whole-turn offset to its neighbours, growing the
        // link tree Prim-style from the best-agreeing link so one
        // wrong-winding link cannot decide a fiber's offset when a clean link
        // to the same fiber exists.
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
        const double rRefCm = median(controlRadii) * toCm;

        NetworkDraft draft;
        draft.networkIndex = networkIndex;
        draft.rRefCm = rRefCm;
        draft.loXCm = std::numeric_limits<double>::infinity();
        draft.hiXCm = -std::numeric_limits<double>::infinity();
        draft.loYCm = std::numeric_limits<double>::infinity();
        draft.hiYCm = -std::numeric_limits<double>::infinity();

        std::unordered_map<std::size_t, FiberGeometry> geometry;
        geometry.reserve(component.size());
        for (const std::size_t member : component) {
            const PreparedFiber& entry = prepared.at(member);
            const InputFiber& fiber = *entry.input;
            std::vector<QPointF> unrolled(fiber.linePoints.size());
            for (std::size_t i = 0; i < fiber.linePoints.size(); ++i) {
                unrolled[i] = QPointF((entry.thetaLine[i] + entry.offset) * rRefCm,
                                      fiber.linePoints[i][2] * toCm);
            }

            FiberGeometry geo;
            const std::vector<double> rawArclength = arclengths(unrolled);
            smoothPolyline(unrolled, sigmaCm, kResampleStepCm, geo.sampleArclength,
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

            // line_points overshoot the outermost control points by over a cm
            // on many fibers; those tails carry no segment metadata and are
            // not drawn, so they are clipped out of the geometry entirely --
            // otherwise label anchors and the extents would be computed from
            // invisible curve.
            const std::size_t begin =
                searchSortedLeft(geo.sampleArclength, geo.controlArclength.front());
            const std::size_t end =
                searchSortedRight(geo.sampleArclength, geo.controlArclength.back());
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

            for (const QPointF& point : geo.samples) {
                draft.loXCm = std::min(draft.loXCm, point.x());
                draft.hiXCm = std::max(draft.hiXCm, point.x());
                draft.loYCm = std::min(draft.loYCm, point.y());
                draft.hiYCm = std::max(draft.hiYCm, point.y());
            }
            geometry.emplace(member, std::move(geo));
        }
        if (!(draft.loXCm <= draft.hiXCm) || !(draft.loYCm <= draft.hiYCm)) {
            continue;
        }

        // Traced runs draw solid, thick and vivid; segments that are only
        // interpolations draw thin, dashed and faded -- "dashed = not real
        // trace data" at a glance.
        for (const std::size_t member : component) {
            const InputFiber& fiber = *prepared.at(member).input;
            const FiberGeometry& geo = geometry.at(member);
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
            draft.fibers.push_back(std::move(placedFiber));
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

        const double padX = std::max(kPadFraction * (draft.hiXCm - draft.loXCm), kMinPadXCm);
        const double padY = std::max(kPadFraction * (draft.hiYCm - draft.loYCm), kMinPadYCm);
        draft.loXCm -= padX;
        draft.hiXCm += padX;
        draft.loYCm -= padY;
        draft.hiYCm += padY;
        drafts.push_back(std::move(draft));
    }
    if (drafts.empty()) {
        return result;
    }

    // Panels ordered inner -> outer by median distance from the umbilicus.
    // Unrolled length starts at 0 on the left and runs continuously through
    // every panel: each next panel starts on the global tick grid, so every
    // panel shows the same labeling interval, and the gap between networks is
    // whatever that snap requires (at least minGapCm).
    std::stable_sort(drafts.begin(), drafts.end(),
                     [](const NetworkDraft& a, const NetworkDraft& b) {
                         return a.rRefCm < b.rRefCm;
                     });

    const double tickCm = params.panelTickCm > 0.0 ? params.panelTickCm : 5.0;
    double panelStart = 0.0;
    int windingNumber = 0;
    result.yMinCm = std::numeric_limits<double>::infinity();
    result.yMaxCm = -std::numeric_limits<double>::infinity();
    result.networks.reserve(drafts.size());
    for (NetworkDraft& draft : drafts) {
        const double width = draft.hiXCm - draft.loXCm;
        const double shift = panelStart - draft.loXCm;

        PlacedNetwork network;
        network.networkIndex = draft.networkIndex;
        network.rRefCm = draft.rRefCm;
        network.x0Cm = panelStart;
        network.x1Cm = panelStart + width;
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

        const double circumference = kTwoPi * draft.rRefCm;
        if (circumference > 0.0) {
            const long long first =
                static_cast<long long>(std::ceil(draft.loXCm / circumference));
            const long long lastMark =
                static_cast<long long>(std::floor(draft.hiXCm / circumference));
            for (long long mark = first; mark <= lastMark; ++mark) {
                network.windings.push_back(WindingMark{
                    static_cast<double>(mark) * circumference + shift, windingNumber});
                ++windingNumber;
            }
        }

        result.yMinCm = std::min(result.yMinCm, draft.loYCm);
        result.yMaxCm = std::max(result.yMaxCm, draft.hiYCm);
        result.widthCm = panelStart + width;
        result.networks.push_back(std::move(network));
        panelStart = tickCm * std::ceil((panelStart + width + params.minGapCm) / tickCm);
    }
    return result;
}

} // namespace vc3d::fiber_map
