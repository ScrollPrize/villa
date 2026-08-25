#include "FiberWindingSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace vc3d::fiber_map::winding
{

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
// Sample points per fiber for the local radial-ordering cost. Ordinal
// comparisons need coverage, not density.
constexpr std::size_t kOrdinalSamples = 48;
// An island whose runner-up shift scores within this fraction of the
// runner-up's own cost is ambiguous: ordinal costs scale with the pair
// count, so a purely absolute margin stops registering near-ties the moment
// the neighbourhoods hold more than a handful of samples - while a winner
// whose runner-up carries real violations stays decisive at any scale.
constexpr double kRelativeAmbiguityFraction = 0.15;
// Coordinate-ascent search window (turns) and pass cap. The window keeps each
// step's candidate evaluation cheap; the pass cap bounds total travel (window
// times passes), and real data has shown slack chains packed tens of windings
// from their ordinal optimum, so travel is what the cap must budget for.
// Convergence exits early, so quiet solves never pay for the headroom.
constexpr long long kAscentWindow = 4;
constexpr int kAscentPasses = 40;

double wrappedDelta(double a, double b)
{
    double d = std::fmod(a - b + M_PI, kTwoPi);
    if (d < 0.0) {
        d += kTwoPi;
    }
    return d - M_PI;
}

// k[to] - k[from] >= weight. Equalities are a pair of mirrored constraints
// dropped together.
struct Constraint {
    std::size_t from = 0;
    std::size_t to = 0;
    long long weight = 0;
    double confidence = 0.0;
    // >= 0: index into the merged crossing list; < 0: -(linkIndex + 1).
    long long source = 0;
    long long pair = -1;
    bool active = true;
};

struct RawCrossing {
    Crossing crossing;
};

// A V fiber split into z-monotone branches, each re-ordered to ascending z so
// overlap queries can binary-search. Reordering the points does not change the
// segment set, only the direction each segment is walked in.
struct Branch {
    std::vector<double> psi;
    std::vector<double> z;
    std::vector<double> r;
    double psiMin = 0.0;
    double psiMax = 0.0;
    double rScale = 0.0;
};

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

std::vector<Branch> splitBranches(const std::vector<double>& psi,
                                  const std::vector<double>& z,
                                  const std::vector<double>& r)
{
    std::vector<Branch> branches;
    if (z.size() < 2) {
        return branches;
    }
    std::size_t start = 0;
    int direction = 0;
    const auto emit = [&](std::size_t begin, std::size_t end) {
        if (end - begin < 1) {
            return;
        }
        Branch branch;
        const std::size_t count = end - begin + 1;
        branch.psi.resize(count);
        branch.z.resize(count);
        branch.r.resize(count);
        const bool ascending = z[end] >= z[begin];
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t src = ascending ? begin + i : end - i;
            branch.psi[i] = psi[src];
            branch.z[i] = z[src];
            branch.r[i] = r[src];
        }
        branch.psiMin = *std::min_element(branch.psi.begin(), branch.psi.end());
        branch.psiMax = *std::max_element(branch.psi.begin(), branch.psi.end());
        branch.rScale = median(branch.r);
        branches.push_back(std::move(branch));
    };
    for (std::size_t i = 1; i < z.size(); ++i) {
        const double delta = z[i] - z[i - 1];
        if (delta == 0.0) {
            continue;
        }
        const int sign = delta > 0.0 ? 1 : -1;
        if (direction == 0) {
            direction = sign;
        } else if (sign != direction) {
            emit(start, i - 1);
            start = i - 1;
            direction = sign;
        }
    }
    emit(start, z.size() - 1);
    return branches;
}

// Evenly spread sample indices for the ordinal cost.
std::vector<std::size_t> sampleIndices(std::size_t count)
{
    std::vector<std::size_t> indices;
    if (count == 0) {
        return indices;
    }
    if (count <= kOrdinalSamples) {
        indices.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            indices[i] = i;
        }
        return indices;
    }
    indices.reserve(kOrdinalSamples);
    for (std::size_t i = 0; i < kOrdinalSamples; ++i) {
        indices.push_back(i * (count - 1) / (kOrdinalSamples - 1));
    }
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

struct OrdinalPoint {
    double z = 0.0;
    double psi = 0.0;
    double r = 0.0;
    std::size_t fiber = 0;
};

} // namespace

SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                          const std::vector<LinkInput>& links,
                          const SolverParams& params)
{
    SolveResult result;
    const std::size_t count = fibers.size();
    result.placements.assign(count, Placement{});
    result.linkTurnErrors.assign(links.size(),
                                 std::numeric_limits<double>::infinity());
    if (count == 0) {
        return result;
    }

    // --- Chirality: the winding coordinate must grow outward. Inferred from
    // the theta-radius covariance summed over every fiber; multi-turn H fibers
    // dominate by construction because covariance scales with angular span.
    int chirality = params.chiralityOverride;
    if (chirality == 0) {
        // Radius one whole turn along the same fiber is the same ray one
        // winding out: crumpling in angle cancels exactly and only the
        // spiral's sign survives (z drift along the turn does not cancel,
        // which is one reason each fiber gets one vote rather than one vote
        // per sample - no single dense or drifting fiber can flip the map).
        // Fibers that never wrap a full turn measure the crumple, not the
        // chirality, so the covariance fallback only decides when no fiber
        // wraps.
        int turnVotes = 0;
        int covarianceVotes = 0;
        bool haveTurnEvidence = false;
        for (const FiberTrace& fiber : fibers) {
            const std::size_t n = fiber.theta.size();
            if (n < 2 || fiber.radius.size() != n) {
                continue;
            }
            const bool ascending = fiber.theta.back() >= fiber.theta.front();
            double lagSum = 0.0;
            std::size_t j = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double target = ascending ? fiber.theta[i] + kTwoPi
                                                : fiber.theta[i] - kTwoPi;
                while (j < n && (ascending ? fiber.theta[j] < target
                                           : fiber.theta[j] > target)) {
                    ++j;
                }
                if (j >= n || j == 0) {
                    continue;
                }
                // Interpolate the radius at exactly one turn's lag, so the
                // vote is not polluted by however far the next sample
                // overshoots the turn.
                const double span = fiber.theta[j] - fiber.theta[j - 1];
                const double t = span != 0.0
                    ? (target - fiber.theta[j - 1]) / span
                    : 0.0;
                const double lagged =
                    fiber.radius[j - 1] + t * (fiber.radius[j] - fiber.radius[j - 1]);
                lagSum += ascending ? lagged - fiber.radius[i]
                                    : fiber.radius[i] - lagged;
            }
            if (lagSum != 0.0) {
                turnVotes += lagSum > 0.0 ? 1 : -1;
                haveTurnEvidence = true;
                continue;
            }
            double meanTheta = 0.0;
            double meanR = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                meanTheta += fiber.theta[i];
                meanR += fiber.radius[i];
            }
            meanTheta /= static_cast<double>(n);
            meanR /= static_cast<double>(n);
            double covariance = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                covariance +=
                    (fiber.theta[i] - meanTheta) * (fiber.radius[i] - meanR);
            }
            if (covariance != 0.0) {
                covarianceVotes += covariance > 0.0 ? 1 : -1;
            }
        }
        // Covariance only decides when no fiber wrapped at all; a TIE among
        // wrapping fibers resolves to the deterministic default rather than
        // letting one short crumpled fiber flip the map.
        const int vote = haveTurnEvidence ? turnVotes : covarianceVotes;
        chirality = vote < 0 ? -1 : 1;
    }
    result.chirality = chirality;

    // psi = s * theta - 2*pi*gauge: everything below works in a frame where
    // the winding coordinate grows outward AND every fiber's own median sits
    // within one turn of zero. The canonical gauge matters because the
    // densest-from-below solve floors every fiber at zero: without it that
    // floor lives in each fiber's arbitrary unwrap branch, and two physically
    // identical inputs whose gauges differ produce different maps. The
    // caller-facing turn offsets compensate on output, so W = s*theta/2pi +
    // turns holds in the caller's own gauge.
    std::vector<std::vector<double>> psi(count);
    std::vector<long long> gauge(count, 0);
    for (std::size_t f = 0; f < count; ++f) {
        psi[f].resize(fibers[f].theta.size());
        for (std::size_t i = 0; i < fibers[f].theta.size(); ++i) {
            psi[f][i] = chirality * fibers[f].theta[i];
        }
        if (!psi[f].empty()) {
            // floor(x + 0.5), not llround: rounding halves away from zero is
            // not translation-equivariant, so a whole-turn input re-gauge
            // could change the canonical gauge by two at a half-turn median.
            gauge[f] = static_cast<long long>(
                std::floor(median(psi[f]) / kTwoPi + 0.5));
            for (double& value : psi[f]) {
                value -= kTwoPi * static_cast<double>(gauge[f]);
            }
        }
    }
    const auto usable = [&](std::size_t f) {
        return fibers[f].theta.size() >= 2 &&
               fibers[f].radius.size() == fibers[f].theta.size() &&
               fibers[f].z.size() == fibers[f].theta.size();
    };

    // --- Crossing detection: transversal intersections of H polylines with V
    // polylines in (psi, z), over every 2*pi translate that can meet the V
    // branch's lift. The integer gap n is exact at an intersection because
    // both curves pass through the same lifted point.
    std::vector<Crossing> raw;
    const double maxStep = params.maxStepTurns * kTwoPi;
    for (std::size_t v = 0; v < count; ++v) {
        if (fibers[v].hvTag != 'V' || !usable(v)) {
            continue;
        }
        const std::vector<Branch> branches =
            splitBranches(psi[v], fibers[v].z, fibers[v].radius);
        for (const Branch& branch : branches) {
            const double branchZLo = branch.z.front();
            const double branchZHi = branch.z.back();
            for (std::size_t h = 0; h < count; ++h) {
                if (fibers[h].hvTag != 'H' || !usable(h)) {
                    continue;
                }
                const std::vector<double>& hPsi = psi[h];
                const std::vector<double>& hZ = fibers[h].z;
                const std::vector<double>& hR = fibers[h].radius;
                for (std::size_t i = 0; i + 1 < hPsi.size(); ++i) {
                    const double zLo = std::min(hZ[i], hZ[i + 1]);
                    const double zHi = std::max(hZ[i], hZ[i + 1]);
                    if (zHi < branchZLo || zLo > branchZHi) {
                        continue;
                    }
                    if (std::min(hR[i], hR[i + 1]) < params.minUmbilicusRadiusVx ||
                        std::abs(hPsi[i + 1] - hPsi[i]) > maxStep) {
                        ++result.gatedSegmentCount;
                        continue;
                    }
                    // Candidate 2*pi translates of this H segment into the
                    // branch's lift window.
                    const double segPsiLo = std::min(hPsi[i], hPsi[i + 1]);
                    const double segPsiHi = std::max(hPsi[i], hPsi[i + 1]);
                    const long long mLo = static_cast<long long>(
                        std::floor((branch.psiMin - segPsiHi) / kTwoPi));
                    const long long mHi = static_cast<long long>(
                        std::ceil((branch.psiMax - segPsiLo) / kTwoPi));
                    // V segments overlapping the H segment's z range, found by
                    // binary search on the branch's ascending z.
                    const auto zBegin = std::lower_bound(branch.z.begin(),
                                                         branch.z.end(), zLo);
                    std::size_t j0 = static_cast<std::size_t>(
                        zBegin - branch.z.begin());
                    j0 = j0 > 0 ? j0 - 1 : 0;
                    for (long long m = mLo; m <= mHi; ++m) {
                        const double x0 = hPsi[i] + kTwoPi * static_cast<double>(m);
                        const double x1 = hPsi[i + 1] + kTwoPi * static_cast<double>(m);
                        if (std::max(x0, x1) < branch.psiMin ||
                            std::min(x0, x1) > branch.psiMax) {
                            continue;
                        }
                        for (std::size_t j = j0;
                             j + 1 < branch.z.size() && branch.z[j] <= zHi; ++j) {
                            if (branch.z[j + 1] < zLo) {
                                continue;
                            }
                            if (std::min(branch.r[j], branch.r[j + 1]) <
                                    params.minUmbilicusRadiusVx ||
                                std::abs(branch.psi[j + 1] - branch.psi[j]) > maxStep) {
                                ++result.gatedSegmentCount;
                                continue;
                            }
                            const double rx = x1 - x0;
                            const double rz = hZ[i + 1] - hZ[i];
                            const double sx = branch.psi[j + 1] - branch.psi[j];
                            const double sz = branch.z[j + 1] - branch.z[j];
                            const double denom = rx * sz - rz * sx;
                            const double qpx = branch.psi[j] - x0;
                            const double qpz = branch.z[j] - hZ[i];
                            if (denom == 0.0) {
                                continue;
                            }
                            const double t = (qpx * sz - qpz * sx) / denom;
                            const double u = (qpx * rz - qpz * rx) / denom;
                            // Half-open on both segments so a crossing at a
                            // shared interior vertex is counted once - except
                            // that each polyline's FINAL segment closes at its
                            // end, so a crossing at a terminal vertex (or at a
                            // branch apex, which is the reversed end of both
                            // branches) is owned rather than lost. The apex's
                            // double detection is exactly what the dedup
                            // clustering exists to merge.
                            const bool tEnd = i + 2 == hPsi.size();
                            const bool uEnd = j + 2 == branch.z.size();
                            if (t < 0.0 || u < 0.0 ||
                                (tEnd ? t > 1.0 : t >= 1.0) ||
                                (uEnd ? u > 1.0 : u >= 1.0)) {
                                continue;
                            }
                            const double rH = hR[i] + t * (hR[i + 1] - hR[i]);
                            const double rV =
                                branch.r[j] + u * (branch.r[j + 1] - branch.r[j]);
                            // Transversality in arc-length-scaled coordinates:
                            // psi is radians, z voxels, so psi is scaled by the
                            // crossing's own radius - a branch-wide scale would
                            // let geometry far along the branch decide whether
                            // THIS pass counts as transversal.
                            const double rScale = 0.5 * (rH + rV);
                            const double hx = rx * rScale;
                            const double vx = sx * rScale;
                            const double hNorm = std::hypot(hx, rz);
                            const double vNorm = std::hypot(vx, sz);
                            if (hNorm == 0.0 || vNorm == 0.0) {
                                continue;
                            }
                            const double transversality =
                                std::abs(hx * sz - rz * vx) / (hNorm * vNorm);
                            if (transversality < params.minTransversality) {
                                ++result.tangentialCount;
                                continue;
                            }
                            Crossing crossing;
                            crossing.hFiber = h;
                            crossing.vFiber = v;
                            crossing.declarable =
                                fibers[h].trusted && fibers[v].trusted;
                            crossing.zVx = hZ[i] + t * rz;
                            crossing.psiH = hPsi[i] + t * (hPsi[i + 1] - hPsi[i]);
                            // The translate integer IS the turn gap, exactly;
                            // reconstructing it from large-angle subtraction
                            // would only reintroduce floating point.
                            crossing.n = m;
                            crossing.deltaR = rH - rV;
                            const double magnitude = std::abs(crossing.deltaR);
                            if (magnitude <= params.tieBandVx) {
                                crossing.kind = CrossingKind::Tie;
                                crossing.confidence = 0.9 * transversality *
                                    (1.0 - 0.2 * magnitude /
                                               std::max(params.tieBandVx, 1e-9));
                            } else {
                                crossing.kind = crossing.deltaR < 0.0
                                    ? CrossingKind::Inside
                                    : CrossingKind::Outside;
                                crossing.confidence = transversality *
                                    std::min(1.0, (magnitude - params.tieBandVx) /
                                                      (3.0 * params.tieBandVx));
                            }
                            if (!crossing.declarable) {
                                crossing.confidence *=
                                    params.untrustedConfidenceFactor;
                            }
                            raw.push_back(crossing);
                        }
                    }
                }
            }
        }
    }

    // --- Merge duplicate detections: one physical traversal seen by several
    // segment pairs (or twice across a branch split) is one piece of evidence.
    std::vector<std::size_t> order(raw.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&raw](std::size_t a, std::size_t b) {
        const Crossing& ca = raw[a];
        const Crossing& cb = raw[b];
        return std::tie(ca.hFiber, ca.vFiber, ca.n, ca.zVx, ca.deltaR) <
               std::tie(cb.hFiber, cb.vFiber, cb.n, cb.zVx, cb.deltaR);
    });
    std::vector<Crossing> merged;
    std::size_t index = 0;
    while (index < order.size()) {
        std::size_t end = index + 1;
        const Crossing& first = raw[order[index]];
        auto key = std::tie(first.hFiber, first.vFiber, first.n);
        std::size_t best = index;
        while (end < order.size()) {
            const Crossing& next = raw[order[end]];
            // One physical traversal seen twice has nearly the same z AND
            // nearly the same radial separation. The kind is deliberately not
            // part of the identity - duplicate detections straddling the tie
            // band must merge, not turn into a manufactured conflict - while
            // the deltaR gate keeps genuinely distinct traversals apart (two
            // branches of a U-shaped fiber can share z, n and kind at wildly
            // different radii).
            if (std::tie(next.hFiber, next.vFiber, next.n) != key ||
                next.zVx - first.zVx > params.zMergeVx ||
                std::abs(next.deltaR - first.deltaR) > params.tieBandVx) {
                break;
            }
            if (next.confidence > raw[order[best]].confidence) {
                best = end;
            }
            ++end;
        }
        Crossing representative = raw[order[best]];
        representative.mergedCount = static_cast<int>(end - index);
        representative.confidence = std::min(
            2.0, representative.confidence *
                     (1.0 + 0.25 * static_cast<double>(representative.mergedCount - 1)));
        merged.push_back(representative);
        index = end;
    }

    // --- Constraint graph.
    std::vector<Constraint> constraints;
    const auto addPair = [&constraints](std::size_t from, std::size_t to,
                                        long long weight, double confidence,
                                        long long source) {
        // Equality: to - from == weight, as a mirrored pair sharing one fate.
        Constraint forward{from, to, weight, confidence, source, -1, true};
        Constraint backward{to, from, -weight, confidence, source, -1, true};
        forward.pair = static_cast<long long>(constraints.size() + 1);
        backward.pair = static_cast<long long>(constraints.size());
        constraints.push_back(forward);
        constraints.push_back(backward);
    };

    for (std::size_t c = 0; c < merged.size(); ++c) {
        const Crossing& crossing = merged[c];
        const long long source = static_cast<long long>(c);
        switch (crossing.kind) {
        case CrossingKind::Inside:
            // W_h <= W_v: same winding or further inward, never a forced gap
            // (papyrus structure: same-winding H passes inside its V).
            constraints.push_back(Constraint{crossing.hFiber, crossing.vFiber,
                                             -crossing.n, crossing.confidence,
                                             source, -1, true});
            break;
        case CrossingKind::Outside:
            // W_h >= W_v + 1: strictly outward.
            constraints.push_back(Constraint{crossing.vFiber, crossing.hFiber,
                                             1 + crossing.n, crossing.confidence,
                                             source, -1, true});
            break;
        case CrossingKind::Tie:
            // Same winding: k_v - k_h == -n.
            addPair(crossing.hFiber, crossing.vFiber, -crossing.n,
                    crossing.confidence, source);
            ++result.tieCount;
            break;
        }
    }

    std::vector<bool> linkValid(links.size(), false);
    for (std::size_t l = 0; l < links.size(); ++l) {
        const LinkInput& link = links[l];
        if (link.fiberA >= count || link.fiberB >= count ||
            !usable(link.fiberA) || !usable(link.fiberB) ||
            link.pointA >= psi[link.fiberA].size() ||
            link.pointB >= psi[link.fiberB].size()) {
            continue;
        }
        linkValid[l] = true;
        const double delta =
            (psi[link.fiberB][link.pointB] - psi[link.fiberA][link.pointA]) / kTwoPi;
        const long long a = static_cast<long long>(std::llround(delta));
        const double residual = std::abs(delta - static_cast<double>(a));
        // A clean link outranks any single crossing; a link half a turn out
        // ranks below everything. The repair loop's seen-count discount is
        // what keeps even a clean-looking wrong link from consuming several
        // correct crossings.
        double confidence = 1.5 *
            std::max(0.0, 1.0 - residual / std::max(params.linkSuspectTurns, 1e-9));
        if (!fibers[link.fiberA].trusted || !fibers[link.fiberB].trusted) {
            // The residual itself rides on interpolated unwrapping, so it is
            // as suspect as the geometry it was measured over.
            confidence *= params.untrustedConfidenceFactor;
        }
        // W_A(pA) == W_B(pB) is k_A - k_B == a; addPair encodes to - from.
        addPair(link.fiberA, link.fiberB, -a, confidence,
                -(static_cast<long long>(l) + 1));
        result.placements[link.fiberA].linked = true;
        result.placements[link.fiberB].linked = true;
    }

    // --- Repair: while a positive cycle exists, drop the cycle's weakest
    // constraint, discounting by how often a constraint has already sat in a
    // detected cycle.
    std::vector<int> seen(constraints.size(), 0);
    std::vector<long long> x(count, 0);
    std::vector<long long> pred(count, -1);
    const auto dropConstraint = [&](std::size_t ci) {
        constraints[ci].active = false;
        if (constraints[ci].pair >= 0) {
            constraints[static_cast<std::size_t>(constraints[ci].pair)].active = false;
        }
        const long long source = constraints[ci].source;
        if (source >= 0) {
            merged[static_cast<std::size_t>(source)].status = CrossingStatus::Dropped;
            ++result.droppedCrossingCount;
        } else {
            result.droppedLinks.push_back(static_cast<std::size_t>(-source - 1));
        }
    };
    for (;;) {
        std::fill(x.begin(), x.end(), 0);
        std::fill(pred.begin(), pred.end(), -1);
        std::size_t relaxed = count;
        bool changed = true;
        for (std::size_t pass = 0; pass <= count && changed; ++pass) {
            changed = false;
            for (std::size_t ci = 0; ci < constraints.size(); ++ci) {
                const Constraint& constraint = constraints[ci];
                if (!constraint.active) {
                    continue;
                }
                if (x[constraint.from] + constraint.weight > x[constraint.to]) {
                    x[constraint.to] = x[constraint.from] + constraint.weight;
                    pred[constraint.to] = static_cast<long long>(ci);
                    relaxed = constraint.to;
                    changed = true;
                }
            }
        }
        if (!changed) {
            break;
        }
        // Walk predecessors until a node repeats: that node sits on a
        // predecessor cycle, which is the positive cycle (or feeds off one).
        std::vector<char> visited(count, 0);
        std::size_t node = relaxed;
        while (pred[node] >= 0 && visited[node] == 0) {
            visited[node] = 1;
            node = constraints[static_cast<std::size_t>(pred[node])].from;
        }
        std::vector<std::size_t> cycle;
        if (pred[node] >= 0) {
            std::size_t walk = node;
            do {
                const std::size_t ci = static_cast<std::size_t>(pred[walk]);
                cycle.push_back(ci);
                walk = constraints[ci].from;
            } while (walk != node && cycle.size() <= constraints.size());
        } else {
            // Defensive: the chain died before looping. Dropping the edge that
            // performed the final relaxation still makes progress.
            cycle.push_back(static_cast<std::size_t>(pred[relaxed]));
        }
        // Score with the counts from previous cycles, then record this one.
        std::size_t victim = cycle.front();
        double victimScore = std::numeric_limits<double>::infinity();
        for (const std::size_t ci : cycle) {
            const double score =
                constraints[ci].confidence /
                (1.0 + 2.0 * static_cast<double>(seen[ci]));
            if (score < victimScore ||
                (score == victimScore && ci < victim)) {
                victimScore = score;
                victim = ci;
            }
        }
        for (const std::size_t ci : cycle) {
            ++seen[ci];
            if (constraints[ci].pair >= 0) {
                ++seen[static_cast<std::size_t>(constraints[ci].pair)];
            }
        }
        dropConstraint(victim);
    }
    // x now holds the densest-from-below solution (all-zero super-source
    // longest paths) of the feasible graph.
    std::vector<long long> k(x);

    // --- Components over the surviving constraints.
    std::vector<std::size_t> parent(count);
    for (std::size_t i = 0; i < count; ++i) {
        parent[i] = i;
    }
    const auto findRoot = [&parent](std::size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (const Constraint& constraint : constraints) {
        if (!constraint.active) {
            continue;
        }
        const std::size_t a = findRoot(constraint.from);
        const std::size_t b = findRoot(constraint.to);
        if (a != b) {
            parent[a] = b;
        }
    }
    std::map<std::size_t, std::vector<std::size_t>> componentsByRoot;
    for (std::size_t i = 0; i < count; ++i) {
        componentsByRoot[findRoot(i)].push_back(i);
    }
    std::vector<std::vector<std::size_t>> components;
    for (auto& entry : componentsByRoot) {
        components.push_back(std::move(entry.second));
    }
    std::sort(components.begin(), components.end(),
              [](const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
                  if (a.size() != b.size()) {
                      return a.size() > b.size();
                  }
                  return a.front() < b.front();
              });

    // The primary component must actually carry a crossing constraint: a
    // link-only network, however large, proves no winding. Fall back to the
    // largest component only when no crossings survived anywhere.
    std::set<std::size_t> rootsWithCrossings;
    for (const Constraint& constraint : constraints) {
        if (constraint.active && constraint.source >= 0) {
            rootsWithCrossings.insert(findRoot(constraint.from));
        }
    }
    std::size_t primaryIndex = components.size();
    for (std::size_t c = 0; c < components.size(); ++c) {
        if (rootsWithCrossings.count(findRoot(components[c].front())) != 0) {
            primaryIndex = c;
            break;
        }
    }

    // Movable blocks: fibers locked together by equality constraints (links
    // and ties) can only satisfy their local radial ordering by moving as one
    // unit - each member alone reads lo == hi and could never move.
    std::vector<std::size_t> blockParent(count);
    for (std::size_t i = 0; i < count; ++i) {
        blockParent[i] = i;
    }
    const auto blockRoot = [&blockParent](std::size_t i) {
        while (blockParent[i] != i) {
            blockParent[i] = blockParent[blockParent[i]];
            i = blockParent[i];
        }
        return i;
    };
    for (const Constraint& constraint : constraints) {
        if (!constraint.active || constraint.pair < 0) {
            continue;
        }
        const std::size_t a = blockRoot(constraint.from);
        const std::size_t b = blockRoot(constraint.to);
        if (a != b) {
            blockParent[a] = b;
        }
    }
    std::vector<std::size_t> blockOf(count);
    for (std::size_t i = 0; i < count; ++i) {
        blockOf[i] = blockRoot(i);
    }

    // --- Local radial-ordering cost. One z-sorted point set over every fiber;
    // membership in the comparison set is a flag consulted per query.
    std::vector<OrdinalPoint> points;
    for (std::size_t f = 0; f < count; ++f) {
        if (!usable(f)) {
            continue;
        }
        for (const std::size_t i : sampleIndices(psi[f].size())) {
            points.push_back(OrdinalPoint{fibers[f].z[i], psi[f][i],
                                          fibers[f].radius[i], f});
        }
    }
    std::sort(points.begin(), points.end(),
              [](const OrdinalPoint& a, const OrdinalPoint& b) { return a.z < b.z; });
    std::vector<double> pointZ(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        pointZ[i] = points[i].z;
    }
    std::vector<bool> active(count, false);
    constexpr std::size_t kNoBlock = std::numeric_limits<std::size_t>::max();

    // Ordering violations of fiber f at offset turns kf against the active
    // set. Neighbouring samples share a ray to within the window, so their
    // winding difference is near-integer; the tie band says whether the radii
    // demand the same winding, and a strict order is only asserted once |dr|
    // clears the crumple-slope allowances for the pair's z and arc
    // separation - anything in between carries no information. Pairs inside
    // excludeBlock are skipped: a block evaluating its own move must not
    // score against members it is about to move with.
    const auto ordinalCost = [&](std::size_t f, long long kf,
                                 std::size_t excludeBlock, std::size_t* pairs) {
        double cost = 0.0;
        std::size_t pairCount = 0;
        for (const std::size_t i : sampleIndices(psi[f].size())) {
            const double z = fibers[f].z[i];
            const double p = psi[f][i];
            const double r = fibers[f].radius[i];
            const auto lo = std::lower_bound(pointZ.begin(), pointZ.end(),
                                             z - params.neighborhoodZVx);
            const auto hi = std::upper_bound(pointZ.begin(), pointZ.end(),
                                             z + params.neighborhoodZVx);
            for (auto it = lo; it != hi; ++it) {
                const OrdinalPoint& q = points[static_cast<std::size_t>(
                    it - pointZ.begin())];
                if (q.fiber == f || !active[q.fiber] ||
                    (excludeBlock != kNoBlock && blockOf[q.fiber] == excludeBlock)) {
                    continue;
                }
                const double arc =
                    std::abs(wrappedDelta(p, q.psi)) * 0.5 * (r + q.r);
                if (arc > params.neighborhoodArcVx) {
                    continue;
                }
                const double wp = p / kTwoPi + static_cast<double>(kf);
                const double wq = q.psi / kTwoPi +
                                  static_cast<double>(k[q.fiber]);
                const long long dw = std::llround(wp - wq);
                const double dr = r - q.r;
                const double strictFloor = params.tieBandVx +
                    params.radialSlopePerZVx * std::abs(z - q.z) +
                    params.radialSlopePerArcVx * arc;
                if (std::abs(dr) <= params.tieBandVx) {
                    ++pairCount;
                    cost += 0.5 * static_cast<double>(std::min<long long>(
                                      std::llabs(dw), 2));
                } else if (std::abs(dr) > strictFloor) {
                    ++pairCount;
                    if (dw == 0 || (dw > 0) != (dr > 0.0)) {
                        cost += 1.0;
                    }
                }
            }
        }
        if (pairs != nullptr) {
            *pairs = pairCount;
        }
        return cost;
    };

    // Slack moves within the feasible interval toward the best local radial
    // ordering, one equality block at a time; constraints internal to the
    // moving block cancel, and everything else is clamped against the
    // neighbours' current values, so feasibility is invariant.
    const auto ascend = [&](const std::vector<std::size_t>& members) {
        std::map<std::size_t, std::vector<std::size_t>> blocks;
        for (const std::size_t f : members) {
            if (usable(f)) {
                blocks[blockOf[f]].push_back(f);
            }
        }
        for (int pass = 0; pass < kAscentPasses; ++pass) {
            bool changed = false;
            for (const auto& [block, blockMembers] : blocks) {
                long long deltaLo = -kAscentWindow;
                long long deltaHi = kAscentWindow;
                for (const Constraint& constraint : constraints) {
                    if (!constraint.active) {
                        continue;
                    }
                    const bool fromIn = blockOf[constraint.from] == block;
                    const bool toIn = blockOf[constraint.to] == block;
                    if (fromIn == toIn) {
                        continue;
                    }
                    if (toIn) {
                        deltaLo = std::max(deltaLo, k[constraint.from] +
                                                        constraint.weight -
                                                        k[constraint.to]);
                    } else {
                        deltaHi = std::min(deltaHi, k[constraint.to] -
                                                        constraint.weight -
                                                        k[constraint.from]);
                    }
                }
                if (deltaLo > 0 || deltaHi < 0 || deltaLo == deltaHi) {
                    continue;
                }
                const auto costAt = [&](long long delta) {
                    double cost = 0.0;
                    for (const std::size_t f : blockMembers) {
                        cost += ordinalCost(f, k[f] + delta, block, nullptr);
                    }
                    return cost;
                };
                long long best = 0;
                double bestCost = costAt(0);
                for (long long delta = deltaLo; delta <= deltaHi; ++delta) {
                    if (delta == 0) {
                        continue;
                    }
                    const double cost = costAt(delta);
                    if (cost < bestCost ||
                        (cost == bestCost &&
                         std::llabs(delta) < std::llabs(best))) {
                        bestCost = cost;
                        best = delta;
                    }
                }
                if (best != 0) {
                    for (const std::size_t f : blockMembers) {
                        k[f] += best;
                    }
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
    };

    // W of fiber f at sample i under the current k.
    const auto windingAt = [&](std::size_t f, std::size_t i) {
        return psi[f][i] / kTwoPi + static_cast<double>(k[f]);
    };
    const auto componentMinWinding = [&](const std::vector<std::size_t>& members) {
        double minW = std::numeric_limits<double>::infinity();
        for (const std::size_t f : members) {
            for (std::size_t i = 0; i < psi[f].size(); ++i) {
                minW = std::min(minW, windingAt(f, i));
            }
        }
        return std::isfinite(minW) ? minW : 0.0;
    };

    // --- Primary component: gauge fixed at innermost winding zero, slack
    // spent on local ordering against its own members. Without any surviving
    // crossing there is no primary at all - nothing proves a winding - and
    // every component runs the island path below with nothing anchored,
    // which reports it unresolved rather than inventing an anchor.
    if (primaryIndex < components.size()) {
        const std::vector<std::size_t>& primary = components[primaryIndex];
        for (const std::size_t f : primary) {
            active[f] = true;
        }
        ascend(primary);
        const long long shift = static_cast<long long>(
            std::floor(componentMinWinding(primary)));
        for (const std::size_t f : primary) {
            k[f] -= shift;
            result.placements[f].anchor = ComponentAnchor::Primary;
        }
    }

    // --- Islands, largest first: rigid shift by the same ordinal cost against
    // everything anchored so far, then their own slack ascent. Anchored
    // islands join the comparison set, so ordering is defined but
    // deterministic.
    for (std::size_t c = 0; c < components.size(); ++c) {
        if (c == primaryIndex) {
            continue;
        }
        const std::vector<std::size_t>& island = components[c];
        ++result.islandCount;
        // Candidate shifts implied by neighbouring anchored samples - but
        // only pairs that would actually score (tie or strict) may nominate:
        // a shift suggested by dead-zone geometry would be a guess that every
        // candidate then scores at zero.
        std::map<long long, std::size_t> candidates;
        for (const std::size_t f : island) {
            if (!usable(f)) {
                continue;
            }
            for (const std::size_t i : sampleIndices(psi[f].size())) {
                const double z = fibers[f].z[i];
                const double p = psi[f][i];
                const double r = fibers[f].radius[i];
                const auto lo = std::lower_bound(pointZ.begin(), pointZ.end(),
                                                 z - params.neighborhoodZVx);
                const auto hi = std::upper_bound(pointZ.begin(), pointZ.end(),
                                                 z + params.neighborhoodZVx);
                for (auto it = lo; it != hi; ++it) {
                    const OrdinalPoint& q = points[static_cast<std::size_t>(
                        it - pointZ.begin())];
                    if (!active[q.fiber]) {
                        continue;
                    }
                    const double arc =
                        std::abs(wrappedDelta(p, q.psi)) * 0.5 * (r + q.r);
                    if (arc > params.neighborhoodArcVx) {
                        continue;
                    }
                    const double dr = r - q.r;
                    const double strictFloor = params.tieBandVx +
                        params.radialSlopePerZVx * std::abs(z - q.z) +
                        params.radialSlopePerArcVx * arc;
                    if (std::abs(dr) > params.tieBandVx &&
                        std::abs(dr) <= strictFloor) {
                        continue;
                    }
                    const double wp = windingAt(f, i);
                    const double wq = q.psi / kTwoPi +
                                      static_cast<double>(k[q.fiber]);
                    ++candidates[std::llround(wq - wp)];
                }
            }
        }
        if (candidates.empty()) {
            // Nothing informative anywhere near: not comparable, not guessed.
            const long long shift = static_cast<long long>(
                std::floor(componentMinWinding(island)));
            for (const std::size_t f : island) {
                k[f] -= shift;
                result.placements[f].anchor = ComponentAnchor::Unresolved;
            }
            ++result.unresolvedCount;
            continue;
        }
        std::map<long long, double> costs;
        for (const auto& [delta, votes] : candidates) {
            (void)votes;
            for (const long long shift : {delta - 1, delta, delta + 1}) {
                costs.emplace(shift, 0.0);
            }
        }
        for (auto& [shift, cost] : costs) {
            for (const std::size_t f : island) {
                if (usable(f)) {
                    cost += ordinalCost(f, k[f] + shift, kNoBlock, nullptr);
                }
            }
        }
        long long bestShift = 0;
        double bestCost = std::numeric_limits<double>::infinity();
        double secondCost = std::numeric_limits<double>::infinity();
        for (const auto& [shift, cost] : costs) {
            if (cost < bestCost) {
                secondCost = bestCost;
                bestCost = cost;
                bestShift = shift;
            } else if (cost == bestCost &&
                       std::llabs(shift) < std::llabs(bestShift)) {
                // An exact cost tie: the old best is a genuine runner-up.
                secondCost = cost;
                bestShift = shift;
            } else {
                secondCost = std::min(secondCost, cost);
            }
        }
        const bool ambiguous =
            std::isfinite(secondCost) &&
            secondCost - bestCost <=
                std::max(params.anchorAmbiguityMargin,
                         kRelativeAmbiguityFraction * secondCost);
        for (const std::size_t f : island) {
            k[f] += bestShift;
            result.placements[f].anchor = ambiguous
                ? ComponentAnchor::AmbiguousRadius
                : ComponentAnchor::Radius;
            active[f] = true;
        }
        ascend(island);
    }

    // --- Outputs.
    for (std::size_t f = 0; f < count; ++f) {
        Placement& placement = result.placements[f];
        placement.turns = static_cast<double>(k[f] - gauge[f]);
        if (usable(f)) {
            double lo = std::numeric_limits<double>::infinity();
            double hi = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < psi[f].size(); ++i) {
                const double w = windingAt(f, i);
                lo = std::min(lo, w);
                hi = std::max(hi, w);
            }
            placement.windingLo = lo;
            placement.windingHi = hi;
        }
    }
    for (std::size_t l = 0; l < links.size(); ++l) {
        if (!linkValid[l]) {
            continue;
        }
        const LinkInput& link = links[l];
        result.linkTurnErrors[l] = std::abs(
            (psi[link.fiberA][link.pointA] / kTwoPi + static_cast<double>(k[link.fiberA])) -
            (psi[link.fiberB][link.pointB] / kTwoPi + static_cast<double>(k[link.fiberB])));
    }
    std::sort(result.droppedLinks.begin(), result.droppedLinks.end());
    // Violation of each crossing against the final map, in the canonical
    // gauge (before psiH is restored below): the H side reads directly off
    // the crossing, the V side off the V trace's nearest z sample.
    for (Crossing& crossing : merged) {
        const std::vector<double>& vz = fibers[crossing.vFiber].z;
        if (vz.empty()) {
            continue;
        }
        std::size_t nearest = 0;
        double bestDz = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < vz.size(); ++i) {
            const double dz = std::abs(vz[i] - crossing.zVx);
            if (dz < bestDz) {
                bestDz = dz;
                nearest = i;
            }
        }
        const double wH = crossing.psiH / kTwoPi +
                          static_cast<double>(k[crossing.hFiber]);
        const double wV = psi[crossing.vFiber][nearest] / kTwoPi +
                          static_cast<double>(k[crossing.vFiber]);
        switch (crossing.kind) {
        case CrossingKind::Inside:   // demanded W_h <= W_v
            crossing.violationTurns = std::max(0.0, wH - wV);
            break;
        case CrossingKind::Outside:  // demanded W_h >= W_v + 1
            crossing.violationTurns = std::max(0.0, wV + 1.0 - wH);
            break;
        case CrossingKind::Tie:      // demanded W_h == W_v
            crossing.violationTurns = std::abs(wH - wV);
            break;
        }
    }
    // Sheet drift: rule 1 constrains "that section" of an H fiber; one k per
    // fiber assumes the annotation stays on one sheet. Repeated drops against
    // DISTINCT evidence on one H fiber - different V fibers or different
    // turns - are the signature of that assumption failing, surfaced rather
    // than solved; two drops of one contested traversal are not.
    std::map<std::size_t, std::set<std::pair<std::size_t, long long>>> dropsPerFiber;
    for (const Crossing& crossing : merged) {
        // Only declarable, actually-violated drops are drift evidence: a
        // contradiction against (or from) interpolated geometry says nothing
        // about the H fiber, and a drop the final map satisfies anyway is
        // repair debris, not evidence of anything.
        if (crossing.status == CrossingStatus::Dropped && crossing.declarable &&
            crossing.violationTurns >= params.declarationViolationTurns) {
            dropsPerFiber[crossing.hFiber].emplace(crossing.vFiber, crossing.n);
        }
    }
    for (const auto& [f, evidence] : dropsPerFiber) {
        if (evidence.size() >= 2) {
            result.placements[f].sheetDriftSuspect = true;
        }
    }
    // Crossing psiH goes back out in the caller's gauge, matching turns.
    for (Crossing& crossing : merged) {
        crossing.psiH +=
            kTwoPi * static_cast<double>(gauge[crossing.hFiber]);
    }
    result.crossings = std::move(merged);
    return result;
}

} // namespace vc3d::fiber_map::winding
