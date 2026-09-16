#include "FiberWindingSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
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

// What a constraint was built from: a merged crossing, an input link, or a
// traversal group standing in for its member crossings.
enum class SourceKind { Crossing, Link, Group };
struct SourceRef {
    SourceKind kind = SourceKind::Crossing;
    std::size_t index = 0;
};

// k[to] - k[from] >= weight. Equalities are a pair of mirrored constraints
// dropped together.
struct Constraint {
    std::size_t from = 0;
    std::size_t to = 0;
    long long weight = 0;
    double confidence = 0.0;
    SourceRef source;
    long long pair = -1;
    bool active = true;
};

struct RawCrossing {
    Crossing crossing;
};

// A V fiber split into z-monotone branches, each re-ordered to ascending z so
// overlap queries can binary-search. Reordering the points does not change the
// segment set, only the direction each segment is walked in.
using Branch = CanonicalTrace::Branch;

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

// Non-finite coordinates would reach sorts (strict-weak-ordering violation)
// and float-to-integer casts (UB); a trace carrying any is unusable.
bool traceValuesFinite(const FiberTrace& fiber)
{
    const auto allFinite = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    return allFinite(fiber.theta) && allFinite(fiber.radius) &&
           allFinite(fiber.z);
}

// floor(x + 0.5), not llround: rounding halves away from zero is not
// translation-equivariant, so a whole-turn input re-gauge could change the
// canonical gauge by two at a half-turn median.
long long canonicalGauge(std::vector<double> psi)
{
    if (psi.empty()) {
        return 0;
    }
    return static_cast<long long>(std::floor(median(std::move(psi)) / kTwoPi + 0.5));
}

std::vector<Branch> splitBranches(const std::vector<double>& psi,
                                  const std::vector<double>& z,
                                  const std::vector<double>& r)
{
    std::vector<Branch> branches;
    if (z.size() < 2) {
        return branches;
    }
    // Vertex identity: consecutive samples at one (psi, z) are one vertex.
    std::vector<std::size_t> vertexOf(z.size(), 0);
    for (std::size_t i = 1; i < z.size(); ++i) {
        vertexOf[i] = (psi[i] == psi[i - 1] && z[i] == z[i - 1]) ? vertexOf[i - 1]
                                                                  : vertexOf[i - 1] + 1;
    }
    std::size_t start = 0;
    int direction = 0;
    // Named to survive Qt's `emit` macro: with the project PCH, Qt headers
    // reach even this Qt-free TU, and a lambda named `emit` fails to parse.
    const auto emitBranch = [&](std::size_t begin, std::size_t end) {
        if (end - begin < 1) {
            return;
        }
        Branch branch;
        const std::size_t count = end - begin + 1;
        branch.psi.resize(count);
        branch.z.resize(count);
        branch.r.resize(count);
        branch.vertexId.resize(count);
        const bool ascending = z[end] >= z[begin];
        branch.forwardAscending = ascending;
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t src = ascending ? begin + i : end - i;
            branch.psi[i] = psi[src];
            branch.z[i] = z[src];
            branch.r[i] = r[src];
            branch.vertexId[i] = vertexOf[src];
        }
        branch.psiMin = *std::min_element(branch.psi.begin(), branch.psi.end());
        branch.psiMax = *std::max_element(branch.psi.begin(), branch.psi.end());
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
            emitBranch(start, i - 1);
            start = i - 1;
            direction = sign;
        }
    }
    emitBranch(start, z.size() - 1);
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

int inferChirality(const std::vector<FiberTrace>& fibers, int chiralityOverride)
{
    int chirality = chiralityOverride;
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
            if (n < 2 || fiber.radius.size() != n || !traceValuesFinite(fiber)) {
                continue;
            }
            const bool ascending = fiber.theta.back() >= fiber.theta.front();
            // The one-turn-lag sweep walks a single monotone cursor, so a
            // fiber whose theta locally reverses would pair samples from
            // unrelated sections and cast a garbage turn vote; such a fiber
            // votes through its covariance instead.
            bool monotone = true;
            for (std::size_t i = 1; i < n && monotone; ++i) {
                const double step = fiber.theta[i] - fiber.theta[i - 1];
                monotone = ascending ? step >= 0.0 : step <= 0.0;
            }
            double lagSum = 0.0;
            std::size_t j = 0;
            for (std::size_t i = 0; monotone && i < n; ++i) {
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
    return chirality;
}

CanonicalTrace canonicalizeTrace(const FiberTrace& fiber, int chirality)
{
    CanonicalTrace trace;
    trace.hvTag = fiber.hvTag;
    trace.trusted = fiber.trusted;
    trace.kollesisStartSample = fiber.kollesisStartSample;
    trace.kollesisEndSample = fiber.kollesisEndSample;
    trace.onKollesis = fiber.onKollesis;
    const bool usable = fiber.theta.size() >= 2 &&
                        fiber.radius.size() == fiber.theta.size() &&
                        fiber.z.size() == fiber.theta.size() &&
                        traceValuesFinite(fiber);
    if (!usable) {
        return trace;
    }
    trace.psi.resize(fiber.theta.size());
    for (std::size_t i = 0; i < fiber.theta.size(); ++i) {
        trace.psi[i] = chirality * fiber.theta[i];
    }
    trace.gauge = canonicalGauge(trace.psi);
    for (double& value : trace.psi) {
        value -= kTwoPi * static_cast<double>(trace.gauge);
    }
    trace.radius = fiber.radius;
    trace.z = fiber.z;
    if (trace.hvTag == 'V') {
        trace.branches = splitBranches(trace.psi, trace.z, trace.radius);
    }
    return trace;
}

// The detection loop for one (H, V) pair, over the V trace's precomputed
// z-monotone branches. Everything here is pair-local: the produced crossings
// carry no indices, no global ids, and no dependence on any other pair.
PairDetections detectPairCrossings(const CanonicalTrace& hTrace,
                                   const CanonicalTrace& vTrace,
                                   const SolverParams& params)
{
    PairDetections result;
    if (hTrace.psi.empty() || vTrace.psi.empty()) {
        return result;
    }
    const bool trusted = hTrace.trusted && vTrace.trusted;
    // Transversal events, merged by the classification into representatives
    // as before, and shallow (tangential) events, which are recorded and
    // counted but never merged into a representative or constraining on
    // their own - exactly the passes the transversality gate used to discard
    // unrecorded.
    std::vector<Crossing> raw;
    std::vector<Crossing> shallow;
    std::size_t detectionCount = 0;
    // Translates on which an event may have gone unseen: a gated segment
    // whose translate could have met the other fiber, or a pair of exactly
    // parallel owner segments the intersection cannot be placed on.
    std::set<long long> gapTranslates;
    std::set<long long> unresolvedTranslates;
    // Which side of the directed segment a->b the point p lies on, in
    // (psi, z): +1, -1, or 0 on the line.
    const auto sideOf = [](double ax, double az, double bx, double bz, double px, double pz) {
        const double cross = (bx - ax) * (pz - az) - (bz - az) * (px - ax);
        return cross > 0.0 ? 1 : (cross < 0.0 ? -1 : 0);
    };
    const double maxStep = params.maxStepTurns * kTwoPi;
    const std::vector<double>& hPsi = hTrace.psi;
    const std::vector<double>& hZ = hTrace.z;
    const std::vector<double>& hR = hTrace.radius;
    // The last segment of positive length on each polyline: the one whose end
    // closes the trace, whatever repeated samples trail it.
    const auto lastRealSegment = [](const std::vector<double>& psi, const std::vector<double>& z) {
        std::size_t last = 0;
        for (std::size_t i = 0; i + 1 < psi.size(); ++i) {
            if (psi[i + 1] != psi[i] || z[i + 1] != z[i]) {
                last = i;
            }
        }
        return last;
    };
    // The nearest sample before/after `index` that differs from it, or npos.
    const auto distinctBefore = [](const std::vector<double>& psi, const std::vector<double>& z,
                                   std::size_t index) {
        for (std::size_t k = index; k > 0; --k) {
            if (psi[k - 1] != psi[index] || z[k - 1] != z[index]) {
                return k - 1;
            }
        }
        return static_cast<std::size_t>(-1);
    };
    const auto distinctAfter = [](const std::vector<double>& psi, const std::vector<double>& z,
                                  std::size_t index) {
        for (std::size_t k = index + 1; k < psi.size(); ++k) {
            if (psi[k] != psi[index] || z[k] != z[index]) {
                return k;
            }
        }
        return static_cast<std::size_t>(-1);
    };
    const std::size_t hLastSegment = lastRealSegment(hPsi, hZ);
    for (std::size_t branchIndex = 0; branchIndex < vTrace.branches.size(); ++branchIndex) {
        const Branch& branch = vTrace.branches[branchIndex];
        const double branchZLo = branch.z.front();
        const double branchZHi = branch.z.back();
        const std::size_t branchLastSegment = lastRealSegment(branch.psi, branch.z);
        // The branch is stored ascending in z; its segments are walked back
        // in the fiber's own polyline order for the orientation sign.
        const int branchDirection = branch.forwardAscending ? 1 : -1;
        for (std::size_t i = 0; i + 1 < hPsi.size(); ++i) {
            const double zLo = std::min(hZ[i], hZ[i + 1]);
            const double zHi = std::max(hZ[i], hZ[i + 1]);
            if (zHi < branchZLo || zLo > branchZHi) {
                continue;
            }
            // Candidate 2*pi translates of this H segment into the branch's
            // lift window - computed before the gate, because a gated segment
            // still says which translates it could have met.
            const double segPsiLo = std::min(hPsi[i], hPsi[i + 1]);
            const double segPsiHi = std::max(hPsi[i], hPsi[i + 1]);
            const long long mLo = static_cast<long long>(
                std::floor((branch.psiMin - segPsiHi) / kTwoPi));
            const long long mHi = static_cast<long long>(
                std::ceil((branch.psiMax - segPsiLo) / kTwoPi));
            if (std::min(hR[i], hR[i + 1]) < params.minUmbilicusRadiusVx ||
                std::abs(hPsi[i + 1] - hPsi[i]) > maxStep) {
                ++result.gatedSegmentCount;
                for (long long m = mLo; m <= mHi; ++m) {
                    const double lo = segPsiLo + kTwoPi * static_cast<double>(m);
                    const double hi = segPsiHi + kTwoPi * static_cast<double>(m);
                    if (hi >= branch.psiMin && lo <= branch.psiMax) {
                        gapTranslates.insert(m);
                    }
                }
                continue;
            }
            // V segments overlapping the H segment's z range, found by
            // binary search on the branch's ascending z.
            const auto zBegin = std::lower_bound(branch.z.begin(),
                                                 branch.z.end(), zLo);
            std::size_t j0 = static_cast<std::size_t>(zBegin - branch.z.begin());
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
                    const bool psiOverlap =
                        std::max(x0, x1) >= std::min(branch.psi[j], branch.psi[j + 1]) &&
                        std::min(x0, x1) <= std::max(branch.psi[j], branch.psi[j + 1]);
                    // Overlap of positive length along the coordinate the
                    // segments extend in (height for near-vertical ones).
                    // Collinear segments sharing only an endpoint are left to
                    // the neighbouring segments' own intersections; the
                    // contact itself is not recorded.
                    const bool alongZ = std::abs(x1 - x0) < std::abs(hZ[i + 1] - hZ[i]);
                    const bool overlapProper = alongZ
                        ? (zHi > std::min(branch.z[j], branch.z[j + 1]) &&
                           zLo < std::max(branch.z[j], branch.z[j + 1]))
                        : (std::max(x0, x1) > std::min(branch.psi[j], branch.psi[j + 1]) &&
                           std::min(x0, x1) < std::max(branch.psi[j], branch.psi[j + 1]));
                    if (std::min(branch.r[j], branch.r[j + 1]) <
                            params.minUmbilicusRadiusVx ||
                        std::abs(branch.psi[j + 1] - branch.psi[j]) > maxStep) {
                        ++result.gatedSegmentCount;
                        if (psiOverlap) {
                            gapTranslates.insert(m);
                        }
                        continue;
                    }
                    const double rx = x1 - x0;
                    const double rz = hZ[i + 1] - hZ[i];
                    const double sx = branch.psi[j + 1] - branch.psi[j];
                    const double sz = branch.z[j + 1] - branch.z[j];
                    // A zero-length segment (a repeated sample) meets nothing
                    // its neighbours do not; it is no unresolved intersection.
                    if ((rx == 0.0 && rz == 0.0) || (sx == 0.0 && sz == 0.0)) {
                        continue;
                    }
                    const double denom = rx * sz - rz * sx;
                    const double qpx = branch.psi[j] - x0;
                    const double qpz = branch.z[j] - hZ[i];
                    if (denom == 0.0) {
                        // Parallel owner segments. Disjoint parallels meet
                        // nowhere; collinear ones overlapping in angle share a
                        // stretch on which no intersection can be placed, so
                        // the translate's count is not to be trusted.
                        const bool collinear = rx * qpz - rz * qpx == 0.0;
                        if (collinear && overlapProper) {
                            unresolvedTranslates.insert(m);
                            ++result.unresolvedCount;
                        }
                        continue;
                    }
                    const double t = (qpx * sz - qpz * sx) / denom;
                    const double u = (qpx * rz - qpz * rx) / denom;
                    // Half-open on both segments so a crossing at a shared
                    // interior vertex is counted once - except that each
                    // polyline's FINAL segment closes at its end, so a
                    // crossing at a terminal vertex (or at a branch apex,
                    // which is the reversed end of both branches) is owned
                    // rather than lost. The apex's double detection is
                    // exactly what the dedup clustering exists to merge.
                    const bool tEnd = i == hLastSegment;
                    const bool uEnd = j == branchLastSegment;
                    if (t < 0.0 || u < 0.0 ||
                        (tEnd ? t > 1.0 : t >= 1.0) ||
                        (uEnd ? u > 1.0 : u >= 1.0)) {
                        continue;
                    }
                    const double rH = hR[i] + t * (hR[i + 1] - hR[i]);
                    const double rV =
                        branch.r[j] + u * (branch.r[j + 1] - branch.r[j]);
                    // Transversality in arc-length-scaled coordinates: psi is
                    // radians, z voxels, so psi is scaled by the crossing's
                    // own radius - a branch-wide scale would let geometry far
                    // along the branch decide whether THIS pass counts as
                    // transversal.
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
                    Crossing crossing;
                    crossing.zVx = hZ[i] + t * rz;
                    crossing.psiH = hPsi[i] + t * (hPsi[i + 1] - hPsi[i]);
                    // The translate integer IS the turn gap, exactly;
                    // reconstructing it from large-angle subtraction would
                    // only reintroduce floating point.
                    crossing.n = m;
                    crossing.deltaR = rH - rV;
                    crossing.transversality = transversality;
                    crossing.orientation = (denom > 0.0 ? 1 : -1) * branchDirection;
                    crossing.hSegment = i;
                    crossing.hT = t;
                    crossing.vBranch = branchIndex;
                    crossing.detection = detectionCount++;
                    if (u == 0.0) {
                        crossing.vSample = branch.vertexId[j];
                    } else if (u == 1.0) {
                        crossing.vSample = branch.vertexId[j + 1];
                    }
                    // A hit exactly at a vertex interior to a polyline is a
                    // crossing only if the vertex's two incident segments
                    // leave on opposite sides of the other segment; on one
                    // side it is a touch: the polyline came up to the other
                    // and turned back.
                    // A hit at a vertex of either polyline is a crossing only
                    // if, going round the hit point, the two H rays alternate
                    // with the two V rays (H, V, H, V): the polylines
                    // separate each other there. H, H, V, V is a touch: one
                    // polyline came up to the other and turned back. The
                    // incident rays are taken past any repeated samples; a
                    // polyline not at a vertex contributes its segment's two
                    // half-rays. Cyclic order of directions survives the
                    // anisotropic (psi, z) axes.
                    const bool hVertex = t == 0.0 && i > 0;
                    const bool vVertex = u == 0.0 && j > 0;
                    if (hVertex || vVertex) {
                        const double px = x0 + t * rx;
                        const double pz = hZ[i] + t * rz;
                        double rays[4][2];
                        bool defined = true;
                        if (hVertex) {
                            const std::size_t prev = distinctBefore(hPsi, hZ, i);
                            const std::size_t next = distinctAfter(hPsi, hZ, i);
                            defined = prev != static_cast<std::size_t>(-1) &&
                                      next != static_cast<std::size_t>(-1);
                            if (defined) {
                                rays[0][0] = hPsi[prev] + kTwoPi * static_cast<double>(m) - px;
                                rays[0][1] = hZ[prev] - pz;
                                rays[1][0] = hPsi[next] + kTwoPi * static_cast<double>(m) - px;
                                rays[1][1] = hZ[next] - pz;
                            }
                        } else {
                            rays[0][0] = -rx;
                            rays[0][1] = -rz;
                            rays[1][0] = rx;
                            rays[1][1] = rz;
                        }
                        if (vVertex) {
                            const std::size_t prev = distinctBefore(branch.psi, branch.z, j);
                            const std::size_t next = distinctAfter(branch.psi, branch.z, j);
                            defined = defined && prev != static_cast<std::size_t>(-1) &&
                                      next != static_cast<std::size_t>(-1);
                            if (defined) {
                                rays[2][0] = branch.psi[prev] - px;
                                rays[2][1] = branch.z[prev] - pz;
                                rays[3][0] = branch.psi[next] - px;
                                rays[3][1] = branch.z[next] - pz;
                            }
                        } else {
                            rays[2][0] = -sx;
                            rays[2][1] = -sz;
                            rays[3][0] = sx;
                            rays[3][1] = sz;
                        }
                        if (defined) {
                            double angle[4];
                            for (int r = 0; r < 4; ++r) {
                                angle[r] = std::atan2(rays[r][1], rays[r][0]);
                            }
                            int order[4] = {0, 1, 2, 3};
                            std::sort(order, order + 4,
                                      [&angle](int a, int b) { return angle[a] < angle[b]; });
                            // Alternating iff no two H rays (0, 1) are cyclic
                            // neighbours.
                            bool alternating = true;
                            for (int r = 0; r < 4; ++r) {
                                const bool aIsH = order[r] < 2;
                                const bool bIsH = order[(r + 1) % 4] < 2;
                                if (aIsH == bIsH) {
                                    alternating = false;
                                }
                            }
                            if (!alternating) {
                                crossing.touch = true;
                            }
                        }
                    }
                    // No tie band: the sign of deltaR is the whole
                    // classification. A same-winding contact reads inside
                    // (H on the sheet front), which the weak constraint
                    // absorbs at equality; a contact whose noise flips the
                    // sign becomes a strict outside, accepted as the price of
                    // not inventing an equality from a sub-band radial
                    // measurement. Confidence is asymmetric to match what
                    // each claim risks: the weak inside ("same or further
                    // in") is only false when the H fiber is truly a full
                    // winding outside - a wrap-scale radial error - so it is
                    // high at any margin; the strict outside asserts a whole
                    // winding of separation off the radial sign alone, so it
                    // earns confidence with radial margin and a sign-of-noise
                    // contact loses repair conflicts.
                    crossing.kind = crossing.deltaR <= 0.0
                        ? CrossingKind::Inside
                        : CrossingKind::Outside;
                    if (transversality < params.minTransversality) {
                        ++result.tangentialCount;
                        crossing.tangential = true;
                        crossing.confidence = 0.0;
                        shallow.push_back(crossing);
                        continue;
                    }
                    crossing.confidence =
                        crossing.kind == CrossingKind::Inside
                            ? 0.9 * transversality
                            : transversality *
                                  std::min(1.0,
                                           crossing.deltaR /
                                               (3.0 * std::max(params.tieBandVx,
                                                               1e-9)));
                    if (!trusted) {
                        crossing.confidence *= params.untrustedConfidenceFactor;
                    }
                    raw.push_back(crossing);
                }
            }
        }
    }

    result.raw = std::move(raw);
    result.shallow = std::move(shallow);
    result.detectionCount = detectionCount;
    result.gapTranslates.assign(gapTranslates.begin(), gapTranslates.end());
    result.unresolvedTranslates.assign(unresolvedTranslates.begin(),
                                       unresolvedTranslates.end());
    return result;
}

bool identicalPairDetections(const PairDetections& a, const PairDetections& b)
{
    const auto sameDouble = [](double x, double y) {
        return std::memcmp(&x, &y, sizeof(double)) == 0;
    };
    const auto sameCrossing = [&](const Crossing& x, const Crossing& y) {
        return sameDouble(x.zVx, y.zVx) && sameDouble(x.psiH, y.psiH) && x.n == y.n &&
               sameDouble(x.deltaR, y.deltaR) && sameDouble(x.confidence, y.confidence) &&
               x.mergedCount == y.mergedCount && x.kind == y.kind &&
               sameDouble(x.transversality, y.transversality) && x.tangential == y.tangential &&
               x.orientation == y.orientation && x.hSegment == y.hSegment &&
               sameDouble(x.hT, y.hT) && x.vSample == y.vSample && x.touch == y.touch &&
               x.vBranch == y.vBranch && x.detection == y.detection;
    };
    if (a.raw.size() != b.raw.size() || a.shallow.size() != b.shallow.size() ||
        a.detectionCount != b.detectionCount || a.gapTranslates != b.gapTranslates ||
        a.unresolvedTranslates != b.unresolvedTranslates ||
        a.gatedSegmentCount != b.gatedSegmentCount || a.tangentialCount != b.tangentialCount ||
        a.unresolvedCount != b.unresolvedCount) {
        return false;
    }
    for (std::size_t i = 0; i < a.raw.size(); ++i) {
        if (!sameCrossing(a.raw[i], b.raw[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.shallow.size(); ++i) {
        if (!sameCrossing(a.shallow[i], b.shallow[i])) {
            return false;
        }
    }
    return true;
}

std::vector<SeamAnchor> seamAnchors(const std::vector<CanonicalTrace>& traces,
                                    std::size_t hIndex, std::size_t vIndex,
                                    const std::vector<LinkInput>& links)
{
    std::vector<SeamAnchor> anchors;
    if (hIndex >= traces.size() || vIndex >= traces.size() || !traces[vIndex].onKollesis) {
        return anchors;
    }
    const CanonicalTrace& h = traces[hIndex];
    const CanonicalTrace& v = traces[vIndex];
    for (const std::size_t hSample : {h.kollesisStartSample, h.kollesisEndSample}) {
        if (hSample == kNoSample || hSample >= h.psi.size()) {
            continue;
        }
        // The pair's link nearest the tagged end along the H fiber. Two
        // links at one H sample naming different V samples say nothing
        // (the annotation tool links a control once; an import may not).
        SeamAnchor anchor;
        anchor.hSample = hSample;
        std::size_t bestDistance = static_cast<std::size_t>(-1);
        bool disagree = false;
        for (const LinkInput& link : links) {
            std::size_t hLink = kNoSample;
            std::size_t vLink = kNoSample;
            if (link.fiberA == hIndex && link.fiberB == vIndex) {
                hLink = link.pointA;
                vLink = link.pointB;
            } else if (link.fiberB == hIndex && link.fiberA == vIndex) {
                hLink = link.pointB;
                vLink = link.pointA;
            } else {
                continue;
            }
            if (hLink >= h.psi.size() || vLink >= v.psi.size()) {
                continue;
            }
            const std::size_t distance = hLink > hSample ? hLink - hSample : hSample - hLink;
            if (distance < bestDistance) {
                bestDistance = distance;
                anchor.hLinkSample = hLink;
                anchor.vSample = vLink;
                disagree = false;
            } else if (distance == bestDistance && anchor.vSample != vLink) {
                disagree = true;
            }
        }
        if (anchor.hLinkSample == kNoSample || disagree) {
            continue;
        }
        anchors.push_back(anchor);
    }
    return anchors;
}

PairCrossings classifyPairCrossings(const PairDetections& detections,
                                    const CanonicalTrace& hTrace,
                                    const CanonicalTrace& vTrace,
                                    const std::vector<SeamAnchor>& seams,
                                    const SolverParams& params)
{
    PairCrossings result;
    result.gatedSegmentCount = detections.gatedSegmentCount;
    result.tangentialCount = detections.tangentialCount;
    result.unresolvedCount = detections.unresolvedCount;
    if (hTrace.psi.empty() || vTrace.psi.empty()) {
        return result;
    }
    const bool trusted = hTrace.trusted && vTrace.trusted;
    const std::vector<double>& hPsi = hTrace.psi;
    const std::vector<double>& hZ = hTrace.z;
    std::vector<Crossing> raw = detections.raw;
    std::vector<Crossing> shallow = detections.shallow;
    const std::size_t detectionCount = detections.detectionCount;
    const std::set<long long> gapTranslates(detections.gapTranslates.begin(),
                                            detections.gapTranslates.end());
    const std::set<long long> unresolvedTranslates(detections.unresolvedTranslates.begin(),
                                                   detections.unresolvedTranslates.end());

    // Pair-local sort and merge of the transversal detections into the
    // representatives the legacy constraint path is built from, unchanged:
    // one physical traversal seen by several segment pairs (or twice across
    // a branch split) is one piece of evidence. Stable, so equal-key
    // crossings keep deterministic encounter order - the constraint index
    // downstream breaks repair ties.
    std::vector<std::size_t> order(raw.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&raw](std::size_t a, std::size_t b) {
                         const Crossing& ca = raw[a];
                         const Crossing& cb = raw[b];
                         return std::tie(ca.n, ca.zVx, ca.deltaR) <
                                std::tie(cb.n, cb.zVx, cb.deltaR);
                     });
    // One physical traversal seen twice has nearly the same z AND nearly the
    // same radial separation. The kind is deliberately not part of the
    // identity - duplicate detections straddling the (confidence-scale) band
    // must merge, not turn into a manufactured conflict - while the deltaR
    // gate keeps genuinely distinct traversals apart (two branches of a
    // U-shaped fiber can share z, n and kind at wildly different radii). A
    // radially distinct traversal interleaved in z must not split a cluster,
    // so mismatches within the z window are skipped over, not treated as the
    // cluster's end.
    // Per detection id: its representative, for the events below.
    std::vector<std::size_t> representativeOf(detectionCount, 0);
    std::vector<std::vector<std::size_t>> clusterDetections;
    // Clusters first (over raw indices, in sorted order), then the seam
    // reading, then the representatives, so a reclassified encounter is
    // represented by its new reading whatever the confidences were.
    std::vector<std::vector<std::size_t>> clusterRawIndices;
    {
        std::vector<char> consumed(order.size(), 0);
        for (std::size_t index = 0; index < order.size(); ++index) {
            if (consumed[index]) {
                continue;
            }
            const Crossing& first = raw[order[index]];
            consumed[index] = 1;
            std::vector<std::size_t> cluster{order[index]};
            for (std::size_t scan = index + 1; scan < order.size(); ++scan) {
                const Crossing& next = raw[order[scan]];
                if (next.n != first.n || next.zVx - first.zVx > params.zMergeVx) {
                    break;
                }
                if (consumed[scan] ||
                    std::abs(next.deltaR - first.deltaR) > params.tieBandVx) {
                    continue;
                }
                consumed[scan] = 1;
                cluster.push_back(order[scan]);
            }
            clusterRawIndices.push_back(std::move(cluster));
        }
    }

    // The kollesis seam encounter of a tagged H end (see SeamAnchor). The
    // encounter is named by a V branch and a 2*pi translate: the branches
    // holding the linked V sample's vertex (two at a fold apex) and the
    // translate that lifts the linked H sample onto it; when none of those
    // saw an encounter - the annotator having linked to the V's nearest
    // control on another of its height folds - every limb with a detection
    // on the translate the linked H sample lifts to at that limb (at the
    // linked control's height clamped to the limb, so a control that
    // climbed past the V still marks the encounter). On each candidate branch
    // the detection nearest the linked H sample along the H fiber,
    // transversal or shallow, is that branch's encounter; the seam is the
    // encounter nearest the link, and where two limbs meet the H fiber at one place (a V folded
    // in height at one angle) the limb the end actually sits against: the
    // radially thinnest, and at an exact tie in thickness the one read
    // Outside, the reading a seam falsifies. (The residual tolerance ranks
    // limbs against the best, it is no overrun limit: a lone limb reads
    // however far the end overran.) The encounter is read as a whole - the detection's
    // proximity cluster and the detections within the merge height AND the
    // radial tie band of it, the same nearness the merge itself uses, so a
    // radially distinct crossing at the same height stays what it is - as
    // Inside: the glued sheets are one winding, and "same or inward" is true
    // whichever sheet the V is on. Nothing is read where no detection sits
    // on the link's own translate (an H that first meets the V a turn later
    // keeps that crossing's radial reading).
    constexpr double kSeamResidualTieTurns = 0.05;
    constexpr double kSeamAlongTieSamples = 0.5;
    const auto readSeam = [&](const SeamAnchor& anchor) {
        if (anchor.hSample >= hPsi.size() || anchor.hLinkSample >= hPsi.size() ||
            anchor.vSample >= vTrace.psi.size()) {
            return;
        }
        const double zLink = hZ[anchor.hLinkSample];
        const double psiLink = hPsi[anchor.hLinkSample];
        const double hPosition = static_cast<double>(anchor.hLinkSample);
        // Position along the H polyline, in samples.
        const auto along = [](const Crossing& detection) {
            return static_cast<double>(detection.hSegment) + detection.hT;
        };
        // The detection of (translate, branch) nearest the linked H sample,
        // or null; ties go to the transversal record, then the earlier one.
        const auto nearestOn = [&](long long n, std::size_t branch) -> const Crossing* {
            const Crossing* nearest = nullptr;
            double bestDistance = std::numeric_limits<double>::infinity();
            for (const std::vector<Crossing>* list : {&raw, &shallow}) {
                for (const Crossing& detection : *list) {
                    if (detection.n != n || detection.vBranch != branch) {
                        continue;
                    }
                    const double distance = std::abs(along(detection) - hPosition);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        nearest = &detection;
                    }
                }
            }
            return nearest;
        };
        struct Candidate {
            std::size_t branch = 0;
            long long n = 0;
            double residual = 0.0;
            const Crossing* nearest = nullptr;
            double distance = 0.0;
        };
        const auto candidateOn = [&](std::size_t branch, long long n, double residual) {
            Candidate candidate{branch, n, residual, nearestOn(n, branch), 0.0};
            if (candidate.nearest != nullptr) {
                candidate.distance = std::abs(along(*candidate.nearest) - hPosition);
            }
            return candidate;
        };
        std::vector<Candidate> candidates;
        const auto anyEncounter = [&]() {
            return std::any_of(candidates.begin(), candidates.end(),
                               [](const Candidate& c) { return c.nearest != nullptr; });
        };
        {
            // The linked V sample's vertex (the run of identical projected
            // samples it belongs to), and every branch holding that vertex.
            std::size_t vertex = 0;
            for (std::size_t i = 1; i <= anchor.vSample; ++i) {
                if (vTrace.psi[i] != vTrace.psi[i - 1] || vTrace.z[i] != vTrace.z[i - 1]) {
                    ++vertex;
                }
            }
            const double turns = (vTrace.psi[anchor.vSample] - psiLink) / kTwoPi;
            const long long n = static_cast<long long>(std::llround(turns));
            for (std::size_t b = 0; b < vTrace.branches.size(); ++b) {
                const Branch& branch = vTrace.branches[b];
                if (std::find(branch.vertexId.begin(), branch.vertexId.end(), vertex) ==
                    branch.vertexId.end()) {
                    continue;
                }
                candidates.push_back(candidateOn(b, n, 0.0));
            }
        }
        if (!anyEncounter()) {
            // Every (translate, limb) that saw a detection, with the residual
            // of lifting the linked H sample onto that limb at its height
            // (clamped to the limb's range: the linked control may sit above
            // or below the V - a control that climbed past it, say - and
            // still mark this encounter).
            candidates.clear();
            std::set<std::pair<long long, std::size_t>> seen;
            for (const std::vector<Crossing>* list : {&raw, &shallow}) {
                for (const Crossing& detection : *list) {
                    if (!seen.insert({detection.n, detection.vBranch}).second) {
                        continue;
                    }
                    const Branch& branch = vTrace.branches[detection.vBranch];
                    if (branch.z.size() < 2) {
                        continue;
                    }
                    const double zAt = std::clamp(zLink, branch.z.front(), branch.z.back());
                    const auto upper = std::lower_bound(branch.z.begin(), branch.z.end(), zAt);
                    std::size_t j = static_cast<std::size_t>(upper - branch.z.begin());
                    j = j > 0 ? j - 1 : 0;
                    if (j + 1 >= branch.z.size()) {
                        j = branch.z.size() - 2;
                    }
                    const double span = branch.z[j + 1] - branch.z[j];
                    const double u =
                        span > 0.0 ? std::clamp((zAt - branch.z[j]) / span, 0.0, 1.0) : 0.0;
                    const double psiV = branch.psi[j] + u * (branch.psi[j + 1] - branch.psi[j]);
                    const double turns = (psiV - psiLink) / kTwoPi;
                    // Only the translate the link lifts to on this limb: a
                    // detection of another turn is another encounter.
                    if (std::llround(turns) != detection.n) {
                        continue;
                    }
                    candidates.push_back(candidateOn(
                        detection.vBranch, detection.n,
                        std::abs(turns - static_cast<double>(detection.n))));
                }
            }
        }
        // Only branches that saw the encounter; among those with the
        // smallest lift residual (within kSeamResidualTieTurns of the best),
        // the encounters nearest the linked H sample along the H fiber (within
        // kSeamAlongTieSamples of the nearest - one place), and of those the
        // radially thinnest, Outside before Inside at an exact tie. Each cut
        // is against the best of the previous, never chained pairwise; only
        // two encounters at one place with one and the same deltaR are left
        // to limb order.
        const Candidate* chosen = nullptr;
        double bestResidual = std::numeric_limits<double>::infinity();
        for (const Candidate& candidate : candidates) {
            if (candidate.nearest != nullptr) {
                bestResidual = std::min(bestResidual, candidate.residual);
            }
        }
        const auto residualEligible = [&](const Candidate& candidate) {
            return candidate.nearest != nullptr &&
                   candidate.residual <= bestResidual + kSeamResidualTieTurns;
        };
        double bestDistance = std::numeric_limits<double>::infinity();
        for (const Candidate& candidate : candidates) {
            if (residualEligible(candidate)) {
                bestDistance = std::min(bestDistance, candidate.distance);
            }
        }
        for (const Candidate& candidate : candidates) {
            if (!residualEligible(candidate) ||
                candidate.distance > bestDistance + kSeamAlongTieSamples) {
                continue;
            }
            if (chosen == nullptr) {
                chosen = &candidate;
                continue;
            }
            const double thickness = std::abs(candidate.nearest->deltaR);
            const double chosenThickness = std::abs(chosen->nearest->deltaR);
            if (thickness < chosenThickness ||
                (thickness == chosenThickness &&
                 candidate.nearest->deltaR > chosen->nearest->deltaR)) {
                chosen = &candidate;
            }
        }
        if (chosen == nullptr) {
            return;
        }
        const long long seamTranslate = chosen->n;
        const std::size_t seamBranch = chosen->branch;
        const Crossing& seed = *chosen->nearest;
        // Within the merge's own nearness of the seed: the same encounter.
        const auto partOfEncounter = [&](const Crossing& detection) {
            return detection.n == seamTranslate && detection.vBranch == seamBranch &&
                   std::abs(detection.zVx - seed.zVx) <= params.zMergeVx &&
                   std::abs(detection.deltaR - seed.deltaR) <= params.tieBandVx;
        };
        const auto readAsSeam = [&](Crossing& detection) {
            detection.kollesis = true;
            detection.kind = CrossingKind::Inside;
            if (!detection.tangential) {
                detection.confidence = 0.9 * detection.transversality;
                if (!trusted) {
                    detection.confidence *= params.untrustedConfidenceFactor;
                }
            }
        };
        // The encounter: the seed's own cluster when the seed is transversal
        // (a cluster is one encounter by construction), and every detection,
        // transversal or shallow, within the merge's nearness of the seed.
        for (const std::vector<std::size_t>& cluster : clusterRawIndices) {
            const bool seeded = std::any_of(
                cluster.begin(), cluster.end(),
                [&](std::size_t rawIndex) { return &raw[rawIndex] == &seed; });
            if (!seeded) {
                continue;
            }
            for (const std::size_t rawIndex : cluster) {
                readAsSeam(raw[rawIndex]);
            }
        }
        for (Crossing& detection : raw) {
            if (partOfEncounter(detection)) {
                readAsSeam(detection);
            }
        }
        for (Crossing& detection : shallow) {
            if (partOfEncounter(detection)) {
                readAsSeam(detection);
            }
        }
    };
    for (const SeamAnchor& anchor : seams) {
        readSeam(anchor);
    }

    for (const std::vector<std::size_t>& cluster : clusterRawIndices) {
        std::size_t best = cluster.front();
        for (const std::size_t rawIndex : cluster) {
            if (raw[rawIndex].confidence > raw[best].confidence) {
                best = rawIndex;
            }
        }
        Crossing representative = raw[best];
        representative.mergedCount = static_cast<int>(cluster.size());
        representative.confidence = std::min(
            2.0, representative.confidence *
                     (1.0 + 0.25 * static_cast<double>(representative.mergedCount - 1)));
        std::vector<std::size_t> detectionIds;
        detectionIds.reserve(cluster.size());
        for (const std::size_t rawIndex : cluster) {
            detectionIds.push_back(raw[rawIndex].detection);
            representativeOf[raw[rawIndex].detection] = result.crossings.size();
        }
        clusterDetections.push_back(std::move(detectionIds));
        result.crossings.push_back(representative);
    }
    // Shallow detections follow the representatives, in the same
    // deterministic order, each standing for itself.
    std::stable_sort(shallow.begin(), shallow.end(),
                     [](const Crossing& a, const Crossing& b) {
                         return std::tie(a.n, a.zVx, a.deltaR) <
                                std::tie(b.n, b.zVx, b.deltaR);
                     });
    for (const Crossing& event : shallow) {
        representativeOf[event.detection] = result.crossings.size();
        clusterDetections.push_back({event.detection});
        result.crossings.push_back(event);
    }
    // The raw record per detection id, for representatives that end up
    // standing for covered and uncovered detections alike.
    std::vector<const Crossing*> recordOf(detectionCount, nullptr);
    for (const Crossing& detection : raw) {
        recordOf[detection.detection] = &detection;
    }
    for (const Crossing& detection : shallow) {
        recordOf[detection.detection] = &detection;
    }

    // Resolved events: every detection, with the two records of a V vertex
    // shared by two branches (a fold apex, or any vertex the ownership rule
    // hands to both) collapsed into one event when their orientations agree
    // - the H fiber passes straight through the vertex - and both flagged as
    // touches when they oppose: the V fiber came up to the H fiber at its
    // apex and retraced, crossing nothing.
    const auto sameVertexHit = [](const Crossing& a, const Crossing& b) {
        return a.vSample != Crossing::kNoSample && a.vSample == b.vSample &&
               a.n == b.n && a.hSegment == b.hSegment && a.hT == b.hT;
    };
    std::vector<Crossing> events;
    // Per detection id: the event it is part of.
    std::vector<std::size_t> eventOfDetection(detectionCount, 0);
    {
        std::vector<Crossing> all;
        all.reserve(raw.size() + shallow.size());
        all.insert(all.end(), raw.begin(), raw.end());
        all.insert(all.end(), shallow.begin(), shallow.end());
        std::vector<std::size_t> byHit(all.size());
        for (std::size_t i = 0; i < byHit.size(); ++i) {
            byHit[i] = i;
        }
        std::stable_sort(byHit.begin(), byHit.end(),
                         [&all](std::size_t a, std::size_t b) {
                             const Crossing& ca = all[a];
                             const Crossing& cb = all[b];
                             return std::tie(ca.n, ca.hSegment, ca.hT, ca.vSample, ca.vBranch) <
                                    std::tie(cb.n, cb.hSegment, cb.hT, cb.vSample, cb.vBranch);
                         });
        // Per record: the record it was collapsed into (itself when kept).
        std::vector<std::size_t> keptRecord(all.size());
        for (std::size_t i = 0; i < keptRecord.size(); ++i) {
            keptRecord[i] = i;
        }
        for (std::size_t k = 1; k < byHit.size(); ++k) {
            Crossing& a = all[byHit[k - 1]];
            Crossing& b = all[byHit[k]];
            if (!sameVertexHit(a, b) || keptRecord[byHit[k - 1]] != byHit[k - 1]) {
                continue;
            }
            if (a.orientation == b.orientation) {
                // One event: keep the more confident record (then the lower
                // branch), standing for both detections.
                const bool keepA = a.confidence > b.confidence ||
                                   (a.confidence == b.confidence && a.vBranch <= b.vBranch);
                Crossing& kept = keepA ? a : b;
                kept.mergedCount = a.mergedCount + b.mergedCount;
                keptRecord[keepA ? byHit[k] : byHit[k - 1]] = keepA ? byHit[k - 1] : byHit[k];
            } else {
                a.touch = true;
                b.touch = true;
            }
        }
        std::vector<std::size_t> eventOfRecord(all.size(), 0);
        std::vector<std::size_t> keptOrder;
        for (std::size_t i = 0; i < all.size(); ++i) {
            if (keptRecord[i] == i) {
                keptOrder.push_back(i);
            }
        }
        std::stable_sort(keptOrder.begin(), keptOrder.end(),
                         [&all](std::size_t a, std::size_t b) {
                             const Crossing& ca = all[a];
                             const Crossing& cb = all[b];
                             return std::tie(ca.n, ca.vBranch, ca.zVx, ca.hSegment, ca.hT, ca.vSample) <
                                    std::tie(cb.n, cb.vBranch, cb.zVx, cb.hSegment, cb.hT, cb.vSample);
                         });
        for (const std::size_t record : keptOrder) {
            eventOfRecord[record] = events.size();
            Crossing event = all[record];
            event.representative = representativeOf[event.detection];
            events.push_back(std::move(event));
        }
        for (std::size_t i = 0; i < all.size(); ++i) {
            eventOfDetection[all[i].detection] = eventOfRecord[keptRecord[i]];
        }
    }

    // Traversal groups: every (translate, V branch) with at least two
    // counted events, counted over the events themselves.
    std::map<std::pair<long long, std::size_t>, std::vector<std::size_t>> eventsByKey;
    for (std::size_t e = 0; e < events.size(); ++e) {
        // Touches cross nothing; seam encounters are annotation-classified,
        // not radial evidence. Neither counts.
        if (events[e].touch || events[e].kollesis) {
            continue;
        }
        eventsByKey[{events[e].n, events[e].vBranch}].push_back(e);
    }
    // Which side of the branch's angular locus an H endpoint lies on, for
    // translate n: +1 / -1 when the endpoint clears the locus by the
    // clearance, 0 when it is too close; `covered` false when the branch does
    // not span that height, in which case the traversal's completeness is
    // unknown and the group takes no verdict.
    const double clearance = params.endpointClearanceTurns * kTwoPi;
    const auto endpointSide = [&](std::size_t sample, long long n, const Branch& branch,
                                  bool& covered) {
        const double z = hZ[sample];
        const double psi = hPsi[sample] + kTwoPi * static_cast<double>(n);
        covered = false;
        if (branch.z.size() < 2 || z < branch.z.front() || z > branch.z.back()) {
            return 0;
        }
        covered = true;
        const auto upper = std::lower_bound(branch.z.begin(), branch.z.end(), z);
        std::size_t j = static_cast<std::size_t>(upper - branch.z.begin());
        j = j > 0 ? j - 1 : 0;
        if (j + 1 >= branch.z.size()) {
            j = branch.z.size() - 2;
        }
        const double span = branch.z[j + 1] - branch.z[j];
        const double u = span > 0.0 ? std::clamp((z - branch.z[j]) / span, 0.0, 1.0) : 0.0;
        const double psiV = branch.psi[j] + u * (branch.psi[j + 1] - branch.psi[j]);
        const double d = psi - psiV;
        if (std::abs(d) < clearance) {
            return 0;
        }
        return d > 0.0 ? 1 : -1;
    };
    for (const auto& [key, members] : eventsByKey) {
        if (members.size() < 2) {
            continue;
        }
        const long long n = key.first;
        const Branch& branch = vTrace.branches[key.second];
        CrossingGroup group;
        group.n = n;
        group.vBranch = key.second;
        group.members = members;
        group.multiplicity = static_cast<int>(members.size());
        group.minAbsDeltaR = std::numeric_limits<double>::infinity();
        bool anyInside = false;
        bool anyOutside = false;
        double transversalitySum = 0.0;
        for (const std::size_t e : members) {
            const Crossing& event = events[e];
            const bool inside = event.kind == CrossingKind::Inside;
            anyInside = anyInside || inside;
            anyOutside = anyOutside || !inside;
            group.orientationSum += event.orientation;
            if (inside) {
                ++group.insideCount;
                group.insideOrientationSum += event.orientation;
            }
            if (event.deltaR == 0.0) {
                group.onCurtain = true;
            }
            group.minAbsDeltaR = std::min(group.minAbsDeltaR, std::abs(event.deltaR));
            transversalitySum += event.transversality;
        }
        group.meanTransversality = transversalitySum / static_cast<double>(members.size());
        group.mixedSigns = anyInside && anyOutside;
        group.coverageGap = gapTranslates.count(n) != 0;
        group.unresolved = unresolvedTranslates.count(n) != 0;
        // Completeness of the count is decided where the curtain is: the
        // stretch of the H trace whose lifted angle lies within the branch's
        // angular window (its psi range plus the clearance) at this
        // translate. Every sample of that stretch must stay within the
        // branch's height range - an excursion above or below it, at the V
        // fiber's angle, could cross the fiber's untraced continuation unseen
        // and come back with the count off by two - and the stretch must be
        // entered from one side of the window and left to the other. A
        // stretch that begins or ends at the trace's own end is judged there
        // by the local side test against the V fiber's angle at that height.
        // A multi-turn H fiber therefore passes on each turn that crosses the
        // V fiber cleanly, whatever it does elsewhere.
        // The stretch is taken segment by segment, each clipped to the window,
        // so a single long segment jumping across the V fiber's angle is seen
        // whether or not a sample lands inside; the clipped ends' heights
        // bound the segment's heights inside the window (it is straight).
        {
            const double windowLo = branch.psiMin - clearance;
            const double windowHi = branch.psiMax + clearance;
            const double windowMid = 0.5 * (branch.psiMin + branch.psiMax);
            const double lift = kTwoPi * static_cast<double>(n);
            const double zLoBranch = branch.z.front();
            const double zHiBranch = branch.z.back();
            bool excursion = false;
            bool any = false;
            int sideA = 0;
            int sideB = 0;
            for (std::size_t i = 0; i + 1 < hPsi.size(); ++i) {
                const double a = hPsi[i] + lift;
                const double b = hPsi[i + 1] + lift;
                const double segLo = std::min(a, b);
                const double segHi = std::max(a, b);
                if (segHi < windowLo || segLo > windowHi) {
                    continue;
                }
                // Parameter range of the segment inside the window.
                double t0 = 0.0;
                double t1 = 1.0;
                if (b != a) {
                    const double tLo = (windowLo - a) / (b - a);
                    const double tHi = (windowHi - a) / (b - a);
                    t0 = std::clamp(std::min(tLo, tHi), 0.0, 1.0);
                    t1 = std::clamp(std::max(tLo, tHi), 0.0, 1.0);
                }
                const double z0 = hZ[i] + t0 * (hZ[i + 1] - hZ[i]);
                const double z1 = hZ[i] + t1 * (hZ[i + 1] - hZ[i]);
                if (z0 < zLoBranch || z0 > zHiBranch || z1 < zLoBranch || z1 > zHiBranch) {
                    excursion = true;
                }
                if (!any) {
                    // Entering: from the side the segment's start lies on, or,
                    // when the trace itself begins inside the window, by the
                    // local test at its first sample.
                    if (a < windowLo || a > windowHi) {
                        sideA = a > windowMid ? 1 : -1;
                    } else {
                        bool covered = false;
                        sideA = endpointSide(0, n, branch, covered);
                        sideA = covered ? sideA : 0;
                    }
                }
                any = true;
                // Leaving (updated at every overlapping segment; the last one
                // stands): to the side the segment's end lies on, or the local
                // test at the trace's last sample when it ends inside.
                if (b < windowLo || b > windowHi) {
                    sideB = b > windowMid ? 1 : -1;
                } else if (i + 2 == hPsi.size()) {
                    bool covered = false;
                    sideB = endpointSide(hPsi.size() - 1, n, branch, covered);
                    sideB = covered ? sideB : 0;
                } else {
                    sideB = 0;
                }
            }
            group.traversalCovered =
                any && !excursion && sideA != 0 && sideB != 0 && sideA != sideB;
        }
        group.hasVerdict = group.multiplicity >= 3 && group.mixedSigns &&
                           (group.orientationSum % 2 != 0) && !group.coverageGap &&
                           !group.unresolved && !group.onCurtain && group.traversalCovered;
        if (group.hasVerdict) {
            group.verdict = (group.insideCount % 2 == 1) ? CrossingKind::Inside
                                                         : CrossingKind::Outside;
            group.confidence =
                std::clamp(group.minAbsDeltaR / (3.0 * std::max(params.tieBandVx, 1e-9)),
                           0.0, 1.0) *
                group.meanTransversality;
            if (!trusted) {
                group.confidence *= params.untrustedConfidenceFactor;
            }
        }
        const long long groupIndex = static_cast<long long>(result.groups.size());
        for (const std::size_t e : members) {
            events[e].groupIndex = groupIndex;
        }
        result.groups.push_back(std::move(group));
    }

    // Representatives against the groups: one standing only for detections
    // whose events sit in verdict groups is covered (the groups constrain in
    // its place); one standing for covered and uncovered detections alike
    // constrains for the uncovered ones only - it becomes the merge's
    // representative of just those: the most confident uncovered record (the
    // earliest in cluster order on a tie, as the merge picks), boosted by
    // their count. Its display group is its own detection's.
    const auto verdictOf = [&](std::size_t detection) {
        const long long g = events[eventOfDetection[detection]].groupIndex;
        return g >= 0 && result.groups[static_cast<std::size_t>(g)].hasVerdict;
    };
    for (std::size_t r = 0; r < result.crossings.size(); ++r) {
        Crossing& representative = result.crossings[r];
        representative.groupIndex = events[eventOfDetection[representative.detection]].groupIndex;
        int uncovered = 0;
        const Crossing* best = nullptr;
        for (const std::size_t detection : clusterDetections[r]) {
            if (verdictOf(detection)) {
                continue;
            }
            ++uncovered;
            if (best == nullptr || recordOf[detection]->confidence > best->confidence) {
                best = recordOf[detection];
            }
        }
        if (uncovered == 0) {
            representative.coveredByGroups = true;
        } else if (uncovered < static_cast<int>(clusterDetections[r].size()) &&
                   !representative.tangential && best != nullptr) {
            const long long displayGroup = representative.groupIndex;
            representative = *best;
            representative.mergedCount = uncovered;
            representative.confidence = std::min(
                2.0, best->confidence * (1.0 + 0.25 * static_cast<double>(uncovered - 1)));
            representative.groupIndex = displayGroup;
        }
    }
    result.events = std::move(events);
    return result;
}

bool identicalPairCrossings(const PairCrossings& a, const PairCrossings& b)
{
    const auto sameDouble = [](double x, double y) {
        return std::memcmp(&x, &y, sizeof(double)) == 0;
    };
    const auto sameCrossing = [&](const Crossing& x, const Crossing& y) {
        return x.hFiber == y.hFiber && x.vFiber == y.vFiber &&
               sameDouble(x.violationTurns, y.violationTurns) && sameDouble(x.zVx, y.zVx) &&
               sameDouble(x.psiH, y.psiH) && x.n == y.n && sameDouble(x.deltaR, y.deltaR) &&
               sameDouble(x.confidence, y.confidence) && x.mergedCount == y.mergedCount &&
               x.kind == y.kind && x.status == y.status &&
               sameDouble(x.transversality, y.transversality) && x.tangential == y.tangential &&
               x.orientation == y.orientation && x.hSegment == y.hSegment &&
               sameDouble(x.hT, y.hT) && x.vSample == y.vSample && x.touch == y.touch &&
               x.vBranch == y.vBranch && x.detection == y.detection &&
               x.representative == y.representative && x.kollesis == y.kollesis &&
               x.coveredByGroups == y.coveredByGroups && x.groupIndex == y.groupIndex;
    };
    const auto sameGroup = [&](const CrossingGroup& x, const CrossingGroup& y) {
        return x.hFiber == y.hFiber && x.vFiber == y.vFiber && x.n == y.n &&
               x.vBranch == y.vBranch && x.members == y.members &&
               x.multiplicity == y.multiplicity && x.insideCount == y.insideCount &&
               x.orientationSum == y.orientationSum &&
               x.insideOrientationSum == y.insideOrientationSum &&
               x.mixedSigns == y.mixedSigns && x.coverageGap == y.coverageGap &&
               x.unresolved == y.unresolved && x.onCurtain == y.onCurtain &&
               x.traversalCovered == y.traversalCovered &&
               sameDouble(x.minAbsDeltaR, y.minAbsDeltaR) &&
               sameDouble(x.meanTransversality, y.meanTransversality) &&
               x.hasVerdict == y.hasVerdict && x.verdict == y.verdict &&
               sameDouble(x.confidence, y.confidence) && x.status == y.status &&
               sameDouble(x.violationTurns, y.violationTurns);
    };
    if (a.crossings.size() != b.crossings.size() || a.events.size() != b.events.size() ||
        a.groups.size() != b.groups.size() || a.gatedSegmentCount != b.gatedSegmentCount ||
        a.tangentialCount != b.tangentialCount || a.unresolvedCount != b.unresolvedCount) {
        return false;
    }
    for (std::size_t i = 0; i < a.crossings.size(); ++i) {
        if (!sameCrossing(a.crossings[i], b.crossings[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.events.size(); ++i) {
        if (!sameCrossing(a.events[i], b.events[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < a.groups.size(); ++i) {
        if (!sameGroup(a.groups[i], b.groups[i])) {
            return false;
        }
    }
    return true;
}

SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                          const std::vector<LinkInput>& links,
                          const SolverParams& params)
{
    const int chirality = inferChirality(fibers, params.chiralityOverride);
    const auto detectBegin = std::chrono::steady_clock::now();
    std::vector<CanonicalTrace> canonical(fibers.size());
    for (std::size_t f = 0; f < fibers.size(); ++f) {
        canonical[f] = canonicalizeTrace(fibers[f], chirality);
    }
    std::deque<PairCrossings> shards;
    std::vector<PairDetection> detections;
    for (std::size_t h = 0; h < fibers.size(); ++h) {
        if (canonical[h].hvTag != 'H' || canonical[h].psi.empty()) {
            continue;
        }
        for (std::size_t v = 0; v < fibers.size(); ++v) {
            if (canonical[v].hvTag != 'V' || canonical[v].psi.empty()) {
                continue;
            }
            shards.push_back(classifyPairCrossings(
                detectPairCrossings(canonical[h], canonical[v], params), canonical[h],
                canonical[v], seamAnchors(canonical, h, v, links), params));
            detections.push_back(PairDetection{h, v, &shards.back()});
        }
    }
    const double detectMs = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - detectBegin)
                                .count();
    SolveResult result = solveWindings(fibers, links, params, chirality, detections);
    result.detectMs += detectMs;
    return result;
}

SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                          const std::vector<LinkInput>& links,
                          const SolverParams& params,
                          const int chirality,
                          const std::vector<PairDetection>& detections)
{
    SolveResult result;
    const std::size_t count = fibers.size();
    result.placements.assign(count, Placement{});
    result.linkTurnErrors.assign(links.size(),
                                 std::numeric_limits<double>::infinity());
    if (count == 0) {
        return result;
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
    std::vector<char> finite(count, 0);
    for (std::size_t f = 0; f < count; ++f) {
        finite[f] = traceValuesFinite(fibers[f]) ? 1 : 0;
    }
    std::vector<std::vector<double>> psi(count);
    std::vector<long long> gauge(count, 0);
    for (std::size_t f = 0; f < count; ++f) {
        psi[f].resize(fibers[f].theta.size());
        for (std::size_t i = 0; i < fibers[f].theta.size(); ++i) {
            psi[f][i] = chirality * fibers[f].theta[i];
        }
        if (!psi[f].empty() && finite[f] != 0) {
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
               fibers[f].z.size() == fibers[f].theta.size() && finite[f] != 0;
    };

    const auto detectBegin = std::chrono::steady_clock::now();
    // --- Assemble the detection shards in canonical (hFiber, vFiber) order:
    // each shard is internally merged and (n, z, deltaR)-sorted, and the
    // pair indices lead the global sort key, so the concatenation IS the
    // globally sorted merged crossing list - constraint order is therefore
    // identical however the shards were produced (fresh or cached).
    std::vector<const PairDetection*> ordered_detections;
    ordered_detections.reserve(detections.size());
    for (const PairDetection& detection : detections) {
        // Defensive: a null shard or an out-of-range endpoint would corrupt
        // the solve silently; such a shard is a caller bug and is skipped.
        if (detection.detection == nullptr || detection.hFiber >= count ||
            detection.vFiber >= count) {
            continue;
        }
        ordered_detections.push_back(&detection);
    }
    std::stable_sort(ordered_detections.begin(), ordered_detections.end(),
                     [](const PairDetection* a, const PairDetection* b) {
                         return std::tie(a->hFiber, a->vFiber) <
                                std::tie(b->hFiber, b->vFiber);
                     });
    std::vector<Crossing> merged;
    std::vector<Crossing>& events = result.events;
    std::vector<CrossingGroup>& groups = result.groups;
    for (const PairDetection* detection : ordered_detections) {
        result.gatedSegmentCount += detection->detection->gatedSegmentCount;
        result.tangentialCount += detection->detection->tangentialCount;
        result.unresolvedIntersectionCount += detection->detection->unresolvedCount;
        const std::size_t crossingBase = merged.size();
        const std::size_t eventBase = events.size();
        const std::size_t groupBase = groups.size();
        for (Crossing crossing : detection->detection->crossings) {
            crossing.hFiber = detection->hFiber;
            crossing.vFiber = detection->vFiber;
            if (crossing.groupIndex >= 0) {
                crossing.groupIndex += static_cast<long long>(groupBase);
            }
            merged.push_back(crossing);
        }
        for (Crossing event : detection->detection->events) {
            event.hFiber = detection->hFiber;
            event.vFiber = detection->vFiber;
            event.representative += crossingBase;
            if (event.groupIndex >= 0) {
                event.groupIndex += static_cast<long long>(groupBase);
            }
            if (event.kollesis) {
                ++result.kollesisCrossingCount;
            }
            events.push_back(event);
        }
        for (CrossingGroup group : detection->detection->groups) {
            group.hFiber = detection->hFiber;
            group.vFiber = detection->vFiber;
            for (std::size_t& member : group.members) {
                member += eventBase;
            }
            groups.push_back(std::move(group));
        }
    }
    const auto detectEnd = std::chrono::steady_clock::now();
    result.detectMs =
        std::chrono::duration<double, std::milli>(detectEnd - detectBegin).count();

    // --- Constraint graph.
    std::vector<Constraint> constraints;
    const auto addPair = [&constraints](std::size_t from, std::size_t to,
                                        long long weight, double confidence,
                                        SourceRef source) {
        // Equality: to - from == weight, as a mirrored pair sharing one fate.
        Constraint forward{from, to, weight, confidence, source, -1, true};
        Constraint backward{to, from, -weight, confidence, source, -1, true};
        forward.pair = static_cast<long long>(constraints.size() + 1);
        backward.pair = static_cast<long long>(constraints.size());
        constraints.push_back(forward);
        constraints.push_back(backward);
    };

    for (std::size_t c = 0; c < merged.size(); ++c) {
        Crossing& crossing = merged[c];
        const SourceRef source{SourceKind::Crossing, c};
        // A representative standing only for detections whose traversal
        // groups took a verdict does not constrain on its own; the groups do,
        // below.
        if (crossing.coveredByGroups) {
            crossing.status = CrossingStatus::InGroup;
            continue;
        }
        // A shallow pass is an event for the count and the record, never a
        // constraint on its own.
        if (crossing.tangential) {
            continue;
        }
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

    // Traversal groups with a verdict, one constraint each, after the
    // crossings and before the links - in shard order, so fresh and cached
    // builds emit identically.
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const CrossingGroup& group = groups[g];
        if (!group.hasVerdict) {
            continue;
        }
        const SourceRef groupSource{SourceKind::Group, g};
        if (group.verdict == CrossingKind::Inside) {
            // W_h <= W_v: k_v - k_h >= -n.
            constraints.push_back(Constraint{group.hFiber, group.vFiber, -group.n,
                                             group.confidence, groupSource, -1, true});
        } else {
            // W_h >= W_v + 1: k_h - k_v >= n + 1.
            constraints.push_back(Constraint{group.vFiber, group.hFiber, 1 + group.n,
                                             group.confidence, groupSource, -1, true});
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
                SourceRef{SourceKind::Link, l});
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
        const SourceRef source = constraints[ci].source;
        switch (source.kind) {
        case SourceKind::Crossing:
            merged[source.index].status = CrossingStatus::Dropped;
            ++result.droppedCrossingCount;
            break;
        case SourceKind::Link:
            result.droppedLinks.push_back(source.index);
            break;
        case SourceKind::Group:
            groups[source.index].status = CrossingStatus::Dropped;
            ++result.droppedGroupCount;
            break;
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
    // link-only network, however large, proves no winding. When no crossing
    // survived anywhere there is no primary at all, and every component runs
    // the island path against an empty anchored set - honestly unresolved.
    std::set<std::size_t> rootsWithCrossings;
    for (const Constraint& constraint : constraints) {
        if (constraint.active && constraint.source.kind != SourceKind::Link) {
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
    //
    // Everything k-independent is hoisted out of the pass loop: each block's
    // boundary constraints (bounds are integer max/min, so subsetting cannot
    // change them) and each member's ordinal pair list - the neighbourhood
    // test uses only z and arc, never k, so the pair set, its order, and
    // each pair's tie/strict classification are fixed for the whole ascent.
    // The per-candidate cost then evaluates the identical expressions over
    // the identical pairs in the identical order as the unhoisted form.
    const auto ascend = [&](const std::vector<std::size_t>& members) {
        std::map<std::size_t, std::vector<std::size_t>> blocks;
        for (const std::size_t f : members) {
            if (usable(f)) {
                blocks[blockOf[f]].push_back(f);
            }
        }
        // Boundary constraints per block, in constraint order.
        std::map<std::size_t, std::vector<const Constraint*>> boundary;
        for (const Constraint& constraint : constraints) {
            if (!constraint.active) {
                continue;
            }
            const std::size_t fromBlock = blockOf[constraint.from];
            const std::size_t toBlock = blockOf[constraint.to];
            if (fromBlock == toBlock) {
                continue;
            }
            if (blocks.count(fromBlock) != 0) {
                boundary[fromBlock].push_back(&constraint);
            }
            if (blocks.count(toBlock) != 0) {
                boundary[toBlock].push_back(&constraint);
            }
        }
        // Ordinal pairs per member, in ordinalCost's own iteration order.
        struct OrdinalPair {
            std::size_t fiber = 0;
            double wpBase = 0.0;
            double wqBase = 0.0;
            bool tie = false;
            bool outward = false;  // dr > 0 for strict pairs
        };
        std::map<std::size_t, std::vector<OrdinalPair>> pairsOf;
        for (const auto& [block, blockMembers] : blocks) {
            for (const std::size_t f : blockMembers) {
                std::vector<OrdinalPair>& list = pairsOf[f];
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
                            blockOf[q.fiber] == block) {
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
                        if (std::abs(dr) <= params.tieBandVx) {
                            list.push_back(OrdinalPair{q.fiber, p / kTwoPi,
                                                       q.psi / kTwoPi, true,
                                                       false});
                        } else if (std::abs(dr) > strictFloor) {
                            list.push_back(OrdinalPair{q.fiber, p / kTwoPi,
                                                       q.psi / kTwoPi, false,
                                                       dr > 0.0});
                        }
                    }
                }
            }
        }
        const auto pairCost = [&](const std::vector<OrdinalPair>& list,
                                  long long kf) {
            double cost = 0.0;
            for (const OrdinalPair& pair : list) {
                const double wp = pair.wpBase + static_cast<double>(kf);
                const double wq = pair.wqBase +
                                  static_cast<double>(k[pair.fiber]);
                const long long dw = std::llround(wp - wq);
                if (pair.tie) {
                    cost += 0.5 * static_cast<double>(std::min<long long>(
                                      std::llabs(dw), 2));
                } else if (dw == 0 || (dw > 0) != pair.outward) {
                    cost += 1.0;
                }
            }
            return cost;
        };
        for (int pass = 0; pass < kAscentPasses; ++pass) {
            bool changed = false;
            for (const auto& [block, blockMembers] : blocks) {
                long long deltaLo = -kAscentWindow;
                long long deltaHi = kAscentWindow;
                const auto boundaryIt = boundary.find(block);
                if (boundaryIt != boundary.end()) {
                    for (const Constraint* constraint : boundaryIt->second) {
                        if (blockOf[constraint->to] == block) {
                            deltaLo = std::max(deltaLo, k[constraint->from] +
                                                            constraint->weight -
                                                            k[constraint->to]);
                        } else {
                            deltaHi = std::min(deltaHi, k[constraint->to] -
                                                            constraint->weight -
                                                            k[constraint->from]);
                        }
                    }
                }
                if (deltaLo > 0 || deltaHi < 0 || deltaLo == deltaHi) {
                    continue;
                }
                const auto costAt = [&](long long delta) {
                    double cost = 0.0;
                    for (const std::size_t f : blockMembers) {
                        cost += pairCost(pairsOf[f], k[f] + delta);
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
    // Violation of each crossing against the final map, exactly: at the
    // crossing both fibers pass through the same lifted point, so
    // W_v(c) - W_h(c) = n + k_v - k_h with n the crossing's exact translate
    // integer - no geometry, no nearest-sample approximation.
    for (Crossing& crossing : merged) {
        const long long gap =
            crossing.n + k[crossing.vFiber] - k[crossing.hFiber];
        switch (crossing.kind) {
        case CrossingKind::Inside:   // demanded W_h <= W_v, i.e. gap >= 0
            crossing.violationTurns =
                static_cast<double>(std::max<long long>(0, -gap));
            break;
        case CrossingKind::Outside:  // demanded W_h >= W_v + 1, i.e. gap <= -1
            crossing.violationTurns =
                static_cast<double>(std::max<long long>(0, gap + 1));
            break;
        case CrossingKind::Tie:      // demanded W_h == W_v
            crossing.violationTurns = static_cast<double>(std::llabs(gap));
            break;
        }
    }
    // Events: their own violation, and the status of what constrained for
    // them - their group when it took a verdict, else their representative.
    for (Crossing& event : events) {
        const long long gap = event.n + k[event.vFiber] - k[event.hFiber];
        event.violationTurns = event.kind == CrossingKind::Inside
            ? static_cast<double>(std::max<long long>(0, -gap))
            : static_cast<double>(std::max<long long>(0, gap + 1));
        const bool grouped = event.groupIndex >= 0 &&
                             groups[static_cast<std::size_t>(event.groupIndex)].hasVerdict;
        event.status = grouped ? CrossingStatus::InGroup : merged[event.representative].status;
    }
    // A group's violation is against its verdict, the constraint it stood for.
    for (CrossingGroup& group : groups) {
        if (!group.hasVerdict) {
            continue;
        }
        const long long gap = group.n + k[group.vFiber] - k[group.hFiber];
        group.violationTurns = group.verdict == CrossingKind::Inside
            ? static_cast<double>(std::max<long long>(0, -gap))
            : static_cast<double>(std::max<long long>(0, gap + 1));
    }
    // Sheet drift: rule 1 constrains "that section" of an H fiber; one k per
    // fiber assumes the annotation stays on one sheet. Repeated drops against
    // DISTINCT evidence on one H fiber - different V fibers or different
    // turns - are the signature of that assumption failing, surfaced rather
    // than solved; two drops of one contested traversal are not.
    std::map<std::size_t, std::set<std::pair<std::size_t, long long>>> dropsPerFiber;
    for (const Crossing& crossing : merged) {
        // Only actually-violated drops are drift evidence: a drop the final
        // map satisfies anyway is repair debris, not evidence of anything.
        if (crossing.status == CrossingStatus::Dropped &&
            crossing.violationTurns >= params.declarationViolationTurns) {
            dropsPerFiber[crossing.hFiber].emplace(crossing.vFiber, crossing.n);
        }
    }
    for (const CrossingGroup& group : groups) {
        if (group.hasVerdict && group.status == CrossingStatus::Dropped &&
            group.violationTurns >= params.declarationViolationTurns) {
            dropsPerFiber[group.hFiber].emplace(group.vFiber, group.n);
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
    for (Crossing& event : events) {
        event.psiH += kTwoPi * static_cast<double>(gauge[event.hFiber]);
    }
    result.crossings = std::move(merged);
    result.solveMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - detectEnd)
                         .count();
    return result;
}

} // namespace vc3d::fiber_map::winding
