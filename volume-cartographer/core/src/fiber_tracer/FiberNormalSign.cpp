#include "vc/fiber_tracer/FiberNormalSign.hpp"

#include <vc_object_store/geometry/FiberCrossingWeight.hpp>

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vc::fiber_tracer
{

double fiberSignResidual(const double delta, const double target)
{
    // The period is two: even phases are one class, odd phases the other.
    double wrapped = std::fmod(delta - target + 1.0, 2.0);
    if (wrapped < 0.0) wrapped += 2.0;
    return wrapped - 1.0;
}

double fiberSignCost(const double residual, const double epsilon)
{
    const double magnitude = std::abs(residual);
    if (epsilon <= 0.0) return magnitude;
    return magnitude < epsilon ? 0.5 * residual * residual / epsilon : magnitude - 0.5 * epsilon;
}

double fiberSignWindingDistanceFactor(const double distance, const FiberSignDistanceWeight mode)
{
    // The shared rule (vc_object_store geometry/FiberCrossingWeight.hpp): the
    // winding graph weighs a crossing the same way when it is asked to.
    if (mode == FiberSignDistanceWeight::None) return 1.0;
    return vc::geometry::fiberCrossingDistanceFactor(distance);
}

double fiberSignEventWeight(const double dot, const double windingDistance, const FiberSignOptions& options)
{
    return fiberSignWindingDistanceFactor(windingDistance, options.distanceWeight)
           * vc::geometry::fiberCrossingDotWeight(dot, options.dotWeightPower);
}

namespace
{

constexpr double kTiny = std::numeric_limits<double>::min();
// Loops are split at a fixed granularity, so every reduction sums in the same
// order whatever the thread count: the solution does not depend on threads.
// Small enough that a few hundred thousand fibers still fill a worker pool.
constexpr std::size_t kChunk = 4096;

double sigmoid(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

template <typename Body>
void forChunks(const std::size_t count, const int threads, Body&& body)
{
    if (count == 0) return;
    const std::size_t chunks = (count + kChunk - 1) / kChunk;
    const auto run = [&](const std::size_t lo, const std::size_t hi) {
        for (std::size_t chunk = lo; chunk < hi; ++chunk)
            body(chunk * kChunk, std::min((chunk + 1) * kChunk, count));
    };
    if (threads <= 1 || chunks == 1) {
        run(0, chunks);
        return;
    }
    cv::parallel_for_(cv::Range(0, static_cast<int>(chunks)), [&](const cv::Range& range) {
        run(static_cast<std::size_t>(range.start), static_cast<std::size_t>(range.end));
    }, static_cast<double>(threads));
}

template <typename Body>
double chunkedSum(const std::size_t count, const int threads, Body&& body)
{
    if (count == 0) return 0.0;
    const std::size_t chunks = (count + kChunk - 1) / kChunk;
    std::vector<double> partial(chunks, 0.0);
    forChunks(count, threads, [&](const std::size_t lo, const std::size_t hi) {
        partial[lo / kChunk] = body(lo, hi);
    });
    double total = 0.0;
    for (const double value : partial) total += value;
    return total;
}

// The prepared sign problem: the graph, its weights and the current state.
struct Problem {
    std::size_t nodes = 0;
    std::size_t events = 0;
    int threads = 1;
    double epsilon = 0.05;

    std::vector<std::int64_t> ea, eb;
    std::vector<double> dot, distance;
    std::vector<double> target;
    std::vector<double> weight;     // normalized; drives the solve
    std::vector<double> rawWeight;  // |dot| times the factor; drives the metrics

    std::vector<double> priorTarget, priorWeight, priorConstant, rejection;
    std::vector<double> agreement;
    std::vector<double> phase, keep;

    // Incidence of every event on both of its nodes: a self event appears
    // twice on its node, as the reference's two scatter adds do.
    std::vector<std::int64_t> rowStart, incidentEvent, incidentPartner;
    std::vector<double> incidentSign;  // -1 on the a side, +1 on the b side
};

struct Terms {
    double pair = 0.0, prior = 0.0, broken = 0.0, total = 0.0;
};

Terms objectiveOf(const Problem& problem, const double epsilon)
{
    Terms terms;
    terms.pair = chunkedSum(problem.events, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
        double sum = 0.0;
        for (std::size_t e = lo; e < hi; ++e) {
            const std::int64_t a = problem.ea[e], b = problem.eb[e];
            const double residual = fiberSignResidual(problem.phase[b] - problem.phase[a], problem.target[e]);
            sum += problem.weight[e] * problem.keep[a] * problem.keep[b] * fiberSignCost(residual, epsilon);
        }
        return sum;
    });
    // The sign stage never masks the prior with keep: rejecting a fiber must
    // not also discount the prior that holds its sign.
    terms.prior = chunkedSum(problem.nodes, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
        double sum = 0.0;
        for (std::size_t v = lo; v < hi; ++v)
            sum += problem.priorWeight[v]
                       * fiberSignCost(fiberSignResidual(problem.phase[v], problem.priorTarget[v]), epsilon)
                   + problem.priorConstant[v];
        return sum;
    });
    terms.broken = chunkedSum(problem.nodes, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
        double sum = 0.0;
        for (std::size_t v = lo; v < hi; ++v) sum += problem.rejection[v] * (1.0 - problem.keep[v]);
        return sum;
    });
    terms.total = terms.pair + terms.prior + terms.broken;
    return terms;
}

// One bounded keep update from `previous`, with the diagonal majorizer that
// covers the coupling between the two nodes of an event.
void keepStep(Problem& problem, const std::vector<double>& previous, std::vector<double>& eventCost)
{
    const double epsilon = problem.epsilon;
    eventCost.resize(problem.events);
    forChunks(problem.events, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
        for (std::size_t e = lo; e < hi; ++e) {
            const double residual =
                fiberSignResidual(problem.phase[problem.eb[e]] - problem.phase[problem.ea[e]], problem.target[e]);
            eventCost[e] = problem.weight[e] * fiberSignCost(residual, epsilon);
        }
    });
    forChunks(problem.nodes, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
        for (std::size_t v = lo; v < hi; ++v) {
            double gradient = -problem.rejection[v];
            double diagonal = 0.0;
            for (std::int64_t k = problem.rowStart[v]; k < problem.rowStart[v + 1]; ++k) {
                const double cost = eventCost[problem.incidentEvent[k]];
                gradient += cost * previous[problem.incidentPartner[k]];
                diagonal += cost;
            }
            if (diagonal > 0.0)
                problem.keep[v] = std::clamp(previous[v] - gradient / std::max(diagonal, kTiny), 0.0, 1.0);
            else if (gradient < 0.0)
                problem.keep[v] = 1.0;
            else if (gradient > 0.0)
                problem.keep[v] = 0.0;
            else
                problem.keep[v] = previous[v];
        }
    });
}

// Jacobi-preconditioned conjugate gradients on the reweighted quadratic
//   (diag(unary) + L(pairWeight)) x = rhs,
// started at the current phase. Returns the iterations taken.
int conjugateGradient(const Problem& problem, const std::vector<double>& pairWeight,
                      const std::vector<double>& unaryWeight, const std::vector<double>& diagonal,
                      const std::vector<double>& rhs, std::vector<double>& x, const int maxIterations,
                      const double tolerance)
{
    const std::size_t n = problem.nodes;
    const int threads = problem.threads;
    std::vector<double> inverse(n, 0.0), residual(n), z(n), direction(n), applied(n);
    forChunks(n, threads, [&](const std::size_t lo, const std::size_t hi) {
        for (std::size_t v = lo; v < hi; ++v)
            inverse[v] = diagonal[v] > 0.0 ? 1.0 / diagonal[v] : 0.0;
    });
    const auto matvec = [&](const std::vector<double>& in, std::vector<double>& out) {
        forChunks(n, threads, [&](const std::size_t lo, const std::size_t hi) {
            for (std::size_t v = lo; v < hi; ++v) {
                // The quadratic's matrix is a weighted graph Laplacian plus
                // the unary diagonal: on either side of an event the row
                // contributes c (x_v - x_partner), so one read per incidence.
                double sum = unaryWeight[v] * in[v];
                const double self = in[v];
                for (std::int64_t k = problem.rowStart[v]; k < problem.rowStart[v + 1]; ++k)
                    sum += pairWeight[problem.incidentEvent[k]] * (self - in[problem.incidentPartner[k]]);
                out[v] = sum;
            }
        });
    };
    const auto dot = [&](const std::vector<double>& u, const std::vector<double>& v) {
        return chunkedSum(n, threads, [&](const std::size_t lo, const std::size_t hi) {
            double sum = 0.0;
            for (std::size_t k = lo; k < hi; ++k) sum += u[k] * v[k];
            return sum;
        });
    };
    matvec(x, applied);
    forChunks(n, threads, [&](const std::size_t lo, const std::size_t hi) {
        for (std::size_t v = lo; v < hi; ++v) {
            residual[v] = rhs[v] - applied[v];
            z[v] = residual[v] * inverse[v];
            direction[v] = z[v];
        }
    });
    double rz = dot(residual, z);
    const double first = rz;
    int step = 0;
    for (step = 0; step < maxIterations; ++step) {
        matvec(direction, applied);
        const double pap = dot(direction, applied);
        const double alpha = pap > kTiny ? rz / std::max(pap, kTiny) : 0.0;
        forChunks(n, threads, [&](const std::size_t lo, const std::size_t hi) {
            for (std::size_t v = lo; v < hi; ++v) {
                x[v] += alpha * direction[v];
                residual[v] -= alpha * applied[v];
                z[v] = residual[v] * inverse[v];
            }
        });
        const double nextRz = dot(residual, z);
        const double beta = rz > kTiny ? nextRz / std::max(rz, kTiny) : 0.0;
        forChunks(n, threads, [&](const std::size_t lo, const std::size_t hi) {
            for (std::size_t v = lo; v < hi; ++v) direction[v] = z[v] + beta * direction[v];
        });
        rz = nextRz;
        if (((step + 1) % 10 == 0 || step + 1 == maxIterations) && rz <= tolerance * tolerance * std::max(first, kTiny))
            break;
    }
    return step + 1;
}

// One residual per event: the wrapped difference of its two phases. The wrap
// shifts the residual by an even constant and leaves the derivatives at +-1,
// so the analytic Jacobian is exact away from the jumps.
class WrappedPairCost final : public ceres::SizedCostFunction<1, 1, 1>
{
public:
    explicit WrappedPairCost(const double target) : target_(target) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        residuals[0] = fiberSignResidual(parameters[1][0] - parameters[0][0], target_);
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) jacobians[0][0] = -1.0;
            if (jacobians[1] != nullptr) jacobians[1][0] = 1.0;
        }
        return true;
    }

private:
    double target_;
};

class WrappedUnaryCost final : public ceres::SizedCostFunction<1, 1>
{
public:
    explicit WrappedUnaryCost(const double target) : target_(target) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        residuals[0] = fiberSignResidual(parameters[0][0], target_);
        if (jacobians != nullptr && jacobians[0] != nullptr) jacobians[0][0] = 1.0;
        return true;
    }

private:
    double target_;
};

FiberSignMetrics metricsOf(const Problem& problem)
{
    const std::size_t n = problem.nodes, m = problem.events;
    std::vector<double> score(n), sign(n);
    std::vector<char> kept(n);
    for (std::size_t v = 0; v < n; ++v) {
        score[v] = std::abs(fiberSignResidual(problem.phase[v], 0.0));
        sign[v] = score[v] <= 0.5 ? 1.0 : -1.0;  // the outward gauge of the solve
        kept[v] = problem.keep[v] > 0.5 ? 1 : 0;
    }
    // Nearly parallel axes: the strict metric's 30 degree window.
    const double strictDot = std::cos(30.0 * 3.14159265358979323846 / 180.0);
    double weightAll = 0.0, weightWrong = 0.0, weightKept = 0.0, weightKeptWrong = 0.0;
    std::int64_t keptEvents = 0, strict = 0, strictWrong = 0, strictKept = 0, strictKeptWrong = 0;
    for (std::size_t e = 0; e < m; ++e) {
        const std::int64_t a = problem.ea[e], b = problem.eb[e];
        const bool wrong = problem.dot[e] * sign[a] * sign[b] < 0.0;
        const bool retained = kept[a] != 0 && kept[b] != 0;
        const double weight = problem.rawWeight[e];
        weightAll += weight;
        if (wrong) weightWrong += weight;
        if (retained) {
            ++keptEvents;
            weightKept += weight;
            if (wrong) weightKeptWrong += weight;
        }
        const double turns = std::abs(problem.distance[e]);
        if (std::abs(problem.dot[e]) > strictDot && std::isfinite(turns) && turns < 1.0) {
            ++strict;
            if (wrong) ++strictWrong;
            if (retained) {
                ++strictKept;
                if (wrong) ++strictKeptWrong;
            }
        }
    }
    const double quiet = std::numeric_limits<double>::quiet_NaN();
    const double denominator = std::max(weightAll, 1e-12);
    double brokenFibers = 0.0, expectedBroken = 0.0, uncertain = 0.0, agreement = 0.0;
    for (std::size_t v = 0; v < n; ++v) {
        brokenFibers += problem.keep[v] <= 0.5 ? 1.0 : 0.0;
        expectedBroken += 1.0 - problem.keep[v];
        uncertain += score[v] > 0.1 && score[v] < 0.9 ? 1.0 : 0.0;
        agreement += problem.agreement[v] * sign[v];
    }
    FiberSignMetrics metrics;
    metrics.brokenFibersPercent = n > 0 ? 100.0 * brokenFibers / double(n) : quiet;
    metrics.expectedBrokenPercent = n > 0 ? 100.0 * expectedBroken / double(n) : quiet;
    metrics.retainedEventsPercent = m > 0 ? 100.0 * double(keptEvents) / double(m) : quiet;
    metrics.weightedDisagreementPercent = 100.0 * weightWrong / denominator;
    metrics.weightedRetainedDisagreementPercent = weightKept > 0.0 ? 100.0 * weightKeptWrong / weightKept : quiet;
    metrics.retainedDisagreementOriginalWeightPercent = 100.0 * weightKeptWrong / denominator;
    metrics.outwardAgreement = n > 0 ? agreement / double(n) : quiet;
    metrics.uncertainSignsPercent = n > 0 ? 100.0 * uncertain / double(n) : quiet;
    metrics.strictEvents = strict;
    metrics.strictDisagreementPercent = 100.0 * double(strictWrong) / double(std::max<std::int64_t>(strict, 1));
    metrics.strictRetainedDisagreementPercent =
        100.0 * double(strictKeptWrong) / double(std::max<std::int64_t>(strictKept, 1));
    return metrics;
}

Problem build(const std::vector<double>& agreement, const std::vector<FiberSignEvent>& events,
              const FiberSignOptions& options, const std::vector<int>& fixedSigns, FiberSignResult& result)
{
    if (!(options.epsilon > 0.0 && options.epsilon < 1.0)) throw std::invalid_argument("epsilon must lie in (0, 1)");
    if (options.iterations < 1 || options.cgIterations < 1 || options.keepRounds < 1)
        throw std::invalid_argument("iteration counts must be positive");
    if (!(options.cgTolerance > 0.0) || options.tolerance < 0.0)
        throw std::invalid_argument("invalid solver tolerances");
    if (!(options.dotWeightPower >= 1.0)) throw std::invalid_argument("dotWeightPower must be at least 1");
    if (!(options.priorWeight >= 0.0) || !(options.fixedSignWeight >= 0.0) || !(options.brokenPenalty > 0.0))
        throw std::invalid_argument("invalid prior or rejection weight");
    if (agreement.empty()) throw std::invalid_argument("no fibers");
    if (!fixedSigns.empty() && fixedSigns.size() != agreement.size())
        throw std::invalid_argument("fixedSigns must hold one entry per fiber");

    Problem problem;
    const std::size_t n = agreement.size();
    problem.nodes = n;
    problem.events = events.size();
    problem.threads = std::max(1, options.threads);
    problem.epsilon = options.epsilon;
    problem.agreement = agreement;
    problem.phase.resize(n);
    problem.keep.assign(n, options.fixedKeep ? 1.0 : 0.99);
    problem.priorTarget.resize(n);
    problem.priorWeight.resize(n);
    problem.priorConstant.resize(n);
    problem.rejection.assign(n, options.brokenPenalty / double(n));

    for (std::size_t v = 0; v < n; ++v) {
        const double a = agreement[v];
        if (!std::isfinite(a) || std::abs(a) > 1.000001) throw std::invalid_argument("radial prior outside [-1, 1]");
        // The radial prior initializes and weakly anchors the phase; a fiber
        // without one (a == 0) starts just inside its class and is decided by
        // the crossings alone.
        problem.phase[v] = 1.0 - sigmoid(a == 0.0 ? 0.001 : 2.0 * a);
        problem.priorTarget[v] = a < 0.0 ? 1.0 : 0.0;
        problem.priorWeight[v] = options.priorWeight * std::abs(a) / double(n);
        problem.priorConstant[v] = options.priorWeight * (1.0 - std::abs(a)) / (2.0 * double(n));
        if (!fixedSigns.empty() && fixedSigns[v] != 0) {
            if (fixedSigns[v] != 1 && fixedSigns[v] != -1) throw std::invalid_argument("a fixed sign must be +1 or -1");
            // A fixed sign is given inward: inward +1 is the odd phase class.
            problem.priorTarget[v] = fixedSigns[v] > 0 ? 1.0 : 0.0;
            problem.priorWeight[v] = options.fixedSignWeight;
            problem.priorConstant[v] = 0.0;
            problem.phase[v] = problem.priorTarget[v];
            ++result.fixedNodes;
        }
    }

    const std::size_t m = events.size();
    problem.ea.resize(m);
    problem.eb.resize(m);
    problem.dot.resize(m);
    problem.distance.resize(m);
    problem.target.resize(m);
    problem.weight.resize(m);
    problem.rawWeight.resize(m);
    double total = 0.0;
    for (std::size_t e = 0; e < m; ++e) {
        const FiberSignEvent& event = events[e];
        if (event.a < 0 || event.b < 0 || std::size_t(event.a) >= n || std::size_t(event.b) >= n)
            throw std::invalid_argument("event fiber index out of range");
        if (!std::isfinite(event.dot) || std::abs(event.dot) > 1.000001)
            throw std::invalid_argument("event dot product outside [-1, 1]");
        const double dot = std::clamp(event.dot, -1.0, 1.0);
        problem.ea[e] = event.a;
        problem.eb[e] = event.b;
        problem.dot[e] = dot;
        problem.distance[e] = event.windingDistance;
        problem.target[e] = dot < 0.0 ? 1.0 : 0.0;
        problem.rawWeight[e] = fiberSignEventWeight(dot, event.windingDistance, options);
        total += problem.rawWeight[e];
        if (event.a == event.b) ++result.selfEvents;
        if (std::isfinite(event.windingDistance)) ++result.eventsWithWindingDistance;
    }
    const double scale = std::max(total, 1e-12);
    for (std::size_t e = 0; e < m; ++e) problem.weight[e] = problem.rawWeight[e] / scale;

    problem.rowStart.assign(n + 1, 0);
    for (std::size_t e = 0; e < m; ++e) {
        ++problem.rowStart[std::size_t(problem.ea[e]) + 1];
        ++problem.rowStart[std::size_t(problem.eb[e]) + 1];
    }
    for (std::size_t v = 0; v < n; ++v) problem.rowStart[v + 1] += problem.rowStart[v];
    problem.incidentEvent.resize(problem.rowStart[n]);
    problem.incidentPartner.resize(problem.rowStart[n]);
    problem.incidentSign.resize(problem.rowStart[n]);
    std::vector<std::int64_t> cursor(problem.rowStart.begin(), problem.rowStart.end() - 1);
    const auto place = [&](const std::int64_t node, const std::size_t event, const std::int64_t partner,
                           const double sign) {
        const std::int64_t slot = cursor[std::size_t(node)]++;
        problem.incidentEvent[std::size_t(slot)] = std::int64_t(event);
        problem.incidentPartner[std::size_t(slot)] = partner;
        problem.incidentSign[std::size_t(slot)] = sign;
    };
    for (std::size_t e = 0; e < m; ++e) {
        place(problem.ea[e], e, problem.eb[e], -1.0);
        place(problem.eb[e], e, problem.ea[e], 1.0);
    }
    for (std::size_t v = 0; v < n; ++v)
        if (problem.rowStart[v + 1] == problem.rowStart[v]) ++result.isolatedNodes;

    result.nodes = std::int64_t(n);
    result.events = std::int64_t(m);
    result.nodeChunks = std::int64_t((n + kChunk - 1) / kChunk);
    result.eventChunks = std::int64_t(m == 0 ? 0 : (m + kChunk - 1) / kChunk);
    return problem;
}

void solveIrls(Problem& problem, const FiberSignOptions& options, FiberSignResult& result,
               const std::chrono::steady_clock::time_point started, const FiberSignStepCallback& onStep)
{
    const std::size_t n = problem.nodes, m = problem.events;
    const double epsilon = options.epsilon;
    const auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };
    const auto record = [&](const int step, const int linear, const int backtracks) {
        const Terms terms = objectiveOf(problem, epsilon);
        if (options.history) {
            FiberSignStep row;
            row.step = step;
            row.objective = terms.total;
            row.pairLoss = terms.pair;
            row.priorLoss = terms.prior;
            row.brokenLoss = terms.broken;
            row.exactObjective = objectiveOf(problem, 0.0).total;
            row.linearIterations = linear;
            row.backtracks = backtracks;
            row.seconds = seconds();
            result.history.push_back(row);
        }
        return terms.total;
    };

    double previous = record(0, 0, 0);
    std::vector<double> pairWeight(m), pairTarget(m), unaryWeight(n), unaryTarget(n), diagonal(n), rhs(n),
        proposed(n), oldPhase(n), oldKeep(n), eventCost;
    int stalled = 0, lastStep = 0;
    for (int step = 1; step <= options.iterations; ++step) {
        // The caller's step hook: progress, and the chance to stop.
        if (onStep && !onStep(step, options.iterations)) break;
        lastStep = step;
        forChunks(m, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
            for (std::size_t e = lo; e < hi; ++e) {
                const std::int64_t a = problem.ea[e], b = problem.eb[e];
                const double delta = problem.phase[b] - problem.phase[a];
                const double residual = fiberSignResidual(delta, problem.target[e]);
                // A self event has no phase derivative, but still weighs on
                // its fiber's keep.
                pairWeight[e] = a == b ? 0.0
                                       : problem.weight[e] * problem.keep[a] * problem.keep[b]
                                             / std::max(std::abs(residual), epsilon);
                pairTarget[e] = delta - residual;
            }
        });
        forChunks(n, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
            for (std::size_t v = lo; v < hi; ++v) {
                const double residual = fiberSignResidual(problem.phase[v], problem.priorTarget[v]);
                unaryWeight[v] = problem.priorWeight[v] / std::max(std::abs(residual), epsilon);
                unaryTarget[v] = problem.phase[v] - residual;
                double sum = unaryWeight[v];
                double right = unaryWeight[v] * unaryTarget[v];
                for (std::int64_t k = problem.rowStart[v]; k < problem.rowStart[v + 1]; ++k) {
                    const std::int64_t e = problem.incidentEvent[k];
                    sum += pairWeight[e];
                    right += problem.incidentSign[k] * pairWeight[e] * pairTarget[e];
                }
                diagonal[v] = sum;
                rhs[v] = right;
            }
        });
        proposed = problem.phase;
        const int linear = conjugateGradient(problem, pairWeight, unaryWeight, diagonal, rhs, proposed,
                                             options.cgIterations, options.cgTolerance);
        for (std::size_t v = 0; v < n; ++v)
            if (!std::isfinite(proposed[v])) throw std::runtime_error("nonfinite phase update");
        result.linearIterations += linear;

        oldPhase = problem.phase;
        oldKeep = problem.keep;
        bool accepted = false;
        int backtracks = 0;
        for (backtracks = 0; backtracks <= 20; ++backtracks) {
            const double fraction = std::ldexp(1.0, -backtracks);
            forChunks(n, problem.threads, [&](const std::size_t lo, const std::size_t hi) {
                for (std::size_t v = lo; v < hi; ++v)
                    problem.phase[v] = oldPhase[v] + (proposed[v] - oldPhase[v]) * fraction;
            });
            if (options.fixedKeep)
                problem.keep = oldKeep;
            else
                keepStep(problem, oldKeep, eventCost);
            if (objectiveOf(problem, epsilon).total <= previous) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            problem.phase = oldPhase;
            problem.keep = oldKeep;
        }
        const double current = record(step, linear, accepted ? backtracks : 21);
        stalled = previous - current <= options.tolerance * std::max(1.0, std::abs(previous)) ? stalled + 1 : 0;
        previous = current;
        if (stalled >= 5) break;
    }
    result.iterations = lastStep;
    result.converged = stalled >= 5;
    result.linearSolver = "jacobi_preconditioned_conjugate_gradients";
    result.threadsUsed = problem.threads;
}

void solveCeres(Problem& problem, const FiberSignOptions& options, FiberSignResult& result,
                const std::chrono::steady_clock::time_point started)
{
    const std::size_t n = problem.nodes, m = problem.events;
    const double epsilon = options.epsilon;
    const auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };
    const auto record = [&](const int step, const int linear, const int backtracks) {
        const Terms terms = objectiveOf(problem, epsilon);
        if (options.history) {
            FiberSignStep row;
            row.step = step;
            row.objective = terms.total;
            row.pairLoss = terms.pair;
            row.priorLoss = terms.prior;
            row.brokenLoss = terms.broken;
            row.exactObjective = objectiveOf(problem, 0.0).total;
            row.linearIterations = linear;
            row.backtracks = backtracks;
            row.seconds = seconds();
            result.history.push_back(row);
        }
        return terms.total;
    };

    // Huber of the wrapped residual, scaled per residual block: Ceres' cost
    // 0.5 * a * rho(r^2) with rho = Huber(epsilon) and a = w / epsilon is
    // exactly w * (r^2 / (2 eps) below eps, |r| - eps/2 above). One shared
    // loss carries every block's scale.
    ceres::HuberLoss huber(epsilon);
    double previous = record(0, 0, 0);
    std::vector<double> oldPhase, oldKeep, eventCost;
    int stalled = 0, lastRound = 0;
    const int rounds = options.fixedKeep ? 1 : options.keepRounds;
    for (int round = 1; round <= rounds; ++round) {
        lastRound = round;
        oldPhase = problem.phase;
        oldKeep = problem.keep;
        ceres::Problem::Options problemOptions;
        problemOptions.loss_function_ownership = ceres::TAKE_OWNERSHIP;
        problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
        ceres::Problem ceresProblem(problemOptions);
        for (std::size_t e = 0; e < m; ++e) {
            const std::int64_t a = problem.ea[e], b = problem.eb[e];
            if (a == b) continue;
            const double weight = problem.weight[e] * problem.keep[a] * problem.keep[b];
            if (!(weight > 0.0)) continue;
            ceresProblem.AddResidualBlock(
                new WrappedPairCost(problem.target[e]),
                new ceres::ScaledLoss(&huber, weight / epsilon, ceres::DO_NOT_TAKE_OWNERSHIP),
                &problem.phase[std::size_t(a)], &problem.phase[std::size_t(b)]);
        }
        for (std::size_t v = 0; v < n; ++v) {
            if (!(problem.priorWeight[v] > 0.0)) continue;
            ceresProblem.AddResidualBlock(
                new WrappedUnaryCost(problem.priorTarget[v]),
                new ceres::ScaledLoss(&huber, problem.priorWeight[v] / epsilon, ceres::DO_NOT_TAKE_OWNERSHIP),
                &problem.phase[v]);
        }
        ceres::Solver::Options solverOptions;
        switch (options.linearSolver) {
            case FiberSignLinearSolver::SparseSchur:
                solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
                break;
            case FiberSignLinearSolver::IterativeSchur:
                solverOptions.linear_solver_type = ceres::ITERATIVE_SCHUR;
                solverOptions.preconditioner_type = ceres::SCHUR_JACOBI;
                break;
            case FiberSignLinearSolver::Cgnr:
                solverOptions.linear_solver_type = ceres::CGNR;
                solverOptions.preconditioner_type = ceres::JACOBI;
                break;
            case FiberSignLinearSolver::DenseNormalCholesky:
                solverOptions.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
                break;
            case FiberSignLinearSolver::DenseQr:
                solverOptions.linear_solver_type = ceres::DENSE_QR;
                break;
            case FiberSignLinearSolver::DenseSchur:
                solverOptions.linear_solver_type = ceres::DENSE_SCHUR;
                break;
            case FiberSignLinearSolver::SparseNormalCholesky:
            default:
                solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                break;
        }
        // A Schur form needs an independent set of fibers to eliminate; with no
        // ordering given Ceres computes one from the graph itself.
        solverOptions.sparse_linear_algebra_library_type =
            ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ? ceres::SUITE_SPARSE
                                                                                  : ceres::EIGEN_SPARSE;
        solverOptions.num_threads = std::max(1, options.threads);
        solverOptions.max_num_iterations = options.iterations;
        solverOptions.function_tolerance = options.tolerance;
        solverOptions.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(solverOptions, &ceresProblem, &summary);
        if (summary.termination_type == ceres::FAILURE)
            throw std::runtime_error("Ceres sign solve failed: " + summary.message);
        result.linearSolver = ceres::LinearSolverTypeToString(summary.linear_solver_type_used);
        result.sparseLibrary =
            ceres::SparseLinearAlgebraLibraryTypeToString(summary.sparse_linear_algebra_library_type);
        result.threadsUsed = summary.num_threads_used;
        const int linear = int(summary.num_successful_steps + summary.num_unsuccessful_steps);
        result.linearIterations += linear;
        if (!options.fixedKeep) keepStep(problem, oldKeep, eventCost);
        double current = objectiveOf(problem, epsilon).total;
        int backtracks = 0;
        if (current > previous) {
            // The phase solve cannot raise its own terms, but the bounded keep
            // update is only a majorizer: reject a round that raised the
            // periodic objective rather than reporting a worse state.
            problem.phase = oldPhase;
            problem.keep = oldKeep;
            backtracks = 1;
            current = previous;
        }
        record(round, linear, backtracks);
        stalled = previous - current <= options.tolerance * std::max(1.0, std::abs(previous)) ? stalled + 1 : 0;
        previous = current;
        if (backtracks != 0 || stalled >= 5) {
            result.converged = true;
            break;
        }
    }
    result.iterations = lastRound;
}

}  // namespace

FiberSignResult solveFiberNormalSigns(const std::vector<double>& agreement,
                                      const std::vector<FiberSignEvent>& events, const FiberSignOptions& options,
                                      const std::vector<int>& fixedSigns,
    const FiberSignStepCallback& onStep)
{
    FiberSignResult result;
    Problem problem = build(agreement, events, options, fixedSigns, result);
    const auto started = std::chrono::steady_clock::now();
    if (options.backend == FiberSignBackend::Ceres)
        solveCeres(problem, options, result, started);
    else
        solveIrls(problem, options, result, started, onStep);
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    const Terms terms = objectiveOf(problem, options.epsilon);
    result.objective = terms.total;
    result.pairLoss = terms.pair;
    result.priorLoss = terms.prior;
    result.brokenLoss = terms.broken;
    result.exactObjective = objectiveOf(problem, 0.0).total;
    result.metrics = metricsOf(problem);

    const std::size_t n = problem.nodes;
    result.phase = problem.phase;
    result.keep = problem.keep;
    result.sign.resize(n);
    result.signProbability.resize(n);
    result.broken.resize(n);
    for (std::size_t v = 0; v < n; ++v) {
        const double score = std::abs(fiberSignResidual(problem.phase[v], 0.0));
        // Stored signs are inward: the negated outward gauge of the solve.
        result.sign[v] = score <= 0.5 ? std::int8_t(-1) : std::int8_t(1);
        result.signProbability[v] = score;
        result.broken[v] = problem.keep[v] <= 0.5 ? 1 : 0;
    }
    return result;
}

}  // namespace vc::fiber_tracer
