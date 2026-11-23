#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <functional>
#include <vector>

// Whole-fiber normal signs from perpendicular crossings, on the CPU.
//
// Each fiber carries one sign-free normal axis field; a crossing of two fibers
// witnesses whether their axes agree there (dot > 0) or oppose (dot < 0).
// Choosing one sign per fiber so that as much of that evidence as possible is
// satisfied is the sign problem: a period-two L1 labelling, solved here as the
// phase relaxation of the reference wrapped-IRLS stage. Phase x_v has period
// two (even phases are one class, odd the other) and an edge's residual is the
// wrapped difference
//     r_e = wrap(x_b - x_a - t_e),   wrap(d) = fmod(d + 1, 2) - 1 in [-1, 1),
// with t_e = 1 when the crossing opposes the two axes and 0 when it agrees.
// With the Huber-L1 cost h(r) = r^2/(2 eps) below eps and |r| - eps/2 above,
// the objective over the phases x and the per-fiber keep variables k in [0, 1]
// is
//     sum_e w_e k_a k_b h(r_e)                 crossing evidence
//   + sum_v (p_v h(wrap(x_v - q_v)) + c_v)     per-fiber prior (radial, or a fixed sign)
//   + sum_v z_v (1 - k_v)                      the cost of rejecting a fiber.
// A crossing's weight w_e is |dot| to a power times a winding-distance factor,
// so well-aligned crossings less than a winding apart dominate. The objective
// is not convex: both backends descend it from the radial initialization and
// report the value they reach, never a certificate of the global optimum.
//
// Two backends solve that same objective, neither needing a GPU:
//  - Irls reproduces the reference stage: each step reweights the wrapped
//    residuals into a quadratic solved by Jacobi-preconditioned conjugate
//    gradients, then takes one bounded keep update, backtracking whenever the
//    periodic objective fails to improve.
//  - Ceres states the wrapped residuals directly as a Ceres problem with a
//    per-edge Huber loss (identical cost, see FiberNormalSign.cpp) and a
//    sparse linear solver, alternating with the same keep update.
//
// Signs are reported in the inward convention (the normal points toward the
// umbilicus), matching the stored normal_sign column.
namespace vc::fiber_tracer
{

// One perpendicular crossing: the two fibers' node indices, the dot product of
// their normal axes there (sign-free evidence in [-1, 1]) and the measured
// winding distance in turns (NaN when the extraction measured none).
struct FiberSignEvent {
    std::int64_t a = 0;
    std::int64_t b = 0;
    double dot = 0.0;
    double windingDistance = std::numeric_limits<double>::quiet_NaN();
};

enum class FiberSignBackend { Irls, Ceres };

// Per-event factor from the measured winding distance. InverseSquare weights a
// crossing d turns apart by min(1, 1/d^2), so a far crossing still counts but
// only decides a sign when nothing closer does; an unmeasured distance counts
// as within one winding. None drops the factor.
enum class FiberSignDistanceWeight { InverseSquare, None };

// The Ceres backend's linear solver. The quadratic is a graph Laplacian with
// one scalar per fiber: the sparse forms factorize it directly, the Schur forms
// eliminate an automatically chosen independent set of fibers first, the dense
// forms ignore the sparsity (worth it only for a few thousand fibers, such as a
// manual scaffold) and Cgnr is conjugate gradients on the normal equations, the
// same family as the IRLS backend's inner solve. Ignored by the IRLS backend.
enum class FiberSignLinearSolver {
    SparseNormalCholesky,
    SparseSchur,
    IterativeSchur,
    Cgnr,
    DenseNormalCholesky,
    DenseQr,
    DenseSchur
};

struct FiberSignOptions {
    FiberSignBackend backend = FiberSignBackend::Irls;
    FiberSignLinearSolver linearSolver = FiberSignLinearSolver::SparseNormalCholesky;
    FiberSignDistanceWeight distanceWeight = FiberSignDistanceWeight::InverseSquare;
    // Crossing weight |dot|^dotWeightPower; at least 1.
    double dotWeightPower = 2.0;
    // Scale of the per-fiber radial prior, and the weight a fixed sign gets
    // instead (above the total crossing weight, so the sign holds).
    double priorWeight = 1e-3;
    double fixedSignWeight = 10.0;
    // Cost of rejecting one fiber, per fiber.
    double brokenPenalty = 0.1;
    // Huber transition of the L1 cost, in phase units; 0 < epsilon < 1.
    double epsilon = 0.05;
    // Irls: reweighting steps. Ceres: trust-region iterations per solve.
    int iterations = 100;
    // Irls: conjugate-gradient iterations per step, and its relative tolerance.
    int cgIterations = 100;
    double cgTolerance = 1e-5;
    // Ceres: rounds of (phase solve, keep update). One round when keep is fixed.
    int keepRounds = 5;
    // Relative objective improvement below which a step has stalled; five
    // stalled steps end the solve.
    double tolerance = 1e-7;
    // Hold every keep at one: sign-only alignment, no rejection.
    bool fixedKeep = false;
    int threads = 1;
    // Record one history row per step.
    bool history = false;
};

// Agreement of the solved signs with the evidence. Shares the reference
// stage's definitions, so the numbers compare directly. A share whose
// population is empty is NaN.
struct FiberSignMetrics {
    double brokenFibersPercent = 0.0;
    double expectedBrokenPercent = 0.0;
    double retainedEventsPercent = 0.0;
    // Weight-weighted share of disagreeing crossings: over all of them, over
    // the retained ones, and the retained ones against the total weight.
    double weightedDisagreementPercent = 0.0;
    double weightedRetainedDisagreementPercent = 0.0;
    double retainedDisagreementOriginalWeightPercent = 0.0;
    // Mean of the radial prior times the solved outward sign.
    double outwardAgreement = 0.0;
    double uncertainSignsPercent = 0.0;
    // The strict internal metric: unweighted share of disagreeing crossings
    // among those whose axes are within 30 degrees and whose winding distance
    // is below one turn.
    std::int64_t strictEvents = 0;
    double strictDisagreementPercent = 0.0;
    double strictRetainedDisagreementPercent = 0.0;
};

struct FiberSignStep {
    int step = 0;
    double objective = 0.0;
    double pairLoss = 0.0;
    double priorLoss = 0.0;
    double brokenLoss = 0.0;
    // The same objective with epsilon 0: the wrapped L1 value, not a bound.
    double exactObjective = 0.0;
    int linearIterations = 0;
    int backtracks = 0;
    double seconds = 0.0;
};

struct FiberSignResult {
    // Per fiber. phase is unwrapped; sign and signProbability are its readout
    // in the inward convention, broken is keep <= 0.5.
    std::vector<double> phase;
    std::vector<double> keep;
    std::vector<std::int8_t> sign;
    std::vector<double> signProbability;
    std::vector<char> broken;

    double objective = 0.0;
    double pairLoss = 0.0;
    double priorLoss = 0.0;
    double brokenLoss = 0.0;
    double exactObjective = 0.0;
    FiberSignMetrics metrics;

    int iterations = 0;
    // CG iterations (Irls) or trust-region iterations (Ceres), summed.
    std::int64_t linearIterations = 0;
    bool converged = false;
    double seconds = 0.0;

    std::int64_t nodes = 0;
    std::int64_t events = 0;
    std::int64_t selfEvents = 0;
    std::int64_t isolatedNodes = 0;
    std::int64_t fixedNodes = 0;
    std::int64_t eventsWithWindingDistance = 0;
    // Work units the node and event loops split into: one means the solve ran
    // serially whatever `threads` asked for.
    std::int64_t nodeChunks = 0;
    std::int64_t eventChunks = 0;
    // What the backend actually ran with. The Ceres backend reports its linear
    // solver, its sparse library and the threads it used; the IRLS backend
    // reports its conjugate gradients and the threads it was given.
    std::string linearSolver;
    std::string sparseLibrary;
    int threadsUsed = 0;
    std::vector<FiberSignStep> history;
};

// The wrapped difference of a phase delta and a target, in [-1, 1).
[[nodiscard]] double fiberSignResidual(double delta, double target = 0.0);
// Huber-L1 cost of one residual; epsilon 0 gives |r|.
[[nodiscard]] double fiberSignCost(double residual, double epsilon);
// The winding-distance factor of one event.
[[nodiscard]] double fiberSignWindingDistanceFactor(double distance, FiberSignDistanceWeight mode);
// The weight of one crossing: |dot|^dotWeightPower times the winding-distance factor.
[[nodiscard]] double fiberSignEventWeight(double dot, double windingDistance, const FiberSignOptions& options);

// `agreement` holds one radial prior per fiber in [-1, 1] (0 where none is
// known). `fixedSigns`, when not empty, holds one entry per fiber: 0 to solve
// it, or +1/-1 for an inward sign held by a strong prior. Event indices
// address `agreement`. Throws std::invalid_argument on invalid input.
// `onStep`, when given, is called after each reweighting step with the step
// number and the planned number of steps; returning false stops the solve and
// keeps the signs of the last step (a cancelled run discards them anyway).
using FiberSignStepCallback = std::function<bool(int step, int steps)>;
[[nodiscard]] FiberSignResult solveFiberNormalSigns(
    const std::vector<double>& agreement,
    const std::vector<FiberSignEvent>& events,
    const FiberSignOptions& options = {},
    const std::vector<int>& fixedSigns = {},
    const FiberSignStepCallback& onStep = {});

}  // namespace vc::fiber_tracer
