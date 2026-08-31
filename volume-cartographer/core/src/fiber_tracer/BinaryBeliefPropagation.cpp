#include "vc/fiber_tracer/BinaryBeliefPropagation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#include <omp.h>

namespace vc::fiber_tracer
{
namespace
{

constexpr std::size_t kParallelFactorThreshold = 32'768;

double logAddExp(double a, double b)
{
    const double maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

double updateMessage(double cavityLogOdds, double logSamePotential, double logDifferentPotential)
{
    const double targetOne = logAddExp(logDifferentPotential, cavityLogOdds + logSamePotential);
    const double targetZero = logAddExp(logSamePotential, cavityLogOdds + logDifferentPotential);
    return targetOne - targetZero;
}

double probabilityOne(double logOdds)
{
    if (logOdds >= 0.0) {
        const double exponential = std::exp(-logOdds);
        return 1.0 / (1.0 + exponential);
    }
    const double exponential = std::exp(logOdds);
    return exponential / (1.0 + exponential);
}

void validate(std::size_t nodeCount, std::span<const BinaryPairwiseFactor> factors, std::span<const BinaryBeliefState> fixedStates, const BinaryBeliefPropagationConfig& config)
{
    if (nodeCount == 0)
        throw std::invalid_argument("Binary BP requires at least one node");
    if (fixedStates.size() != nodeCount) {
        throw std::invalid_argument("Binary BP fixed-state count does not match node count");
    }
    if (!std::isfinite(config.temperature) || !(config.temperature > 0.0)) {
        throw std::invalid_argument("Binary BP temperature must be finite and positive");
    }
    if (!std::isfinite(config.messageDamping) || !(config.messageDamping > 0.0) || config.messageDamping > 1.0) {
        throw std::invalid_argument("Binary BP message damping must be in (0, 1]");
    }
    if (!std::isfinite(config.messageResidualTolerance) || config.messageResidualTolerance < 0.0) {
        throw std::invalid_argument("Binary BP residual tolerance must be finite and nonnegative");
    }
    if (config.maximumMessageIterations == 0) {
        throw std::invalid_argument("Binary BP message iteration limit must be positive");
    }
    if (config.parallelWorkers == 0) {
        throw std::invalid_argument("Binary BP parallel worker count must be positive");
    }
    for (const auto state : fixedStates) {
        if (state != BinaryBeliefState::Free && state != BinaryBeliefState::Zero && state != BinaryBeliefState::One) {
            throw std::invalid_argument("Binary BP fixed state is invalid");
        }
    }
    for (const auto& factor : factors) {
        if (factor.a >= nodeCount || factor.b >= nodeCount || factor.a == factor.b) {
            throw std::invalid_argument("Binary BP factor references an invalid node pair");
        }
        if (!std::isfinite(factor.sameCost) || !std::isfinite(factor.differentCost) || factor.sameCost < 0.0 || factor.differentCost < 0.0) {
            throw std::invalid_argument("Binary BP factor costs must be finite and nonnegative");
        }
    }
}

}  // namespace

BinaryBeliefPropagationReport solveBinaryPairwiseSumProduct(
    std::size_t nodeCount,
    std::span<const BinaryPairwiseFactor> factors,
    std::span<const BinaryBeliefState> fixedStates,
    const BinaryBeliefPropagationConfig& config,
    const BinaryBeliefPropagationProgressCallback& progress)
{
    const auto totalStart = std::chrono::steady_clock::now();
    validate(nodeCount, factors, fixedStates, config);

    std::vector<double> logSame(factors.size());
    std::vector<double> logDifferent(factors.size());
    for (std::size_t index = 0; index < factors.size(); ++index) {
        const double minimumCost = std::min(factors[index].sameCost, factors[index].differentCost);
        logSame[index] = -(factors[index].sameCost - minimumCost) / config.temperature;
        logDifferent[index] = -(factors[index].differentCost - minimumCost) / config.temperature;
        if (!std::isfinite(logSame[index]) || !std::isfinite(logDifferent[index])) {
            throw std::invalid_argument("Binary BP temperature is too small for factor costs");
        }
    }

    std::vector<double> totalLogOdds(nodeCount, 0.0);

    // CSR incoming-message slots preserve each node's original factor order.
    // Node totals are contiguous and bit-for-bit equivalent to the serial edge
    // pass, while factor slots map updates back into the next CSR buffer.
    std::vector<std::size_t> incidentOffsets(nodeCount + 1, 0);
    for (const auto& factor : factors) {
        ++incidentOffsets[factor.a + 1];
        ++incidentOffsets[factor.b + 1];
    }
    for (std::size_t node = 0; node < nodeCount; ++node)
        incidentOffsets[node + 1] += incidentOffsets[node];
    std::vector<std::size_t> slotToA(factors.size());
    std::vector<std::size_t> slotToB(factors.size());
    std::vector<std::size_t> nextIncident = incidentOffsets;
    for (std::size_t index = 0; index < factors.size(); ++index) {
        slotToA[index] = nextIncident[factors[index].a]++;
        slotToB[index] = nextIncident[factors[index].b]++;
    }
    std::vector<double> incomingMessages(incidentOffsets.back(), 0.0);
    std::vector<double> nextIncomingMessages(incidentOffsets.back(), 0.0);

    const std::size_t usefulWorkers = std::max<std::size_t>(1, std::min(nodeCount, std::max<std::size_t>(1, factors.size())));
    const std::size_t runtimeWorkers = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
    const int workers = static_cast<int>(std::min({
        config.parallelWorkers,
        usefulWorkers,
        runtimeWorkers,
        static_cast<std::size_t>(std::numeric_limits<int>::max()),
    }));
    const bool useParallel = workers > 1 && factors.size() >= kParallelFactorThreshold;

    BinaryBeliefPropagationReport report;
    report.logOdds.resize(nodeCount);
    report.probabilityOne.resize(nodeCount);
    const auto setupEnd = std::chrono::steady_clock::now();
    double residual = 0.0;
    bool stop = false;
    std::exception_ptr progressFailure;
    auto phaseStart = setupEnd;

#pragma omp parallel num_threads(workers) if (useParallel) shared(residual, stop, report, incomingMessages, nextIncomingMessages, progressFailure)
    {
#pragma omp single
        {
            report.effectiveWorkers = static_cast<std::size_t>(omp_get_num_threads());
            phaseStart = std::chrono::steady_clock::now();
        }

        for (std::size_t iteration = 0; iteration < config.maximumMessageIterations; ++iteration) {
#pragma omp for schedule(static)
            for (std::size_t node = 0; node < nodeCount; ++node) {
                double total = 0.0;
                for (std::size_t incident = incidentOffsets[node]; incident < incidentOffsets[node + 1]; ++incident)
                    total += incomingMessages[incident];
                totalLogOdds[node] = total;
            }

#pragma omp single
            {
                const auto now = std::chrono::steady_clock::now();
                report.nodeTotalMilliseconds += std::chrono::duration<double, std::milli>(now - phaseStart).count();
                phaseStart = now;
                residual = 0.0;
            }

#pragma omp for schedule(static) reduction(max : residual)
            for (std::size_t index = 0; index < factors.size(); ++index) {
                const auto& factor = factors[index];
                const double oldAToB = incomingMessages[slotToB[index]];
                const double oldBToA = incomingMessages[slotToA[index]];
                const auto outgoing = [&](std::size_t node, double cavityLogOdds) {
                    if (fixedStates[node] == BinaryBeliefState::One)
                        return logSame[index] - logDifferent[index];
                    if (fixedStates[node] == BinaryBeliefState::Zero)
                        return logDifferent[index] - logSame[index];
                    return updateMessage(cavityLogOdds, logSame[index], logDifferent[index]);
                };
                const double rawAToB = outgoing(factor.a, totalLogOdds[factor.a] - oldBToA);
                const double rawBToA = outgoing(factor.b, totalLogOdds[factor.b] - oldAToB);
                const double nextAToBValue = oldAToB + config.messageDamping * (rawAToB - oldAToB);
                const double nextBToAValue = oldBToA + config.messageDamping * (rawBToA - oldBToA);
                nextIncomingMessages[slotToB[index]] = nextAToBValue;
                nextIncomingMessages[slotToA[index]] = nextBToAValue;
                residual = std::max({
                    residual,
                    std::abs(nextAToBValue - oldAToB),
                    std::abs(nextBToAValue - oldBToA),
                });
            }

#pragma omp single
            {
                incomingMessages.swap(nextIncomingMessages);
                report.messageIterations = iteration + 1;
                report.messageResidual = residual;
                stop = residual <= config.messageResidualTolerance;
                report.messageConverged = stop;
                const auto now = std::chrono::steady_clock::now();
                report.messageUpdateMilliseconds += std::chrono::duration<double, std::milli>(now - phaseStart).count();
                if (progress) {
                    try {
                        progress({
                            report.messageIterations,
                            config.maximumMessageIterations,
                            report.messageResidual,
                            false,
                        });
                    } catch (...) {
                        progressFailure = std::current_exception();
                        stop = true;
                    }
                }
                phaseStart = std::chrono::steady_clock::now();
            }
            if (stop)
                break;
        }

#pragma omp for schedule(static)
        for (std::size_t node = 0; node < nodeCount; ++node) {
            double total = 0.0;
            for (std::size_t incident = incidentOffsets[node]; incident < incidentOffsets[node + 1]; ++incident)
                total += incomingMessages[incident];
            totalLogOdds[node] = total;
        }

#pragma omp for schedule(static)
        for (std::size_t node = 0; node < nodeCount; ++node) {
            if (fixedStates[node] == BinaryBeliefState::One) {
                report.logOdds[node] = std::numeric_limits<double>::infinity();
                report.probabilityOne[node] = 1.0;
            } else if (fixedStates[node] == BinaryBeliefState::Zero) {
                report.logOdds[node] = -std::numeric_limits<double>::infinity();
                report.probabilityOne[node] = 0.0;
            } else {
                report.logOdds[node] = totalLogOdds[node];
                report.probabilityOne[node] = probabilityOne(totalLogOdds[node]);
            }
        }
    }
    const auto solveEnd = std::chrono::steady_clock::now();
    report.setupMilliseconds = std::chrono::duration<double, std::milli>(setupEnd - totalStart).count();
    report.solveMilliseconds = std::chrono::duration<double, std::milli>(solveEnd - setupEnd).count();
    report.elapsedMilliseconds = std::chrono::duration<double, std::milli>(solveEnd - totalStart).count();
    if (progressFailure)
        std::rethrow_exception(progressFailure);
    if (progress) {
        progress({
            report.messageIterations,
            config.maximumMessageIterations,
            report.messageResidual,
            true,
        });
    }
    return report;
}

}  // namespace vc::fiber_tracer
