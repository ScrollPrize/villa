#include "vc/lasagna/MaxflowGraph.hpp"
#include "vc/lasagna/EclMaxflow.hpp"
#include "vc/lasagna/LaplaceAmgx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr double kDefaultLaplaceSolverTolerance = 1.0e-8;
constexpr double kLaplaceRatioConfidenceFactor = 100.0;

void printUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " <manifest.lasagna.json> "
              << "--src x,y,z [--src x,y,z ...] --sink x,y,z [--sink x,y,z ...] "
              << "[--margin-base-voxels 1000] [--threshold 110] "
              << "[--run-ecl] [--runs N] [--terminal-flood-depth 10] "
              << "[--terminal-flood-capacity 1024] [--terminal-flood-decay 2] "
              << "[--terminal-region-iterations 0] [--terminal-flood-sweep] "
              << "[--terminal-fringe-sweep] [--run-laplace-amgx] "
              << "[--laplace-lambda-sweep 0.015625,0.0078125,...] "
              << "[--laplace-adaptive-lambda] [--laplace-adaptive-start-lambda 0.00048828125] "
              << "[--laplace-length-sweep 8,16,32,...] "
              << "[--laplace-source-depth 0] [--amgx-config file]\n";
}

vc::lasagna::MaxflowDouble3 parsePoint(const std::string& value, const char* name)
{
    vc::lasagna::MaxflowDouble3 point;
    char trailing = '\0';
    if (std::sscanf(value.c_str(), "%lf,%lf,%lf%c", &point.x, &point.y, &point.z, &trailing) != 3) {
        throw std::invalid_argument(std::string(name) + " must be formatted as x,y,z");
    }
    return point;
}

int64_t parseNonNegativeInt64(const std::string& value, const char* name)
{
    size_t consumed = 0;
    int64_t parsed = 0;
    try {
        parsed = std::stoll(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    if (consumed != value.size() || parsed < 0) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    return parsed;
}

uint8_t parseThreshold(const std::string& value)
{
    const int64_t parsed = parseNonNegativeInt64(value, "threshold");
    if (parsed > 255) {
        throw std::invalid_argument("threshold must be in [0,255]");
    }
    return static_cast<uint8_t>(parsed);
}

int parsePositiveInt(const std::string& value, const char* name)
{
    const int64_t parsed = parseNonNegativeInt64(value, name);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive int32 value");
    }
    return static_cast<int>(parsed);
}

double parsePositiveDouble(const std::string& value, const char* name)
{
    size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive number");
    }
    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be a positive number");
    }
    return parsed;
}

std::vector<double> parsePositiveDoubleList(const std::string& value, const char* name)
{
    std::vector<double> parsed;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string item = value.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::invalid_argument(std::string(name) + " must not contain empty entries");
        }
        size_t consumed = 0;
        double number = 0.0;
        try {
            number = std::stod(item, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument(std::string(name) + " must contain positive numbers");
        }
        if (consumed != item.size() || !std::isfinite(number) || number <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must contain positive numbers");
        }
        parsed.push_back(number);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (parsed.empty()) {
        throw std::invalid_argument(std::string(name) + " must contain at least one value");
    }
    return parsed;
}

std::string boxToString(const vc::lasagna::MaxflowBox3& box)
{
    return "[" + std::to_string(box.begin.x) + "," + std::to_string(box.begin.y) + "," +
           std::to_string(box.begin.z) + "]..[" + std::to_string(box.end.x) + "," +
           std::to_string(box.end.y) + "," + std::to_string(box.end.z) + "]";
}

vc::lasagna::MaxflowInt3 boxSize(const vc::lasagna::MaxflowBox3& box)
{
    return {
        box.end.x - box.begin.x,
        box.end.y - box.begin.y,
        box.end.z - box.begin.z,
    };
}

uint64_t boxVoxelCount(const vc::lasagna::MaxflowBox3& box)
{
    const auto size = boxSize(box);
    if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(size.x) *
           static_cast<uint64_t>(size.y) *
           static_cast<uint64_t>(size.z);
}

std::string sizeToString(const vc::lasagna::MaxflowInt3& size)
{
    return std::to_string(size.x) + "," + std::to_string(size.y) + "," + std::to_string(size.z);
}

void printTerminal(
    const char* label,
    size_t index,
    const vc::lasagna::MaxflowTerminal& terminal)
{
    std::cout << label << "[" << index << "] pred_dt voxel xyz: "
              << terminal.predVoxelXYZ.x << ","
              << terminal.predVoxelXYZ.y << ","
              << terminal.predVoxelXYZ.z
              << " node=" << terminal.node
              << " exact=" << (terminal.exact ? "yes" : "nearest") << '\n';
}

struct FlowRunResult {
    size_t sourceIndex = 0;
    size_t sinkIndex = 0;
    int32_t sourceNode = -1;
    int32_t sinkNode = -1;
    bool skipped = false;
    std::string error;
    vc::lasagna::EclMaxflowResult flow;
    vc::lasagna::EclTerminalExpansion sourceExpansion;
    vc::lasagna::EclTerminalExpansion sinkExpansion;
    bool usedTerminalFlood = false;
    int terminalFloodDepth = 0;
    int terminalFloodCapacity = 0;
    int terminalFloodDecay = 0;
    uint64_t sourceFloodNodes = 0;
    uint64_t sinkFloodNodes = 0;
    int64_t sourceFloodTerminalCapacity = 0;
    int64_t sinkFloodTerminalCapacity = 0;
    bool usedTerminalRegions = false;
    uint64_t terminalRegionIterations = 0;
    int32_t terminalEdgeCapacity = 0;
    bool terminalExpansionLimitReached = false;
    double wallSeconds = 0.0;
};

struct TerminalFlood {
    std::vector<int32_t> distance;
    std::vector<int32_t> shellCapacity;
    uint64_t nodes = 0;
    int64_t totalShellCapacity = 0;
};

struct TerminalFringeRegions {
    std::vector<uint8_t> sourceRegion;
    std::vector<uint8_t> sinkRegion;
    bool saturated = false;
    int saturationDepth = -1;
    int appliedDepth = 0;
};

struct TerminalFringeStats {
    bool saturated = false;
    int saturationDepth = -1;
    int appliedDepth = 0;
};

enum class FloodScheduleKind {
    Geometric,
    Linear
};

struct FloodSchedule {
    FloodScheduleKind kind = FloodScheduleKind::Geometric;
    int startCapacity = 1024;
    double factor = 2.0;
    int step = 1;
    int maxDepth = 10;
    std::string label;
};

struct TerminalBoundary {
    int64_t capacity = 0;
    uint64_t regionNodes = 0;
    bool touchesOpposite = false;
    std::vector<int32_t> frontier;
};

vc::lasagna::MaxflowGraph buildTerminalRegionGraph(
    const vc::lasagna::MaxflowGraph& base,
    const std::vector<uint8_t>& sourceRegion,
    const std::vector<uint8_t>& sinkRegion,
    int32_t terminalCapacity);

uint64_t countRegionNodes(const std::vector<uint8_t>& region)
{
    return static_cast<uint64_t>(std::count(region.begin(), region.end(), static_cast<uint8_t>(1)));
}

int32_t terminalEdgeCapacityForGraph(const vc::lasagna::MaxflowGraph& graph)
{
    int64_t totalCapacity = 0;
    for (uint64_t node = 0; node < graph.stats.graphNodes; ++node) {
        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            totalCapacity += graph.capacity[static_cast<size_t>(edge)];
        }
    }
    const int64_t safeCap = std::min<int64_t>(
        totalCapacity + 1,
        std::numeric_limits<int32_t>::max() / 16);
    if (safeCap <= 0) {
        return 1;
    }
    return static_cast<int32_t>(safeCap);
}

int32_t geometricCapacity(int baseCapacity, double decay, int distance)
{
    double cap = static_cast<double>(baseCapacity);
    for (int i = 0; i < distance; ++i) {
        cap /= decay;
        if (cap < 1.0) {
            return 0;
        }
    }
    if (cap > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(cap);
}

int32_t scheduleCapacity(const FloodSchedule& schedule, int distance)
{
    if (distance > schedule.maxDepth) {
        return 0;
    }
    if (schedule.kind == FloodScheduleKind::Linear) {
        const int64_t cap = static_cast<int64_t>(schedule.startCapacity) -
                            static_cast<int64_t>(schedule.step) * distance;
        return cap > 0 ? static_cast<int32_t>(std::min<int64_t>(
                             cap,
                             std::numeric_limits<int32_t>::max()))
                       : 0;
    }
    return geometricCapacity(schedule.startCapacity, schedule.factor, distance);
}

int depthUntilOneGeometric(int startCapacity, double factor)
{
    int depth = 0;
    while (geometricCapacity(startCapacity, factor, depth) > 1 &&
           depth < std::numeric_limits<int>::max() / 2) {
        ++depth;
    }
    return depth;
}

int depthUntilOneLinear(int startCapacity, int step)
{
    if (step <= 0) {
        return 0;
    }
    return static_cast<int>((static_cast<int64_t>(startCapacity) - 1) / step);
}

TerminalFlood terminalFlood(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t seedNode,
    const FloodSchedule& schedule)
{
    if (seedNode < 0 || seedNode >= static_cast<int64_t>(graph.stats.graphNodes)) {
        throw std::runtime_error("terminal flood seed is outside graph");
    }
    if (schedule.maxDepth < 0) {
        throw std::runtime_error("terminal flood depth must be non-negative");
    }
    if (schedule.startCapacity <= 0) {
        throw std::runtime_error("terminal flood capacity must be positive");
    }
    if (schedule.kind == FloodScheduleKind::Geometric && !(schedule.factor > 1.0)) {
        throw std::runtime_error("terminal flood decay must be greater than 1");
    }
    if (schedule.kind == FloodScheduleKind::Linear && schedule.step <= 0) {
        throw std::runtime_error("terminal flood linear step must be positive");
    }

    TerminalFlood flood;
    flood.distance.assign(static_cast<size_t>(graph.stats.graphNodes), -1);
    flood.shellCapacity.assign(static_cast<size_t>(graph.stats.graphNodes), 0);
    std::deque<int32_t> queue;
    flood.distance[static_cast<size_t>(seedNode)] = 0;
    queue.push_back(seedNode);

    while (!queue.empty()) {
        const int32_t node = queue.front();
        queue.pop_front();
        const int dist = flood.distance[static_cast<size_t>(node)];
        const int32_t cap = scheduleCapacity(schedule, dist);
        if (cap > 0) {
            flood.shellCapacity[static_cast<size_t>(node)] = cap;
            ++flood.nodes;
            flood.totalShellCapacity += cap;
        }
        if (dist >= schedule.maxDepth) {
            continue;
        }
        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            if (flood.distance[static_cast<size_t>(dst)] < 0) {
                flood.distance[static_cast<size_t>(dst)] = dist + 1;
                queue.push_back(dst);
            }
        }
    }
    return flood;
}

TerminalFlood terminalFlood(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t seedNode,
    int maxDepth,
    int baseCapacity,
    int decay)
{
    FloodSchedule schedule;
    schedule.kind = FloodScheduleKind::Geometric;
    schedule.startCapacity = baseCapacity;
    schedule.factor = static_cast<double>(decay);
    schedule.maxDepth = maxDepth;
    return terminalFlood(graph, seedNode, schedule);
}

vc::lasagna::MaxflowGraph buildTerminalFloodGraph(
    const vc::lasagna::MaxflowGraph& base,
    int32_t sourceNode,
    int32_t sinkNode,
    const TerminalFlood& sourceFlood,
    const TerminalFlood& sinkFlood,
    int32_t terminalCapacity)
{
    vc::lasagna::MaxflowGraph graph;
    const uint64_t baseNodes = base.stats.graphNodes;
    const uint64_t sourceSuper = baseNodes;
    const uint64_t sinkSuper = baseNodes + 1;
    graph.stats = base.stats;
    graph.stats.graphNodes = baseNodes + 2;
    graph.nindex.assign(static_cast<size_t>(graph.stats.graphNodes) + 1, 0);

    for (uint64_t node = 0; node < baseNodes; ++node) {
        graph.nindex[static_cast<size_t>(node) + 1] =
            base.nindex[static_cast<size_t>(node) + 1] - base.nindex[static_cast<size_t>(node)];
        if (static_cast<int32_t>(node) == sinkNode) {
            ++graph.nindex[static_cast<size_t>(node) + 1];
        }
    }
    ++graph.nindex[static_cast<size_t>(sourceSuper) + 1];
    for (size_t i = 1; i < graph.nindex.size(); ++i) {
        graph.nindex[i] += graph.nindex[i - 1];
    }

    graph.stats.directedEdges = base.stats.directedEdges + 2;
    graph.nlist.assign(static_cast<size_t>(graph.stats.directedEdges), 0);
    graph.capacity.assign(static_cast<size_t>(graph.stats.directedEdges), 1);
    std::vector<uint64_t> cursor = graph.nindex;
    for (uint64_t node = 0; node < baseNodes; ++node) {
        for (uint64_t edge = base.nindex[static_cast<size_t>(node)];
             edge < base.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = base.nlist[static_cast<size_t>(edge)];
            const uint64_t out = cursor[static_cast<size_t>(node)]++;
            graph.nlist[static_cast<size_t>(out)] = dst;
            int32_t capacity = base.capacity[static_cast<size_t>(edge)];
            const int32_t sourceDist = sourceFlood.distance[static_cast<size_t>(node)];
            const int32_t dstSourceDist = sourceFlood.distance[static_cast<size_t>(dst)];
            if (sourceDist >= 0 && dstSourceDist == sourceDist + 1) {
                capacity = std::max(capacity, sourceFlood.shellCapacity[static_cast<size_t>(node)]);
            }
            const int32_t sinkDist = sinkFlood.distance[static_cast<size_t>(node)];
            const int32_t dstSinkDist = sinkFlood.distance[static_cast<size_t>(dst)];
            if (dstSinkDist >= 0 && sinkDist == dstSinkDist + 1) {
                capacity = std::max(capacity, sinkFlood.shellCapacity[static_cast<size_t>(dst)]);
            }
            graph.capacity[static_cast<size_t>(out)] = capacity;
        }
        if (static_cast<int32_t>(node) == sinkNode) {
            const uint64_t out = cursor[static_cast<size_t>(node)]++;
            graph.nlist[static_cast<size_t>(out)] = static_cast<int32_t>(sinkSuper);
            graph.capacity[static_cast<size_t>(out)] = terminalCapacity;
        }
    }
    const uint64_t out = cursor[static_cast<size_t>(sourceSuper)]++;
    graph.nlist[static_cast<size_t>(out)] = sourceNode;
    graph.capacity[static_cast<size_t>(out)] = terminalCapacity;
    return graph;
}

vc::lasagna::EclMaxflowResult runEclWithTerminalFlood(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    int floodDepth,
    int floodCapacity,
    int floodDecay,
    FlowRunResult& result)
{
    const auto sourceFlood = terminalFlood(graph, sourceNode, floodDepth, floodCapacity, floodDecay);
    const auto sinkFlood = terminalFlood(graph, sinkNode, floodDepth, floodCapacity, floodDecay);
    result.usedTerminalFlood = true;
    result.terminalFloodDepth = floodDepth;
    result.terminalFloodCapacity = floodCapacity;
    result.terminalFloodDecay = floodDecay;
    result.sourceFloodNodes = sourceFlood.nodes;
    result.sinkFloodNodes = sinkFlood.nodes;
    result.sourceFloodTerminalCapacity = sourceFlood.totalShellCapacity;
    result.sinkFloodTerminalCapacity = sinkFlood.totalShellCapacity;

    const auto terminalGraph = buildTerminalFloodGraph(
        graph,
        sourceNode,
        sinkNode,
        sourceFlood,
        sinkFlood,
        terminalEdgeCapacityForGraph(graph));
    vc::lasagna::EclMaxflowOptions eclOptions;
    eclOptions.runs = runs;
    eclOptions.computeMinCut = false;
    return vc::lasagna::runEclMaxflow(
        terminalGraph,
        static_cast<int32_t>(graph.stats.graphNodes),
        static_cast<int32_t>(graph.stats.graphNodes + 1),
        eclOptions);
}

vc::lasagna::EclMaxflowResult runEclWithTerminalFloodSchedule(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    const FloodSchedule& schedule)
{
    const auto sourceFlood = terminalFlood(graph, sourceNode, schedule);
    const auto sinkFlood = terminalFlood(graph, sinkNode, schedule);
    const auto terminalGraph = buildTerminalFloodGraph(
        graph,
        sourceNode,
        sinkNode,
        sourceFlood,
        sinkFlood,
        terminalEdgeCapacityForGraph(graph));
    vc::lasagna::EclMaxflowOptions eclOptions;
    eclOptions.runs = runs;
    eclOptions.computeMinCut = false;
    return vc::lasagna::runEclMaxflow(
        terminalGraph,
        static_cast<int32_t>(graph.stats.graphNodes),
        static_cast<int32_t>(graph.stats.graphNodes + 1),
        eclOptions);
}

vc::lasagna::EclMaxflowResult runEclWithTerminalFloodScheduleQuiet(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    const FloodSchedule& schedule)
{
    std::fflush(stdout);
    const int savedStdout = dup(STDOUT_FILENO);
    const int devNull = open("/dev/null", O_WRONLY);
    if (savedStdout < 0 || devNull < 0) {
        if (savedStdout >= 0) close(savedStdout);
        if (devNull >= 0) close(devNull);
        return runEclWithTerminalFloodSchedule(graph, sourceNode, sinkNode, runs, schedule);
    }
    dup2(devNull, STDOUT_FILENO);
    close(devNull);

    try {
        auto result = runEclWithTerminalFloodSchedule(graph, sourceNode, sinkNode, runs, schedule);
        std::fflush(stdout);
        dup2(savedStdout, STDOUT_FILENO);
        close(savedStdout);
        return result;
    } catch (...) {
        std::fflush(stdout);
        dup2(savedStdout, STDOUT_FILENO);
        close(savedStdout);
        throw;
    }
}

TerminalFringeRegions terminalFringeRegions(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int maxDepth)
{
    if (sourceNode < 0 || sourceNode >= static_cast<int64_t>(graph.stats.graphNodes)) {
        throw std::runtime_error("terminal fringe source is outside graph");
    }
    if (sinkNode < 0 || sinkNode >= static_cast<int64_t>(graph.stats.graphNodes)) {
        throw std::runtime_error("terminal fringe sink is outside graph");
    }
    if (maxDepth < 0) {
        throw std::runtime_error("terminal fringe depth must be non-negative");
    }

    TerminalFringeRegions regions;
    const size_t nodeCount = static_cast<size_t>(graph.stats.graphNodes);
    regions.sourceRegion.assign(nodeCount, 0);
    regions.sinkRegion.assign(nodeCount, 0);
    regions.sourceRegion[static_cast<size_t>(sourceNode)] = 1;
    regions.sinkRegion[static_cast<size_t>(sinkNode)] = 1;

    std::vector<int32_t> sourceFrontier{sourceNode};
    std::vector<int32_t> sinkFrontier{sinkNode};
    std::vector<uint8_t> sourceProposal(nodeCount, 0);
    std::vector<uint8_t> sinkProposal(nodeCount, 0);

    const auto collectProposals = [&graph](
                                      const std::vector<int32_t>& frontier,
                                      const std::vector<uint8_t>& ownRegion,
                                      const std::vector<uint8_t>& otherRegion,
                                      std::vector<uint8_t>& proposal,
                                      std::vector<int32_t>& proposed) {
        bool touchesOther = false;
        for (const int32_t node : frontier) {
            if (otherRegion[static_cast<size_t>(node)] != 0) {
                touchesOther = true;
                continue;
            }
            for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
                 edge < graph.nindex[static_cast<size_t>(node) + 1];
                 ++edge) {
                const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
                const size_t dstIndex = static_cast<size_t>(dst);
                if (otherRegion[dstIndex] != 0) {
                    touchesOther = true;
                    continue;
                }
                if (ownRegion[dstIndex] == 0 && proposal[dstIndex] == 0) {
                    proposal[dstIndex] = 1;
                    proposed.push_back(dst);
                }
            }
        }
        return touchesOther;
    };

    for (int depth = 0; depth < maxDepth; ++depth) {
        if (sourceFrontier.empty() && sinkFrontier.empty()) {
            break;
        }

        std::vector<int32_t> sourceProposed;
        std::vector<int32_t> sinkProposed;
        bool touchesOther = collectProposals(
            sourceFrontier,
            regions.sourceRegion,
            regions.sinkRegion,
            sourceProposal,
            sourceProposed);
        touchesOther = collectProposals(
            sinkFrontier,
            regions.sinkRegion,
            regions.sourceRegion,
            sinkProposal,
            sinkProposed) || touchesOther;

        for (const int32_t node : sourceProposed) {
            if (sinkProposal[static_cast<size_t>(node)] != 0) {
                touchesOther = true;
                break;
            }
        }

        if (touchesOther) {
            for (const int32_t node : sourceProposed) {
                sourceProposal[static_cast<size_t>(node)] = 0;
            }
            for (const int32_t node : sinkProposed) {
                sinkProposal[static_cast<size_t>(node)] = 0;
            }
            regions.saturated = true;
            regions.saturationDepth = depth + 1;
            break;
        }

        for (const int32_t node : sourceProposed) {
            regions.sourceRegion[static_cast<size_t>(node)] = 1;
            sourceProposal[static_cast<size_t>(node)] = 0;
        }
        for (const int32_t node : sinkProposed) {
            regions.sinkRegion[static_cast<size_t>(node)] = 1;
            sinkProposal[static_cast<size_t>(node)] = 0;
        }

        std::vector<int32_t> nextSourceFrontier;
        std::vector<int32_t> nextSinkFrontier;
        for (const int32_t node : sourceProposed) {
            if (regions.sinkRegion[static_cast<size_t>(node)] == 0) {
                nextSourceFrontier.push_back(node);
            }
        }
        for (const int32_t node : sinkProposed) {
            if (regions.sourceRegion[static_cast<size_t>(node)] == 0) {
                nextSinkFrontier.push_back(node);
            }
        }

        sourceFrontier = std::move(nextSourceFrontier);
        sinkFrontier = std::move(nextSinkFrontier);
        regions.appliedDepth = depth + 1;
    }
    return regions;
}

vc::lasagna::EclMaxflowResult runEclWithTerminalFringe(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    int fringeDepth,
    TerminalFringeStats* stats = nullptr)
{
    const auto regions = terminalFringeRegions(graph, sourceNode, sinkNode, fringeDepth);
    if (stats != nullptr) {
        stats->saturated = regions.saturated;
        stats->saturationDepth = regions.saturationDepth;
        stats->appliedDepth = regions.appliedDepth;
    }
    const auto terminalGraph = buildTerminalRegionGraph(
        graph,
        regions.sourceRegion,
        regions.sinkRegion,
        1);
    vc::lasagna::EclMaxflowOptions eclOptions;
    eclOptions.runs = runs;
    eclOptions.computeMinCut = false;
    return vc::lasagna::runEclMaxflow(
        terminalGraph,
        static_cast<int32_t>(graph.stats.graphNodes),
        static_cast<int32_t>(graph.stats.graphNodes + 1),
        eclOptions);
}

vc::lasagna::EclMaxflowResult runEclWithTerminalFringeQuiet(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    int fringeDepth,
    TerminalFringeStats* stats = nullptr)
{
    std::fflush(stdout);
    const int savedStdout = dup(STDOUT_FILENO);
    const int devNull = open("/dev/null", O_WRONLY);
    if (savedStdout < 0 || devNull < 0) {
        if (savedStdout >= 0) close(savedStdout);
        if (devNull >= 0) close(devNull);
        return runEclWithTerminalFringe(graph, sourceNode, sinkNode, runs, fringeDepth, stats);
    }
    dup2(devNull, STDOUT_FILENO);
    close(devNull);

    try {
        auto result = runEclWithTerminalFringe(graph, sourceNode, sinkNode, runs, fringeDepth, stats);
        std::fflush(stdout);
        dup2(savedStdout, STDOUT_FILENO);
        close(savedStdout);
        return result;
    } catch (...) {
        std::fflush(stdout);
        dup2(savedStdout, STDOUT_FILENO);
        close(savedStdout);
        throw;
    }
}

std::vector<FloodSchedule> makeFloodSweepSchedules()
{
    const std::vector<int> starts{128, 256, 512, 1024};
    const std::vector<double> factors{4.0, 2.0, 1.5, 1.25};
    const std::vector<int> linearSteps{1, 4, 16};

    std::vector<FloodSchedule> schedules;
    for (int start : starts) {
        for (double factor : factors) {
            FloodSchedule schedule;
            schedule.kind = FloodScheduleKind::Geometric;
            schedule.startCapacity = start;
            schedule.factor = factor;
            schedule.maxDepth = depthUntilOneGeometric(start, factor);
            std::ostringstream label;
            label << "geom start=" << start << " factor=" << factor
                  << " depth=" << schedule.maxDepth;
            schedule.label = label.str();
            schedules.push_back(std::move(schedule));
        }
        for (int step : linearSteps) {
            FloodSchedule schedule;
            schedule.kind = FloodScheduleKind::Linear;
            schedule.startCapacity = start;
            schedule.step = step;
            schedule.maxDepth = depthUntilOneLinear(start, step);
            std::ostringstream label;
            label << "linear start=" << start << " step=" << step
                  << " depth=" << schedule.maxDepth;
            schedule.label = label.str();
            schedules.push_back(std::move(schedule));
        }
    }
    return schedules;
}

std::vector<FloodSchedule> makeLinearSweepSchedules()
{
    const std::vector<int> starts{128, 256, 512, 1024};
    const std::vector<int> linearSteps{1, 4, 16};

    std::vector<FloodSchedule> schedules;
    for (int start : starts) {
        for (int step : linearSteps) {
            FloodSchedule schedule;
            schedule.kind = FloodScheduleKind::Linear;
            schedule.startCapacity = start;
            schedule.step = step;
            schedule.maxDepth = depthUntilOneLinear(start, step);
            std::ostringstream label;
            label << "linear start=" << start << " step=" << step
                  << " depth=" << schedule.maxDepth;
            schedule.label = label.str();
            schedules.push_back(std::move(schedule));
        }
    }
    return schedules;
}

const char* scheduleKindName(FloodScheduleKind kind)
{
    switch (kind) {
    case FloodScheduleKind::Geometric:
        return "geom";
    case FloodScheduleKind::Linear:
        return "linear";
    }
    return "unknown";
}

std::string scheduleParamString(const FloodSchedule& schedule)
{
    std::ostringstream out;
    if (schedule.kind == FloodScheduleKind::Geometric) {
        out << schedule.factor;
    } else {
        out << schedule.step;
    }
    return out.str();
}

std::string ratioString(int64_t numerator, int64_t denominator)
{
    if (denominator == 0) {
        return "inf";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << static_cast<double>(numerator) / static_cast<double>(denominator);
    return out.str();
}

std::string ratioString(double numerator, double denominator)
{
    if (denominator == 0.0) {
        return "inf";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << numerator / denominator;
    return out.str();
}

std::string laplaceRatioString(double numerator, double denominator)
{
    constexpr double noiseFloor = kDefaultLaplaceSolverTolerance * kLaplaceRatioConfidenceFactor;
    const std::string ratio = ratioString(numerator, denominator);
    if (std::max(std::abs(numerator), std::abs(denominator)) < noiseFloor) {
        return "(" + ratio + ")";
    }
    return ratio;
}

std::string valueString(double value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(6) << value;
    return out.str();
}

double laplaceConfidenceFloor()
{
    return kDefaultLaplaceSolverTolerance * kLaplaceRatioConfidenceFactor;
}

struct LaplaceCliEvaluation {
    double lambda = 0.0;
    std::string status;
    double solveSeconds = 0.0;
    bool solved = false;
    std::vector<double> sinkValues;
    double maxAbsSinkValue = 0.0;
    double minAbsSinkValue = 0.0;
};

bool laplaceAccepted(const LaplaceCliEvaluation& evaluation)
{
    return evaluation.solved && evaluation.maxAbsSinkValue >= laplaceConfidenceFloor();
}

bool laplaceFastStop(const LaplaceCliEvaluation& evaluation)
{
    return evaluation.sinkValues.size() == 2 &&
           laplaceAccepted(evaluation) &&
           evaluation.minAbsSinkValue < laplaceConfidenceFloor();
}

TerminalBoundary terminalBoundary(
    const vc::lasagna::MaxflowGraph& graph,
    const std::vector<uint8_t>& region,
    const std::vector<uint8_t>& oppositeRegion,
    vc::lasagna::EclTerminalSide side)
{
    TerminalBoundary boundary;
    boundary.regionNodes = countRegionNodes(region);
    std::vector<uint8_t> inFrontier(region.size(), 0);
    const auto addFrontier = [&](int32_t node) {
        const size_t idx = static_cast<size_t>(node);
        if (oppositeRegion[idx] != 0) {
            boundary.touchesOpposite = true;
            return;
        }
        if (region[idx] == 0 && inFrontier[idx] == 0) {
            inFrontier[idx] = 1;
            boundary.frontier.push_back(node);
        }
    };

    for (uint64_t src = 0; src < graph.stats.graphNodes; ++src) {
        for (uint64_t edge = graph.nindex[static_cast<size_t>(src)];
             edge < graph.nindex[static_cast<size_t>(src) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            if (side == vc::lasagna::EclTerminalSide::Source) {
                if (region[static_cast<size_t>(src)] != 0 && region[static_cast<size_t>(dst)] == 0) {
                    boundary.capacity += graph.capacity[static_cast<size_t>(edge)];
                    addFrontier(dst);
                }
            } else {
                if (region[static_cast<size_t>(src)] == 0 && region[static_cast<size_t>(dst)] != 0) {
                    boundary.capacity += graph.capacity[static_cast<size_t>(edge)];
                    addFrontier(static_cast<int32_t>(src));
                }
            }
        }
    }
    return boundary;
}

vc::lasagna::MaxflowGraph buildTerminalRegionGraph(
    const vc::lasagna::MaxflowGraph& base,
    const std::vector<uint8_t>& sourceRegion,
    const std::vector<uint8_t>& sinkRegion,
    int32_t terminalCapacity)
{
    vc::lasagna::MaxflowGraph graph;
    const uint64_t baseNodes = base.stats.graphNodes;
    const uint64_t sourceSuper = baseNodes;
    const uint64_t sinkSuper = baseNodes + 1;
    graph.stats = base.stats;
    graph.stats.graphNodes = baseNodes + 2;
    graph.nindex.assign(static_cast<size_t>(graph.stats.graphNodes) + 1, 0);

    uint64_t terminalEdges = 0;
    for (uint64_t node = 0; node < baseNodes; ++node) {
        graph.nindex[static_cast<size_t>(node) + 1] =
            base.nindex[static_cast<size_t>(node) + 1] - base.nindex[static_cast<size_t>(node)];
        if (sinkRegion[static_cast<size_t>(node)] != 0) {
            ++graph.nindex[static_cast<size_t>(node) + 1];
            ++terminalEdges;
        }
        if (sourceRegion[static_cast<size_t>(node)] != 0) {
            ++graph.nindex[static_cast<size_t>(sourceSuper) + 1];
            ++terminalEdges;
        }
    }
    for (size_t i = 1; i < graph.nindex.size(); ++i) {
        graph.nindex[i] += graph.nindex[i - 1];
    }

    graph.stats.directedEdges = base.stats.directedEdges + terminalEdges;
    graph.nlist.assign(static_cast<size_t>(graph.stats.directedEdges), 0);
    graph.capacity.assign(static_cast<size_t>(graph.stats.directedEdges), 1);
    std::vector<uint64_t> cursor = graph.nindex;
    for (uint64_t node = 0; node < baseNodes; ++node) {
        for (uint64_t edge = base.nindex[static_cast<size_t>(node)];
             edge < base.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const uint64_t out = cursor[static_cast<size_t>(node)]++;
            graph.nlist[static_cast<size_t>(out)] = base.nlist[static_cast<size_t>(edge)];
            graph.capacity[static_cast<size_t>(out)] = base.capacity[static_cast<size_t>(edge)];
        }
        if (sinkRegion[static_cast<size_t>(node)] != 0) {
            const uint64_t out = cursor[static_cast<size_t>(node)]++;
            graph.nlist[static_cast<size_t>(out)] = static_cast<int32_t>(sinkSuper);
            graph.capacity[static_cast<size_t>(out)] = terminalCapacity;
        }
        if (sourceRegion[static_cast<size_t>(node)] != 0) {
            const uint64_t out = cursor[static_cast<size_t>(sourceSuper)]++;
            graph.nlist[static_cast<size_t>(out)] = static_cast<int32_t>(node);
            graph.capacity[static_cast<size_t>(out)] = terminalCapacity;
        }
    }
    return graph;
}

vc::lasagna::EclMaxflowResult runEclWithTerminalExpansion(
    const vc::lasagna::MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs,
    uint64_t maxIterations,
    FlowRunResult& result)
{
    std::vector<uint8_t> sourceRegion(static_cast<size_t>(graph.stats.graphNodes), 0);
    std::vector<uint8_t> sinkRegion(static_cast<size_t>(graph.stats.graphNodes), 0);
    sourceRegion[static_cast<size_t>(sourceNode)] = 1;
    sinkRegion[static_cast<size_t>(sinkNode)] = 1;
    result.terminalEdgeCapacity = terminalEdgeCapacityForGraph(graph);

    vc::lasagna::EclMaxflowResult flow;
    for (uint64_t iter = 0; iter <= maxIterations; ++iter) {
        const auto terminalGraph = buildTerminalRegionGraph(
            graph,
            sourceRegion,
            sinkRegion,
            result.terminalEdgeCapacity);
        vc::lasagna::EclMaxflowOptions eclOptions;
        eclOptions.runs = runs;
        eclOptions.computeMinCut = false;
        flow = vc::lasagna::runEclMaxflow(
            terminalGraph,
            static_cast<int32_t>(graph.stats.graphNodes),
            static_cast<int32_t>(graph.stats.graphNodes + 1),
            eclOptions);

        const auto sourceBoundary = terminalBoundary(
            graph,
            sourceRegion,
            sinkRegion,
            vc::lasagna::EclTerminalSide::Source);
        const auto sinkBoundary = terminalBoundary(
            graph,
            sinkRegion,
            sourceRegion,
            vc::lasagna::EclTerminalSide::Sink);
        result.sourceExpansion.valid = true;
        result.sourceExpansion.side = vc::lasagna::EclTerminalSide::Source;
        result.sourceExpansion.iterations = iter;
        result.sourceExpansion.regionNodes = sourceBoundary.regionNodes;
        result.sourceExpansion.finalBoundaryCapacity = sourceBoundary.capacity;
        result.sourceExpansion.finalBoundaryIsMinCut = sourceBoundary.capacity == flow.maxFlow;
        result.sourceExpansion.touchedOppositeTerminal = sourceBoundary.touchesOpposite;
        result.sinkExpansion.valid = true;
        result.sinkExpansion.side = vc::lasagna::EclTerminalSide::Sink;
        result.sinkExpansion.iterations = iter;
        result.sinkExpansion.regionNodes = sinkBoundary.regionNodes;
        result.sinkExpansion.finalBoundaryCapacity = sinkBoundary.capacity;
        result.sinkExpansion.finalBoundaryIsMinCut = sinkBoundary.capacity == flow.maxFlow;
        result.sinkExpansion.touchedOppositeTerminal = sinkBoundary.touchesOpposite;

        if (iter == maxIterations) {
            result.terminalExpansionLimitReached =
                sourceBoundary.capacity == flow.maxFlow || sinkBoundary.capacity == flow.maxFlow;
            break;
        }

        bool absorbed = false;
        if (sourceBoundary.capacity == flow.maxFlow && !sourceBoundary.touchesOpposite) {
            for (int32_t node : sourceBoundary.frontier) {
                if (sourceRegion[static_cast<size_t>(node)] == 0) {
                    sourceRegion[static_cast<size_t>(node)] = 1;
                    ++result.sourceExpansion.absorbedNodes;
                    absorbed = true;
                }
            }
        }
        if (sinkBoundary.capacity == flow.maxFlow && !sinkBoundary.touchesOpposite) {
            for (int32_t node : sinkBoundary.frontier) {
                if (sinkRegion[static_cast<size_t>(node)] == 0) {
                    sinkRegion[static_cast<size_t>(node)] = 1;
                    ++result.sinkExpansion.absorbedNodes;
                    absorbed = true;
                }
            }
        }
        if (!absorbed) {
            break;
        }
        result.terminalRegionIterations = iter + 1;
    }
    result.sourceExpansion.regionNodes = countRegionNodes(sourceRegion);
    result.sinkExpansion.regionNodes = countRegionNodes(sinkRegion);
    return flow;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage(argv[0]);
        return argc < 2 ? 2 : 0;
    }

    try {
        std::filesystem::path manifestPath;
        vc::lasagna::MaxflowManifestBuildOptions options;
        bool runEcl = false;
        int runs = 1;
        int terminalFloodDepth = 10;
        int terminalFloodCapacity = 1024;
        int terminalFloodDecay = 2;
        uint64_t terminalRegionIterations = 0;
        bool terminalFloodSweep = false;
        bool terminalFringeSweep = false;
        bool runLaplaceAmgx = false;
        bool laplaceAdaptiveLambda = false;
        double laplaceAdaptiveStartLambda = 1.0 / 2048.0;
        std::vector<double> laplaceLambdas{
            1.0 / 64.0,
            1.0 / 128.0,
            1.0 / 256.0,
            1.0 / 512.0,
            1.0 / 1024.0,
            1.0 / 2048.0,
            1.0 / 4096.0,
            1.0 / 8192.0,
            1.0 / 16384.0,
            1.0 / 32768.0,
            1.0 / 65536.0,
            1.0 / 131072.0,
        };
        int laplaceSourceDepth = 0;
        std::filesystem::path amgxConfigPath;

        manifestPath = argv[1];
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto requireValue = [&](const char* opt) -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(opt) + " requires a value");
                }
                return argv[++i];
            };

            if (arg == "--src") {
                options.sourcesBase.push_back(parsePoint(requireValue("--src"), "--src"));
            } else if (arg == "--sink") {
                options.sinksBase.push_back(parsePoint(requireValue("--sink"), "--sink"));
            } else if (arg == "--margin-base-voxels") {
                options.marginBaseVoxels = parseNonNegativeInt64(
                    requireValue("--margin-base-voxels"),
                    "margin-base-voxels");
            } else if (arg == "--threshold") {
                options.threshold = parseThreshold(requireValue("--threshold"));
            } else if (arg == "--run-ecl") {
                runEcl = true;
            } else if (arg == "--runs") {
                runs = parsePositiveInt(requireValue("--runs"), "runs");
            } else if (arg == "--terminal-flood-depth") {
                const auto parsed = parseNonNegativeInt64(
                    requireValue("--terminal-flood-depth"),
                    "terminal-flood-depth");
                if (parsed > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("terminal-flood-depth must fit int32");
                }
                terminalFloodDepth = static_cast<int>(parsed);
            } else if (arg == "--terminal-flood-capacity") {
                terminalFloodCapacity = parsePositiveInt(
                    requireValue("--terminal-flood-capacity"),
                    "terminal-flood-capacity");
            } else if (arg == "--terminal-flood-decay") {
                terminalFloodDecay = parsePositiveInt(
                    requireValue("--terminal-flood-decay"),
                    "terminal-flood-decay");
                if (terminalFloodDecay <= 1) {
                    throw std::invalid_argument("terminal-flood-decay must be greater than 1");
                }
            } else if (arg == "--terminal-region-iterations") {
                terminalRegionIterations = static_cast<uint64_t>(parseNonNegativeInt64(
                    requireValue("--terminal-region-iterations"),
                    "terminal-region-iterations"));
            } else if (arg == "--terminal-flood-sweep") {
                terminalFloodSweep = true;
            } else if (arg == "--terminal-fringe-sweep") {
                terminalFringeSweep = true;
            } else if (arg == "--run-laplace-amgx") {
                runLaplaceAmgx = true;
            } else if (arg == "--laplace-adaptive-lambda") {
                laplaceAdaptiveLambda = true;
            } else if (arg == "--laplace-adaptive-start-lambda") {
                laplaceAdaptiveStartLambda = parsePositiveDouble(
                    requireValue("--laplace-adaptive-start-lambda"),
                    "laplace-adaptive-start-lambda");
            } else if (arg == "--laplace-lambda-sweep") {
                laplaceLambdas = parsePositiveDoubleList(
                    requireValue("--laplace-lambda-sweep"),
                    "laplace-lambda-sweep");
            } else if (arg == "--laplace-length-sweep") {
                const auto lengths = parsePositiveDoubleList(
                    requireValue("--laplace-length-sweep"),
                    "laplace-length-sweep");
                laplaceLambdas.clear();
                laplaceLambdas.reserve(lengths.size());
                for (double length : lengths) {
                    laplaceLambdas.push_back(1.0 / (length * length));
                }
            } else if (arg == "--laplace-source-depth") {
                const auto parsed = parseNonNegativeInt64(
                    requireValue("--laplace-source-depth"),
                    "laplace-source-depth");
                if (parsed > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("laplace-source-depth must fit int32");
                }
                laplaceSourceDepth = static_cast<int>(parsed);
            } else if (arg == "--amgx-config") {
                amgxConfigPath = requireValue("--amgx-config");
            } else if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        if (options.sourcesBase.empty() || options.sinksBase.empty()) {
            throw std::invalid_argument("--src and --sink are required");
        }
        std::sort(laplaceLambdas.begin(), laplaceLambdas.end(), std::greater<double>());
        options.sourceBase = options.sourcesBase.front();
        options.sinkBase = options.sinksBase.front();

        const auto wallStart = std::chrono::steady_clock::now();
        const auto report = vc::lasagna::buildMaxflowGraphFromManifest(manifestPath, options);
        const auto wallTime = std::chrono::steady_clock::now() - wallStart;

        const auto& graph = report.graph;
        const auto& stats = graph.stats;
        const auto& pred = report.predDt;

        std::cout.imbue(std::locale::classic());
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Lasagna maxflow graph build\n";
        std::cout << "manifest: " << report.manifest.manifestPath << '\n';
        std::cout << "pred_dt zarr: " << pred.zarrPath << '\n';
        std::cout << "crop base xyz: " << boxToString(pred.cropBaseXYZ) << '\n';
        std::cout << "crop pred_dt xyz: " << boxToString(pred.cropPredXYZ) << '\n';
        std::cout << "used pred_dt size xyz: " << sizeToString(boxSize(pred.cropPredXYZ)) << '\n';
        std::cout << "used pred_dt voxels: " << boxVoxelCount(pred.cropPredXYZ) << '\n';
        std::cout << "pred_dt shape zyx: " << pred.shapeZYX[0] << "," << pred.shapeZYX[1] << ","
                  << pred.shapeZYX[2] << '\n';
        std::cout << "pred_dt chunk zyx: " << pred.chunksZYX[0] << "," << pred.chunksZYX[1] << ","
                  << pred.chunksZYX[2] << '\n';
        std::cout << "pred_dt base spacing: " << pred.spacingBase << '\n';
        std::cout << "threshold: pred_dt > " << static_cast<int>(options.threshold) << '\n';
        for (size_t i = 0; i < report.sources.size(); ++i) {
            printTerminal("source", i, report.sources[i]);
        }
        for (size_t i = 0; i < report.sinks.size(); ++i) {
            printTerminal("sink", i, report.sinks[i]);
        }
        std::cout << "passable voxels: " << stats.passableVoxels << '\n';
        std::cout << "voxel nodes: " << stats.leafVoxels << '\n';
        std::cout << "graph nodes: " << stats.graphNodes << '\n';
        std::cout << "directed edges: " << stats.directedEdges << '\n';
        std::cout << "undirected edges: " << stats.undirectedEdges << '\n';
        std::cout << "average edges/node: " << stats.averageEdgesPerNode << '\n';
        std::cout << "total face-contact capacity: " << stats.totalExternalCapacity << '\n';
        std::cout << "uncontracted passable graph estimate: nodes="
                  << stats.uncontractedPassableVoxelGraphNodes
                  << " undirected_edges=" << stats.uncontractedPassableVoxelGraphUndirectedEdges << '\n';
        std::cout << "contraction ratio: " << stats.contractionRatio << '\n';
        std::cout << "graph build time seconds: " << stats.buildTime.count() << '\n';
        std::cout << "graph pipeline total time seconds: " << std::chrono::duration<double>(wallTime).count() << '\n';

        if (runLaplaceAmgx) {
            std::cout << "AMGX screened-Laplace results:\n";
            std::cout << "  available: " << (vc::lasagna::laplaceAmgxAvailable() ? "yes" : "no") << '\n';
            std::cout << "  source_depth: " << laplaceSourceDepth << '\n';
            std::cout << "  confidence_floor: " << valueString(laplaceConfidenceFloor()) << '\n';
            if (laplaceAdaptiveLambda) {
                std::cout << "  adaptive_start_lambda: " << valueString(laplaceAdaptiveStartLambda) << '\n';
                std::cout << "  adaptive_bracket_factor: 16\n";
                std::cout << "  adaptive_min_refine_factor: 2\n";
            } else {
                std::cout << "  lambdas: ";
                for (size_t i = 0; i < laplaceLambdas.size(); ++i) {
                    if (i != 0) {
                        std::cout << ",";
                    }
                    std::cout << valueString(laplaceLambdas[i]);
                }
                std::cout << '\n';
            }

            const auto evaluateLaplace = [&](
                                             int32_t sourceNode,
                                             double lambda) -> LaplaceCliEvaluation {
                LaplaceCliEvaluation evaluation;
                evaluation.lambda = lambda;
                evaluation.sinkValues.assign(report.sinks.size(), 0.0);
                try {
                    vc::lasagna::LaplaceAmgxOptions laplaceOptions;
                    laplaceOptions.lambda = lambda;
                    laplaceOptions.sourceDepth = laplaceSourceDepth;
                    laplaceOptions.configPath = amgxConfigPath;
                    const auto result = vc::lasagna::solveScreenedLaplaceAmgx(
                        graph,
                        sourceNode,
                        laplaceOptions);
                    evaluation.status = result.status;
                    evaluation.solveSeconds = result.solveSeconds;
                    evaluation.solved = result.success;
                    for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                        const int32_t sinkNode = report.sinks[sinkIndex].node;
                        if (sinkNode < 0 ||
                            static_cast<uint64_t>(sinkNode) >= graph.stats.graphNodes) {
                            throw std::runtime_error("Laplace sink node is outside graph");
                        }
                        evaluation.sinkValues[sinkIndex] =
                            result.valuesByNode[static_cast<size_t>(sinkNode)];
                    }
                    if (!evaluation.sinkValues.empty()) {
                        evaluation.minAbsSinkValue = std::numeric_limits<double>::infinity();
                        for (double value : evaluation.sinkValues) {
                            const double absValue = std::abs(value);
                            evaluation.maxAbsSinkValue = std::max(evaluation.maxAbsSinkValue, absValue);
                            evaluation.minAbsSinkValue = std::min(evaluation.minAbsSinkValue, absValue);
                        }
                    }
                } catch (const std::exception& e) {
                    evaluation.status = e.what();
                }
                return evaluation;
            };

            const auto printLaplaceHeader = [&](bool includeNote) {
                std::cout << std::left
                          << std::setw(8) << "source"
                          << std::setw(14) << "lambda"
                          << std::setw(14) << "status"
                          << std::setw(14) << "solve_s";
                if (includeNote) {
                    std::cout << std::setw(12) << "note";
                }
                for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                    std::ostringstream col;
                    col << "t" << sinkIndex;
                    std::cout << std::setw(16) << col.str();
                }
                if (report.sinks.size() >= 2) {
                    std::cout << std::setw(16) << "t0/t1";
                }
                std::cout << '\n';
            };

            const auto printLaplaceRow = [&](
                                             size_t sourceIndex,
                                             const LaplaceCliEvaluation& evaluation,
                                             const char* note) {
                std::cout << std::left
                          << std::setw(8) << sourceIndex
                          << std::setw(14) << valueString(evaluation.lambda)
                          << std::setw(14) << (evaluation.solved ? evaluation.status : "error")
                          << std::setw(14) << valueString(evaluation.solveSeconds);
                if (note != nullptr) {
                    std::cout << std::setw(12) << note;
                }
                for (double value : evaluation.sinkValues) {
                    std::cout << std::setw(16) << (evaluation.solved ? valueString(value) : "-");
                }
                if (report.sinks.size() >= 2) {
                    std::cout << std::setw(16)
                              << (evaluation.solved
                                      ? laplaceRatioString(evaluation.sinkValues[0], evaluation.sinkValues[1])
                                      : "-");
                }
                if (!evaluation.solved && !evaluation.status.empty()) {
                    std::cout << "  " << evaluation.status;
                }
                std::cout << '\n';
            };

            printLaplaceHeader(laplaceAdaptiveLambda);

            for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                const int32_t sourceNode = report.sources[sourceIndex].node;
                if (laplaceAdaptiveLambda) {
                    constexpr double bracketFactor = 16.0;
                    constexpr double minRefineFactor = 2.0;
                    constexpr int maxProbeSteps = 16;
                    constexpr int maxRefineSteps = 16;

                    LaplaceCliEvaluation best;
                    bool haveBest = false;
                    double rejectedHigh = 0.0;
                    double acceptedLow = 0.0;

                    auto start = evaluateLaplace(sourceNode, laplaceAdaptiveStartLambda);
                    printLaplaceRow(sourceIndex, start, "start");
                    if (laplaceFastStop(start)) {
                        printLaplaceRow(sourceIndex, start, "best");
                        continue;
                    }
                    if (laplaceAccepted(start)) {
                        best = start;
                        haveBest = true;
                        acceptedLow = start.lambda;
                        double probeLambda = start.lambda;
                        for (int step = 0; step < maxProbeSteps; ++step) {
                            probeLambda *= bracketFactor;
                            auto probe = evaluateLaplace(sourceNode, probeLambda);
                            printLaplaceRow(sourceIndex, probe, "probe_up");
                            if (laplaceFastStop(probe)) {
                                best = probe;
                                haveBest = true;
                                acceptedLow = probe.lambda;
                                break;
                            }
                            if (laplaceAccepted(probe)) {
                                best = probe;
                                haveBest = true;
                                acceptedLow = probe.lambda;
                            } else {
                                rejectedHigh = probe.lambda;
                                break;
                            }
                        }
                    } else {
                        rejectedHigh = start.lambda;
                        double probeLambda = start.lambda;
                        for (int step = 0; step < maxProbeSteps; ++step) {
                            probeLambda /= bracketFactor;
                            auto probe = evaluateLaplace(sourceNode, probeLambda);
                            printLaplaceRow(sourceIndex, probe, "probe_down");
                            if (laplaceAccepted(probe)) {
                                best = probe;
                                haveBest = true;
                                acceptedLow = probe.lambda;
                                break;
                            }
                        }
                    }

                    if (haveBest && rejectedHigh > acceptedLow && acceptedLow > 0.0) {
                        for (int step = 0;
                             step < maxRefineSteps && rejectedHigh / acceptedLow > minRefineFactor;
                             ++step) {
                            const double midLambda = std::sqrt(rejectedHigh * acceptedLow);
                            auto refine = evaluateLaplace(sourceNode, midLambda);
                            printLaplaceRow(sourceIndex, refine, "refine");
                            if (laplaceFastStop(refine)) {
                                best = refine;
                                acceptedLow = refine.lambda;
                                break;
                            }
                            if (laplaceAccepted(refine)) {
                                best = refine;
                                acceptedLow = refine.lambda;
                            } else {
                                rejectedHigh = refine.lambda;
                            }
                        }
                    }

                    if (haveBest) {
                        printLaplaceRow(sourceIndex, best, "best");
                    } else {
                        std::cout << "adaptive source[" << sourceIndex
                                  << "] found no accepted lambda\n";
                    }
                    continue;
                }

                for (double lambda : laplaceLambdas) {
                    const auto evaluation = evaluateLaplace(sourceNode, lambda);
                    printLaplaceRow(sourceIndex, evaluation, nullptr);
                }
            }
            return 0;
        }

        if (runEcl) {
            if (terminalFloodSweep) {
                const auto schedules = makeFloodSweepSchedules();
                std::cout << "ECL-MaxFlow flood sweep results:\n";
                std::cout << "  available: " << (vc::lasagna::eclMaxflowAvailable() ? "yes" : "no") << '\n';
                std::cout << "  schedules: " << schedules.size() << '\n';

                std::cout << std::left
                          << std::setw(8) << "kind"
                          << std::setw(8) << "start"
                          << std::setw(9) << "param"
                          << std::setw(8) << "depth";
                for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                    for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                        std::ostringstream col;
                        col << "s" << sourceIndex << "_t" << sinkIndex;
                        std::cout << std::setw(12) << col.str();
                    }
                }
                if (report.sinks.size() >= 2) {
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        std::ostringstream col;
                        col << "s" << sourceIndex << "_t0/t1";
                        std::cout << std::setw(12) << col.str();
                    }
                }
                std::cout << '\n';

                for (const auto& schedule : schedules) {
                    std::vector<std::vector<int64_t>> flows(
                        report.sources.size(),
                        std::vector<int64_t>(report.sinks.size(), 0));
                    std::vector<std::vector<std::string>> cells(
                        report.sources.size(),
                        std::vector<std::string>(report.sinks.size()));
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            const int32_t sourceNode = report.sources[sourceIndex].node;
                            const int32_t sinkNode = report.sinks[sinkIndex].node;
                            if (sourceNode == sinkNode) {
                                cells[sourceIndex][sinkIndex] = "skip";
                                continue;
                            }
                            try {
                                const auto flow = runEclWithTerminalFloodScheduleQuiet(
                                    graph,
                                    sourceNode,
                                    sinkNode,
                                    runs,
                                    schedule);
                                flows[sourceIndex][sinkIndex] = flow.maxFlow;
                                cells[sourceIndex][sinkIndex] = std::to_string(flow.maxFlow);
                            } catch (const std::exception& e) {
                                cells[sourceIndex][sinkIndex] = "error";
                            }
                        }
                    }

                    std::cout << std::left
                              << std::setw(8) << scheduleKindName(schedule.kind)
                              << std::setw(8) << schedule.startCapacity
                              << std::setw(9) << scheduleParamString(schedule)
                              << std::setw(8) << schedule.maxDepth;
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            std::cout << std::setw(12) << cells[sourceIndex][sinkIndex];
                        }
                    }
                    if (report.sinks.size() >= 2) {
                        for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                            std::cout << std::setw(12)
                                      << ratioString(flows[sourceIndex][0], flows[sourceIndex][1]);
                        }
                    }
                    std::cout << '\n';
                }
                return 0;
            }

            if (terminalFringeSweep) {
                const auto schedules = makeLinearSweepSchedules();
                std::cout << "ECL-MaxFlow linear flood vs hard fringe sweep results:\n";
                std::cout << "  available: " << (vc::lasagna::eclMaxflowAvailable() ? "yes" : "no") << '\n';
                std::cout << "  schedules: " << schedules.size() << '\n';

                std::cout << std::left
                          << std::setw(8) << "start"
                          << std::setw(8) << "step"
                          << std::setw(8) << "depth";
                for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                    for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                        std::ostringstream col;
                        col << "lin_s" << sourceIndex << "_t" << sinkIndex;
                        std::cout << std::setw(14) << col.str();
                    }
                }
                if (report.sinks.size() >= 2) {
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        std::ostringstream col;
                        col << "lin_s" << sourceIndex << "_t0/t1";
                        std::cout << std::setw(14) << col.str();
                    }
                }
                for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                    std::ostringstream col;
                    col << "fr_depth_s" << sourceIndex;
                    std::cout << std::setw(14) << col.str();
                }
                for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                    for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                        std::ostringstream col;
                        col << "fr_s" << sourceIndex << "_t" << sinkIndex;
                        std::cout << std::setw(14) << col.str();
                    }
                }
                for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                    for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                        std::ostringstream col;
                        col << "fr_sat_s" << sourceIndex << "_t" << sinkIndex;
                        std::cout << std::setw(14) << col.str();
                    }
                }
                if (report.sinks.size() >= 2) {
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        std::ostringstream col;
                        col << "fr_s" << sourceIndex << "_t0/t1";
                        std::cout << std::setw(14) << col.str();
                    }
                }
                std::cout << '\n';

                for (const auto& schedule : schedules) {
                    std::vector<std::vector<int64_t>> linearFlows(
                        report.sources.size(),
                        std::vector<int64_t>(report.sinks.size(), 0));
                    std::vector<std::vector<int64_t>> fringeFlows(
                        report.sources.size(),
                        std::vector<int64_t>(report.sinks.size(), 0));
                    std::vector<std::vector<std::string>> linearCells(
                        report.sources.size(),
                        std::vector<std::string>(report.sinks.size()));
                    std::vector<std::vector<std::string>> fringeCells(
                        report.sources.size(),
                        std::vector<std::string>(report.sinks.size()));
                    std::vector<std::vector<std::string>> fringeSaturationCells(
                        report.sources.size(),
                        std::vector<std::string>(report.sinks.size()));
                    std::vector<int> commonFringeDepth(
                        report.sources.size(),
                        schedule.maxDepth);

                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        const int32_t sourceNode = report.sources[sourceIndex].node;
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            const int32_t sinkNode = report.sinks[sinkIndex].node;
                            if (sourceNode == sinkNode) {
                                fringeSaturationCells[sourceIndex][sinkIndex] = "skip";
                                continue;
                            }
                            try {
                                const auto regions = terminalFringeRegions(
                                    graph,
                                    sourceNode,
                                    sinkNode,
                                    schedule.maxDepth);
                                if (regions.saturated) {
                                    fringeSaturationCells[sourceIndex][sinkIndex] =
                                        std::to_string(regions.saturationDepth);
                                    commonFringeDepth[sourceIndex] = std::min(
                                        commonFringeDepth[sourceIndex],
                                        std::max(0, regions.saturationDepth - 1));
                                } else {
                                    fringeSaturationCells[sourceIndex][sinkIndex] = "-";
                                }
                            } catch (const std::exception&) {
                                fringeSaturationCells[sourceIndex][sinkIndex] = "error";
                            }
                        }
                    }

                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            const int32_t sourceNode = report.sources[sourceIndex].node;
                            const int32_t sinkNode = report.sinks[sinkIndex].node;
                            if (sourceNode == sinkNode) {
                                linearCells[sourceIndex][sinkIndex] = "skip";
                                fringeCells[sourceIndex][sinkIndex] = "skip";
                                continue;
                            }
                            try {
                                const auto flow = runEclWithTerminalFloodScheduleQuiet(
                                    graph,
                                    sourceNode,
                                    sinkNode,
                                    runs,
                                    schedule);
                                linearFlows[sourceIndex][sinkIndex] = flow.maxFlow;
                                linearCells[sourceIndex][sinkIndex] = std::to_string(flow.maxFlow);
                            } catch (const std::exception&) {
                                linearCells[sourceIndex][sinkIndex] = "error";
                            }
                            try {
                                const auto flow = runEclWithTerminalFringeQuiet(
                                    graph,
                                    sourceNode,
                                    sinkNode,
                                    runs,
                                    commonFringeDepth[sourceIndex]);
                                fringeFlows[sourceIndex][sinkIndex] = flow.maxFlow;
                                fringeCells[sourceIndex][sinkIndex] = std::to_string(flow.maxFlow);
                            } catch (const std::exception&) {
                                fringeCells[sourceIndex][sinkIndex] = "error";
                            }
                        }
                    }

                    std::cout << std::left
                              << std::setw(8) << schedule.startCapacity
                              << std::setw(8) << schedule.step
                              << std::setw(8) << schedule.maxDepth;
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            std::cout << std::setw(14) << linearCells[sourceIndex][sinkIndex];
                        }
                    }
                    if (report.sinks.size() >= 2) {
                        for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                            std::cout << std::setw(14)
                                      << ratioString(linearFlows[sourceIndex][0], linearFlows[sourceIndex][1]);
                        }
                    }
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        std::cout << std::setw(14) << commonFringeDepth[sourceIndex];
                    }
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            std::cout << std::setw(14) << fringeCells[sourceIndex][sinkIndex];
                        }
                    }
                    for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                        for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                            std::cout << std::setw(14) << fringeSaturationCells[sourceIndex][sinkIndex];
                        }
                    }
                    if (report.sinks.size() >= 2) {
                        for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                            std::cout << std::setw(14)
                                      << ratioString(fringeFlows[sourceIndex][0], fringeFlows[sourceIndex][1]);
                        }
                    }
                    std::cout << '\n';
                }
                return 0;
            }

            std::vector<FlowRunResult> flowResults;
            flowResults.reserve(report.sources.size() * report.sinks.size());
            for (size_t sourceIndex = 0; sourceIndex < report.sources.size(); ++sourceIndex) {
                for (size_t sinkIndex = 0; sinkIndex < report.sinks.size(); ++sinkIndex) {
                    FlowRunResult result;
                    result.sourceIndex = sourceIndex;
                    result.sinkIndex = sinkIndex;
                    result.sourceNode = report.sources[sourceIndex].node;
                    result.sinkNode = report.sinks[sinkIndex].node;
                    if (result.sourceNode == result.sinkNode) {
                        result.skipped = true;
                        result.error = "source and sink resolved to the same graph node";
                        flowResults.push_back(std::move(result));
                        continue;
                    }
                    try {
                        const auto flowStart = std::chrono::steady_clock::now();
                        if (terminalFloodDepth > 0) {
                            result.flow = runEclWithTerminalFlood(
                                graph,
                                result.sourceNode,
                                result.sinkNode,
                                runs,
                                terminalFloodDepth,
                                terminalFloodCapacity,
                                terminalFloodDecay,
                                result);
                        } else if (terminalRegionIterations > 0) {
                            result.usedTerminalRegions = true;
                            result.flow = runEclWithTerminalExpansion(
                                graph,
                                result.sourceNode,
                                result.sinkNode,
                                runs,
                                terminalRegionIterations,
                                result);
                        } else {
                            result.flow = vc::lasagna::runEclMaxflow(
                                graph,
                                result.sourceNode,
                                result.sinkNode,
                                runs);
                            result.sourceExpansion =
                                vc::lasagna::expandTerminalRegionAcrossMinCutBoundaries(
                                    graph,
                                    result.sourceNode,
                                    result.sinkNode,
                                    result.flow.maxFlow,
                                    vc::lasagna::EclTerminalSide::Source,
                                    terminalRegionIterations);
                            result.sinkExpansion =
                                vc::lasagna::expandTerminalRegionAcrossMinCutBoundaries(
                                    graph,
                                    result.sinkNode,
                                    result.sourceNode,
                                    result.flow.maxFlow,
                                    vc::lasagna::EclTerminalSide::Sink,
                                    terminalRegionIterations);
                        }
                        result.wallSeconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - flowStart).count();
                    } catch (const std::exception& e) {
                        result.error = e.what();
                    }
                    flowResults.push_back(std::move(result));
                }
            }

            std::cout << "ECL-MaxFlow results:\n";
            std::cout << "  available: " << (vc::lasagna::eclMaxflowAvailable() ? "yes" : "no") << '\n';
            std::cout << "  pairs: " << flowResults.size() << '\n';
            for (const auto& result : flowResults) {
                std::cout << "  source[" << result.sourceIndex << "] node=" << result.sourceNode
                          << " -> sink[" << result.sinkIndex << "] node=" << result.sinkNode;
                if (result.skipped) {
                    std::cout << " skipped: " << result.error << '\n';
                } else if (!result.error.empty()) {
                    std::cout << " error: " << result.error << '\n';
                } else {
                    std::cout << " max_flow=" << result.flow.maxFlow
                              << " runs=" << result.flow.runs
                              << " median_runtime_seconds=" << result.flow.medianRuntimeSeconds
                              << " throughput_gigaedges_per_second=" << result.flow.throughputGigaEdgesPerSecond
                              << " wall_time_seconds=" << result.wallSeconds;
                    if (result.usedTerminalFlood) {
                        std::cout << " terminal_flood_mode=yes"
                                  << " terminal_flood_depth=" << result.terminalFloodDepth
                                  << " terminal_flood_capacity=" << result.terminalFloodCapacity
                                  << " terminal_flood_decay=" << result.terminalFloodDecay
                                  << " terminal_seed_capacity=infinite"
                                  << " source_flood_nodes=" << result.sourceFloodNodes
                                  << " source_flood_shell_capacity_sum="
                                  << result.sourceFloodTerminalCapacity
                                  << " sink_flood_nodes=" << result.sinkFloodNodes
                                  << " sink_flood_shell_capacity_sum="
                                  << result.sinkFloodTerminalCapacity;
                    }
                    if (result.usedTerminalRegions) {
                        std::cout << " terminal_region_mode=yes"
                                  << " terminal_edge_capacity=" << result.terminalEdgeCapacity
                                  << " terminal_region_iterations=" << result.terminalRegionIterations
                                  << " terminal_expansion_limit_reached="
                                  << (result.terminalExpansionLimitReached ? "yes" : "no");
                    }
                    if (result.flow.minCut.valid) {
                        std::cout << " min_cut_capacity=" << result.flow.minCut.cutCapacity
                                  << " min_cut_directed_edges=" << result.flow.minCut.cutDirectedEdges
                                  << " min_cut_source_side_nodes=" << result.flow.minCut.sourceReachableNodes
                                  << " min_cut_sink_side_nodes=" << result.flow.minCut.sinkSideNodes;
                    }
                    if (result.sourceExpansion.valid) {
                        std::cout << " source_terminal_region_nodes=" << result.sourceExpansion.regionNodes
                                  << " source_terminal_absorbed_nodes=" << result.sourceExpansion.absorbedNodes
                                  << " source_terminal_iterations=" << result.sourceExpansion.iterations
                                  << " source_terminal_boundary_capacity="
                                  << result.sourceExpansion.finalBoundaryCapacity
                                  << " source_terminal_boundary_is_min_cut="
                                  << (result.sourceExpansion.finalBoundaryIsMinCut ? "yes" : "no");
                    }
                    if (result.sinkExpansion.valid) {
                        std::cout << " sink_terminal_region_nodes=" << result.sinkExpansion.regionNodes
                                  << " sink_terminal_absorbed_nodes=" << result.sinkExpansion.absorbedNodes
                                  << " sink_terminal_iterations=" << result.sinkExpansion.iterations
                                  << " sink_terminal_boundary_capacity="
                                  << result.sinkExpansion.finalBoundaryCapacity
                                  << " sink_terminal_boundary_is_min_cut="
                                  << (result.sinkExpansion.finalBoundaryIsMinCut ? "yes" : "no");
                    }
                    std::cout << '\n';
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
