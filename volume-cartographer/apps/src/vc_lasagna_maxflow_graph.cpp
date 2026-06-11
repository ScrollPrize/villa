#include "vc/lasagna/MaxflowGraph.hpp"
#include "vc/lasagna/EclMaxflow.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void printUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " <manifest.lasagna.json> "
              << "--src x,y,z [--src x,y,z ...] --sink x,y,z [--sink x,y,z ...] "
              << "[--margin-base-voxels 1000] [--threshold 110] "
              << "[--run-ecl] [--runs N]\n";
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
    double wallSeconds = 0.0;
};

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

        if (runEcl) {
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
                        result.flow = vc::lasagna::runEclMaxflow(
                            graph,
                            result.sourceNode,
                            result.sinkNode,
                            runs);
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
                              << " wall_time_seconds=" << result.wallSeconds << '\n';
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
