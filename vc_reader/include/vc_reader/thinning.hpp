#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace vcr::thinning {
enum class Backend { Cpu, Cuda };
enum class Algorithm { ZhangSuen, GuoHall, ZhangSuenGuoHall };
Algorithm algorithmFromName(std::string_view name);
struct Point {
    int x = 0, y = 0;
    bool operator==(const Point&) const = default;
};
using Traces = std::vector<std::vector<Point>>;
struct Stats {
    double distanceTransformSeconds = 0.0;
    double seedDetectionSeconds = 0.0;
    double tracePathsSeconds = 0.0;
    uint64_t seedCount = 0;
    uint64_t traceCount = 0;        // raw traces produced before pruning
    uint64_t traceSteps = 0;
    uint64_t candidateEvaluations = 0;
    uint64_t tracesPruned = 0;      // traces dropped by pruning pass
    uint64_t tracesKept = 0;        // traces retained after pruning

    void accumulate(const Stats& other) {
        distanceTransformSeconds += other.distanceTransformSeconds;
        seedDetectionSeconds += other.seedDetectionSeconds;
        tracePathsSeconds += other.tracePathsSeconds;
        seedCount += other.seedCount;
        traceCount += other.traceCount;
        traceSteps += other.traceSteps;
        candidateEvaluations += other.candidateEvaluations;
        tracesPruned += other.tracesPruned;
        tracesKept += other.tracesKept;
    }
};


// Polyline lengths retain double accumulation to preserve pruning decisions.
struct PruneParams { double minLengthPx = 0; };
// One scratch object per calling thread. Buffers are reused across slices.
struct Scratch {
    std::vector<std::uint8_t> distance, visited;
    std::vector<Point> seedPoints, firstPath, secondPath;
};

// Ordered distance-ridge tracing of a uint8 YX mask (any nonzero is foreground).
// Input may have row padding; stride=0 means cols. Output, if supplied, is
// contiguous rows*cols bytes. Input/output may alias. Traces use XY coordinates.
// Output records visit indices saturated at 255; seeds are 1. With pruning and
// traces requested, kept traces are rasterized as 255. Pruning requires traces.
// CUDA traces independent connected components on device, preserving seed order.
// Empty grids clear traces. Invalid sizes/pointers/parameters throw.
void trace(const std::uint8_t* input, std::size_t rows, std::size_t cols,
           std::uint8_t* output, Traces* traces, Scratch& scratch,
           Stats* stats = nullptr, PruneParams prune = {},
           Backend backend = Backend::Cpu, std::size_t stride = 0);
void prune(Traces& traces, PruneParams params, Stats* stats = nullptr);

// Independent YX binary skeletons in contiguous (slices,rows,cols) storage.
// Any nonzero input is foreground; output is 0/255. Aliasing is supported.
// preserveBorder=true leaves each slice's outer pixels unchanged; false treats
// pixels outside each slice as background. Both subpasses run to convergence.
void skeleton(const std::uint8_t* input, std::uint8_t* output,
              std::size_t rows, std::size_t cols, std::size_t slices = 1,
              Algorithm algorithm = Algorithm::ZhangSuen, bool preserveBorder = true);
#ifdef VC_READER_CUDA
namespace cuda {
// Full distance/seed/tracing pass on a resident YX mask. Output visit indices
// match trace(); no intermediate image is downloaded. No polyline pruning.
void traceDevice(const std::uint8_t* input, std::uint8_t* output,
                 std::size_t rows, std::size_t cols);
// Host input/output, including transfers. Uses the calling thread's device/stream.
void skeleton(const std::uint8_t* input, std::uint8_t* output,
              std::size_t rows, std::size_t cols, std::size_t slices = 1,
              Algorithm algorithm = Algorithm::ZhangSuen, bool preserveBorder = true);
// Device input/output on the calling thread's device. Completes on vc_reader's
// stream before returning. External producers must finish writes before calling.
void skeletonDevice(const std::uint8_t* input, std::uint8_t* output,
                    std::size_t rows, std::size_t cols, std::size_t slices = 1,
                    Algorithm algorithm = Algorithm::ZhangSuen, bool preserveBorder = true);
}
#endif
}
