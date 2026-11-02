#pragma once
// Sampling volumes on the GPU through vc_reader's CUDA reader, for VC3D's viewers and the
// command line tools. Every call here also works in a build without CUDA, where no reader
// exists. See docs/native_sampling.md.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vcr {
class Array;
struct ElementType;
namespace reader_cuda { class Reader; class Pool; }
namespace zarr { struct OpenedArray; }
}

namespace vc::cuda {

// A reader keeping up to `cacheBytes` of `array` on the device; null when the build or
// machine has no CUDA device, or `array` is not 3-D uint8.
std::shared_ptr<vcr::reader_cuda::Reader> makeReader(vcr::Array& array, const vcr::ElementType& type,
                                                     std::size_t cacheBytes);
// One device brick cache of `cacheBytes` for several readers (e.g. every level of the
// volumes a viewer opens); null when the build or machine has no CUDA device.
std::shared_ptr<vcr::reader_cuda::Pool> makePool(std::size_t cacheBytes);
// A reader of `array` whose bricks live in `pool`, sharing its budget and LRU order; null
// as makeReader, or when pool is null. Arrays whose bricks are larger than the pool's
// slots (odd chunk edges over 32) get a cache of their own of the pool's budget.
std::shared_ptr<vcr::reader_cuda::Reader> makeReader(vcr::Array& array, const vcr::ElementType& type,
                                                     const std::shared_ptr<vcr::reader_cuda::Pool>& pool);

// Trilinear / nearest values at zyx index points, on `reader` when given. A GPU failure
// frees the reader's bricks, logs once per process and samples on the CPU. Null samples on
// the CPU, as Array::sample_linear and Array::sample_nearest do.
void sampleLinear(vcr::Array& array, const vcr::ElementType& type, vcr::reader_cuda::Reader* reader,
                  const double* zyx, std::size_t n, float* out);
void sampleNearest(vcr::Array& array, vcr::reader_cuda::Reader* reader, const double* zyx, std::size_t n,
                   std::uint8_t* out);
// The voxels at zyx integer indices (`reader` must be set; throws on a GPU failure so the
// caller can use its own CPU read).
void readVoxels(vcr::reader_cuda::Reader& reader, const std::int64_t* zyx, std::size_t n, std::uint8_t* out);

// Tools: --cuda turns GPU sampling on for this process. enableForTool throws when there
// is no CUDA device (the user asked for it explicitly).
constexpr std::size_t kDefaultToolCacheGb = 4;
void enableForTool(std::size_t cacheBytes);
bool enabledForTool();
// Removes --cuda and --cuda-cache-gb N from argv, enabling GPU sampling when present.
// Prints the problem and exits on a malformed option or a missing device.
void applyToolOptions(int& argc, char** argv);
// The option lines for a tool's --help.
const char* toolOptionsHelp();

// The tool's reader of `level`, made on first use and kept until the process exits; every
// level's reader shares one device cache of the --cuda-cache-gb budget; null when --cuda is off or the level is not 3-D uint8 (noted in the log).
vcr::reader_cuda::Reader* toolReader(const vcr::zarr::OpenedArray& level);
// The reader toolReader made for `array`, or null.
vcr::reader_cuda::Reader* toolReaderOf(const vcr::Array& array);

}  // namespace vc::cuda
