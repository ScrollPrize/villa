#pragma once
// Patch-lattice views for self-supervised training: oriented crops of a token
// lattice, teacher/student placement, token correspondence between two oriented
// crops, token masks, and masking of the voxels of masked tokens.
//
// Arrays are C order, ZYX. A token is one cell of a patch lattice (patch size P
// per axis); origins and shapes of crops are in token units.
//
// Orientations: id = perm_index * 8 + flip_bits, 48 in all. perm_index indexes
// the lexicographic permutations (0,1,2) (0,2,1) (1,0,2) (1,2,0) (2,0,1) (2,1,0).
// An oriented view of crop C is V = flip(transpose(C, perm)), as numpy.transpose;
// flip bit k (value 1 << k) reverses output axis k. Output axis k has extent
// n[perm[k]], and V[i] = C[u] with u[perm[k]] = flip_k ? n[perm[k]] - 1 - i_k : i_k.
//
// Random streams: every seed starts an independent SplitMix64 stream
// (splitmix.hpp, shared with transforms): output k = 1, 2, ... is
// mix(seed + k * 0x9e3779b97f4a7c15). Real draws use the top 53 bits of one
// output, (bits >> 11) * 2^-53, uniform in [0, 1); integers in [0, n) take
// outputs until one is at least 2^64 mod n and return it mod n (unbiased).
// Draws are consumed in the order each function documents.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vcr::views {
using Index3 = std::array<std::int64_t, 3>;

inline constexpr int ORIENTATION_COUNT = 48;
struct Orientation {
    std::array<int, 3> perm{0, 1, 2};
    std::array<bool, 3> flips{false, false, false};
};
// Throws std::invalid_argument outside [0, 48) or for a non-permutation.
Orientation orientation(int id);
int orientationId(const Orientation& orientation);
// ids[i] is uniform over [0, 48): the first integer draw of seeds[i]'s stream.
void drawOrientations(const std::uint64_t* seeds, std::size_t count, std::int32_t* ids);

// support[b, t] = 1 iff every voxel of token cell t of valid[b] is nonzero.
// valid is (batch, shape); support is (batch, shape / patch). Every axis of
// shape must be a positive multiple of patch.
void tokenSupport(const std::uint8_t* valid, std::size_t batch, const Index3& shape,
                  const Index3& patch, std::uint8_t* support);

// Teacher and student crop placement on a token support grid (nonzero = supported).
struct PlanOptions {
    std::size_t sets = 1;
    double minOverlapFraction = 0.0;
    std::int64_t minOverlapCount = 0;
    std::int64_t maxStudentAttempts = 1;
    std::int64_t maxTeacherAttempts = 1;
    std::uint64_t seed = 0;
};
struct ViewSets {
    bool ok = false;                          // every set was placed
    std::size_t sets = 0, students = 0;
    std::vector<std::uint8_t> setOk;          // (sets)
    std::vector<std::int32_t> teacherOrigins; // (sets, 3)
    std::vector<std::int32_t> studentOrigins; // (sets, students, 3)
    std::vector<std::int64_t> studentOverlap; // (sets, students) supported overlap tokens
    std::vector<std::int32_t> studentAttempts;// (sets, students) proposals used by the accepted placement
    std::vector<std::int32_t> teacherAttempts;// (sets)
    std::vector<std::int64_t> rejections;     // (sets) rejected student proposals, all teacher attempts
};
// Sets are placed in order from one stream of options.seed. A teacher attempt
// draws its origin uniformly per axis (z, y, x) over [0, T - n]; then each
// student in order proposes origins the same way, up to maxStudentAttempts
// times. A proposal is accepted when O, the supported tokens in its intersection
// with the teacher box, satisfies O >= minOverlapCount and
// O >= minOverlapFraction * prod(student shape). A student that exhausts its
// attempts starts a new teacher attempt. A set whose teacher attempts run out
// has setOk = 0 and zero origins; ok is then false. Contained and coincident
// crops are allowed. Throws std::invalid_argument for a shape that is not
// positive or does not fit the grid, and for invalid options.
ViewSets planViewSets(const std::uint8_t* support, const Index3& tokens, const Index3& teacher,
                      const std::vector<Index3>& students, const PlanOptions& options);

// Token correspondence of `count` student/teacher crop pairs sharing one
// student and one teacher shape (unoriented, token units).
struct Supports {
    const std::uint8_t* data = nullptr; // (grids, shape) token support, nonzero = supported
    std::size_t grids = 0;
    Index3 shape{};
};
// Host arrays in every backend; validated before any work.
struct Pairs {
    std::size_t count = 0;
    Index3 studentShape{}, teacherShape{};
    const std::int64_t* supportIndex = nullptr;       // (count) grid of each pair
    const std::int64_t* studentOrigins = nullptr;     // (count, 3)
    const std::int64_t* studentOrientations = nullptr;// (count)
    const std::int64_t* teacherOrigins = nullptr;     // (count, 3)
    const std::int64_t* teacherOrientations = nullptr;// (count)
};
// For pair p and oriented student token i (C order over the oriented student
// shape), g = student origin + its unoriented crop coordinates. studentValid =
// 1 iff g is inside the grid and supported. teacherIndex is the C-order index of
// g in the oriented teacher crop when g lies inside the teacher box and the
// student token is valid, else -1. lattice (optional) receives g, ZYX.
// Outputs: teacherIndex int32 (count, prod(student)), studentValid
// (count, prod(student)), lattice int32 (count, prod(student), 3). Integer
// arithmetic only; CPU and CUDA results are identical. Throws
// std::invalid_argument for invalid shapes, indices or orientations, and when
// teacher indices or lattice coordinates do not fit int32.
void correspondence(const Supports& support, const Pairs& pairs, std::int32_t* teacherIndex,
                    std::uint8_t* studentValid, std::int32_t* lattice = nullptr);

// Token masks, one independent stream per seed. Generation is sequential and
// produces small per-view grids, so it runs on the host only; upload the result.
enum class MaskMode { Random, Multiblock };
struct MaskOptions {
    MaskMode mode = MaskMode::Multiblock;
    double probability = 1.0;                     // chance a mask is applied
    double fraction = 0.5;                        // masked fraction when applied
    double minBlockTokens = 4;
    std::optional<double> maxBlockTokens;         // default: all tokens
    double minAspect = 0.3;
    std::optional<double> maxAspect;              // default: 1 / minAspect
    std::int64_t attempts = 10;
    bool cyclicShift = true;
};
// masks (count, shape) receives 0/1; applied and counts (count) may be null.
// Per seed: applied = uniform < probability (an unapplied mask is empty). The
// target count is n = round_half_up(fraction * N) of N tokens. Random mode masks
// n distinct tokens by partial Fisher-Yates. Multiblock mode (DINOv3's masking
// generator with 3D cuboids) adds blocks of at most min(n - count, maxBlockTokens)
// new tokens until n are masked or a block fails; each block makes `attempts`
// tries of: volume uniform between minBlockTokens and the cap, depth and planar
// aspect log-uniform in [minAspect, maxAspect], d = round(cbrt(volume / depth) *
// depth), h = round(sqrt(max(volume / d, 1) * planar)), w = round(sqrt(max(volume
// / d, 1) / planar)) (each at least 1, ties to even), and a uniform corner (z, y,
// x) when the block fits. Any shortfall is masked uniformly among unmasked
// tokens; cyclicShift then rolls the mask by a uniform shift per axis. Applied
// masks always have exactly n tokens. Results depend only on each seed.
void patchMasks(const Index3& shape, const std::uint64_t* seeds, std::size_t count,
                const MaskOptions& options, std::uint8_t* masks, bool* applied = nullptr,
                std::int64_t* counts = nullptr);

// Copies images (batch, channels, shape) to out with every voxel of masked
// token cells set to fill, in all channels. masks are (batch, shape / patch),
// nonzero = masked. out may alias images. Float16 and BFloat16 receive fill
// rounded to nearest, ties to even.
enum class Real { Float32, Float16, BFloat16 };
struct CellImages {
    const void* images = nullptr;
    void* out = nullptr;
    Real type = Real::Float32;
    std::size_t batch = 0, channels = 0;
    Index3 shape{}, patch{};
};
void maskPatchCells(const CellImages& images, const std::uint8_t* masks, float fill);

#ifdef VC_READER_CUDA
namespace cuda {
// Device pointers on the calling thread's CUDA device. Work is enqueued on the
// calling thread's vc_reader stream (device::StreamScope) and is not awaited:
// synchronize or order consumers on that stream. Pairs stay host arrays.
void tokenSupport(const std::uint8_t* valid, std::size_t batch, const Index3& shape,
                  const Index3& patch, std::uint8_t* support);
void correspondence(const Supports& support, const Pairs& pairs, std::int32_t* teacherIndex,
                    std::uint8_t* studentValid, std::int32_t* lattice = nullptr);
void maskPatchCells(const CellImages& images, const std::uint8_t* masks, float fill);
}  // namespace cuda
#endif
}  // namespace vcr::views
