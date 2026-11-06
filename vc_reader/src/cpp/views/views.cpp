#include "common.hpp"
#include <algorithm>

namespace vcr::views {
using namespace detail;

Orientation orientation(int id)
{
    require(id >= 0 && id < ORIENTATION_COUNT, "orientation id must be in [0, 48)");
    Orientation o;
    for (int k = 0; k < 3; ++k) {
        o.perm[k] = permutations[id / 8][k];
        o.flips[k] = (id >> k) & 1;
    }
    return o;
}

int orientationId(const Orientation& o)
{
    for (int p = 0; p < 6; ++p)
        if (o.perm[0] == permutations[p][0] && o.perm[1] == permutations[p][1] && o.perm[2] == permutations[p][2])
            return p * 8 + int(o.flips[0]) + 2 * int(o.flips[1]) + 4 * int(o.flips[2]);
    throw std::invalid_argument("perm must be a permutation of (0, 1, 2)");
}

void drawOrientations(const std::uint64_t* seeds, std::size_t count, std::int32_t* ids)
{
    require(count == 0 || (seeds && ids), "seed and id arrays are required");
    for (std::size_t i = 0; i < count; ++i) ids[i] = std::int32_t(Stream(seeds[i]).below(ORIENTATION_COUNT));
}

void tokenSupport(const std::uint8_t* valid, std::size_t batch, const Index3& shape, const Index3& patch,
                  std::uint8_t* support)
{
    const auto grid = cells(shape, patch);
    require(batch == 0 || (valid && support), "valid and support arrays are required");
    const auto voxels = product(shape);
    std::size_t out = 0;
    for (std::size_t b = 0; b < batch; ++b)
        for (std::int64_t z = 0; z < grid[0]; ++z)
            for (std::int64_t y = 0; y < grid[1]; ++y)
                for (std::int64_t x = 0; x < grid[2]; ++x)
                    support[out++] = cellSupported(valid + b * voxels, shape.data(), patch.data(), z, y, x);
}

void correspondence(const Supports& support, const Pairs& pairs, std::int32_t* teacherIndex,
                    std::uint8_t* studentValid, std::int32_t* lattice)
{
    const auto g = pairGeometry(support, pairs);
    const auto plans = pairPlans(support, pairs, g);
    if (!pairs.count) return;
    require(support.data && teacherIndex && studentValid, "support and output arrays are required");
    for (std::size_t p = 0; p < pairs.count; ++p) {
        const auto offset = std::int64_t(p) * g.studentTokens;
        for (std::int64_t i = 0; i < g.studentTokens; ++i)
            correspond(g, plans[p], support.data, i, teacherIndex + offset, studentValid + offset,
                       lattice ? lattice + 3 * offset : nullptr);
    }
}

namespace {
template <class T>
void maskCells(const CellImages& c, const Index3& grid, const std::uint8_t* masks, T fill)
{
    const auto* in = static_cast<const T*>(c.images);
    auto* out = static_cast<T*>(c.out);
    const bool inPlace = c.images == c.out;
    const auto voxels = std::size_t(product(c.shape)), tokens = std::size_t(product(grid));
    const auto width = std::size_t(c.shape[2]), patchX = std::size_t(c.patch[2]);
    for (std::size_t b = 0; b < c.batch; ++b)
        for (std::size_t ch = 0; ch < c.channels; ++ch) {
            const auto image = (b * c.channels + ch) * voxels;
            for (std::int64_t z = 0; z < c.shape[0]; ++z)
                for (std::int64_t y = 0; y < c.shape[1]; ++y) {
                    const auto row = image + (std::size_t(z) * c.shape[1] + y) * width;
                    const auto* cellRow = masks + b * tokens + (std::size_t(z / c.patch[0]) * grid[1] + y / c.patch[1]) * grid[2];
                    for (std::int64_t x = 0; x < grid[2]; ++x) {
                        const auto begin = row + std::size_t(x) * patchX;
                        if (cellRow[x]) std::fill(out + begin, out + begin + patchX, fill);
                        else if (!inPlace) std::copy(in + begin, in + begin + patchX, out + begin);
                    }
                }
        }
}
}  // namespace

void maskPatchCells(const CellImages& images, const std::uint8_t* masks, float fill)
{
    const auto grid = validateCells(images);
    if (!images.batch || !images.channels) return;
    require(masks != nullptr, "masks are required");
    const auto bits = fillBits(images.type, fill);
    if (images.type == Real::Float32) maskCells<std::uint32_t>(images, grid, masks, bits);
    else maskCells<std::uint16_t>(images, grid, masks, std::uint16_t(bits));
}
}  // namespace vcr::views
