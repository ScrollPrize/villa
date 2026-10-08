#include "vc/fiber_tracer/FiberSkeleton.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace vc::fiber_tracer
{
namespace
{

// Raster order of a 3x3x3 neighborhood:
// index = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1), so the center is 13.
constexpr std::size_t kCenter = 13;

// Border directions of the six thinning subiterations, in the order used by
// Lee's algorithm as implemented in ITK and scikit-image: -Z, +Z, +X, -X, -Y,
// +Y. A voxel is a border point for a direction when the neighbor in that
// direction is background.
constexpr std::array<std::size_t, 6> kBorderNeighbor{4, 22, 14, 12, 10, 16};

constexpr std::array<int, 3> neighborOffset(std::size_t index) noexcept
{
    return {static_cast<int>(index / 9) - 1,
            static_cast<int>((index / 3) % 3) - 1,
            static_cast<int>(index % 3) - 1};
}

// Adjacency inside one 3x3x3 neighborhood, as bit masks over the 27 slots.
struct NeighborhoodTables {
    std::array<std::uint32_t, 27> adjacent26{};
    std::array<std::uint32_t, 27> adjacent6{};
    std::uint32_t eighteen = 0;   // the 18-neighborhood, center excluded
    std::uint32_t face = 0;       // the six 6-neighbors
};

NeighborhoodTables buildNeighborhoodTables() noexcept
{
    NeighborhoodTables tables;
    for (std::size_t first = 0; first < 27; ++first) {
        if (first == kCenter)
            continue;
        const auto a = neighborOffset(first);
        const int radius = std::abs(a[0]) + std::abs(a[1]) + std::abs(a[2]);
        if (radius <= 2)
            tables.eighteen |= 1U << first;
        if (radius == 1)
            tables.face |= 1U << first;
        for (std::size_t second = 0; second < 27; ++second) {
            if (second == kCenter || second == first)
                continue;
            const auto b = neighborOffset(second);
            const int dz = std::abs(a[0] - b[0]);
            const int dy = std::abs(a[1] - b[1]);
            const int dx = std::abs(a[2] - b[2]);
            if (dz <= 1 && dy <= 1 && dx <= 1)
                tables.adjacent26[first] |= 1U << second;
            if (dz + dy + dx == 1)
                tables.adjacent6[first] |= 1U << second;
        }
    }
    return tables;
}

const NeighborhoodTables& neighborhoodTables() noexcept
{
    static const NeighborhoodTables tables = buildNeighborhoodTables();
    return tables;
}

// Closure of `seed` inside `domain` under the given adjacency table.
std::uint32_t maskClosure(
    std::uint32_t seed, std::uint32_t domain,
    const std::array<std::uint32_t, 27>& adjacency) noexcept
{
    std::uint32_t reached = seed;
    std::uint32_t frontier = seed;
    while (frontier != 0) {
        std::uint32_t next = 0;
        std::uint32_t pending = frontier;
        while (pending != 0) {
            const auto slot = static_cast<std::size_t>(
                std::countr_zero(pending));
            pending &= pending - 1;
            next |= adjacency[slot] & domain & ~reached;
        }
        reached |= next;
        frontier = next;
    }
    return reached;
}

}  // namespace

bool fiberSkeletonSimplePoint(
    const std::array<std::uint8_t, 27>& neighborhood) noexcept
{
    const auto& tables = neighborhoodTables();

    std::uint32_t foreground = 0;
    for (std::size_t slot = 0; slot < 27; ++slot) {
        if (slot != kCenter && neighborhood[slot] != 0)
            foreground |= 1U << slot;
    }
    // An isolated voxel is not simple: removing it deletes a component.
    if (foreground == 0)
        return false;

    // The foreground of the 26-neighborhood must form one 26-connected
    // component, otherwise removal disconnects it locally.
    const std::uint32_t firstForeground =
        1U << static_cast<std::size_t>(std::countr_zero(foreground));
    if (maskClosure(firstForeground, foreground, tables.adjacent26) !=
        foreground) {
        return false;
    }

    // Exactly one 6-connected background component of the 18-neighborhood may
    // be 6-adjacent to the center, otherwise removal opens a cavity or tunnel.
    const std::uint32_t background = tables.eighteen & ~foreground;
    std::uint32_t remaining = background;
    int adjacentComponents = 0;
    while (remaining != 0) {
        const std::uint32_t seed =
            1U << static_cast<std::size_t>(std::countr_zero(remaining));
        const std::uint32_t component =
            maskClosure(seed, background, tables.adjacent6);
        remaining &= ~component;
        if ((component & tables.face) != 0) {
            ++adjacentComponents;
            if (adjacentComponents > 1)
                return false;
        }
    }
    return adjacentComponents == 1;
}

void thinFiberSkeletonMask(
    const std::array<std::size_t, 3>& shapeZYX, std::vector<std::uint8_t>& mask)
{
    const std::size_t sizeZ = shapeZYX[0];
    const std::size_t sizeY = shapeZYX[1];
    const std::size_t sizeX = shapeZYX[2];
    const std::size_t total = sizeZ * sizeY * sizeX;
    if (mask.size() != total)
        throw std::invalid_argument("thinning mask size does not match shape");
    if (total == 0)
        return;
    for (auto& value : mask) {
        if (value > 1)
            throw std::invalid_argument("thinning mask must be zero or one");
    }

    const std::ptrdiff_t strideZ = static_cast<std::ptrdiff_t>(sizeY * sizeX);
    const std::ptrdiff_t strideY = static_cast<std::ptrdiff_t>(sizeX);
    std::array<std::ptrdiff_t, 27> offsets{};
    for (std::size_t slot = 0; slot < 27; ++slot) {
        const auto delta = neighborOffset(slot);
        offsets[slot] = delta[0] * strideZ + delta[1] * strideY + delta[2];
    }

    std::array<std::uint8_t, 27> neighborhood{};
    const auto gather = [&](std::size_t flat) noexcept {
        const std::size_t x = flat % sizeX;
        const std::size_t y = (flat / sizeX) % sizeY;
        const std::size_t z = flat / (sizeX * sizeY);
        if (z > 0 && z + 1 < sizeZ && y > 0 && y + 1 < sizeY && x > 0 &&
            x + 1 < sizeX) {
            const auto* base = mask.data() + flat;
            for (std::size_t slot = 0; slot < 27; ++slot)
                neighborhood[slot] = base[offsets[slot]];
            return;
        }
        for (std::size_t slot = 0; slot < 27; ++slot) {
            const auto delta = neighborOffset(slot);
            const std::ptrdiff_t nz = static_cast<std::ptrdiff_t>(z) + delta[0];
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(y) + delta[1];
            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(x) + delta[2];
            const bool inside = nz >= 0 && ny >= 0 && nx >= 0 &&
                nz < static_cast<std::ptrdiff_t>(sizeZ) &&
                ny < static_cast<std::ptrdiff_t>(sizeY) &&
                nx < static_cast<std::ptrdiff_t>(sizeX);
            neighborhood[slot] = inside
                ? mask[static_cast<std::size_t>(
                      (nz * strideZ) + (ny * strideY) + nx)]
                : static_cast<std::uint8_t>(0);
        }
    };

    std::vector<std::size_t> foreground;
    foreground.reserve(total / 8 + 1);
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (mask[flat] != 0)
            foreground.push_back(flat);
    }

    // A voxel's classification can only change when its own neighborhood
    // changed, so after the first pass only the neighbors of removed voxels
    // need to be revisited. `foreground` stays sorted in raster order, which
    // keeps the candidate order and therefore the result deterministic, and
    // lets the revisit flags be indexed by position instead of by voxel.
    std::vector<std::uint8_t> revisit(foreground.size(), 1);
    std::vector<std::uint8_t> revisitNext(foreground.size(), 0);
    std::vector<std::size_t> candidates;
    const auto markRevisit = [&](std::size_t flat) {
        const auto found = std::lower_bound(
            foreground.begin(), foreground.end(), flat);
        if (found != foreground.end() && *found == flat) {
            revisitNext[static_cast<std::size_t>(
                found - foreground.begin())] = 1;
        }
    };

    bool passRemoved = true;
    while (passRemoved) {
        passRemoved = false;
        std::fill(revisitNext.begin(), revisitNext.end(), 0);
        for (const std::size_t border : kBorderNeighbor) {
            candidates.clear();
            for (std::size_t position = 0; position < foreground.size();
                 ++position) {
                const std::size_t flat = foreground[position];
                if (mask[flat] == 0 || revisit[position] == 0)
                    continue;
                gather(flat);
                if (neighborhood[border] != 0)
                    continue;
                int neighbors = 0;
                for (std::size_t slot = 0; slot < 27; ++slot) {
                    if (slot != kCenter)
                        neighbors += neighborhood[slot];
                }
                // Keep line endpoints: a voxel with a single neighbor.
                if (neighbors <= 1)
                    continue;
                if (!fiberSkeletonSimplePoint(neighborhood))
                    continue;
                candidates.push_back(flat);
            }

            // Candidates were classified against the same mask, so each
            // removal is re-verified against the mask as it now stands.
            for (const std::size_t flat : candidates) {
                mask[flat] = 0;
                gather(flat);
                if (!fiberSkeletonSimplePoint(neighborhood)) {
                    mask[flat] = 1;
                    continue;
                }
                passRemoved = true;
                for (std::size_t slot = 0; slot < 27; ++slot) {
                    if (slot == kCenter)
                        continue;
                    const auto neighbor =
                        static_cast<std::ptrdiff_t>(flat) + offsets[slot];
                    if (neighbor >= 0 &&
                        neighbor < static_cast<std::ptrdiff_t>(total)) {
                        markRevisit(static_cast<std::size_t>(neighbor));
                    }
                }
            }
        }
        // Compact the foreground and its revisit flags together so positions
        // stay aligned for the next pass.
        std::size_t kept = 0;
        for (std::size_t position = 0; position < foreground.size();
             ++position) {
            if (mask[foreground[position]] == 0)
                continue;
            foreground[kept] = foreground[position];
            revisitNext[kept] = revisitNext[position];
            ++kept;
        }
        foreground.resize(kept);
        revisitNext.resize(kept);
        revisit.swap(revisitNext);
        revisitNext.assign(foreground.size(), 0);
    }
}

std::vector<std::int32_t> labelFiberSkeletonComponents(
    const std::array<std::size_t, 3>& shapeZYX,
    const std::vector<std::uint8_t>& mask, std::size_t& componentCount)
{
    const std::size_t sizeZ = shapeZYX[0];
    const std::size_t sizeY = shapeZYX[1];
    const std::size_t sizeX = shapeZYX[2];
    const std::size_t total = sizeZ * sizeY * sizeX;
    if (mask.size() != total)
        throw std::invalid_argument("label mask size does not match shape");
    std::vector<std::int32_t> labels(total, 0);
    componentCount = 0;
    if (total == 0)
        return labels;

    std::vector<std::size_t> stack;
    for (std::size_t seed = 0; seed < total; ++seed) {
        if (mask[seed] == 0 || labels[seed] != 0)
            continue;
        ++componentCount;
        const auto label = static_cast<std::int32_t>(componentCount);
        labels[seed] = label;
        stack.clear();
        stack.push_back(seed);
        while (!stack.empty()) {
            const std::size_t flat = stack.back();
            stack.pop_back();
            const std::size_t x = flat % sizeX;
            const std::size_t y = (flat / sizeX) % sizeY;
            const std::size_t z = flat / (sizeX * sizeY);
            for (std::size_t slot = 0; slot < 27; ++slot) {
                if (slot == kCenter)
                    continue;
                const auto delta = neighborOffset(slot);
                const std::ptrdiff_t nz =
                    static_cast<std::ptrdiff_t>(z) + delta[0];
                const std::ptrdiff_t ny =
                    static_cast<std::ptrdiff_t>(y) + delta[1];
                const std::ptrdiff_t nx =
                    static_cast<std::ptrdiff_t>(x) + delta[2];
                if (nz < 0 || ny < 0 || nx < 0 ||
                    nz >= static_cast<std::ptrdiff_t>(sizeZ) ||
                    ny >= static_cast<std::ptrdiff_t>(sizeY) ||
                    nx >= static_cast<std::ptrdiff_t>(sizeX)) {
                    continue;
                }
                const auto neighbor = static_cast<std::size_t>(
                    (nz * static_cast<std::ptrdiff_t>(sizeY * sizeX)) +
                    (ny * static_cast<std::ptrdiff_t>(sizeX)) + nx);
                if (mask[neighbor] == 0 || labels[neighbor] != 0)
                    continue;
                labels[neighbor] = label;
                stack.push_back(neighbor);
            }
        }
    }
    return labels;
}

std::vector<std::uint8_t> thresholdFiberSkeletonCrop(
    const FiberSkeletonCrop& crop, int thresholdU8,
    std::size_t minComponentVoxels)
{
    if (thresholdU8 < 1 || thresholdU8 > 255)
        throw std::invalid_argument("skeleton threshold must be in [1, 255]");
    if (crop.presence.size() != crop.voxelCount())
        throw std::invalid_argument("skeleton crop presence size mismatch");

    std::vector<std::uint8_t> mask(crop.voxelCount(), 0);
    const auto threshold = static_cast<std::uint8_t>(thresholdU8);
    for (std::size_t flat = 0; flat < mask.size(); ++flat)
        mask[flat] = crop.presence[flat] >= threshold ? 1 : 0;

    if (minComponentVoxels > 1) {
        std::size_t componentCount = 0;
        const auto labels =
            labelFiberSkeletonComponents(crop.shapeZYX, mask, componentCount);
        std::vector<std::size_t> sizes(componentCount + 1, 0);
        for (const auto label : labels) {
            if (label > 0)
                ++sizes[static_cast<std::size_t>(label)];
        }
        for (std::size_t flat = 0; flat < mask.size(); ++flat) {
            const auto label = labels[flat];
            if (label > 0 &&
                sizes[static_cast<std::size_t>(label)] < minComponentVoxels) {
                mask[flat] = 0;
            }
        }
    }
    return mask;
}

}  // namespace vc::fiber_tracer
