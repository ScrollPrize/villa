#include "common.hpp"
#include "../rounding.hpp"
#include <algorithm>
#include <cmath>

namespace vcr::views {
using namespace detail;

namespace {
// Masks k tokens chosen uniformly without replacement among the unmasked ones
// (partial Fisher-Yates over their ascending indices).
void maskUniform(std::uint8_t* mask, std::int64_t tokens, std::int64_t k, Stream& r, std::vector<std::int64_t>& free)
{
    free.clear();
    for (std::int64_t i = 0; i < tokens; ++i)
        if (!mask[i]) free.push_back(i);
    const auto n = std::int64_t(free.size());
    for (std::int64_t i = 0; i < k; ++i) {
        const auto j = i + std::int64_t(r.below(std::uint64_t(n - i)));
        std::swap(free[std::size_t(i)], free[std::size_t(j)]);
        mask[free[std::size_t(i)]] = 1;
    }
}

struct Blocks {
    Index3 shape;
    double minTokens, logMinAspect, logMaxAspect;
    std::int64_t attempts;

    // A cuboid of at most `cap` new tokens, or 0 after `attempts` failures.
    std::int64_t add(std::uint8_t* mask, double cap, Stream& r) const
    {
        for (std::int64_t attempt = 0; attempt < attempts; ++attempt) {
            const double volume = r.uniform(minTokens, cap);
            const double depthScale = std::exp(r.uniform(logMinAspect, logMaxAspect));
            const double planarAspect = std::exp(r.uniform(logMinAspect, logMaxAspect));
            const double base = std::cbrt(volume / depthScale);
            const double d = std::max(1.0, round_half_even(base * depthScale));
            const double area = std::max(volume / d, 1.0);
            const double h = std::max(1.0, round_half_even(std::sqrt(area * planarAspect)));
            const double w = std::max(1.0, round_half_even(std::sqrt(area / planarAspect)));
            if (d > double(shape[0]) || h > double(shape[1]) || w > double(shape[2])) continue;
            const Index3 size{std::int64_t(d), std::int64_t(h), std::int64_t(w)};
            Index3 corner{};
            for (int a = 0; a < 3; ++a) corner[a] = std::int64_t(r.below(std::uint64_t(shape[a] - size[a] + 1)));
            std::int64_t masked = 0;
            for (std::int64_t z = corner[0]; z < corner[0] + size[0]; ++z)
                for (std::int64_t y = corner[1]; y < corner[1] + size[1]; ++y) {
                    const auto* row = mask + (z * shape[1] + y) * shape[2];
                    for (std::int64_t x = corner[2]; x < corner[2] + size[2]; ++x) masked += row[x];
                }
            const std::int64_t added = size[0] * size[1] * size[2] - masked;
            if (added > 0 && double(added) <= cap) {
                for (std::int64_t z = corner[0]; z < corner[0] + size[0]; ++z)
                    for (std::int64_t y = corner[1]; y < corner[1] + size[1]; ++y)
                        std::fill_n(mask + (z * shape[1] + y) * shape[2] + corner[2], size[2], std::uint8_t(1));
                return added;
            }
        }
        return 0;
    }
};

void roll(std::uint8_t* mask, std::vector<std::uint8_t>& scratch, const Index3& shape, const Index3& shift)
{
    scratch.assign(mask, mask + product(shape));
    for (std::int64_t z = 0; z < shape[0]; ++z)
        for (std::int64_t y = 0; y < shape[1]; ++y) {
            const auto* source = scratch.data() + (z * shape[1] + y) * shape[2];
            auto* target = mask + (((z + shift[0]) % shape[0]) * shape[1] + (y + shift[1]) % shape[1]) * shape[2];
            const auto x = shift[2];
            std::copy(source, source + shape[2] - x, target + x);
            std::copy(source + shape[2] - x, source + shape[2], target);
        }
}
}  // namespace

void patchMasks(const Index3& shape, const std::uint64_t* seeds, std::size_t count, const MaskOptions& o,
                std::uint8_t* masks, bool* applied, std::int64_t* counts)
{
    for (int a = 0; a < 3; ++a) require(shape[a] > 0, "mask shape must be positive");
    const std::int64_t tokens = product(shape);
    require(std::isfinite(o.probability) && o.probability >= 0 && o.probability <= 1, "probability must be in [0, 1]");
    require(std::isfinite(o.fraction) && o.fraction >= 0 && o.fraction <= 1, "fraction must be in [0, 1]");
    require(o.mode == MaskMode::Random || o.mode == MaskMode::Multiblock, "unknown mask mode");
    require(std::isfinite(o.minBlockTokens) && o.minBlockTokens >= 0, "min_block_tokens must be finite and nonnegative");
    const double maxTokens = o.maxBlockTokens.value_or(double(tokens));
    require(std::isfinite(maxTokens) && maxTokens > 0, "max_block_tokens must be finite and positive");
    require(std::isfinite(o.minAspect) && o.minAspect > 0, "min_aspect must be finite and positive");
    const double maxAspect = o.maxAspect.value_or(1.0 / o.minAspect);
    require(std::isfinite(maxAspect) && maxAspect > 0, "max_aspect must be finite and positive");
    require(o.attempts >= 0, "attempts must be nonnegative");
    require(count == 0 || (seeds && masks), "seed and mask arrays are required");

    const Blocks blocks{shape, o.minBlockTokens, std::log(o.minAspect), std::log(maxAspect), o.attempts};
    std::vector<std::int64_t> free;
    std::vector<std::uint8_t> scratch;
    for (std::size_t b = 0; b < count; ++b) {
        auto* mask = masks + b * std::size_t(tokens);
        std::fill_n(mask, tokens, std::uint8_t(0));
        Stream r(seeds[b]);
        const bool on = r.uniform() < o.probability;
        if (applied) applied[b] = on;
        if (counts) counts[b] = 0;
        if (!on) continue;
        // Round half up.
        const std::int64_t target = std::clamp<std::int64_t>(nearest_index(o.fraction * double(tokens)), 0, tokens);
        if (o.mode == MaskMode::Random) {
            maskUniform(mask, tokens, target, r, free);
        } else {
            std::int64_t masked = 0;
            while (masked < target) {
                const auto added = blocks.add(mask, std::min(double(target - masked), maxTokens), r);
                if (!added) break;
                masked += added;
            }
            if (masked < target) maskUniform(mask, tokens, target - masked, r, free);
            if (o.cyclicShift) {
                Index3 shift{};
                for (int a = 0; a < 3; ++a) shift[a] = std::int64_t(r.below(std::uint64_t(shape[a])));
                roll(mask, scratch, shape, shift);
            }
        }
        if (counts) counts[b] = target;
    }
}
}  // namespace vcr::views
