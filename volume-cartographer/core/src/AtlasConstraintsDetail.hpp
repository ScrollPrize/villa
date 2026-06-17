#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace vc::atlas::detail {

struct LinkDedupEntry {
    size_t fiberA = 0;
    double arclengthA = 0.0;
    size_t fiberB = 0;
    double arclengthB = 0.0;
};

[[nodiscard]] inline std::optional<double> sourceIndexToArclength(
    const std::vector<double>& cumulativeArclength,
    double sourceIndex)
{
    if (cumulativeArclength.empty() || !std::isfinite(sourceIndex)) {
        return std::nullopt;
    }

    const double clamped = std::clamp(sourceIndex,
                                      0.0,
                                      static_cast<double>(cumulativeArclength.size() - 1));
    const auto lo = static_cast<size_t>(std::floor(clamped));
    const auto hi = static_cast<size_t>(std::ceil(clamped));
    if (lo >= cumulativeArclength.size() || hi >= cumulativeArclength.size() ||
        !std::isfinite(cumulativeArclength[lo]) ||
        !std::isfinite(cumulativeArclength[hi])) {
        return std::nullopt;
    }
    if (lo == hi) {
        return cumulativeArclength[lo];
    }
    const double t = clamped - static_cast<double>(lo);
    return cumulativeArclength[lo] * (1.0 - t) + cumulativeArclength[hi] * t;
}

[[nodiscard]] inline std::optional<LinkDedupEntry> makeLinkDedupEntry(
    size_t firstFiber,
    double firstArclength,
    size_t secondFiber,
    double secondArclength)
{
    if (!std::isfinite(firstArclength) || !std::isfinite(secondArclength)) {
        return std::nullopt;
    }

    LinkDedupEntry entry{firstFiber, firstArclength, secondFiber, secondArclength};
    if (entry.fiberB < entry.fiberA ||
        (entry.fiberA == entry.fiberB && entry.arclengthB < entry.arclengthA)) {
        std::swap(entry.fiberA, entry.fiberB);
        std::swap(entry.arclengthA, entry.arclengthB);
    }
    return entry;
}

[[nodiscard]] inline bool linkDedupMatches(const LinkDedupEntry& kept,
                                           const LinkDedupEntry& candidate,
                                           double arclengthTolerance)
{
    return kept.fiberA == candidate.fiberA &&
           kept.fiberB == candidate.fiberB &&
           std::abs(kept.arclengthA - candidate.arclengthA) <= arclengthTolerance &&
           std::abs(kept.arclengthB - candidate.arclengthB) <= arclengthTolerance;
}

[[nodiscard]] inline bool containsLinkDedupEntry(const std::vector<LinkDedupEntry>& entries,
                                                 const LinkDedupEntry& candidate,
                                                 double arclengthTolerance)
{
    return std::any_of(entries.begin(), entries.end(), [&](const LinkDedupEntry& kept) {
        return linkDedupMatches(kept, candidate, arclengthTolerance);
    });
}

} // namespace vc::atlas::detail
