#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vc::fiber_tracer::detail
{

struct FiberAnchorSupportSpan {
    uint32_t z = 0;
    uint32_t y = 0;
    uint32_t xBegin = 0;
    uint32_t xEnd = 0;

    bool operator==(const FiberAnchorSupportSpan&) const = default;
};

[[nodiscard]] inline std::vector<FiberAnchorSupportSpan>
buildFiberAnchorSupportStencil(
    size_t cellSize,
    size_t sampleHalo,
    double maximumSupportRadius)
{
    if (sampleHalo > std::numeric_limits<uint32_t>::max() / 2 ||
        cellSize > std::numeric_limits<uint32_t>::max() - 2 * sampleHalo) {
        throw std::overflow_error("fiber anchor support stencil is too large");
    }
    const size_t extent = cellSize + 2 * sampleHalo;
    const double pivot = 0.5 * static_cast<double>(cellSize - 1);
    const double radiusSquared =
        maximumSupportRadius * maximumSupportRadius + 1.0e-12;
    std::vector<FiberAnchorSupportSpan> spans;
    spans.reserve(extent * extent);
    for (size_t z = 0; z < extent; ++z) {
        for (size_t y = 0; y < extent; ++y) {
            bool insideSpan = false;
            size_t spanBegin = 0;
            for (size_t x = 0; x <= extent; ++x) {
                bool retained = false;
                if (x < extent) {
                    const auto relativeZ = static_cast<double>(z) -
                        static_cast<double>(sampleHalo);
                    const auto relativeY = static_cast<double>(y) -
                        static_cast<double>(sampleHalo);
                    const auto relativeX = static_cast<double>(x) -
                        static_cast<double>(sampleHalo);
                    const bool owned =
                        relativeZ >= 0.0 && relativeZ < cellSize &&
                        relativeY >= 0.0 && relativeY < cellSize &&
                        relativeX >= 0.0 && relativeX < cellSize;
                    const double dz = relativeZ - pivot;
                    const double dy = relativeY - pivot;
                    const double dx = relativeX - pivot;
                    retained = owned ||
                        dz * dz + dy * dy + dx * dx <= radiusSquared;
                }
                if (retained && !insideSpan) {
                    insideSpan = true;
                    spanBegin = x;
                } else if (!retained && insideSpan) {
                    spans.push_back({
                        static_cast<uint32_t>(z),
                        static_cast<uint32_t>(y),
                        static_cast<uint32_t>(spanBegin),
                        static_cast<uint32_t>(x),
                    });
                    insideSpan = false;
                }
            }
        }
    }
    return spans;
}

[[nodiscard]] inline size_t fiberAnchorSupportStencilSize(
    const std::vector<FiberAnchorSupportSpan>& spans)
{
    size_t result = 0;
    for (const auto& span : spans)
        result += static_cast<size_t>(span.xEnd - span.xBegin);
    return result;
}

template <typename Visitor>
inline void visitFiberAnchorSupportStencilTileIndices(
    const std::vector<FiberAnchorSupportSpan>& spans,
    const std::array<size_t, 3>& cellSampleBegin,
    const std::array<size_t, 3>& tileSampleBegin,
    const std::array<size_t, 3>& tileSampleShape,
    Visitor&& visitor)
{
    const size_t plane = tileSampleShape[1] * tileSampleShape[2];
    for (const auto& span : spans) {
        const size_t z = cellSampleBegin[0] - tileSampleBegin[0] + span.z;
        const size_t y = cellSampleBegin[1] - tileSampleBegin[1] + span.y;
        size_t index = z * plane + y * tileSampleShape[2] +
            cellSampleBegin[2] - tileSampleBegin[2] + span.xBegin;
        const size_t end = index + span.xEnd - span.xBegin;
        for (; index < end; ++index)
            visitor(static_cast<uint32_t>(index));
    }
}

} // namespace vc::fiber_tracer::detail
