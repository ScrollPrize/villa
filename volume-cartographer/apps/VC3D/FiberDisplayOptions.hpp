#pragma once

// Catalog filters and overlay settings of the Fibers panel, shared by the
// panel and the store overlay. Lengths are millimetres, distances volume voxels.
struct FiberDisplayOptions {
    bool showFibers = true;
    // Colour fibers by linked component and ring their linked control points.
    bool showLinked = false;
    bool rainbow = false;
    bool horizontal = true;
    bool vertical = true;
    double minLengthMm = 0.0;
    // Per viewport; 0 shows every intersecting fiber.
    int maxDisplayed = 5000;
    // Half thickness of the slab around a slice in which fibers are drawn.
    double sliceDistance = 3.0;
    double lineWidth = 3.0;
    bool operator==(const FiberDisplayOptions&) const = default;
};
