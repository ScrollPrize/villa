#pragma once

// Shared enumeration describing how segmentation edit handles propagate their influence.
enum class SegmentationInfluenceMode
{
    GridChebyshev = 0,
    GeodesicCircular = 1
};

