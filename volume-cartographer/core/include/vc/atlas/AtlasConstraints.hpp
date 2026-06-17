#pragma once

#include "vc/atlas/Atlas.hpp"
#include "vc/core/PointCollections.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace vc::lasagna {
class LasagnaNormalSampler;
}

namespace vc::atlas {

struct AtlasConstraintExportOptions {
    bool closeCycles = true;
    bool exportLineConstraints = true;
    bool exportCrossWindingConstraints = true;
    double closeMinSignedWinding = -0.5;
    double closeMaxSignedWinding = 0.0;
    double closeAtlasWindingThreshold = 0.1;
    double lineMaxWindingStep = 0.25;
    double crossWindingTarget = 1.0;
    double crossWindingTolerance = 0.1;
    double crossZThreshold = 25.0;
    double intersectionMaxDistance = 500.0;
    double intersectionMaxSampleSpacing = 100.0;
    int intersectionSeedStride = 100;
    double intersectionClusterArclength = 8.0;
    int intersectionMaxIterations = 50;
    double intersectionDeduplicateArclength = 4.0;
    size_t greedyBeamWidth = 32;
    std::filesystem::path debugImagesDir;
};

struct AtlasConstraintExportReport {
    size_t atlasFibers = 0;
    size_t sourceLinks = 0;
    size_t temporaryLinks = 0;
    size_t lineCollections = 0;
    size_t linePoints = 0;
    size_t crossCollections = 0;
    size_t crossPoints = 0;
    size_t debugImagesWritten = 0;
    size_t skippedLargeLineSteps = 0;
};

struct AtlasConstraintExportResult {
    PointCollections collections;
    AtlasConstraintExportReport report;
};

[[nodiscard]] AtlasConstraintExportResult exportAtlasConstraints(
    const LasagnaAtlasExport& exportData,
    const QuadSurface* baseSurface,
    const vc::lasagna::LasagnaNormalSampler* windingSampler,
    const AtlasConstraintExportOptions& options = {});

} // namespace vc::atlas
