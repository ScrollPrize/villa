#pragma once

#include "vc/atlas/Atlas.hpp"
#include "vc/core/PointCollections.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    double crossWindingTolerance = 0.2;
    double crossZThreshold = 4000.0;
    double intersectionMaxDistance = 500.0;
    double intersectionMaxSampleSpacing = 100.0;
    int intersectionSeedStride = 100;
    double intersectionClusterArclength = 8.0;
    int intersectionMaxIterations = 50;
    double intersectionDeduplicateArclength = 4.0;
    size_t greedyBeamWidth = 32;
    std::filesystem::path debugImagesDir;
    std::filesystem::path debugDirectory;
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

struct AtlasConstraintLinkDebugRow {
    std::string kind;
    std::filesystem::path firstFiber;
    std::filesystem::path secondFiber;
    double firstSource = 0.0;
    double secondSource = 0.0;
    double firstWinding = 0.0;
    double secondWinding = 0.0;
    double atlasWindingDelta = 0.0;
    std::optional<double> signedWindingDistance;
    std::optional<int> desiredWindingDelta;
};

struct AtlasConstraintExportResult {
    PointCollections collections;
    AtlasConstraintExportReport report;
    std::vector<AtlasConstraintLinkDebugRow> linkDebugRows;
};

[[nodiscard]] AtlasConstraintExportResult exportAtlasConstraints(
    const LasagnaAtlasExport& exportData,
    const QuadSurface* baseSurface,
    const vc::lasagna::LasagnaNormalSampler* windingSampler,
    const AtlasConstraintExportOptions& options = {});

} // namespace vc::atlas
