#include "vc/fiber_tracer/FiberletPipeline.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace vc::fiber_tracer {
FiberletDatasetMetadata makeFiberletDatasetMetadata(
    FiberletDatasetKind kind, const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& anchors, const FiberletPathConfig& paths,
    int storageChunkSideBaseVoxels, const nlohmann::json& sources,
    const std::vector<cv::Vec3d>& corridorReferenceBase,
    double corridorRadiusBaseVoxels,
    const FiberletGeometryCacheProfile& cacheProfile,
    FiberletStorageProfile storageProfile,
    std::string_view selectionIdentity)
{
    const double cellSideBase = static_cast<double>(anchors.cellSizePredictionVoxels) * grid.predictionToBaseScale;
    const auto roundedCellSideBase = std::llround(cellSideBase);
    if (!(roundedCellSideBase > 0) || std::abs(cellSideBase - static_cast<double>(roundedCellSideBase)) > 1.0e-9 ||
        storageChunkSideBaseVoxels % roundedCellSideBase != 0) {
        throw std::invalid_argument("storage chunk side must be an exact multiple of the anchor cell side in base voxels");
    }
    const std::int64_t unitsPerChunk = storageChunkSideBaseVoxels / roundedCellSideBase;
    std::array<std::int64_t, 3> cellShape{};
    std::array<std::int32_t, 3> chunkShape{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        cellShape[axis] = static_cast<std::int64_t>(
            (grid.shapeZYX[axis] + static_cast<std::size_t>(anchors.cellSizePredictionVoxels) - 1) /
            static_cast<std::size_t>(anchors.cellSizePredictionVoxels));
        const auto chunks = (cellShape[axis] + unitsPerChunk - 1) / unitsPerChunk;
        if (chunks > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("fiberlet storage chunk grid exceeds int32");
        chunkShape[axis] = static_cast<std::int32_t>(chunks);
    }
    std::int64_t maximumReach = 0;
    for (const auto& offset : vc::fiber_tracer::fiberletCellNeighborhoodOffsets(paths.cellRadius, paths.neighborhoodMarginCells)) {
        for (const auto coordinate : offset)
            maximumReach = std::max(maximumReach, static_cast<std::int64_t>(std::abs(coordinate)));
    }
    vc::fiber_tracer::FiberletDatasetMetadata metadata;
    metadata.kind = kind;
    metadata.profile = storageProfile;
    metadata.chunkGridShapeZYX = chunkShape;
    metadata.coordinateOriginZYX = {0, 0, 0};
    metadata.coordinateUnitsPerChunkZYX = {unitsPerChunk, unitsPerChunk, unitsPerChunk};
    metadata.maximumEndpointReachCoordinateUnitsZYX = {maximumReach, maximumReach, maximumReach};
    metadata.spatialChunkSideBaseVoxels = static_cast<std::uint32_t>(storageChunkSideBaseVoxels);
    if (storageProfile == vc::fiber_tracer::FiberletStorageProfile::CompactDirectionsFixedCost) {
        metadata.costBits = 16;
    }
    metadata.predictionToBaseScale = grid.predictionToBaseScale;
    metadata.sources = sources;
    nlohmann::json corridor = nlohmann::json::array();
    for (const auto& point : corridorReferenceBase)
        corridor.push_back({point[0], point[1], point[2]});
    metadata.processing = {
        {"contract_version",
         vc::fiber_tracer::kFiberletGenerationContractVersion},
        {"grid", {
            {"shape_zyx", grid.shapeZYX},
            {"prediction_to_base", grid.predictionToBaseScale},
            {"coordinate_order", "zyx_storage_xyz_vectors"},
        }},
        {"selection", {
            {"algorithm", selectionIdentity},
            {"corridor_radius_base", corridorRadiusBaseVoxels},
            {"corridor_reference_base_xyz", corridor},
        }},
        {"anchors", {
            {"cell_size_prediction", anchors.cellSizePredictionVoxels},
            {"gaussian_sigma_prediction", anchors.gaussianSigmaPredictionVoxels},
            {"peak_sigma_prediction", anchors.peakSigmaPredictionVoxels},
            {"peak_axial_sigma_prediction", anchors.peakAxialSigmaPredictionVoxels},
            {"peak_grid_step_prediction", anchors.peakGridStepPredictionVoxels},
            {"peak_gradient_weight", anchors.peakGradientWeight},
            {"peak_gradient_reliability_scale", anchors.peakGradientReliabilityScale},
            {"gaussian_cutoff_sigmas", anchors.gaussianCutoffSigmas},
            {"local_window_radius_prediction", anchors.localWindowRadiusPredictionVoxels},
            {"axial_support_half_width_prediction", anchors.axialSupportHalfWidthPredictionVoxels},
            {"position_convergence_tolerance_prediction", anchors.positionConvergenceTolerancePredictionVoxels},
            {"nms_maximum_angle_degrees", anchors.nmsMaximumAngleDegrees},
            {"nms_transverse_radius_prediction", anchors.nmsTransverseRadiusPredictionVoxels},
            {"nms_longitudinal_radius_prediction", anchors.nmsLongitudinalRadiusPredictionVoxels},
            {"observation_presence_floor", anchors.observationPresenceFloor},
            {"minimum_aligned_support", anchors.minimumAlignedSupport},
            {"robust_maximum_trim_mass_fraction", anchors.robustMaximumTrimMassFraction},
            {"robust_mad_multiplier", anchors.robustMadMultiplier},
            {"robust_minimum_angle_degrees", anchors.robustMinimumAngleDegrees},
            {"merge_maximum_angle_degrees", anchors.mergeMaximumAngleDegrees},
            {"merge_maximum_absolute_objective_loss", anchors.mergeMaximumAbsoluteObjectiveLoss},
            {"merge_maximum_relative_objective_loss", anchors.mergeMaximumRelativeObjectiveLoss},
            {"maximum_seed_count", anchors.maximumSeedCount},
            {"maximum_iterations", anchors.maximumIterations},
            {"convergence_tolerance", anchors.convergenceTolerance},
        }},
        {"paths", {
            {"cell_radius", paths.cellRadius},
            {"neighborhood_margin_cells", paths.neighborhoodMarginCells},
            {"longitudinal_step_prediction", paths.longitudinalStepPredictionVoxels},
            {"transverse_step_prediction", paths.transverseStepPredictionVoxels},
            {"maximum_endpoint_angle_degrees", paths.maximumEndpointAngleDegrees},
            {"maximum_prediction_deviation_degrees", paths.maximumPredictionDeviationDegrees},
            {"corridor_radius_prediction", paths.corridorRadiusPredictionVoxels},
            {"invalid_prediction_cost_per_voxel", paths.invalidPredictionCostPerVoxel},
            {"smoothness_weight", paths.smoothnessWeight},
            {"smoothness_normal_weight", paths.smoothnessNormalWeight},
            {"smoothness_tangent_weight", paths.smoothnessTangentWeight},
            {"smoothness_free_angle_degrees", paths.smoothnessFreeAngleDegrees},
        }},
        {"layout", {
            {"sparse", true},
            {"chunk_grid_shape_zyx", chunkShape},
            {"coordinate_units_per_chunk_zyx", metadata.coordinateUnitsPerChunkZYX},
            {"maximum_endpoint_reach_coordinate_units_zyx", metadata.maximumEndpointReachCoordinateUnitsZYX},
            {"arrays", {"anchors", "prefix", "routes"}},
        }},
        {"storage", {
            {"profile", static_cast<int>(storageProfile)},
            {"spatial_chunk_side_base", storageChunkSideBaseVoxels},
            {"coordinate_bits", metadata.coordinateBits},
            {"delta_bits", metadata.deltaBits},
            {"route_count_bits", metadata.routeCountBits},
            {"route_lattice_bits", metadata.routeLatticeBits},
            {"cost_bits", metadata.costBits},
            {"position_quantum_base", metadata.positionQuantumBaseVoxels == 0
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(metadata.positionQuantumBaseVoxels)},
            {"compact_directions", cacheProfile.geometry.compactDirections},
            {"route_cost_density_schema", kind == vc::fiber_tracer::FiberletDatasetKind::Anchors
                 ? nlohmann::json(nullptr)
                 : nlohmann::json("sqrt_u16_max256_v1")},
            {"codec_envelope", "vc_fiberlet_chunk_v2"},
        }},
    };
    if (anchors.fitMethod != "legacy") {
        metadata.processing["anchors"]["weighted_fit"] = {
            {"schema",1},{"method",anchors.fitMethod},{"iterations",anchors.weightedFitIterations},
            {"presence_exponent",anchors.weightedFitPresenceExponent},
            {"direction_exponent",anchors.weightedFitDirectionExponent},
            {"local_sigma",anchors.weightedFitLocalSigma},{"damping",anchors.weightedFitDamping},
            {"owned_cell_only",true},{"nms",false}};
    }
    vc::fiber_tracer::finalizeFiberletDatasetIdentity(metadata);
    return metadata;
}

void configureCanonicalFiberletGeneration(FiberletOnDemandConfig& config) {
    config.anchorConfig.parallelThreads = 1;
    config.pathConfig.parallelThreads = 1;
    config.geometryQuantization = {.positionQuantumBaseVoxels = 0.0, .compactDirections = true};
    config.anchorCellPredicate = [](const std::array<std::size_t, 3>&) { return true; };
    config.anchorRetainPredicate = [](const FiberAnchor&) { return FiberAnchorRetainEvaluation{true, {}, {}}; };
    config.pointPredicate = [](const cv::Vec3d&) { return true; };
}
}
