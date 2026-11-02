#pragma once
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/PointCollections.hpp"
#include <nlohmann/json.hpp>

namespace vcr { class Array; }
class SurfacePatchIndex;

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const nlohmann::json &params, float voxelsize = 1.0);
// Grows a surface in `volume`, a uint8 CT level (give it a MemoryCache: the tracer
// samples it point by point); coordinates are that level's voxels.
QuadSurface *tracer(vcr::Array& volume, float scale, cv::Vec3f origin, const nlohmann::json &params, float voxelsize = 1.0, QuadSurface* resume_surf = nullptr, const std::filesystem::path& tgt_path = "", const nlohmann::json& meta_params = {}, const PointCollections &corrections = PointCollections(), const cv::Mat* allowed_growth_mask = nullptr);
