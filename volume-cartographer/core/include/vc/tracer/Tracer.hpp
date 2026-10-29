#pragma once
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/PointCollections.hpp"
#include "utils/Json.hpp"
#include "vc/tracer/DirectionField.hpp"
#include "vc_reader/slice.hpp"

class SurfacePatchIndex;

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize = 1.0);
// Grows a surface in `volume`, a uint8 CT level registered with a chunk service;
// coordinates are that level's voxels.
QuadSurface *tracer(vcr::slice::Cached& volume, float scale, cv::Vec3f origin, const utils::Json &params, const std::string &cache_root = "", float voxelsize = 1.0, const std::vector<DirectionField> &direction_fields = {}, QuadSurface* resume_surf = nullptr, const std::filesystem::path& tgt_path = "", const utils::Json& meta_params = {}, const PointCollections &corrections = PointCollections(), const cv::Mat* allowed_growth_mask = nullptr);
