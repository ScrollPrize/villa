#pragma once
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/PointCollections.hpp"
#include "utils/Json.hpp"
#include "vc/tracer/DirectionField.hpp"

class Volume;
class SurfacePatchIndex;

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize = 1.0);
QuadSurface *tracer(Volume& volume, float scale, int level, cv::Vec3f origin, const utils::Json &params, const std::string &cache_root = "", float voxelsize = 1.0, const std::vector<DirectionField> &direction_fields = {}, QuadSurface* resume_surf = nullptr, const std::filesystem::path& tgt_path = "", const utils::Json& meta_params = {}, const PointCollections &corrections = PointCollections(), const cv::Mat* allowed_growth_mask = nullptr);
