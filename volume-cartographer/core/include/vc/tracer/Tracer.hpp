#pragma once
#include "vc/core/util/Surface.hpp"

struct Chunked3dFloatFromUint8;
struct Chunked3dVec3fFromUint8;

struct DirectionField
{
    std::string direction;
    std::unique_ptr<Chunked3dVec3fFromUint8> field_ptr;
    std::unique_ptr<Chunked3dFloatFromUint8> weight_ptr;
};

QuadSurface *grow_surf_from_surfs(SurfaceMeta *seed, const std::vector<SurfaceMeta*> &surfs_v, const nlohmann::json &params, float voxelsize = 1.0);
QuadSurface *space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, const nlohmann::json &params, const std::string &cache_root = "", float voxelsize = 1.0, std::vector<DirectionField> const &direction_fields = {}, QuadSurface* resume_surf = nullptr);
