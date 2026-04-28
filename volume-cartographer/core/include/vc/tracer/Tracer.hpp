#pragma once
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"
#include "utils/Json.hpp"

namespace vc { class VcDataset; }
namespace vc::cache { class BlockPipeline; }
class SurfacePatchIndex;

struct Chunked3dFloatFromUint8;
struct Chunked3dVec3fFromUint8;

struct DirectionField
{
    DirectionField(std::string dir,
                   std::unique_ptr<Chunked3dVec3fFromUint8> field,
                   std::unique_ptr<Chunked3dFloatFromUint8> weight_dataset,
                   float weight = 1.0f);
    ~DirectionField();
    DirectionField(DirectionField&&) noexcept;
    DirectionField& operator=(DirectionField&&) noexcept;

    std::string direction;
    std::unique_ptr<Chunked3dVec3fFromUint8> field_ptr;
    std::unique_ptr<Chunked3dFloatFromUint8> weight_ptr;
    float weight{1.0f};
};

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize = 1.0);
QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize, SurfacePatchIndex* surface_patch_index);
QuadSurface *tracer(vc::VcDataset *ds, float scale, vc::cache::BlockPipeline *cache, int level, cv::Vec3f origin, const utils::Json &params, const std::string &cache_root = "", float voxelsize = 1.0, const std::vector<DirectionField> &direction_fields = {}, QuadSurface* resume_surf = nullptr, const std::filesystem::path& tgt_path = "", const utils::Json& meta_params = {}, const VCCollection &corrections = VCCollection(), const cv::Mat* allowed_growth_mask = nullptr);
