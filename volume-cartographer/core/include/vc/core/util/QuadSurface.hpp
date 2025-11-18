#pragma once

#include <filesystem>
#include <set>
#include <optional>
#include <unordered_map>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <nlohmann/json_fwd.hpp>
#include <z5/dataset.hxx>

#include "Surface.hpp"

// Forward declarations
template<typename T>
class ChunkCache;

// Surface loading and channel flags
#define SURF_LOAD_IGNORE_MASK 1
#define SURF_CHANNEL_NORESIZE 1

// Debug prefix for auto-generated surfaces
#define Z_DBG_GEN_PREFIX "auto_grown_"

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);

// Options for writing TIFF from QuadSurface
struct TiffWriteOptions {
    enum class Compression { NONE, LZW, DEFLATE };
    enum class Predictor  { NONE, HORIZONTAL, FLOATINGPOINT };

    bool forceBigTiff = false;
    int  tileSize = 1024;                 // square tiles
    Compression compression = Compression::LZW;
    Predictor  predictor   = Predictor::FLOATINGPOINT; // for float32
};

//quads based surface class with a pointer implementing a nominal scale of 1 voxel
class QuadSurface : public Surface
{
public:
    cv::Vec3f pointer() override;
    QuadSurface() = default;
    // points will be cloned in constructor
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    // points will not be cloned in constructor, but pointer stored
    QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale);
    ~QuadSurface() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc_raw(const cv::Vec3f &ptr);
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000) override;
    cv::Size size();
    [[nodiscard]] cv::Vec2f scale() const;
    [[nodiscard]] cv::Vec3f center() const;

    void save(const std::string &path, const std::string &uuid, bool force_overwrite = false);
    void save(const std::filesystem::path &path, bool force_overwrite = false);
    void save_meta();
    // Configure how TIFFs are written
    void setTiffWriteOptions(const TiffWriteOptions& opt) { _tiff_opts = opt; }
    Rect3D bbox();

    bool containsPoint(const cv::Vec3f& point, float tolerance) const;

    virtual cv::Mat_<cv::Vec3f> rawPoints() { return *_points; }
    virtual cv::Mat_<cv::Vec3f> *rawPointsPtr() { return _points; }
    virtual const cv::Mat_<cv::Vec3f> *rawPointsPtr() const { return _points; }

    friend QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search, int step_out);
    friend QuadSurface *smooth_vc_segmentation(QuadSurface *src);
    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name, int flags = 0);
    void invalidateCache();
    void saveOverwrite();
    void saveSnapshot(int maxBackups = 10);
    std::vector<std::string> channelNames() const;
protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    cv::Mat_<cv::Vec3f>* _points = nullptr;
    cv::Rect _bounds;
    TiffWriteOptions _tiff_opts;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
};

QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags = 0);
QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search = 100, int step_out = 5);
QuadSurface *smooth_vc_segmentation(QuadSurface *src);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

QuadSurface* surface_diff(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
QuadSurface* surface_union(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
QuadSurface* surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);

// Control CUDA usage in GrowPatch (space_tracing_quad_phys). Default is true.
void set_space_tracing_use_cuda(bool enable);

void generate_mask(QuadSurface* surf,
                            cv::Mat_<uint8_t>& mask,
                            cv::Mat_<uint8_t>& img,
                            z5::Dataset* ds_high = nullptr,
                            z5::Dataset* ds_low = nullptr,
                            ChunkCache<uint8_t>* cache = nullptr);

class MultiSurfaceIndex {
private:
    struct Cell {
        std::vector<int> patch_indices;
    };

    std::unordered_map<uint64_t, Cell> grid;
    float cell_size;
    std::vector<Rect3D> patch_bboxes;

    uint64_t hash(int x, int y, int z) const {
        // Ensure non-negative values for hashing
        uint32_t ux = static_cast<uint32_t>(x + 1000000);
        uint32_t uy = static_cast<uint32_t>(y + 1000000);
        uint32_t uz = static_cast<uint32_t>(z + 1000000);
        return (static_cast<uint64_t>(ux) << 40) |
               (static_cast<uint64_t>(uy) << 20) |
               static_cast<uint64_t>(uz);
    }

public:
    MultiSurfaceIndex(float cell_sz = 100.0f) : cell_size(cell_sz) {}

    void addPatch(int idx, QuadSurface* patch) {
        Rect3D bbox = patch->bbox();
        patch_bboxes.push_back(bbox);

        // Expand bbox slightly to handle edge cases
        int x0 = std::floor((bbox.low[0] - cell_size) / cell_size);
        int y0 = std::floor((bbox.low[1] - cell_size) / cell_size);
        int z0 = std::floor((bbox.low[2] - cell_size) / cell_size);
        int x1 = std::ceil((bbox.high[0] + cell_size) / cell_size);
        int y1 = std::ceil((bbox.high[1] + cell_size) / cell_size);
        int z1 = std::ceil((bbox.high[2] + cell_size) / cell_size);

        for (int z = z0; z <= z1; z++) {
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    grid[hash(x, y, z)].patch_indices.push_back(idx);
                }
            }
        }
    }

    std::vector<int> getCandidatePatches(const cv::Vec3f& point, float tolerance = 0.0f) const {
        // Get the cell containing this point
        int x = std::floor(point[0] / cell_size);
        int y = std::floor(point[1] / cell_size);
        int z = std::floor(point[2] / cell_size);

        // If tolerance is specified, check neighboring cells too
        std::set<int> unique_patches;

        if (tolerance > 0) {
            int cell_radius = std::ceil(tolerance / cell_size);
            for (int dz = -cell_radius; dz <= cell_radius; dz++) {
                for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                    for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                        auto it = grid.find(hash(x + dx, y + dy, z + dz));
                        if (it != grid.end()) {
                            for (int idx : it->second.patch_indices) {
                                unique_patches.insert(idx);
                            }
                        }
                    }
                }
            }
        } else {
            auto it = grid.find(hash(x, y, z));
            if (it != grid.end()) {
                for (int idx : it->second.patch_indices) {
                    unique_patches.insert(idx);
                }
            }
        }

        // Filter by bounding box for extra safety
        std::vector<int> result;
        for (int idx : unique_patches) {
            const Rect3D& bbox = patch_bboxes[idx];
            if (point[0] >= bbox.low[0] - tolerance &&
                point[0] <= bbox.high[0] + tolerance &&
                point[1] >= bbox.low[1] - tolerance &&
                point[1] <= bbox.high[1] + tolerance &&
                point[2] >= bbox.low[2] - tolerance &&
                point[2] <= bbox.high[2] + tolerance) {
                result.push_back(idx);
            }
        }

        return result;
    }

    size_t getCellCount() const { return grid.size(); }
    size_t getPatchCount() const { return patch_bboxes.size(); }
};
