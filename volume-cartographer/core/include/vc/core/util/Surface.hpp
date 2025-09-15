#pragma once
#include <filesystem>
#include <set>

#include <opencv2/core.hpp> 
#include <nlohmann/json_fwd.hpp>
#include <z5/dataset.hxx>

#include "Slicing.hpp"


#define Z_DBG_GEN_PREFIX "auto_grown_"


struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);



//base surface class
class Surface
{
public:
    virtual ~Surface();

    // get a central location point
    virtual cv::Vec3f pointer() = 0;

    //move pointer within internal coordinate system
    virtual void move(cv::Vec3f &ptr, const cv::Vec3f &offset) = 0;
    //does the pointer location contain valid surface data
    virtual bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //nominal pointer coordinates (in "output" coordinates)
    virtual cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //read coord at pointer location, potentially with (3) offset
    virtual cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) = 0;
    //coordgenerator relative to ptr&offset
    //needs to be deleted after use
    virtual void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) = 0;
    nlohmann::json *meta = nullptr;
    std::filesystem::path path;
    std::string id;
};

class PlaneSurface : public Surface
{
public:
    //Surface API FIXME
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset);
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override { return true; };
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) override { abort(); };

    PlaneSurface() {};
    PlaneSurface(cv::Vec3f origin_, cv::Vec3f normal_);

    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    float pointDist(cv::Vec3f wp);
    cv::Vec3f project(cv::Vec3f wp, float render_scale = 1.0, float coord_scale = 1.0);
    void setNormal(cv::Vec3f normal);
    void setOrigin(cv::Vec3f origin);
    cv::Vec3f origin();
    float scalarp(cv::Vec3f point) const;
protected:
    void update();
    cv::Vec3f _normal = {0,0,1};
    cv::Vec3f _origin = {0,0,0};
    cv::Matx33d _M;
    cv::Vec3d _T;
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

    void save(const std::string &path, const std::string &uuid);
    void save(std::filesystem::path &path);
    void save_meta();
    Rect3D bbox();

    bool containsPoint(const cv::Vec3f& point, float tolerance) const;

    virtual cv::Mat_<cv::Vec3f> rawPoints() { return *_points; }
    virtual cv::Mat_<cv::Vec3f> *rawPointsPtr() { return _points; }

    friend QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search, int step_out);
    friend QuadSurface *smooth_vc_segmentation(QuadSurface *src);
    friend class ControlPointSurface;
    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name);
protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    cv::Mat_<cv::Vec3f>* _points = nullptr;
    cv::Rect _bounds;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
};


//surface representing some operation on top of a base surface
//by default all ops but gen() are forwarded to the base
class DeltaSurface : public Surface
{
public:
    //default - just assign base ptr, override if additional processing necessary
    //like relocate ctrl points, mark as dirty, ...
    virtual void setBase(Surface *base);
    DeltaSurface(Surface *base);

    virtual cv::Vec3f pointer() override;

    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override = 0;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000) override;

protected:
    Surface *_base = nullptr;
};

//might in the future have more properties! or those props are handled in whatever class manages a set of control points ...
class SurfaceControlPoint
{
public:
    SurfaceControlPoint(Surface *base, const cv::Vec3f &ptr_, const cv::Vec3f &control);
    cv::Vec3f ptr; //location of control point in base surface
    cv::Vec3f orig_wp; //the original 3d location where the control point was created
    cv::Vec3f normal; //original normal
    cv::Vec3f control_point; //actual control point location - should be in line with _orig_wp along the normal, but could change if the underlaying surface changes!
};

class ControlPointSurface : public DeltaSurface
{
public:
    ControlPointSurface(Surface *base) : DeltaSurface(base) {};
    void addControlPoint(const cv::Vec3f &base_ptr, cv::Vec3f control_point);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    void setBase(Surface *base);

protected:
    std::vector<SurfaceControlPoint> _controls;
};

class RefineCompSurface : public DeltaSurface
{
public:
    RefineCompSurface(z5::Dataset *ds, ChunkCache *cache, QuadSurface *base = nullptr);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    float start = 0;
    float stop = -100;
    float step = 2.0;
    float low = 0.1;
    float high = 1.0;
protected:
    z5::Dataset *_ds;
    ChunkCache *_cache;
};

class SurfaceMeta
{
public:
    SurfaceMeta() {};
    SurfaceMeta(const std::filesystem::path &path_, const nlohmann::json &json);
    SurfaceMeta(const std::filesystem::path &path_);
    ~SurfaceMeta();
    void readOverlapping();
    QuadSurface *surface();
    void setSurface(QuadSurface *surf);
    std::string name();
    std::filesystem::path path;
    QuadSurface *_surf = nullptr;
    Rect3D bbox;
    nlohmann::json *meta = nullptr;
    std::set<std::string> overlapping_str;
    std::set<SurfaceMeta*> overlapping;
};

QuadSurface *load_quad_from_tifxyz(const std::string &path);
QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search = 100, int step_out = 5);
QuadSurface *smooth_vc_segmentation(QuadSurface *src);

bool overlap(SurfaceMeta &a, SurfaceMeta &b, int max_iters = 1000);
bool contains(SurfaceMeta &a, const cv::Vec3f &loc, int max_iters = 1000);
bool contains(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);
bool contains_any(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);

//TODO constrain to visible area? or add visible area display?
void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries = 10);

float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, float init_step = 16.0, float min_step = 0.125);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);

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
                            ChunkCache* cache = nullptr);

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