#pragma once
// Triangle meshes of QuadSurface grids and the way back: a UV-mapped mesh
// rasterized onto a regular grid in UV space (a flattened surface). Shared by
// the OBJ converters and SLIM flattening.

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

class QuadSurface;

namespace vc::meshing {

// Keeps ~`ratio` of the grid's points: every stride-th row and column with
// stride = max(2, round(1/sqrt(ratio))). A ratio outside (0, 1) keeps all
// (stride 1). `stride` receives the stride used.
cv::Mat_<cv::Vec3f> decimateGrid(const cv::Mat_<cv::Vec3f>& points, float ratio, int* stride = nullptr);

// The triangles of a grid's valid quads (loc_valid). Vertices are numbered in
// first-use order while walking quads row by row, each quad adding its new
// corners in the order (r,c) (r,c+1) (r+1,c) (r+1,c+1); each quad gives the
// faces (r+1,c)(r,c)(r,c+1) and (r+1,c)(r,c+1)(r+1,c+1).
struct GridMesh {
    std::vector<cv::Vec3f> positions;
    std::vector<cv::Vec2i> cells;  // (row, col) of each vertex in the grid
    std::vector<cv::Vec3i> faces;  // vertex indices, two per quad in quad order
};
GridMesh meshGrid(const cv::Mat_<cv::Vec3f>& points);

// UVs for the vertices of `fine` (meshGrid of a grid) from `coarseUVs`, the
// UVs of `coarse` = meshGrid(coarsePoints), coarsePoints = decimateGrid(grid,
// ..., stride). Vertices are matched by grid position: fine cell (r, c) lies at
// (r, c) / stride in the decimated grid, so a vertex on a decimated sample takes
// that sample's UV exactly and one inside a decimated quad the barycentric mix
// of the quad's triangle. A vertex outside the decimated quads (a hole, or the
// rows/columns after the last sample) takes the nearest decimated vertex within
// two cells; with none it stays invalid.
struct LiftedUVs {
    std::vector<cv::Vec2f> uvs;
    std::vector<uint8_t> valid;
};
LiftedUVs liftGridUVs(const GridMesh& fine, const cv::Mat_<cv::Vec3f>& coarsePoints, const GridMesh& coarse,
                      const std::vector<cv::Vec2d>& coarseUVs, int stride);

// A triangle of a UV-mapped mesh: vertex and texture-coordinate indices.
struct UvFace {
    int v[3];
    int vt[3];
};

// Rasterizes a UV-mapped triangle mesh onto a regular grid spanning its UV
// bounds; each grid point takes the barycentric mix of its triangle's 3D
// corners (exact fixed-point coverage, top-left fill rule, first triangle wins).
class UvGridRasterizer {
public:
    void setUVMetric(bool v) { uv_is_metric = v; }
    void setUVToObj(float r) { uv_to_obj = r; }
    void setUVDownsample(float f) { uv_downsample = std::max(1.0f, f); }
    void setGridCapPixels(uint64_t cap) { grid_cap_pixels = cap; }
    // Grid cells per UV unit of the source surface: the output grid gets the
    // source's sampling density and scale.
    void setSrcScale(const cv::Vec2f& s) { src_scale = s; }
    // The source's approval channel, resampled through setGridUV's cells.
    void setSrcApproval(const cv::Mat& m) { src_approval = m; }

    void setMesh(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec2f> uvs, std::vector<UvFace> faces);
    // Per-vertex (col, row) in the source grid; returns false (and keeps none)
    // when the count does not match the vertices.
    bool setGridUV(std::vector<cv::Vec2f> cells);

    bool hasGridUV() const { return !grid_uv.empty(); }
    bool hasApproval() const { return !src_approval.empty() && !grid_uv.empty(); }
    const cv::Mat_<cv::Vec3b>& outApproval() const { return out_approval; }

    void determineGridDimensions(float stretch_factor = 1.0f);
    QuadSurface* createQuadSurface(float mesh_units = 1.0f);

private:
    void calculateScaleFromGrid(const cv::Mat_<cv::Vec3f>& points, float mesh_units = 1.0f);
    void rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const UvFace& face);
    cv::Vec3b sampleSrcApproval(float colf, float rowf) const;

    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec2f> uvs;
    std::vector<UvFace> faces;
    std::vector<cv::Vec2f> grid_uv;  // per-vertex source (col,row); empty if none

    cv::Vec2f uv_min, uv_max;
    cv::Vec2i grid_size;
    cv::Vec2f scale;
    // UV handling
    bool uv_is_metric = true;   // if true, scale comes from UV spacing
    float uv_to_obj   = 1.0f;   // OBJ units per 1 UV unit (used when uv_is_metric)
    // UV decimation / capping
    float uv_downsample = 1.0f;            // user-specified uniform decimation
    uint64_t grid_cap_pixels = 0;          // cap total pixel count (0=off)
    // Approval resampling: original mask (BGR or 1-ch) + output accumulator.
    cv::Mat src_approval;                  // empty if not resampling approval
    cv::Mat_<cv::Vec3b> out_approval;      // sized to grid_size when active
    cv::Vec2f src_scale = cv::Vec2f(0.f, 0.f);  // <=0 = unset
};

} // namespace vc::meshing
