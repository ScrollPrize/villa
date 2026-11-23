// Audit tests for the mesh <-> tifxyz tools (vc::meshing), the written tifxyz scale: metric mode and decimation. Cases B1 and B2 of
// the audit of 2026-11-07; both fail on main 7e6b82164 and pass with the fix.
// Same style as core/test/test_surface_meshing.cpp (doctest). Written by the audit reader "mesh", 2026-11-07.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceMeshing.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace vc::meshing;

namespace {
// A lattice plane of rows x cols vertices, `step` voxels apart, UV = voxel offsets (as vc_tifxyz2obj
// writes them for a surface of scale 1/step), faces wound as meshGrid winds them.
UvMesh plane(int rows, int cols, float step = 4.f)
{
    UvMesh m;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            m.vertices.push_back({10000.f + step * c, 8000.f + step * r, 6000.f});  // flat: 3-D spacing == step
            m.uvs.push_back({step * c, step * r});
        }
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r + 1 < rows; ++r)
        for (int c = 0; c + 1 < cols; ++c) {
            m.faces.push_back({{id(r + 1, c), id(r, c), id(r, c + 1)}, {id(r + 1, c), id(r, c), id(r, c + 1)}});
            m.faces.push_back({{id(r + 1, c), id(r, c + 1), id(r + 1, c + 1)}, {id(r + 1, c), id(r, c + 1), id(r + 1, c + 1)}});
        }
    return m;
}

bool valid(const cv::Vec3f& p) { return p[0] != -1.f; }

int validCount(const cv::Mat_<cv::Vec3f>& g)
{
    int n = 0;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) n += valid(g(r, c));
    return n;
}

// Median 3-D distance between horizontally adjacent valid grid points: voxels per cell.
double voxelsPerCell(const cv::Mat_<cv::Vec3f>& g)
{
    std::vector<double> d;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c + 1 < g.cols; ++c)
            if (valid(g(r, c)) && valid(g(r, c + 1))) d.push_back(cv::norm(g(r, c + 1) - g(r, c)));
    REQUIRE(!d.empty());
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return d[d.size() / 2];
}
}

TEST_CASE("B1: metric mode writes grid cells per OBJ unit, as its readers expect")
{
    auto m = plane(31, 41);
    for (float stretch : {0.25f, 4.f}) {
        UvGridRasterizer r;
        r.setMesh(m.vertices, m.uvs, m.faces);
        r.determineGridDimensions(stretch);
        std::unique_ptr<QuadSurface> s(r.createQuadSurface(1.f));
        const double cells_per_voxel = 1.0 / voxelsPerCell(*s->rawPointsPtr());
        // as built: 4 at stretch 0.25 (16 x too large), 0.25 at stretch 4 (16 x too small)
        CHECK(std::abs(s->scale()[0] - cells_per_voxel) < 1e-3 * cells_per_voxel);
        CHECK(std::abs(s->scale()[1] - cells_per_voxel) < 1e-3 * cells_per_voxel);
        // QuadSurface::size() = cols / scale is the nominal extent in voxels: 164 for 41 cells of 4 voxels (160 + one cell)
        CHECK(std::abs(s->size().width - 160) <= 4);
    }
    // mesh_units does not change a cells-per-unit scale (positions stay in OBJ units)
    UvGridRasterizer r;
    r.setMesh(m.vertices, m.uvs, m.faces);
    r.determineGridDimensions(1.f);
    std::unique_ptr<QuadSurface> s(r.createQuadSurface(7.91f));
    CHECK(std::abs(s->scale()[0] - 1.f) < 1e-3f);
}

TEST_CASE("B2: --uv-downsample halves the scale with the grid")
{
    auto m = plane(61, 81, 1.f);
    for (bool source : {false, true}) {
        UvGridRasterizer r;
        if (source) r.setSrcScale({1.f, 1.f});
        r.setUVDownsample(2.f);
        r.setMesh(m.vertices, m.uvs, m.faces);
        r.determineGridDimensions(1.f);
        std::unique_ptr<QuadSurface> s(r.createQuadSurface(1.f));
        REQUIRE(s->rawPointsPtr()->cols == 41);
        CHECK(std::abs(voxelsPerCell(*s->rawPointsPtr()) - 2.0) < 1e-3);
        CHECK(std::abs(s->scale()[0] - 0.5f) < 1e-4f);  // as built: 1.0
    }
}

