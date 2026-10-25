// Grid meshing and the grid-space UV lift used by SLIM flattening.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceMeshing.hpp"

using namespace vc::meshing;

namespace {
cv::Mat_<cv::Vec3f> sheet(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) points(r, c) = {2.f * c, 2.f * r, 0.1f * c * r};
    return points;
}

void invalidate(cv::Mat_<cv::Vec3f>& points, int r, int c) { points(r, c) = {-1.f, -1.f, -1.f}; }

// UVs that are an affine function of the decimated grid position: an exact
// lift reproduces the function at every lifted vertex.
cv::Vec2d affine(double r, double c) { return {3.0 * c + 0.5 * r + 1.0, -2.0 * r + 0.25 * c - 5.0}; }
}

TEST_CASE("meshGrid triangulates valid quads and numbers vertices in first-use order")
{
    auto points = sheet(3, 3);
    invalidate(points, 2, 2);
    const auto mesh = meshGrid(points);
    // Quads (0,0) (0,1) (1,0) are valid; (1,1) touches the invalid corner.
    CHECK(mesh.faces.size() == 6);
    CHECK(mesh.positions.size() == 8);
    CHECK(mesh.cells[0] == cv::Vec2i(0, 0));
    CHECK(mesh.cells[1] == cv::Vec2i(0, 1));
    CHECK(mesh.cells[2] == cv::Vec2i(1, 0));
    CHECK(mesh.cells[3] == cv::Vec2i(1, 1));
    CHECK(mesh.faces[0] == cv::Vec3i(2, 0, 1));
    CHECK(mesh.faces[1] == cv::Vec3i(2, 1, 3));
}

TEST_CASE("decimateGrid keeps every stride-th row and column")
{
    const auto points = sheet(10, 9);
    int stride = 0;
    const auto coarse = decimateGrid(points, 0.25f, &stride);
    CHECK(stride == 2);
    CHECK(coarse.rows == 5);
    CHECK(coarse.cols == 5);
    CHECK(coarse(4, 4) == points(8, 8));
    CHECK(decimateGrid(points, 1.0f, &stride).size() == points.size());
    CHECK(stride == 1);
}

TEST_CASE("liftGridUVs is exact in grid space when border rows and columns are invalid")
{
    // vc_obj_uv_lift aligned the coarse and fine UV bounding boxes, which
    // misplaced every vertex when the first valid rows of the two grids
    // differ (here: fine row 3 vs decimated row 2 = fine row 4).
    auto points = sheet(21, 17);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < points.cols; ++c) invalidate(points, r, c);
    for (int r = 0; r < points.rows; ++r) invalidate(points, r, 0);
    invalidate(points, 10, 9);  // an interior hole

    int stride = 0;
    const auto coarsePoints = decimateGrid(points, 0.25f, &stride);
    REQUIRE(stride == 2);
    const auto coarse = meshGrid(coarsePoints);
    std::vector<cv::Vec2d> coarseUVs;
    for (const auto& cell : coarse.cells) coarseUVs.push_back(affine(cell[0], cell[1]));

    const auto fine = meshGrid(points);
    const auto lifted = liftGridUVs(fine, coarsePoints, coarse, coarseUVs, stride);
    REQUIRE(lifted.uvs.size() == fine.cells.size());

    int onSamples = 0, inside = 0;
    for (std::size_t i = 0; i < fine.cells.size(); ++i) {
        const int r = fine.cells[i][0], c = fine.cells[i][1];
        const double gr = double(r) / stride, gc = double(c) / stride;
        const int qr = int(gr), qc = int(gc);
        const bool inQuad = qr + 1 < coarsePoints.rows && qc + 1 < coarsePoints.cols &&
                            coarsePoints(qr, qc)[0] != -1 && coarsePoints(qr + 1, qc)[0] != -1 &&
                            coarsePoints(qr, qc + 1)[0] != -1 && coarsePoints(qr + 1, qc + 1)[0] != -1;
        if (!inQuad) continue;
        ++inside;
        REQUIRE(lifted.valid[i]);
        const cv::Vec2d expected = affine(gr, gc);
        CHECK(std::abs(lifted.uvs[i][0] - expected[0]) < 1e-4);
        CHECK(std::abs(lifted.uvs[i][1] - expected[1]) < 1e-4);
        if (r % stride == 0 && c % stride == 0) {
            ++onSamples;
            // A vertex on a decimated sample takes that sample's own UV.
            CHECK(lifted.uvs[i][0] == float(expected[0]));
            CHECK(lifted.uvs[i][1] == float(expected[1]));
        }
    }
    CHECK(inside > 150);
    CHECK(onSamples > 40);
    // Vertices beyond the decimated quads (the last fine row/column here) still lift.
    for (std::size_t i = 0; i < fine.cells.size(); ++i) CHECK(lifted.valid[i]);
}
