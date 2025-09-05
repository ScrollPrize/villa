// vc_tifxyz2rgb.cpp
// Convert a tifxyz quadmesh to an RGB visualization of normalized (x,y,z).
// R,G,B = min-max normalized X,Y,Z over valid nodes.
//
// Usage:
//   ./tifxyz_to_rgb <mesh.tifxyz> [out.png]
// Produces an 8-bit RGB image (PNG/TIF supported by OpenCV).

#include <iostream>
#include <limits>
#include <filesystem>
#include <cmath>
#include <memory>
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "vc/core/util/Surface.hpp"  // QuadSurface, load_quad_from_tifxyz

namespace fs = std::filesystem;

static inline bool is_valid_point(const cv::Vec3f& p) {
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return false;
    // In this codebase invalid nodes are set to (-1,-1,-1)
    return !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <mesh.tifxyz> [output_image.png]\n";
        return 1;
    }

    const fs::path in_path = argv[1];

    fs::path out_path;
    if (argc == 3) {
        out_path = fs::path(argv[2]);
    } else {
        // <parent-parent>/rgb/<stem>.png
        std::string stem = in_path.stem().string();
        const std::string suffix = "_xyz_rgb";
        if (stem.size() >= suffix.size() &&
            stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
            stem.erase(stem.size() - suffix.size());
        }
        out_path = in_path.parent_path().parent_path() / "rgb" / (stem + ".png");
    }

    // 1) Load tifxyz as QuadSurface
    std::unique_ptr<QuadSurface> surf(load_quad_from_tifxyz(in_path.string()));
    if (!surf) {
        std::cerr << "Error: failed to load tifxyz: " << in_path << "\n";
        return 2;
    }

    cv::Mat_<cv::Vec3f> points = surf->rawPoints(); // rows x cols, (x,y,z) per grid node
    if (points.empty()) {
        std::cerr << "Error: empty points matrix in tifxyz.\n";
        return 3;
    }

    const int rows = points.rows;
    const int cols = points.cols;

    // 2) Compute per-axis min/max over valid nodes
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    int valid_count = 0;
    for (int j = 0; j < rows; ++j) {
        const cv::Vec3f* rowp = points.ptr<cv::Vec3f>(j);
        for (int i = 0; i < cols; ++i) {
            const cv::Vec3f& p = rowp[i];
            if (!is_valid_point(p)) continue;
            minX = std::min(minX, p[0]); maxX = std::max(maxX, p[0]);
            minY = std::min(minY, p[1]); maxY = std::max(maxY, p[1]);
            minZ = std::min(minZ, p[2]); maxZ = std::max(maxZ, p[2]);
            ++valid_count;
        }
    }

    if (valid_count == 0) {
        std::cerr << "Error: no valid nodes found in the mesh.\n";
        return 4;
    }

    const float eps = 1e-12f;
    const float rangeX = std::max(maxX - minX, eps);
    const float rangeY = std::max(maxY - minY, eps);
    const float rangeZ = std::max(maxZ - minZ, eps);

    // 3) Normalize (x,y,z) → [0,255] and pack into 8-bit RGB
    cv::Mat rgb(rows, cols, CV_8UC3, cv::Scalar(0,0,0)); // invalid nodes -> black
    for (int j = 0; j < rows; ++j) {
        const cv::Vec3f* rowp = points.ptr<cv::Vec3f>(j);
        cv::Vec3b* outp = rgb.ptr<cv::Vec3b>(j);
        for (int i = 0; i < cols; ++i) {
            const cv::Vec3f& p = rowp[i];
            if (!is_valid_point(p)) { outp[i] = cv::Vec3b(0,0,0); continue; }

            float nx = (p[0] - minX) / rangeX; // [0,1]
            float ny = (p[1] - minY) / rangeY;
            float nz = (p[2] - minZ) / rangeZ;

            // OpenCV uses BGR order for 8-bit images; the requirement asked for RGB.
            // We'll still write RGB logically (R=x, G=y, B=z) then swap to BGR for OpenCV storage.
            unsigned char R = static_cast<unsigned char>(std::round(255.0f * std::clamp(nx, 0.0f, 1.0f)));
            unsigned char G = static_cast<unsigned char>(std::round(255.0f * std::clamp(ny, 0.0f, 1.0f)));
            unsigned char B = static_cast<unsigned char>(std::round(255.0f * std::clamp(nz, 0.0f, 1.0f)));

            outp[i] = cv::Vec3b(B, G, R); // store as BGR for OpenCV
        }
    }

    // 4) Save
    fs::create_directories(out_path.parent_path());
    if (!cv::imwrite(out_path.string(), rgb)) {
        std::cerr << "Error: could not write output image to " << out_path << "\n";
        return 5;
    }

    std::cout << "Wrote normalized XYZ RGB image: " << out_path << "\n"
              << " - size: " << cols << " x " << rows << "\n"
              << " - X range: [" << minX << ", " << maxX << "]\n"
              << " - Y range: [" << minY << ", " << maxY << "]\n"
              << " - Z range: [" << minZ << ", " << maxZ << "]\n";

    return 0;
}
