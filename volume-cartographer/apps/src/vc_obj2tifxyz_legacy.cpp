#include "vc/core/util/QuadSurface.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>



struct Vertex {
    cv::Vec3f pos;
};

struct UV {
    cv::Vec2f coord;
};

struct Face {
    int v[3];  // vertex indices
    int vt[3]; // texture coordinate indices
};

class ObjToTifxyzConverter {
private:
    std::vector<Vertex> vertices;
    std::vector<UV> uvs;
    std::vector<Face> faces;

    cv::Vec2f uv_min, uv_max;
    cv::Vec2i grid_size;
    cv::Vec2d full_resolution;  // Full voxel resolution before downsampling
    float step_size;

public:
    bool loadObj(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open OBJ file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;
                
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;
            
            if (prefix == "v") {
                Vertex v;
                iss >> v.pos[0] >> v.pos[1] >> v.pos[2];
                vertices.push_back(v);
            }
            else if (prefix == "vt") {
                UV uv;
                iss >> uv.coord[0] >> uv.coord[1];
                uvs.push_back(uv);
            }
            else if (prefix == "f") {
                Face face;
                std::string vertex_str;
                int idx = 0;
                
                while (iss >> vertex_str && idx < 3) {
                    // Parse vertex/texture/normal indices
                    std::replace(vertex_str.begin(), vertex_str.end(), '/', ' ');
                    std::istringstream viss(vertex_str);
                    
                    viss >> face.v[idx];
                    face.v[idx]--; // Convert to 0-based
                    
                    if (viss >> face.vt[idx]) {
                        face.vt[idx]--; // Convert to 0-based
                    } else {
                        face.vt[idx] = -1;
                    }
                    
                    idx++;
                }
                
                if (idx == 3) {
                    faces.push_back(face);
                }
            }
        }
        
        file.close();
        
        std::cout << "Loaded OBJ file:" << std::endl;
        std::cout << "  Vertices: " << vertices.size() << std::endl;
        std::cout << "  UVs: " << uvs.size() << std::endl;
        std::cout << "  Faces: " << faces.size() << std::endl;
        
        return !vertices.empty() && !faces.empty() && !uvs.empty();
    }
    
    void determineGridDimensions(float step = 20.0f) {
        step_size = step;

        // Find UV bounds from all UVs used in faces
        uv_min = cv::Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        uv_max = cv::Vec2f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

        // Compute directional scale factors by analyzing UV-to-3D mapping gradients
        // For each triangle, compute the Jacobian of the mapping to get stretch in U and V
        double sum_scale_u = 0.0, sum_scale_v = 0.0;
        double sum_weight = 0.0;

        for (const auto& face : faces) {
            // Check valid indices
            bool valid = true;
            for (int i = 0; i < 3; i++) {
                if (face.v[i] < 0 || face.v[i] >= (int)vertices.size() ||
                    face.vt[i] < 0 || face.vt[i] >= (int)uvs.size()) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            // Update UV bounds
            for (int i = 0; i < 3; i++) {
                cv::Vec2f uv = uvs[face.vt[i]].coord;
                uv_min[0] = std::min(uv_min[0], uv[0]);
                uv_min[1] = std::min(uv_min[1], uv[1]);
                uv_max[0] = std::max(uv_max[0], uv[0]);
                uv_max[1] = std::max(uv_max[1], uv[1]);
            }

            // Get triangle vertices and UVs
            cv::Vec3f p0 = vertices[face.v[0]].pos;
            cv::Vec3f p1 = vertices[face.v[1]].pos;
            cv::Vec3f p2 = vertices[face.v[2]].pos;
            cv::Vec2f uv0 = uvs[face.vt[0]].coord;
            cv::Vec2f uv1 = uvs[face.vt[1]].coord;
            cv::Vec2f uv2 = uvs[face.vt[2]].coord;

            // Compute edge vectors in UV and 3D
            cv::Vec2f e1_uv = uv1 - uv0;
            cv::Vec2f e2_uv = uv2 - uv0;
            cv::Vec3f e1_3d = p1 - p0;
            cv::Vec3f e2_3d = p2 - p0;

            // Compute UV triangle area for weighting
            double uv_cross = e1_uv[0] * e2_uv[1] - e1_uv[1] * e2_uv[0];
            double uv_area = 0.5 * std::abs(uv_cross);
            if (uv_area < 1e-12) continue;  // Skip degenerate triangles

            // Solve for the Jacobian: [dp/du, dp/dv] = [e1_3d, e2_3d] * inv([e1_uv, e2_uv]^T)
            // inv([a b; c d]) = 1/(ad-bc) * [d -b; -c a]
            double det = uv_cross;
            double inv_det = 1.0 / det;

            // dp/du = (e2_uv[1] * e1_3d - e1_uv[1] * e2_3d) / det
            // dp/dv = (-e2_uv[0] * e1_3d + e1_uv[0] * e2_3d) / det
            cv::Vec3f dp_du = (e2_uv[1] * e1_3d - e1_uv[1] * e2_3d) * inv_det;
            cv::Vec3f dp_dv = (-e2_uv[0] * e1_3d + e1_uv[0] * e2_3d) * inv_det;

            // Scale factors are the magnitudes of the gradients
            double scale_u = std::sqrt(dp_du.dot(dp_du));
            double scale_v = std::sqrt(dp_dv.dot(dp_dv));

            // Weight by triangle area
            sum_scale_u += scale_u * uv_area;
            sum_scale_v += scale_v * uv_area;
            sum_weight += uv_area;
        }

        // Average scale factors
        double avg_scale_u = (sum_weight > 0) ? sum_scale_u / sum_weight : 1.0;
        double avg_scale_v = (sum_weight > 0) ? sum_scale_v / sum_weight : 1.0;

        cv::Vec2f uv_range = uv_max - uv_min;

        std::cout << "UV bounds: [" << uv_min[0] << ", " << uv_min[1] << "] to ["
                  << uv_max[0] << ", " << uv_max[1] << "]" << std::endl;
        std::cout << "Directional scales: U=" << avg_scale_u << ", V=" << avg_scale_v << std::endl;

        // Compute full voxel resolution
        double full_res_u = uv_range[0] * avg_scale_u;
        double full_res_v = uv_range[1] * avg_scale_v;

        // Store for UV-to-grid mapping in rasterization
        full_resolution = cv::Vec2d(full_res_u, full_res_v);

        // Downsample by step_size for actual grid/tif dimensions
        grid_size[0] = static_cast<int>(std::ceil(full_res_u / step_size)) + 1;
        grid_size[1] = static_cast<int>(std::ceil(full_res_v / step_size)) + 1;

        std::cout << "Full resolution: " << full_res_u << " x " << full_res_v << std::endl;
        std::cout << "Grid dimensions: " << grid_size[0] << " x " << grid_size[1] << std::endl;
        std::cout << "Step size: " << step_size << ", Scale: " << (1.0f / step_size) << std::endl;
    }
    
    QuadSurface* createQuadSurface() {
        // Create points matrix initialized with invalid values
        cv::Mat_<cv::Vec3f>* points = new cv::Mat_<cv::Vec3f>(grid_size[1], grid_size[0], cv::Vec3f(-1, -1, -1));

        // Rasterize triangles onto the grid
        for (const auto& face : faces) {
            rasterizeTriangle(*points, face);
        }

        // Count valid points
        int valid_count = 0;
        for (int y = 0; y < grid_size[1]; y++) {
            for (int x = 0; x < grid_size[0]; x++) {
                if ((*points)(y, x)[0] != -1) {
                    valid_count++;
                }
            }
        }

        std::cout << "Valid grid points: " << valid_count << " / " << (grid_size[0] * grid_size[1])
                  << " (" << (100.0f * valid_count / (grid_size[0] * grid_size[1])) << "%)" << std::endl;

        // Scale = 1/step_size (matching GrowPatch.cpp pattern)
        cv::Vec2f scale = {1.0f / step_size, 1.0f / step_size};

        return new QuadSurface(points, scale);
    }


private:
    void rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const Face& face) {
        // Get triangle vertices and UVs
        cv::Vec3f v0 = vertices[face.v[0]].pos;
        cv::Vec3f v1 = vertices[face.v[1]].pos;
        cv::Vec3f v2 = vertices[face.v[2]].pos;

        cv::Vec2f uv0 = uvs[face.vt[0]].coord;
        cv::Vec2f uv1 = uvs[face.vt[1]].coord;
        cv::Vec2f uv2 = uvs[face.vt[2]].coord;

        // Transform UVs to grid coordinates (downsampled)
        // 1. Map UV from [uv_min, uv_max] to full voxel resolution [0, full_resolution]
        // 2. Divide by step_size to get grid coordinates
        cv::Vec2f uv_range = uv_max - uv_min;

        // Map to full resolution, then downsample to grid coordinates
        uv0 = (uv0 - uv_min);
        uv0[0] = uv0[0] / uv_range[0] * full_resolution[0] / step_size;
        uv0[1] = uv0[1] / uv_range[1] * full_resolution[1] / step_size;

        uv1 = (uv1 - uv_min);
        uv1[0] = uv1[0] / uv_range[0] * full_resolution[0] / step_size;
        uv1[1] = uv1[1] / uv_range[1] * full_resolution[1] / step_size;

        uv2 = (uv2 - uv_min);
        uv2[0] = uv2[0] / uv_range[0] * full_resolution[0] / step_size;
        uv2[1] = uv2[1] / uv_range[1] * full_resolution[1] / step_size;
        
        // Find bounding box in grid coordinates
        int min_x = std::max(0, static_cast<int>(std::floor(std::min({uv0[0], uv1[0], uv2[0]}))) - 1);
        int max_x = std::min(grid_size[0] - 1, static_cast<int>(std::ceil(std::max({uv0[0], uv1[0], uv2[0]}))) + 1);
        int min_y = std::max(0, static_cast<int>(std::floor(std::min({uv0[1], uv1[1], uv2[1]}))) - 1);
        int max_y = std::min(grid_size[1] - 1, static_cast<int>(std::ceil(std::max({uv0[1], uv1[1], uv2[1]}))) + 1);
        
        // Rasterize triangle
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                cv::Vec2f p(x, y);
                
                // Compute barycentric coordinates
                cv::Vec3f bary = computeBarycentric(p, uv0, uv1, uv2);
                
                // Check if point is inside triangle
                if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
                    // Interpolate 3D position
                    cv::Vec3f pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2;
                    
                    // Only update if not already set (first triangle wins)
                    if (points(y, x)[0] == -1) {
                        points(y, x) = pos;
                    }
                }
            }
        }
    }
    
    cv::Vec3f computeBarycentric(const cv::Vec2f& p, const cv::Vec2f& a, const cv::Vec2f& b, const cv::Vec2f& c) {
        cv::Vec2f v0 = c - a;
        cv::Vec2f v1 = b - a;
        cv::Vec2f v2 = p - a;
        
        float dot00 = v0.dot(v0);
        float dot01 = v0.dot(v1);
        float dot02 = v0.dot(v2);
        float dot11 = v1.dot(v1);
        float dot12 = v1.dot(v2);
        
        float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        
        return cv::Vec3f(1.0f - u - v, v, u);
    }
};

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        std::cout << "usage: " << argv[0] << " <input.obj> <output_directory> [step_size]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  step_size: UV units per grid cell (default: 20)" << std::endl;
        std::cout << "             Scale will be 1/step_size (default: 0.05)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir" << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir 10" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    std::filesystem::path output_dir = argv[2];
    float step_size = 20.0f;

    if (argc >= 4) {
        step_size = std::atof(argv[3]);
        if (step_size <= 0) {
            std::cerr << "Invalid step size: " << step_size << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (!std::filesystem::exists(obj_path)) {
        std::cerr << "Input file does not exist: " << obj_path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Converting OBJ to tifxyz format" << std::endl;
    std::cout << "Input: " << obj_path << std::endl;
    std::cout << "Output: " << output_dir << std::endl;
    std::cout << "Step size: " << step_size << " (scale: " << (1.0f / step_size) << ")" << std::endl;

    ObjToTifxyzConverter converter;

    // Load OBJ file
    if (!converter.loadObj(obj_path.string())) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return EXIT_FAILURE;
    }

    // Determine grid dimensions from UV coordinates
    converter.determineGridDimensions(step_size);

    // Create quad surface
    QuadSurface* surf = converter.createQuadSurface();
    if (!surf) {
        std::cerr << "Failed to create quad surface" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Generate a UUID for the surface
    std::string uuid = output_dir.filename().string();
    if (uuid.empty()) {
        uuid = obj_path.stem().string();
    }
    
    std::cout << "Saving to tifxyz format..." << std::endl;
    
    try {
        surf->save(output_dir.string(), uuid);
        std::cout << "Successfully converted to tifxyz format" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving tifxyz: " << e.what() << std::endl;
        delete surf;
        return EXIT_FAILURE;
    }

    delete surf;
    return EXIT_SUCCESS;
}
