#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <limits>

#include <opencv2/imgproc.hpp>



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
    cv::Vec2f scale;
    
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
    
    void determineGridDimensions(float stretch_factor = 1000.0f) {
        // Find UV bounds from all UVs used in faces
        uv_min = cv::Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        uv_max = cv::Vec2f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
        
        for (const auto& face : faces) {
            for (int i = 0; i < 3; i++) {
                if (face.vt[i] >= 0 && face.vt[i] < uvs.size()) {
                    cv::Vec2f uv = uvs[face.vt[i]].coord;
                    uv_min[0] = std::min(uv_min[0], uv[0]);
                    uv_min[1] = std::min(uv_min[1], uv[1]);
                    uv_max[0] = std::max(uv_max[0], uv[0]);
                    uv_max[1] = std::max(uv_max[1], uv[1]);
                }
            }
        }
        
        std::cout << "UV bounds: [" << uv_min[0] << ", " << uv_min[1] << "] to [" 
                  << uv_max[0] << ", " << uv_max[1] << "]" << std::endl;
        
        // Two-pass approach:
        // Pass 1: Create preliminary grid to measure scale
        cv::Vec2f uv_range = uv_max - uv_min;
        cv::Vec2i preliminary_grid_size;
        preliminary_grid_size[0] = static_cast<int>(std::ceil(uv_range[0] * stretch_factor)) + 1;
        preliminary_grid_size[1] = static_cast<int>(std::ceil(uv_range[1] * stretch_factor)) + 1;
        
        std::cout << "Creating preliminary grid: " << preliminary_grid_size[0] << " x " << preliminary_grid_size[1] << " to measure scale..." << std::endl;
        
        // Temporary scale for preliminary rasterization
        cv::Vec2f temp_scale;
        temp_scale[0] = uv_range[0] * stretch_factor / (preliminary_grid_size[0] - 1);
        temp_scale[1] = uv_range[1] * stretch_factor / (preliminary_grid_size[1] - 1);
        
        // Create preliminary grid
        cv::Mat_<cv::Vec3f> preliminary_points(preliminary_grid_size[1], preliminary_grid_size[0], cv::Vec3f(-1, -1, -1));
        
        // Rasterize with temporary scale
        scale = temp_scale;
        grid_size = preliminary_grid_size;
        for (const auto& face : faces) {
            rasterizeTriangle(preliminary_points, face);
        }
        
        // Calculate actual scale from preliminary grid
        calculateScaleFromGrid(preliminary_points);
        cv::Vec2f measured_scale = scale;
        
        // Pass 2: Calculate final grid dimensions from measured scale
        // The measured scale tells us the physical distance between adjacent grid points
        // We multiply by stretch_factor to get the number of grid points needed
        grid_size[0] = static_cast<int>(std::round(measured_scale[0] * stretch_factor)) + 1;
        grid_size[1] = static_cast<int>(std::round(measured_scale[1] * stretch_factor)) + 1;
        
        std::cout << "Final grid dimensions: " << grid_size[0] << " x " << grid_size[1] << std::endl;
        std::cout << "Scale factors: " << measured_scale[0] << ", " << measured_scale[1] << std::endl;
        
        // Set final scale
        scale = measured_scale;
    }
    
    QuadSurface* createQuadSurface(float mesh_units = 1.0f, int step = 1) {
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
        
        // Always calculate scale from the grid to preserve anisotropic scaling
        calculateScaleFromGrid(*points, mesh_units);

        if (step != 1) {

            cv::Size small = cv::Size(points->cols/step, points->rows/step);
            //crop to nearest multiple
            *points = points->operator()(cv::Rect(0,0,small.width*step, small.height*step));
            cv::resize(*points, *points, small, 0, 0, cv::INTER_NEAREST);

            scale /= step;
        }
        
        return new QuadSurface(points, scale);
    }
    
    void calculateScaleFromGrid(const cv::Mat_<cv::Vec3f>& points, float mesh_units = 1.0f) {
        // Based on vc_segmentation_scales from Slicing.cpp
        double sum_x = 0;
        double sum_y = 0;
        int count = 0;
        
        // Skip borders (10% on each side) to avoid artifacts
        int jmin = points.rows * 0.1 + 1;
        int jmax = points.rows * 0.9;
        int imin = points.cols * 0.1 + 1;
        int imax = points.cols * 0.9;
        int step = 4;
        
        // For small grids, use all points
        if (points.rows < 20 || points.cols < 20) {
            jmin = 1;
            jmax = points.rows;
            imin = 1;
            imax = points.cols;
            step = 1;
        }
        
        // Calculate average distance between adjacent points
        for (int j = jmin; j < jmax; j += step) {
            for (int i = imin; i < imax; i += step) {
                // Skip invalid points
                if (points(j, i)[0] == -1 || points(j, i-1)[0] == -1 || points(j-1, i)[0] == -1)
                    continue;
                
                // Distance to neighbor in X direction
                cv::Vec3f v = points(j, i) - points(j, i-1);
                double dist_x = std::sqrt(v.dot(v));
                if (dist_x > 0) {
                    sum_x += dist_x;
                }
                
                // Distance to neighbor in Y direction
                v = points(j, i) - points(j-1, i);
                double dist_y = std::sqrt(v.dot(v));
                if (dist_y > 0) {
                    sum_y += dist_y;
                }
                count++;
            }
        }
        
        if (count > 0 && sum_x > 0 && sum_y > 0) {
            // Scale is the average distance between points, adjusted by mesh units
            scale[0] = (sum_x / count) * mesh_units;
            scale[1] = (sum_y / count) * mesh_units;
        } else {
            // Fallback to UV-based scale if we couldn't calculate from grid
            std::cerr << "Warning: Could not calculate scale from grid, using UV-based fallback" << std::endl;
            // scale already set in determineGridDimensions
        }
        
        std::cout << "Calculated scale factors from grid: " << scale[0] << ", " << scale[1] << " micrometers" << std::endl;
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
        
        // Transform UVs to grid coordinates
        // Map from [uv_min, uv_max] to [0, grid_size-1]
        cv::Vec2f uv_range = uv_max - uv_min;
        uv0 = (uv0 - uv_min);
        uv0[0] = uv0[0] / uv_range[0] * (grid_size[0] - 1);
        uv0[1] = uv0[1] / uv_range[1] * (grid_size[1] - 1);
        
        uv1 = (uv1 - uv_min);
        uv1[0] = uv1[0] / uv_range[0] * (grid_size[0] - 1);
        uv1[1] = uv1[1] / uv_range[1] * (grid_size[1] - 1);
        
        uv2 = (uv2 - uv_min);
        uv2[0] = uv2[0] / uv_range[0] * (grid_size[0] - 1);
        uv2[1] = uv2[1] / uv_range[1] * (grid_size[1] - 1);
        
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
    if (argc < 3 || argc > 6) {
        std::cout << "usage: " << argv[0] << " <input.obj> <output_directory> [stretch_factor] [mesh_units] [step_size]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  stretch_factor: UV scaling factor (default: 1000.0)" << std::endl;
        std::cout << "  mesh_units: Units of the mesh coordinates in micrometers (default: 1.0)" << std::endl;
        std::cout << "  step size: quadmesh stepping factor (default 20)" << std::endl;
        std::cout << std::endl;
        std::cout << "Note: Scale factors are automatically calculated from the mesh grid structure." << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    std::filesystem::path output_dir = argv[2];
    float stretch_factor = 1000.0f;
    float mesh_units = 1.0f;  // mesh units in micrometers
    int step = 20;
    
    if (argc >= 4) {
        stretch_factor = std::atof(argv[3]);
        if (stretch_factor <= 0) {
            std::cerr << "Invalid stretch factor: " << stretch_factor << std::endl;
            return EXIT_FAILURE;
        }
    }
    
    if (argc >= 5) {
        mesh_units = std::atof(argv[4]);
        if (mesh_units <= 0) {
            std::cerr << "Invalid mesh units: " << mesh_units << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (argc >= 6) {
        step = std::atoi(argv[5]);
        if (mesh_units <= 0) {
            std::cerr << "invalid step size: " << step << std::endl;
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
    std::cout << "Stretch factor: " << stretch_factor << std::endl;
    std::cout << "Mesh units: " << mesh_units << " micrometers" << std::endl;
    std::cout << "Step size: " << step << std::endl;
    
    ObjToTifxyzConverter converter;
    
    // Load OBJ file
    if (!converter.loadObj(obj_path.string())) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Determine grid dimensions from UV coordinates
    converter.determineGridDimensions(stretch_factor);
    
    // Create quad surface
    QuadSurface* surf = converter.createQuadSurface(mesh_units, step);
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
