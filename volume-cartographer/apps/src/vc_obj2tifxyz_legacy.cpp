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
#include <cmath>
#include <chrono>
#include <random>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
    
    cv::Vec2f uv_min{0.f, 0.f};
    cv::Vec2f uv_max{0.f, 0.f};
    cv::Vec2i grid_size{0, 0};
    cv::Vec2f scale{0.f, 0.f};

    struct MappedMat : public cv::Mat_<cv::Vec3f> {
        void* mapped = nullptr;
        size_t length = 0;
        int fd = -1;
        std::filesystem::path path;

        MappedMat(int rows, int cols, void* ptr, size_t len, int fd_, std::filesystem::path p)
            : cv::Mat_<cv::Vec3f>(rows, cols, reinterpret_cast<cv::Vec3f*>(ptr))
            , mapped(ptr)
            , length(len)
            , fd(fd_)
            , path(std::move(p))
        {}

        ~MappedMat() {
            if (mapped && mapped != MAP_FAILED) {
                munmap(mapped, length);
            }
            if (fd >= 0) {
                close(fd);
            }
            if (!path.empty()) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        }
    };
    
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
    
    QuadSurface* createQuadSurface(float mesh_units = 1.0f, float uv_pixels_per_unit = 20.0f) {
        if (uv_pixels_per_unit <= 0.0f) {
            std::cerr << "Invalid UV pixel density: " << uv_pixels_per_unit << std::endl;
            return nullptr;
        }

        if (!computeUVBounds()) {
            return nullptr;
        }

        const cv::Vec2f uv_range = uv_max - uv_min;
        if (uv_range[0] <= 0.0f || uv_range[1] <= 0.0f) {
            std::cerr << "Invalid UV bounds (zero area)." << std::endl;
            return nullptr;
        }

        grid_size[0] = std::max(2, static_cast<int>(std::ceil(uv_range[0] * uv_pixels_per_unit)) + 1);
        grid_size[1] = std::max(2, static_cast<int>(std::ceil(uv_range[1] * uv_pixels_per_unit)) + 1);

        const double grid_megs = static_cast<double>(grid_size[0]) * static_cast<double>(grid_size[1]) * sizeof(cv::Vec3f) / (1024.0 * 1024.0);
        std::cout << "UV bounds: [" << uv_min[0] << ", " << uv_min[1] << "] to [" 
                  << uv_max[0] << ", " << uv_max[1] << "]" << std::endl;
        std::cout << "Grid dimensions: " << grid_size[0] << " x " << grid_size[1]
                  << " (~" << uv_pixels_per_unit << " px per UV unit)"
                  << " ~" << grid_megs << " MB" << std::endl;

        bool used_mmap = false;
        cv::Mat_<cv::Vec3f>* points = allocatePointGrid(grid_size[1], grid_size[0], &used_mmap);
        if (!points) {
            std::cerr << "Failed to allocate point grid." << std::endl;
            return nullptr;
        }
        if (used_mmap) {
            std::cout << "Using mmap-backed grid for rasterization." << std::endl;
        }
        const float uv_to_px = uv_pixels_per_unit;

        for (const auto& face : faces) {
            rasterizeTriangle(*points, face, uv_to_px);
        }

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

        if (valid_count == 0) {
            std::cerr << "No valid grid points rasterized; check UV parametrization or step size." << std::endl;
            delete points;
            return nullptr;
        }

        calculateScaleFromGrid(*points, mesh_units);

        return new QuadSurface(points, scale);
    }
    
    void calculateScaleFromGrid(const cv::Mat_<cv::Vec3f>& points, float mesh_units = 1.0f) {
        double sum_x = 0;
        double sum_y = 0;
        int count = 0;
        
        // Skip borders (10% on each side) to avoid artifacts
        int jmin = static_cast<int>(points.rows * 0.1) + 1;
        int jmax = static_cast<int>(points.rows * 0.9);
        int imin = static_cast<int>(points.cols * 0.1) + 1;
        int imax = static_cast<int>(points.cols * 0.9);
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
                if (points(j, i)[0] == -1 || points(j, i-1)[0] == -1 || points(j-1, i)[0] == -1)
                    continue;
                
                cv::Vec3f v = points(j, i) - points(j, i-1);
                double dist_x = std::sqrt(v.dot(v));
                if (dist_x > 0) {
                    sum_x += dist_x;
                }
                
                v = points(j, i) - points(j-1, i);
                double dist_y = std::sqrt(v.dot(v));
                if (dist_y > 0) {
                    sum_y += dist_y;
                }
                count++;
            }
        }
        
        if (count > 0 && sum_x > 0 && sum_y > 0) {
            scale[0] = static_cast<float>((sum_x / count) * mesh_units);
            scale[1] = static_cast<float>((sum_y / count) * mesh_units);
        } else {
            std::cerr << "Warning: Could not calculate scale from grid; leaving scale unchanged." << std::endl;
        }
        
        std::cout << "Calculated scale factors from grid: " << scale[0] << ", " << scale[1] << " micrometers" << std::endl;
    }
    
private:
    bool computeUVBounds() {
        const float inf = std::numeric_limits<float>::infinity();
        uv_min = {inf, inf};
        uv_max = {-inf, -inf};
        bool found = false;

        for (const auto& face : faces) {
            for (int i = 0; i < 3; i++) {
                if (face.vt[i] >= 0 && face.vt[i] < static_cast<int>(uvs.size())) {
                    cv::Vec2f uv = uvs[face.vt[i]].coord;
                    uv_min[0] = std::min(uv_min[0], uv[0]);
                    uv_min[1] = std::min(uv_min[1], uv[1]);
                    uv_max[0] = std::max(uv_max[0], uv[0]);
                    uv_max[1] = std::max(uv_max[1], uv[1]);
                    found = true;
                }
            }
        }

        if (!found) {
            std::cerr << "No valid UV coordinates found for faces." << std::endl;
            return false;
        }
        return true;
    }

    cv::Mat_<cv::Vec3f>* allocatePointGrid(int rows, int cols, bool* used_mmap = nullptr) {
        const size_t length = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(cv::Vec3f);
        const auto tmp_dir = std::filesystem::temp_directory_path();

        if (used_mmap) {
            *used_mmap = false;
        }

        if (length > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
            std::cerr << "Grid too large for mmap backing; falling back to heap allocation." << std::endl;
            return new cv::Mat_<cv::Vec3f>(rows, cols, cv::Vec3f(-1, -1, -1));
        }

        // Build a simple unique filename
        std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
        std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::path tmp_path;
        int fd = -1;
        void* ptr = MAP_FAILED;

        for (int attempt = 0; attempt < 5; ++attempt) {
            tmp_path = tmp_dir / ("vc_obj2tifxyz_" + std::to_string(dist(rng)) + ".bin");
            fd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0) {
                continue;
            }
            if (ftruncate(fd, static_cast<off_t>(length)) != 0) {
                ::close(fd);
                fd = -1;
                std::error_code ec;
                std::filesystem::remove(tmp_path, ec);
                continue;
            }
            ptr = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr == MAP_FAILED) {
                ::close(fd);
                fd = -1;
                std::error_code ec;
                std::filesystem::remove(tmp_path, ec);
                continue;
            }
            break;
        }

        if (ptr == MAP_FAILED || fd < 0) {
            // Fallback to in-memory allocation
            std::cerr << "Warning: falling back to heap allocation for point grid." << std::endl;
            auto* mat = new cv::Mat_<cv::Vec3f>(rows, cols, cv::Vec3f(-1, -1, -1));
            return mat;
        }

        auto* mapped = new MappedMat(rows, cols, ptr, length, fd, tmp_path);
        mapped->setTo(cv::Vec3f(-1.f, -1.f, -1.f));
        if (used_mmap) {
            *used_mmap = true;
        }
        return mapped;
    }

    void rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const Face& face, float uv_to_px) {
        for (int k = 0; k < 3; ++k) {
            if (face.v[k] < 0 || face.v[k] >= static_cast<int>(vertices.size())) {
                return;
            }
            if (face.vt[k] < 0 || face.vt[k] >= static_cast<int>(uvs.size())) {
                return;
            }
        }

        const cv::Vec3f v0 = vertices[face.v[0]].pos;
        const cv::Vec3f v1 = vertices[face.v[1]].pos;
        const cv::Vec3f v2 = vertices[face.v[2]].pos;
        
        cv::Vec2f uv0 = (uvs[face.vt[0]].coord - uv_min) * uv_to_px;
        cv::Vec2f uv1 = (uvs[face.vt[1]].coord - uv_min) * uv_to_px;
        cv::Vec2f uv2 = (uvs[face.vt[2]].coord - uv_min) * uv_to_px;

        int min_x = std::max(0, static_cast<int>(std::floor(std::min({uv0[0], uv1[0], uv2[0]}))));
        int max_x = std::min(grid_size[0] - 1, static_cast<int>(std::ceil(std::max({uv0[0], uv1[0], uv2[0]}))));
        int min_y = std::max(0, static_cast<int>(std::floor(std::min({uv0[1], uv1[1], uv2[1]}))));
        int max_y = std::min(grid_size[1] - 1, static_cast<int>(std::ceil(std::max({uv0[1], uv1[1], uv2[1]}))));
        
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                cv::Vec2f p(static_cast<float>(x), static_cast<float>(y));
                cv::Vec3f bary = computeBarycentric(p, uv0, uv1, uv2);
                
                const float eps = -1e-4f;
                if (bary[0] >= eps && bary[1] >= eps && bary[2] >= eps) {
                    cv::Vec3f pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2;
                    
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
        
        float denom = dot00 * dot11 - dot01 * dot01;
        if (std::abs(denom) < 1e-20f || !std::isfinite(denom)) {
            return {-1.f, -1.f, -1.f};
        }

        float invDenom = 1.0f / denom;
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        
        return cv::Vec3f(1.0f - u - v, v, u);
    }
};

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 5) {
        std::cout << "usage: " << argv[0] << " <input.obj> <output_directory> [mesh_units] [uv_pixels_per_unit]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format using UV-grid projection." << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  mesh_units        : micrometers per OBJ unit (default: 1.0)" << std::endl;
        std::cout << "  uv_pixels_per_unit: UV pixel density (default: 20.0)" << std::endl;
        std::cout << std::endl;
        std::cout << "Note: UV parameterization is rasterized with barycentric interpolation; no stretch factor is needed." << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir 1.0 20" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    std::filesystem::path output_dir = argv[2];
    float mesh_units = 1.0f;  // mesh units in micrometers
    float uv_pixels_per_unit = 20.0f;
    
    if (argc >= 4) {
        mesh_units = std::atof(argv[3]);
        if (mesh_units <= 0) {
            std::cerr << "Invalid mesh units: " << mesh_units << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (argc >= 5) {
        uv_pixels_per_unit = std::atof(argv[4]);
        if (uv_pixels_per_unit <= 0) {
            std::cerr << "Invalid UV pixel density: " << uv_pixels_per_unit << std::endl;
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
    std::cout << "Mesh units: " << mesh_units << " micrometers" << std::endl;
    std::cout << "UV pixels per unit: " << uv_pixels_per_unit << std::endl;
    
    ObjToTifxyzConverter converter;
    
    // Load OBJ file
    if (!converter.loadObj(obj_path.string())) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Create quad surface directly from UV grid projection
    QuadSurface* surf = converter.createQuadSurface(mesh_units, uv_pixels_per_unit);
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
