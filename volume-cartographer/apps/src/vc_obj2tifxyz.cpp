#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"
#include <cctype>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <limits>

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

// Helpers for flag parsing
static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}

class ObjToTifxyzConverter {
private:
    std::vector<Vertex> vertices;
    std::vector<UV> uvs;
    std::vector<Face> faces;
    
    cv::Vec2f uv_min, uv_max;
    cv::Vec2i grid_size;
    cv::Vec2f scale;
    // UV handling
    bool uv_is_metric = true;   // if true, scale comes from UV spacing
    float uv_to_obj   = 1.0f;   // OBJ units per 1 UV unit (used when uv_is_metric)
    // UV decimation / capping
    float uv_downsample = 1.0f;            // user-specified uniform decimation
    uint64_t grid_cap_pixels = 0;          // cap total pixel count (0=off)

public:
    void setUVMetric(bool v) { uv_is_metric = v; }
    void setUVToObj(float r) { uv_to_obj = r; }
    void setUVDownsample(float f) { uv_downsample = std::max(1.0f, f); }
    void setGridCapPixels(uint64_t cap) { grid_cap_pixels = cap; }

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
    
    void determineGridDimensions(float stretch_factor = 1.0f) {
        // Find UV bounds from all UVs used in faces
        uv_min = cv::Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        uv_max = cv::Vec2f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
        
        for (const auto& face : faces) {
            for (int i = 0; i < 3; i++) {
                if (face.vt[i] >= 0 && face.vt[i] < (int)uvs.size()) {
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
        
        // Build raw grid (pre-decimation) from UV range and stretch factor
        const cv::Vec2f uv_range = uv_max - uv_min;
        const int gw_raw = std::max(2, static_cast<int>(std::ceil(uv_range[0] * stretch_factor)) + 1);
        const int gh_raw = std::max(2, static_cast<int>(std::ceil(uv_range[1] * stretch_factor)) + 1);

        // Apply uniform decimation / capping to the resolution (NOT to the scale)
        const auto apply_decimation = [](int n, double d) {
            if (d <= 1.0) return n;
            // keep endpoints: 1 + floor((n-1)/d)
            const double eff = std::max(1.0, d);
            return std::max(2, static_cast<int>(std::floor((n - 1) / eff)) + 1);
        };

        double decim = std::max(1.0, static_cast<double>(uv_downsample));
        if (grid_cap_pixels > 0) {
            const long double raw_px = static_cast<long double>(gw_raw) * static_cast<long double>(gh_raw);
            if (raw_px > static_cast<long double>(grid_cap_pixels)) {
                const double need = std::sqrt(static_cast<double>(raw_px / static_cast<long double>(grid_cap_pixels)));
                decim = std::max(decim, std::ceil(need));
            }
        }
        const int gw_dec = apply_decimation(gw_raw, decim);
        const int gh_dec = apply_decimation(gh_raw, decim);
        grid_size[0] = gw_dec;
        grid_size[1] = gh_dec;

        // Effective decimation per axis (accounts for rounding at endpoints)
        const double eff_decim_x = (gw_dec > 1) ? (double)(gw_raw - 1) / (double)(gw_dec - 1) : 1.0;
        const double eff_decim_y = (gh_dec > 1) ? (double)(gh_raw - 1) / (double)(gh_dec - 1) : 1.0;

        if (uv_is_metric) {
            // --- Preserve physical pixel size (scale) from the RAW grid ---
            // du_raw/dv_raw are UV units per pixel *before* decimation.
            const float du_raw = (gw_raw > 1) ? uv_range[0] / float(gw_raw - 1) : 0.f;
            const float dv_raw = (gh_raw > 1) ? uv_range[1] / float(gh_raw - 1) : 0.f;
            scale[0] = du_raw * uv_to_obj; // OBJ units per pixel (unchanged by decimation)
            scale[1] = dv_raw * uv_to_obj;

            std::cout << "UV-metric mode: grid " << grid_size[0] << " x " << grid_size[1]
                      << "  scale(OBJ units): " << scale[0] << ", " << scale[1] << std::endl;
            if (decim > 1.0) {
                std::cout << "  (applied UV decimation ~" << decim << "x per axis; "
                          << "effective decim [" << eff_decim_x << ", " << eff_decim_y << "]; "
                          << "preserved per-pixel physical scale)\n";
            }
        } else {
            // --- Legacy non-metric: measure from 3D on decimated grid, then compensate ---
            std::cout << "Creating preliminary grid " << grid_size[0] << " x " << grid_size[1]
                      << " to measure scale from 3D..." << std::endl;

            cv::Mat_<cv::Vec3f> preliminary_points(grid_size[1], grid_size[0], cv::Vec3f(-1, -1, -1));
            for (const auto& face : faces) {
                rasterizeTriangle(preliminary_points, face);
            }
            // This fills 'scale' with the distance between adjacent samples on the DECIMATED grid
            calculateScaleFromGrid(preliminary_points);

            // Compensate by the effective decimation so that scale matches the raw-grid pixel size
            if (scale[0] > 0) scale[0] = static_cast<float>(scale[0] / eff_decim_x);
            if (scale[1] > 0) scale[1] = static_cast<float>(scale[1] / eff_decim_y);

            // Keep original behavior for the final log (do not fight it further)
            const cv::Vec2f measured = scale;
            grid_size[0] = std::max(2, static_cast<int>(std::round(measured[0] * stretch_factor)) + 1);
            grid_size[1] = std::max(2, static_cast<int>(std::round(measured[1] * stretch_factor)) + 1);

            std::cout << "Final grid: " << grid_size[0] << " x " << grid_size[1] << std::endl;
            std::cout << "Scale(OBJ units; compensated to raw-pixel spacing): "
                      << measured[0] << ", " << measured[1] << std::endl;
        }
    }
    
    QuadSurface* createQuadSurface(float mesh_units = 1.0f) {
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
        
        // Scale is currently in OBJ units. Convert to micrometers now.
        if (valid_count == 0) {
            std::cerr << "Warning: no valid grid points were rasterized." << std::endl;
        }
        if (uv_is_metric) {
            scale[0] *= mesh_units;
            scale[1] *= mesh_units;
            std::cout << "Scale from UV (micrometers): " << scale[0] << ", " << scale[1] << std::endl;
        } else {
            // Measure from 3D to preserve anisotropy and noise-robustness (already compensated in determineGridDimensions)
            calculateScaleFromGrid(*points, mesh_units);
        }
        
        return new QuadSurface(points, scale);
    }
    
    void calculateScaleFromGrid(const cv::Mat_<cv::Vec3f>& points, float mesh_units = 1.0f) {
        // Based on vc_segmentation_scales from Slicing.cpp
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
            scale[0] = static_cast<float>((sum_x / count) * mesh_units);
            scale[1] = static_cast<float>((sum_y / count) * mesh_units);
        } else {
            // Fallback to UV-based scale if we couldn't calculate from grid
            std::cerr << "Warning: Could not calculate scale from grid, using UV-based fallback" << std::endl;
            // scale already set in determineGridDimensions
        }
        
        std::cout << "Calculated scale factors from grid: " << scale[0] << ", " << scale[1] << " micrometers" << std::endl;
    }
    
private:
    void rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const Face& face) {
        // Validate indices
        for (int k = 0; k < 3; ++k) {
            if (face.v[k] < 0 || face.v[k] >= (int)vertices.size()) {
                return; // skip invalid vertex indices
            }
            if (face.vt[k] < 0 || face.vt[k] >= (int)uvs.size()) {
                return; // skip faces with missing/invalid UVs
            }
        }
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
        const float rx = std::max(uv_range[0], 1e-12f);
        const float ry = std::max(uv_range[1], 1e-12f);
        uv0 = (uv0 - uv_min);
        uv0[0] = uv0[0] / rx * (grid_size[0] - 1);
        uv0[1] = uv0[1] / ry * (grid_size[1] - 1);
        
        uv1 = (uv1 - uv_min);
        uv1[0] = uv1[0] / rx * (grid_size[0] - 1);
        uv1[1] = uv1[1] / ry * (grid_size[1] - 1);
        
        uv2 = (uv2 - uv_min);
        uv2[0] = uv2[0] / rx * (grid_size[0] - 1);
        uv2[1] = uv2[1] / ry * (grid_size[1] - 1);
        
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
        
        const float denom = (dot00 * dot11 - dot01 * dot01);
        if (std::fabs(denom) < 1e-20f || !std::isfinite(denom)) {
            // Degenerate triangle in UV space -> mark "outside"
            return cv::Vec3f(-1.f, -1.f, -1.f);
        }
        float invDenom = 1.0f / denom;
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        
        return cv::Vec3f(1.0f - u - v, v, u);
    }
};

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "usage: " << argv[0]
                  << " <input.obj> <output_directory> [stretch_factor] [mesh_units]"
                  << " [--uv-metric] [--uv-to-obj=<ratio>] [--uv-downsample=<f>]"
                  << " [--grid-cap=<pixels>]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  stretch_factor: UV scaling factor (default: 1.0)" << std::endl;
        std::cout << "  mesh_units    : micrometers per OBJ unit (default: 1.0)" << std::endl;
        std::cout << "Flags:" << std::endl;
        std::cout << "  --uv-metric         : UVs are metric (default; UV units == OBJ units unless --uv-to-obj is set)" << std::endl;
        std::cout << "  --uv-non-metric     : Revert to legacy behavior (measure scale from 3D mesh)" << std::endl;
        std::cout << "  --uv-to-obj=<ratio> : OBJ units per 1 UV unit (default: 1.0). Only used with --uv-metric." << std::endl;
        std::cout << "  --uv-downsample=<f> : Uniform UV decimation factor (>=1.0). Reduces grid by ~f^2." << std::endl;
        std::cout << "  --grid-cap=<pixels> : Upper bound on total grid pixels. Implies extra decimation if needed." << std::endl;
        std::cout << std::endl;
        std::cout << "Note: Scale factors are automatically calculated from the mesh grid structure." << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir                       (legacy behavior)" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir 800 1.0 --uv-metric  (UV is metric, OBJ units == UV units)" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir --uv-metric --uv-to-obj=0.001" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    std::filesystem::path output_dir = argv[2];
    float stretch_factor = 1.0f;
    float mesh_units = 1.0f;      // micrometers per OBJ unit
    bool  uv_metric = true;       // treat UV as metric parametrization
    float uv_to_obj = 1.0f;       // OBJ units per UV unit (only when uv_metric)
    float uv_downsample = 1.0f;
    uint64_t grid_cap = 0;

    // Backward-compatible parsing:
    // positional numbers: stretch_factor, mesh_units
    // optional flags: --uv-metric (no-op default)  --uv-non-metric  --uv-to-obj=<ratio>
    int consumed_numbers = 0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (starts_with(a, "--uv-metric")) {
            uv_metric = true;
            continue;
        }
        if (starts_with(a, "--uv-non-metric")) {
            uv_metric = false;
            continue;
        }
        if (starts_with(a, "--uv-to-obj=")) {
            try {
                uv_to_obj = std::stof(a.substr(std::string("--uv-to-obj=").size()));
                continue;
            } catch (...) {
                std::cerr << "Invalid value for --uv-to-obj\n";
                return EXIT_FAILURE;
            }
        }
        if (starts_with(a, "--uv-downsample=")) {
            try {
                uv_downsample = std::stof(a.substr(std::string("--uv-downsample=").size()));
                if (uv_downsample < 1.0f) throw 1;
                continue;
            } catch (...) {
                std::cerr << "Invalid value for --uv-downsample (must be >= 1)\n";
                return EXIT_FAILURE;
            }
        }
        if (starts_with(a, "--grid-cap=")) {
            try {
                grid_cap = static_cast<uint64_t>(std::stoll(a.substr(std::string("--grid-cap=").size())));
                continue;
            } catch (...) {
                std::cerr << "Invalid value for --grid-cap\n";
                return EXIT_FAILURE;
            }
        }
        // numbers (legacy positional)
        if (consumed_numbers == 0) {
            stretch_factor = std::atof(a.c_str());
            if (stretch_factor <= 0) {
                std::cerr << "Invalid stretch factor: " << a << std::endl;
                return EXIT_FAILURE;
            }
            consumed_numbers++;
        } else if (consumed_numbers == 1) {
            mesh_units = std::atof(a.c_str());
            if (mesh_units <= 0) {
                std::cerr << "Invalid mesh units: " << a << std::endl;
                return EXIT_FAILURE;
            }
            consumed_numbers++;
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
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
    std::cout << "Mesh units: " << mesh_units << " micrometers per OBJ unit" << std::endl;
    if (uv_metric) {
        std::cout << "UV mode: metric (default)" << std::endl;
        std::cout << "UV->OBJ scale: " << uv_to_obj << " OBJ units / UV unit" << std::endl;
    } else {
        std::cout << "UV mode: non-metric (scale measured from mesh)" << std::endl;
    }
    if (uv_downsample > 1.0f)
        std::cout << "UV downsample: " << uv_downsample << "x per axis\n";
    if (grid_cap > 0)
        std::cout << "Grid cap: " << grid_cap << " pixels (pre-rasterization)\n";

    ObjToTifxyzConverter converter;
    converter.setUVMetric(uv_metric);
    converter.setUVToObj(uv_to_obj);
    converter.setUVDownsample(uv_downsample);
    converter.setGridCapPixels(grid_cap);

    // Load OBJ file
    if (!converter.loadObj(obj_path.string())) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Determine grid dimensions from UV coordinates
    converter.determineGridDimensions(stretch_factor);
    
    // Create quad surface
    QuadSurface* surf = converter.createQuadSurface(mesh_units);
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
