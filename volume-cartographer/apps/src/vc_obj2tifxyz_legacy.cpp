#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreSurfaces.hpp"
#include "vc/core/util/SurfaceMeshing.hpp"
#include <vc_object_store/Uuid.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>



using vc::meshing::StepUvRasterizer;
using ScaleMethod = StepUvRasterizer::ScaleMethod;

namespace {
bool loadObj(const std::string& filename, std::vector<cv::Vec3f>& vertices, std::vector<cv::Vec2f>& uvs,
             std::vector<vc::meshing::UvFace>& faces)
{
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
            cv::Vec3f v;
            iss >> v[0] >> v[1] >> v[2];
            vertices.push_back(v);
        }
        else if (prefix == "vt") {
            cv::Vec2f uv;
            iss >> uv[0] >> uv[1];
            uvs.push_back(uv);
        }
        else if (prefix == "f") {
            vc::meshing::UvFace face;
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
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " <input.obj> <output_directory | --store=<store>> [step_size] [--uuid=<id>]"
                  << " [--scale-method=median|mean | --uv-scale=<s>[,<sv>]]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  step_size: UV units per grid cell (default: 20)" << std::endl;
        std::cout << "             Scale will be 1/step_size (default: 0.05)" << std::endl;
        std::cout << "  --uuid=<id>: Metadata UUID. Defaults to a fresh random UUID." << std::endl;
        std::cout << "  --store=<store>: write a new surface object instead of a directory (prints surface_uuid:)," << std::endl;
        std::cout << "             with --name=<name>, --tag=<tag> (repeatable) and --volume-uuid=<id>" << std::endl;
        std::cout << "  --scale-method=median|mean: how the UV-to-3D scale (voxels per UV unit) that sizes" << std::endl;
        std::cout << "             the grid is estimated from the mesh: area weighted median of the per" << std::endl;
        std::cout << "             triangle Jacobian norms (default, robust to a few stretched triangles)" << std::endl;
        std::cout << "             or area weighted mean (the behaviour before the median)" << std::endl;
        std::cout << "  --uv-scale=<s> or <su>,<sv>: give the UV-to-3D scale instead of estimating it" << std::endl;
        std::cout << "             (manual method; for U and V, or one value for both)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir" << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir 10" << std::endl;
        std::cout << "Example: " << argv[0] << " mesh.obj output_dir 4 --uv-scale=0.2" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    // The output is a directory, or a store given as --store=PATH.
    const bool output_positional = std::string(argv[2]).rfind("--", 0) != 0;
    std::filesystem::path output_dir = output_positional ? argv[2] : "";
    StoreOutputArgs store_output;
    float step_size = 20.0f;
    std::string uuid_override;
    ScaleMethod scale_method = ScaleMethod::Median;
    bool method_given = false, uv_scale_given = false;
    cv::Vec2d manual_scale(1.0, 1.0);

    bool consumed_step_size = false;
    for (int i = output_positional ? 3 : 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (store_output.parse(arg)) continue;
        if (arg.rfind("--uuid=", 0) == 0) {
            uuid_override = arg.substr(std::string("--uuid=").size());
            if (uuid_override.empty()) {
                std::cerr << "Invalid value for --uuid (must be non-empty)" << std::endl;
                return EXIT_FAILURE;
            }
            continue;
        }
        if (arg.rfind("--scale-method=", 0) == 0) {
            const std::string m = arg.substr(std::string("--scale-method=").size());
            if (m == "median") scale_method = ScaleMethod::Median;
            else if (m == "mean") scale_method = ScaleMethod::Mean;
            else {
                std::cerr << "Invalid value for --scale-method (median or mean): " << m << std::endl;
                return EXIT_FAILURE;
            }
            method_given = true;
            continue;
        }
        if (arg.rfind("--uv-scale=", 0) == 0) {
            const std::string v = arg.substr(std::string("--uv-scale=").size());
            const auto comma = v.find(',');
            char* end_u = nullptr;
            char* end_v = nullptr;
            const std::string su = v.substr(0, comma);
            const std::string sv = (comma == std::string::npos) ? su : v.substr(comma + 1);
            manual_scale[0] = std::strtod(su.c_str(), &end_u);
            manual_scale[1] = std::strtod(sv.c_str(), &end_v);
            if (su.empty() || sv.empty() || *end_u != '\0' || *end_v != '\0' ||
                !(manual_scale[0] > 0) || !(manual_scale[1] > 0) ||
                !std::isfinite(manual_scale[0]) || !std::isfinite(manual_scale[1])) {
                std::cerr << "Invalid value for --uv-scale (one or two positive numbers): " << v << std::endl;
                return EXIT_FAILURE;
            }
            uv_scale_given = true;
            continue;
        }
        if (consumed_step_size) {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return EXIT_FAILURE;
        }
        step_size = std::atof(arg.c_str());
        if (step_size <= 0) {
            std::cerr << "Invalid step size: " << arg << std::endl;
            return EXIT_FAILURE;
        }
        consumed_step_size = true;
    }

    if (output_dir.empty() == store_output.store.empty() ||
        (store_output.store.empty() && (!store_output.name.empty() || !store_output.tags.empty() || store_output.volume))) {
        std::cerr << "Give an output directory or --store=<store>; --name, --tag and --volume-uuid need --store" << std::endl;
        return EXIT_FAILURE;
    }
    if (!store_output.store.empty() && !uuid_override.empty()) {
        std::cerr << "--uuid names tifxyz output; stored surfaces get their UUID from the store" << std::endl;
        return EXIT_FAILURE;
    }

    if (method_given && uv_scale_given) {
        std::cerr << "--scale-method and --uv-scale are exclusive" << std::endl;
        return EXIT_FAILURE;
    }
    if (uv_scale_given) scale_method = ScaleMethod::Manual;

    if (!std::filesystem::exists(obj_path)) {
        std::cerr << "Input file does not exist: " << obj_path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Converting OBJ to tifxyz format" << std::endl;
    std::cout << "Input: " << obj_path << std::endl;
    std::cout << "Output: " << output_dir << std::endl;
    std::cout << "Step size: " << step_size << " (scale: " << (1.0f / step_size) << ")" << std::endl;

    StepUvRasterizer converter;
    {
        std::vector<cv::Vec3f> vertices;
        std::vector<cv::Vec2f> uvs;
        std::vector<vc::meshing::UvFace> faces;
        if (!loadObj(obj_path.string(), vertices, uvs, faces)) {
            std::cerr << "Failed to load OBJ file" << std::endl;
            return EXIT_FAILURE;
        }
        converter.setMesh(std::move(vertices), std::move(uvs), std::move(faces));
    }

    // Determine grid dimensions from UV coordinates
    converter.determineGridDimensions(step_size, scale_method, manual_scale);

    // Create quad surface
    QuadSurface* surf = converter.createQuadSurface();
    if (!surf) {
        std::cerr << "Failed to create quad surface" << std::endl;
        return EXIT_FAILURE;
    }
    
    if (!store_output.store.empty()) {
        try {
            nlohmann::json parameters = {{"input", obj_path.filename().string()}, {"arguments", std::vector<std::string>(argv + 2, argv + argc)}};
            StoreSurfaces store(store_output.store, std::filesystem::path(argv[0]).filename().string());
            const auto uuid = store.create(*surf, store_output.name.empty() ? obj_path.stem().string() : store_output.name,
                                           parameters, {}, store_output.tags, store_output.volume);
            store.finish();
            std::cout << "surface_uuid: " << uuid << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error storing surface: " << e.what() << std::endl;
            delete surf;
            return EXIT_FAILURE;
        }
        delete surf;
        return EXIT_SUCCESS;
    }

    // A converted mesh is a new object: fresh UUID unless --uuid names one.
    surf->meta["uuid"] = uuid_override.empty() ? vc::objects::util::newUuid() : uuid_override;
    
    std::cout << "Saving to tifxyz format..." << std::endl;
    
    try {
        surf->save(output_dir);
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
