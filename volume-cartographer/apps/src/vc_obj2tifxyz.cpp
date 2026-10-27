#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreSurfaces.hpp"
#include "vc/core/util/SurfaceMeshing.hpp"
#include "vc/core/util/Slicing.hpp"
#include <vc_object_store/Uuid.hpp>
#include <cctype>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>

// Helpers for flag parsing
static bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}

// The vertices, texture coordinates and triangles of an OBJ file.
struct ObjMesh {
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec2f> uvs;
    std::vector<vc::meshing::UvFace> faces;

    bool load(const std::string& filename) {
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
};

// The per-vertex grid-UV sidecar (<obj>.griduv: "col row" per line, in OBJ
// vertex order) written by vc_tifxyz2obj; empty if absent.
static std::vector<cv::Vec2f> loadGridUV(const std::string& obj_filename) {
    std::vector<cv::Vec2f> grid_uv;
    std::ifstream file(obj_filename + ".griduv");
    float col = 0, row = 0;
    while (file >> col >> row) {
        grid_uv.emplace_back(col, row);
    }
    return grid_uv;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "usage: " << argv[0]
                  << " <input.obj> <output_directory | --store=<store>> [stretch_factor] [mesh_units]"
                  << " [--uv-metric] [--uv-to-obj=<ratio>] [--uv-downsample=<f>]"
                  << " [--grid-cap=<pixels>] [--uuid=<id>]" << std::endl;
        std::cout << "Converts an OBJ file to tifxyz format" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  stretch_factor: UV scaling factor (default: 1.0)" << std::endl;
        std::cout << "  mesh_units    : micrometers per OBJ unit (default: 1.0)" << std::endl;
        std::cout << "Flags:" << std::endl;
        std::cout << "  --store=<store>     : write a new surface object instead of a directory (prints surface_uuid:)," << std::endl;
        std::cout << "                        with --name=<name>, --tag=<tag> (repeatable), --volume-uuid=<id> and" << std::endl;
        std::cout << "                        --source-surface=<uuid> in place of --tifxyz-source" << std::endl;
        std::cout << "  --uv-metric         : UVs are metric (default; UV units == OBJ units unless --uv-to-obj is set)" << std::endl;
        std::cout << "  --uv-non-metric     : Revert to legacy behavior (measure scale from 3D mesh)" << std::endl;
        std::cout << "  --uv-to-obj=<ratio> : OBJ units per 1 UV unit (default: 1.0). Only used with --uv-metric." << std::endl;
        std::cout << "  --uv-downsample=<f> : Uniform UV decimation factor (>=1.0). Reduces grid by ~f^2." << std::endl;
        std::cout << "  --grid-cap=<pixels> : Upper bound on total grid pixels. Implies extra decimation if needed." << std::endl;
        std::cout << "  --uuid=<id>         : Metadata UUID. Defaults to a fresh random UUID." << std::endl;
        std::cout << "  --tifxyz-source=<dir>: Original tifxyz being flattened. Its meta.json scale sizes the" << std::endl;
        std::cout << "                        output grid to the input sampling density (output scale == input" << std::endl;
        std::cout << "                        scale). If <dir>/approval.tif exists, it is resampled onto the new" << std::endl;
        std::cout << "                        grid via the <input.obj>.griduv sidecar. Absent: legacy metric sizing." << std::endl;
        std::cout << std::endl;
        std::cout << "Note: Scale factors are automatically calculated from the mesh grid structure." << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir                       (legacy behavior)" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir 800 1.0 --uv-metric  (UV is metric, OBJ units == UV units)" << std::endl;
        std::cout << "  " << argv[0] << " mesh.obj outdir --uv-metric --uv-to-obj=0.001" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path obj_path = argv[1];
    // The output is a directory, or a store given as --store=PATH.
    const bool output_positional = std::string(argv[2]).rfind("--", 0) != 0;
    std::filesystem::path output_dir = output_positional ? argv[2] : "";
    StoreOutputArgs store_output;
    std::string source_surface;
    float stretch_factor = 1.0f;
    float mesh_units = 1.0f;      // micrometers per OBJ unit
    bool  uv_metric = true;       // treat UV as metric parametrization
    float uv_to_obj = 1.0f;       // OBJ units per UV unit (only when uv_metric)
    float uv_downsample = 1.0f;
    uint64_t grid_cap = 0;
    std::string tifxyz_source;  // original tifxyz dir: provides scale + approval mask
    std::string uuid_override;

    // Backward-compatible parsing:
    // positional numbers: stretch_factor, mesh_units
    // optional flags: --uv-metric (no-op default)  --uv-non-metric  --uv-to-obj=<ratio>
    int consumed_numbers = 0;
    for (int i = output_positional ? 3 : 2; i < argc; ++i) {
        std::string a = argv[i];
        if (store_output.parse(a)) continue;
        if (starts_with(a, "--source-surface=")) {
            source_surface = a.substr(std::string("--source-surface=").size());
            continue;
        }
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
        if (starts_with(a, "--tifxyz-source=")) {
            tifxyz_source = a.substr(std::string("--tifxyz-source=").size());
            continue;
        }
        if (starts_with(a, "--uuid=")) {
            uuid_override = a.substr(std::string("--uuid=").size());
            if (uuid_override.empty()) {
                std::cerr << "Invalid value for --uuid (must be non-empty)\n";
                return EXIT_FAILURE;
            }
            continue;
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

    if (output_dir.empty() == store_output.store.empty() ||
        (store_output.store.empty() && (!store_output.name.empty() || !store_output.tags.empty() ||
                                        store_output.volume || !source_surface.empty()))) {
        std::cerr << "Give an output directory or --store=<store>; --name, --tag, --volume-uuid and --source-surface need --store\n";
        return EXIT_FAILURE;
    }
    if (!store_output.store.empty() && (!uuid_override.empty() || !tifxyz_source.empty())) {
        std::cerr << "In store mode the source is --source-surface=<uuid> and the store assigns the UUID\n";
        return EXIT_FAILURE;
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

    vc::meshing::UvGridRasterizer converter;
    converter.setUVMetric(uv_metric);
    converter.setUVToObj(uv_to_obj);
    converter.setUVDownsample(uv_downsample);
    converter.setGridCapPixels(grid_cap);

    // Load OBJ file
    ObjMesh mesh;
    if (!mesh.load(obj_path.string())) {
        std::cerr << "Failed to load OBJ file" << std::endl;
        return EXIT_FAILURE;
    }
    const std::size_t vertex_count = mesh.vertices.size();
    converter.setMesh(std::move(mesh.vertices), std::move(mesh.uvs), std::move(mesh.faces));

    // Source tifxyz (optional): provides the scale that sizes the output grid
    // to the input sampling density, and (if present) the approval mask to
    // resample. Absent -> legacy metric sizing, no approval channel.
    utils::Json sourceMeta = utils::Json::object();
    if (!tifxyz_source.empty() || !source_surface.empty()) {
        try {
            std::unique_ptr<QuadSurface> src(source_surface.empty()
                ? load_quad_from_tifxyz(tifxyz_source)
                : StoreSurfaces(store_output.store, "vc_obj2tifxyz").load(source_surface));
            if (src->meta.is_object())
                sourceMeta = src->meta;
            const cv::Vec2f s = src->scale();
            if (s[0] > 0.f && s[1] > 0.f) {
                converter.setSrcScale(s);
                std::cout << "Source tifxyz scale: " << s[0] << ", " << s[1]
                          << " (sizing output grid to input density)\n";
            } else {
                std::cerr << "warning: source scale invalid; falling back to metric sizing\n";
            }
            // Approval resample: needs both the source mask and the per-vertex
            // grid-UV sidecar emitted by vc_tifxyz2obj. Skip cleanly if either
            // is missing.
            cv::Mat ap = src->channel("approval", SURF_CHANNEL_NORESIZE);
            if (ap.empty()) {
                std::cout << "note: source has no approval channel; nothing to resample\n";
            } else {
                auto grid_uv = loadGridUV(obj_path.string());
                const std::size_t entries = grid_uv.size();
                if (!grid_uv.empty() && !converter.setGridUV(std::move(grid_uv))) {
                    std::cerr << "warning: grid-UV sidecar has " << entries
                              << " entries but OBJ has " << vertex_count
                              << " vertices; skipping approval resample\n";
                }
                if (!converter.hasGridUV()) {
                    std::cerr << "note: no usable grid-UV sidecar; approval will not be resampled\n";
                } else {
                    std::cout << "Loaded grid-UV sidecar: " << entries << " entries\n";
                    converter.setSrcApproval(ap);
                    std::cout << "Resampling approval (" << ap.cols << "x" << ap.rows
                              << ", " << ap.channels() << "ch)\n";
                }
            }
        } catch (...) {
            std::cerr << "warning: could not load tifxyz source: " << tifxyz_source << "\n";
        }
    }

    // Determine grid dimensions from UV coordinates
    converter.determineGridDimensions(stretch_factor);
    
    // Create quad surface
    QuadSurface* surf = converter.createQuadSurface(mesh_units);
    if (!surf) {
        std::cerr << "Failed to create quad surface" << std::endl;
        return EXIT_FAILURE;
    }
    surf->meta = std::move(sourceMeta);

    // Attach the resampled approval mask so save() writes approval.tif at the
    // new grid size alongside x/y/z.tif.
    if (converter.hasApproval()) {
        surf->setChannel("approval", converter.outApproval());
        std::cout << "Attached resampled approval channel\n";
    }
    
    if (!store_output.store.empty()) {
        try {
            nlohmann::json parameters = {{"input", obj_path.filename().string()}, {"arguments", std::vector<std::string>(argv + 2, argv + argc)}};
            StoreSurfaces store(store_output.store, std::filesystem::path(argv[0]).filename().string());
            const auto uuid = store.create(*surf, store_output.name.empty() ? obj_path.stem().string() : store_output.name,
                                           parameters, source_surface.empty() ? std::vector<std::string>{} : std::vector<std::string>{source_surface}, store_output.tags, store_output.volume);
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
