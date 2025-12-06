#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"

#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>


using json = nlohmann::json;

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};


std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d%H%M%S");
    ss << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

QuadSurface* load_surface(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Surface path does not exist: " + path.string());
    }

    std::filesystem::path meta_path = path / "meta.json";
    if (!std::filesystem::exists(meta_path)) {
        throw std::runtime_error("No meta.json found at: " + path.string());
    }

    std::ifstream meta_f(meta_path);
    json meta = json::parse(meta_f);

    std::string format = meta.value("format", "unknown");

    if (format == "tifxyz") {
        return load_quad_from_tifxyz(path.string()).release();
    } else {
        throw std::runtime_error("Unknown surface format: " + format);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5 || argc > 7) {
        std::cout << "Usage: " << argv[0] << " <surface-a> <surface-b> <operation> <output-name> [tolerance] [params.json]" << std::endl;
        std::cout << std::endl;
        std::cout << "Operations:" << std::endl;
        std::cout << "  diff         - Returns points in surface-a that are not in surface-b" << std::endl;
        std::cout << "  union        - Combines points from both surfaces" << std::endl;
        std::cout << "  intersection - Returns only points that exist in both surfaces" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  output-name  - Base name for output (will append operation and timestamp)" << std::endl;
        std::cout << "  tolerance    - Distance threshold for considering points as same (default: 2.0)" << std::endl;
        std::cout << "  params.json  - Additional parameters file (optional)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  " << argv[0] << " ./seg1 ./seg2 diff ./output_seg 2.0" << std::endl;
        std::cout << "  Creates: ./output_seg_diff_20250817144348966/" << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path surface_a_path = argv[1];
    std::filesystem::path surface_b_path = argv[2];
    std::string operation = argv[3];
    std::filesystem::path output_base = argv[4];

    float tolerance = 2.0;
    if (argc >= 6) {
        tolerance = std::stof(argv[5]);
    }

    json params;
    if (argc == 7) {
        std::filesystem::path params_path = argv[6];
        if (std::filesystem::exists(params_path)) {
            std::ifstream params_f(params_path);
            params = json::parse(params_f);
        }
    }

    // Validate operation
    if (operation != "diff" && operation != "union" && operation != "intersection") {
        std::cerr << "Error: Invalid operation '" << operation << "'" << std::endl;
        std::cerr << "Valid operations are: diff, union, intersection" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        // Load surface A
        std::cout << "Loading surface A from: " << surface_a_path << std::endl;
        QuadSurface* surf_a = nullptr;
        {
            MeasureLife timer("Loading surface A");
            surf_a = load_surface(surface_a_path);
        }

        if (!surf_a) {
            std::cerr << "Error: Failed to load surface A" << std::endl;
            return EXIT_FAILURE;
        }

        // Load surface B
        std::cout << "Loading surface B from: " << surface_b_path << std::endl;
        QuadSurface* surf_b = nullptr;
        {
            MeasureLife timer("Loading surface B");
            surf_b = load_surface(surface_b_path);
        }

        if (!surf_b) {
            std::cerr << "Error: Failed to load surface B" << std::endl;
            delete surf_a;
            return EXIT_FAILURE;
        }

        // Print surface info
        std::cout << "Surface A: " << surf_a->size().width << "x" << surf_a->size().height
                  << " scale: [" << surf_a->scale()[0] << ", " << surf_a->scale()[1] << "]" << std::endl;
        std::cout << "Surface B: " << surf_b->size().width << "x" << surf_b->size().height
                  << " scale: [" << surf_b->scale()[0] << ", " << surf_b->scale()[1] << "]" << std::endl;

        // Perform the operation
        std::unique_ptr<QuadSurface> result;
        std::cout << "Performing " << operation << " operation with tolerance=" << tolerance << std::endl;

        {
            MeasureLife timer("Computing " + operation);

            if (operation == "diff") {
                result = surface_diff(surf_a, surf_b, tolerance);
            } else if (operation == "union") {
                result = surface_union(surf_a, surf_b, tolerance);
            } else if (operation == "intersection") {
                result = surface_intersection(surf_a, surf_b, tolerance);
            }
        }

        if (!result) {
            std::cerr << "Error: Operation failed" << std::endl;
            delete surf_a;
            delete surf_b;
            return EXIT_FAILURE;
        }

        // Prepare metadata
        if (!result->meta) {
            result->meta = std::make_unique<json>();
        }

        // Generate output path - append operation and timestamp to the base name
        std::string timestamp = get_timestamp();
        std::string uuid = operation + "_" + timestamp;

        // Build the full output path
        std::filesystem::path output_path;
        if (output_base.has_parent_path()) {
            // If output_base has a path component, use it
            output_path = output_base.parent_path() / (output_base.filename().string() + "_" + uuid);
        } else {
            // If it's just a name, use current directory
            output_path = output_base.string() + "_" + uuid;
        }

        // Save the result
        {
            MeasureLife timer("Saving result");
            result->save(output_path, uuid);
        }

        std::cout << "Result saved to: " << output_path << std::endl;

        // Print statistics
        std::cout << "Result surface contains " << result->countValidPoints() << " valid points" << std::endl;

        if (result->meta) {
            const double area_vx2 = vc::surface::computeSurfaceAreaVox2(result->rawPoints());
            (*result->meta)["area_vx2"] = area_vx2;

            if (params.contains("voxelsize")) {
                const double voxelsize = params["voxelsize"];
                if (std::isfinite(voxelsize) && voxelsize > 0.0) {
                    const double area_cm2 = area_vx2 * voxelsize * voxelsize / 1e8;
                    (*result->meta)["area_cm2"] = area_cm2;
                    std::cout << "Area: " << area_cm2 << " cm²" << std::endl;
                }
            }
        }

        // Update metadata with final statistics
        result->save_meta();

        std::cout << "Operation completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
