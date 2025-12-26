#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input-file", po::value<std::string>(), "input tifxyz file")
        ("rotate,r", po::value<float>()->required(), "Rotate the point grid by a given angle in degrees.")
        ("paths,p", po::value<std::vector<std::string>>()->multitoken(), "Path arguments (currently unused).");

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("paths", -1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

        if (vm.count("help")) {
            std::cout << "usage: " << argv[0] << " <tifxyz> -r/--rotate angle_deg [-p/--paths ...]\n" << desc << std::endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << "usage: " << argv[0] << " <tifxyz> -r/--rotate angle_deg [-p/--paths ...]\n" << desc << std::endl;
        return EXIT_FAILURE;
    }

    if (!vm.count("input-file")) {
        std::cerr << "Error: No input tiffxyz file specified." << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path input_path = vm["input-file"].as<std::string>();
    float rotation_angle = vm["rotate"].as<float>();

    // Load the surface
    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(input_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading tifxyz file: " << input_path << " - " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Apply rotation using QuadSurface::rotate() which handles points and channels
    std::cout << "Rotating surface by " << rotation_angle << " degrees..." << std::endl;
    surf->rotate(rotation_angle);

    // Generate output filename
    float normalized_angle = fmod(rotation_angle, 360.0f);
    if (normalized_angle < 0) normalized_angle += 360.0f;

    std::string angle_str = std::to_string(static_cast<int>(normalized_angle));
    std::filesystem::path output_path = input_path.parent_path() / (input_path.stem().string() + "_r" + angle_str + input_path.extension().string());

    // Recalculate and update the surface area
    double area = vc::surface::computeSurfaceAreaVox2(surf->rawPoints());
    if (!surf->meta) {
        surf->meta = std::make_unique<nlohmann::json>();
    }
    (*surf->meta)["area"] = area;

    surf->save(output_path, true);
    std::cout << "Saved rotated surface to: " << output_path << std::endl;

    return EXIT_SUCCESS;
}
