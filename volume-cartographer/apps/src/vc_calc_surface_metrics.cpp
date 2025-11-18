#include "vc/ui/surface_metrics.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <opencv2/imgcodecs.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    po::options_description desc("Calculate surface metrics based on a point collection.");
    desc.add_options()
        ("help,h", "Print help")
        ("collection", po::value<std::string>(), "Input point collection file (.json)")
        ("surface", po::value<std::string>(), "Input surface file (.tif)")
        ("winding", po::value<std::string>(), "Input winding file (.tif)")
        ("output", po::value<std::string>(), "Output metrics file (.json)")
        ("z_min", po::value<int>()->default_value(-1), "Minimum slice index to consider (optional)")
        ("z_max", po::value<int>()->default_value(-1), "Maximum slice index to consider (optional)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (!vm.count("collection") || !vm.count("surface") || !vm.count("output")) {
        std::cerr << "Error: --collection, --surface, and --output are required." << std::endl;
        return 1;
    }

    std::string collection_path = vm["collection"].as<std::string>();
    std::string surface_path = vm["surface"].as<std::string>();
    std::string output_path = vm["output"].as<std::string>();
    int z_min = vm["z_min"].as<int>();
    int z_max = vm["z_max"].as<int>();

    VCCollection collection;
    if (!collection.loadFromJSON(collection_path)) {
        std::cerr << "Error: Failed to load point collection from " << collection_path << std::endl;
        return 1;
    }

    QuadSurface* surface = load_quad_from_tifxyz(surface_path);
    if (!surface) {
        std::cerr << "Error: Failed to load surface from " << surface_path << std::endl;
        return 1;
    }
    if (z_min == -1)
        z_min = (int)surface->bbox().low[2];
    if (z_max == -1)
        z_min = (int)surface->bbox().high[2];

    nlohmann::json metrics = calc_point_metrics(collection, surface, z_min, z_max);

    if (vm.count("winding")) {
        std::string winding_path = vm["winding"].as<std::string>();
        cv::Mat_<float> winding = cv::imread(winding_path, cv::IMREAD_UNCHANGED);
        if (winding.empty()) {
            std::cerr << "Error: Failed to load winding from " << winding_path << std::endl;
            return 1;
        }
        nlohmann::json winding_metrics = calc_point_winding_metrics(collection, surface, winding, z_min, z_max);
        metrics.update(winding_metrics);
    }

    delete surface;

    std::ofstream o(output_path);
    if (!o.is_open()) {
        std::cerr << "Error: Failed to open output file " << output_path << std::endl;
        return 1;
    }

    o << metrics.dump(4);
    o.close();

    std::cout << "Successfully calculated metrics and saved to " << output_path << std::endl;

    return 0;
}