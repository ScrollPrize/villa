/**
 * @file vc_flatten.cpp
 * @brief ABF++ mesh flattening tool for TIFXYZ surfaces
 *
 * Computes a low-distortion 2D parameterization of a surface mesh using
 * Angle-Based Flattening (ABF++) followed by Least Squares Conformal Maps (LSCM).
 *
 * The output is a new TIFXYZ surface where the 3D positions are rearranged
 * according to the computed UV parameterization, reducing angular and area
 * distortion in rendered textures.
 */

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/ABFFlattening.hpp"

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    po::options_description desc("vc_flatten - ABF++ mesh flattening\n\nUsage");
    desc.add_options()
        ("help,h", "Show this help message")
        ("input,i", po::value<std::string>()->required(),
            "Input TIFXYZ directory")
        ("output,o", po::value<std::string>(),
            "Output TIFXYZ directory (default: <input>_flat)")
        ("iterations", po::value<int>()->default_value(10),
            "Maximum ABF++ iterations (default: 10)")
        ("downsample", po::value<int>()->default_value(1),
            "Downsample factor for faster computation (1=full, 2=half, 4=quarter)")
        ("lscm-only", po::bool_switch()->default_value(false),
            "Use only LSCM, skip ABF++ angle optimization")
        ("no-scale", po::bool_switch()->default_value(false),
            "Don't scale output to match original 3D surface area")
        ("uv-only", po::bool_switch()->default_value(false),
            "Only compute and save UV channel, don't create new surface");

    po::positional_options_description pos;
    pos.add("input", 1);
    pos.add("output", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
            .options(desc)
            .positional(pos)
            .run(), vm);

        if (vm.count("help") || argc < 2) {
            std::cout << desc << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  vc_flatten /path/to/segment" << std::endl;
            std::cout << "  vc_flatten -i segment -o segment_flat --iterations 20" << std::endl;
            std::cout << "  vc_flatten segment --lscm-only" << std::endl;
            std::cout << "  vc_flatten segment --uv-only  # Store UV channel without rasterizing" << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information." << std::endl;
        return 1;
    }

    fs::path inputPath = vm["input"].as<std::string>();
    fs::path outputPath;

    if (vm.count("output")) {
        outputPath = vm["output"].as<std::string>();
    } else {
        outputPath = inputPath.parent_path() / (inputPath.filename().string() + "_flat");
    }

    // Configure ABF++
    vc::ABFConfig config;
    config.maxIterations = static_cast<std::size_t>(vm["iterations"].as<int>());
    config.downsampleFactor = vm["downsample"].as<int>();
    config.useABF = !vm["lscm-only"].as<bool>();
    config.scaleToOriginalArea = !vm["no-scale"].as<bool>();

    bool uvOnly = vm["uv-only"].as<bool>();

    // Load surface
    std::cout << "Loading surface: " << inputPath << std::endl;
    auto surf = load_quad_from_tifxyz(inputPath.string());
    if (!surf) {
        std::cerr << "Failed to load surface: " << inputPath << std::endl;
        return 1;
    }

    std::cout << "Surface loaded:" << std::endl;
    std::cout << "  Grid size: " << surf->rawPointsPtr()->cols << " x " << surf->rawPointsPtr()->rows << std::endl;
    std::cout << "  Scale: " << surf->scale()[0] << ", " << surf->scale()[1] << std::endl;
    std::cout << "  Valid points: " << surf->countValidPoints() << std::endl;
    std::cout << "  Valid quads: " << surf->countValidQuads() << std::endl;

    std::cout << "\nFlattening configuration:" << std::endl;
    std::cout << "  Max iterations: " << config.maxIterations << std::endl;
    std::cout << "  Downsample factor: " << config.downsampleFactor << std::endl;
    std::cout << "  Use ABF++: " << (config.useABF ? "yes" : "no (LSCM only)") << std::endl;
    std::cout << "  Scale to original area: " << (config.scaleToOriginalArea ? "yes" : "no") << std::endl;
    std::cout << "  Mode: " << (uvOnly ? "UV channel only" : "create new surface") << std::endl;

    if (uvOnly) {
        // Just compute and save UV channel
        std::cout << "\nComputing UV coordinates..." << std::endl;
        if (!vc::abfFlattenInPlace(*surf, config)) {
            std::cerr << "Flattening failed" << std::endl;
            return 1;
        }

        // Save the original surface with UV channel
        std::cout << "Saving surface with UV channel to: " << outputPath << std::endl;
        surf->save(outputPath.string(), outputPath.filename().string(), true);
    } else {
        // Create a new surface with rearranged positions
        std::cout << "\nFlattening mesh..." << std::endl;
        QuadSurface* flatSurf = vc::abfFlattenToNewSurface(*surf, config);
        if (!flatSurf) {
            std::cerr << "Flattening failed" << std::endl;
            return 1;
        }

        std::cout << "\nFlattened surface:" << std::endl;
        std::cout << "  Grid size: " << flatSurf->rawPointsPtr()->cols << " x " << flatSurf->rawPointsPtr()->rows << std::endl;
        std::cout << "  Scale: " << flatSurf->scale()[0] << ", " << flatSurf->scale()[1] << std::endl;
        std::cout << "  Valid points: " << flatSurf->countValidPoints() << std::endl;
        std::cout << "  Valid quads: " << flatSurf->countValidQuads() << std::endl;

        // Save the flattened surface
        std::cout << "\nSaving flattened surface to: " << outputPath << std::endl;
        flatSurf->save(outputPath.string(), outputPath.filename().string(), true);

        delete flatSurf;
    }

    std::cout << "Done!" << std::endl;

    return 0;
}
