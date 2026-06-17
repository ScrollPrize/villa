#include "vc/atlas/AtlasConstraints.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <locale>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path atlasDir;
    fs::path projectVolpkgJson;
    fs::path volpkgRoot;
    fs::path fiberPathRoot;
    fs::path lasagnaManifest;
    fs::path output;
    vc::atlas::AtlasConstraintExportOptions exportOptions;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " <atlas_dir> [project.volpkg.json] --output <pointcollections.json>\n\n"
        << "Exports atlas line and cross-winding constraints as VC3D point collections.\n"
        << "Options:\n"
        << "  --output PATH                         Output point collection JSON (required)\n"
        << "  --lasagna-manifest PATH               Override selected Lasagna dataset\n"
        << "  --fiber-path PATH                     Base directory for atlas fiber JSON paths\n"
        << "  --debug-image PATH                    Write one debug image; extension selects format\n"
        << "  --dbg-dir DIR                         Write aggregate and per-constraint LZW TIFFs\n"
        << "  --line-max-step W                     Max on-fiber point spacing in windings (default 0.25)\n"
        << "  --cross-target W                      Cross-winding target distance (default 1.0)\n"
        << "  --cross-tolerance W                   Cross-winding tolerance (default 0.1)\n"
        << "  --cross-z-threshold Z                 Cross candidate z threshold (default 25.0)\n"
        << "  --close-min-signed-winding W          Cycle-close signed winding min (default -0.5)\n"
        << "  --close-max-signed-winding W          Cycle-close signed winding max (default 0.0)\n"
        << "  --close-atlas-winding-threshold W     Cycle-close atlas winding diff threshold (default 0.1)\n"
        << "  --greedy-beam-width N                 Line cover beam width (default 32)\n"
        << "  --no-cycle-close                      Disable temporary cycle-closing links\n"
        << "  --no-lines                            Do not export line constraints\n"
        << "  --no-cross                            Do not export cross-winding constraints\n";
}

std::string optionValue(const std::string& arg, const std::string& name, int& i, int argc, char** argv)
{
    const std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
    }
    if (arg == name) {
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value after " + name);
        }
        return argv[++i];
    }
    return {};
}

bool readDoubleOption(const std::string& arg,
                      const std::string& name,
                      int& i,
                      int argc,
                      char** argv,
                      double& out)
{
    const auto value = optionValue(arg, name, i, argc, argv);
    if (value.empty()) {
        return false;
    }
    out = std::stod(value);
    return true;
}

bool readSizeOption(const std::string& arg,
                    const std::string& name,
                    int& i,
                    int argc,
                    char** argv,
                    size_t& out)
{
    const auto value = optionValue(arg, name, i, argc, argv);
    if (value.empty()) {
        return false;
    }
    out = static_cast<size_t>(std::stoull(value));
    return true;
}

Options parseArgs(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--no-cycle-close") {
            options.exportOptions.closeCycles = false;
            continue;
        }
        if (arg == "--no-lines") {
            options.exportOptions.exportLineConstraints = false;
            continue;
        }
        if (arg == "--no-cross") {
            options.exportOptions.exportCrossWindingConstraints = false;
            continue;
        }
        if (auto value = optionValue(arg, "--output", i, argc, argv); !value.empty()) {
            options.output = value;
            continue;
        }
        if (auto value = optionValue(arg, "--lasagna-manifest", i, argc, argv); !value.empty()) {
            options.lasagnaManifest = value;
            continue;
        }
        if (auto value = optionValue(arg, "--fiber-path", i, argc, argv); !value.empty()) {
            options.fiberPathRoot = value;
            continue;
        }
        if (auto value = optionValue(arg, "--debug-image", i, argc, argv); !value.empty()) {
            options.exportOptions.debugImagesDir = value;
            continue;
        }
        if (auto value = optionValue(arg, "--dbg-dir", i, argc, argv); !value.empty()) {
            options.exportOptions.debugDirectory = value;
            continue;
        }
        if (auto value = optionValue(arg, "--debug-images-dir", i, argc, argv); !value.empty()) {
            options.exportOptions.debugDirectory = value;
            continue;
        }
        if (readDoubleOption(arg, "--line-max-step", i, argc, argv, options.exportOptions.lineMaxWindingStep) ||
            readDoubleOption(arg, "--cross-target", i, argc, argv, options.exportOptions.crossWindingTarget) ||
            readDoubleOption(arg, "--cross-tolerance", i, argc, argv, options.exportOptions.crossWindingTolerance) ||
            readDoubleOption(arg, "--cross-z-threshold", i, argc, argv, options.exportOptions.crossZThreshold) ||
            readDoubleOption(arg, "--close-min-signed-winding", i, argc, argv, options.exportOptions.closeMinSignedWinding) ||
            readDoubleOption(arg, "--close-max-signed-winding", i, argc, argv, options.exportOptions.closeMaxSignedWinding) ||
            readDoubleOption(arg, "--close-atlas-winding-threshold", i, argc, argv, options.exportOptions.closeAtlasWindingThreshold) ||
            readSizeOption(arg, "--greedy-beam-width", i, argc, argv, options.exportOptions.greedyBeamWidth)) {
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (options.atlasDir.empty()) {
            options.atlasDir = arg;
        } else if (options.projectVolpkgJson.empty()) {
            options.projectVolpkgJson = arg;
        } else {
            throw std::invalid_argument("too many positional arguments");
        }
    }
    if (options.atlasDir.empty()) {
        throw std::invalid_argument("missing atlas_dir");
    }
    if (options.output.empty()) {
        throw std::invalid_argument("missing --output");
    }
    if (!options.projectVolpkgJson.empty() &&
        (!fs::is_regular_file(options.projectVolpkgJson) ||
         options.projectVolpkgJson.filename().string().find(".volpkg.json") == std::string::npos)) {
        throw std::invalid_argument("second positional argument must be a *.volpkg.json file when provided");
    }
    return options;
}

void resolveProjectContext(Options& options)
{
    if (options.projectVolpkgJson.empty()) {
        if (options.lasagnaManifest.empty() && options.exportOptions.closeCycles) {
            throw std::runtime_error(
                "missing Lasagna manifest; pass --lasagna-manifest, provide project.volpkg.json, or use --no-cycle-close");
        }
        return;
    }
    const auto pkg = VolumePkg::load(options.projectVolpkgJson);
    if (!pkg) {
        throw std::runtime_error("failed to load project file: " +
                                 options.projectVolpkgJson.string());
    }
    options.volpkgRoot = fs::path(pkg->getVolpkgDirectory());
    if (options.volpkgRoot.empty()) {
        throw std::runtime_error("failed to resolve volpkg root from " +
                                 options.projectVolpkgJson.string());
    }
    if (options.lasagnaManifest.empty()) {
        options.lasagnaManifest = pkg->selectedLasagnaDatasetPath();
    }
    if (options.lasagnaManifest.empty() && options.exportOptions.closeCycles) {
        throw std::runtime_error(
            "missing Lasagna manifest; set selected Lasagna dataset in project, pass --lasagna-manifest, or use --no-cycle-close");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::cout.imbue(std::locale::classic());
        Options options = parseArgs(argc, argv);
        resolveProjectContext(options);

        const auto exportData =
            vc::atlas::loadLasagnaAtlasExport(
                options.atlasDir,
                options.volpkgRoot,
                options.fiberPathRoot);
        QuadSurface baseSurface(exportData.basePath);

        std::unique_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
        std::optional<vc::lasagna::LasagnaDataset> dataset;
        if (options.exportOptions.closeCycles) {
            dataset = vc::lasagna::LasagnaDataset::open(options.lasagnaManifest);
            sampler = std::make_unique<vc::lasagna::LasagnaNormalSampler>(*dataset);
        }

        auto result = vc::atlas::exportAtlasConstraints(
            exportData,
            &baseSurface,
            sampler.get(),
            options.exportOptions);
        if (!result.collections.saveToJSON(options.output.string())) {
            throw std::runtime_error("failed to write " + options.output.string());
        }

        std::cout << "wrote " << options.output << '\n'
                  << "atlas_fibers=" << result.report.atlasFibers
                  << " source_links=" << result.report.sourceLinks
                  << " temporary_links=" << result.report.temporaryLinks
                  << " line_collections=" << result.report.lineCollections
                  << " line_points=" << result.report.linePoints
                  << " cross_collections=" << result.report.crossCollections
                  << " cross_points=" << result.report.crossPoints
                  << " debug_images=" << result.report.debugImagesWritten
                  << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "vc_atlas_constraints_export: " << ex.what() << '\n';
        return 1;
    }
}
