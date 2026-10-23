// vc_extend_edge: grow a tifxyz segment past its edges along the sheet,
// headless, with the same edge extension and sheet walk as VC3D's Manual Add
// ("Extensions follow the sheet").
#include "segmentation/tools/AutoExtend.hpp"
#include "segmentation/tools/SheetNormalField.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include "utils/Json.hpp"

#include <boost/program_options.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace
{

std::shared_ptr<Volume> openVolume(const std::string& where)
{
    if (where.rfind("http://", 0) == 0 || where.rfind("https://", 0) == 0 || where.rfind("s3://", 0) == 0) {
        return Volume::NewFromUrl(where);
    }
    return Volume::New(fs::path(where));
}

} // namespace

int main(int argc, char** argv)
{
    po::options_description desc(
        "vc_extend_edge: grow a tifxyz segment past its edges along the sheet.\n"
        "Each round extends every row or column of each side by --cells vertices and walks\n"
        "each new line along the structure tensor sheet normals of --normals-volume. Use the\n"
        "CT volume, or a surface prediction volume of the same scan where one exists (it gave\n"
        "more vertices on the sheet in our PHerc0139 tests). Remote chunks are cached under\n"
        "the cache root (see vc_cache_root).\n\nOptions");
    desc.add_options()
        ("help,h", "show this help")
        ("input,i", po::value<std::string>()->required(), "input tifxyz segment directory")
        ("output,o", po::value<std::string>()->required(), "output tifxyz directory (a new segment)")
        ("normals-volume,n", po::value<std::string>()->required(),
         "volume the sheet normals come from: a local zarr path or an http(s)/s3 URL")
        ("sides", po::value<std::string>()->default_value("up,down,left,right"),
         "comma separated sides to grow: up, down, left, right (grid rows and columns)")
        ("cells", po::value<int>()->default_value(10), "vertices added per line per round")
        ("rounds", po::value<int>()->default_value(8), "rounds of growth")
        ("threads", po::value<int>()->default_value(0), "worker threads for the walk (0: all cores)")
        ("overwrite", po::bool_switch(), "replace --output if it exists")
        ("json", po::bool_switch(), "print the report as JSON");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << '\n';
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n" << desc << '\n';
        return 2;
    }

    segmentation::AutoExtendOptions options;
    options.cells = vm["cells"].as<int>();
    options.rounds = vm["rounds"].as<int>();
    options.threads = vm["threads"].as<int>();
    options.sides.clear();
    {
        std::stringstream list(vm["sides"].as<std::string>());
        std::string name;
        while (std::getline(list, name, ',')) {
            const auto side = segmentation::extendSideFromName(name);
            if (!side) {
                std::cerr << "error: unknown side '" << name << "'\n";
                return 2;
            }
            options.sides.push_back(*side);
        }
    }

    const fs::path input = vm["input"].as<std::string>();
    const fs::path output = vm["output"].as<std::string>();
    const bool overwrite = vm["overwrite"].as<bool>();
    if (fs::exists(output) && !overwrite) {
        std::cerr << "error: " << output << " exists (pass --overwrite to replace it)\n";
        return 2;
    }

    const auto started = std::chrono::steady_clock::now();
    std::unique_ptr<QuadSurface> surface;
    std::shared_ptr<Volume> volume;
    try {
        surface = load_quad_from_tifxyz(input.string());
        volume = openVolume(vm["normals-volume"].as<std::string>());
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    const cv::Size before = surface->rawPointsPtr()->size();

    SheetNormalField normals(volume);
    std::string status;
    const auto result = segmentation::autoExtendSurface(
        *surface, [&](const cv::Vec3f& world) { return normals.normalAt(world); }, options, &status);
    if (!result) {
        std::cerr << "error: " << status << '\n';
        return 1;
    }

    try {
        surface->saveAsNew(output, overwrite);
    } catch (const std::exception& e) {
        std::cerr << "error saving " << output << ": " << e.what() << '\n';
        return 1;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const cv::Size after = surface->rawPointsPtr()->size();

    if (vm["json"].as<bool>()) {
        auto report = utils::Json::object();
        report["input"] = input.string();
        report["output"] = output.string();
        report["normals_volume"] = vm["normals-volume"].as<std::string>();
        report["cells"] = options.cells;
        report["rounds"] = options.rounds;
        report["grid_before"] = utils::Json::array();
        report["grid_before"].push_back(before.height);
        report["grid_before"].push_back(before.width);
        report["grid_after"] = utils::Json::array();
        report["grid_after"].push_back(after.height);
        report["grid_after"].push_back(after.width);
        report["added"] = result->added;
        report["seconds"] = seconds;
        auto steps = utils::Json::array();
        for (const auto& step : result->steps) {
            auto s = utils::Json::object();
            s["round"] = step.round;
            s["side"] = segmentation::extendSideName(step.side);
            s["lines"] = step.lines;
            s["added"] = step.added;
            s["steps_without_normal"] = step.stepsWithoutNormal;
            if (!step.skipped.empty()) {
                s["skipped"] = step.skipped;
            }
            steps.push_back(s);
        }
        report["steps"] = steps;
        std::cout << report.dump(2) << '\n';
    } else {
        for (const auto& step : result->steps) {
            std::cout << "round " << step.round << ' ' << segmentation::extendSideName(step.side) << ": ";
            if (!step.skipped.empty()) {
                std::cout << "skipped (" << step.skipped << ")\n";
            } else {
                std::cout << step.added << " vertices on " << step.lines << " lines, "
                          << step.stepsWithoutNormal << " steps without a normal\n";
            }
        }
        std::cout << "grid " << before.height << "x" << before.width << " -> " << after.height << "x"
                  << after.width << ", " << result->added << " vertices added in " << seconds << " s\n"
                  << "saved " << output.string() << '\n';
    }
    return 0;
}
