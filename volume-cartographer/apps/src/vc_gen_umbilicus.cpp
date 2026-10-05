// vc_gen_umbilicus: build a scroll's umbilicus by running align_and_extract_umbilicus over the
// xy slices of a normal grid volume.
//
// One xy grid slice is one cross section of the scroll, and align_and_extract_umbilicus reduces
// it to the single point the sheet normals of that section point away from. This tool runs that
// function, the one in core/src/normalgridtools.cpp and not a copy of it, over a set of slices
// and writes the points it returns as an umbilicus file, which is the form
// Umbilicus::FromFile and resolveScrollUmbilicus already read.
//
// Only the xy plane is offered, because an umbilicus is a curve with one point per z, and the xy
// grids are the slices whose centre that point is. The xz and yz grids of the same volume hold
// the same segments cut the other way and have no such centre.
//
// The estimate is randomised: the function draws 10,000 segments and 1,000 candidate points from
// its generator, so two runs on one slice give two different answers, by up to a few millimetres
// on a well behaved slice and much more on one where the hill climb leaves the grid. Pass --seed
// to fix the draw, and the run repeats exactly. Without --seed the shared generator is seeded
// from the hardware, as it is everywhere else in the library, and the run does not repeat; the
// tool then refuses --threads above 1, because that generator is shared mutable state.
//
// It does not build normal grids, and does not need to: vc_gen_normalgrids already builds them
// from a prediction volume, reading every non zero voxel of a uint8 zarr with fill_value 0 as
// predicted surface. From predictions to an umbilicus is therefore two commands:
//
//     vc_gen_normalgrids -i predictions.zarr -o normal_grids/ --direction xy
//     vc_gen_umbilicus   -i normal_grids/    -o umbilicus.json --seed 1
//
// lasagna/labels_to_lasagna_normals.py runs the first of those on a binarised label volume and is
// the worked example of that step.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/core.hpp>

#include "utils/Json.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/normalgridtools.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

using utils::Json;
using vc::core::util::GridStore;
using vc::core::util::NormalGridVolume;
using vc::core::util::align_and_extract_umbilicus;

namespace {

constexpr int kXyPlane = 0;

struct Estimate {
    int slice = 0;
    int repeat = 0;
    std::optional<std::uint32_t> seed;
    cv::Vec2f point{0.0f, 0.0f};
    cv::Size gridSize{0, 0};
    bool inside = false;
};

// Slice index of a grid file, from the %06d.grid name vc_gen_normalgrids writes.
std::optional<int> sliceIndexOfFile(const fs::path& path)
{
    const std::string stem = path.stem().string();
    if (stem.empty() ||
        !std::all_of(stem.begin(), stem.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return std::nullopt;
    }
    try {
        return std::stoi(stem);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// The slices present in a directory of grid files, in order.
std::vector<int> slicesInDirectory(const fs::path& dir)
{
    std::vector<int> slices;
    if (!fs::is_directory(dir)) {
        return slices;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".grid") {
            continue;
        }
        // A slice the fetcher recorded as absent is an empty file, not a grid.
        if (entry.file_size() == 0) {
            continue;
        }
        if (const auto idx = sliceIndexOfFile(entry.path())) {
            slices.push_back(*idx);
        }
    }
    std::sort(slices.begin(), slices.end());
    return slices;
}

// "4000", "1000,2000,3000", "1000:9000" or "1000:9000:500". A range is inclusive of its end when
// the step divides the span, which is how a caller asking for 1000:9000:1000 expects to be given
// the slice at 9000.
std::vector<int> parseSlices(const std::string& spec)
{
    std::vector<int> slices;
    if (spec.find(':') != std::string::npos) {
        std::vector<long long> parts;
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ':')) {
            parts.push_back(std::stoll(token));
        }
        if (parts.size() < 2 || parts.size() > 3) {
            throw std::runtime_error("--slices range must be first:last or first:last:step");
        }
        const long long step = parts.size() == 3 ? parts[2] : 1;
        if (step <= 0) {
            throw std::runtime_error("--slices step must be positive");
        }
        for (long long v = parts[0]; v <= parts[1]; v += step) {
            slices.push_back(static_cast<int>(v));
        }
        return slices;
    }

    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            slices.push_back(static_cast<int>(std::stoll(token)));
        }
    }
    return slices;
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

void writeCsv(const fs::path& path, const std::vector<Estimate>& estimates, double scale)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open csv for writing: " + path.string());
    }
    // Enough digits that a coordinate several million grid units out, which is where a walk of
    // the published objective ends up, is still written exactly rather than rounded to six
    // significant figures.
    out << std::setprecision(12);
    out << "slice,repeat,seed,grid_x,grid_y,volume_x,volume_y,volume_z,"
           "grid_width,grid_height,inside\n";
    for (const auto& e : estimates) {
        out << e.slice << ',' << e.repeat << ',';
        if (e.seed) {
            out << *e.seed;
        }
        out << ',' << e.point[0] << ',' << e.point[1] << ','
            << e.point[0] / scale << ',' << e.point[1] / scale << ','
            << e.slice / scale << ','
            << e.gridSize.width << ',' << e.gridSize.height << ','
            << (e.inside ? 1 : 0) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    po::options_description desc(
        "Build an umbilicus from a normal grid volume, by running "
        "align_and_extract_umbilicus on its xy slices");
    desc.add_options()
        ("help,h", "Print this help message")
        ("input,i", po::value<std::string>(),
            "Normal grid volume directory, a directory of NNNNNN.grid files, or one .grid file")
        ("output,o", po::value<std::string>(),
            "Umbilicus json to write (control_points, as Umbilicus::FromFile reads them)")
        ("csv", po::value<std::string>(),
            "Also write every single estimate here, one row per slice and repeat")
        ("slices", po::value<std::string>(),
            "Slices to estimate: a list 1000,2000 or a range 1000:9000[:500]. "
            "The default is every slice present")
        ("level", po::value<int>()->default_value(0),
            "Level of a multiscale normal grid volume")
        ("seed", po::value<std::uint32_t>(),
            "Base seed. Slice z repeat r uses seed + z + 1000 * r, so a run repeats exactly. "
            "Without it the library's shared hardware seeded generator is used, as published")
        ("repeats", po::value<int>()->default_value(1),
            "Estimates per slice. The control point written is their coordinatewise median")
        ("threads", po::value<int>()->default_value(1),
            "Slices estimated at once. Above 1 requires --seed")
        ("coordinate-scale", po::value<double>(),
            "Grid units per volume voxel, when the input carries no metadata.json to state it. "
            "The written points are grid coordinates divided by this")
        ("volume", po::value<std::string>(),
            "Volume store the grids were built from, stamped into the file as provenance")
        ("voxelsize-um", po::value<double>(),
            "Voxel size in micrometres of the frame the points are written in, stamped into "
            "the file")
        ("volume-width", po::value<int>(), "Width in voxels of that frame, stamped into the file")
        ("volume-height", po::value<int>(), "Height in voxels of that frame, stamped into the file")
        ("volume-slices", po::value<int>(), "Slice count of that frame, stamped into the file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help") || argc == 1) {
        std::cout << desc << std::endl;
        std::cout
            << "\nNormal grids from predictions are vc_gen_normalgrids' job, not this tool's:\n"
            << "  vc_gen_normalgrids -i predictions.zarr -o normal_grids/ --direction xy\n"
            << "  vc_gen_umbilicus   -i normal_grids/    -o umbilicus.json --seed 1\n"
            << std::endl;
        return EXIT_SUCCESS;
    }

    if (!vm.count("input") || !vm.count("output")) {
        std::cerr << "Error: --input and --output are required." << std::endl;
        return EXIT_FAILURE;
    }

    const fs::path input = vm["input"].as<std::string>();
    const fs::path output = vm["output"].as<std::string>();
    const int level = vm["level"].as<int>();
    const int repeats = vm["repeats"].as<int>();
    const int threads = vm["threads"].as<int>();
    const std::optional<std::uint32_t> baseSeed =
        vm.count("seed") ? std::optional<std::uint32_t>(vm["seed"].as<std::uint32_t>())
                         : std::nullopt;

    if (repeats < 1) {
        std::cerr << "Error: --repeats must be at least 1." << std::endl;
        return EXIT_FAILURE;
    }
    if (threads < 1) {
        std::cerr << "Error: --threads must be at least 1." << std::endl;
        return EXIT_FAILURE;
    }
    if (threads > 1 && !baseSeed) {
        // Without a seed the function draws from one static generator shared by every caller, so
        // concurrent calls race on it. Refusing is better than a run that is neither repeatable
        // nor sound.
        std::cerr << "Error: --threads above 1 needs --seed, because the unseeded generator "
                     "is shared between calls." << std::endl;
        return EXIT_FAILURE;
    }
    if (!fs::exists(input)) {
        std::cerr << "Error: input does not exist: " << input << std::endl;
        return EXIT_FAILURE;
    }

    // Where the grids are and what one grid unit is worth.
    //
    // Three inputs are accepted. A normal grid volume directory, which states its own scale and
    // is read through NormalGridVolume so that multiscale levels and remote streaming work. A
    // plain directory of NNNNNN.grid files, which is what a partial download looks like and
    // states nothing, so its scale has to be given. And one .grid file, for a single slice.
    std::vector<int> slices;
    std::unique_ptr<NormalGridVolume> volume;
    fs::path looseDir;
    fs::path singleFile;
    double coordinateScale = 1.0;

    try {
        if (fs::is_directory(input) && fs::exists(input / "metadata.json")) {
            volume = std::make_unique<NormalGridVolume>(input.string(), level);
            coordinateScale = volume->coordinateScale();
            looseDir = input / "xy";
            if (fs::is_directory(looseDir / std::to_string(volume->level()))) {
                looseDir /= std::to_string(volume->level());
            }
        } else if (fs::is_directory(input)) {
            looseDir = input;
        } else {
            singleFile = input;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: cannot open the input: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("coordinate-scale")) {
        coordinateScale = vm["coordinate-scale"].as<double>();
    }
    if (!(coordinateScale > 0.0)) {
        std::cerr << "Error: --coordinate-scale must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    try {
        if (vm.count("slices")) {
            slices = parseSlices(vm["slices"].as<std::string>());
        } else if (!singleFile.empty()) {
            const auto idx = sliceIndexOfFile(singleFile);
            if (!idx) {
                std::cerr << "Error: cannot read a slice index from " << singleFile.filename()
                          << "; pass --slices with the one index this file holds." << std::endl;
                return EXIT_FAILURE;
            }
            slices.push_back(*idx);
        } else {
            slices = slicesInDirectory(looseDir);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (slices.empty()) {
        std::cerr << "Error: no slices to estimate." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "vc_gen_umbilicus: " << slices.size() << " slices, " << repeats
              << " estimate(s) each, coordinate scale " << coordinateScale;
    if (baseSeed) {
        std::cout << ", base seed " << *baseSeed;
    } else {
        std::cout << ", unseeded";
    }
    std::cout << std::endl;

    // One task is one estimate, so that the slices of a scroll fill the cores evenly even when
    // one of them walks much longer than the others.
    struct Task {
        size_t sliceIndex = 0;
        int repeat = 0;
    };
    std::vector<Task> tasks;
    tasks.reserve(slices.size() * static_cast<size_t>(repeats));
    for (size_t s = 0; s < slices.size(); ++s) {
        for (int r = 0; r < repeats; ++r) {
            tasks.push_back(Task{s, r});
        }
    }

    std::vector<std::optional<Estimate>> results(tasks.size());
    std::atomic<size_t> nextTask{0};
    std::mutex loadMutex;
    std::mutex reportMutex;
    std::atomic<size_t> failures{0};

    const auto loadGrid = [&](int slice) -> std::shared_ptr<const GridStore> {
        if (volume) {
            std::lock_guard<std::mutex> lock(loadMutex);
            return volume->get_grid(kXyPlane, slice);
        }
        char name[64];
        std::snprintf(name, sizeof(name), "%06d.grid", slice);
        const fs::path path = singleFile.empty() ? looseDir / name : singleFile;
        return std::make_shared<GridStore>(path.string());
    };

    const auto work = [&]() {
        while (true) {
            const size_t t = nextTask.fetch_add(1);
            if (t >= tasks.size()) {
                return;
            }
            const Task task = tasks[t];
            const int slice = slices[task.sliceIndex];
            try {
                const auto grid = loadGrid(slice);
                if (!grid) {
                    throw std::runtime_error("slice not found");
                }
                Estimate e;
                e.slice = slice;
                e.repeat = task.repeat;
                e.gridSize = grid->size();
                if (baseSeed) {
                    e.seed = static_cast<std::uint32_t>(*baseSeed + slice +
                                                        1000 * static_cast<std::uint32_t>(task.repeat));
                }
                e.point = align_and_extract_umbilicus(*grid, e.seed);
                if (std::isnan(e.point[0]) || std::isnan(e.point[1])) {
                    std::lock_guard<std::mutex> lock(reportMutex);
                    std::cerr << "slice " << slice << ": no segments, skipped" << std::endl;
                    ++failures;
                    continue;
                }
                e.inside = e.point[0] >= 0.0f && e.point[0] <= e.gridSize.width &&
                           e.point[1] >= 0.0f && e.point[1] <= e.gridSize.height;
                results[t] = e;
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lock(reportMutex);
                std::cerr << "slice " << slice << ": " << ex.what() << std::endl;
                ++failures;
            }
        }
    };

    if (threads == 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) {
            pool.emplace_back(work);
        }
        for (auto& thread : pool) {
            thread.join();
        }
    }

    std::vector<Estimate> estimates;
    estimates.reserve(results.size());
    for (const auto& r : results) {
        if (r) {
            estimates.push_back(*r);
        }
    }
    std::sort(estimates.begin(), estimates.end(), [](const Estimate& a, const Estimate& b) {
        return a.slice != b.slice ? a.slice < b.slice : a.repeat < b.repeat;
    });

    if (estimates.empty()) {
        std::cerr << "Error: not one slice produced an estimate." << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("csv")) {
        try {
            writeCsv(vm["csv"].as<std::string>(), estimates, coordinateScale);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    // One control point per slice, the coordinatewise median of that slice's repeats, carried
    // from grid units into the frame the grids were built from.
    Json points = Json::array();
    size_t outside = 0;
    for (size_t i = 0; i < estimates.size();) {
        size_t j = i;
        std::vector<double> xs;
        std::vector<double> ys;
        size_t insideCount = 0;
        while (j < estimates.size() && estimates[j].slice == estimates[i].slice) {
            xs.push_back(estimates[j].point[0]);
            ys.push_back(estimates[j].point[1]);
            insideCount += estimates[j].inside ? 1 : 0;
            ++j;
        }
        Json point = Json::object();
        point["x"] = median(xs) / coordinateScale;
        point["y"] = median(ys) / coordinateScale;
        point["z"] = estimates[i].slice / coordinateScale;
        point["estimates"] = static_cast<int>(j - i);
        point["inside"] = insideCount == j - i;
        points.push_back(point);
        if (insideCount != j - i) {
            ++outside;
        }
        i = j;
    }

    Json metadata = Json::object();
    metadata["total_points"] = static_cast<int>(points.size());
    metadata["tool"] = "vc_gen_umbilicus";
    metadata["source"] = fs::absolute(input).string();
    metadata["plane"] = "xy";
    metadata["coordinate_scale"] = coordinateScale;
    metadata["repeats"] = repeats;
    if (baseSeed) {
        metadata["seed"] = static_cast<int64_t>(*baseSeed);
    }
    if (vm.count("volume")) {
        metadata["volume"] = vm["volume"].as<std::string>();
    }
    if (vm.count("voxelsize-um")) {
        metadata["voxelsize_um"] = vm["voxelsize-um"].as<double>();
    }
    // The grid dimension triplet is stamped only when all three are given. Two of three is read
    // by Umbilicus::LoadFileInfo as a malformed statement and refuses the file, and the tool
    // cannot supply the third itself: the grids state their extent in x and y but not the slice
    // count of the volume they came from, since a sparse or partial set of grid files is
    // indistinguishable from a short volume.
    const bool haveTriplet = vm.count("volume-width") && vm.count("volume-height") &&
                             vm.count("volume-slices");
    if (haveTriplet) {
        metadata["volume_width"] = vm["volume-width"].as<int>();
        metadata["volume_height"] = vm["volume-height"].as<int>();
        metadata["volume_slices"] = vm["volume-slices"].as<int>();
    } else if (vm.count("volume-width") || vm.count("volume-height") || vm.count("volume-slices")) {
        std::cerr << "Error: --volume-width, --volume-height and --volume-slices go together or "
                     "not at all." << std::endl;
        return EXIT_FAILURE;
    }

    Json document = Json::object();
    document["metadata"] = metadata;
    document["control_points"] = points;

    std::ofstream out(output);
    if (!out) {
        std::cerr << "Error: cannot open " << output << " for writing." << std::endl;
        return EXIT_FAILURE;
    }
    out << document.dump(2) << std::endl;
    if (!out) {
        std::cerr << "Error: failed writing " << output << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "wrote " << points.size() << " control points to " << output << std::endl;
    if (outside > 0) {
        std::cout << outside << " of " << points.size()
                  << " lie outside the grid they were estimated on" << std::endl;
    }
    if (failures > 0) {
        std::cout << failures << " estimate(s) failed and are absent from the file" << std::endl;
    }
    return EXIT_SUCCESS;
}
