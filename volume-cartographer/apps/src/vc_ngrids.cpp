#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/core/types.hpp>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

static void print_usage() {
    std::cout
        << "vc_ngrids: Read a NormalGridVolume and derive/visualize data.\n\n"
        << "Usage: vc_ngrids [options]\n\n"
        << "Options:\n"
        << "  -h, --help              Print this help message\n"
        << "  -i, --input PATH        Input NormalGridVolume directory (required)\n"
        << "  -c, --crop x0 y0 z0 x1 y1 z1   Crop bounding box in voxel coords (half-open)\n"
        << "      --vis-obj PATH      Write visualization as OBJ (vertices + polyline edges)\n\n"
        << "Notes:\n"
        << "  - Input is the directory created by vc_gen_normalgrids (contains metadata.json and xy/xz/yz).\n"
        << "  - Crop is optional; if omitted the full extent from metadata/available grids is used.\n";
}

struct CropBox3i {
    cv::Point3i min; // inclusive
    cv::Point3i max; // exclusive
};

static CropBox3i crop_from_args(const std::vector<int>& v) {
    // v = {x0,y0,z0,x1,y1,z1}
    if (v.size() != 6) {
        throw std::runtime_error("--crop expects 6 integers: x0 y0 z0 x1 y1 z1");
    }
    CropBox3i c;
    c.min = cv::Point3i(v[0], v[1], v[2]);
    c.max = cv::Point3i(v[3], v[4], v[5]);
    if (c.max.x < c.min.x || c.max.y < c.min.y || c.max.z < c.min.z) {
        throw std::runtime_error("--crop invalid: max must be >= min in all dimensions");
    }
    return c;
}

struct ObjWriter {
    explicit ObjWriter(const fs::path& path) : out(path) {
        if (!out) {
            throw std::runtime_error("Failed to open output file for writing: " + path.string());
        }
        out << "# vc_ngrids visualization\n";
    }

    void write_polyline(const std::vector<cv::Point3f>& pts) {
        if (pts.size() < 2) return;

        std::vector<size_t> idx;
        idx.reserve(pts.size());

        for (const auto& p : pts) {
            out << "v " << p.x << " " << p.y << " " << p.z << "\n";
            idx.push_back(++vtx_count);
        }

        out << "l";
        for (const auto i : idx) {
            out << " " << i;
        }
        out << "\n";
    }

    std::ofstream out;
    size_t vtx_count = 0;
};

static void add_gridstore_paths_as_obj_polylines(
    ObjWriter& obj,
    const vc::core::util::GridStore& grid,
    int plane_idx,
    int slice_idx,
    const CropBox3i& crop) {
    const auto paths = grid.get_all();
    for (const auto& path_ptr : paths) {
        if (!path_ptr || path_ptr->size() < 2) continue;

        std::vector<cv::Point3f> pts;
        pts.reserve(path_ptr->size());

        for (const auto& p2 : *path_ptr) {
            // GridStore coordinates are 2D pixel coords within the slice image.
            // We map them back into a 3D voxel coordinate depending on plane.
            // Dataset axis convention in vc_gen_normalgrids is assumed: (z,y,x).
            cv::Point3f p3;
            if (plane_idx == 0) {
                // xy: fixed z
                p3 = cv::Point3f(static_cast<float>(p2.x), static_cast<float>(p2.y), static_cast<float>(slice_idx));
            } else if (plane_idx == 1) {
                // xz: fixed y
                p3 = cv::Point3f(static_cast<float>(p2.x), static_cast<float>(slice_idx), static_cast<float>(p2.y));
            } else {
                // yz: fixed x
                p3 = cv::Point3f(static_cast<float>(slice_idx), static_cast<float>(p2.x), static_cast<float>(p2.y));
            }

            // Optional crop filtering (in 3D).
            if (p3.x < crop.min.x || p3.y < crop.min.y || p3.z < crop.min.z) continue;
            if (p3.x >= crop.max.x) continue;
            if (p3.y >= crop.max.y) continue;
            if (p3.z >= crop.max.z) continue;

            pts.push_back(p3);
        }

        if (pts.size() >= 2) {
            obj.write_polyline(pts);
        }
    }
}

static std::shared_ptr<const vc::core::util::GridStore> try_load_grid(
    const fs::path& base,
    const std::string& plane_dir,
    int slice_idx) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%06d.grid", slice_idx);
    fs::path grid_path = base / plane_dir / filename;
    if (!fs::exists(grid_path)) return nullptr;
    return std::make_shared<vc::core::util::GridStore>(grid_path.string());
}

static int align_down(int v, int step) {
    if (step <= 1) return v;
    if (v >= 0) return (v / step) * step;
    // For negative, ensure we still go downwards.
    return -(((-v + step - 1) / step) * step);
}

static void run_vis_obj(const fs::path& input_dir, const fs::path& out_obj, const std::optional<CropBox3i>& crop_opt) {
    vc::core::util::NormalGridVolume ngv(input_dir.string());
    const int sparse_volume = ngv.metadata().value("sparse-volume", 1);

    // If no crop provided, use a permissive default. (We still only iterate slices that exist.)
    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Point3i(0, 0, 0),
        cv::Point3i(std::numeric_limits<int>::max() / 4,
                    std::numeric_limits<int>::max() / 4,
                    std::numeric_limits<int>::max() / 4),
    });

    ObjWriter obj(out_obj);

    struct PlaneCfg {
        int plane_idx;
        const char* dir;
        // which crop axis is the slice axis: 0=x,1=y,2=z
        int slice_axis;
    };
    const PlaneCfg planes[3] = {
        {0, "xy", 2},
        {1, "xz", 1},
        {2, "yz", 0},
    };

    for (const auto& pc : planes) {
        const int crop_min = (pc.slice_axis == 0) ? crop.min.x : (pc.slice_axis == 1) ? crop.min.y : crop.min.z;
        const int crop_max = (pc.slice_axis == 0) ? crop.max.x : (pc.slice_axis == 1) ? crop.max.y : crop.max.z;
        int slice_start = align_down(crop_min, sparse_volume);
        if (slice_start < crop_min) slice_start += sparse_volume;

        for (int slice = slice_start; slice < crop_max; slice += std::max(1, sparse_volume)) {
            auto grid = try_load_grid(input_dir, pc.dir, slice);
            if (!grid) continue;
            add_gridstore_paths_as_obj_polylines(obj, *grid, pc.plane_idx, slice, crop);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("vc_ngrids options");
    desc.add_options()
        ("help,h", "Print help")
        ("input,i", po::value<std::string>()->required(), "Input NormalGridVolume directory")
        ("crop,c", po::value<std::vector<int>>()->multitoken(), "Crop x0 y0 z0 x1 y1 z1")
        ("vis-obj", po::value<std::string>(), "Write visualization OBJ file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help") || argc == 1) {
            print_usage();
            std::cout << "\n" << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage();
        std::cout << "\n" << desc << std::endl;
        return 1;
    }

    const fs::path input_dir(vm["input"].as<std::string>());
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: input is not a directory: " << input_dir << std::endl;
        return 1;
    }

    std::optional<CropBox3i> crop;
    if (vm.count("crop")) {
        crop = crop_from_args(vm["crop"].as<std::vector<int>>());
    }

    if (vm.count("vis-obj")) {
        run_vis_obj(input_dir, fs::path(vm["vis-obj"].as<std::string>()), crop);
        return 0;
    }

    std::cerr << "Error: no output specified. Use --vis-obj.\n\n";
    print_usage();
    return 1;
}
