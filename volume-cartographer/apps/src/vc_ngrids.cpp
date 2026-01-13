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
        << "      --vis-ply PATH      Write visualization as PLY with vertex colors\n\n"
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

struct PlyWriter {
    struct Vtx {
        float x, y, z;
        uint8_t r, g, b;
    };

    explicit PlyWriter(const fs::path& path) : path(path) {}

    void write_polyline(const std::vector<cv::Point3f>& pts, const cv::Vec3b& color) {
        if (pts.size() < 2) return;

        const size_t base = vertices.size();
        vertices.reserve(vertices.size() + pts.size());
        edges.reserve(edges.size() + (pts.size() - 1));

        for (const auto& p : pts) {
            vertices.push_back(Vtx{
                p.x,
                p.y,
                p.z,
                color[2], // OpenCV Vec3b is BGR; PLY expects RGB
                color[1],
                color[0],
            });
        }

        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            edges.emplace_back(static_cast<uint32_t>(base + i), static_cast<uint32_t>(base + i + 1));
        }
    }

    void flush_ascii() const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Failed to open output file for writing: " + path.string());
        }

        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment vc_ngrids visualization\n";
        out << "element vertex " << vertices.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "element edge " << edges.size() << "\n";
        out << "property int vertex1\n";
        out << "property int vertex2\n";
        out << "end_header\n";

        for (const auto& v : vertices) {
            out << v.x << " " << v.y << " " << v.z << " "
                << static_cast<int>(v.r) << " " << static_cast<int>(v.g) << " " << static_cast<int>(v.b) << "\n";
        }
        for (const auto& e : edges) {
            out << e.first << " " << e.second << "\n";
        }
    }

    fs::path path;
    std::vector<Vtx> vertices;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
};

static void add_gridstore_paths_as_ply_polylines(
    PlyWriter& ply,
    const vc::core::util::GridStore& grid,
    int plane_idx,
    int slice_idx,
    const CropBox3i& crop,
    const cv::Vec3b& color_bgr) {
    const auto paths = grid.get_all();
    for (const auto& path_ptr : paths) {
        if (!path_ptr || path_ptr->size() < 2) continue;

        std::vector<cv::Point3f> pts;
        pts.reserve(path_ptr->size());

        for (const auto& p2 : *path_ptr) {
            cv::Point3f p3;
            if (plane_idx == 0) {
                p3 = cv::Point3f(static_cast<float>(p2.x), static_cast<float>(p2.y), static_cast<float>(slice_idx));
            } else if (plane_idx == 1) {
                p3 = cv::Point3f(static_cast<float>(p2.x), static_cast<float>(slice_idx), static_cast<float>(p2.y));
            } else {
                p3 = cv::Point3f(static_cast<float>(slice_idx), static_cast<float>(p2.x), static_cast<float>(p2.y));
            }

            if (p3.x < crop.min.x || p3.y < crop.min.y || p3.z < crop.min.z) continue;
            if (p3.x >= crop.max.x) continue;
            if (p3.y >= crop.max.y) continue;
            if (p3.z >= crop.max.z) continue;

            pts.push_back(p3);
        }

        if (pts.size() >= 2) {
            ply.write_polyline(pts, color_bgr);
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

static void run_vis_ply(const fs::path& input_dir, const fs::path& out_ply, const std::optional<CropBox3i>& crop_opt) {
    vc::core::util::NormalGridVolume ngv(input_dir.string());
    const int sparse_volume = ngv.metadata().value("sparse-volume", 1);

    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Point3i(0, 0, 0),
        cv::Point3i(std::numeric_limits<int>::max() / 4,
                    std::numeric_limits<int>::max() / 4,
                    std::numeric_limits<int>::max() / 4),
    });

    PlyWriter ply(out_ply);

    struct PlaneCfg {
        int plane_idx;
        const char* dir;
        int slice_axis;
        cv::Vec3b color_bgr;
    };
    const PlaneCfg planes[3] = {
        {0, "xy", 2, cv::Vec3b(0, 0, 255)},   // red
        {1, "xz", 1, cv::Vec3b(0, 255, 0)},   // green
        {2, "yz", 0, cv::Vec3b(255, 0, 0)},   // blue
    };

    for (const auto& pc : planes) {
        const int crop_min = (pc.slice_axis == 0) ? crop.min.x : (pc.slice_axis == 1) ? crop.min.y : crop.min.z;
        const int crop_max = (pc.slice_axis == 0) ? crop.max.x : (pc.slice_axis == 1) ? crop.max.y : crop.max.z;
        int slice_start = align_down(crop_min, sparse_volume);
        if (slice_start < crop_min) slice_start += sparse_volume;

        for (int slice = slice_start; slice < crop_max; slice += std::max(1, sparse_volume)) {
            auto grid = try_load_grid(input_dir, pc.dir, slice);
            if (!grid) continue;
            add_gridstore_paths_as_ply_polylines(ply, *grid, pc.plane_idx, slice, crop, pc.color_bgr);
        }
    }

    ply.flush_ascii();
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("vc_ngrids options");
    desc.add_options()
        ("help,h", "Print help")
        ("input,i", po::value<std::string>()->required(), "Input NormalGridVolume directory")
        ("crop,c", po::value<std::vector<int>>()->multitoken(), "Crop x0 y0 z0 x1 y1 z1")
        ("vis-ply", po::value<std::string>(), "Write visualization PLY file (with colors)");

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

    if (vm.count("vis-ply")) {
        run_vis_ply(input_dir, fs::path(vm["vis-ply"].as<std::string>()), crop);
        return 0;
    }

    std::cerr << "Error: no output specified. Use --vis-ply.\n\n";
    print_usage();
    return 1;
}
