#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <iomanip>
#include <omp.h>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <ceres/ceres.h>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/GridStore.hpp"

#include "discrete.hpp"
#include "continous.hpp"
#include "spiral.hpp"
#include "spiral2.hpp"
#include "spiral2cont.hpp"
#include "continuous3d.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;




class VideoCallback : public ceres::IterationCallback {
public:
    VideoCallback(const std::vector<double>* positions, cv::Mat* frame, cv::VideoWriter* writer)
        : positions_(positions), frame_(frame), writer_(writer) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        if (writer_ && writer_->isOpened()) {
            // Redraw the entire trace for this frame
            frame_->setTo(cv::Scalar(0,0,0)); // Clear frame
            for (size_t i = 0; i < positions_->size() / 2 - 1; ++i) {
                cv::Point p1((*positions_)[i*2], (*positions_)[i*2+1]);
                cv::Point p2((*positions_)[(i+1)*2], (*positions_)[(i+1)*2+1]);
                cv::line(*frame_, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
            writer_->write(*frame_);
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    const std::vector<double>* positions_;
    cv::Mat* frame_;
    cv::VideoWriter* writer_;
};

struct SkeletonVertex {
    cv::Point pos;
};

struct SkeletonEdge {
    std::vector<cv::Point> path;
    int id;
};

using SkeletonGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SkeletonVertex, SkeletonEdge>;
 
int main(int argc, char** argv) {
    po::options_description desc("Diffuse winding number through a voxel grid.");
    desc.add_options()
        ("help,h", "Print help")
        ("points", po::value<std::string>(), "Input JSON point sets file (VCCollection format)")
        ("volume", po::value<std::string>(), "Input zarr volume path")
        ("dataset", po::value<std::string>()->default_value("0"), "Dataset within the zarr file (e.g., '0' for scale level)")
        ("output", po::value<std::string>(), "Output image file for visualization")
        ("video-out", po::value<std::string>(), "Output path for the video visualization")
        ("json-out", po::value<std::string>(), "Path to write the spiral wraps to a JSON file.")
        ("json-in", po::value<std::string>(), "Path to read spiral wraps from a JSON file for spiral2cont mode.")
        ("collection-name", po::value<std::string>(), "Name of the collection to use in spiral2cont mode.")
        ("start-winding", po::value<double>(), "Start winding number for the spiral to process in spiral2cont mode.")
        ("end-winding", po::value<double>(), "End winding number for the spiral to process in spiral2cont mode.")
        ("conflicts-out", po::value<std::string>(), "Output path for the conflicts map image")
        ("umbilicus-set", po::value<std::string>()->default_value("umbilicus"), "Name of the point set marking the umbilicus")
        ("iterations", po::value<int>()->default_value(100), "Number of diffusion/refinement iterations")
        ("mode", po::value<std::string>()->default_value("continuous"), "Diffusion mode: 'discrete', 'continuous', 'spiral', 'spiral2', 'spiral2cont', or 'continuous3d'")
        ("ray-step-dist", po::value<float>()->default_value(5.0f), "Distance between rays for sheet constraints in pixels (0 to disable)")
       ("mask", po::value<std::string>(), "Optional mask image to constrain diffusion")
       ("winding", po::value<double>(), "Winding number to process in continuous3d mode.")
       ("collection", po::value<std::string>(), "Name of the point collection to use in continuous3d mode.")
       ("box-size", po::value<std::vector<int>>()->multitoken(), "Box size for 3D volume extraction (W H D).")
       ("debug,d", "Enable debug image outputs")
       ("dampening", po::value<float>()->default_value(0.0f), "Dampening factor for iterative diffusion")
       ("starting-diameter", po::value<double>()->default_value(200.0), "Starting diameter for spiral generation.")
       ("end-diameter", po::value<double>()->default_value(3000.0), "End diameter for spiral generation.")
       ("spiral-step", po::value<double>()->default_value(16.0), "Step size for spiral generation.")
       ("revolutions", po::value<int>()->default_value(5), "Number of revolutions for the spiral.")
       ("no-optimized-spiral", po::bool_switch()->default_value(false), "Use the original simple spiral generation");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (!vm.count("points") || !vm.count("volume") || !vm.count("output")) {
        std::cerr << "Error: Missing required arguments: --points, --volume, --output" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    fs::path points_path = vm["points"].as<std::string>();
    fs::path volume_path = vm["volume"].as<std::string>();
    std::string dataset_name = vm["dataset"].as<std::string>();
    fs::path output_path = vm["output"].as<std::string>();
    fs::path conflicts_path;
    if (vm.count("conflicts-out")) {
        conflicts_path = vm["conflicts-out"].as<std::string>();
    }
    std::string umbilicus_set_name = vm["umbilicus-set"].as<std::string>();
    int iterations = vm["iterations"].as<int>();
    std::string mode = vm["mode"].as<std::string>();
   float ray_step_dist = vm["ray-step-dist"].as<float>();

    std::cout << "Winding Diffusion Tool" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Mode: " << mode << std::endl;

    // Load points
    VCCollection point_collection;
    if (!point_collection.loadFromJSON(points_path.string())) {
        std::cerr << "Error: Failed to load point file: " << points_path << std::endl;
        return 1;
    }
    std::cout << "Loaded " << point_collection.getAllCollections().size() << " point sets from " << points_path << std::endl;

    // Find umbilicus point
    std::optional<cv::Vec3f> umbilicus_point;
    for (const auto& [id, collection] : point_collection.getAllCollections()) {
        if (collection.name == umbilicus_set_name) {
            if (collection.points.empty()) {
                std::cerr << "Error: Umbilicus point set '" << umbilicus_set_name << "' is empty." << std::endl;
                return 1;
            }
            umbilicus_point = collection.points.begin()->second.p;
            break;
        }
    }

    if (!umbilicus_point) {
        std::cerr << "Error: Umbilicus point set '" << umbilicus_set_name << "' not found." << std::endl;
        return 1;
    }
    std::cout << "Found umbilicus point at: " << *umbilicus_point << std::endl;

    // Load volume data using z5 and ChunkedTensor
    z5::filesystem::handle::Group group_handle(volume_path);
    std::unique_ptr<z5::Dataset> ds = z5::openDataset(group_handle, dataset_name);
    if (!ds) {
        std::cerr << "Error: Could not open dataset '" << dataset_name << "' in volume '" << volume_path << "'." << std::endl;
        return 1;
    }
    auto shape = ds->shape();
    std::cout << "Volume shape: (" << shape[0] << ", " << shape[1] << ", " << shape[2] << ")" << std::endl;

    // Extract XY slice
    int z_slice = static_cast<int>(std::round((*umbilicus_point)[2]));
    std::cout << "Extracting slice at z=" << z_slice << std::endl;

    auto start_slice_extraction = std::chrono::high_resolution_clock::now();
    std::clock_t start_cpu = std::clock();

    cv::Mat slice_mat(shape[1], shape[2], CV_8U);
    std::vector<size_t> slice_shape = {1, shape[1], shape[2]};
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> slice_data = xt::zeros<uint8_t>(slice_shape);
    cv::Vec3i offset = {z_slice, 0, 0};
    ChunkCache cache(4llu*1024*1024*1024);
    readArea3D(slice_data, offset, ds.get(), &cache);
    
    for (int y = 0; y < shape[1]; ++y) {
        for (int x = 0; x < shape[2]; ++x) {
            slice_mat.at<uint8_t>(y, x) = slice_data(0, y, x);
        }
    }
    auto end_slice_extraction = std::chrono::high_resolution_clock::now();
    std::clock_t end_cpu = std::clock();

    double real_time = std::chrono::duration<double>(end_slice_extraction - start_slice_extraction).count();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    std::cout << "Slice extraction took " << real_time << " s (real), "
              << cpu_time << " s (user), "
              << (cpu_time / real_time * 100.0) << "%" << std::endl;
    
    std::cout << "Successfully extracted slice." << std::endl;

    cv::Mat mask;
    if (vm.count("mask")) {
        fs::path mask_path = vm["mask"].as<std::string>();
        mask = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            std::cerr << "Error: Failed to load mask image from " << mask_path << std::endl;
            return 1;
        }
        if (mask.size() != slice_mat.size()) {
            std::cerr << "Error: Mask dimensions (" << mask.size() << ") do not match slice dimensions (" << slice_mat.size() << ")." << std::endl;
            return 1;
        }
        for (int y = 0; y < slice_mat.rows; ++y) {
            for (int x = 0; x < slice_mat.cols; ++x) {
                if (mask.at<uint8_t>(y, x) == 0) {
                    slice_mat.at<uint8_t>(y, x) = 0;
                }
            }
        }
        std::cout << "Applied mask from " << mask_path << std::endl;
    }

    if (vm.count("debug")) {
        cv::imwrite("slice.tif", slice_mat);
        std::cout << "Saved debug slice to slice.tif" << std::endl;
    }

    if (vm.count("debug")) {
        cv::imwrite("slice.tif", slice_mat);
        std::cout << "Saved debug slice to slice.tif" << std::endl;
    }

    if (mode == "discrete") {
        return discrete_main(slice_mat, point_collection, umbilicus_point, umbilicus_set_name, iterations, output_path, conflicts_path);
    } else if (mode == "continuous") {
        return continous_main(slice_mat, point_collection, umbilicus_point, umbilicus_set_name, iterations, ray_step_dist, output_path, conflicts_path, mask, vm);
    } else if (mode == "spiral") {
        return spiral_main(slice_mat, umbilicus_point, output_path, ray_step_dist, vm);
    } else if (mode == "spiral2") {
        return spiral2_main(slice_mat, point_collection, umbilicus_point, umbilicus_set_name, vm);
    } else if (mode == "spiral2cont") {
        return spiral2cont_main(slice_mat, umbilicus_point, output_path, vm);
    } else if (mode == "continuous3d") {
        return continuous3d_main(vm);
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'. Use 'discrete', 'continuous', 'spiral', 'spiral2', 'spiral2cont', or 'continuous3d'." << std::endl;
        return 1;
    }
 
    return 0;
}
