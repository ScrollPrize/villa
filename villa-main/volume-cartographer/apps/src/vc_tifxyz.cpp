#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace {

bool isValidPoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && point[1] != -1.0f && point[2] != -1.0f
        && std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

void laplacianSmooth(cv::Mat_<cv::Vec3f>& points, int iterations)
{
    if (points.empty() || iterations <= 0) {
        return;
    }

    constexpr float strength = 0.5f;
    const int neighborRows[] = {-1, 1, 0, 0};
    const int neighborCols[] = {0, 0, -1, 1};

    for (int iter = 0; iter < iterations; ++iter) {
        const cv::Mat_<cv::Vec3f> previous = points.clone();

        for (int row = 0; row < previous.rows; ++row) {
            for (int col = 0; col < previous.cols; ++col) {
                const cv::Vec3f& current = previous(row, col);
                if (!isValidPoint(current)) {
                    points(row, col) = cv::Vec3f(-1.0f, -1.0f, -1.0f);
                    continue;
                }

                cv::Vec3f sum(0.0f, 0.0f, 0.0f);
                int count = 0;
                for (int i = 0; i < 4; ++i) {
                    const int nr = row + neighborRows[i];
                    const int nc = col + neighborCols[i];
                    if (nr < 0 || nr >= previous.rows || nc < 0 || nc >= previous.cols) {
                        continue;
                    }

                    const cv::Vec3f& neighbor = previous(nr, nc);
                    if (!isValidPoint(neighbor)) {
                        continue;
                    }

                    sum += neighbor;
                    ++count;
                }

                if (count == 0) {
                    points(row, col) = current;
                    continue;
                }

                const cv::Vec3f average = sum * (1.0f / static_cast<float>(count));
                points(row, col) = current * (1.0f - strength) + average * strength;
            }
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input-file", po::value<std::string>(), "input tifxyz file")
        ("rotate,r", po::value<float>(), "Rotate the point grid by a given angle in degrees.")
        ("resample,s", po::value<float>(), "Resample the surface by a scale factor (>1 increases density)")
        ("smooth", po::value<int>()->implicit_value(1), "Laplacian smooth the whole tifxyz grid for the given iterations (default: 1)")
        ("interpolation,i", po::value<std::string>()->default_value("bilinear"),
         "Interpolation method: nearest, bilinear (default), cubic, lanczos")
        ("paths,p", po::value<std::vector<std::string>>()->multitoken(), "Path arguments (currently unused).");

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("paths", -1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

        if (vm.count("help")) {
            std::cout << "usage: " << argv[0] << " <tifxyz> [-r/--rotate angle_deg] [-s/--resample factor] [--smooth [iterations]] [-i/--interpolation method]\n" << desc << std::endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << "usage: " << argv[0] << " <tifxyz> [-r/--rotate angle_deg] [-s/--resample factor] [--smooth [iterations]] [-i/--interpolation method]\n" << desc << std::endl;
        return EXIT_FAILURE;
    }

    if (!vm.count("input-file")) {
        std::cerr << "Error: No input tiffxyz file specified." << std::endl;
        return EXIT_FAILURE;
    }

    // Require at least one operation
    if (!vm.count("rotate") && !vm.count("resample") && !vm.count("smooth")) {
        std::cerr << "Error: At least one operation (--rotate, --resample, or --smooth) must be specified." << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path input_path = vm["input-file"].as<std::string>();

    // Parse interpolation method
    int interpFlag = cv::INTER_LINEAR;
    std::string method = vm["interpolation"].as<std::string>();
    if (method == "nearest") {
        interpFlag = cv::INTER_NEAREST;
    } else if (method == "bilinear" || method == "linear") {
        interpFlag = cv::INTER_LINEAR;
    } else if (method == "cubic") {
        interpFlag = cv::INTER_CUBIC;
    } else if (method == "lanczos") {
        interpFlag = cv::INTER_LANCZOS4;
    } else {
        std::cerr << "Warning: Unknown interpolation method '" << method << "', using bilinear." << std::endl;
    }

    // Load the surface
    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(input_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading tifxyz file: " << input_path << " - " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Apply operations (resample first, then rotate)
    if (vm.count("resample")) {
        float factor = vm["resample"].as<float>();
        std::cout << "Resampling surface by factor " << factor << "..." << std::endl;
        surf->resample(factor, interpFlag);
    }

    if (vm.count("rotate")) {
        float angle = vm["rotate"].as<float>();
        std::cout << "Rotating surface by " << angle << " degrees..." << std::endl;
        surf->rotate(angle);
    }

    if (vm.count("smooth")) {
        int iterations = vm["smooth"].as<int>();
        if (iterations < 1) {
            std::cerr << "Error: --smooth iterations must be >= 1." << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Laplacian smoothing surface for " << iterations << " iteration"
                  << (iterations == 1 ? "" : "s") << "..." << std::endl;
        cv::Mat_<cv::Vec3f>* points = surf->rawPointsPtr();
        laplacianSmooth(*points, iterations);
        surf->invalidateCache();
    }

    // Generate output filename
    std::string suffix;
    if (vm.count("resample")) {
        float factor = vm["resample"].as<float>();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << factor;
        suffix += "_s" + oss.str();
    }
    if (vm.count("rotate")) {
        float angle = fmod(vm["rotate"].as<float>(), 360.0f);
        if (angle < 0) angle += 360.0f;
        suffix += "_r" + std::to_string(static_cast<int>(angle));
    }
    if (vm.count("smooth")) {
        int iterations = vm["smooth"].as<int>();
        suffix += "_smooth" + std::to_string(iterations);
    }

    std::filesystem::path output_path = input_path.parent_path() / (input_path.stem().string() + suffix + input_path.extension().string());

    // Recalculate and update the surface area
    double area = vc::surface::computeSurfaceAreaVox2(surf->rawPoints());
    if (surf->meta.is_null()) {
        surf->meta = utils::Json::object();
    }
    surf->meta["area"] = area;

    surf->save(output_path, true);
    std::cout << "Saved transformed surface to: " << output_path << std::endl;

    return EXIT_SUCCESS;
}
