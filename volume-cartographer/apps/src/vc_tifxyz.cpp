#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void rotate_points(cv::Mat_<cv::Vec3f>& points, float angle_deg) {
    if (points.empty()) return;

    // Center of rotation (image center)
    cv::Point2f center(static_cast<float>(points.cols - 1) / 2.0f, static_cast<float>(points.rows - 1) / 2.0f);
    
    // Get the rotation matrix
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    // Create a copy to store rotated points
    cv::Mat_<cv::Vec3f> rotated_points = points.clone();
    
    // Apply rotation to each point
    cv::warpAffine(points, rotated_points, rot_mat, points.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(-1,-1,-1));
    
    points = rotated_points;
}


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
    QuadSurface* surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(input_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading tifxyz file: " << input_path << " - " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> points = surf->rawPoints();

    // Apply rotation
    std::cout << "Rotating points by " << rotation_angle << " degrees..." << std::endl;
    rotate_points(points, rotation_angle);

    // Generate output filename
    float normalized_angle = fmod(rotation_angle, 360.0f);
    if (normalized_angle < 0) normalized_angle += 360.0f;
    
    std::string angle_str = std::to_string(static_cast<int>(normalized_angle));
    std::filesystem::path output_path = input_path.parent_path() / (input_path.stem().string() + "_r" + angle_str + input_path.extension().string());
    
    // Save the modified surface
    QuadSurface rotated_surf(points, surf->scale());
    rotated_surf.save(output_path, true);
    std::cout << "Saved rotated surface to: " << output_path << std::endl;

    delete surf;
    return EXIT_SUCCESS;
}