#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

cv::Mat rotate_channel(const cv::Mat& channel, float angle_deg, int border_mode, const cv::Scalar& border_value) {
    if (channel.empty()) return channel;

    cv::Point2f center(static_cast<float>(channel.cols - 1) / 2.0f, static_cast<float>(channel.rows - 1) / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_deg, 1.0);

    // Calculate the bounding box of the rotated image
    cv::Rect2f bbox = cv::RotatedRect(center, channel.size(), angle_deg).boundingRect2f();

    // Adjust the rotation matrix to translate the image to the center of the new canvas
    rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    cv::Mat rotated_channel;
    cv::warpAffine(channel, rotated_channel, rot_mat, bbox.size(), cv::INTER_LINEAR, border_mode, border_value);

    return rotated_channel;
}

void rotate_points(cv::Mat_<cv::Vec3f>& points, float angle_deg) {
    if (points.empty()) return;

    cv::Mat rotated_mat = rotate_channel(points, angle_deg, cv::BORDER_CONSTANT, cv::Scalar(-1,-1,-1));
    cv::Mat_<cv::Vec3f> rotated_points = rotated_mat;

    // Create a mask of valid points (not equal to the border value)
    cv::Mat mask;
    cv::inRange(rotated_points, cv::Scalar(-1, -1, -1), cv::Scalar(-1, -1, -1), mask);
    // Erode the mask to remove the outermost edge
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(mask, mask, kernel, {-1,-1});

    // Apply the eroded mask back to the points
    rotated_points.setTo(cv::Scalar(-1, -1, -1), mask);
    
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
    
    // Rotate all other channels
    std::unordered_map<std::string, cv::Mat> rotated_channels;
    for (const auto& name : surf->channelNames()) {
        // Points are handled separately
        if (name == "points") continue;

        cv::Mat channel = surf->channel(name);
        if (channel.empty()) continue;

        std::cout << "Rotating channel: " << name << "..." << std::endl;
        
        // For normals, use a different border value
        cv::Scalar border_value(0,0,0);
        if (name == "normals") {
             border_value = cv::Scalar(0,0,0);
        }

        rotated_channels[name] = rotate_channel(channel, rotation_angle, cv::BORDER_CONSTANT, border_value);
    }

    if (rotated_channels.size())
        std::cout << "WARNING: channels support is untested for rotation (including mask) - please check the output" << std::endl;

    // Save the modified surface
    QuadSurface rotated_surf(points, surf->scale());
    if (surf->meta) {
        rotated_surf.meta = new nlohmann::json(*surf->meta);
    } else {
        rotated_surf.meta = new nlohmann::json();
    }

    for (const auto& pair : rotated_channels) {
        rotated_surf.setChannel(pair.first, pair.second);
    }

    // Recalculate and update the surface area
    double area = vc::surface::computeSurfaceAreaVox2(rotated_surf.rawPoints());
    (*rotated_surf.meta)["area"] = area;

    rotated_surf.save(output_path, true);
    std::cout << "Saved rotated surface to: " << output_path << std::endl;

    delete surf;
    return EXIT_SUCCESS;
}
