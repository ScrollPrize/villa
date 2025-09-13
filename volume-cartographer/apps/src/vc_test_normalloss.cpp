#include <boost/program_options.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vc/core/util/CostFunctions.hpp>
#include <vc/core/util/NormalGridVolume.hpp>
#include <vc/core/util/Slicing.hpp>
#include <z5/factory.hxx>
#include <z5/filesystem/handle.hxx>

#include "support.hpp" // For visualize_normal_grid

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "ng-path", po::value<std::string>()->required(),
      "path to normalgrid volume")(
      "vol-path", po::value<std::string>()->required(),
      "path to zarr volume")(
      "seed", po::value<std::vector<double>>()->multitoken()->required(),
      "seed coordinate (x y z)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  try {
    po::notify(vm);
  } catch (const po::error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  const std::string ng_path = vm["ng-path"].as<std::string>();
  const std::string vol_path = vm["vol-path"].as<std::string>();
  const std::vector<double> seed_coords =
      vm["seed"].as<std::vector<double>>();

  if (seed_coords.size() != 3) {
    std::cerr << "Error: --seed requires 3 values for x, y, and z."
              << std::endl;
    return 1;
  }

  const cv::Vec3d seed_point(seed_coords[0], seed_coords[1], seed_coords[2]);

  std::cout << "NormalGrid Path: " << ng_path << std::endl;
  std::cout << "Seed Point: " << seed_point << std::endl;

  vc::core::util::NormalGridVolume ngv(ng_path);

  const double step = ngv.metadata()["spiral-step"];
  std::cout << "Using step size: " << step << std::endl;

  cv::Vec3d p0 = seed_point;
  cv::Vec3d p1 = seed_point + cv::Vec3d(step, 0, 0);
  cv::Vec3d p2 = seed_point + cv::Vec3d(0, step, 0);
  cv::Vec3d p3 = seed_point + cv::Vec3d(step, step, 0);

  ceres::Problem problem;

  // Distance constraints
  problem.AddResidualBlock(DistLoss::Create(step, 1.0), nullptr, &p0[0], &p1[0]);
  problem.AddResidualBlock(DistLoss::Create(step, 1.0), nullptr, &p0[0], &p2[0]);
  problem.AddResidualBlock(DistLoss::Create(step, 1.0), nullptr, &p1[0], &p3[0]);
  problem.AddResidualBlock(DistLoss::Create(step, 1.0), nullptr, &p2[0], &p3[0]);
  problem.AddResidualBlock(DistLoss::Create(step * sqrt(2.0), 1.0), nullptr, &p0[0], &p3[0]);
  problem.AddResidualBlock(DistLoss::Create(step * sqrt(2.0), 1.0), nullptr, &p1[0], &p2[0]);

  // Normal losses
  const float w = 10.0;
  for (int plane_idx = 0; plane_idx < 3; ++plane_idx) {
    problem.AddResidualBlock(
        NormalConstraintPlane::Create(ngv, plane_idx, w), nullptr, &p0[0],
        &p1[0], &p2[0], &p3[0]);
    problem.AddResidualBlock(
        NormalConstraintPlane::Create(ngv, plane_idx, w), nullptr, &p1[0],
        &p3[0], &p0[0], &p2[0]);
    problem.AddResidualBlock(
        NormalConstraintPlane::Create(ngv, plane_idx, w), nullptr, &p2[0],
        &p0[0], &p3[0], &p1[0]);
    problem.AddResidualBlock(
        NormalConstraintPlane::Create(ngv, plane_idx, w), nullptr, &p3[0],
        &p2[0], &p1[0], &p0[0]);
  }


  problem.SetParameterBlockConstant(&p0[0]);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  std::cout << "p0: " << p0 << std::endl;
  std::cout << "p1: " << p1 << std::endl;
  std::cout << "p2: " << p2 << std::endl;
  std::cout << "p3: " << p3 << std::endl;

  z5::filesystem::handle::Group group_handle(vol_path);
  std::unique_ptr<z5::Dataset> ds = z5::openDataset(group_handle, "0");
  if (!ds) {
    std::cerr << "Error: Could not open dataset '0' in volume '" << vol_path
              << "'." << std::endl;
    return 1;
  }

  const int crop_size = 200;
  const int half_crop = crop_size / 2;
  cv::Vec3i offset(static_cast<int>(seed_point[0]) - half_crop,
                   static_cast<int>(seed_point[1]) - half_crop,
                   static_cast<int>(seed_point[2]) - half_crop);

  xt::xtensor<uint8_t, 3, xt::layout_type::column_major> crop =
      xt::zeros<uint8_t>({(size_t)crop_size, (size_t)crop_size, (size_t)crop_size});
  ChunkCache cache(1024*1024*1024);
  readArea3D(crop, offset, ds.get(), &cache);

  cv::Mat slice_xy(crop_size, crop_size, CV_8U);
  cv::Mat slice_xz(crop_size, crop_size, CV_8U);
  cv::Mat slice_yz(crop_size, crop_size, CV_8U);

  for (int i = 0; i < crop_size; ++i) {
    for (int j = 0; j < crop_size; ++j) {
      slice_xy.at<uint8_t>(j, i) = crop(i, j, half_crop);
      slice_xz.at<uint8_t>(j, i) = crop(i, half_crop, j);
      slice_yz.at<uint8_t>(j, i) = crop(half_crop, i, j);
    }
  }

  cv::cvtColor(slice_xy, slice_xy, cv::COLOR_GRAY2BGR);
  cv::cvtColor(slice_xz, slice_xz, cv::COLOR_GRAY2BGR);
  cv::cvtColor(slice_yz, slice_yz, cv::COLOR_GRAY2BGR);

  auto draw_quad = [&](cv::Mat& img, const cv::Vec3d& p0, const cv::Vec3d& p1,
                       const cv::Vec3d& p2, const cv::Vec3d& p3,
                       const cv::Vec3d& offset, int plane_idx) {
    auto project = [&](const cv::Vec3d& p) {
      cv::Vec3d p_local = p - offset;
      if (plane_idx == 0) return cv::Point2f(p_local[1], p_local[2]); // XY
      if (plane_idx == 1) return cv::Point2f(p_local[0], p_local[2]); // XZ
      return cv::Point2f(p_local[0], p_local[1]);                     // YZ
    };

    cv::line(img, project(p0), project(p1), cv::Scalar(0, 0, 255), 1);
    cv::line(img, project(p0), project(p2), cv::Scalar(0, 0, 255), 1);
    cv::line(img, project(p1), project(p3), cv::Scalar(0, 0, 255), 1);
    cv::line(img, project(p2), project(p3), cv::Scalar(0, 0, 255), 1);
  };

  draw_quad(slice_xy, p0, p1, p2, p3, cv::Vec3d(offset[0], offset[1], offset[2]), 0);
  draw_quad(slice_xz, p0, p1, p2, p3, cv::Vec3d(offset[0], offset[1], offset[2]), 1);
  draw_quad(slice_yz, p0, p1, p2, p3, cv::Vec3d(offset[0], offset[1], offset[2]), 2);

  cv::imwrite("slice_xy.tif", slice_xy);
  cv::imwrite("slice_xz.tif", slice_xz);
  cv::imwrite("slice_yz.tif", slice_yz);

  std::cout << "Saved slices to slice_xy.tif, slice_xz.tif, and slice_yz.tif"
            << std::endl;

  for (int plane_idx = 0; plane_idx < 3; ++plane_idx) {
    auto result = ngv.query({(float)seed_point[0], (float)seed_point[1], (float)seed_point[2]}, plane_idx);
    if (result) {
        for (const auto* grid_ptr : {result->grid1, result->grid2}) {
            if (grid_ptr) {
                cv::Mat vis = visualize_normal_grid(*grid_ptr, grid_ptr->size());
                
                cv::Rect crop_rect(offset[plane_idx == 0 ? 1 : 0], offset[plane_idx == 2 ? 1 : 2], crop_size, crop_size);
                cv::Mat vis_crop = vis(crop_rect);

                cv::Vec3d grid_offset = {0,0,0};
                grid_offset[plane_idx == 0 ? 1 : 0] = offset[plane_idx == 0 ? 1 : 0];
                grid_offset[plane_idx == 2 ? 1 : 2] = offset[plane_idx == 2 ? 1 : 2];

                draw_quad(vis_crop, p0, p1, p2, p3, grid_offset, plane_idx);
                
                std::string plane_str = (plane_idx == 0) ? "xy" : ((plane_idx == 1) ? "xz" : "yz");
                static int grid_vis_count = 0;
                std::string filename = "ng_" + plane_str + "_" + std::to_string(grid_vis_count++) + ".tif";
                cv::imwrite(filename, vis_crop);
                std::cout << "Saved normal grid visualization to " << filename << std::endl;
            }
        }
    }
  }

  return 0;
}