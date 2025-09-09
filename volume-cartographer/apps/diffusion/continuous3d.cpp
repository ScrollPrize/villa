#include "continuous3d.hpp"

#include "common.hpp"

#include <vc/ui/VCCollection.hpp>
#include <vc/core/util/GridStore.hpp>
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// A simple tensor implementation based on a vector of cv::Mat_
template <typename T>
class StupidTensor
{
public:
    StupidTensor() {};
    StupidTensor(const StupidTensor<T> &other) {
        create(other.planes[0].size(), other.planes.size());
        for(size_t i=0; i<planes.size(); ++i) {
            other.planes[i].copyTo(planes[i]);
        }
    }
    template <typename O> StupidTensor(const StupidTensor<O> &other) { create(other.planes[0].size(), other.planes.size()); };
    StupidTensor(const cv::Size &size, int d) { create(size, d); };

    StupidTensor<T>& operator=(const StupidTensor<T>& other) {
        if (this != &other) {
            create(other.planes[0].size(), other.planes.size());
            for (size_t i = 0; i < planes.size(); ++i) {
                other.planes[i].copyTo(planes[i]);
            }
        }
        return *this;
    }

    void create(const cv::Size &size, int d)
    {
        planes.resize(d);
        for(auto &p : planes)
            p.create(size);
    }
    void setTo(const T &v)
    {
        for(auto &p : planes)
            p.setTo(v);
    }
    template <typename O> void convertTo(O &out, int code) const
    {
        out.create(planes[0].size(), planes.size());
        for(int z=0;z<planes.size();z++)
            planes[z].convertTo(out.planes[z], code);
    }
    T &at(int z, int y, int x) { return planes[z](y,x); }
    T &operator()(int z, int y, int x) { return at(z,y,x); }
    std::vector<cv::Mat_<T>> planes;
};

int continuous3d_main(const po::variables_map& vm) {

    fs::path points_path = vm["points"].as<std::string>();
    fs::path volume_path = vm["volume"].as<std::string>();
    fs::path output_path = vm["output"].as<std::string>();
    std::string dataset_name = vm["dataset"].as<std::string>();
    double target_winding = vm["winding"].as<double>();
    if (!vm.count("box-size")) {
        std::cerr << "Error: --box-size is required for continuous3d mode." << std::endl;
        return 1;
    }
    if (!vm.count("box-size")) {
        std::cerr << "Error: --box-size is required for continuous3d mode." << std::endl;
        return 1;
    }
    std::vector<int> box_dims = vm["box-size"].as<std::vector<int>>();
    if (box_dims.size() == 1) {
        box_dims.push_back(box_dims[0]);
        box_dims.push_back(box_dims[0]);
    }
    if (box_dims.size() != 3) {
        std::cerr << "Error: --box-size requires one or three values (width, height, depth)." << std::endl;
        return 1;
    }
    int box_w = box_dims[0];
    int box_h = box_dims[1];
    int box_d = box_dims[2];
    int iterations = vm["iterations"].as<int>();

    VCCollection point_collection;
    if (!point_collection.loadFromJSON(points_path.string())) {
        std::cerr << "Error: Failed to load point file: " << points_path << std::endl;
        return 1;
    }

    std::optional<cv::Vec3f> target_point;
    std::string collection_name = vm.count("collection") ? vm["collection"].as<std::string>() : "";

    auto find_point = [&](const auto& collections) {
        for (const auto& [id, collection] : collections) {
            if (!collection_name.empty() && collection.name != collection_name) continue;
            for (const auto& pair : collection.points) {
                const auto& point = pair.second;
                if (std::abs(point.winding_annotation - target_winding) < 1e-6) {
                    target_point = point.p;
                    return;
                }
            }
        }
    };

    find_point(point_collection.getAllCollections());

    if (!target_point) {
        std::cerr << "Error: Point with winding number " << target_winding << " not found." << std::endl;
        return 1;
    }

    std::cout << "Found point " << *target_point << " for winding " << target_winding << std::endl;

    z5::filesystem::handle::Group group_handle(volume_path);
    std::unique_ptr<z5::Dataset> ds = z5::openDataset(group_handle, dataset_name);
    if (!ds) {
        std::cerr << "Error: Could not open dataset '" << dataset_name << "' in volume '" << volume_path << "'." << std::endl;
        return 1;
    }

    auto shape = ds->shape();
    std::cout << "Volume shape: (" << shape[0] << ", " << shape[1] << ", " << shape[2] << ")" << std::endl;

    StupidTensor<uint8_t> volume_slice(cv::Size(box_w, box_h), box_d);
    
    cv::Vec3i offset = {
        static_cast<int>(std::round((*target_point)[2])) - box_d / 2,
        static_cast<int>(std::round((*target_point)[1])) - box_h / 2,
        static_cast<int>(std::round((*target_point)[0])) - box_w / 2
    };

    std::vector<size_t> slice_shape = {(size_t)box_d, (size_t)box_h, (size_t)box_w};
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> slice_data = xt::zeros<uint8_t>(slice_shape);

    ChunkCache cache(4llu*1024*1024*1024);
    readArea3D(slice_data, offset, ds.get(), &cache);

    for (int z = 0; z < box_d; ++z) {
        for (int y = 0; y < box_h; ++y) {
            for (int x = 0; x < box_w; ++x) {
                volume_slice(z, y, x) = slice_data(z, y, x);
            }
        }
    }

    std::cout << "Extracted " << box_w << "x" << box_h << "x" << box_d << " volume around the point." << std::endl;

    int seed_x = box_w / 2;
    int seed_y = box_h / 2;
    int seed_z = box_d / 2;

    cv::imwrite("diffusion_slice_crop.tif", volume_slice.planes[seed_z]);
    std::cout << "Saved debug slice crop to diffusion_slice_crop.tif" << std::endl;

    StupidTensor<float> density(cv::Size(box_w, box_h), box_d);
    density.setTo(0.0f);

    for (int i = 0; i < iterations; ++i) {
        density(seed_z, seed_y, seed_x) += 1000000.0f;

        StupidTensor<float> next_density = density;

        #pragma omp parallel for
        for (int z = 1; z < box_d - 1; ++z) {
            for (int y = 1; y < box_h - 1; ++y) {
                for (int x = 1; x < box_w - 1; ++x) {
                    if (volume_slice(z, y, x) == 0) continue;

                    float current_density = density(z, y, x);
                    if (current_density > 0) {
                        float sum_diff = 0;
                        float diffs[6];
                        
                        int neighbors[6][3] = {
                            {z + 1, y, x}, {z - 1, y, x},
                            {z, y + 1, x}, {z, y - 1, x},
                            {z, y, x + 1}, {z, y, x - 1}
                        };

                        for(int j=0; j<6; ++j) {
                            int nz = neighbors[j][0];
                            int ny = neighbors[j][1];
                            int nx = neighbors[j][2];
                            float diff = 0;
                            if (volume_slice(nz, ny, nx) > 0) {
                                //0.14 < 1/7
                                diff = std::max(0.0, 0.14*(current_density - density(nz, ny, nx)));
                                next_density(nz, ny, nx) += diff;
                            }
                            sum_diff += diff;
                        }

                        next_density(z, y, x) -= sum_diff;
                    }
                }
            }
        }
        density = next_density;
        std::cout << "Iteration " << i+1 << "/" << iterations << std::endl;
    }

    cv::Mat output_slice = density.planes[seed_z];

    cv::Mat normalized_slice = output_slice.clone();
    std::vector<double> row_max_values(box_h, 0.0);

    for (int z = 0; z < box_d; ++z) {
        for (int y = 0; y < box_h; ++y) {
            for (int x = 0; x < box_w; ++x) {
                row_max_values[y] = std::max(row_max_values[y], (double)density(z, y, x));
            }
        }
    }

    for (int y = 0; y < box_h; ++y) {
        for (int x = 0; x < box_w; ++x) {
            normalized_slice.at<float>(y, x) /= row_max_values[y];
        }
    }

    cv::imwrite(output_path.string(), normalized_slice);

    std::cout << "Saved output slice to " << output_path << std::endl;

    return 0;
}
