/*
Author: Julian Schilliger - ThaumatoAnakalyptor - 2024
*/ 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <archive.h>
#include <archive_entry.h>

#include <filesystem>  // Requires C++17
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <mutex>
#include <execution>
#include <tbb/tbb.h>
#include <thread>
#include <iomanip>
#include <cstdio>    // For mkstemp
#include <unistd.h>  // For unlink
#include <filesystem>
#include "happly.h"
#include "nanoflann.h"
#include <omp.h>
#include <tuple>
#include <cmath>
#include "hdf5.h"

namespace fs = std::filesystem;
namespace py = pybind11;
namespace nf = nanoflann;

struct Point {
    float x, y, z, w;   // Coordinates and winding angle
    float nx, ny, nz;   // Normal vector components
    unsigned char r, g, b; // Color components
    bool marked_for_deletion = false;  // Flag to mark points for deletion

    // Default constructor
    Point() = default;

    // Define a constructor that initializes all members
    explicit Point(float x, float y, float z, float w,
          float nx, float ny, float nz,
          unsigned char r, unsigned char g, unsigned char b,
          bool marked_for_deletion)
        : x(x), y(y), z(z), w(w),
          nx(nx), ny(ny), nz(nz),
          r(r), g(g), b(b),
          marked_for_deletion(marked_for_deletion) {}
};

class PointCloud {
public:
    std::vector<Point> pts;

    // Default constructor
    PointCloud() = default;

    // Initialize with a vector of points
    explicit PointCloud(const std::vector<Point>& points) : pts(points) {}

    // Initialize with a NumPy array
    explicit PointCloud(py::array_t<float> points, py::array_t<float> normals, py::array_t<float> colors) {
        auto points_r = points.unchecked<2>();
        auto normals_r = normals.unchecked<2>();
        auto colors_r = colors.unchecked<2>();

        size_t total_points = points_r.shape(0);
        pts.reserve(total_points);

        for (size_t i = 0; i < total_points; ++i) {
            pts.emplace_back(
                points_r(i, 0), points_r(i, 1), points_r(i, 2), points_r(i, 3), // coordinates and winding angle
                normals_r(i, 0), normals_r(i, 1), normals_r(i, 2),  // normal vector components
                static_cast<unsigned char>(colors_r(i, 0) * 255), static_cast<unsigned char>(colors_r(i, 1) * 255), static_cast<unsigned char>(colors_r(i, 2) * 255),  // color components
                false  // not marked for deletion
            );
        }
    }

    // Initialize with a NumPy array, points only
    explicit PointCloud(py::array_t<float> points) {
        auto points_r = points.unchecked<2>();

        size_t total_points = points_r.shape(0);
        pts.reserve(total_points);

        for (size_t i = 0; i < total_points; ++i) {
            pts.emplace_back(
                points_r(i, 0), points_r(i, 1), points_r(i, 2), points_r(i, 3), // coordinates and winding angle
                0.0f, 0.0f, 0.0f,  // normal vector components (defaults)
                0, 0, 0,  // color components (defaults)
                false  // not marked for deletion (defaults)
            );
        }
    }

    // Move constructor
    PointCloud(PointCloud&& other) noexcept 
        : pts(std::move(other.pts)) { }

    // Add a point to the cloud
    void addPoint(const Point& point) {
        pts.push_back(point);
    }

    // Get number of points in the cloud
    size_t size() const {
        return pts.size();
    }

    // Clear all points
    void clear() {
        pts.clear();
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else if (dim == 2) return pts[idx].z;
        return 0; // Should never be reached
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef nf::KDTreeSingleIndexAdaptor<
    nf::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    3, // dimensionality
    size_t // using size_t for indexing
> MyKDTree;

std::string format_filename(int number) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << number;
    return stream.str();
}

class PointCloudLoader {
public:
    PointCloudLoader(const std::vector<std::tuple<std::vector<int>, int, double>>& node_data,
                     const std::string& base_path,
                     const int start_z,
                     const int end_z,
                     bool verbose)
        : node_data_(node_data), base_path_(base_path),
          start_z_(start_z), end_z_(end_z), verbose(verbose) {}

    bool file_exists(const std::string& name) {
        return std::filesystem::exists(name);
    }

    bool extract_ply_from_archive(const std::string& archive_path,
                                  const std::string& ply_filename,
                                  std::string& out_ply_content)
    {
        // RAII struct to ensure archive is freed
        struct ArchiveGuard {
            archive* arch;
            ArchiveGuard(archive* a) : arch(a) {}
            ~ArchiveGuard() {
                if (arch) {
                    archive_read_free(arch);
                }
            }
        };

        archive* a = archive_read_new();
        if (!a) {
            return false;
        }
        ArchiveGuard guard(a);

        archive_read_support_filter_all(a);
        archive_read_support_format_all(a);

        int r = archive_read_open_filename(a, archive_path.c_str(), 65536); // 64 KB buffer
        if (r != ARCHIVE_OK) {
            return false;
        }

        struct archive_entry* entry;
        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string currentFile = archive_entry_pathname(entry);
            if (currentFile == ply_filename) {
                const void* buff;
                size_t size;
                la_int64_t offset;
                std::ostringstream oss;
                while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
                    oss.write(static_cast<const char*>(buff), size);
                }
                out_ply_content = oss.str();
                return true;
            }
            archive_read_data_skip(a);
        }
        return false;
    }

    void process_node(size_t start, size_t end) {
        std::string h5_path = base_path_ + ".h5";
        bool h5_exists = file_exists(h5_path);
        for (size_t index = start; index < end; ++index) {
            auto& node = node_data_[index];
            const auto& xyz = std::get<0>(node);
            size_t patch_nr = std::get<1>(node);
            double winding_angle = std::get<2>(node);

            // Skip nodes outside the Z-range
            if (xyz[1] < start_z_ || xyz[1] > end_z_) {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
                continue;
            }

            // Construct a unique base filename using the node's xyz values
            std::string base_filename = base_path_ + "/" +
                format_filename(xyz[0]) + "_" +
                format_filename(xyz[1]) + "_" +
                format_filename(xyz[2]);

            // Check if an HDF5 file exists (named as base_path_ + ".h5")
            if (h5_exists) {
                // Compute the group name (basename of base_filename)
                std::string group_name = get_basename(base_filename);
                std::string surface_group_name = "surface_" + std::to_string(patch_nr);

                // Open HDF5 file in read-only mode.
                hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                if (file_id < 0) {
                    std::cerr << "Error opening HDF5 file: " << h5_path << std::endl;
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                // Open the group (if it exists)
                hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
                if (group_id < 0) {
                    std::cerr << "Group " << group_name << " not found in HDF5 file: " << h5_path << std::endl;
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                // Open the subgroup for the surface data.
                hid_t surface_group_id = H5Gopen2(group_id, surface_group_name.c_str(), H5P_DEFAULT);
                if (surface_group_id < 0) {
                    std::cerr << "Surface group " << surface_group_name << " not found in group " << group_name << std::endl;
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                // --- Read the "points" dataset ---
                hid_t points_dset = H5Dopen2(surface_group_id, "points", H5P_DEFAULT);
                if (points_dset < 0) {
                    std::cerr << "Dataset 'points' not found in " << surface_group_name << std::endl;
                    H5Gclose(surface_group_id);
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t points_space = H5Dget_space(points_dset);
                int ndims = H5Sget_simple_extent_ndims(points_space);
                if (ndims != 2) {
                    std::cerr << "Dataset 'points' does not have 2 dimensions" << std::endl;
                    H5Sclose(points_space);
                    H5Dclose(points_dset);
                    H5Gclose(surface_group_id);
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hsize_t dims[2];
                H5Sget_simple_extent_dims(points_space, dims, nullptr);
                size_t num_points = dims[0]; // Number of vertices
                // Read points data (assumed to be stored as double, shape: [num_points x 3])
                std::vector<double> points_data(num_points * 3);
                H5Dread(points_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, points_data.data());
                H5Sclose(points_space);
                H5Dclose(points_dset);

                // --- Read the "normals" dataset ---
                hid_t normals_dset = H5Dopen2(surface_group_id, "normals", H5P_DEFAULT);
                if (normals_dset < 0) {
                    std::cerr << "Dataset 'normals' not found in " << surface_group_name << std::endl;
                    H5Gclose(surface_group_id);
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t normals_space = H5Dget_space(normals_dset);
                int ndims_normals = H5Sget_simple_extent_ndims(normals_space);
                hsize_t dims_normals[2];
                H5Sget_simple_extent_dims(normals_space, dims_normals, nullptr);
                std::vector<double> normals_data(num_points * 3);
                H5Dread(normals_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, normals_data.data());
                H5Sclose(normals_space);
                H5Dclose(normals_dset);

                // --- Read the "colors" dataset ---
                hid_t colors_dset = H5Dopen2(surface_group_id, "colors", H5P_DEFAULT);
                if (colors_dset < 0) {
                    std::cerr << "Dataset 'colors' not found in " << surface_group_name << std::endl;
                    H5Gclose(surface_group_id);
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t colors_space = H5Dget_space(colors_dset);
                int ndims_colors = H5Sget_simple_extent_ndims(colors_space);
                hsize_t dims_colors[2];
                H5Sget_simple_extent_dims(colors_space, dims_colors, nullptr);
                std::vector<unsigned char> colors_data(num_points * 3);
                H5Dread(colors_dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, colors_data.data());
                H5Sclose(colors_space);
                H5Dclose(colors_dset);

                // Close groups and file
                H5Gclose(surface_group_id);
                H5Gclose(group_id);
                H5Fclose(file_id);

                // Convert the raw data into points
                std::vector<Point> points;
                points.reserve(num_points);
                for (size_t i = 0; i < num_points; ++i) {
                    double x = points_data[i * 3 + 0];
                    double y = points_data[i * 3 + 1];
                    double z = points_data[i * 3 + 2];
                    bool delete_range = (y < start_z_ || y > end_z_);
                    points.emplace_back(
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z),
                        static_cast<float>(winding_angle),
                        static_cast<float>(normals_data[i * 3 + 0]),
                        static_cast<float>(normals_data[i * 3 + 1]),
                        static_cast<float>(normals_data[i * 3 + 2]),
                        colors_data[i * 3 + 0],
                        colors_data[i * 3 + 1],
                        colors_data[i * 3 + 2],
                        delete_range
                    );
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    cloud_.pts.insert(cloud_.pts.end(), points.begin(), points.end());
                }
            }
            else {
                // Fallback to reading from TAR/7z archive
                std::string archive_path;
                if (file_exists(base_filename + ".tar")) {
                    archive_path = base_filename + ".tar";
                }
                else if (file_exists(base_filename + ".7z")) {
                    archive_path = base_filename + ".7z";
                }
                else {
                    std::cerr << "No archive found for " << base_filename << std::endl;
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                std::string ply_file_name = "surface_" + std::to_string(patch_nr) + ".ply";
                std::string ply_content;
                if (!extract_ply_from_archive(archive_path, ply_file_name, ply_content)) {
                    std::cerr << "Failed to extract PLY file from archive: " << archive_path << std::endl;
                    continue;
                }

                try {
                    std::istringstream plyStream(ply_content);
                    happly::PLYData plyIn(plyStream);
                    auto vertices = plyIn.getVertexData();

                    std::vector<double> x = std::get<0>(std::get<0>(vertices));
                    std::vector<double> y = std::get<1>(std::get<0>(vertices));
                    std::vector<double> z = std::get<2>(std::get<0>(vertices));

                    std::vector<double> nx = std::get<0>(std::get<1>(vertices));
                    std::vector<double> ny = std::get<1>(std::get<1>(vertices));
                    std::vector<double> nz = std::get<2>(std::get<1>(vertices));

                    std::vector<unsigned char> r = std::get<0>(std::get<2>(vertices));
                    std::vector<unsigned char> g = std::get<1>(std::get<2>(vertices));
                    std::vector<unsigned char> b = std::get<2>(std::get<2>(vertices));

                    std::vector<Point> points;
                    points.reserve(x.size());
                    for (size_t i = 0; i < x.size(); ++i) {
                        bool delete_range = y[i] < start_z_ || y[i] > end_z_;
                        points.emplace_back(
                            static_cast<float>(x[i]),
                            static_cast<float>(y[i]),
                            static_cast<float>(z[i]),
                            static_cast<float>(winding_angle),
                            static_cast<float>(nx[i]),
                            static_cast<float>(ny[i]),
                            static_cast<float>(nz[i]),
                            r[i], g[i], b[i],
                            delete_range
                        );
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        print_progress();
                        cloud_.pts.insert(cloud_.pts.end(), points.begin(), points.end());
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing node: " << e.what() << std::endl;
                }
            }
        }
    }

    void find_vertex_counts(size_t start, size_t end) {
        std::string h5_path = base_path_ + ".h5";
        bool h5_exists = file_exists(h5_path);
        for (size_t index = start; index < end; ++index) {
            const auto& xyz = std::get<0>(node_data_[index]);
            size_t patch_nr = std::get<1>(node_data_[index]);

            if (xyz[1] < start_z_ || xyz[1] > end_z_) {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
                continue;
            }

            std::string base_filename = base_path_ + "/" +
                format_filename(xyz[0]) + "_" +
                format_filename(xyz[1]) + "_" +
                format_filename(xyz[2]);

            if (h5_exists) {
                std::string group_name = get_basename(base_filename);
                std::string surface_group_name = "surface_" + std::to_string(patch_nr);

                hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                if (file_id < 0) {
                    std::cerr << "Error opening HDF5 file: " << h5_path << std::endl;
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
                if (group_id < 0) {
                    std::cerr << "Group " << group_name << " not found in HDF5 file: " << h5_path << std::endl;
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t surface_group_id = H5Gopen2(group_id, surface_group_name.c_str(), H5P_DEFAULT);
                if (surface_group_id < 0) {
                    std::cerr << "Surface group " << surface_group_name << " not found in group " << group_name << std::endl;
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                // Open "points" dataset to get the number of vertices.
                hid_t points_dset = H5Dopen2(surface_group_id, "points", H5P_DEFAULT);
                if (points_dset < 0) {
                    std::cerr << "Dataset 'points' not found in " << surface_group_name << std::endl;
                    H5Gclose(surface_group_id);
                    H5Gclose(group_id);
                    H5Fclose(file_id);
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }
                hid_t points_space = H5Dget_space(points_dset);
                int ndims = H5Sget_simple_extent_ndims(points_space);
                hsize_t dims[2];
                H5Sget_simple_extent_dims(points_space, dims, nullptr);
                size_t num_points = dims[0];
                offset_per_node[index] = num_points;
                H5Sclose(points_space);
                H5Dclose(points_dset);
                H5Gclose(surface_group_id);
                H5Gclose(group_id);
                H5Fclose(file_id);
            }
            else {
                // Fallback to using the archive.
                std::string archive_path;
                if (file_exists(base_filename + ".tar")) {
                    archive_path = base_filename + ".tar";
                }
                else if (file_exists(base_filename + ".7z")) {
                    archive_path = base_filename + ".7z";
                }
                else {
                    std::cerr << "No archive found for " << base_filename << std::endl;
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                    continue;
                }

                std::string ply_file_name = "surface_" + std::to_string(patch_nr) + ".ply";
                std::string ply_content;
                if (extract_ply_from_archive(archive_path, ply_file_name, ply_content)) {
                    try {
                        std::istringstream plyStream(ply_content);
                        happly::PLYData plyIn(plyStream);
                        // if (plyIn.hasElement("vertex")) {
                        //     offset_per_node[index] = plyIn.getElement("vertex").getProperty<double>("x").size();
                        // }
                        // Faster way
                        auto vertices = plyIn.getVertexData();
                        offset_per_node[index] = std::get<0>(std::get<0>(vertices)).size();
                        
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error counting points in node: " << e.what() << std::endl;
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mutex_);
            print_progress();
        }
    }

    size_t find_total_points() {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t total_nodes = node_data_.size();
        size_t chunk_size = std::ceil(static_cast<double>(total_nodes) / num_threads);
        
        problem_size = total_nodes;
        progress = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::find_vertex_counts, this, start, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }

        if (verbose) {
            std::cout << std::endl;
        }
        problem_size = -1;
        progress = 0;

        size_t total_points = 0;
        size_t total_points_temp = 0;
        for (size_t i = 0; i < total_nodes; ++i) {
            total_points_temp = total_points;
            total_points += offset_per_node[i];
            offset_per_node[i] = total_points_temp;
        }
        return total_points;
    }

    void load_total_points(size_t total_nodes) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = std::ceil(static_cast<double>(total_nodes) / num_threads);

        problem_size = total_nodes;
        progress = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::process_node, this, start, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    void load_all() {
        size_t total_nodes = node_data_.size();
        offset_per_node = std::make_unique<size_t[]>(total_nodes);
        if (verbose) {
            std::cout << "Loading all nodes..." << std::endl;
        }
        size_t total_points = find_total_points();
        cloud_.pts.reserve(total_points);
        if (verbose) {
            std::cout << "Total points: " << total_points << std::endl;
        }

        load_total_points(total_nodes);
        if (verbose) {
            std::cout << std::endl << "All nodes have been processed." << std::endl;
        }
    }

    void free_memory() {
        cloud_.pts.clear();
        cloud_.pts.shrink_to_fit();
    }

    void deleteMarkedPoints() {
        std::cout << "Deleting marked points..." << std::endl;
        try {
            cloud_.pts.erase(std::remove_if(cloud_.pts.begin(), cloud_.pts.end(), [](const Point& p) {
                return p.marked_for_deletion;
            }), cloud_.pts.end());
        }
        catch (...) {
            std::cerr << "Error deleting marked points" << std::endl;
        }
    }

    void sortPointsWZYX() {
        unsigned int available_threads = std::thread::hardware_concurrency();
        unsigned int num_threads = std::min(available_threads, 32u);
    
        tbb::task_arena limited_arena(num_threads);
        limited_arena.execute([&] {
            std::sort(std::execution::par_unseq, cloud_.pts.begin(), cloud_.pts.end(), [](const Point& a, const Point& b) {
                if (a.w != b.w) return a.w < b.w;
                if (a.z != b.z) return a.z < b.z;
                if (a.y != b.y) return a.y < b.y;
                return a.x < b.x;
            });
        });
    }

    void sortPointsXYZW() {
        unsigned int available_threads = std::thread::hardware_concurrency();
        unsigned int num_threads = std::min(available_threads, 32u);
    
        tbb::task_arena limited_arena(num_threads);
        limited_arena.execute([&] {
            std::sort(std::execution::par_unseq, cloud_.pts.begin(), cloud_.pts.end(), [](const Point& a, const Point& b) {
                if (a.x != b.x) return a.x < b.x;
                if (a.y != b.y) return a.y < b.y;
                if (a.z != b.z) return a.z < b.z;
                return a.w < b.w;
            });
        });
    }

    void processDuplicates() {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t total_points = cloud_.size();
        size_t chunk_size = total_points / num_threads;

        std::vector<size_t> chunk_starts = getChunkStarts(num_threads, total_points, chunk_size);
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(&PointCloudLoader::processDuplicatesThreaded, this,
                                 chunk_starts[i],
                                 (i + 1 < num_threads) ? chunk_starts[i + 1] : total_points);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    void filterPointsUsingKDTree(double spatial_threshold, double angle_threshold) {
        std::unique_ptr<MyKDTree> index;
        // Create a KD-tree for 3D points
        try {
            index = std::make_unique<MyKDTree>(3 /*dim*/, cloud_, nf::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        } catch (...) {
            std::cerr << "Error creating KD-tree, returning" << std::endl;
            return;
        }
        if (index == nullptr) {
            std::cerr << "Error creating KD-tree, returning" << std::endl;
            return;
        }
        try {
            std ::cout << "Building index..." << std::endl;
            index->buildIndex();
        }
        catch (...) {
            std::cerr << "Error building KD-tree" << std::endl;
            return;
        }

        const size_t num_threads = std::thread::hardware_concurrency(); // Number of concurrent threads supported
        std::vector<std::thread> threads(num_threads);
        size_t part_length = cloud_.pts.size() / num_threads;
        progress = 0;
        problem_size = cloud_.pts.size() / 1000;

        std::cout << "Processing points using KD-tree..." << std::endl;
        for (size_t i = 0; i < num_threads; ++i) {
            try {
                size_t start = i * part_length;
                size_t end = (i == num_threads - 1) ? cloud_.pts.size() : start + part_length;
                threads[i] = std::thread(&PointCloudLoader::processSubset, this, index.get(), start, end, spatial_threshold, angle_threshold);
            }
            catch (...) {
                std::cerr << "Error processing subset" << std::endl;
            }
        }

        // Join all threads
        for (auto& thread : threads) {
            thread.join();
        }
        if (verbose) {
            std::cout << std::endl;
        }
        progress = 0;

        std :: cout << "Finished processing points using KD-tree" << std::endl;
        // Apply deletions
        deleteMarkedPoints();
    }

    PointCloud get_results() {
        return std::move(cloud_);
    }

private:
    const int start_z_;
    const int end_z_;
    std::vector<std::tuple<std::vector<int>, int, double>> node_data_;
    PointCloud cloud_;
    std::unique_ptr<size_t[]> offset_per_node;
    std::string base_path_;
    mutable std::mutex mutex_;
    size_t progress = 0;
    size_t problem_size = -1;
    bool verbose;

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // Print every 1% progress
        if (problem_size <= 0 || progress % (problem_size / 100) == 0) {
            std::cout << "Progress: " << progress << "/" << problem_size << "\r";
            std::cout.flush();
        }
    }

    std::vector<size_t> getChunkStarts(size_t num_threads, size_t total_points, size_t chunk_size) {
        std::vector<size_t> chunk_starts(num_threads);
        size_t start = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            chunk_starts[i] = start;
            size_t end = std::min(start + chunk_size, total_points);
            if (end < total_points) {
                while (end < total_points &&
                       cloud_.pts[end - 1].x == cloud_.pts[end].x &&
                       cloud_.pts[end - 1].y == cloud_.pts[end].y &&
                       cloud_.pts[end - 1].z == cloud_.pts[end].z) {
                    ++end;
                }
            }
            start = end;
        }
        return chunk_starts;
    }

    void processDuplicatesThreaded(size_t start, size_t end) {
        auto it = cloud_.pts.begin() + start;
        auto finish = cloud_.pts.begin() + end;

        while (it != finish) {
            auto next = it + 1;
            while (next != finish &&
                   next->x == it->x && next->y == it->y && next->z == it->z) {
                ++next;
            }
            processRange(it, next);
            it = next;
        }
    }

    void processRange(std::vector<Point>::iterator begin, std::vector<Point>::iterator end) {
        if (begin == end) return;

        double x = begin->x;
        double y = begin->y;
        double z = begin->z;
        for (auto it = begin + 1; it != end; ++it) {
            if (it->x != x || it->y != y || it->z != z) {
                std::cout << "Error: Points in range do not have the same x y z values" << std::endl;
                std::cout << "Range: " << begin->x << " " << begin->y << " " << begin->z << " to "
                          << (end - 1)->x << " " << (end - 1)->y << " " << (end - 1)->z << std::endl;
                return;
            }
        }

        size_t best_w_index_range = 0;
        double best_w = begin->w;
        double sum_selected_w = 0;
        size_t count = 0;

        auto start_it = begin;
        auto end_it = begin;
        while (end_it != end) {
            sum_selected_w += end_it->w;
            ++count;
            double mean_iterative_w = sum_selected_w / count;
            while (start_it != end_it &&
                   !(mean_iterative_w - 90 <= start_it->w && start_it->w <= mean_iterative_w + 90)) {
                sum_selected_w -= start_it->w;
                --count;
                ++start_it;
                mean_iterative_w = sum_selected_w / count;
            }
            if (end_it - start_it + 1 > best_w_index_range) {
                best_w_index_range = end_it - start_it + 1;
                best_w = mean_iterative_w;
            }
            ++end_it;
        }
        for (auto it = begin; it != end; ++it) {
            it->marked_for_deletion = true;
        }
        for (auto it = begin; it != end; ++it) {
            if (std::abs(it->w - best_w) < std::abs(begin->w - best_w)) {
                begin = it;
            }
        }
        begin->marked_for_deletion = false;
    }

    void processSubset(MyKDTree* index, size_t start, size_t end, double spatial_threshold, double angle_threshold) {
        for (size_t i = start; i < end; ++i) {
            if (i >= cloud_.pts.size()) {
                continue;
            }
            try {
                std::vector<nf::ResultItem<size_t, double>> ret_matches;
                nf::SearchParameters params;
                const double query_pt[3] = { cloud_.pts[i].x, cloud_.pts[i].y, cloud_.pts[i].z };
                const double radius = spatial_threshold * spatial_threshold;
                index->radiusSearch(&query_pt[0], radius, ret_matches, params);
                for (auto& match : ret_matches) {
                    if (match.first >= cloud_.pts.size()) {
                        continue;
                    }
                    if (i != match.first &&
                        std::abs(cloud_.pts[i].w - cloud_.pts[match.first].w) > angle_threshold) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        cloud_.pts[i].marked_for_deletion = true;
                        cloud_.pts[match.first].marked_for_deletion = true;
                    }
                }
                if (i % 1000 == 0) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    print_progress();
                }
            }
            catch (...) {
                // Handle errors silently.
            }
        }
    }

    // Helper: returns the basename of a given path.
    std::string get_basename(const std::string& path) {
        return std::filesystem::path(path).filename().string();
    }
};

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> to_array(const PointCloud& cloud) {
    // Create NumPy arrays for points, normals, and colors
    size_t total_points = cloud.size();
    py::array_t<float> points, normals, colors;
    points = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)4});
    normals = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)3});
    colors = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)3});

    auto pts = points.mutable_unchecked<2>();  // for direct access without bounds checking
    auto nrm = normals.mutable_unchecked<2>();
    auto clr = colors.mutable_unchecked<2>();

    // add the data to the numpy arrays
    for (size_t i = 0; i < total_points; ++i) {
        // add points x y z and winding angle to points
        pts(i, 0) = cloud.pts[i].x;
        pts(i, 1) = cloud.pts[i].y;
        pts(i, 2) = cloud.pts[i].z;
        pts(i, 3) = cloud.pts[i].w;

        // add normals nx ny nz to normals
        nrm(i, 0) = cloud.pts[i].nx;
        nrm(i, 1) = cloud.pts[i].ny;
        nrm(i, 2) = cloud.pts[i].nz;

        // add colors r g b to colors
        clr(i, 0) = static_cast<float>(cloud.pts[i].r) / 255.0;
        clr(i, 1) = static_cast<float>(cloud.pts[i].g) / 255.0;
        clr(i, 2) = static_cast<float>(cloud.pts[i].b) / 255.0;
    }

    return std::make_tuple(points, normals, colors);
}

py::array_t<bool> vector_to_array(std::vector<bool> selected_originals) {
    // Create NumPy arrays for points mask
    size_t total_points = selected_originals.size();
    py::array_t<bool> points_mask;
    points_mask = py::array_t<bool>(py::array::ShapeContainer{total_points});

    auto pts_mask = points_mask.mutable_unchecked<1>();  // for direct access without bounds checking

    // add the data to the numpy arrays
    for (size_t i = 0; i < total_points; ++i) {
        // add mask entry
        pts_mask(i) = selected_originals[i];
    }

    return points_mask;
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> load_pointclouds(const std::vector<std::tuple<std::vector<int>, int, double>>& nodes, const std::string& path, const int start_z, const int end_z, bool verbose = true) {
    PointCloudLoader processor(nodes, path, start_z, end_z, verbose);
    processor.load_all();
    processor.sortPointsXYZW();
    if (verbose) {
        std::cout << "Sorted points by XYZW" << std::endl;
    }
    processor.processDuplicates();
    if (verbose) {
        std::cout << "Processed duplicates" << std::endl;
    }
    processor.deleteMarkedPoints();
    if (verbose) {
        std::cout << "Deleted marked points" << std::endl;
    }
    processor.filterPointsUsingKDTree(2.0, 90.0);
    if (verbose) {
        std::cout << "Filtered points using KDTree" << std::endl;
    }
    processor.sortPointsWZYX();
    if (verbose) {
        std::cout << "Sorted points by WZYX" << std::endl;
    }
    PointCloud processed_points = std::move(processor.get_results());
    return to_array(std::move(processed_points));
}

py::array_t<bool> upsample_pointclouds(py::array_t<float> original_points, py::array_t<float> selected_subsamples, py::array_t<float> unselected_subsamples) {
    // Create KD-trees for selected and unselected points
    PointCloud selectedPC(selected_subsamples);
    PointCloud unselectedPC(unselected_subsamples);
    MyKDTree tree_selected(3 /*dim*/, selectedPC, {10 /* max leaf */});
    MyKDTree tree_unselected(3 /*dim*/, unselectedPC, {10 /* max leaf */});
    tree_selected.buildIndex();
    tree_unselected.buildIndex();

    std::vector<bool> selected_originals;

    // Assign each original point to selected or unselected based on closest distance
    auto original_points_r = original_points.unchecked<2>();
    std::cout << "Original points shape: " << original_points_r.shape(0) << std::endl;
    for (size_t i = 0; i < original_points.shape(0); ++i) {
        Point pt;
        pt.x = original_points_r(i, 0);
        pt.y = original_points_r(i, 1);
        pt.z = original_points_r(i, 2);
        pt.w = original_points_r(i, 3);
        double query_pt[3] = { pt.x, pt.y, pt.z };
        size_t closest_idx;
        double out_dist_sqr;

        tree_selected.knnSearch(&query_pt[0], 1, &closest_idx, &out_dist_sqr);
        double dist_selected = out_dist_sqr;
        double w_selected = selectedPC.pts[closest_idx].w;  // Assuming PointCloud and MyKDTree provide access to points

        tree_unselected.knnSearch(&query_pt[0], 1, &closest_idx, &out_dist_sqr);
        double dist_unselected = out_dist_sqr;
        double w_unselected = unselectedPC.pts[closest_idx].w;  // Assuming PointCloud and MyKDTree provide access to points

        if (dist_selected > 2.0 && dist_unselected > 2.0) {
            selected_originals.push_back(false);
        }
        else if (dist_selected <= dist_unselected) {
            if (std::abs(pt.w - w_selected) <= 90) {
                selected_originals.push_back(true);
            }
            else {
                selected_originals.push_back(false);
            }
        }
        else {
            if (std::abs(pt.w - w_unselected) <= 90) {
                selected_originals.push_back(true);
            }
            else {
                selected_originals.push_back(false);
            }
        }
    }
    std::cout << "Selected originals size: " << selected_originals.size() << std::endl;
    // Convert selected originals back to numpy array
    return vector_to_array(selected_originals);
}

std::vector<float> umbilicus_xz_at_y(const std::vector<std::vector<float>>& points_array, float y_new) {
    // Resultant vector of interpolated points
    std::vector<float> interpolated_point = points_array[0];
    interpolated_point[1] = y_new;

    // Check if points_array is not empty
    if (points_array.empty()) {
        throw std::invalid_argument("The points array cannot be empty.");
    }

    // Linear interpolation function
    auto linear_interp = [](float x0, float x1, float y0, float y1, float y) {
        if (y0 == y1) {
            return x0;
        }
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0);
    };

    if (points_array[0][1] >= y_new && points_array[0][1] >= y_new) {
        interpolated_point = points_array[0];
        interpolated_point[1] = y_new;
    }
    else if (points_array[points_array.size() - 1][1] <= y_new && points_array[points_array.size() - 1][1] <= y_new) {
        interpolated_point = points_array[points_array.size() - 1];
        interpolated_point[1] = y_new;
    }
    else {
        // Use binary search to find the segment where y_new lies between
        auto it = std::lower_bound(points_array.begin(), points_array.end(), y_new,
            [](const std::vector<float>& point, float value) {
                return point[1] < value;
            });

        // `it` now points to the first element with a y value greater than y_new
        // We interpolate between this element and the one before it
        size_t i = std::distance(points_array.begin(), it) - 1;
        // Perform interpolation
        float x_new = linear_interp(points_array[i][0], points_array[i + 1][0], points_array[i][1], points_array[i + 1][1], y_new);
        float z_new = linear_interp(points_array[i][2], points_array[i + 1][2], points_array[i][1], points_array[i + 1][1], y_new);

        // Add the new point to the list of interpolated points
        interpolated_point = {x_new, y_new, z_new};
    }

    return interpolated_point;
}

// Helper to determine min and max from vector of points
std::pair<float, float> findMinMaxWindingAngles(const std::vector<std::vector<float>>& points) {
    float minAngle = std::numeric_limits<float>::max();
    float maxAngle = std::numeric_limits<float>::lowest();
    for (const auto& point : points) {
        if (point[3] < minAngle) minAngle = point[3];
        if (point[3] > maxAngle) maxAngle = point[3];
    }
    return {minAngle, maxAngle};
}

// Helper to determine min and max from vector of points
std::pair<float, float> findMinMaxZ(const std::vector<std::vector<float>>& points) {
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const auto& point : points) {
        if (point[1] < minZ) minZ = point[1];
        if (point[1] > maxZ) maxZ = point[1];
    }
    return {minZ, maxZ};
}

class EfficientPointsetProcessor {
public:
    EfficientPointsetProcessor(bool verbose) : verbose(verbose) {}

    std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> create_ordered_pointset_processor(
        py::array_t<float> original_points,
        py::array_t<float> original_normals,
        py::array_t<float> umbilicus_points,
        float min_z, float max_z,
        float angleStep = 6, int z_spacing = 10, float max_eucledian_distance = 10,
        float min_wind = 0.0f, float max_wind = 0.0f
        )
    {
        // Check the input dimensions and types are as expected
        if (original_points.ndim() != 2 || original_normals.ndim() != 2) {
            throw std::runtime_error("Expected two-dimensional array for points and normals.");
        }

        if (original_points.shape(1) != 4 || original_normals.shape(1) != 3) {
            throw std::runtime_error("Expected each point to have four and each normal to have three components.");
        }

        // Access the data
        auto points_buf = original_points.unchecked<2>(); // Accessing the data without bounds checking for performance
        auto normals_buf = original_normals.unchecked<2>();
        auto umbilicus_points_buf = umbilicus_points.unchecked<2>();

        // create a vector of vectors to hold the processed points
        std::vector<std::vector<float>> processed_points;
        processed_points.reserve(original_points.shape(0)); // reserve space for all points to improve performance

        // create a vector of vectors to hold the processed normals
        std::vector<std::vector<float>> processed_normals;
        processed_normals.reserve(original_normals.shape(0)); // reserve space for all normals to improve performance

        // create a vector of vectors to hold the umbilicus points
        std::vector<std::vector<float>> umbilicus_points_vector;
        umbilicus_points_vector.reserve(umbilicus_points.shape(0)); // reserve space for all umbilicus points to improve performance

        // Process points: just a placeholder for actual operations
        for (size_t i = 0; i < original_points.shape(0); ++i) {
            std::vector<float> point = {
                points_buf(i, 0), // x coordinate
                points_buf(i, 1), // y coordinate
                points_buf(i, 2),  // z coordinate
                points_buf(i, 3),  // winding angle
            };
            processed_points.push_back(point);

            std::vector<float> normal = {
                normals_buf(i, 0), // x component
                normals_buf(i, 1), // y component
                normals_buf(i, 2), // z component
            };
            processed_normals.push_back(normal);
        }

        // Process umbilicus points
        for (size_t i = 0; i < umbilicus_points.shape(0); ++i) {
            std::vector<float> umbilicus_point = {
                umbilicus_points_buf(i, 0), // x coordinate
                umbilicus_points_buf(i, 1), // y coordinate
                umbilicus_points_buf(i, 2),  // z coordinate
            };
            umbilicus_points_vector.push_back(umbilicus_point);
        }

        auto result = this->processPointsetV2(processed_points, processed_normals, umbilicus_points_vector, angleStep, z_spacing, max_eucledian_distance, min_z, max_z, min_wind, max_wind);

        return std::move(result);
    }

    std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>>
    processPointset(
        std::vector<std::vector<float>>& original_points,
        const std::vector<std::vector<float>>& original_normals,
        const std::vector<std::vector<float>>& umbilicus_points,
        float angleStep, int z_spacing, float max_eucledian_distance, float min_z, float max_z, float min_wind, float max_wind)
    {
        std::cout << "Processing pointset" << std::endl;
        // Preprocess angle step
        int angleSteps = getAngleSteps(angleStep);

        std::cout << "Angle steps: " << angleSteps << std::endl;

        auto [minWind, maxWind] = findMinMaxWindingAngles(original_points);

        if (min_wind != max_wind) {
            minWind = min_wind;
            maxWind = max_wind;
        }

        // put down to complete 360 deg winding
        float k_ = minWind / 360.0;
        int k = std::floor(k_);
        minWind = k * 360.0;

        std::cout << "Found min and max winding angles " << minWind << ", " << maxWind << std::endl;

        // Calculate origin point and radius for each point
        auto [origin_points, radii] = umbilicusToOrigin(original_points, umbilicus_points);

        std::cout << "Calculated origin points and radii" << std::endl;

        // Calculate z positions based on spacing
        std::vector<float> zPositions = calculateZPositions(original_points, z_spacing, min_z, max_z);

        std::cout << "Calculated z positions" << std::endl;

        // Clear the original points to save memory
        original_points.clear();
        original_points.shrink_to_fit();

        std::cout << "Cleared original points" << std::endl;

        std::vector<std::vector<float>> ordered_umbilicus_points(zPositions.size());
        for (size_t i = 0; i < zPositions.size(); ++i) {
            ordered_umbilicus_points[i] = umbilicus_xz_at_y(umbilicus_points, zPositions[i]);
        }

        std::cout << "Calculated ordered umbilicus points" << std::endl;

        // Find size of results based on the min and max angles
        int totalAngles = getAngleIndex(0.0f, maxWind, minWind) + 1;

        std::cout << "Total angles: " << totalAngles << std::endl;

        std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> results(totalAngles * angleSteps);
        // Process points for each angle step
        for (int angleStep_ = 0; angleStep_ < angleSteps / 4; angleStep_++) {
            processAngle(origin_points, original_normals, umbilicus_points, ordered_umbilicus_points, radii, angleStep_, angleSteps, minWind, totalAngles, zPositions, max_eucledian_distance, results);
            std::cout << "Processed angle " << angleStep_ * 360.0 / angleSteps << std::endl;
            ;
        }

        // needs to be done in python since this function gets called repeatedly and the results need to be combined before cutting away
        // // Cut away start and end of results that are empty
        // int first_non_empty = 0;
        // int last_non_empty = results.size() - 1;

        // // Find the first non-empty result
        // while (first_non_empty < results.size() && isEmptyResult(results[first_non_empty])) {
        //     first_non_empty++;
        // }

        // // Find the last non-empty result
        // while (last_non_empty >= 0 && isEmptyResult(results[last_non_empty])) {
        //     last_non_empty--;
        // }

        // std::cout << "First non-empty: " << first_non_empty << " Last non-empty: " << last_non_empty << " Results size: " << results.size() << " new Results size: " << last_non_empty - first_non_empty + 1 << std::endl;

        // if (first_non_empty <= last_non_empty) {
        //     results = std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>>(
        //         results.begin() + first_non_empty, results.begin() + last_non_empty + 1);
        // } else {
        //     // If all results are empty, clear the vector
        //     results.clear();
        // }

        return std::move(results);
    }

    std::vector<std::tuple<
    std::vector<std::vector<float>>,             // For each z bin: ordered ts (radii) values
    std::vector<std::vector<std::vector<float>>>,  // For each z bin: ordered normals
    std::vector<std::vector<float>>,               // Precomputed ordered umbilicus points (one per z bin)
    std::vector<float>                             // Angle vector for this angle step
    >> processPointsetV2(
        const std::vector<std::vector<float>>& sorted_points,  // Pre-sorted by winding angle (4th entry)
        const std::vector<std::vector<float>>& sorted_normals,
        const std::vector<std::vector<float>>& umbilicus_points,
        float angleStep,      // e.g. 6 degrees per step
        int z_spacing,        // spacing used to compute z bins
        float max_distance,   // max allowed z deviation for candidate acceptance
        float min_z, float max_z, 
        float min_wind, float max_wind)
    {
        // Step 1: Precompute z bins and their corresponding umbilicus points.
        std::vector<float> zPositions = calculateZPositions(sorted_points, z_spacing, min_z, max_z);
        std::vector<std::vector<float>> ordered_umbilicus_points(zPositions.size());
        for (size_t i = 0; i < zPositions.size(); ++i) {
            ordered_umbilicus_points[i] = umbilicus_xz_at_y(umbilicus_points, zPositions[i]);
        }

        // Step 2: Determine overall angle range from sorted points.
        if (sorted_points.empty())
            return {};  // no points to process

        auto [minWind, maxWind] = findMinMaxWindingAngles(sorted_points);

        if (min_wind != max_wind) {
            minWind = min_wind;
            maxWind = max_wind;
        }

        int numSteps = static_cast<int>(std::floor((maxWind - minWind) / angleStep)) + 1;
        std::vector<std::tuple<
            std::vector<std::vector<float>>,             // ordered ts (radii) per z bin
            std::vector<std::vector<std::vector<float>>>,  // ordered normals per z bin
            std::vector<std::vector<float>>,               // ordered umbilicus points (common to all bins)
            std::vector<float>                             // angle vector for this angle step
        >> results;
        results.reserve(numSteps);

        size_t nPoints = sorted_points.size();
        size_t currentStart = 0, currentEnd = 0;

        // Step 3: Loop over angle steps from minWind to maxAngle.
        for (int step = 0; step < numSteps; ++step) {
            float targetAngle = minWind + step * angleStep;
            // Use a ±10° window.
            float windowLow = targetAngle - 10.0f;
            float windowHigh = targetAngle + 10.0f;

            // Advance currentStart until point angle >= windowLow.
            while (currentStart < nPoints && sorted_points[currentStart][3] < windowLow) {
                ++currentStart;
            }
            // Ensure currentEnd is not behind currentStart.
            currentEnd = std::max(currentStart, currentEnd);
            // Advance currentEnd until point angle > windowHigh.
            while (currentEnd < nPoints && sorted_points[currentEnd][3] <= windowHigh) {
                ++currentEnd;
            }
            // Now, candidate indices lie in [currentStart, currentEnd).

            // Create candidate bins (one vector per z bin).
            // Each candidate is a pair: (angular metric, index into sorted_points).
            std::vector<std::vector<std::pair<float, size_t>>> bin_candidates(zPositions.size());

            for (size_t i = currentStart; i < currentEnd; ++i) {
                float pointAngle = sorted_points[i][3];
                // Compute angular difference (accounting for wrap-around if needed).
                float diff = std::fabs(pointAngle - targetAngle);
                if (diff > 180.0f)
                    diff = 360.0f - diff;
                float metric = diff; // Use angular difference as the metric.
                
                // Get the point's vertical coordinate (assumed at index 1).
                float point_z = sorted_points[i][1];
                if (point_z < zPositions.front() || point_z > zPositions.back())
                    continue;
                
                // Compute the candidate's z bin index (rounding based on z_spacing).
                int bin_index = static_cast<int>(std::round((point_z - zPositions.front()) / static_cast<float>(z_spacing)));
                if (bin_index < 0 || bin_index >= static_cast<int>(zPositions.size()))
                    continue;
                
                // Accept candidate only if its z difference from the bin center is within max_distance.
                float zDiff = std::fabs(point_z - zPositions[bin_index]);
                if (zDiff > max_distance) {
                    std::cout << "Something is wrong with the sorted points!!!!" << std::endl;
                    continue;
                }
                
                bin_candidates[bin_index].push_back({metric, i});
            }

            // For each z bin, process candidates:
            // 1. Sort candidates by angular metric.
            // 2. Retain the closest 50 candidates.
            // 3. Compute their centroid (in x and z); update the centroid's z to the bin center.
            // 4. Compute each candidate's distance to the centroid and sort.
            // 5. Retain the closest 10 candidates.
            std::vector<std::vector<float>> ordered_ts(zPositions.size());               // To store computed radii per bin.
            std::vector<std::vector<std::vector<float>>> ordered_normals(zPositions.size()); // To store corresponding normals.
            for (size_t bin = 0; bin < zPositions.size(); ++bin) {
                auto& candidates = bin_candidates[bin];
                if (candidates.empty())
                    continue;

                // Sort by angular difference.
                std::sort(candidates.begin(), candidates.end(),
                        [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                            return a.first < b.first;
                        });
                // Keep up to 50 closest candidates.
                if (candidates.size() > 50)
                    candidates.resize(50);

                // Compute centroid (average x and z) of these candidates.
                float sumX = 0.0f, sumZ = 0.0f;
                for (const auto& cand : candidates) {
                    size_t idx = cand.second;
                    sumX += sorted_points[idx][0];
                    sumZ += sorted_points[idx][2];  // Assume index 2 holds the z-coordinate for radius.
                }
                int count = candidates.size();
                float centroidX = sumX / count;
                float centroidZ = sumZ / count;
                // Update centroid's z to the bin's center.
                float updatedCentroidZ = zPositions[bin];

                // Compute distances from each candidate to the centroid.
                std::vector<std::pair<float, size_t>> dist_candidates;
                for (const auto& cand : candidates) {
                    size_t idx = cand.second;
                    float dx = sorted_points[idx][0] - centroidX;
                    float dz = sorted_points[idx][2] - updatedCentroidZ;
                    float distance = std::sqrt(dx * dx + dz * dz);
                    dist_candidates.push_back({distance, idx});
                }
                // Sort candidates by distance to the centroid.
                std::sort(dist_candidates.begin(), dist_candidates.end(),
                        [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
                            return a.first < b.first;
                        });
                // Retain the closest 10 candidates.
                if (dist_candidates.size() > 10)
                    dist_candidates.resize(10);

                // For each selected candidate, compute the "ts" value (radius)
                // using the corresponding umbilicus point and store the normal.
                for (const auto& dc : dist_candidates) {
                    size_t idx = dc.second;
                    std::vector<float> umb = umbilicus_xz_at_y(umbilicus_points, sorted_points[idx][1]);
                    float dx = sorted_points[idx][0] - umb[0];
                    float dz = sorted_points[idx][2] - umb[2];
                    float radius = -std::sqrt(dx * dx + dz * dz);
                    ordered_ts[bin].push_back(radius);
                    ordered_normals[bin].push_back(sorted_normals[idx]);
                }
            }

            // Compute the 3D angle vector for the target angle.
            float rad = targetAngle * M_PI / 180.0f;
            std::vector<float> angle_vector = { std::cos(rad), 0.0f, -std::sin(rad) };

            // Store the result for this angle step.
            results.push_back(std::make_tuple(ordered_ts, ordered_normals, ordered_umbilicus_points, angle_vector));
            // Print every 1% progress
            if (verbose && step % (numSteps / 100) == 0) {
                // Update same line, flush
                std::cout << "\rProcessed angle step at target angle " << std::fixed << std::setprecision(2) << targetAngle << " degrees. Of min/max angles: " << minWind << " / " << maxWind << " degrees. Progress: " << (step * 100) / numSteps << "%" << std::flush;
            }
        }
        std::cout << std::endl;

        return results;
    }

private:
    bool verbose;

    // Adjust angle step st 360 / angleStep is divisible by 4
    float getAngleSteps(float angleStep) {
        int angleSteps = 360.0 / angleStep;
        while (angleSteps % 4 != 0) {
            angleSteps++;
        }
        return angleSteps;
    }

    bool isEmptyResult(const std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>& result) {
        const auto& ordered_pointset = std::get<0>(result);
        for (const auto& pointset : ordered_pointset) {
            if (!pointset.empty()) {
                return false;
            }
        }
        return true;
    }

    // Function to compute vectors from points to their corresponding umbilicus points and their radii
    std::tuple<std::vector<std::vector<float>>, std::vector<float>> umbilicusToOrigin(
        const std::vector<std::vector<float>>& points,
        const std::vector<std::vector<float>>& umbilicus_points) 
    {
        // Vectors to store the results
        std::vector<std::vector<float>> origin_points(points.size(), std::vector<float>(4));
        std::vector<float> radii(points.size());

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < points.size(); ++i) {
            // Find the corresponding umbilicus point based on the y-coordinate
            const std::vector<float> umbilicus_position = umbilicus_xz_at_y(umbilicus_points, points[i][1]);

            // Calculate the vector from the point to the umbilicus point
            origin_points[i][0] = points[i][0] - umbilicus_position[0];
            origin_points[i][1] = points[i][1];
            origin_points[i][2] = points[i][2] - umbilicus_position[2];
            origin_points[i][3] = points[i][3];

            // Calculate the radius (distance in the xz-plane)
            radii[i] = - std::sqrt(
                std::pow(origin_points[i][0], 2) +
                std::pow(origin_points[i][2], 2)
            );
        }

        // Return a tuple of the two vectors
        return std::make_tuple(origin_points, radii);
    }

    // Calculate z positions based on spacing
    std::vector<float> calculateZPositions(const std::vector<std::vector<float>>& points, int z_spacing, float min_z, float max_z) {
        auto [minZ, maxZ] = findMinMaxZ(points);
        minZ = std::max(minZ - 1, min_z);
        maxZ = std::min(maxZ + 1, max_z);
        std::vector<float> zPositions;
        for (float z = minZ; z < maxZ; z += z_spacing) {
            zPositions.push_back(z);
        }
        return zPositions;
    }

    // Find min and max Z values
    std::pair<float, float> findMinMaxZ(const std::vector<std::vector<float>>& points) {
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();
        for (const auto& point : points) {
            if (point[1] < minZ) minZ = point[1];
            if (point[1] > maxZ) maxZ = point[1];
        }
        return {minZ, maxZ};
    }

    // Find angle group index
    int getAngleIndex(float angle, float pointAngle, float minAngle) {
        float delta_angle = pointAngle - minAngle;
        float angle_anligned = delta_angle - angle + 180.0;
        int angle_index = static_cast<int>(angle_anligned / 360.0);
        if (angle_index < 0) {
            std::cout << "Negative angle index: " << angle_index << std::endl;
            angle_index = 0;
        }
        return angle_index;
    }

    void processAngle(
        const std::vector<std::vector<float>>& points,
        const std::vector<std::vector<float>>& normals,
        const std::vector<std::vector<float>>& umbilicus_points,
        const std::vector<std::vector<float>> ordered_umbilicus_points,
        const std::vector<float>& radii,
        int angleStep_,
        int angleSteps,
        float minWind,
        int totalAngles,
        const std::vector<float>& z_positions,
        float max_eucledian_distance,
        std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>>& results) 
    {
        float angle = angleStep_ * 360.0 / angleSteps;

        // Calculate the projection plane normal vectors for the given angle and angle + 90 degrees
        float angle_radians = angle * M_PI / 180.0;
        float angle_plus_90_radians = (angle + 90.0f) * M_PI / 180.0;

        std::vector<float> angle_vector = { std::cos(static_cast<float>(angle_radians)), 0.0f, -std::sin(static_cast<float>(angle_radians)) };
        std::vector<float> angle_vector_plus_90 = { std::cos(static_cast<float>(angle_plus_90_radians)), 0.0f, -std::sin(static_cast<float>(angle_plus_90_radians)) };
        std::vector<float> angle_vector_plus_180 = { std::cos(static_cast<float>(angle_radians + M_PI)), 0.0f, -std::sin(static_cast<float>(angle_radians + M_PI)) };
        std::vector<float> angle_vector_plus_270 = { std::cos(static_cast<float>(angle_plus_90_radians + M_PI)), 0.0f, -std::sin(static_cast<float>(angle_plus_90_radians + M_PI)) };

        std::vector<std::vector<float>> angle_vectors = { angle_vector, angle_vector_plus_90, angle_vector_plus_180, angle_vector_plus_270 };
        std::vector<float> angles = { angle, angle + 90.0f, angle + 180.0f, angle + 270.0f };

        // Group data structures
        std::vector<std::vector<std::vector<float>>> grouped_points(4); // For angle and angle + 180
        std::vector<std::vector<std::vector<float>>> grouped_normals(4);
        std::vector<std::vector<float>> grouped_distances(4);        
        std::vector<std::vector<float>> grouped_ts(4);

        // Thread-local storage for each group
        std::vector<float> distances(points.size());
        std::vector<float> distances_plus_90(points.size());
        std::vector<float> group_indices(points.size());
        std::vector<float> group_indices_plus_90(points.size());

        // Process each point
        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i) {
            // Project the point onto the two planes
            float distance = distanceToProjectionUsingHessian(points[i], angle_vector);
            float distance_plus_90 = distanceToProjectionUsingHessian(points[i], angle_vector_plus_90);

            // Determine which side the point belongs to (0 or 180 degrees) based on the sign of x component in angle + 90 plane
            int group_index = distance_plus_90 > 0 ? 0 : 2; // clockwise to other plane
            int group_index_plus_90 = distance > 0 ? 3 : 1; // counterclockwise to other plane

            // Store the distances and group indices
            distances[i] = distance;
            distances_plus_90[i] = distance_plus_90;
            group_indices[i] = group_index;
            group_indices_plus_90[i] = group_index_plus_90;
        }

        // std::cout << "Computed distances" << std::endl;

        for (size_t i = 0; i < points.size(); ++i) {
            // Filter out points that are beyond the Euclidean distance threshold
            if (std::abs(distances[i]) <= max_eucledian_distance) {
                // std::cout << "Group index: " << group_indices[i] << std::endl;
                grouped_points[group_indices[i]].push_back(points[i]);
                grouped_normals[group_indices[i]].push_back(normals[i]);
                grouped_distances[group_indices[i]].push_back(distances[i]);
                grouped_ts[group_indices[i]].push_back(radii[i]);
            }
            if (std::abs(distances_plus_90[i]) <= max_eucledian_distance) {
                // std::cout << "Group index: " << group_indices_plus_90[i] << std::endl;
                grouped_points[group_indices_plus_90[i]].push_back(points[i]);
                grouped_normals[group_indices_plus_90[i]].push_back(normals[i]);
                grouped_distances[group_indices_plus_90[i]].push_back(distances_plus_90[i]);
                grouped_ts[group_indices_plus_90[i]].push_back(radii[i]);
            }
        }

        // std::cout << "Grouped points" << std::endl;

        // Sort each group by the z-coordinate
        #pragma omp parallel for
        for (int g = 0; g < grouped_points.size(); ++g) {
            // Create indices for sorting
            std::vector<size_t> indices(grouped_points[g].size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort the indices based on the z-coordinate of the points in grouped_points[g]
            std::sort(indices.begin(), indices.end(), [&grouped_points, g](size_t i1, size_t i2) {
                return grouped_points[g][i1][1] < grouped_points[g][i2][1];
            });

            // Apply the sorted indices to reorder points, normals, and ts within each group
            std::vector<std::vector<float>> sorted_points(grouped_points[g].size());
            std::vector<std::vector<float>> sorted_normals(grouped_normals[g].size());
            std::vector<float> sorted_distances(grouped_distances[g].size());
            std::vector<float> sorted_ts(grouped_ts[g].size());

            for (size_t i = 0; i < indices.size(); ++i) {
                sorted_points[i] = std::move(grouped_points[g][indices[i]]);
                sorted_normals[i] = std::move(grouped_normals[g][indices[i]]);
                sorted_distances[i] = std::move(grouped_distances[g][indices[i]]);
                sorted_ts[i] = std::move(grouped_ts[g][indices[i]]);
            }

            // Replace the original group with the sorted data
            grouped_points[g] = std::move(sorted_points);
            grouped_normals[g] = std::move(sorted_normals);
            grouped_distances[g] = std::move(sorted_distances);
            grouped_ts[g] = std::move(sorted_ts);
        }

        // std::cout << "Sorted points" << std::endl;

        // Process each group
        for (int g = 0; g < grouped_points.size(); ++g) {
            // std::cout << "Processing group " << g << std::endl;
            std::vector<float> angle_vector_ = angle_vectors[g];
            // Extract for each z position the points within the threshold range
            std::vector<std::vector<std::vector<float>>> ordered_pointset(totalAngles, std::vector<std::vector<float>>(z_positions.size()));
            std::vector<std::vector<std::vector<float>>> ordered_distances(totalAngles, std::vector<std::vector<float>>(z_positions.size()));
            std::vector<std::vector<std::vector<std::vector<float>>>> ordered_points(totalAngles, std::vector<std::vector<std::vector<float>>>(z_positions.size()));
            std::vector<std::vector<std::vector<std::vector<float>>>> ordered_normals(totalAngles, std::vector<std::vector<std::vector<float>>>(z_positions.size()));

            // Initialize and count points for each z-position
            std::vector<std::vector<bool>> in_range(z_positions.size(), std::vector<bool>(grouped_points[g].size(), false));
            std::vector<std::vector<int>> angle_index_o(z_positions.size(), std::vector<int>(grouped_points[g].size(), 0));
            // Initialize a 2D vector counts: totalAngles x z_positions.size()
            std::vector<std::vector<size_t>> counts(
                totalAngles, 
                std::vector<size_t>(z_positions.size(), 0)
            );

            // std::cout << "Created empty vectors" << std::endl;

            #pragma omp parallel for
            for (size_t j = 0; j < grouped_points[g].size(); ++j) {
                for (size_t i = 0; i < z_positions.size(); ++i) {
                    // Take points within the Euclidean distance threshold
                    if (std::abs(grouped_points[g][j][1] - z_positions[i]) <= max_eucledian_distance) {
                        // split up and group the ordered pointset by the winding angles into 360 degree segments per ordered pointset for angle +- 180 degrees
                        int angle_index = getAngleIndex(angles[g], grouped_points[g][j][3], minWind);
                        angle_index_o[i][j] = angle_index;
                        // Store the selection
                        in_range[i][j] = true;
                        #pragma omp atomic
                        counts[angle_index][i]++;
                    }
                }
            }

            // std::cout << "Counted points" << std::endl;

            // Allocate memory based on counts
            for (size_t o = 0; o < totalAngles; ++o) {
                for (size_t i = 0; i < z_positions.size(); ++i) {
                    ordered_pointset[o][i].resize(counts[o][i]);
                    ordered_distances[o][i].resize(counts[o][i]);
                    ordered_points[o][i].resize(counts[o][i]);
                    ordered_normals[o][i].resize(counts[o][i]);
                }
            }

            // std::cout << "Reserved memory" << std::endl;
            
            // Store points and normals
            #pragma omp parallel for
            for (size_t i = 0; i < z_positions.size(); ++i) {
                // Index for storing the point and normal directly
                std::vector<size_t> index(totalAngles, 0);
                for (size_t j = 0; j < grouped_points[g].size(); ++j) {
                    if (in_range[i][j]) {
                        int o = angle_index_o[i][j];
                        // Check size of ordered_pointset[o][i] and so on
                        if (index[o] >= counts[o][i]) {
                            std::cout << "Error: index is out of bounds" << std::endl;
                            continue;
                        }
                        ordered_pointset[o][i][index[o]] = grouped_ts[g][j];
                        ordered_distances[o][i][index[o]] = grouped_distances[g][j];
                        ordered_points[o][i][index[o]] = grouped_points[g][j];
                        ordered_normals[o][i][index[o]] = grouped_normals[g][j];
                        // Check that ordered_points[o][i] at index[o] has 4 entries
                        if (ordered_points[o][i][index[o]].size() != 4) {
                            std::cout << "Error: ordered points size does not match 4" << std::endl;
                            continue;
                        }
                        index[o]++;
                    }
                }

                // Shrink to fit ordered_points[o][i] and so on to index[o]
                for (size_t o = 0; o < totalAngles; ++o) {
                    ordered_pointset[o][i].resize(index[o]);
                    ordered_distances[o][i].resize(index[o]);
                    ordered_points[o][i].resize(index[o]);
                    ordered_normals[o][i].resize(index[o]);
                }

                // Check that each point has 4 entries
                for (size_t o = 0; o < totalAngles; ++o) {
                    if (ordered_pointset[o][i].size() != counts[o][i]) {
                        std::cout << "Error: ordered pointset size does not match counts " << ordered_pointset[o][i].size() << " != " << counts[o][i] << std::endl;
                    }
                }


                // if (i == z_positions.size() / 2) {
                //     // Check if the index is correct
                //     float min_angl = 10000000000;
                //     float max_angl = -10000000000;
                //     for (size_t p = 0; p < ordered_points[totalAngles/2][i].size(); ++p) {
                //         if (ordered_points[totalAngles/2][i][p][3] < min_angl) {
                //             min_angl = ordered_points[totalAngles/2][i][p][3];
                //         }
                //         if (ordered_points[totalAngles/2][i][p][3] > max_angl) {
                //             max_angl = ordered_points[totalAngles/2][i][p][3];
                //         }
                //     }

                //     std::cout << "Min and max winding angles for z position " << z_positions[i] << " are " << min_angl << ", " << max_angl << " size of points: " << ordered_points[totalAngles/2][i].size() << std::endl;
                // }

                for (size_t o = 0; o < totalAngles; ++o) {
                    // skip if no points
                    if (ordered_pointset[o][i].empty()) {
                        continue;
                    }
                    std::vector<size_t> indices(ordered_points[o][i].size());
                    std::iota(indices.begin(), indices.end(), 0);

                    // sort points from closest to angle ray to furthest
                    std::sort(indices.begin(), indices.end(), [&ordered_points, &ordered_distances, o, i, z_positions](size_t i1, size_t i2) {
                        float x1 = ordered_distances[o][i][i1];
                        float z_diff1 = ordered_points[o][i][i1][1] - z_positions[i];
                        float dist1 = x1 * x1 + z_diff1 * z_diff1;

                        float x2 = ordered_distances[o][i][i2];
                        float z_diff2 = ordered_points[o][i][i2][1] - z_positions[i];
                        float dist2 = x2 * x2 + z_diff2 * z_diff2;

                        return dist1 < dist2;
                    });

                    // Take maximaly 10 smallest normals and pointset
                    if (indices.size() > 10) {
                        indices.resize(10);
                    }

                    std::vector<float> ordered_pointset_(indices.size());
                    std::vector<std::vector<float>> ordered_points_(indices.size());
                    std::vector<std::vector<float>> ordered_normals_(indices.size());
                    for (size_t j = 0; j < indices.size(); ++j) {
                        ordered_pointset_[j] = ordered_pointset[o][i][indices[j]];
                        ordered_points_[j] = ordered_points[o][i][indices[j]];
                        ordered_normals_[j] = ordered_normals[o][i][indices[j]];
                    }

                    // std move to avoid copying
                    ordered_pointset[o][i] = std::move(ordered_pointset_);
                    ordered_points[o][i] = std::move(ordered_points_);
                    ordered_normals[o][i] = std::move(ordered_normals_);

                    if (i == z_positions.size() / 2 && o == totalAngles / 2) {
                        // Find mean and std of winding angles of the points
                        float mean_winding_angle = 0.0;
                        float std_winding_angle = 0.0;
                        for (size_t p = 0; p < ordered_points[o][i].size(); ++p) {
                            mean_winding_angle += ordered_points[o][i][p][3];
                        }
                        mean_winding_angle /= ordered_points[o][i].size();
                        for (size_t p = 0; p < ordered_points[o][i].size(); ++p) {
                            std_winding_angle += std::pow(ordered_points[o][i][p][3] - mean_winding_angle, 2);
                        }
                        std_winding_angle = std::sqrt(std_winding_angle / ordered_points[o][i].size());
                        float closest_angle = mean_winding_angle - angles[g]; // find closest angle for mean
                        closest_angle = closest_angle / 360.0;
                        closest_angle = std::round(closest_angle);
                        closest_angle = closest_angle * 360.0 + angles[g]; // use offset from mean winding angle
                        float alternative_alternative_angle = minWind + o*360.0f + angles[g]; // use offset from min winding angle
                        // std::cout << "Angle: " << angles[g] << " Closest angle: " << closest_angle << " Closest angle alternative calculation: " << alternative_alternative_angle << " Mean winding angle: " << mean_winding_angle << " Std winding angle: " << std_winding_angle << std::endl;
                    }
                }
            }

            // std::cout << "Stored points and normals" << std::endl;

            for (size_t o = 0; o < totalAngles; ++o) {
                int index = angleStep_ + g * angleSteps / 4 + o * angleSteps;
                // std::cout << "Stored points and normals for angle " << o << " and angle step " << angleStep_ << " with index " << index << std::endl;
                // Store the results for each angle index
                if (index >= results.size()) {
                    std::cout << "Index out of bounds: " << index << " " << results.size() << std::endl;
                    continue;
                }
                results[index] = {ordered_pointset[o], ordered_normals[o], ordered_umbilicus_points, angle_vector_};
            }

            // std::cout << "Stored results" << std::endl;
        }
    }

    float distanceToProjectionUsingHessian(const std::vector<float>& point, const std::vector<float>& normal_vector) {
        // Extract the x and z components from the point
        float y = point[0];
        float x = point[2];

        // Extract the components of the normal vector
        float a = normal_vector[0];
        float b = normal_vector[2];

        // Vector is unit vector

        // Rotate the normal vector by 90 degrees clockwise
        float rotated_a = b;
        float rotated_b = -a;

        // Calculate the distance using the Hessian normal form
        float distance = rotated_b * x + rotated_a * y; // No normalization needed as the normal vector is a unit vector

        return distance;
    }

};


std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> create_ordered_pointset(
    py::array_t<float> original_points,
    py::array_t<float> original_normals,
    py::array_t<float> umbilicus_points,
    float angleStep, int z_spacing, float max_eucledian_distance, bool verbose, float min_z, float max_z, float min_wind, float max_wind
    )
{
    EfficientPointsetProcessor processor(verbose);
    if (verbose) {
        std::cout << "Creating ordered pointset" << std::endl;
    }
    return std::move(processor.create_ordered_pointset_processor(std::move(original_points), std::move(original_normals), umbilicus_points, min_z, max_z, angleStep, z_spacing, max_eucledian_distance, min_wind, max_wind));
    if (verbose) {
        std::cout << "Finished creating ordered pointset" << std::endl;
    }
}

class OrderedPointsetOptimizer {
public:
    OrderedPointsetOptimizer(
        std::vector<std::vector<float>> input_ordered_pointset,
        std::vector<std::vector<bool>> fixed_points,
        std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices,
        float learning_rate = 0.1,
        size_t iterations = 3,
        float error_val_d = 0.01,
        float unfix_factor = 3.0,
        bool verbose = false)
        :   input_ordered_pointset(input_ordered_pointset), 
            new_interpolated_ts(input_ordered_pointset), 
            fixed_points(fixed_points), 
            neighbours_indices(neighbours_indices), 
            learning_rate(learning_rate),
            iterations(iterations),
            error_val_d(error_val_d),
            verbose(verbose) {}

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    std::vector<std::vector<float>> optimize_ordered_pointset_processor() {
        // Set up progress tracking
        problem_size = iterations;
        progress = 0;
        std::cout << "Optimizing ordered pointset" << std::endl;
        // Iterate over the number of iterations
        for (size_t iter = 0; iter < iterations; ++iter) {
            // Progress tracking
            {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
            }

            // Calculate the total number of vertices and fixed points
            size_t nr_vertices = 0;
            size_t nr_fixed = 0;
            float last_error_val = std::numeric_limits<float>::max();

            // Calculate the total number of vertices and fixed points
            for (const auto& row : fixed_points) {
                nr_vertices += row.size();
                nr_fixed += std::count(row.begin(), row.end(), true);
            }
            size_t nr_floating = nr_vertices - nr_fixed;

            for (size_t opt_iter = 0; opt_iter < 10000; ++opt_iter) { // Maximum of 10000 optimization steps
                float error_val = compute_interpolated_adjacent();

                error_val /= nr_floating; // Normalize error by the number of floating vertices
                // std::cout << "Error value per floating vertex: " << std::setprecision(5) << error_val << std::endl;

                // Check for convergence or if the error increased
                if ((std::abs(last_error_val - error_val) < error_val_d) || (last_error_val - error_val < 0)) {
                    break;
                }
                last_error_val = error_val;
            }

            // Detect and unfix wrong fixed adjacent if necessary
            detect_and_unfix_wrong_fixed_adjacent();
        }
        // Finish progress tracking
        {
            if (verbose) {
                std::cout << std::endl;
            }
        }

        return new_interpolated_ts;
    }

private:
    std::pair<float, bool> get_front(size_t i, size_t j) {
        size_t f_i = neighbours_indices[i][j][0][0];
        size_t f_j = neighbours_indices[i][j][0][1];
        if (f_i == -1 || f_j == -1) {
            return {1, false};
        }
        float ts_f = input_ordered_pointset[f_i][f_j];
        bool fixed_f = fixed_points[f_i][f_j];
        return {ts_f, fixed_f};
    }

    std::pair<float, bool> get_back(size_t i, size_t j) {
        size_t b_i = neighbours_indices[i][j][1][0];
        size_t b_j = neighbours_indices[i][j][1][1];
        if (b_i == -1 || b_j == -1) {
            return {1, false};
        }
        float ts_b = input_ordered_pointset[b_i][b_j];
        bool fixed_b = fixed_points[b_i][b_j];
        return {ts_b, fixed_b};
    }

    std::pair<float, bool> get_top(size_t i, size_t j) {
        size_t t_i = neighbours_indices[i][j][2][0];
        size_t t_j = neighbours_indices[i][j][2][1];
        if (t_i == -1 || t_j == -1) {
            return {1, false};
        }
        float ts_t = input_ordered_pointset[t_i][t_j];
        bool fixed_t = fixed_points[t_i][t_j];
        return {ts_t, fixed_t};
    }

    std::pair<float, bool> get_bottom(size_t i, size_t j) {
        size_t bo_i = neighbours_indices[i][j][3][0];
        size_t bo_j = neighbours_indices[i][j][3][1];
        if (bo_i == -1 || bo_j == -1) {
            return {1, false};
        }
        float ts_bo = input_ordered_pointset[bo_i][bo_j];
        bool fixed_bo = fixed_points[bo_i][bo_j];
        return {ts_bo, fixed_bo};
    }

    std::pair<float, bool> get_left(size_t i, size_t j) {
        size_t l_i = neighbours_indices[i][j][4][0];
        size_t l_j = neighbours_indices[i][j][4][1];
        if (l_i == -1 || l_j == -1) {
            return {1, false};
        }
        float ts_l = input_ordered_pointset[l_i][l_j];
        bool fixed_l = fixed_points[l_i][l_j];
        return {ts_l, fixed_l};
    }

    std::pair<float, bool> get_right(size_t i, size_t j) {
        size_t r_i = neighbours_indices[i][j][5][0];
        size_t r_j = neighbours_indices[i][j][5][1];
        if (r_i == -1 || r_j == -1) {
            return {1, false};
        }
        float ts_r = input_ordered_pointset[r_i][r_j];
        bool fixed_r = fixed_points[r_i][r_j];
        return {ts_r, fixed_r};
    }

    std::pair<float, bool> get_left_new(size_t i, size_t j) {
        size_t l_i = neighbours_indices[i][j][4][0];
        size_t l_j = neighbours_indices[i][j][4][1];
        if (l_i == -1 || l_j == -1) {
            return {1, false};
        }
        float ts_l = new_interpolated_ts[l_i][l_j];
        bool fixed_l = fixed_points[l_i][l_j];
        return {ts_l, fixed_l};
    }

    std::pair<float, bool> get_right_new(size_t i, size_t j) {
        size_t r_i = neighbours_indices[i][j][5][0];
        size_t r_j = neighbours_indices[i][j][5][1];
        if (r_i == -1 || r_j == -1) {
            return {1, false};
        }
        float ts_r = new_interpolated_ts[r_i][r_j];
        bool fixed_r = fixed_points[r_i][r_j];
        return {ts_r, fixed_r};
    }

    float solve_for_t_individual(float r, float l, float m_r, bool valid_mr, float m_l, bool valid_ml, float m_ts, bool valid_mts, float a = 1.0) {
        float t_ts = m_ts;
        float t_total = t_ts;
        float count_total = a;
        if (r != 1 && valid_mr) {
            float t_r = r - m_r;
            t_total += t_r;
            count_total += 1.0;
        }
        if (l != 1 && valid_ml) {
            float t_l = l - m_l;
            t_total += t_l;
            count_total += 1.0;
        }
        t_total /= count_total;

        if (t_total > 0.0) {
            t_total = 0.0;
        }

        return t_total;
    }

    std::pair<bool, bool> side_of(float ts_, float n) {
        return {ts_ > n, ts_ == n};
    }

    float respect_non_overlapping(size_t i, size_t j, float new_ts_d) {
        float old_ts = input_ordered_pointset[i][j];

        auto [ts_l, fixed_l] = get_left(i, j);  // old left
        auto [ts_r, fixed_r] = get_right(i, j); // old right
        auto [ts_ln, fixed_ln] = get_left_new(i, j);  // new left
        auto [ts_rn, fixed_rn] = get_right_new(i, j); // new right

        // Check left boundary
        if (ts_l != 1) {
            auto [side_old_l, invalid_l] = side_of(old_ts, ts_l);
            if (invalid_l) {
                std::cout << "Invalid left boundary for point " << i << ", " << j << " with ts_l: " << ts_l << " and old_ts: " << old_ts << std::endl;
            }
            assert(!invalid_l);
            if (!invalid_l) {
                auto [side_new_l, invalid_new_l] = side_of(old_ts + new_ts_d, ts_l);
                if (side_old_l != side_new_l) {
                    new_ts_d = (ts_l - old_ts) * 0.5;
                    // new_ts_d = 0.0f;
                }
            }
        }

        // Check right boundary
        if (ts_r != 1) {
            auto [side_old_r, invalid_r] = side_of(old_ts, ts_r);
            if (invalid_r) {
                std::cout << "Invalid right boundary for point " << i << ", " << j << " with ts_r: " << ts_r << " and old_ts: " << old_ts << std::endl;
            }
            assert(!invalid_r);
            if (!invalid_r) {
                auto [side_new_r, invalid_new_r] = side_of(old_ts + new_ts_d, ts_r);
                if (side_old_r != side_new_r) {
                    new_ts_d = (ts_r - old_ts) * 0.5;
                    // new_ts_d = 0.0f;
                }
            }
        }

        // Check new left boundary
        if (ts_ln != 1) {
            auto [side_old_ln, invalid_ln] = side_of(old_ts, ts_ln);
            if (invalid_ln) {
                std::cout << "Invalid new left boundary for point " << i << ", " << j << " with ts_ln: " << ts_ln << " and old_ts: " << old_ts << std::endl;
            }
            assert(!invalid_ln);
            if (!invalid_ln) {
                auto [side_new_ln, invalid_new_ln] = side_of(old_ts + new_ts_d, ts_ln);
                if (side_old_ln != side_new_ln) {
                    new_ts_d = (ts_ln - old_ts) * 0.5;
                    // new_ts_d = 0.0f;
                }
            }
        }

        // Check new right boundary
        if (ts_rn != 1) {
            auto [side_old_rn, invalid_rn] = side_of(old_ts, ts_rn);
            if (invalid_rn) {
                std::cout << "Invalid new right boundary for point " << i << ", " << j << " with ts_rn: " << ts_rn << " and old_ts: " << old_ts << std::endl;
            }
            assert(!invalid_rn);
            if (!invalid_rn) {
                auto [side_new_rn, invalid_new_rn] = side_of(old_ts + new_ts_d, ts_rn);
                if (side_old_rn != side_new_rn) {
                    new_ts_d = (ts_rn - old_ts) * 0.5;
                    // new_ts_d = 0.0f;
                }
            }
        }

        return new_ts_d;
    }

    std::tuple<float, float, float, bool, float, bool, float, bool> calculate_neighbors_values(size_t i, size_t j) {
        std::pair<size_t, size_t> dict_key = {i, j};
        std::vector<std::vector<size_t>> same_sheet_neighbors;

        // Populate neighbors
        auto [f, fixed_f] = get_front(i, j);
        if (f != 1) {
            size_t f_i = neighbours_indices[i][j][0][0];
            size_t f_j = neighbours_indices[i][j][0][1];
            same_sheet_neighbors.push_back({f_i, f_j});
        }
        auto [b, fixed_b] = get_back(i, j);
        if (b != 1) {
            size_t b_i = neighbours_indices[i][j][1][0];
            size_t b_j = neighbours_indices[i][j][1][1];
            same_sheet_neighbors.push_back({b_i, b_j});
        }
        auto [t, fixed_t] = get_top(i, j);
        if (t != 1) {
            size_t t_i = neighbours_indices[i][j][2][0];
            size_t t_j = neighbours_indices[i][j][2][1];
            same_sheet_neighbors.push_back({t_i, t_j});
        }
        auto [bo, fixed_bo] = get_bottom(i, j);
        if (bo != 1) {
            size_t bo_i = neighbours_indices[i][j][3][0];
            size_t bo_j = neighbours_indices[i][j][3][1];
            same_sheet_neighbors.push_back({bo_i, bo_j});
        }

        float l = get_left(i, j).first;
        float r = get_right(i, j).first;
        float m_r = 0.0;
        float m_l = 0.0;
        float m_ts = 0.0;

        size_t count_r = 0, count_l = 0;

        // Assuming 'neighbors' includes the indices for the front-back and top-bottom neighbors
        for (auto& n : same_sheet_neighbors) {
            float ts_n = input_ordered_pointset[n[0]][n[1]];
            assert(ts_n <= 0.0);
            
            m_ts += ts_n;

            auto [l_n, fixed_l_n] = get_left(n[0], n[1]);
            if (l_n != 1) {
                m_l += (l_n - ts_n);
                count_l++;
            }
            auto [r_n, fixed_r_n] = get_right(n[0], n[1]);
            if (r_n != 1) {
                m_r += (r_n - ts_n);
                count_r++;
            }
            
        }

        m_ts = !same_sheet_neighbors.empty() ? m_ts / same_sheet_neighbors.size() : 0;
        bool valid_mts = same_sheet_neighbors.size() > 0;
        m_r = count_r > 0 ? m_r / count_r : 0;
        bool valid_mr = count_r > 0;
        m_l = count_l > 0 ? m_l / count_l : 0;
        bool valid_ml = count_l > 0;

        return {r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts};
    }

    void compute_section(int start, int end, float& thread_error_val) {
        for (int i = start; i < end; ++i) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); ++j) {
                if (!fixed_points[i][j]) {
                    auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                    float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                    assert(t <= 0.0);
                    float d_t = t - input_ordered_pointset[i][j];
                    d_t = respect_non_overlapping(i, j, d_t);
                    thread_error_val += std::abs(d_t);
                    new_interpolated_ts[i][j] = input_ordered_pointset[i][j] + learning_rate * d_t; // Apply learning rate
                }
            }
        }
    }

    float compute_interpolated_adjacent() {
        float error_val = 0.0;
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<float> errors(num_threads, 0.0);
        int n = input_ordered_pointset.size();
        int chunk_size = (n + num_threads - 1) / num_threads; // Calculate chunk size for each thread

        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, n);
            threads.emplace_back([this, start, end, &errors, i]() {
                this->compute_section(start, end, errors[i]);
            });
        }

        // Join threads and combine errors
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
            error_val += errors[i];
        }

        // Copy the new interpolated ts to the input ordered pointset
        input_ordered_pointset = new_interpolated_ts;

        return error_val;
    }

    float compute_interpolated_adjacent_st() {
        float error_val = 0.0;
        new_interpolated_ts = input_ordered_pointset; // Copy the original ts

        // Calculate neighbour values for each vertex
        std::vector<std::vector<std::tuple<float, float, float, bool, float, bool, float, bool>>> neighbour_values;
        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            std::vector<std::tuple<float, float, float, bool, float, bool, float, bool>> row;
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                if (fixed_points[i][j]) { // optimization only for fixed points
                    row.push_back({1, 1, 0, false, 0, false, 0, false});
                    continue;
                }
                auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                row.push_back({r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts});
            }
            neighbour_values.push_back(row);
        }

        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                if (!fixed_points[i][j]) {
                    // Fetch the necessary neighbor values
                    auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = neighbour_values[i][j];

                    // Now solve for t_individual using dynamically computed m values
                    float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                    assert(t <= 0.0);
                    float d_t = t - input_ordered_pointset[i][j];
                    
                    // Adjust d_t respecting the non-overlapping constraints
                    d_t = respect_non_overlapping(i, j, d_t);
                    
                    error_val += std::abs(d_t);
                    new_interpolated_ts[i][j] = input_ordered_pointset[i][j] + learning_rate * d_t; // Apply learning rate
                }
            }
        }

        // Copy the new interpolated ts to the input ordered pointset
        input_ordered_pointset = new_interpolated_ts;

        return error_val;
    }

    std::vector<std::vector<float>> compute_interpolated_adjacent_errors() {
        std::vector<std::vector<float>> errors(input_ordered_pointset.size(), std::vector<float>(input_ordered_pointset[0].size(), 0.0f));

        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                assert(t <= 0.0);
                float d_t = t - input_ordered_pointset[i][j];
                d_t = respect_non_overlapping(i, j, d_t);
                errors[i][j] = std::abs(d_t);
            }
        }
        return errors;
    }

    void detect_and_unfix_wrong_fixed_adjacent() {
        auto errors = compute_interpolated_adjacent_errors();
        float sum_errors = 0.0;
        size_t count_fixed = 0;
        float sum_error_unfixed = 0.0;
        size_t count_unfixed = 0;

        // Calculate mean error of fixed points
        for (size_t i = 0; i < fixed_points.size(); i++) {
            for (size_t j = 0; j < fixed_points[i].size(); j++) {
                if (fixed_points[i][j]) {
                    sum_errors += errors[i][j];
                    count_fixed++;
                }
                else {
                    sum_error_unfixed += errors[i][j];
                    count_unfixed++;
                }
            }
        }

        float error_mean_fixed = count_fixed > 0 ? sum_errors / count_fixed : 0.0;
        float error_threshold = unfix_factor * error_mean_fixed;

        float error_mean_unfixed = count_unfixed > 0 ? sum_error_unfixed / count_unfixed : 0.0;
        float fixing_threshold = 0.2 * error_mean_unfixed;

        // Unfix points exceeding the error threshold
        for (size_t i = 0; i < fixed_points.size(); i++) {
            for (size_t j = 0; j < fixed_points[i].size(); j++) {
                if (fixed_points[i][j] && (errors[i][j] > error_threshold)) {
                    fixed_points[i][j] = false;
                }
                if (!fixed_points[i][j] && (errors[i][j] < fixing_threshold)) {
                    fixed_points[i][j] = true;
                }
            }
        }
    }

    std::vector<std::vector<float>> input_ordered_pointset;
    std::vector<std::vector<float>> new_interpolated_ts;
    std::vector<std::vector<bool>> fixed_points;
    std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices;
    float learning_rate = 0.1;
    size_t iterations = 3;
    mutable std::mutex mutex_;
    size_t progress = 0;
    size_t problem_size = -1;
    float error_val_d = 0.01;  // Delta for error value convergence
    float unfix_factor = 3.0;
    bool verbose;
};

std::vector<std::vector<float>> optimize_ordered_pointset(
    std::vector<std::vector<float>> input_ordered_pointset, 
    std::vector<std::vector<bool>> fixed_points, 
    std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices,
    float learning_rate = 0.1,
    int iterations = 3,
    float error_val_d = 0.01,  // Delta for error value convergence
    float unfix_factor = 3.0, // factor of which the error of a node has to be above the mean error to have the node go from fixed to floating during optimization
    bool verbose = true
    )
{
    OrderedPointsetOptimizer optimizer(input_ordered_pointset, fixed_points, neighbours_indices, learning_rate, iterations, error_val_d, unfix_factor, verbose);
    if (verbose) {
        std::cout << "Optimizing ordered pointset" << std::endl;
    }
    return std::move(optimizer.optimize_ordered_pointset_processor());
}


PYBIND11_MODULE(pointcloud_processing, m) {
    m.doc() = "pybind11 module for parallel point cloud processing";

    m.def("load_pointclouds", &load_pointclouds, "Function to load point clouds and return points, normals, and colors.");

    m.def("upsample_pointclouds", &upsample_pointclouds, "Function to load point clouds and return points, normals, and colors.");

    m.def("create_ordered_pointset", &create_ordered_pointset, 
          "Function to create an ordered point set from a point cloud.",
          py::arg("original_points"),
          py::arg("original_normals"),
          py::arg("umbilicus_points"),
          py::arg("angleStep") = 6,
          py::arg("z_spacing") = 10,
          py::arg("max_eucledian_distance") = 10,
          py::arg("verbose") = true,
          py::arg("min_z") = std::numeric_limits<float>::lowest(),
          py::arg("max_z") = std::numeric_limits<float>::max(),
          py::arg("min_wind") = 0.0f,
          py::arg("max_wind") = 0.0f
          );

    m.def("optimize_adjacent", &optimize_ordered_pointset, 
            "Function to optimize an ordered pointset.",
            py::arg("input_ordered_pointset"),
            py::arg("fixed_points"),
            py::arg("neighbours_indices"),
            py::arg("learning_rate") = 0.1,
            py::arg("iterations") = 3,
            py::arg("error_val_d") = 0.01,
            py::arg("unfix_factor") = 3.0,
            py::arg("verbose") = true);
}
