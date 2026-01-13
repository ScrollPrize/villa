#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <chrono>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/core/types.hpp>

#include <ceres/ceres.h>

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
        << "      --vis-ply PATH      Write visualization as PLY with vertex colors\n"
        << "      --fit-normals       Estimate local 3D normals from segments (within crop)\n"
        << "      --vis-normals PATH  Write fitted normals as PLY line segments\n\n"
        << "Notes:\n"
        << "  - Input is the directory created by vc_gen_normalgrids (contains metadata.json and xy/xz/yz).\n"
        << "  - Crop is optional; if omitted the full extent from metadata/available grids is used.\n";
}

struct CropBox3i {
    cv::Vec3i min; // inclusive
    cv::Vec3i max; // exclusive
};

static CropBox3i crop_from_args(const std::vector<int>& v) {
    // v = {x0,y0,z0,x1,y1,z1}
    if (v.size() != 6) {
        throw std::runtime_error("--crop expects 6 integers: x0 y0 z0 x1 y1 z1");
    }
    CropBox3i c;
    c.min = cv::Vec3i(v[0], v[1], v[2]);
    c.max = cv::Vec3i(v[3], v[4], v[5]);
    if (c.max[0] < c.min[0] || c.max[1] < c.min[1] || c.max[2] < c.min[2]) {
        throw std::runtime_error("--crop invalid: max must be >= min in all dimensions");
    }
    return c;
}

struct PlyWriter {
    explicit PlyWriter(const fs::path& path) : path(path) {}

    void begin_ascii_streaming() {
        // We must output vertices first, then edges (PLY element order).
        // To avoid storing all edges in memory OR iterating grids twice,
        // write edges to a temporary file and append it at the end.
        out.open(path, std::ios::in | std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Failed to open output file for writing: " + path.string());
        }

        tmp_edges_path = path;
        tmp_edges_path += ".edges.tmp";
        edges_out.open(tmp_edges_path, std::ios::out | std::ios::trunc);
        if (!edges_out) {
            throw std::runtime_error("Failed to open temp edge file for writing: " + tmp_edges_path.string());
        }

        write_header_with_counts(0, 0);
    }

    void write_polyline_streaming(const std::vector<cv::Point3f>& pts, const cv::Vec3b& color_bgr) {
        if (pts.size() < 2) return;

        const uint32_t base = next_vertex_idx;
        const int r = static_cast<int>(color_bgr[2]);
        const int g = static_cast<int>(color_bgr[1]);
        const int b = static_cast<int>(color_bgr[0]);

        for (const auto& p : pts) {
            out << p.x << " " << p.y << " " << p.z << " " << r << " " << g << " " << b << "\n";
            ++next_vertex_idx;
        }
        for (uint32_t i = 0; i + 1 < pts.size(); ++i) {
            edges_out << (base + i) << " " << (base + i + 1) << "\n";
        }

        vertex_count += pts.size();
        edge_count += (pts.size() - 1);
    }

    void write_segment_streaming(const cv::Point3f& a, const cv::Point3f& b, const cv::Vec3b& color_bgr) {
        const int r = static_cast<int>(color_bgr[2]);
        const int g = static_cast<int>(color_bgr[1]);
        const int bcol = static_cast<int>(color_bgr[0]);

        const uint32_t idx0 = next_vertex_idx++;
        const uint32_t idx1 = next_vertex_idx++;

        out << a.x << " " << a.y << " " << a.z << " " << r << " " << g << " " << bcol << "\n";
        out << b.x << " " << b.y << " " << b.z << " " << r << " " << g << " " << bcol << "\n";
        edges_out << idx0 << " " << idx1 << "\n";

        vertex_count += 2;
        edge_count += 1;
    }

    void end_streaming() {
        edges_out.close();

        // Append edges after vertices.
        {
            std::ifstream edges_in(tmp_edges_path, std::ios::in);
            if (!edges_in) {
                throw std::runtime_error("Failed to open temp edge file for reading: " + tmp_edges_path.string());
            }
            out << edges_in.rdbuf();
        }

        // Patch header in-place with final counts (fixed width => same header length).
        out.seekp(0, std::ios::beg);
        write_header_with_counts(vertex_count, edge_count);
        out.flush();
        out.close();

        std::error_code ec;
        fs::remove(tmp_edges_path, ec);
    }

private:
    void write_header_with_counts(size_t vtx, size_t edg) {
        char vbuf[32];
        char ebuf[32];
        snprintf(vbuf, sizeof(vbuf), "%020zu", vtx);
        snprintf(ebuf, sizeof(ebuf), "%020zu", edg);

        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment vc_ngrids visualization\n";
        out << "element vertex " << vbuf << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "element edge " << ebuf << "\n";
        out << "property int vertex1\n";
        out << "property int vertex2\n";
        out << "end_header\n";
    }

    fs::path path;
    size_t vertex_count = 0;
    size_t edge_count = 0;

    std::fstream out;
    fs::path tmp_edges_path;
    std::ofstream edges_out;
    uint32_t next_vertex_idx = 0;
};

static void add_gridstore_paths_as_ply_polylines(
    PlyWriter& ply,
    const vc::core::util::GridStore& grid,
    int plane_idx,
    int slice_idx,
    const CropBox3i& crop,
    const cv::Vec3b& color_bgr) {
    // Axis mapping for each plane:
    // plane 0 (xy @ z): (u,v,s) = (x,y,z)
    // plane 1 (xz @ y): (u,v,s) = (x,z,y)
    // plane 2 (yz @ x): (u,v,s) = (y,z,x)
    const int u_axis = (plane_idx == 2) ? 1 : 0;
    const int v_axis = (plane_idx == 0) ? 1 : 2;
    const int s_axis = (plane_idx == 0) ? 2 : (plane_idx == 1) ? 1 : 0;

    // Use GridStore ROI query to avoid decompressing/loading all paths in the slice.
    const cv::Rect query(crop.min[u_axis],
                         crop.min[v_axis],
                         crop.max[u_axis] - crop.min[u_axis],
                         crop.max[v_axis] - crop.min[v_axis]);

    const auto paths = grid.get(query);
    for (const auto& path_ptr : paths) {
        if (!path_ptr || path_ptr->size() < 2) continue;

        // Clip each segment against the 3D crop box so that segments crossing the bbox are kept,
        // and segments fully outside are dropped.
        auto clip_segment = [&](cv::Point3f& a, cv::Point3f& b) -> bool {
            // Liang–Barsky style clipping in 3D with t in [0,1].
            float t0 = 0.f;
            float t1 = 1.f;
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float dz = b.z - a.z;

            auto clip_1d = [&](float p, float q) -> bool {
                // p * t <= q
                if (p == 0.f) {
                    return q >= 0.f;
                }
                const float r = q / p;
                if (p < 0.f) {
                    if (r > t1) return false;
                    if (r > t0) t0 = r;
                } else {
                    if (r < t0) return false;
                    if (r < t1) t1 = r;
                }
                return true;
            };

            // Use closed-open bounds [min, max) by clipping to [min, max-eps].
            const float xmax = static_cast<float>(crop.max[0]) - 1e-3f;
            const float ymax = static_cast<float>(crop.max[1]) - 1e-3f;
            const float zmax = static_cast<float>(crop.max[2]) - 1e-3f;
            const float xmin = static_cast<float>(crop.min[0]);
            const float ymin = static_cast<float>(crop.min[1]);
            const float zmin = static_cast<float>(crop.min[2]);

            if (!clip_1d(-dx, a.x - xmin)) return false;
            if (!clip_1d(+dx, xmax - a.x)) return false;
            if (!clip_1d(-dy, a.y - ymin)) return false;
            if (!clip_1d(+dy, ymax - a.y)) return false;
            if (!clip_1d(-dz, a.z - zmin)) return false;
            if (!clip_1d(+dz, zmax - a.z)) return false;

            if (t1 < t0) return false;

            const cv::Point3f a0 = a;
            a = cv::Point3f(a0.x + t0 * dx, a0.y + t0 * dy, a0.z + t0 * dz);
            b = cv::Point3f(a0.x + t1 * dx, a0.y + t1 * dy, a0.z + t1 * dz);
            return true;
        };

        auto p3_of = [&](const cv::Point& p2) {
            float coords[3] = {0.f, 0.f, 0.f};
            coords[u_axis] = static_cast<float>(p2.x);
            coords[v_axis] = static_cast<float>(p2.y);
            coords[s_axis] = static_cast<float>(slice_idx);
            return cv::Point3f(coords[0], coords[1], coords[2]);
        };

        for (size_t i = 0; i + 1 < path_ptr->size(); ++i) {
            cv::Point3f a = p3_of((*path_ptr)[i]);
            cv::Point3f b = p3_of((*path_ptr)[i + 1]);
            if (clip_segment(a, b)) {
                ply.write_segment_streaming(a, b, color_bgr);
            }
        }
    }
}

static bool clip_segment_to_crop(cv::Point3f& a, cv::Point3f& b, const CropBox3i& crop) {
    // Liang–Barsky style clipping in 3D with t in [0,1].
    float t0 = 0.f;
    float t1 = 1.f;
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dz = b.z - a.z;

    auto clip_1d = [&](float p, float q) -> bool {
        // p * t <= q
        if (p == 0.f) {
            return q >= 0.f;
        }
        const float r = q / p;
        if (p < 0.f) {
            if (r > t1) return false;
            if (r > t0) t0 = r;
        } else {
            if (r < t0) return false;
            if (r < t1) t1 = r;
        }
        return true;
    };

    // Use closed-open bounds [min, max) by clipping to [min, max-eps].
    const float xmax = static_cast<float>(crop.max[0]) - 1e-3f;
    const float ymax = static_cast<float>(crop.max[1]) - 1e-3f;
    const float zmax = static_cast<float>(crop.max[2]) - 1e-3f;
    const float xmin = static_cast<float>(crop.min[0]);
    const float ymin = static_cast<float>(crop.min[1]);
    const float zmin = static_cast<float>(crop.min[2]);

    if (!clip_1d(-dx, a.x - xmin)) return false;
    if (!clip_1d(+dx, xmax - a.x)) return false;
    if (!clip_1d(-dy, a.y - ymin)) return false;
    if (!clip_1d(+dy, ymax - a.y)) return false;
    if (!clip_1d(-dz, a.z - zmin)) return false;
    if (!clip_1d(+dz, zmax - a.z)) return false;

    if (t1 < t0) return false;

    const cv::Point3f a0 = a;
    a = cv::Point3f(a0.x + t0 * dx, a0.y + t0 * dy, a0.z + t0 * dz);
    b = cv::Point3f(a0.x + t1 * dx, a0.y + t1 * dy, a0.z + t1 * dz);
    return true;
}

static float dist_sq_point_segment(const cv::Point3f& p, const cv::Point3f& a, const cv::Point3f& b) {
    const cv::Point3f ab = b - a;
    const float ab2 = ab.dot(ab);
    if (ab2 <= 1e-12f) {
        const cv::Point3f d = p - a;
        return d.dot(d);
    }
    const float t = std::max(0.f, std::min(1.f, (p - a).dot(ab) / ab2));
    const cv::Point3f q = a + t * ab;
    const cv::Point3f d = p - q;
    return d.dot(d);
}

struct NormalDotResidual {
    NormalDotResidual(const cv::Point3f& d, double w) : d_(d), w_(w) {}
    template <typename T>
    bool operator()(const T* const n, T* residual) const {
        residual[0] = T(w_) * (n[0] * T(d_.x) + n[1] * T(d_.y) + n[2] * T(d_.z));
        return true;
    }
    cv::Point3f d_;
    double w_;
};

struct UnitNormResidual {
    explicit UnitNormResidual(double w) : w_(w) {}
    template <typename T>
    bool operator()(const T* const n, T* residual) const {
        const T len = ceres::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + T(1e-12));
        residual[0] = T(w_) * (len - T(1.0));
        return true;
    }
    double w_;
};

static cv::Point3f pca_smallest_evec(const std::vector<cv::Point3f>& dirs_unit) {
    // Build 3x3 covariance C = sum d d^T.
    double c[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (const auto& d : dirs_unit) {
        c[0][0] += d.x * d.x;
        c[0][1] += d.x * d.y;
        c[0][2] += d.x * d.z;
        c[1][0] += d.y * d.x;
        c[1][1] += d.y * d.y;
        c[1][2] += d.y * d.z;
        c[2][0] += d.z * d.x;
        c[2][1] += d.z * d.y;
        c[2][2] += d.z * d.z;
    }

    // Power iteration on inverse is overkill; do a simple heuristic init:
    // pick cross of two non-parallel directions.
    for (size_t i = 0; i + 1 < dirs_unit.size(); ++i) {
        const cv::Point3f a = dirs_unit[i];
        const cv::Point3f b = dirs_unit[i + 1];
        const cv::Point3f n(a.y * b.z - a.z * b.y,
                            a.z * b.x - a.x * b.z,
                            a.x * b.y - a.y * b.x);
        const float n2 = n.dot(n);
        if (n2 > 1e-6f) {
            return n * (1.0f / std::sqrt(n2));
        }
    }

    // Fallback.
    return cv::Point3f(0.f, 0.f, 1.f);
}

static bool fit_normal_ceres(const std::vector<cv::Point3f>& dirs_unit, cv::Point3f& out_n) {
    if (dirs_unit.size() < 3) return false;

    double n[3];
    {
        const cv::Point3f init = pca_smallest_evec(dirs_unit);
        n[0] = init.x;
        n[1] = init.y;
        n[2] = init.z;
    }

    ceres::Problem problem;
    for (const auto& d : dirs_unit) {
        auto* cost = new ceres::AutoDiffCostFunction<NormalDotResidual, 1, 3>(new NormalDotResidual(d, 1.0));
        problem.AddResidualBlock(cost, nullptr, n);
    }
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<UnitNormResidual, 1, 3>(new UnitNormResidual(10.0)),
        nullptr,
        n);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 50;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(len > 1e-12)) return false;
    out_n = cv::Point3f(static_cast<float>(n[0] / len), static_cast<float>(n[1] / len), static_cast<float>(n[2] / len));
    return true;
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
        cv::Vec3i(0, 0, 0),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    PlyWriter ply(out_ply);
    ply.begin_ascii_streaming();

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
        const int crop_min = crop.min[pc.slice_axis];
        const int crop_max = crop.max[pc.slice_axis];
        int slice_start = align_down(crop_min, sparse_volume);
        if (slice_start < crop_min) slice_start += sparse_volume;

        for (int slice = slice_start; slice < crop_max; slice += std::max(1, sparse_volume)) {
            auto grid = try_load_grid(input_dir, pc.dir, slice);
            if (!grid) continue;
            add_gridstore_paths_as_ply_polylines(ply, *grid, pc.plane_idx, slice, crop, pc.color_bgr);
        }
    }

    ply.end_streaming();
}

static void run_fit_normals(
    const fs::path& input_dir,
    const fs::path& out_ply,
    const std::optional<CropBox3i>& crop_opt,
    int step = 64,
    float radius = 96.f) {
    vc::core::util::NormalGridVolume ngv(input_dir.string());
    const int sparse_volume = ngv.metadata().value("sparse-volume", 1);

    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Vec3i(0, 0, 0),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    PlyWriter ply(out_ply);
    ply.begin_ascii_streaming();
    const cv::Vec3b color_bgr(0, 255, 255); // yellow

    struct PlaneCfg {
        int plane_idx;
        const char* dir;
        int slice_axis;
    };
    const PlaneCfg planes[3] = {
        {0, "xy", 2},
        {1, "xz", 1},
        {2, "yz", 0},
    };

    const float r2 = radius * radius;
    const float normal_vis_scale = static_cast<float>(step) * 0.5f;

    const int sx0 = align_down(crop.min[0], step);
    const int sy0 = align_down(crop.min[1], step);
    const int sz0 = align_down(crop.min[2], step);

    const int nx = (crop.max[0] - sx0 + step - 1) / step;
    const int ny = (crop.max[1] - sy0 + step - 1) / step;
    const int nz = (crop.max[2] - sz0 + step - 1) / step;
    const int64_t total_samples = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
    int64_t processed = 0;
    int64_t written = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

    for (int z = sz0; z < crop.max[2]; z += step) {
        for (int y = sy0; y < crop.max[1]; y += step) {
            for (int x = sx0; x < crop.max[0]; x += step) {
                const cv::Point3f sample(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                const float sample_arr[3] = {sample.x, sample.y, sample.z};

                std::vector<cv::Point3f> dirs_unit;
                dirs_unit.reserve(256);

                for (const auto& pc : planes) {
                    // plane axes
                    const int u_axis = (pc.plane_idx == 2) ? 1 : 0;
                    const int v_axis = (pc.plane_idx == 0) ? 1 : 2;
                    const int s_axis = (pc.plane_idx == 0) ? 2 : (pc.plane_idx == 1) ? 1 : 0;

                    const int s_center = static_cast<int>(sample_arr[s_axis]);
                    const int s_min = std::max(crop.min[s_axis], static_cast<int>(std::floor(s_center - radius)));
                    const int s_max = std::min(crop.max[s_axis], static_cast<int>(std::ceil(s_center + radius)) + 1);
                    int slice_start = align_down(s_min, sparse_volume);
                    if (slice_start < s_min) slice_start += std::max(1, sparse_volume);

                    // 2D ROI for this plane around the sample
                    const int u0 = std::max(crop.min[u_axis], static_cast<int>(std::floor(sample_arr[u_axis] - radius)));
                    const int v0 = std::max(crop.min[v_axis], static_cast<int>(std::floor(sample_arr[v_axis] - radius)));
                    const int u1 = std::min(crop.max[u_axis], static_cast<int>(std::ceil(sample_arr[u_axis] + radius)) + 1);
                    const int v1 = std::min(crop.max[v_axis], static_cast<int>(std::ceil(sample_arr[v_axis] + radius)) + 1);
                    if (u1 <= u0 || v1 <= v0) continue;

                    const cv::Rect query(u0, v0, u1 - u0, v1 - v0);

                    for (int slice = slice_start; slice < s_max; slice += std::max(1, sparse_volume)) {
                        auto grid = try_load_grid(input_dir, pc.dir, slice);
                        if (!grid) continue;

                        const auto paths = grid->get(query);
                        for (const auto& path_ptr : paths) {
                            if (!path_ptr || path_ptr->size() < 2) continue;

                            auto p3_of = [&](const cv::Point& p2) {
                                float coords[3] = {0.f, 0.f, 0.f};
                                coords[u_axis] = static_cast<float>(p2.x);
                                coords[v_axis] = static_cast<float>(p2.y);
                                coords[s_axis] = static_cast<float>(slice);
                                return cv::Point3f(coords[0], coords[1], coords[2]);
                            };

                            for (size_t i = 0; i + 1 < path_ptr->size(); ++i) {
                                cv::Point3f a = p3_of((*path_ptr)[i]);
                                cv::Point3f b = p3_of((*path_ptr)[i + 1]);

                                if (!clip_segment_to_crop(a, b, crop)) continue;
                                if (dist_sq_point_segment(sample, a, b) > r2) continue;

                                const cv::Point3f d = b - a;
                                const float d2 = d.dot(d);
                                if (d2 <= 1e-6f) continue;
                                const float inv = 1.0f / std::sqrt(d2);
                                dirs_unit.emplace_back(d.x * inv, d.y * inv, d.z * inv);
                            }
                        }
                    }
                }

                cv::Point3f n;
                if (!fit_normal_ceres(dirs_unit, n)) {
                    ++processed;
                } else {
                    const cv::Point3f a = sample;
                    const cv::Point3f b = sample + normal_vis_scale * n;
                    ply.write_segment_streaming(a, b, color_bgr);
                    ++processed;
                    ++written;
                }

                const auto now = std::chrono::steady_clock::now();
                if (now - t_last >= std::chrono::seconds(2)) {
                    const double elapsed = std::chrono::duration<double>(now - t0).count();
                    const double rate = (elapsed > 1e-9) ? (static_cast<double>(processed) / elapsed) : 0.0;
                    const double rem = static_cast<double>(std::max<int64_t>(0, total_samples - processed));
                    const double eta = (rate > 1e-9) ? (rem / rate) : 0.0;
                    std::cerr << "fit-normals: " << processed << "/" << total_samples
                              << " (written " << written << ")"
                              << " | elapsed " << elapsed << "s"
                              << " | rate " << rate << " samples/s"
                              << " | ETA " << eta << "s\n";
                    t_last = now;
                }
            }
        }
    }

    ply.end_streaming();
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("vc_ngrids options");
    desc.add_options()
        ("help,h", "Print help")
        ("input,i", po::value<std::string>()->required(), "Input NormalGridVolume directory")
        ("crop,c", po::value<std::vector<int>>()->multitoken(), "Crop x0 y0 z0 x1 y1 z1")
        ("vis-ply", po::value<std::string>(), "Write visualization PLY file (with colors)")
        ("fit-normals", "Estimate local 3D normals from segments (requires --vis-normals)")
        ("vis-normals", po::value<std::string>(), "Write fitted normals as PLY line segments");

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

    if (vm.count("fit-normals")) {
        if (!vm.count("vis-normals")) {
            std::cerr << "Error: --fit-normals requires --vis-normals PATH\n";
            return 1;
        }
        run_fit_normals(input_dir, fs::path(vm["vis-normals"].as<std::string>()), crop);
        return 0;
    }

    std::cerr << "Error: no output specified. Use --vis-ply or --fit-normals --vis-normals.\n\n";
    print_usage();
    return 1;
}
