#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/core/types.hpp>

#include <ceres/ceres.h>
#include <omp.h>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

#include "z5/factory.hxx"
#include "z5/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/filesystem/handle.hxx"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xadapt.hpp)

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

static void print_usage() {
    std::cout
        << "vc_ngrids: Read a NormalGridVolume and derive/visualize data.\n\n"
        << "Usage: vc_ngrids [options]\n\n"
        << "Options:\n"
        << "  -h, --help              Print this help message\n"
        << "  -i, --input PATH        Input NormalGridVolume directory OR normals zarr root (required)\n"
        << "  -c, --crop x0 y0 z0 x1 y1 z1   Crop bounding box in voxel coords (half-open)\n"
        << "      --vis-ply PATH      Write visualization as PLY with vertex colors\n"
        << "      --fit-normals       Estimate local 3D normals from segments (within crop)\n"
        << "      --vis-normals PATH  Write fitted normals as PLY line segments\n\n"
        << "      --output-zarr PATH  Write fitted normals to a zarr directory (direction-field encoding)\n\n"
        << "Notes:\n"
        << "  - Input can be a directory created by vc_gen_normalgrids (contains metadata.json and xy/xz/yz).\n"
        << "  - Or, input can be a normals zarr root (contains x/0, y/0, z/0 datasets).\n"
        << "  - Crop is optional; if omitted the full extent from metadata/available grids is used.\n";
}

struct CropBox3i {
    cv::Vec3i min; // inclusive
    cv::Vec3i max; // exclusive
};

static inline uint8_t encode_dir_component(float v) {
    // Match direction-field encoding used by Chunked3dVec3fFromUint8:
    // decode: (u8 - 128) / 127 -> [-1, 1]
    if (!std::isfinite(v)) return 128;
    v = std::max(-1.0f, std::min(1.0f, v));
    const int q = static_cast<int>(std::lround(v * 127.0f + 128.0f));
    return static_cast<uint8_t>(std::max(0, std::min(255, q)));
}

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

    // Helpers for manual streaming/merge (avoid making internals public).
    void append_vertex_lines_from_file(const fs::path& vtx_file, size_t vtx_count_to_add) {
        std::ifstream in(vtx_file, std::ios::in);
        if (!in) {
            throw std::runtime_error("Failed to open temp vertex file for merge: " + vtx_file.string());
        }
        out << in.rdbuf();
        vertex_count += vtx_count_to_add;
        next_vertex_idx += static_cast<uint32_t>(vtx_count_to_add);
    }

    void append_edge_lines_from_file_with_offset(const fs::path& edg_file, size_t vtx_offset, size_t edg_count_to_add) {
        std::ifstream in(edg_file, std::ios::in);
        if (!in) {
            throw std::runtime_error("Failed to open temp edge file for merge: " + edg_file.string());
        }
        std::string line;
        size_t got = 0;
        while (got < edg_count_to_add && std::getline(in, line)) {
            size_t a = 0, b = 0;
            if (sscanf(line.c_str(), "%zu %zu", &a, &b) != 2) {
                throw std::runtime_error("Invalid edge line in temp edge file: " + edg_file.string());
            }
            edges_out << (a + vtx_offset) << " " << (b + vtx_offset) << "\n";
            ++got;
        }
        if (got != edg_count_to_add) {
            throw std::runtime_error("Truncated temp edge file for merge: " + edg_file.string());
        }
        edge_count += edg_count_to_add;
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

static inline float dist_sq_point_segment_appx(const cv::Point3f& p, const cv::Point3f& a, const cv::Point3f& b) {
    // Approximate distance: use the segment midpoint instead of true point-to-segment distance.
    const cv::Point3f m = 0.5f * (a + b);
    const cv::Point3f d = p - m;
    return d.dot(d);
}

static inline bool segment_intersects_local_roi_2d(const cv::Point& a, const cv::Point& b, const cv::Rect& roi) {
    // Fast 2D early reject: check segment AABB intersects ROI.
    // This avoids 3D conversion and distance checks for clearly irrelevant segments.
    const int minx = std::min(a.x, b.x);
    const int maxx = std::max(a.x, b.x);
    const int miny = std::min(a.y, b.y);
    const int maxy = std::max(a.y, b.y);

    // roi is [x, x+w) x [y, y+h)
    if (maxx < roi.x) return false;
    if (minx >= roi.x + roi.width) return false;
    if (maxy < roi.y) return false;
    if (miny >= roi.y + roi.height) return false;
    return true;
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

static bool fit_normal_ceres(
    const std::vector<cv::Point3f>& dirs_unit,
    const std::vector<double>& weights,
    cv::Point3f& out_n,
    int* out_num_iterations,
    double* out_rms,
    double* out_solve_seconds) {
    if (dirs_unit.size() < 3) return false;
    if (weights.size() != dirs_unit.size()) return false;

    double n[3];
    {
        const cv::Point3f init = pca_smallest_evec(dirs_unit);
        n[0] = init.x;
        n[1] = init.y;
        n[2] = init.z;
    }

    ceres::Problem problem;
    for (size_t i = 0; i < dirs_unit.size(); ++i) {
        const auto& d = dirs_unit[i];
        const double w = weights[i];
        auto* cost = new ceres::AutoDiffCostFunction<NormalDotResidual, 1, 3>(new NormalDotResidual(d, w));
        problem.AddResidualBlock(cost, nullptr, n);
    }
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<UnitNormResidual, 1, 3>(new UnitNormResidual(10.0)),
        nullptr,
        n);

    ceres::Solver::Options opts;
    // opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opts.linear_solver_type = ceres::SPARSE_SCHUR;
    opts.max_num_iterations = 1000;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const auto solve_t0 = std::chrono::steady_clock::now();
    ceres::Solve(opts, &problem, &summary);
    const auto solve_t1 = std::chrono::steady_clock::now();

    if (out_solve_seconds != nullptr) {
        *out_solve_seconds = std::chrono::duration<double>(solve_t1 - solve_t0).count();
    }

    if (out_num_iterations != nullptr) {
        *out_num_iterations = static_cast<int>(summary.iterations.size());
    }
    if (out_rms != nullptr) {
        // Ceres cost = 1/2 * sum(residual^2)
        const double denom = std::max(1, summary.num_residuals);
        *out_rms = std::sqrt(2.0 * summary.final_cost / denom);
    }

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

static cv::Point3f point_for_slice_query(const cv::Point3f& base, int plane_idx, int slice_idx) {
    cv::Point3f p = base;
    if (plane_idx == 0) {
        // xy @ z
        p.z = static_cast<float>(slice_idx);
    } else if (plane_idx == 1) {
        // xz @ y
        p.y = static_cast<float>(slice_idx);
    } else {
        // yz @ x
        p.x = static_cast<float>(slice_idx);
    }
    return p;
}

static int align_down(int v, int step) {
    if (step <= 1) return v;
    if (v >= 0) return (v / step) * step;
    // For negative, ensure we still go downwards.
    return -(((-v + step - 1) / step) * step);
}

static std::optional<cv::Vec3i> infer_volume_shape_from_grids(const fs::path& ngv_root) {
    // Infer (X,Y,Z) from GridStore slice dimensions.
    // XY: (width,height)=(X,Y)
    // XZ: (width,height)=(X,Z)
    // YZ: (width,height)=(Y,Z)
    auto find_any_valid_grid = [&](const fs::path& dir) -> std::optional<fs::path> {
        // Note: vc_gen_normalgrids may create empty placeholder files for empty slices.
        // Those are not valid GridStore files and must be skipped.
        if (!fs::exists(dir) || !fs::is_directory(dir)) return std::nullopt;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".grid") continue;
            // quick reject: GridStore files have a header; empty placeholder files are size 0
            std::error_code ec;
            const auto sz = fs::file_size(entry.path(), ec);
            if (ec || sz < 16) continue;
            try {
                vc::core::util::GridStore g(entry.path().string());
                const auto s = g.size();
                if (s.width > 0 && s.height > 0) {
                    return entry.path();
                }
            } catch (...) {
                continue;
            }
        }
        return std::nullopt;
    };

    std::optional<int> X, Y, Z;
    auto try_xy = [&]() {
        auto p = find_any_valid_grid(ngv_root / "xy");
        if (!p) return;
        vc::core::util::GridStore g(p->string());
        const auto sz = g.size();
        X = sz.width;
        Y = sz.height;
    };
    auto try_xz = [&]() {
        auto p = find_any_valid_grid(ngv_root / "xz");
        if (!p) return;
        vc::core::util::GridStore g(p->string());
        const auto sz = g.size();
        X = X.value_or(sz.width);
        Z = sz.height;
    };
    auto try_yz = [&]() {
        auto p = find_any_valid_grid(ngv_root / "yz");
        if (!p) return;
        vc::core::util::GridStore g(p->string());
        const auto sz = g.size();
        Y = Y.value_or(sz.width);
        Z = Z.value_or(sz.height);
    };

    try_xy();
    try_xz();
    try_yz();

    if (!X || !Y || !Z) return std::nullopt;
    if (*X <= 0 || *Y <= 0 || *Z <= 0) return std::nullopt;
    return cv::Vec3i(*X, *Y, *Z);
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
            auto grid = ngv.query_nearest(point_for_slice_query(cv::Point3f(0.f, 0.f, 0.f), pc.plane_idx, slice), pc.plane_idx);
            if (!grid) continue;
            add_gridstore_paths_as_ply_polylines(ply, *grid, pc.plane_idx, slice, crop, pc.color_bgr);
        }
    }

    ply.end_streaming();
}

static bool looks_like_normals_zarr_root(const fs::path& input_dir) {
    // Minimal heuristic: groups x,y,z exist and each has a scale dataset directory ("0").
    return fs::is_directory(input_dir / "x") && fs::is_directory(input_dir / "y") && fs::is_directory(input_dir / "z") &&
           fs::is_directory(input_dir / "x" / "0") && fs::is_directory(input_dir / "y" / "0") && fs::is_directory(input_dir / "z" / "0");
}

static void run_vis_normals_zarr_as_ply(const fs::path& zarr_root, const fs::path& out_ply, const std::optional<CropBox3i>& crop_opt) {
    // Determine delimiter from x/0/.zarray (fallback "." to match other tools).
    std::string delim = ".";
    {
        const fs::path zarray_path = zarr_root / "x" / "0" / ".zarray";
        if (fs::exists(zarray_path)) {
            nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_path));
            delim = j.value<std::string>("dimension_separator", ".");
        }
    }

    // Optional metadata written by vc_ngrids --output-zarr.
    cv::Vec3i origin_xyz(0, 0, 0);
    int step = 1;
    {
        z5::filesystem::handle::File rootFile(zarr_root);
        z5::filesystem::handle::Group root(rootFile, "");
        try {
            nlohmann::json attrs;
            z5::filesystem::readAttributes(root, attrs);
            if (attrs.contains("grid_origin_xyz") && attrs["grid_origin_xyz"].is_array() && attrs["grid_origin_xyz"].size() == 3) {
                origin_xyz = cv::Vec3i(attrs["grid_origin_xyz"][0].get<int>(), attrs["grid_origin_xyz"][1].get<int>(), attrs["grid_origin_xyz"][2].get<int>());
            }
            if (attrs.contains("sample_step")) {
                step = std::max(1, attrs["sample_step"].get<int>());
            }
        } catch (...) {
            // Attributes are optional; keep defaults.
        }
    }

    auto open_u8_zyx = [&](const char* axis) -> std::unique_ptr<z5::Dataset> {
        z5::filesystem::handle::File file(zarr_root);
        z5::filesystem::handle::Group axis_group(file, axis);
        z5::filesystem::handle::Dataset ds_handle(axis_group, "0", delim);
        return z5::filesystem::openDataset(ds_handle);
    };

    auto dsx = open_u8_zyx("x");
    auto dsy = open_u8_zyx("y");
    auto dsz = open_u8_zyx("z");
    if (!dsx || !dsy || !dsz) {
        throw std::runtime_error("Failed to open x/y/z datasets under zarr root: " + zarr_root.string());
    }
    if (dsx->shape() != dsy->shape() || dsx->shape() != dsz->shape()) {
        throw std::runtime_error("x/y/z datasets have different shapes under: " + zarr_root.string());
    }

    const auto& shape = dsx->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("Expected 3D datasets (ZYX) for normals zarr under: " + zarr_root.string());
    }
    const size_t Z = shape[0];
    const size_t Y = shape[1];
    const size_t X = shape[2];

    xt::xarray<uint8_t> ax = xt::zeros<uint8_t>({Z, Y, X});
    xt::xarray<uint8_t> ay = xt::zeros<uint8_t>({Z, Y, X});
    xt::xarray<uint8_t> az = xt::zeros<uint8_t>({Z, Y, X});
    z5::types::ShapeType off = {0, 0, 0};
    z5::multiarray::readSubarray<uint8_t>(*dsx, ax, off.begin());
    z5::multiarray::readSubarray<uint8_t>(*dsy, ay, off.begin());
    z5::multiarray::readSubarray<uint8_t>(*dsz, az, off.begin());

    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Vec3i(std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    auto decode = [&](uint8_t u) -> float {
        // (u8 - 128) / 127
        return (static_cast<int>(u) - 128) / 127.0f;
    };

    const float vis_scale = static_cast<float>(step) * 0.5f;
    const cv::Vec3b color_bgr(0, 255, 255); // yellow

    PlyWriter ply(out_ply);
    ply.begin_ascii_streaming();

    for (size_t iz = 0; iz < Z; ++iz) {
        for (size_t iy = 0; iy < Y; ++iy) {
            for (size_t ix = 0; ix < X; ++ix) {
                const uint8_t ux = ax(iz, iy, ix);
                const uint8_t uy = ay(iz, iy, ix);
                const uint8_t uz = az(iz, iy, ix);
                if (ux == 128 && uy == 128 && uz == 128) continue;

                const float nx = decode(ux);
                const float ny = decode(uy);
                const float nz = decode(uz);
                if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) continue;

                const int x = origin_xyz[0] + static_cast<int>(ix) * step;
                const int y = origin_xyz[1] + static_cast<int>(iy) * step;
                const int z = origin_xyz[2] + static_cast<int>(iz) * step;

                if (x < crop.min[0] || x >= crop.max[0] || y < crop.min[1] || y >= crop.max[1] || z < crop.min[2] || z >= crop.max[2]) {
                    continue;
                }

                const cv::Point3f a(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                const cv::Point3f b = a + vis_scale * cv::Point3f(nx, ny, nz);
                ply.write_segment_streaming(a, b, color_bgr);
            }
        }
    }

    ply.end_streaming();
}

static void run_fit_normals(
    const fs::path& input_dir,
    const std::optional<fs::path>& out_ply_opt,
    const std::optional<CropBox3i>& crop_opt,
    const std::optional<fs::path>& out_zarr_opt,
    int step = 16,
    float radius = 96.f) {
    vc::core::util::NormalGridVolume ngv(input_dir.string());
    const int sparse_volume = ngv.metadata().value("sparse-volume", 1);

    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Vec3i(0, 0, 0),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    // Optional PLY output: per-thread temp files then merge.
    const int nthreads = std::max(1, omp_get_max_threads());
    struct ThreadOut {
        fs::path vtx_path;
        fs::path edg_path;
        std::ofstream vtx;
        std::ofstream edg;
        size_t vtx_count = 0;
        size_t edg_count = 0;
    };
    std::vector<ThreadOut> t_out;
    if (out_ply_opt.has_value()) {
        const fs::path& out_ply = *out_ply_opt;
        t_out.resize(static_cast<size_t>(nthreads));
        for (int t = 0; t < nthreads; ++t) {
            t_out[t].vtx_path = out_ply;
            t_out[t].vtx_path += ".normals.vtx.part" + std::to_string(t);
            t_out[t].edg_path = out_ply;
            t_out[t].edg_path += ".normals.edg.part" + std::to_string(t);
            t_out[t].vtx.open(t_out[t].vtx_path, std::ios::out | std::ios::trunc);
            t_out[t].edg.open(t_out[t].edg_path, std::ios::out | std::ios::trunc);
            if (!t_out[t].vtx || !t_out[t].edg) {
                throw std::runtime_error("Failed to open temp normal output files for thread " + std::to_string(t));
            }
        }
    }

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
    const float sigma = radius / 3.0f;
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma + 1e-12f);
    const float normal_vis_scale = static_cast<float>(step) * 0.5f;

    const int sx0 = align_down(crop.min[0], step);
    const int sy0 = align_down(crop.min[1], step);
    const int sz0 = align_down(crop.min[2], step);

    // When writing zarr, allocate a full-sized (downsampled) volume and only fill within crop.
    // This way downstream tools can use global voxel coordinates without needing crop-origin offsets.
    int nx = 0, ny = 0, nz = 0;
    if (out_zarr_opt.has_value()) {
        const auto vol_xyz_opt = infer_volume_shape_from_grids(input_dir);
        if (!vol_xyz_opt.has_value()) {
            throw std::runtime_error("Failed to infer volume shape from normal grids (need at least two plane dirs with .grid files)");
        }
        const cv::Vec3i vol_xyz = *vol_xyz_opt;
        if (crop.max[0] > vol_xyz[0] || crop.max[1] > vol_xyz[1] || crop.max[2] > vol_xyz[2]) {
            std::stringstream msg;
            msg << "Crop max exceeds inferred volume shape: crop.max=(" << crop.max[0] << "," << crop.max[1] << "," << crop.max[2]
                << ") vs inferred vol_xyz=(" << vol_xyz[0] << "," << vol_xyz[1] << "," << vol_xyz[2] << ")";
            throw std::runtime_error(msg.str());
        }
        nx = (vol_xyz[0] + step - 1) / step;
        ny = (vol_xyz[1] + step - 1) / step;
        nz = (vol_xyz[2] + step - 1) / step;
    } else {
        // For PLY-only mode, only consider sample lattice in the crop.
        nx = (crop.max[0] - sx0 + step - 1) / step;
        ny = (crop.max[1] - sy0 + step - 1) / step;
        nz = (crop.max[2] - sz0 + step - 1) / step;
    }

    // Progress reporting should reflect *work done*, i.e. samples evaluated within the crop,
    // not the size of the allocated output lattice.
    const int crop_nx = (crop.max[0] - sx0 + step - 1) / step;
    const int crop_ny = (crop.max[1] - sy0 + step - 1) / step;
    const int crop_nz = (crop.max[2] - sz0 + step - 1) / step;
    const int64_t total_samples = static_cast<int64_t>(crop_nx) * static_cast<int64_t>(crop_ny) * static_cast<int64_t>(crop_nz);

    const int64_t full_samples = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);

    // Stats: iterations-to-solved histogram and RMS histogram.
    // Note: we keep a coarse RMS bucket histogram for printing and a fine histogram for median estimation.
    constexpr int kFineRmsBins = 1000;
    struct FitStats {
        // Iteration buckets: [0-4],[5-9],[10-19],[20-49],[50-99],[100-199],[200+]
        uint64_t iters_buckets[7] = {0, 0, 0, 0, 0, 0, 0};

        // Sample-count buckets (#segments used): [0-511],[512-1023],[1024-2047],[2048-4095],[4096-8191],[8192-16383],[16384+]
        uint64_t samples_buckets[7] = {0, 0, 0, 0, 0, 0, 0};

        // Coarse RMS buckets: [0-0.01),[0.01-0.02),[0.02-0.05),[0.05-0.1),[0.1-0.2),[0.2-0.5),[0.5+)
        uint64_t rms_buckets[7] = {0, 0, 0, 0, 0, 0, 0};

        // Fine RMS histogram for median: bins over [0, 1.0], plus overflow bin.
        uint64_t rms_fine[kFineRmsBins + 1] = {0};

        uint64_t ok_count = 0;
        double rms_sum = 0.0;
        double rms_max = 0.0;

        // Thread-summed timings (seconds).
        uint64_t samples_total = 0;
        double t_ng_read_s = 0.0;
        double t_preproc_s = 0.0;
        double t_solve_s = 0.0;
        double t_overhead_s = 0.0;

        // Debug: distance test rejection rate.
        uint64_t dist_test_total = 0;
        uint64_t dist_test_reject = 0;

        void add(int iters, int samples, double rms) {
            ++ok_count;
            rms_sum += rms;
            rms_max = std::max(rms_max, rms);

            const int ib = (iters < 5) ? 0 : (iters < 10) ? 1 : (iters < 20) ? 2 : (iters < 50) ? 3 : (iters < 100) ? 4 : (iters < 200) ? 5 : 6;
            ++iters_buckets[ib];

            const int sb = (samples < 512) ? 0 : (samples < 1024) ? 1 : (samples < 2048) ? 2 : (samples < 4096) ? 3 : (samples < 8192) ? 4 : (samples < 16384) ? 5 : 6;
            ++samples_buckets[sb];

            const int rb = (rms < 0.01) ? 0 : (rms < 0.02) ? 1 : (rms < 0.05) ? 2 : (rms < 0.10) ? 3 : (rms < 0.20) ? 4 : (rms < 0.50) ? 5 : 6;
            ++rms_buckets[rb];

            // Fine binning.
            constexpr double max_rms = 1.0;
            int fi = 0;
            if (rms >= max_rms) {
                fi = kFineRmsBins; // overflow
            } else if (rms <= 0.0) {
                fi = 0;
            } else {
                fi = static_cast<int>(std::floor((rms / max_rms) * kFineRmsBins));
                fi = std::max(0, std::min(kFineRmsBins - 1, fi));
            }
            ++rms_fine[fi];
        }

        void add_timing(double ng_read_s, double preproc_s, double solve_s, double overhead_s) {
            ++samples_total;
            t_ng_read_s += ng_read_s;
            t_preproc_s += preproc_s;
            t_solve_s += solve_s;
            t_overhead_s += overhead_s;
        }

        void add_dist_test(bool rejected) {
            ++dist_test_total;
            if (rejected) ++dist_test_reject;
        }
    };

    auto stats_of = [&]() -> std::vector<FitStats> {
        return std::vector<FitStats>(static_cast<size_t>(std::max(1, omp_get_max_threads())));
    };

    std::vector<FitStats> stats = stats_of();

    auto merge_stats = [&](FitStats& acc, const FitStats& s) {
        for (int i = 0; i < 7; ++i) {
            acc.iters_buckets[i] += s.iters_buckets[i];
            acc.samples_buckets[i] += s.samples_buckets[i];
            acc.rms_buckets[i] += s.rms_buckets[i];
        }
        for (int i = 0; i < kFineRmsBins + 1; ++i) {
            acc.rms_fine[i] += s.rms_fine[i];
        }
        acc.ok_count += s.ok_count;
        acc.rms_sum += s.rms_sum;
        acc.rms_max = std::max(acc.rms_max, s.rms_max);

        acc.samples_total += s.samples_total;
        acc.t_ng_read_s += s.t_ng_read_s;
        acc.t_preproc_s += s.t_preproc_s;
        acc.t_solve_s += s.t_solve_s;
        acc.t_overhead_s += s.t_overhead_s;

        acc.dist_test_total += s.dist_test_total;
        acc.dist_test_reject += s.dist_test_reject;
    };

    auto summarize_stats = [&](const std::vector<FitStats>& per_thread) -> FitStats {
        FitStats acc;
        for (const auto& s : per_thread) {
            merge_stats(acc, s);
        }
        return acc;
    };

    auto estimate_median_from_fine = [&](const FitStats& acc) -> double {
        if (acc.ok_count == 0) return 0.0;
        const uint64_t target = (acc.ok_count - 1) / 2; // lower median
        uint64_t cum = 0;
        int idx = 0;
        for (; idx < kFineRmsBins + 1; ++idx) {
            cum += acc.rms_fine[idx];
            if (cum > target) break;
        }
        if (idx >= kFineRmsBins) {
            return 1.0; // overflow bin => >= 1.0
        }
        // Bin center.
        const double bin_w = 1.0 / kFineRmsBins;
        return (idx + 0.5) * bin_w;
    };

    // Optional output: store fitted normals on the sample lattice.
    // Encoding matches direction-field zarrs: 3 uint8 volumes x,y,z, decoded as (v-128)/127.
    std::vector<uint8_t> enc_x;
    std::vector<uint8_t> enc_y;
    std::vector<uint8_t> enc_z;
    if (out_zarr_opt.has_value()) {
        const size_t n = static_cast<size_t>(std::max<int64_t>(0, full_samples));
        enc_x.assign(n, 128);
        enc_y.assign(n, 128);
        enc_z.assign(n, 128);
    }
    int64_t processed = 0;
    int64_t written = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;

    auto report_progress = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        const double rate = (elapsed > 1e-9) ? (static_cast<double>(processed) / elapsed) : 0.0;
        const double rem = static_cast<double>(std::max<int64_t>(0, total_samples - processed));
        const double eta = (rate > 1e-9) ? (rem / rate) : 0.0;
        std::cerr << "fit-normals: " << processed << "/" << total_samples
                  << " (written " << written << ")"
                  << " | elapsed " << elapsed << "s"
                  << " | rate " << rate << " samples/s"
                  << " | ETA " << eta << "s\n";
    };

    auto report_stats = [&]() {
        const FitStats acc = summarize_stats(stats);
        const double avg = (acc.ok_count > 0) ? (acc.rms_sum / static_cast<double>(acc.ok_count)) : 0.0;
        const double med = estimate_median_from_fine(acc);
        const double work = acc.t_ng_read_s + acc.t_preproc_s + acc.t_solve_s + acc.t_overhead_s;
        const double png = (work > 1e-12) ? (100.0 * acc.t_ng_read_s / work) : 0.0;
        const double ppp = (work > 1e-12) ? (100.0 * acc.t_preproc_s / work) : 0.0;
        const double ps = (work > 1e-12) ? (100.0 * acc.t_solve_s / work) : 0.0;
        const double po = (work > 1e-12) ? (100.0 * acc.t_overhead_s / work) : 0.0;
        std::cerr << "fit-normals stats: ok=" << acc.ok_count
                  << " | rms(avg/med/max)=" << avg << "/" << med << "/" << acc.rms_max << "\n";
        std::cerr << "  iters buckets [0-4,5-9,10-19,20-49,50-99,100-199,200+]: "
                  << acc.iters_buckets[0] << "," << acc.iters_buckets[1] << "," << acc.iters_buckets[2] << "," << acc.iters_buckets[3]
                  << "," << acc.iters_buckets[4] << "," << acc.iters_buckets[5] << "," << acc.iters_buckets[6] << "\n";
        std::cerr << "  samples buckets [0-511,512-1023,1024-2047,2048-4095,4096-8191,8192-16383,16384+]: "
                  << acc.samples_buckets[0] << "," << acc.samples_buckets[1] << "," << acc.samples_buckets[2] << "," << acc.samples_buckets[3]
                  << "," << acc.samples_buckets[4] << "," << acc.samples_buckets[5] << "," << acc.samples_buckets[6] << "\n";
        std::cerr << "  rms buckets [<0.01,<0.02,<0.05,<0.1,<0.2,<0.5,>=0.5]: "
                  << acc.rms_buckets[0] << "," << acc.rms_buckets[1] << "," << acc.rms_buckets[2] << "," << acc.rms_buckets[3]
                  << "," << acc.rms_buckets[4] << "," << acc.rms_buckets[5] << "," << acc.rms_buckets[6] << "\n";

        // Time breakdown is thread-summed (can exceed wall time with OpenMP).
        std::cerr << "  time(thread-summed): samples=" << acc.samples_total
                  << " | ng_read=" << acc.t_ng_read_s << "s (" << png << "%)"
                  << " | preproc=" << acc.t_preproc_s << "s (" << ppp << "%)"
                  << " | solve=" << acc.t_solve_s << "s (" << ps << "%)"
                  << " | overhead=" << acc.t_overhead_s << "s (" << po << "%)\n";

        const double rej = (acc.dist_test_total > 0) ? (100.0 * static_cast<double>(acc.dist_test_reject) / static_cast<double>(acc.dist_test_total)) : 0.0;
        std::cerr << "  dist2>r2 rejects: " << acc.dist_test_reject << "/" << acc.dist_test_total << " (" << rej << "%)\n";
    };

    // Only compute normals inside crop, but write them into the full lattice when out_zarr is enabled.
    #pragma omp parallel for collapse(3) schedule(dynamic,1)
    for (int z = sz0; z < crop.max[2]; z += step) {
        for (int y = sy0; y < crop.max[1]; y += step) {
            for (int x = sx0; x < crop.max[0]; x += step) {
                const int tid = omp_get_thread_num();
                ThreadOut* tout = nullptr;
                if (out_ply_opt.has_value()) {
                    tout = &t_out[static_cast<size_t>(tid)];
                }

                const cv::Point3f sample(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                const float sample_arr[3] = {sample.x, sample.y, sample.z};

                const auto t_sample0 = std::chrono::steady_clock::now();

                std::vector<cv::Point3f> dirs_unit;
                dirs_unit.reserve(256);
                std::vector<double> weights;
                weights.reserve(256);

                double t_ng_read_s = 0.0;
                double t_preproc_s = 0.0;
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
                        const auto t_read0 = std::chrono::steady_clock::now();
                        auto grid = ngv.query_nearest(point_for_slice_query(sample, pc.plane_idx, slice), pc.plane_idx);
                        if (!grid) continue;

                        const auto paths = grid->get(query);
                        const auto t_read1 = std::chrono::steady_clock::now();
                        t_ng_read_s += std::chrono::duration<double>(t_read1 - t_read0).count();

                        const auto t_pp0 = std::chrono::steady_clock::now();
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
                                const cv::Point a2 = (*path_ptr)[i];
                                const cv::Point b2 = (*path_ptr)[i + 1];

                                // 2D early reject before 3D conversion.
                                if (!segment_intersects_local_roi_2d(a2, b2, query)) continue;

                                cv::Point3f a = p3_of(a2);
                                cv::Point3f b = p3_of(b2);

                                if (!clip_segment_to_crop(a, b, crop)) continue;
                                const float dist2 = dist_sq_point_segment_appx(sample, a, b);
                                const bool reject = (dist2 > r2);
                                stats[static_cast<size_t>(tid)].add_dist_test(reject);
                                if (reject) continue;

                                const cv::Point3f d = b - a;
                                const float seglen2 = d.dot(d);
                                if (seglen2 <= 1e-6f) continue;
                                const float inv = 1.0f / std::sqrt(seglen2);
                                dirs_unit.emplace_back(d.x * inv, d.y * inv, d.z * inv);

                                // Gaussian falloff: nearer segments contribute more.
                                const double w = std::exp(-static_cast<double>(dist2) * static_cast<double>(inv_two_sigma2));
                                weights.push_back(w);
                            }
                        }
                        const auto t_pp1 = std::chrono::steady_clock::now();
                        t_preproc_s += std::chrono::duration<double>(t_pp1 - t_pp0).count();
                    }
                }

                cv::Point3f n;
                int iters = 0;
                double rms = 0.0;
                double t_solve_s = 0.0;
                const bool ok = fit_normal_ceres(dirs_unit, weights, n, &iters, &rms, &t_solve_s);
                if (ok) {
                    stats[static_cast<size_t>(tid)].add(iters, static_cast<int>(dirs_unit.size()), rms);
                    if (tout != nullptr) {
                        const cv::Point3f a = sample;
                        const cv::Point3f b = sample + normal_vis_scale * n;
                        const int r = static_cast<int>(color_bgr[2]);
                        const int g = static_cast<int>(color_bgr[1]);
                        const int bc = static_cast<int>(color_bgr[0]);

                        const size_t idx0 = tout->vtx_count;
                        const size_t idx1 = tout->vtx_count + 1;
                        tout->vtx << a.x << " " << a.y << " " << a.z << " " << r << " " << g << " " << bc << "\n";
                        tout->vtx << b.x << " " << b.y << " " << b.z << " " << r << " " << g << " " << bc << "\n";
                        tout->edg << idx0 << " " << idx1 << "\n";
                        tout->vtx_count += 2;
                        tout->edg_count += 1;
                    }

                    if (out_zarr_opt.has_value()) {
                        const int ix = x / step;
                        const int iy = y / step;
                        const int iz = z / step;

                        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
                            std::stringstream msg;
                            msg << "Output index out of range while writing normals zarr:"
                                << " ix/iy/iz=(" << ix << "," << iy << "," << iz << ")"
                                << " nx/ny/nz=(" << nx << "," << ny << "," << nz << ")"
                                << " at xyz=(" << x << "," << y << "," << z << ") step=" << step;
                            throw std::runtime_error(msg.str());
                        }

                        const size_t lin = (static_cast<size_t>(iz) * static_cast<size_t>(ny) + static_cast<size_t>(iy)) * static_cast<size_t>(nx) + static_cast<size_t>(ix);
                        if (lin >= enc_x.size()) {
                            std::stringstream msg;
                            msg << "Linear index out of range while writing normals zarr:"
                                << " lin=" << lin << " enc_size=" << enc_x.size()
                                << " ix/iy/iz=(" << ix << "," << iy << "," << iz << ")"
                                << " nx/ny/nz=(" << nx << "," << ny << "," << nz << ")";
                            throw std::runtime_error(msg.str());
                        }
                        enc_x[lin] = encode_dir_component(n.x);
                        enc_y[lin] = encode_dir_component(n.y);
                        enc_z[lin] = encode_dir_component(n.z);
                    }
                }

                const auto t_sample1 = std::chrono::steady_clock::now();
                const double t_total_s = std::chrono::duration<double>(t_sample1 - t_sample0).count();
                const double t_overhead_s = std::max(0.0, t_total_s - t_ng_read_s - t_preproc_s - t_solve_s);
                stats[static_cast<size_t>(tid)].add_timing(t_ng_read_s, t_preproc_s, t_solve_s, t_overhead_s);

                #pragma omp atomic
                processed += 1;
                if (ok) {
                    #pragma omp atomic
                    written += 1;
                }

                // Only one thread reports.
                const auto now = std::chrono::steady_clock::now();
                if (tid == 0 && now - t_last >= std::chrono::seconds(10)) {
                    #pragma omp critical
                    {
                        const auto now2 = std::chrono::steady_clock::now();
                        if (now2 - t_last >= std::chrono::seconds(10)) {
                            report_progress();
                            report_stats();
                            t_last = now2;
                        }
                    }
                }
            }
        }
    }

    // Final report.
    report_progress();
    report_stats();

    if (out_ply_opt.has_value()) {
        const fs::path& out_ply = *out_ply_opt;

        // Close per-thread temp files before merge.
        for (auto& to : t_out) {
            to.vtx.close();
            to.edg.close();
        }

        // Merge into a single PLY (streaming).
        PlyWriter merged(out_ply);
        merged.begin_ascii_streaming();

        // Write vertices by concatenating temp vertex files.
        for (const auto& to : t_out) {
            merged.append_vertex_lines_from_file(to.vtx_path, to.vtx_count);
        }

        // Write edges with per-thread vertex offset.
        size_t vtx_offset = 0;
        for (const auto& to : t_out) {
            merged.append_edge_lines_from_file_with_offset(to.edg_path, vtx_offset, to.edg_count);
            vtx_offset += to.vtx_count;
        }

        merged.end_streaming();

        // Cleanup temp files.
        for (const auto& to : t_out) {
            std::error_code ec;
            fs::remove(to.vtx_path, ec);
            fs::remove(to.edg_path, ec);
        }
    }

    if (out_zarr_opt.has_value()) {
        const fs::path out_zarr = *out_zarr_opt;

        // Create datasets with Zarr metadata using '/' as dimension_separator.
        // NOTE: direction-field readers in vc_grow_seg_from_seed expect:
        //   <root>/{x,y,z}/0/.zarray
        // and will read the delimiter from that .zarray.
        z5::filesystem::handle::File outFile(out_zarr);
        z5::createFile(outFile, true);

        // Ensure groups exist so z5 can infer the zarr format when creating datasets.
        z5::createGroup(outFile, "x");
        z5::createGroup(outFile, "y");
        z5::createGroup(outFile, "z");

        const std::vector<size_t> shape = {static_cast<size_t>(nz), static_cast<size_t>(ny), static_cast<size_t>(nx)}; // ZYX
        const std::vector<size_t> chunks = {std::min<size_t>(64, shape[0]), std::min<size_t>(64, shape[1]), std::min<size_t>(64, shape[2])};
        nlohmann::json compOpts = {{"cname", "zstd"}, {"clevel", 1}, {"shuffle", 0}};

        z5::filesystem::handle::Group gx(outFile, "x");
        z5::filesystem::handle::Group gy(outFile, "y");
        z5::filesystem::handle::Group gz(outFile, "z");

        auto dsx = z5::createDataset(gx, "0", "uint8", shape, chunks, std::string("blosc"), compOpts, /*fillValue=*/0, /*zarrDelimiter=*/"/");
        auto dsy = z5::createDataset(gy, "0", "uint8", shape, chunks, std::string("blosc"), compOpts, /*fillValue=*/0, /*zarrDelimiter=*/"/");
        auto dsz = z5::createDataset(gz, "0", "uint8", shape, chunks, std::string("blosc"), compOpts, /*fillValue=*/0, /*zarrDelimiter=*/"/");

        auto ax = xt::adapt(enc_x, shape);
        auto ay = xt::adapt(enc_y, shape);
        auto az = xt::adapt(enc_z, shape);
        z5::types::ShapeType off = {0, 0, 0};
        z5::multiarray::writeSubarray<uint8_t>(dsx, ax, off.begin());
        z5::multiarray::writeSubarray<uint8_t>(dsy, ay, off.begin());
        z5::multiarray::writeSubarray<uint8_t>(dsz, az, off.begin());

        // Minimal attrs on root.
        nlohmann::json attrs;
        attrs["source"] = "vc_ngrids";
        attrs["note_axes_order"] = "ZYX";
        attrs["encoding"] = "uint8_dir";
        attrs["decode"] = "(v-128)/127";
        attrs["sample_step"] = step;
        attrs["radius"] = radius;
        attrs["crop_min_xyz"] = {crop.min[0], crop.min[1], crop.min[2]};
        attrs["crop_max_xyz"] = {crop.max[0], crop.max[1], crop.max[2]};
        attrs["grid_shape_zyx"] = {shape[0], shape[1], shape[2]};
        z5::filesystem::handle::Group root(outFile, "");
        z5::filesystem::writeAttributes(root, attrs);
    }
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
        ("vis-normals", po::value<std::string>(), "Write fitted normals as PLY line segments")
        ("output-zarr", po::value<std::string>(), "Write fitted normals to a zarr directory (direction-field encoding)");

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

    const bool input_is_normals_zarr = looks_like_normals_zarr_root(input_dir);

    std::optional<CropBox3i> crop;
    if (vm.count("crop")) {
        crop = crop_from_args(vm["crop"].as<std::vector<int>>());
    }

    if (vm.count("vis-ply")) {
        if (input_is_normals_zarr) {
            run_vis_normals_zarr_as_ply(input_dir, fs::path(vm["vis-ply"].as<std::string>()), crop);
        } else {
            run_vis_ply(input_dir, fs::path(vm["vis-ply"].as<std::string>()), crop);
        }
        return 0;
    }

    if (vm.count("fit-normals")) {
        if (input_is_normals_zarr) {
            std::cerr << "Error: --fit-normals is not supported when --input is a normals zarr (use --vis-ply).\n";
            return 1;
        }
        if (!vm.count("vis-normals") && !vm.count("output-zarr")) {
            std::cerr << "Error: --fit-normals requires --vis-normals PATH and/or --output-zarr PATH\n";
            return 1;
        }
        std::optional<fs::path> out_zarr;
        if (vm.count("output-zarr")) {
            out_zarr = fs::path(vm["output-zarr"].as<std::string>());
        }

        std::optional<fs::path> out_ply;
        if (vm.count("vis-normals")) {
            out_ply = fs::path(vm["vis-normals"].as<std::string>());
        }
        run_fit_normals(input_dir, out_ply, crop, out_zarr);
        return 0;
    }

    std::cerr << "Error: no output specified. Use --vis-ply or --fit-normals --vis-normals.\n\n";
    print_usage();
    return 1;
}
