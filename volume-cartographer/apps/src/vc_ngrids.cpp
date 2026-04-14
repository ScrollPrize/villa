#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <random>

#include <unordered_map>
#include <unordered_set>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>
#include <omp.h>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SparseChunkSpool.hpp"
#include "vc/core/util/Zarr.hpp"

#include "vc/core/types/VcDataset.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

using json = nlohmann::json;

static void write_metrics_json(const fs::path& path, const json& metrics) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open metrics json for writing: " + path.string());
    }
    out << metrics.dump(2) << "\n";
    if (!out) {
        throw std::runtime_error("Failed writing metrics json: " + path.string());
    }
}

static void print_usage() {
    std::cout
        << "vc_ngrids: Read a NormalGridVolume and derive/visualize data.\n\n"
        << "Usage: vc_ngrids [options]\n\n"
        << "Options:\n"
        << "  -h, --help              Print this help message\n"
        << "  -i, --input PATH        Input NormalGridVolume directory OR normals zarr root (required)\n"
        << "  -c, --crop x0 y0 z0 x1 y1 z1   Crop bounding box in voxel coords (half-open)\n"
        << "      --surf PATH         Optional tifxyz surface directory (for edge-sampled normal visualization / mesh export)\n"
        << "      --vis-ply PATH      Write visualization as PLY with vertex colors\n"
        << "      --vis-surf PATH     Write --surf tifxyz surface as a quad-mesh PLY (faces)\n"
        << "      --fit-normals       Estimate local 3D normals from segments (within crop)\n"
        << "      --vis-normals PATH  Write fitted normals as PLY line segments\n\n"
        << "      --output-zarr PATH  Write fitted normals to a zarr directory (direction-field encoding)\n\n"
        << "      --metrics-json PATH Write structured run metrics json for fit/align modes\n\n"
        << "      --align-normals     Align normals in an existing normals zarr (requires --output-zarr)\n\n"
        << "Notes:\n"
        << "  - Input can be a directory created by vc_gen_normalgrids (contains metadata.json and xy/xz/yz).\n"
        << "  - Or, input can be a normals zarr root (contains x/0, y/0, z/0 datasets).\n"
        << "  - Crop is optional; if omitted the full extent from metadata/available grids is used.\n";
}

struct CropBox3i {
    cv::Vec3i min; // inclusive
    cv::Vec3i max; // exclusive
};

struct SliceGridCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

struct ThreadSliceGridCache {
    explicit ThreadSliceGridCache(size_t maxEntries = 128) : maxEntries(maxEntries) {}

    std::shared_ptr<const vc::core::util::GridStore> get(
        vc::core::util::NormalGridVolume& ngv, int planeIdx, int sliceIdx)
    {
        const uint64_t key =
            (static_cast<uint64_t>(static_cast<uint32_t>(planeIdx)) << 32) |
            static_cast<uint32_t>(sliceIdx);
        auto it = entries.find(key);
        if (it != entries.end()) {
            ++stats.hits;
            return it->second;
        }

        ++stats.misses;
        auto grid = ngv.get_grid(planeIdx, sliceIdx);
        entries.emplace(key, grid);
        order.push_back(key);
        if (entries.size() > maxEntries && !order.empty()) {
            const uint64_t victim = order.front();
            order.pop_front();
            const auto erased = entries.erase(victim);
            if (erased > 0) {
                ++stats.evictions;
            }
        }
        return grid;
    }

    SliceGridCacheStats stats;
    size_t maxEntries = 128;
    std::unordered_map<uint64_t, std::shared_ptr<const vc::core::util::GridStore>> entries;
    std::deque<uint64_t> order;
};

struct FitOutputMaterializeStats {
    size_t touchedChunks = 0;
    uint64_t records = 0;
    size_t spillFiles = 0;
    double materializeSeconds = 0.0;
    size_t inMemoryBudgetBytes = 0;
};

static FitOutputMaterializeStats materialize_fit_output_spool(
    const fs::path& out_zarr,
    vc::core::util::SparseChunkSpool& spool,
    size_t numThreads)
{
    FitOutputMaterializeStats stats;
    const auto touched = spool.touchedChunks();
    const auto spool_stats = spool.stats();
    stats.touchedChunks = touched.size();
    stats.records = spool_stats.appendedRecords;
    stats.spillFiles = spool_stats.spillFiles;
    stats.inMemoryBudgetBytes = spool_stats.inMemoryBudgetBytes;
    if (touched.empty()) {
        return stats;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto chunkShape = spool.chunkShape();
    const size_t chunkElems = chunkShape[0] * chunkShape[1] * chunkShape[2];
    std::atomic<size_t> nextIndex{0};
    std::exception_ptr firstError;
    std::mutex errorMutex;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (size_t tid = 0; tid < numThreads; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            try {
                vc::VcDataset dsx(out_zarr / "x" / "0");
                vc::VcDataset dsy(out_zarr / "y" / "0");
                vc::VcDataset dsz(out_zarr / "z" / "0");
                vc::VcDataset ds_fit_rms(out_zarr / "fit_rms" / "0");
                vc::VcDataset ds_fit_frac(out_zarr / "fit_frac_short_paths" / "0");
                vc::VcDataset ds_fit_rad(out_zarr / "fit_used_radius" / "0");
                vc::VcDataset ds_fit_sc(out_zarr / "fit_segment_count" / "0");

                std::vector<uint8_t> buf_x(chunkElems, 128);
                std::vector<uint8_t> buf_y(chunkElems, 128);
                std::vector<uint8_t> buf_z(chunkElems, 128);
                std::vector<uint8_t> buf_fit_rms(chunkElems, 0);
                std::vector<uint8_t> buf_fit_frac(chunkElems, 0);
                std::vector<uint8_t> buf_fit_rad(chunkElems, 0);
                std::vector<uint8_t> buf_fit_sc(chunkElems, 0);
                std::vector<vc::core::util::SparseChunkRecordU8x7> records;

                while (true) {
                    const size_t idx = nextIndex.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= touched.size()) {
                        break;
                    }

                    const auto chunk = touched[idx];
                    records.clear();
                    if (!spool.readChunkRecords(chunk, records)) {
                        continue;
                    }

                    std::fill(buf_x.begin(), buf_x.end(), 128);
                    std::fill(buf_y.begin(), buf_y.end(), 128);
                    std::fill(buf_z.begin(), buf_z.end(), 128);
                    std::fill(buf_fit_rms.begin(), buf_fit_rms.end(), 0);
                    std::fill(buf_fit_frac.begin(), buf_fit_frac.end(), 0);
                    std::fill(buf_fit_rad.begin(), buf_fit_rad.end(), 0);
                    std::fill(buf_fit_sc.begin(), buf_fit_sc.end(), 0);

                    for (const auto& record : records) {
                        const size_t lin =
                            (static_cast<size_t>(record.z) * chunkShape[1] + static_cast<size_t>(record.y))
                            * chunkShape[2] + static_cast<size_t>(record.x);
                        buf_x[lin] = record.values[0];
                        buf_y[lin] = record.values[1];
                        buf_z[lin] = record.values[2];
                        buf_fit_rms[lin] = record.values[3];
                        buf_fit_frac[lin] = record.values[4];
                        buf_fit_rad[lin] = record.values[5];
                        buf_fit_sc[lin] = record.values[6];
                    }

                    dsx.writeChunk(chunk.z, chunk.y, chunk.x, buf_x.data(), buf_x.size());
                    dsy.writeChunk(chunk.z, chunk.y, chunk.x, buf_y.data(), buf_y.size());
                    dsz.writeChunk(chunk.z, chunk.y, chunk.x, buf_z.data(), buf_z.size());
                    ds_fit_rms.writeChunk(chunk.z, chunk.y, chunk.x, buf_fit_rms.data(), buf_fit_rms.size());
                    ds_fit_frac.writeChunk(chunk.z, chunk.y, chunk.x, buf_fit_frac.data(), buf_fit_frac.size());
                    ds_fit_rad.writeChunk(chunk.z, chunk.y, chunk.x, buf_fit_rad.data(), buf_fit_rad.size());
                    ds_fit_sc.writeChunk(chunk.z, chunk.y, chunk.x, buf_fit_sc.data(), buf_fit_sc.size());
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!firstError) {
                    firstError = std::current_exception();
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    if (firstError) {
        std::rethrow_exception(firstError);
    }

    stats.materializeSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return stats;
}

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

static void write_quad_surface_as_ply_quads(const ::QuadSurface& surf, const fs::path& out_ply) {
    const cv::Mat_<cv::Vec3f>* pts = surf.rawPointsPtr();
    if (!pts || pts->empty()) {
        throw std::runtime_error("Surface has no points");
    }

    const int rows = pts->rows;
    const int cols = pts->cols;

    // Map valid grid points to contiguous vertex indices.
    std::vector<int> vid(static_cast<size_t>(rows) * static_cast<size_t>(cols), -1);
    std::vector<cv::Vec3f> verts;
    verts.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols) / 2);

    auto lin = [&](int r, int c) -> size_t { return static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c); };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = (*pts)(r, c);
            if (p[0] == -1.f) continue;
            const int idx = static_cast<int>(verts.size());
            vid[lin(r, c)] = idx;
            verts.push_back(p);
        }
    }

    // Count faces (valid quads only).
    size_t face_count = 0;
    for (int r = 0; r + 1 < rows; ++r) {
        for (int c = 0; c + 1 < cols; ++c) {
            const int v00 = vid[lin(r, c)];
            const int v01 = vid[lin(r, c + 1)];
            const int v10 = vid[lin(r + 1, c)];
            const int v11 = vid[lin(r + 1, c + 1)];
            if (v00 < 0 || v01 < 0 || v10 < 0 || v11 < 0) continue;
            ++face_count;
        }
    }

    std::ofstream out(out_ply, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open output PLY for writing: " + out_ply.string());
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment vc_ngrids --vis-surf\n";
    out << "element vertex " << verts.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element face " << face_count << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (const auto& p : verts) {
        out << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    for (int r = 0; r + 1 < rows; ++r) {
        for (int c = 0; c + 1 < cols; ++c) {
            const int v00 = vid[lin(r, c)];
            const int v01 = vid[lin(r, c + 1)];
            const int v10 = vid[lin(r + 1, c)];
            const int v11 = vid[lin(r + 1, c + 1)];
            if (v00 < 0 || v01 < 0 || v10 < 0 || v11 < 0) continue;
            // Quad winding: p00 -> p01 -> p11 -> p10 (consistent grid winding).
            out << "4 " << v00 << " " << v01 << " " << v11 << " " << v10 << "\n";
        }
    }
}

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

// First-order normal field around a sample point x0:
//   n(x) = n0 + J * (x - x0)
// where J is 3x3 (row-major). We only *output* n0 (normalized), but fitting J
// improves the local fit when normals vary spatially (curvature).
struct AffineNormalFitFunctor {
    using Scalar = double;
    enum { NUM_RESIDUALS = Eigen::Dynamic, NUM_PARAMETERS = 12 };

    AffineNormalFitFunctor(const std::array<std::vector<cv::Point3f>, 3>& dirs_unit,
                           const std::array<std::vector<double>, 3>& weights,
                           const std::array<std::vector<cv::Point3f>, 3>& deltas_xyz,
                           const std::array<double, 3>& plane_scales,
                           double unit_norm_w,
                           double jacobian_w)
        : dirs_unit_(dirs_unit),
          weights_(weights),
          deltas_xyz_(deltas_xyz),
          plane_scales_(plane_scales),
          unit_norm_w_(unit_norm_w),
          jacobian_w_(jacobian_w) {
        num_residuals_ = static_cast<int>(dirs_unit_[0].size() + dirs_unit_[1].size() + dirs_unit_[2].size()) + 1 + 9;
    }

    int NumResiduals() const {
        return num_residuals_;
    }

    template <typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        const T* n0 = parameters;
        const T* J = parameters + 3;
        int idx = 0;
        for (int p = 0; p < 3; ++p) {
            const auto& dirs = dirs_unit_[p];
            const auto& deltas = deltas_xyz_[p];
            const auto& wts = weights_[p];
            const double s = plane_scales_[p];
            for (size_t k = 0; k < dirs.size(); ++k) {
                const auto& d = dirs[k];
                const auto& delta = deltas[k];
                const double w_eff = wts[k] * s;
                const double w_sqrt = std::sqrt(std::max(0.0, w_eff));
                const T w = T(w_sqrt);

                const T dx = T(delta.x);
                const T dy = T(delta.y);
                const T dz = T(delta.z);

                const T nx = n0[0] + J[0] * dx + J[1] * dy + J[2] * dz;
                const T ny = n0[1] + J[3] * dx + J[4] * dy + J[5] * dz;
                const T nz = n0[2] + J[6] * dx + J[7] * dy + J[8] * dz;

                residuals[idx++] = w * (nx * T(d.x) + ny * T(d.y) + nz * T(d.z));
            }
        }

        using std::sqrt;
        using ceres::sqrt;
        const T len = sqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2] + T(1e-12));
        residuals[idx++] = T(unit_norm_w_) * (len - T(1.0));

        for (int i = 0; i < 9; ++i) {
            residuals[idx++] = T(jacobian_w_) * J[i];
        }
        return true;
    }

  private:
    const std::array<std::vector<cv::Point3f>, 3>& dirs_unit_;
    const std::array<std::vector<double>, 3>& weights_;
    const std::array<std::vector<cv::Point3f>, 3>& deltas_xyz_;
    const std::array<double, 3> plane_scales_;
    const double unit_norm_w_;
    const double jacobian_w_;
    int num_residuals_ = 0;
};

static bool fit_normal_tiny(
    std::array<std::vector<cv::Point3f>, 3>& dirs_unit_by_plane,
    std::array<std::vector<double>, 3>& weights_by_plane,
    const cv::Point3f& sample_xyz,
    const std::array<std::vector<cv::Point3f>, 3>& deltas_xyz_by_plane,
    cv::Point3f& out_n,
    int* out_num_iterations,
    double* out_rms,
    double* out_solve_seconds,
    double* inout_init_n) {
    (void)sample_xyz;
    for (int p = 0; p < 3; ++p) {
        if (weights_by_plane[p].size() != dirs_unit_by_plane[p].size()) return false;
    }

    // Determine how many samples we will use per plane.
    const size_t n0_count = dirs_unit_by_plane[0].size();
    const size_t n1_count = dirs_unit_by_plane[1].size();
    const size_t n2_count = dirs_unit_by_plane[2].size();
    const size_t total = n0_count + n1_count + n2_count;
    if (total < 3) return false;

    // Re-weight so each plane contributes equally (counts only).
    // scale[p] = (total/3) / n_p (0 if n_p==0)
    const double target = static_cast<double>(total) / 3.0;
    const double s0 = (n0_count > 0) ? (target / static_cast<double>(n0_count)) : 0.0;
    const double s1 = (n1_count > 0) ? (target / static_cast<double>(n1_count)) : 0.0;
    const double s2 = (n2_count > 0) ? (target / static_cast<double>(n2_count)) : 0.0;

    double n0[3];
    if (inout_init_n != nullptr) {
        // Reuse previous solution for warm start.
        n0[0] = inout_init_n[0];
        n0[1] = inout_init_n[1];
        n0[2] = inout_init_n[2];
    } else {
        n0[0] = 1;
        n0[1] = 1;
        n0[2] = 1;
    }

    // For RMS reporting: normalize by the *sum of weights* (not sample count).
    // Note: residual is scaled by sqrt(weight), so cost is ~ 1/2 * sum(weight * dot^2).
    double weight_sum = 0.0;
    for (size_t k = 0; k < n0_count; ++k) {
        const double w_eff = weights_by_plane[0][k] * s0;
        weight_sum += std::max(0.0, w_eff);
    }
    for (size_t k = 0; k < n1_count; ++k) {
        const double w_eff = weights_by_plane[1][k] * s1;
        weight_sum += std::max(0.0, w_eff);
    }
    for (size_t k = 0; k < n2_count; ++k) {
        const double w_eff = weights_by_plane[2][k] * s2;
        weight_sum += std::max(0.0, w_eff);
    }

    const std::array<double, 3> plane_scales = {s0, s1, s2};
    constexpr double kUnitNormWeight = 10.0;
    constexpr double kJacobianWeight = 0.05;
    AffineNormalFitFunctor functor(dirs_unit_by_plane,
                                   weights_by_plane,
                                   deltas_xyz_by_plane,
                                   plane_scales,
                                   kUnitNormWeight,
                                   kJacobianWeight);
    using AutoDiffFn = ceres::TinySolverAutoDiffFunction<AffineNormalFitFunctor, Eigen::Dynamic, 12>;
    AutoDiffFn f(functor);
    ceres::TinySolver<AutoDiffFn> solver;
    solver.options.max_num_iterations = 1000;

    typename ceres::TinySolver<AutoDiffFn>::Parameters params;
    params.setZero();
    params[0] = n0[0];
    params[1] = n0[1];
    params[2] = n0[2];

    const auto solve_t0 = std::chrono::steady_clock::now();
    const auto summary = solver.Solve(f, &params);
    const auto solve_t1 = std::chrono::steady_clock::now();

    if (out_solve_seconds != nullptr) {
        *out_solve_seconds = std::chrono::duration<double>(solve_t1 - solve_t0).count();
    }

    if (out_num_iterations != nullptr) {
        *out_num_iterations = static_cast<int>(summary.iterations);
    }
    if (out_rms != nullptr) {
        // cost = 1/2 * sum(residual^2)
        const double denom = std::max(1e-12, weight_sum);
        *out_rms = std::sqrt(2.0 * summary.final_cost / denom);
    }

    n0[0] = params[0];
    n0[1] = params[1];
    n0[2] = params[2];
    const double len2 = n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2];
    if (!(len2 > 1e-24)) return false;
    const double len = std::sqrt(len2);
    out_n = cv::Point3f(static_cast<float>(n0[0] / len), static_cast<float>(n0[1] / len), static_cast<float>(n0[2] / len));

    if (inout_init_n != nullptr) {
        inout_init_n[0] = out_n.x;
        inout_init_n[1] = out_n.y;
        inout_init_n[2] = out_n.z;
    }
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
        try {
            nlohmann::json attrs = vc::readZarrAttributes(zarr_root);
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

    auto open_u8_zyx = [&](const char* axis) -> std::unique_ptr<vc::VcDataset> {
        return std::make_unique<vc::VcDataset>(zarr_root / axis / "0");
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

    std::vector<uint8_t> ax(Z * Y * X, 0);
    std::vector<uint8_t> ay(Z * Y * X, 0);
    std::vector<uint8_t> az(Z * Y * X, 0);
    {
        std::vector<size_t> off = {0, 0, 0};
        std::vector<size_t> regionShape = {Z, Y, X};
        dsx->readRegion(off, regionShape, ax.data());
        dsy->readRegion(off, regionShape, ay.data());
        dsz->readRegion(off, regionShape, az.data());
    }

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
                const uint8_t ux = ax[iz * Y * X + iy * X + ix];
                const uint8_t uy = ay[iz * Y * X + iy * X + ix];
                const uint8_t uz = az[iz * Y * X + iy * X + ix];
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

static void run_vis_normals_zarr_on_surf_edges_as_ply(
    const fs::path& zarr_root,
    const fs::path& surf_tifxyz,
    const fs::path& out_ply,
    const std::optional<CropBox3i>& crop_opt) {
    // Determine delimiter from x/0/.zarray (fallback ".").
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
        try {
            nlohmann::json attrs = vc::readZarrAttributes(zarr_root);
            if (attrs.contains("grid_origin_xyz") && attrs["grid_origin_xyz"].is_array() && attrs["grid_origin_xyz"].size() == 3) {
                origin_xyz = cv::Vec3i(attrs["grid_origin_xyz"][0].get<int>(), attrs["grid_origin_xyz"][1].get<int>(), attrs["grid_origin_xyz"][2].get<int>());
            }
            if (attrs.contains("sample_step")) {
                step = std::max(1, attrs["sample_step"].get<int>());
            }
        } catch (...) {
            // optional
        }
    }

    auto open_u8_zyx = [&](const char* axis) -> std::unique_ptr<vc::VcDataset> {
        return std::make_unique<vc::VcDataset>(zarr_root / axis / "0");
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

    std::vector<uint8_t> ax(Z * Y * X, 0);
    std::vector<uint8_t> ay(Z * Y * X, 0);
    std::vector<uint8_t> az(Z * Y * X, 0);
    {
        std::vector<size_t> off = {0, 0, 0};
        std::vector<size_t> regionShape = {Z, Y, X};
        dsx->readRegion(off, regionShape, ax.data());
        dsy->readRegion(off, regionShape, ay.data());
        dsz->readRegion(off, regionShape, az.data());
    }

    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Vec3i(std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    auto decode = [&](uint8_t u) -> float { return (static_cast<int>(u) - 128) / 127.0f; };
    auto is_fill = [&](size_t iz, size_t iy, size_t ix) -> bool {
        return ax[iz * Y * X + iy * X + ix] == 128 && ay[iz * Y * X + iy * X + ix] == 128 && az[iz * Y * X + iy * X + ix] == 128;
    };

    auto sample_trilinear = [&](const cv::Point3f& p_xyz, cv::Point3f& out_n_xyz) -> bool {
        // Convert voxel coordinates to grid coordinates.
        const double gx = (static_cast<double>(p_xyz.x) - origin_xyz[0]) / static_cast<double>(step);
        const double gy = (static_cast<double>(p_xyz.y) - origin_xyz[1]) / static_cast<double>(step);
        const double gz = (static_cast<double>(p_xyz.z) - origin_xyz[2]) / static_cast<double>(step);

        int x0 = static_cast<int>(std::floor(gx));
        int y0 = static_cast<int>(std::floor(gy));
        int z0 = static_cast<int>(std::floor(gz));
        x0 = std::clamp(x0, 0, static_cast<int>(X) - 2);
        y0 = std::clamp(y0, 0, static_cast<int>(Y) - 2);
        z0 = std::clamp(z0, 0, static_cast<int>(Z) - 2);
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if (is_fill(static_cast<size_t>(z0), static_cast<size_t>(y0), static_cast<size_t>(x0)) ||
            is_fill(static_cast<size_t>(z1), static_cast<size_t>(y0), static_cast<size_t>(x0)) ||
            is_fill(static_cast<size_t>(z0), static_cast<size_t>(y1), static_cast<size_t>(x0)) ||
            is_fill(static_cast<size_t>(z1), static_cast<size_t>(y1), static_cast<size_t>(x0)) ||
            is_fill(static_cast<size_t>(z0), static_cast<size_t>(y0), static_cast<size_t>(x1)) ||
            is_fill(static_cast<size_t>(z1), static_cast<size_t>(y0), static_cast<size_t>(x1)) ||
            is_fill(static_cast<size_t>(z0), static_cast<size_t>(y1), static_cast<size_t>(x1)) ||
            is_fill(static_cast<size_t>(z1), static_cast<size_t>(y1), static_cast<size_t>(x1))) {
            return false;
        }

        double tx = gx - x0;
        double ty = gy - y0;
        double tz = gz - z0;
        tx = std::clamp(tx, 0.0, 1.0);
        ty = std::clamp(ty, 0.0, 1.0);
        tz = std::clamp(tz, 0.0, 1.0);

        auto lerp = [](double a, double b, double t) { return (1.0 - t) * a + t * b; };
        auto tri = [&](double c000, double c100, double c010, double c110,
                       double c001, double c101, double c011, double c111) -> double {
            const double c00 = lerp(c000, c001, tx);
            const double c01 = lerp(c010, c011, tx);
            const double c10 = lerp(c100, c101, tx);
            const double c11 = lerp(c110, c111, tx);
            const double c0 = lerp(c00, c01, ty);
            const double c1 = lerp(c10, c11, ty);
            return lerp(c0, c1, tz);
        };

        auto c = [&](const std::vector<uint8_t>& a, int zz, int yy, int xx) -> double {
            return static_cast<double>(decode(a[static_cast<size_t>(zz) * Y * X + static_cast<size_t>(yy) * X + static_cast<size_t>(xx)]));
        };

        const double nx = tri(
            c(ax, z0, y0, x0), c(ax, z1, y0, x0), c(ax, z0, y1, x0), c(ax, z1, y1, x0),
            c(ax, z0, y0, x1), c(ax, z1, y0, x1), c(ax, z0, y1, x1), c(ax, z1, y1, x1));
        const double ny = tri(
            c(ay, z0, y0, x0), c(ay, z1, y0, x0), c(ay, z0, y1, x0), c(ay, z1, y1, x0),
            c(ay, z0, y0, x1), c(ay, z1, y0, x1), c(ay, z0, y1, x1), c(ay, z1, y1, x1));
        const double nz = tri(
            c(az, z0, y0, x0), c(az, z1, y0, x0), c(az, z0, y1, x0), c(az, z1, y1, x0),
            c(az, z0, y0, x1), c(az, z1, y0, x1), c(az, z0, y1, x1), c(az, z1, y1, x1));

        const double nlen2 = nx * nx + ny * ny + nz * nz;
        if (!(nlen2 > 1e-24)) return false;
        const double nlen = std::sqrt(nlen2);
        out_n_xyz = cv::Point3f(static_cast<float>(nx / nlen), static_cast<float>(ny / nlen), static_cast<float>(nz / nlen));
        return true;
    };

    // Load surface.
    auto surf = load_quad_from_tifxyz(surf_tifxyz.string());
    if (!surf) {
        throw std::runtime_error("Failed to load tifxyz surface: " + surf_tifxyz.string());
    }
    const cv::Mat_<cv::Vec3f>* pts = surf->rawPointsPtr();
    if (!pts || pts->empty()) {
        throw std::runtime_error("Surface has no points: " + surf_tifxyz.string());
    }

    const cv::Vec3b color_bgr(0, 255, 255); // yellow
    const float vis_scale = static_cast<float>(step) * 0.5f;

    PlyWriter ply(out_ply);
    ply.begin_ascii_streaming();

    auto in_crop = [&](const cv::Point3f& p) -> bool {
        const int x = static_cast<int>(std::floor(p.x));
        const int y = static_cast<int>(std::floor(p.y));
        const int z = static_cast<int>(std::floor(p.z));
        return x >= crop.min[0] && x < crop.max[0] && y >= crop.min[1] && y < crop.max[1] && z >= crop.min[2] && z < crop.max[2];
    };

    const int rows = pts->rows;
    const int cols = pts->cols;

    auto emit_edge_normal = [&](const cv::Vec3f& a, const cv::Vec3f& b) {
        if (a[0] == -1.f || b[0] == -1.f) return;
        const cv::Point3f mid = 0.5f * cv::Point3f(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
        if (!in_crop(mid)) return;
        cv::Point3f n;
        if (!sample_trilinear(mid, n)) return;
        const cv::Point3f p0 = mid;
        const cv::Point3f p1 = mid + vis_scale * n;
        ply.write_segment_streaming(p0, p1, color_bgr);
    };

    // Emit normals sampled at edge midpoints along the quad grid lines.
    // Horizontal edges (row, col) -> (row, col+1)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c + 1 < cols; ++c) {
            emit_edge_normal((*pts)(r, c), (*pts)(r, c + 1));
        }
    }
    // Vertical edges (row, col) -> (row+1, col)
    for (int r = 0; r + 1 < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            emit_edge_normal((*pts)(r, c), (*pts)(r + 1, c));
        }
    }

    ply.end_streaming();
}

static inline uint8_t flip_u8_dir_component(uint8_t u) {
    // decode(u)=(u-128)/127. Flipping the decoded value maps approximately to (256-u).
    const int v = 256 - static_cast<int>(u);
    return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

struct CropIndexBox3z {
    // ZYX indices (half-open) in the zarr grid.
    std::vector<size_t> off = {0, 0, 0};
    std::vector<size_t> shape = {0, 0, 0};
};

static CropIndexBox3z crop_to_zarr_zyx(
    const CropBox3i& crop_xyz,
    const cv::Vec3i& origin_xyz,
    int step,
    const std::vector<size_t>& full_shape_zyx) {
    // Convert voxel crop box to zarr indices (ZYX) using floor/ceil so we cover the requested voxel region.
    const auto floor_div = [](int a, int b) -> int {
        // b>0
        if (a >= 0) return a / b;
        return -(((-a + b - 1) / b));
    };
    const auto ceil_div = [&](int a, int b) -> int {
        // ceil(a/b)
        if (a >= 0) return (a + b - 1) / b;
        return -((-a) / b);
    };

    const int ix0 = floor_div(crop_xyz.min[0] - origin_xyz[0], step);
    const int iy0 = floor_div(crop_xyz.min[1] - origin_xyz[1], step);
    const int iz0 = floor_div(crop_xyz.min[2] - origin_xyz[2], step);
    const int ix1 = ceil_div(crop_xyz.max[0] - origin_xyz[0], step);
    const int iy1 = ceil_div(crop_xyz.max[1] - origin_xyz[1], step);
    const int iz1 = ceil_div(crop_xyz.max[2] - origin_xyz[2], step);

    const int X = static_cast<int>(full_shape_zyx.at(2));
    const int Y = static_cast<int>(full_shape_zyx.at(1));
    const int Z = static_cast<int>(full_shape_zyx.at(0));

    const int cx0 = std::max(0, std::min(X, ix0));
    const int cy0 = std::max(0, std::min(Y, iy0));
    const int cz0 = std::max(0, std::min(Z, iz0));
    const int cx1 = std::max(0, std::min(X, ix1));
    const int cy1 = std::max(0, std::min(Y, iy1));
    const int cz1 = std::max(0, std::min(Z, iz1));

    CropIndexBox3z out;
    out.off = {static_cast<size_t>(cz0), static_cast<size_t>(cy0), static_cast<size_t>(cx0)};
    out.shape = {static_cast<size_t>(std::max(0, cz1 - cz0)),
                 static_cast<size_t>(std::max(0, cy1 - cy0)),
                 static_cast<size_t>(std::max(0, cx1 - cx0))};
    return out;
}

static double dot_decoded_u8(uint8_t ax, uint8_t ay, uint8_t az, uint8_t bx, uint8_t by, uint8_t bz) {
    const auto d = [](uint8_t u) -> double { return (static_cast<int>(u) - 128) / 127.0; };
    return d(ax) * d(bx) + d(ay) * d(by) + d(az) * d(bz);
}

static bool is_valid_normal_u8(uint8_t ux, uint8_t uy, uint8_t uz) {
    // In our direction-field encoding, (128,128,128) is the neutral fill value meaning "no normal".
    return !(ux == 128 && uy == 128 && uz == 128);
}

static std::optional<size_t> memAvailableBytes()
{
#ifdef __linux__
    std::ifstream in("/proc/meminfo");
    if (!in) {
        return std::nullopt;
    }

    std::string key;
    size_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value * 1024ull;
        }
    }
#endif
    return std::nullopt;
}

struct NormalsZarrLayout {
    std::vector<size_t> fullShapeZyx;
    std::vector<size_t> chunkShapeZyx;
    cv::Vec3i originXyz = cv::Vec3i(0, 0, 0);
    int sampleStep = 1;
};

struct NormalsCropU8 {
    NormalsZarrLayout layout;
    CropIndexBox3z cropZyx;
    std::vector<uint8_t> x;
    std::vector<uint8_t> y;
    std::vector<uint8_t> z;
};

static std::unique_ptr<vc::VcDataset> open_normals_axis_dataset(
    const fs::path& zarr_root,
    const char* axis)
{
    return std::make_unique<vc::VcDataset>(zarr_root / axis / "0");
}

static void assert_normals_fill_value_128(const fs::path& zarr_root, const char* axis)
{
    const fs::path zarray_path = zarr_root / axis / "0" / ".zarray";
    if (!fs::exists(zarray_path)) {
        throw std::runtime_error(
            std::string("Missing ") + axis + "/0/.zarray under normals zarr root: " +
            zarr_root.string());
    }

    nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_path));
    if (!j.contains("fill_value")) {
        throw std::runtime_error(
            std::string("Missing fill_value in ") + axis + "/0/.zarray under normals zarr root: " +
            zarr_root.string());
    }

    const int fv = j["fill_value"].get<int>();
    if (fv != 128) {
        std::stringstream msg;
        msg << "Normals zarr has unexpected fill_value=" << fv
            << " for " << axis << "/0; expected 128";
        throw std::runtime_error(msg.str());
    }
}

static NormalsZarrLayout read_normals_zarr_layout(const fs::path& zarr_root)
{
    auto dsx = open_normals_axis_dataset(zarr_root, "x");
    auto dsy = open_normals_axis_dataset(zarr_root, "y");
    auto dsz = open_normals_axis_dataset(zarr_root, "z");
    if (!dsx || !dsy || !dsz) {
        throw std::runtime_error("Failed to open x/y/z datasets under zarr root: " + zarr_root.string());
    }
    if (dsx->shape() != dsy->shape() || dsx->shape() != dsz->shape()) {
        throw std::runtime_error("x/y/z datasets have different shapes under: " + zarr_root.string());
    }
    if (dsx->shape().size() != 3) {
        throw std::runtime_error("Expected 3D datasets (ZYX) for normals zarr under: " + zarr_root.string());
    }

    assert_normals_fill_value_128(zarr_root, "x");
    assert_normals_fill_value_128(zarr_root, "y");
    assert_normals_fill_value_128(zarr_root, "z");

    NormalsZarrLayout layout;
    layout.fullShapeZyx = {dsx->shape()[0], dsx->shape()[1], dsx->shape()[2]};
    layout.chunkShapeZyx = {
        dsx->defaultChunkShape()[0],
        dsx->defaultChunkShape()[1],
        dsx->defaultChunkShape()[2],
    };

    try {
        nlohmann::json attrs = vc::readZarrAttributes(zarr_root);
        if (attrs.contains("grid_origin_xyz") &&
            attrs["grid_origin_xyz"].is_array() &&
            attrs["grid_origin_xyz"].size() == 3) {
            layout.originXyz = cv::Vec3i(
                attrs["grid_origin_xyz"][0].get<int>(),
                attrs["grid_origin_xyz"][1].get<int>(),
                attrs["grid_origin_xyz"][2].get<int>());
        }
        if (attrs.contains("sample_step")) {
            layout.sampleStep = std::max(1, attrs["sample_step"].get<int>());
        }
    } catch (...) {
        // Attributes are optional; keep defaults.
    }

    return layout;
}

static NormalsCropU8 load_normals_crop_u8(
    const fs::path& zarr_root,
    const std::optional<CropBox3i>& crop_opt,
    double* read_seconds = nullptr)
{
    const auto read_start = std::chrono::steady_clock::now();
    NormalsCropU8 crop;
    crop.layout = read_normals_zarr_layout(zarr_root);

    const CropBox3i crop_xyz = crop_opt.value_or(CropBox3i{
        cv::Vec3i(std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    crop.cropZyx = crop_to_zarr_zyx(
        crop_xyz,
        crop.layout.originXyz,
        crop.layout.sampleStep,
        crop.layout.fullShapeZyx);
    const size_t CZ = crop.cropZyx.shape[0];
    const size_t CY = crop.cropZyx.shape[1];
    const size_t CX = crop.cropZyx.shape[2];
    if (CZ == 0 || CY == 0 || CX == 0) {
        throw std::runtime_error("--crop maps to empty region in normals zarr index space");
    }

    auto dsx = open_normals_axis_dataset(zarr_root, "x");
    auto dsy = open_normals_axis_dataset(zarr_root, "y");
    auto dsz = open_normals_axis_dataset(zarr_root, "z");

    crop.x.assign(CZ * CY * CX, 0);
    crop.y.assign(CZ * CY * CX, 0);
    crop.z.assign(CZ * CY * CX, 0);

    const std::vector<size_t> off(crop.cropZyx.off.begin(), crop.cropZyx.off.end());
    const std::vector<size_t> region_shape(crop.cropZyx.shape.begin(), crop.cropZyx.shape.end());
    dsx->readRegion(off, region_shape, crop.x.data());
    dsy->readRegion(off, region_shape, crop.y.data());
    dsz->readRegion(off, region_shape, crop.z.data());

    if (read_seconds != nullptr) {
        *read_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - read_start).count();
    }
    return crop;
}

static void create_normals_output_zarr(
    const fs::path& out_zarr,
    const NormalsZarrLayout& layout,
    const std::optional<CropIndexBox3z>& crop_zyx = std::nullopt)
{
    std::filesystem::create_directories(out_zarr);
    vc::createZarrDataset(out_zarr / "x", "0", layout.fullShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);
    vc::createZarrDataset(out_zarr / "y", "0", layout.fullShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);
    vc::createZarrDataset(out_zarr / "z", "0", layout.fullShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);

    nlohmann::json attrs;
    attrs["source"] = "vc_ngrids";
    attrs["description"] = "Direction-field encoded normals: x/0,y/0,z/0 uint8 with decode (u8-128)/127";
    attrs["encoding"] = {
        {"type", "direction-field-u8"},
        {"decode", "(u8-128)/127"},
        {"fill_value", 128},
    };
    attrs["grid_origin_xyz"] = {
        layout.originXyz[0],
        layout.originXyz[1],
        layout.originXyz[2],
    };
    attrs["sample_step"] = layout.sampleStep;
    attrs["grid_shape_zyx"] = {
        layout.fullShapeZyx[0],
        layout.fullShapeZyx[1],
        layout.fullShapeZyx[2],
    };
    if (crop_zyx.has_value()) {
        attrs["crop_off_zyx"] = {
            crop_zyx->off[0],
            crop_zyx->off[1],
            crop_zyx->off[2],
        };
        attrs["crop_shape_zyx"] = {
            crop_zyx->shape[0],
            crop_zyx->shape[1],
            crop_zyx->shape[2],
        };
    }
    vc::writeZarrAttributes(out_zarr, attrs);
}

static void write_normals_crop_u8(
    const fs::path& out_zarr,
    const CropIndexBox3z& crop_zyx,
    const std::vector<uint8_t>& ax,
    const std::vector<uint8_t>& ay,
    const std::vector<uint8_t>& az)
{
    auto out_dsx = open_normals_axis_dataset(out_zarr, "x");
    auto out_dsy = open_normals_axis_dataset(out_zarr, "y");
    auto out_dsz = open_normals_axis_dataset(out_zarr, "z");

    const std::vector<size_t> crop_off(crop_zyx.off.begin(), crop_zyx.off.end());
    const std::vector<size_t> crop_shape(crop_zyx.shape.begin(), crop_zyx.shape.end());
    out_dsx->writeRegion(crop_off, crop_shape, ax.data());
    out_dsy->writeRegion(crop_off, crop_shape, ay.data());
    out_dsz->writeRegion(crop_off, crop_shape, az.data());
}

static void run_align_normals_zarr(
    const fs::path& zarr_root,
    const fs::path& out_zarr,
    const std::optional<CropBox3i>& crop_opt,
    int seed_samples = 100,
    int radius = 2,
    int candidate_samples_per_iter = 100,
    const std::optional<fs::path>& metrics_json_path = std::nullopt) {
    const auto total_start = std::chrono::steady_clock::now();
    json metrics;
    metrics["mode"] = "align-normals";
    metrics["input"] = zarr_root.string();
    metrics["output"] = out_zarr.string();
    metrics["omp_threads"] = std::max(1, omp_get_max_threads());
    metrics["seed_samples"] = seed_samples;
    metrics["radius"] = radius;
    metrics["candidate_samples_per_iter"] = candidate_samples_per_iter;

    double read_seconds = 0.0;
    NormalsCropU8 normals = load_normals_crop_u8(zarr_root, crop_opt, &read_seconds);
    metrics["read_seconds"] = read_seconds;
    const auto& crop_zyx = normals.cropZyx;
    const auto& full_shape = normals.layout.fullShapeZyx;
    const cv::Vec3i origin_xyz = normals.layout.originXyz;
    const int step = normals.layout.sampleStep;
    const size_t CZ = crop_zyx.shape[0];
    const size_t CY = crop_zyx.shape[1];
    const size_t CX = crop_zyx.shape[2];
    auto& ax = normals.x;
    auto& ay = normals.y;
    auto& az = normals.z;
    const CropBox3i crop_xyz = crop_opt.value_or(CropBox3i{
        cv::Vec3i(std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4,
                  std::numeric_limits<int>::min() / 4),
        cv::Vec3i(std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4,
                  std::numeric_limits<int>::max() / 4),
    });

    const size_t N = CZ * CY * CX;
    auto lin_of = [&](size_t iz, size_t iy, size_t ix) -> size_t {
        return (iz * CY + iy) * CX + ix;
    };
    auto idx_of_lin = [&](size_t lin, size_t& iz, size_t& iy, size_t& ix) {
        ix = lin % CX;
        const size_t t = lin / CX;
        iy = t % CY;
        iz = t / CY;
    };

    // State bits:
    // bit0: aligned
    // bit1: flipped relative to original
    // bit2: in fringe (internal)
    constexpr uint8_t kAligned = 1u << 0;
    constexpr uint8_t kFlip = 1u << 1;
    constexpr uint8_t kInFringe = 1u << 2;
    std::vector<uint8_t> state(N, 0);
    std::vector<size_t> fringe;
    fringe.reserve(1024);

    auto neighbor_iter = [&](size_t iz, size_t iy, size_t ix, int rad, const auto& fn) {
        const int z0 = std::max(0, static_cast<int>(iz) - rad);
        const int y0 = std::max(0, static_cast<int>(iy) - rad);
        const int x0 = std::max(0, static_cast<int>(ix) - rad);
        const int z1 = std::min(static_cast<int>(CZ) - 1, static_cast<int>(iz) + rad);
        const int y1 = std::min(static_cast<int>(CY) - 1, static_cast<int>(iy) + rad);
        const int x1 = std::min(static_cast<int>(CX) - 1, static_cast<int>(ix) + rad);
        for (int zz = z0; zz <= z1; ++zz) {
            for (int yy = y0; yy <= y1; ++yy) {
                for (int xx = x0; xx <= x1; ++xx) {
                    fn(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx));
                }
            }
        }
    };

    auto divergence_parallel_invariant = [&](size_t lin, int rad, int& out_count) -> double {
        size_t iz, iy, ix;
        idx_of_lin(lin, iz, iy, ix);
        const uint8_t ux = ax[lin_of(iz, iy, ix)];
        const uint8_t uy = ay[lin_of(iz, iy, ix)];
        const uint8_t uz = az[lin_of(iz, iy, ix)];
        if (!is_valid_normal_u8(ux, uy, uz)) {
            out_count = 0;
            return 1e9;
        }
        double acc = 0.0;
        int cnt = 0;
        neighbor_iter(iz, iy, ix, rad, [&](size_t zz, size_t yy, size_t xx) {
            const uint8_t vx = ax[lin_of(zz, yy, xx)];
            const uint8_t vy = ay[lin_of(zz, yy, xx)];
            const uint8_t vz = az[lin_of(zz, yy, xx)];
            if (!is_valid_normal_u8(vx, vy, vz)) return;
            const double d = dot_decoded_u8(ux, uy, uz, vx, vy, vz);
            acc += (1.0 - std::abs(d));
            ++cnt;
        });
        out_count = cnt;
        if (cnt <= 1) return 1e9;
        return acc / static_cast<double>(cnt);
    };

    // Collect valid normals in the crop.
    std::vector<size_t> valid_lin;
    valid_lin.reserve(N / 4);
    for (size_t iz = 0; iz < CZ; ++iz) {
        for (size_t iy = 0; iy < CY; ++iy) {
            for (size_t ix = 0; ix < CX; ++ix) {
                if (is_valid_normal_u8(ax[lin_of(iz, iy, ix)], ay[lin_of(iz, iy, ix)], az[lin_of(iz, iy, ix)])) {
                    valid_lin.push_back(lin_of(iz, iy, ix));
                }
            }
        }
    }
    if (valid_lin.empty()) {
        throw std::runtime_error("No valid normals found in selected crop region");
    }

    // Seed selection: pick the locally smoothest (parallel-invariant) from random samples.
    std::mt19937_64 rng(0xA11F10);
    std::uniform_int_distribution<size_t> pick_valid(0, valid_lin.size() - 1);
    size_t seed_lin = valid_lin[pick_valid(rng)];
    double best_div = 1e9;
    for (int s = 0; s < std::max(1, seed_samples); ++s) {
        const size_t lin = valid_lin[pick_valid(rng)];
        int cnt = 0;
        const double div = divergence_parallel_invariant(lin, radius, cnt);
        if (cnt < 8) continue;
        if (div < best_div) {
            best_div = div;
            seed_lin = lin;
        }
    }

    // Seed: mark aligned; orientation arbitrary (flip=0).
    state[seed_lin] |= kAligned;

    auto oriented_u8_at = [&](size_t lin, uint8_t& ox, uint8_t& oy, uint8_t& oz) {
        size_t iz, iy, ix;
        idx_of_lin(lin, iz, iy, ix);
        ox = ax[lin_of(iz, iy, ix)];
        oy = ay[lin_of(iz, iy, ix)];
        oz = az[lin_of(iz, iy, ix)];
        if (state[lin] & kFlip) {
            ox = flip_u8_dir_component(ox);
            oy = flip_u8_dir_component(oy);
            oz = flip_u8_dir_component(oz);
        }
    };

    auto add_fringe_neighbors = [&](size_t lin) {
        size_t iz, iy, ix;
        idx_of_lin(lin, iz, iy, ix);

        // Parallelize neighbor checks, but keep fringe insertions serial (global state).
        // Use the same threshold as candidate scoring.
        const size_t fringe_size = fringe.size();
        std::vector<size_t> to_add;
        to_add.reserve(26);

        #pragma omp parallel if(fringe_size >= 10000)
        {
            std::vector<size_t> local;
            local.reserve(26);
            #pragma omp for collapse(3) schedule(dynamic,1) nowait
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int zz = static_cast<int>(iz) + dz;
                        const int yy = static_cast<int>(iy) + dy;
                        const int xx = static_cast<int>(ix) + dx;
                        if (zz < 0 || yy < 0 || xx < 0 || zz >= static_cast<int>(CZ) || yy >= static_cast<int>(CY) || xx >= static_cast<int>(CX)) continue;
                        const size_t nlin = lin_of(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx));
                        if (state[nlin] & kAligned) continue;
                        if (!is_valid_normal_u8(ax[lin_of(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx))],
                                                ay[lin_of(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx))],
                                                az[lin_of(static_cast<size_t>(zz), static_cast<size_t>(yy), static_cast<size_t>(xx))])) {
                            continue;
                        }
                        if (state[nlin] & kInFringe) continue;
                        local.push_back(nlin);
                    }
                }
            }

            #pragma omp critical
            {
                to_add.insert(to_add.end(), local.begin(), local.end());
            }
        }

        for (const size_t nlin : to_add) {
            if (state[nlin] & kAligned) continue;
            if (state[nlin] & kInFringe) continue;
            state[nlin] |= kInFringe;
            fringe.push_back(nlin);
        }
    };

    auto score_candidate = [&](size_t cand_lin, int neigh_rad, int& aligned_neighbors, bool& out_flip) -> double {
        size_t iz, iy, ix;
        idx_of_lin(cand_lin, iz, iy, ix);
        const uint8_t cx = ax[lin_of(iz, iy, ix)];
        const uint8_t cy = ay[lin_of(iz, iy, ix)];
        const uint8_t cz = az[lin_of(iz, iy, ix)];
        if (!is_valid_normal_u8(cx, cy, cz)) {
            aligned_neighbors = 0;
            out_flip = false;
            return -1e9;
        }

        // We intentionally do NOT normalize by neighbor count.
        // This biases the selection toward candidates with more already-aligned neighbors.
        double sum_dot_no = 0.0;
        double sum_dot_fl = 0.0;
        int cnt = 0;
        neighbor_iter(iz, iy, ix, neigh_rad, [&](size_t zz, size_t yy, size_t xx) {
            const size_t nlin = lin_of(zz, yy, xx);
            if (!(state[nlin] & kAligned)) return;
            uint8_t nx, ny, nz;
            oriented_u8_at(nlin, nx, ny, nz);
            const double d = dot_decoded_u8(cx, cy, cz, nx, ny, nz);
            sum_dot_no += d;
            sum_dot_fl += -d;
            ++cnt;
        });
        aligned_neighbors = cnt;
        if (cnt == 0) {
            out_flip = false;
            return -1e9;
        }
        if (sum_dot_fl > sum_dot_no) {
            out_flip = true;
            return sum_dot_fl;
        }
        out_flip = false;
        return sum_dot_no;
    };

    add_fringe_neighbors(seed_lin);
    size_t aligned_count = 1;
    const auto align_start = std::chrono::steady_clock::now();

    // One RNG per OpenMP thread for the lifetime of this function.
    // Used for candidate sampling inside the OpenMP candidate loop.
    const int omp_threads = std::max(1, omp_get_max_threads());
    std::vector<std::mt19937_64> omp_rng(static_cast<size_t>(omp_threads));
    for (int t = 0; t < omp_threads; ++t) {
        // Deterministic but distinct streams.
        omp_rng[static_cast<size_t>(t)] = std::mt19937_64(0xA11F10ull + 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(t + 1));
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto t_last = t0;
    while (aligned_count < valid_lin.size() && !fringe.empty()) {
        const auto nowp = std::chrono::steady_clock::now();
        if (nowp - t_last >= std::chrono::seconds(1)) {
            const double elapsed = std::chrono::duration<double>(nowp - t0).count();
            const double rate = (elapsed > 1e-9) ? (static_cast<double>(aligned_count) / elapsed) : 0.0;
            const double rem = static_cast<double>((valid_lin.size() > aligned_count) ? (valid_lin.size() - aligned_count) : 0);
            const double eta = (rate > 1e-9) ? (rem / rate) : 0.0;
            std::cerr << "align-normals: aligned " << aligned_count << "/" << valid_lin.size()
                      << " | fringe=" << fringe.size()
                      << " | elapsed=" << elapsed << "s"
                      << " | rate=" << rate << " vox/s"
                      << " | ETA=" << eta << "s\n";
            t_last = nowp;
        }

        const size_t fringe_size = fringe.size();
        std::uniform_int_distribution<size_t> pick_fr(0, fringe_size - 1);
        const int tries = std::min<int>(candidate_samples_per_iter, static_cast<int>(fringe_size));

        // Candidate selection + scoring: parallelize, but keep state mutations serial.
        struct BestCand {
            size_t idx_in_fringe = 0;
            size_t cand_lin = 0;
            double score = -1e9;
            int cnt = -1;
            bool flip = false;
        };

        BestCand best;
        best.idx_in_fringe = pick_fr(rng);
        best.cand_lin = fringe[best.idx_in_fringe];

        #pragma omp parallel if(fringe_size >= 10000)
        {
            const int tid = omp_get_thread_num();
            BestCand local;
            local.score = -1e9;
            local.cnt = -1;
            std::uniform_int_distribution<size_t> pick(0, fringe_size - 1);

            // OpenMP loop runs over #cands (tries) with dynamic scheduling.
            #pragma omp for schedule(dynamic,1) nowait
            for (int t = 0; t < tries; ++t) {
                (void)t;
                const size_t cand_idx = pick(omp_rng[static_cast<size_t>(tid)]);
                const size_t cand = fringe[cand_idx];
                if (state[cand] & kAligned) continue;
                int cnt = 0;
                bool fl = false;
                const double s = score_candidate(cand, /*neigh_rad=*/radius, cnt, fl);
                if (cnt <= 0) continue;
                if (s > local.score || (s == local.score && cnt > local.cnt)) {
                    local.score = s;
                    local.cnt = cnt;
                    local.cand_lin = cand;
                    local.idx_in_fringe = cand_idx;
                    local.flip = fl;
                }
            }

            #pragma omp critical
            {
                if (local.cnt > 0 && (local.score > best.score || (local.score == best.score && local.cnt > best.cnt))) {
                    best = local;
                }
            }
        }

        const size_t best_idx_in_fringe = best.idx_in_fringe;
        const size_t best_cand = best.cand_lin;
        const double best_score = best.score;
        const int best_cnt = best.cnt;
        const bool best_flip = best.flip;

        if (best_cnt <= 0) {
            // Drop one stale fringe entry.
            const size_t drop = pick_fr(rng);
            state[fringe[drop]] &= ~kInFringe;
            fringe[drop] = fringe.back();
            fringe.pop_back();
            continue;
        }

        state[best_cand] &= ~kInFringe;
        state[best_cand] |= kAligned;
        if (best_flip) state[best_cand] |= kFlip;
        ++aligned_count;

        // Remove best_cand from fringe (swap-erase by index; avoids linear scan).
        // If this fails, it indicates a bug (e.g. duplicates or stale index), so fail hard.
        if (!(best_idx_in_fringe < fringe.size() && fringe[best_idx_in_fringe] == best_cand)) {
            std::stringstream msg;
            msg << "Invariant violated: best_idx_in_fringe does not point to best_cand (idx=" << best_idx_in_fringe
                << ", best_cand=" << best_cand << ", fringe_size=" << fringe.size() << ")";
            throw std::runtime_error(msg.str());
        }
        fringe[best_idx_in_fringe] = fringe.back();
        fringe.pop_back();

        add_fringe_neighbors(best_cand);
    }

    // Apply flips to the crop arrays.
    for (size_t lin = 0; lin < N; ++lin) {
        if (!(state[lin] & kAligned)) continue;
        if (!(state[lin] & kFlip)) continue;
        size_t iz, iy, ix;
        idx_of_lin(lin, iz, iy, ix);
        ax[lin_of(iz, iy, ix)] = flip_u8_dir_component(ax[lin_of(iz, iy, ix)]);
        ay[lin_of(iz, iy, ix)] = flip_u8_dir_component(ay[lin_of(iz, iy, ix)]);
        az[lin_of(iz, iy, ix)] = flip_u8_dir_component(az[lin_of(iz, iy, ix)]);
    }

    metrics["align_seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - align_start).count();
    const auto write_start = std::chrono::steady_clock::now();
    create_normals_output_zarr(out_zarr, normals.layout, crop_zyx);
    write_normals_crop_u8(out_zarr, crop_zyx, ax, ay, az);

    nlohmann::json attrs = vc::readZarrAttributes(out_zarr);
    attrs["align_normals"] = true;
    attrs["align_seed_samples"] = seed_samples;
    attrs["align_radius_step_units"] = radius;
    attrs["align_candidate_samples_per_iter"] = candidate_samples_per_iter;
    attrs["crop_min_xyz"] = {crop_xyz.min[0], crop_xyz.min[1], crop_xyz.min[2]};
    attrs["crop_max_xyz"] = {crop_xyz.max[0], crop_xyz.max[1], crop_xyz.max[2]};
    vc::writeZarrAttributes(out_zarr, attrs);

    metrics["write_seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - write_start).count();
    metrics["total_seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    metrics["full_shape_zyx"] = {full_shape[0], full_shape[1], full_shape[2]};
    metrics["crop_off_zyx"] = {crop_zyx.off[0], crop_zyx.off[1], crop_zyx.off[2]};
    metrics["crop_shape_zyx"] = {crop_zyx.shape[0], crop_zyx.shape[1], crop_zyx.shape[2]};
    metrics["valid_normals"] = valid_lin.size();
    metrics["aligned_normals"] = aligned_count;
    if (metrics_json_path.has_value()) {
        write_metrics_json(*metrics_json_path, metrics);
    }
}

struct FitPlaneConfig {
    int plane_idx = 0;
    const char* dir = "";
    int slice_axis = 0;
};

struct FitPlaneSliceEntry {
    int slice = 0;
    std::shared_ptr<const vc::core::util::GridStore> grid;
};

struct FitPlaneSliceTable {
    FitPlaneConfig cfg;
    std::vector<FitPlaneSliceEntry> entries;
};

struct FitGridAccessPlan {
    bool preload = false;
    size_t estimatedBytes = 0;
    size_t budgetBytes = 0;
    size_t availableBytes = 0;
    double preloadSeconds = 0.0;
    std::array<FitPlaneSliceTable, 3> planes;
};

static fs::path normal_grid_slice_path(
    const fs::path& input_dir,
    const char* plane_dir,
    int slice)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "%06d.grid", slice);
    return input_dir / plane_dir / filename;
}

static FitGridAccessPlan build_fit_grid_access_plan(
    const fs::path& input_dir,
    vc::core::util::NormalGridVolume& ngv,
    const CropBox3i& read_box,
    int sparse_volume)
{
    const FitPlaneConfig plane_cfgs[3] = {
        {0, "xy", 2},
        {1, "xz", 1},
        {2, "yz", 0},
    };

    FitGridAccessPlan plan;
    const auto available = memAvailableBytes();
    plan.availableBytes = available.value_or(0);
    if (available.has_value()) {
        plan.budgetBytes = std::min<size_t>(
            256ull << 20,
            std::max<size_t>(64ull << 20, *available / 16));
    } else {
        plan.budgetBytes = 128ull << 20;
    }

    for (size_t plane = 0; plane < std::size(plane_cfgs); ++plane) {
        plan.planes[plane].cfg = plane_cfgs[plane];
        const auto& cfg = plan.planes[plane].cfg;
        const int axis = cfg.slice_axis;
        const int s_min = read_box.min[axis];
        const int s_max = read_box.max[axis];
        int slice_start = align_down(s_min, sparse_volume);
        if (slice_start < s_min) {
            slice_start += std::max(1, sparse_volume);
        }

        for (int slice = slice_start; slice < s_max; slice += std::max(1, sparse_volume)) {
            const fs::path grid_path = normal_grid_slice_path(input_dir, cfg.dir, slice);
            if (!fs::exists(grid_path)) {
                continue;
            }

            plan.planes[plane].entries.push_back(FitPlaneSliceEntry{slice, nullptr});
            std::error_code ec;
            plan.estimatedBytes += fs::file_size(grid_path, ec);
        }
    }

    plan.preload = plan.estimatedBytes > 0 && plan.estimatedBytes <= plan.budgetBytes;
    if (!plan.preload) {
        return plan;
    }

    const auto preload_start = std::chrono::steady_clock::now();
    for (auto& plane : plan.planes) {
        for (auto& entry : plane.entries) {
            entry.grid = ngv.get_grid(plane.cfg.plane_idx, entry.slice);
        }
    }
    plan.preloadSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - preload_start).count();
    return plan;
}

static void run_fit_normals(
    const fs::path& input_dir,
    const std::optional<fs::path>& out_ply_opt,
    const std::optional<CropBox3i>& crop_opt,
    const std::optional<fs::path>& out_zarr_opt,
    int step = 16,
    float radius = 128.f,
    bool dbg_tif = false,
    const std::optional<fs::path>& metrics_json_path = std::nullopt) {
    const auto total_start = std::chrono::steady_clock::now();
    json metrics;
    metrics["mode"] = "fit-normals";
    metrics["input"] = input_dir.string();
    if (out_ply_opt.has_value()) metrics["output_ply"] = out_ply_opt->string();
    if (out_zarr_opt.has_value()) metrics["output_zarr"] = out_zarr_opt->string();
    metrics["step"] = step;
    metrics["radius"] = radius;
    metrics["omp_threads"] = std::max(1, omp_get_max_threads());

    vc::core::util::NormalGridVolume ngv(input_dir.string());
    ngv.resetCacheStats();
    const int sparse_volume = ngv.metadata().value("sparse-volume", 1);

    // We only *output* samples within crop, but we must *read* enough context
    // (potentially larger than crop) so fits near the crop boundary have full support.
    const auto vol_xyz_opt = infer_volume_shape_from_grids(input_dir);
    if (!vol_xyz_opt.has_value()) {
        throw std::runtime_error("Failed to infer volume shape from normal grids (required to expand fit read region beyond crop)");
    }
    const cv::Vec3i vol_xyz = *vol_xyz_opt;
    const CropBox3i crop = crop_opt.value_or(CropBox3i{
        cv::Vec3i(0, 0, 0),
        vol_xyz,
    });

    // Adaptive radius (per-sample): start at 32, double to 512, and choose the first
    // radius for which >=min_samples are found in any 2 of the 3 planes.
    constexpr int kMinRadius = 32;
    constexpr int kMaxRadius = 512;
    constexpr int kMinSamplesPerPlane = 256;

    // Read support must cover the *largest* possible radius, even though we only output inside crop.
    const int rad_i = kMaxRadius;
    CropBox3i read_box;
    read_box.min = cv::Vec3i(
        std::max(0, crop.min[0] - rad_i),
        std::max(0, crop.min[1] - rad_i),
        std::max(0, crop.min[2] - rad_i));
    read_box.max = cv::Vec3i(
        std::min(vol_xyz[0], crop.max[0] + rad_i),
        std::min(vol_xyz[1], crop.max[1] + rad_i),
        std::min(vol_xyz[2], crop.max[2] + rad_i));

    FitGridAccessPlan grid_access_plan = build_fit_grid_access_plan(
        input_dir,
        ngv,
        read_box,
        sparse_volume);
    metrics["grid_access"] = {
        {"mode", grid_access_plan.preload ? "preload" : "stream"},
        {"estimated_bytes", static_cast<long long>(grid_access_plan.estimatedBytes)},
        {"budget_bytes", static_cast<long long>(grid_access_plan.budgetBytes)},
        {"available_bytes", static_cast<long long>(grid_access_plan.availableBytes)},
        {"preload_seconds", grid_access_plan.preloadSeconds},
        {"xy_slices", static_cast<long long>(grid_access_plan.planes[0].entries.size())},
        {"xz_slices", static_cast<long long>(grid_access_plan.planes[1].entries.size())},
        {"yz_slices", static_cast<long long>(grid_access_plan.planes[2].entries.size())},
    };

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

    // For --fit-normals + --vis-normals: color by fit RMS.
    // RMS=0 => blue, RMS>=0.5 => red, linear blend in between.
    auto rms_to_color_bgr = [&](double rms) -> cv::Vec3b {
        const double t = std::clamp(rms / 0.5, 0.0, 1.0);
        const int b = static_cast<int>(std::lround((1.0 - t) * 255.0));
        const int g = 0;
        const int r = static_cast<int>(std::lround(t * 255.0));
        return cv::Vec3b(static_cast<uint8_t>(b), static_cast<uint8_t>(g), static_cast<uint8_t>(r));
    };

    auto has_enough_plane_support = [&](const std::array<std::vector<cv::Point3f>, 3>& dirs_unit) -> bool {
        int ok = 0;
        for (int p = 0; p < 3; ++p) {
            if (static_cast<int>(dirs_unit[p].size()) >= kMinSamplesPerPlane) ++ok;
        }
        return ok >= 2;
    };

    auto clear_fit_buffers = [&](std::array<std::vector<cv::Point3f>, 3>& dirs_unit,
                                 std::array<std::vector<double>, 3>& weights,
                                 std::array<std::vector<cv::Point3f>, 3>& deltas_xyz) {
        for (int p = 0; p < 3; ++p) {
            dirs_unit[p].clear();
            weights[p].clear();
            deltas_xyz[p].clear();
        }
    };

    // gather_samples_for_radius is defined after FitStats/stats so it can record dist-test stats.
    const float normal_vis_scale = static_cast<float>(step) * 0.5f;

    const int sx0 = align_down(crop.min[0], step);
    const int sy0 = align_down(crop.min[1], step);
    const int sz0 = align_down(crop.min[2], step);

    // When writing zarr, keep global lattice coordinates but only materialize
    // the crop-sized subregion to the output datasets.
    int full_nx = 0, full_ny = 0, full_nz = 0;
    if (out_zarr_opt.has_value()) {
        if (crop.max[0] > vol_xyz[0] || crop.max[1] > vol_xyz[1] || crop.max[2] > vol_xyz[2]) {
            std::stringstream msg;
            msg << "Crop max exceeds inferred volume shape: crop.max=(" << crop.max[0] << "," << crop.max[1] << "," << crop.max[2]
                << ") vs inferred vol_xyz=(" << vol_xyz[0] << "," << vol_xyz[1] << "," << vol_xyz[2] << ")";
            throw std::runtime_error(msg.str());
        }
        full_nx = (vol_xyz[0] + step - 1) / step;
        full_ny = (vol_xyz[1] + step - 1) / step;
        full_nz = (vol_xyz[2] + step - 1) / step;
    }

    // Progress reporting should reflect *work done*, i.e. samples evaluated within the crop,
    // not the size of the full output lattice.
    const int crop_nx = (crop.max[0] - sx0 + step - 1) / step;
    const int crop_ny = (crop.max[1] - sy0 + step - 1) / step;
    const int crop_nz = (crop.max[2] - sz0 + step - 1) / step;
    const int64_t total_samples = static_cast<int64_t>(crop_nx) * static_cast<int64_t>(crop_ny) * static_cast<int64_t>(crop_nz);
    const int crop_off_x = sx0 / step;
    const int crop_off_y = sy0 / step;
    const int crop_off_z = sz0 / step;
    metrics["crop_min_xyz"] = {crop.min[0], crop.min[1], crop.min[2]};
    metrics["crop_max_xyz"] = {crop.max[0], crop.max[1], crop.max[2]};
    metrics["read_box_min_xyz"] = {read_box.min[0], read_box.min[1], read_box.min[2]};
    metrics["read_box_max_xyz"] = {read_box.max[0], read_box.max[1], read_box.max[2]};
    metrics["crop_shape_samples_zyx"] = {crop_nz, crop_ny, crop_nx};
    metrics["crop_off_samples_zyx"] = {crop_off_z, crop_off_y, crop_off_x};
    metrics["total_samples"] = total_samples;

    std::vector<size_t> output_shape_zyx;
    std::vector<size_t> output_chunks_zyx;
    fs::path spool_dir;
    std::unique_ptr<vc::core::util::SparseChunkSpool> output_spool;
    std::vector<std::unique_ptr<vc::core::util::SparseChunkSpoolBuffer>> output_spool_buffers;
    if (out_zarr_opt.has_value()) {
        output_shape_zyx = {
            static_cast<size_t>(full_nz),
            static_cast<size_t>(full_ny),
            static_cast<size_t>(full_nx)};
        output_chunks_zyx = {
            std::min<size_t>(64, output_shape_zyx[0]),
            std::min<size_t>(64, output_shape_zyx[1]),
            std::min<size_t>(64, output_shape_zyx[2])};
        spool_dir = *out_zarr_opt / ".vc_ngrids_fit_spool";
        std::error_code ec;
        fs::remove_all(spool_dir, ec);
        output_spool = std::make_unique<vc::core::util::SparseChunkSpool>(
            spool_dir,
            vc::core::util::Shape3{output_chunks_zyx[0], output_chunks_zyx[1], output_chunks_zyx[2]},
            vc::core::util::Shape3{output_shape_zyx[0], output_shape_zyx[1], output_shape_zyx[2]},
            128ull * 1024ull * 1024ull);
        output_spool_buffers.reserve(static_cast<size_t>(nthreads));
        for (int t = 0; t < nthreads; ++t) {
            output_spool_buffers.push_back(
                std::make_unique<vc::core::util::SparseChunkSpoolBuffer>(*output_spool));
        }
    }

    // Stats: iterations-to-solved histogram and RMS histogram.
    // Note: we keep a coarse RMS bucket histogram for printing and a fine histogram for median estimation.
    constexpr int kFineRmsBins = 1000;
    struct FitStats {
        // Iteration buckets: [0-4],[5-9],[10-19],[20-49],[50-99],[100-199],[200+]
        uint64_t iters_buckets[7] = {0, 0, 0, 0, 0, 0, 0};

        // Sample-count buckets (#segments used): [0-127],[128-255],[256-511],[512-1023],[1024-2047],[2048-4095],[4096+]
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
        // Time spent inside the solver call itself (as reported by fit_normal_tiny).
        double t_solve_s = 0.0;
        // Total time spent in the fit_normal_tiny() call (includes setup, residual creation, etc.).
        double t_solve_call_s = 0.0;
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

            const int sb = (samples < 128) ? 0 : (samples < 256) ? 1 : (samples < 512) ? 2 : (samples < 1024) ? 3 : (samples < 2048) ? 4 : (samples < 4096) ? 5 : 6;
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

        void add_timing(double ng_read_s, double preproc_s, double solve_s, double solve_call_s, double overhead_s) {
            ++samples_total;
            t_ng_read_s += ng_read_s;
            t_preproc_s += preproc_s;
            t_solve_s += solve_s;
            t_solve_call_s += solve_call_s;
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
    const size_t thread_grid_cache_capacity = std::max<size_t>(
        128,
        std::min<size_t>(
            1024,
            std::max({
                grid_access_plan.planes[0].entries.size(),
                grid_access_plan.planes[1].entries.size(),
                grid_access_plan.planes[2].entries.size(),
            })));
    std::vector<ThreadSliceGridCache> thread_grid_cache;
    if (!grid_access_plan.preload) {
        thread_grid_cache.reserve(static_cast<size_t>(std::max(1, omp_get_max_threads())));
        for (int t = 0; t < std::max(1, omp_get_max_threads()); ++t) {
            thread_grid_cache.emplace_back(thread_grid_cache_capacity);
        }
    }

    auto gather_samples_for_radius = [&](
        const cv::Point3f& sample,
        float rad,
        int tid,
        std::array<std::vector<cv::Point3f>, 3>& dirs_unit,
        std::array<std::vector<double>, 3>& weights,
        std::array<std::vector<cv::Point3f>, 3>& deltas_xyz,
        vc::core::util::GridStore::QueryScratch& grid_query_scratch,
        int& used_segments_total,
        int& used_segments_short_paths,
        double& t_ng_read_s,
        double& t_preproc_s) {

        // Only use paths that are long enough to be meaningful.
        constexpr int kMinSegmentsPerPath = 1;
        // "Short path" threshold used for diagnostics (not for filtering; see dbg + zarr outputs).
        // A path with a single segment (2 points) is considered "short".
        constexpr int kShortPathMaxSegments = 1;

        const float r2 = rad * rad;
        const float sigma = rad / 2.0f;
        const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma + 1e-12f);
        (void)inv_two_sigma2;
        const float sample_arr[3] = {sample.x, sample.y, sample.z};

        for (const auto& plane : grid_access_plan.planes) {
            const auto& pc = plane.cfg;
            // plane axes
            const int u_axis = (pc.plane_idx == 2) ? 1 : 0;
            const int v_axis = (pc.plane_idx == 0) ? 1 : 2;
            const int s_axis = (pc.plane_idx == 0) ? 2 : (pc.plane_idx == 1) ? 1 : 0;

            const int s_center = static_cast<int>(sample_arr[s_axis]);
            const int s_min = std::max(read_box.min[s_axis], static_cast<int>(std::floor(s_center - rad)));
            const int s_max = std::min(read_box.max[s_axis], static_cast<int>(std::ceil(s_center + rad)) + 1);
            auto slice_it = std::lower_bound(
                plane.entries.begin(),
                plane.entries.end(),
                s_min,
                [](const FitPlaneSliceEntry& entry, int slice_value) {
                    return entry.slice < slice_value;
                });

            for (; slice_it != plane.entries.end() && slice_it->slice < s_max; ++slice_it) {
                const int slice = slice_it->slice;
                // Shrink the 2D query rect based on distance in the slice axis:
                // at offset ds from the center, the in-slice radius is sqrt(r^2 - ds^2).
                const float ds = std::abs(static_cast<float>(slice) - sample_arr[s_axis]);
                if (ds > rad) continue;
                const float r_eff = std::sqrt(std::max(0.0f, rad * rad - ds * ds));

                const int u0 = std::max(read_box.min[u_axis], static_cast<int>(std::floor(sample_arr[u_axis] - r_eff)));
                const int v0 = std::max(read_box.min[v_axis], static_cast<int>(std::floor(sample_arr[v_axis] - r_eff)));
                const int u1 = std::min(read_box.max[u_axis], static_cast<int>(std::ceil(sample_arr[u_axis] + r_eff)) + 1);
                const int v1 = std::min(read_box.max[v_axis], static_cast<int>(std::ceil(sample_arr[v_axis] + r_eff)) + 1);
                if (u1 <= u0 || v1 <= v0) continue;

                const cv::Rect query(u0, v0, u1 - u0, v1 - v0);

                const auto t_read0 = std::chrono::steady_clock::now();
                std::shared_ptr<const vc::core::util::GridStore> grid;
                if (grid_access_plan.preload) {
                    grid = slice_it->grid;
                } else {
                    grid = thread_grid_cache[static_cast<size_t>(tid)].get(ngv, pc.plane_idx, slice);
                }
                if (!grid) continue;
                grid->forEach(
                    query,
                    grid_query_scratch,
                    [&](const std::shared_ptr<std::vector<cv::Point>>&) {});
                const auto t_read1 = std::chrono::steady_clock::now();
                t_ng_read_s += std::chrono::duration<double>(t_read1 - t_read0).count();

                const auto t_pp0 = std::chrono::steady_clock::now();
                for (const auto& path_ptr : grid_query_scratch.results) {
                    if (!path_ptr || path_ptr->size() < 2) continue;

                    const int seg_count = static_cast<int>(path_ptr->size()) - 1;
                    if (seg_count < kMinSegmentsPerPath) continue;
                    const bool is_short_path = (seg_count <= kShortPathMaxSegments);

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

                        if (!clip_segment_to_crop(a, b, read_box)) continue;
                        const float dist2 = dist_sq_point_segment_appx(sample, a, b);
                        const bool reject = (dist2 > r2);
                        stats[static_cast<size_t>(tid)].add_dist_test(reject);
                        if (reject) continue;

                        const cv::Point3f d = b - a;
                        const float seglen2 = d.dot(d);
                        if (seglen2 <= 1e-6f) continue;
                        const float inv = 1.0f / std::sqrt(seglen2);
                        dirs_unit[pc.plane_idx].emplace_back(d.x * inv, d.y * inv, d.z * inv);

                        // Delta from sample point for first-order curvature term.
                        // Use the segment midpoint as the constraint location.
                        const cv::Point3f m = 0.5f * (a + b);
                        deltas_xyz[pc.plane_idx].emplace_back(m.x - sample.x, m.y - sample.y, m.z - sample.z);

                    // Weights are proportional to path length:
                    // seg_count=1 -> 0.1, seg_count>=10 -> 1.0.
                    const double path_w = std::max(0.1, std::min(1.0, static_cast<double>(seg_count) / 10.0));
                    const double w = path_w; // switchable: path_w * std::exp(-static_cast<double>(dist2) * static_cast<double>(inv_two_sigma2))
                    weights[pc.plane_idx].push_back(w);

                        ++used_segments_total;
                        if (is_short_path) ++used_segments_short_paths;
                    }
                }
                const auto t_pp1 = std::chrono::steady_clock::now();
                t_preproc_s += std::chrono::duration<double>(t_pp1 - t_pp0).count();
            }
        }
    };

    // Per-thread warm start for the solver.
    struct WarmStart {
        double n[3] = {0.0, 0.0, 0.0};
        bool has = false;
    };
    std::vector<WarmStart> warm(static_cast<size_t>(std::max(1, omp_get_max_threads())));

    struct FitBuffers {
        std::array<std::vector<cv::Point3f>, 3> dirs_unit;
        std::array<std::vector<double>, 3> weights;
        std::array<std::vector<cv::Point3f>, 3> deltas_xyz;
        vc::core::util::GridStore::QueryScratch gridQueryScratch;
    };
    std::vector<FitBuffers> fit_buffers(static_cast<size_t>(std::max(1, omp_get_max_threads())));
    for (auto& fb : fit_buffers) {
        for (int p = 0; p < 3; ++p) {
            fb.dirs_unit[p].reserve(512);
            fb.weights[p].reserve(512);
            fb.deltas_xyz[p].reserve(512);
        }
    }

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
        acc.t_solve_call_s += s.t_solve_call_s;
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

    auto encode_u8_from_range = [](double v, double lo, double hi) -> uint8_t {
        if (!(hi > lo)) return 0;
        if (!std::isfinite(v)) return 0;
        const double t = (v - lo) / (hi - lo);
        const double tc = std::max(0.0, std::min(1.0, t));
        const int q = static_cast<int>(std::lround(tc * 255.0));
        return static_cast<uint8_t>(std::max(0, std::min(255, q)));
    };
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
        const double work = acc.t_ng_read_s + acc.t_preproc_s + acc.t_solve_call_s + acc.t_overhead_s;
        const double png = (work > 1e-12) ? (100.0 * acc.t_ng_read_s / work) : 0.0;
        const double ppp = (work > 1e-12) ? (100.0 * acc.t_preproc_s / work) : 0.0;
        const double ps = (work > 1e-12) ? (100.0 * acc.t_solve_call_s / work) : 0.0;
        const double po = (work > 1e-12) ? (100.0 * acc.t_overhead_s / work) : 0.0;
        std::cerr << "fit-normals stats: ok=" << acc.ok_count
                  << " | rms(avg/med/max)=" << avg << "/" << med << "/" << acc.rms_max << "\n";
        std::cerr << "  iters buckets [0-4,5-9,10-19,20-49,50-99,100-199,200+]: "
                  << acc.iters_buckets[0] << "," << acc.iters_buckets[1] << "," << acc.iters_buckets[2] << "," << acc.iters_buckets[3]
                  << "," << acc.iters_buckets[4] << "," << acc.iters_buckets[5] << "," << acc.iters_buckets[6] << "\n";
        std::cerr << "  samples buckets [0-127,128-255,256-511,512-1023,1024-2047,2048-4095,4096+]: "
                  << acc.samples_buckets[0] << "," << acc.samples_buckets[1] << "," << acc.samples_buckets[2] << "," << acc.samples_buckets[3]
                  << "," << acc.samples_buckets[4] << "," << acc.samples_buckets[5] << "," << acc.samples_buckets[6] << "\n";
        std::cerr << "  rms buckets [<0.01,<0.02,<0.05,<0.1,<0.2,<0.5,>=0.5]: "
                  << acc.rms_buckets[0] << "," << acc.rms_buckets[1] << "," << acc.rms_buckets[2] << "," << acc.rms_buckets[3]
                  << "," << acc.rms_buckets[4] << "," << acc.rms_buckets[5] << "," << acc.rms_buckets[6] << "\n";

        // Time breakdown is thread-summed (can exceed wall time with OpenMP).
        const double solve_call_overhead_s = std::max(0.0, acc.t_solve_call_s - acc.t_solve_s);
        std::cerr << "  time(thread-summed): samples=" << acc.samples_total
                  << " | ng_read=" << acc.t_ng_read_s << "s (" << png << "%)"
                  << " | preproc=" << acc.t_preproc_s << "s (" << ppp << "%)"
                  << " | solve_call=" << acc.t_solve_call_s << "s (" << ps << "%); tiny_solve=" << acc.t_solve_s << "s; solve_call_overhead=" << solve_call_overhead_s << "s"
                  << " | overhead=" << acc.t_overhead_s << "s (" << po << "%)\n";

        const double rej = (acc.dist_test_total > 0) ? (100.0 * static_cast<double>(acc.dist_test_reject) / static_cast<double>(acc.dist_test_total)) : 0.0;
        std::cerr << "  dist2>r2 rejects: " << acc.dist_test_reject << "/" << acc.dist_test_total << " (" << rej << "%)\n";
    };

    // Optional debug outputs (first z-layer only, output-sample grid size).
    const int dbg_z = sz0;
    cv::Mat1f dbg_rms;
    cv::Mat1f dbg_used_radius;
    cv::Mat1f dbg_nsamp_sum;
    cv::Mat1f dbg_nsamp_xy;
    cv::Mat1f dbg_nsamp_xz;
    cv::Mat1f dbg_nsamp_yz;
    cv::Mat1f dbg_frac_used_short_paths;
    if (dbg_tif) {
        const float nanf = std::numeric_limits<float>::quiet_NaN();
        dbg_rms = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_used_radius = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_nsamp_sum = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_nsamp_xy = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_nsamp_xz = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_nsamp_yz = cv::Mat1f(crop_ny, crop_nx, nanf);
        dbg_frac_used_short_paths = cv::Mat1f(crop_ny, crop_nx, nanf);
    }

    // Only compute normals inside crop, but write them into the full lattice when out_zarr is enabled.
    #pragma omp parallel for collapse(2) schedule(dynamic,1)
    for (int z = sz0; z < crop.max[2]; z += step) {
        for (int y = sy0; y < crop.max[1]; y += step) {
            for (int x = sx0; x < crop.max[0]; x += step) {
                const int tid = omp_get_thread_num();
                ThreadOut* tout = nullptr;
                if (out_ply_opt.has_value()) {
                    tout = &t_out[static_cast<size_t>(tid)];
                }

                const cv::Point3f sample(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));

                const auto t_sample0 = std::chrono::steady_clock::now();

                FitBuffers& fb = fit_buffers[static_cast<size_t>(tid)];
                auto& dirs_unit = fb.dirs_unit;
                auto& weights = fb.weights;
                auto& deltas_xyz = fb.deltas_xyz;

                double t_ng_read_s = 0.0;
                double t_preproc_s = 0.0;

                int used_rad = kMaxRadius;
                int used_segments_total = 0;
                int used_segments_short_paths = 0;
                bool skip_fit_due_to_low_segments = false;
                for (int rad = kMinRadius; rad <= kMaxRadius; rad *= 1.1) {
                    clear_fit_buffers(dirs_unit, weights, deltas_xyz);
                    used_segments_total = 0;
                    used_segments_short_paths = 0;
                    gather_samples_for_radius(sample,
                                              static_cast<float>(rad),
                                              tid,
                                              dirs_unit,
                                              weights,
                                              deltas_xyz,
                                              fb.gridQueryScratch,
                                              used_segments_total,
                                              used_segments_short_paths,
                                              t_ng_read_s,
                                              t_preproc_s);
                    used_rad = rad;

                    // If at the initial radius there are too few segments overall, skip this normal completely.
                    // Treat as a failed fit (same behavior as fit_normal_tiny returning false).
                    if (rad == kMinRadius && used_segments_total < 50) {
                        skip_fit_due_to_low_segments = true;
                        break;
                    }

                    if (has_enough_plane_support(dirs_unit)) break;
                }

                cv::Point3f n;
                int iters = 0;
                double rms = 0.0;
                double t_solve_s = 0.0;
                double t_solve_call_s = 0.0;
                WarmStart& ws = warm[static_cast<size_t>(tid)];
                double* init_ptr = ws.has ? ws.n : nullptr;
                const auto t_solve_call0 = std::chrono::steady_clock::now();
                const bool ok = (!skip_fit_due_to_low_segments) &&
                                fit_normal_tiny(dirs_unit, weights, sample, deltas_xyz, n, &iters, &rms, &t_solve_s, init_ptr);
                const auto t_solve_call1 = std::chrono::steady_clock::now();
                t_solve_call_s = std::chrono::duration<double>(t_solve_call1 - t_solve_call0).count();

                // Debug images: first z-layer only.
                if (dbg_tif && z == dbg_z) {
                    const int col = (x - sx0) / step;
                    const int row = (y - sy0) / step;
                    if (row >= 0 && col >= 0 && row < crop_ny && col < crop_nx) {
                        const int n_xy = static_cast<int>(dirs_unit[0].size());
                        const int n_xz = static_cast<int>(dirs_unit[1].size());
                        const int n_yz = static_cast<int>(dirs_unit[2].size());
                        const int n_sum = n_xy + n_xz + n_yz;
                        dbg_used_radius(row, col) = static_cast<float>(used_rad);
                        dbg_nsamp_xy(row, col) = static_cast<float>(n_xy);
                        dbg_nsamp_xz(row, col) = static_cast<float>(n_xz);
                        dbg_nsamp_yz(row, col) = static_cast<float>(n_yz);
                        dbg_nsamp_sum(row, col) = static_cast<float>(n_sum);
                        dbg_rms(row, col) = ok ? static_cast<float>(rms) : 1.0f;

                        if (used_segments_total > 0) {
                            dbg_frac_used_short_paths(row, col) = static_cast<float>(
                                static_cast<double>(used_segments_short_paths) / static_cast<double>(used_segments_total));
                        }
                    }
                }

                    if (ok) {
                        ws.n[0] = n.x;
                        ws.n[1] = n.y;
                        ws.n[2] = n.z;
                        ws.has = true;
                        const int nsamp = static_cast<int>(dirs_unit[0].size() + dirs_unit[1].size() + dirs_unit[2].size());
                        stats[static_cast<size_t>(tid)].add(iters, nsamp, rms);
                        if (tout != nullptr) {
                            const cv::Vec3b color_bgr = rms_to_color_bgr(rms);
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
                        const int ix = (x - sx0) / step;
                        const int iy = (y - sy0) / step;
                        const int iz = (z - sz0) / step;

                        if (ix < 0 || iy < 0 || iz < 0 || ix >= crop_nx || iy >= crop_ny || iz >= crop_nz) {
                            std::stringstream msg;
                            msg << "Crop-local output index out of range while writing normals zarr:"
                                << " ix/iy/iz=(" << ix << "," << iy << "," << iz << ")"
                                << " crop_nx/ny/nz=(" << crop_nx << "," << crop_ny << "," << crop_nz << ")"
                                << " at xyz=(" << x << "," << y << "," << z << ") step=" << step;
                            throw std::runtime_error(msg.str());
                        }

                        const double frac_short = (used_segments_total > 0)
                            ? (static_cast<double>(used_segments_short_paths) / static_cast<double>(used_segments_total))
                            : 0.0;
                        output_spool_buffers[static_cast<size_t>(tid)]->emit(
                            static_cast<size_t>(crop_off_z + iz),
                            static_cast<size_t>(crop_off_y + iy),
                            static_cast<size_t>(crop_off_x + ix),
                            std::array<uint8_t, 7>{
                                encode_dir_component(n.x),
                                encode_dir_component(n.y),
                                encode_dir_component(n.z),
                                encode_u8_from_range(rms, 0.0, 1.0),
                                encode_u8_from_range(frac_short, 0.0, 1.0),
                                encode_u8_from_range(static_cast<double>(used_rad), 2.0, 512.0),
                                encode_u8_from_range(static_cast<double>(used_segments_total), 0.0, 8192.0),
                            });
                    }
                }

                const auto t_sample1 = std::chrono::steady_clock::now();
                const double t_total_s = std::chrono::duration<double>(t_sample1 - t_sample0).count();
                const double t_overhead_s = std::max(0.0, t_total_s - t_ng_read_s - t_preproc_s - t_solve_call_s);
                stats[static_cast<size_t>(tid)].add_timing(t_ng_read_s, t_preproc_s, t_solve_s, t_solve_call_s, t_overhead_s);

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

    SliceGridCacheStats slice_grid_cache_stats;
    for (const auto& cache : thread_grid_cache) {
        slice_grid_cache_stats.hits += cache.stats.hits;
        slice_grid_cache_stats.misses += cache.stats.misses;
        slice_grid_cache_stats.evictions += cache.stats.evictions;
    }
    const auto ngv_cache_stats = ngv.cacheStats();

    if (dbg_tif) {
        // Write float TIFFs into the current working directory.
        // NOTE: OpenCV will write 32-bit float TIFFs when passed CV_32F mats.
        cv::imwrite((fs::path("dbg_fit_rms.tif")).string(), dbg_rms);
        cv::imwrite((fs::path("dbg_fit_used_radius.tif")).string(), dbg_used_radius);
        cv::imwrite((fs::path("dbg_fit_sample_count_sum.tif")).string(), dbg_nsamp_sum);
        cv::imwrite((fs::path("dbg_fit_sample_count_xy.tif")).string(), dbg_nsamp_xy);
        cv::imwrite((fs::path("dbg_fit_sample_count_xz.tif")).string(), dbg_nsamp_xz);
        cv::imwrite((fs::path("dbg_fit_sample_count_yz.tif")).string(), dbg_nsamp_yz);
        cv::imwrite((fs::path("dbg_fit_frac_used_short_paths.tif")).string(), dbg_frac_used_short_paths);
    }

    if (output_spool) {
        for (auto& spool_buffer : output_spool_buffers) {
            spool_buffer->flushAll();
        }
    }

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

    FitOutputMaterializeStats materialize_stats;
    if (out_zarr_opt.has_value()) {
        const fs::path out_zarr = *out_zarr_opt;

        NormalsZarrLayout output_layout;
        output_layout.fullShapeZyx = output_shape_zyx;
        output_layout.chunkShapeZyx = output_chunks_zyx;
        output_layout.sampleStep = step;
        create_normals_output_zarr(
            out_zarr,
            output_layout,
            CropIndexBox3z{
                {static_cast<size_t>(crop_off_z), static_cast<size_t>(crop_off_y), static_cast<size_t>(crop_off_x)},
                {static_cast<size_t>(crop_nz), static_cast<size_t>(crop_ny), static_cast<size_t>(crop_nx)},
            });
        vc::createZarrDataset(out_zarr / "fit_rms", "0", output_shape_zyx, output_chunks_zyx, vc::VcDtype::uint8, "blosc", "/");
        vc::createZarrDataset(out_zarr / "fit_frac_short_paths", "0", output_shape_zyx, output_chunks_zyx, vc::VcDtype::uint8, "blosc", "/");
        vc::createZarrDataset(out_zarr / "fit_used_radius", "0", output_shape_zyx, output_chunks_zyx, vc::VcDtype::uint8, "blosc", "/");
        vc::createZarrDataset(out_zarr / "fit_segment_count", "0", output_shape_zyx, output_chunks_zyx, vc::VcDtype::uint8, "blosc", "/");

        materialize_stats = materialize_fit_output_spool(out_zarr, *output_spool, static_cast<size_t>(nthreads));

        nlohmann::json attrs = vc::readZarrAttributes(out_zarr);
        attrs["radius"] = radius;
        attrs["crop_min_xyz"] = {crop.min[0], crop.min[1], crop.min[2]};
        attrs["crop_max_xyz"] = {crop.max[0], crop.max[1], crop.max[2]};

        // Diagnostics.
        attrs["fit_rms_group"] = "fit_rms/0";
        attrs["fit_frac_short_paths_group"] = "fit_frac_short_paths/0";
        attrs["fit_used_radius_group"] = "fit_used_radius/0";
        attrs["fit_segment_count_group"] = "fit_segment_count/0";
        attrs["fit_rms_decode"] = "ok? (v/255) : 1.0 (failed fits stored as 255)";
        attrs["fit_frac_short_paths_decode"] = "v/255";
        attrs["fit_used_radius_decode"] = "2 + (v/255)*(512-2)";
        attrs["fit_segment_count_decode"] = "(v/255)*8192";
        vc::writeZarrAttributes(out_zarr, attrs);

        std::error_code ec;
        fs::remove_all(spool_dir, ec);
    }

    const FitStats final_stats = summarize_stats(stats);
    metrics["total_seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    metrics["written_samples"] = written;
    metrics["fit_ok_count"] = final_stats.ok_count;
    metrics["fit_rms_avg"] = (final_stats.ok_count > 0)
        ? (final_stats.rms_sum / static_cast<double>(final_stats.ok_count))
        : 0.0;
    metrics["fit_rms_median"] = estimate_median_from_fine(final_stats);
    metrics["fit_rms_max"] = final_stats.rms_max;
    metrics["thread_summed_seconds"] = {
        {"ng_read", final_stats.t_ng_read_s},
        {"preproc", final_stats.t_preproc_s},
        {"solve", final_stats.t_solve_s},
        {"solve_call", final_stats.t_solve_call_s},
        {"overhead", final_stats.t_overhead_s},
    };
    metrics["distance_rejects"] = {
        {"rejected", final_stats.dist_test_reject},
        {"total", final_stats.dist_test_total},
    };
    metrics["normal_grid_volume_cache"] = {
        {"grid_hits", ngv_cache_stats.gridHits},
        {"grid_misses", ngv_cache_stats.gridMisses},
        {"live_entries", ngv_cache_stats.liveGridEntries},
        {"decoded_path_hits", ngv_cache_stats.decodedPathHits},
        {"decoded_path_misses", ngv_cache_stats.decodedPathMisses},
        {"decoded_path_evictions", ngv_cache_stats.decodedPathEvictions},
        {"decoded_path_entries", ngv_cache_stats.decodedPathEntries},
        {"decoded_path_bytes", ngv_cache_stats.decodedPathBytes},
    };
    metrics["thread_slice_grid_cache"] = {
        {"hits", slice_grid_cache_stats.hits},
        {"misses", slice_grid_cache_stats.misses},
        {"evictions", slice_grid_cache_stats.evictions},
    };
    if (out_zarr_opt.has_value()) {
        metrics["output_spool"] = {
            {"touched_chunks", materialize_stats.touchedChunks},
            {"records", materialize_stats.records},
            {"spill_files", materialize_stats.spillFiles},
            {"materialize_seconds", materialize_stats.materializeSeconds},
            {"in_memory_budget_bytes", materialize_stats.inMemoryBudgetBytes},
        };
    }
    if (metrics_json_path.has_value()) {
        write_metrics_json(*metrics_json_path, metrics);
    }
}

} // namespace

int main(int argc, char** argv) {
    po::options_description desc("vc_ngrids options");
    desc.add_options()
        ("help,h", "Print help")
        ("input,i", po::value<std::string>()->required(), "Input NormalGridVolume directory")
        ("crop,c", po::value<std::vector<int>>()->multitoken(), "Crop x0 y0 z0 x1 y1 z1")
        ("surf", po::value<std::string>(), "Optional tifxyz surface directory")
        ("vis-ply", po::value<std::string>(), "Write visualization PLY file (with colors)")
        ("vis-surf", po::value<std::string>(), "Write --surf tifxyz surface as a quad-mesh PLY")
        ("fit-normals", "Estimate local 3D normals from segments (requires --vis-normals)")
        ("vis-normals", po::value<std::string>(), "Write fitted normals as PLY line segments")
        ("dbg-tif", "Write debug float TIFFs for --fit-normals (workdir): rms, used radius, sample counts (first z layer only)")
        ("output-zarr", po::value<std::string>(), "Write fitted normals to a zarr directory (direction-field encoding)")
        ("metrics-json", po::value<std::string>(), "Write structured fit/align metrics json")
        ("align-normals", "Align normals in an existing normals zarr (requires --input zarr and --output-zarr)");

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

    std::optional<fs::path> surf_path;
    const std::optional<fs::path> metrics_json = vm.count("metrics-json")
        ? std::optional<fs::path>(fs::path(vm["metrics-json"].as<std::string>()))
        : std::nullopt;
    if (vm.count("surf")) {
        surf_path = fs::path(vm["surf"].as<std::string>());
        if (!fs::exists(*surf_path) || !fs::is_directory(*surf_path)) {
            std::cerr << "Error: --surf is not a directory: " << *surf_path << "\n";
            return 1;
        }
    }

    if (vm.count("vis-surf")) {
        if (!surf_path.has_value()) {
            std::cerr << "Error: --vis-surf requires --surf PATH\n";
            return 1;
        }
        auto surf = load_quad_from_tifxyz(surf_path->string());
        if (!surf) {
            std::cerr << "Error: failed to load tifxyz surface: " << surf_path->string() << "\n";
            return 1;
        }
        write_quad_surface_as_ply_quads(*surf, fs::path(vm["vis-surf"].as<std::string>()));
    }

    if (vm.count("vis-ply")) {
        if (input_is_normals_zarr) {
            if (surf_path.has_value()) {
                run_vis_normals_zarr_on_surf_edges_as_ply(input_dir, *surf_path, fs::path(vm["vis-ply"].as<std::string>()), crop);
            } else {
                run_vis_normals_zarr_as_ply(input_dir, fs::path(vm["vis-ply"].as<std::string>()), crop);
            }
        } else {
            if (surf_path.has_value()) {
                throw std::runtime_error("--surf is only supported with normals zarr input (--input <zarr_root>) for --vis-ply");
            }
            run_vis_ply(input_dir, fs::path(vm["vis-ply"].as<std::string>()), crop);
        }
        return 0;
    }

    if (vm.count("align-normals")) {
        if (!input_is_normals_zarr) {
            std::cerr << "Error: --align-normals requires --input to be a normals zarr root\n";
            return 1;
        }
        if (!vm.count("output-zarr")) {
            std::cerr << "Error: --align-normals requires --output-zarr PATH\n";
            return 1;
        }
        run_align_normals_zarr(input_dir,
                               fs::path(vm["output-zarr"].as<std::string>()),
                               crop,
                               100,
                               2,
                               100,
                               metrics_json);
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
        run_fit_normals(input_dir,
                        out_ply,
                        crop,
                        out_zarr,
                        /*step=*/16,
                        /*radius=*/64.f,
                        /*dbg_tif=*/vm.count("dbg-tif") > 0,
                        metrics_json);
        return 0;
    }

    std::cerr << "Error: no output specified. Use --vis-ply or --fit-normals --vis-normals.\n\n";
    print_usage();
    return 1;
}
